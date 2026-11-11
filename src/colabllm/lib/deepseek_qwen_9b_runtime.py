"""Hardware-adaptive 256K-context planning, GGUF + MTP sidecar staging, and llama.cpp runtime helpers for DeepSeek-V4-Pro-Qwen3.5-9B.

Targets `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` GGUF conversions + native MTP speculative decoding:
  - Default on L4 (`23,034 MiB` VRAM) and T4 (`15,360 MiB` VRAM): `i1-Q4_K_M` (`5.63 GB` / `5.24 GiB` imatrix-calibrated 4-bit)
    from `mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF` (with `--quant Q8_0`, `i1-Q6_K`, `f16`, etc. optional)
  - Native Multi-Token Prediction (`MTP`) draft sidecar (`DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.mtp-Q8_0.gguf`, `~2.32 GiB`),
    remotely exported once via `convert_hf_to_gguf.py --remote ... --mtp --outtype q8_0`, cached on Google Drive,
    and served with `llama-server --spec-type draft-mtp`
  - Multimodal vision projector: `DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.mmproj-f16.gguf` (`0.92 GB` / `0.86 GiB`)
served via `llama-server` (`b11540`) with native `--api-key-file` Bearer authentication on a Colab L4 GPU by default.

Imported both locally (`src/colabllm/services/deepseek-qwen-9b.py` CLI `plan` / validation / unit tests, stdlib only)
and inside the generated Colab notebook (`src/colabllm/services/deepseek-qwen-9b.ipynb`).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from typing import Any, Callable

_ENV_SPAWN_LOCK = threading.Lock()

DEFAULT_REPO = "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-GGUF"
DEFAULT_REVISION = "2f01d0e8b732843d317e97165ff1c5f135b09147"
IMATRIX_REPO = "mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF"
IMATRIX_REVISION = "fff50f6bb44cefedd08217dbc285fb2c41213fff"

DEFAULT_QUANT = "auto"
L4_DEFAULT_QUANT = "i1-Q4_K_M"
T4_DEFAULT_QUANT = "i1-Q4_K_M"
Q8_MIN_VRAM_MIB = 20_000

UPSTREAM_HF_REPO = "xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic"
MTP_SIDECAR_FILENAME = "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.mtp-Q8_0.gguf"
MTP_WEIGHTS_MIB = 2_320
MTP_ATTN_LAYERS = 1
MTP_MODES = ("auto", "on", "off")
DEFAULT_MTP_MODE = "auto"
DEFAULT_MTP_DRAFT_MAX = 3
DEFAULT_MTP_P_MIN = 0.60
MIN_MTP_AUTO_KV_BUDGET_MIB = 2_048

DEFAULT_ALIAS = "deepseek-v4-pro-qwen3.5-9b"
DEFAULT_LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
# ggml-org/llama.cpp tag b11540 (native qwen35 hybrid Gated DeltaNet + draft-mtp + qwen3vl_merger vision + enable_thinking)
DEFAULT_LLAMA_COMMIT = "1e6f04a75e91da4bb66b5248f3fd4bdeff169d41"

NATIVE_ORIG_CONTEXT = 262_144
DEFAULT_CONTEXT = 262_144
MODEL_MAX_CONTEXT = 1_048_576
CPU_AUTO_CONTEXT = 16_384
MIN_CONTEXT = 8_192
CONTEXT_ALIGN = 16_384

MODEL_NUM_LAYERS = 32
FULL_ATTN_LAYERS = 8   # block_count=32, full_attention_interval=4 -> 8 GQA full-attn + 24 Gated DeltaNet SSM
ATTN_KV_HEADS = 4
ATTN_KEY_LEN = 256
ATTN_VAL_LEN = 256

VISION_MMPROJ_MIB = 876
SSM_RECURRENT_STATE_MIB = 52          # 24 linear-attention Gated DeltaNet layers (f32 128x128x32 state + conv1d)
FIXED_GPU_RUNTIME_RESERVE_MIB = 2_560  # CUDA context + FlashAttn workspace + 27-layer qwen3vl ViT activations

SUPPORTED_CACHE_TYPES = ("q8_0", "f16", "q4_0")
REASONING_MODES = ("on", "off", "auto")

VISION_PROJECTOR_SPEC: dict[str, Any] = {
    "repo": DEFAULT_REPO,
    "revision": DEFAULT_REVISION,
    "name": "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.mmproj-f16.gguf",
    "size": 918_166_368,
    "sha256": "1a2244e013aa758c5c3cf6b384400165aad63eac437ebb4314d21bced8f16862",
}


def _spec(repo: str, revision: str, filename: str, size: int, sha256: str) -> dict[str, Any]:
    return {
        "repo": repo,
        "revision": revision,
        "name": filename,
        "size": size,
        "sha256": sha256,
        "weights_mib": int(math.ceil(size / (1024 * 1024))),
    }


PINNED_QUANT_SPECS: dict[str, dict[str, Any]] = {
    # Primary GGUF repo (`Q8_0` default on L4, `f16` unquantized, plus standard K-quants)
    "Q8_0": _spec(
        DEFAULT_REPO, DEFAULT_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q8_0.gguf",
        9_527_503_840, "0bc2a8896ff0929f07c50611e08abc7dfea847807c5ef89afd26ac2ea709b834",
    ),
    "f16": _spec(
        DEFAULT_REPO, DEFAULT_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.f16.gguf",
        17_920_699_360, "4f1db3705cbef5143c24854a3bd82d7086f995c355a056326f477e17096e7bea",
    ),
    "Q6_K": _spec(
        DEFAULT_REPO, DEFAULT_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q6_K.gguf",
        7_359_261_664, "1c24a104546efc4b94e4b58cb06ce60fe47da3119b7211de816ec1da0ff1a1e0",
    ),
    "Q5_K_M": _spec(
        DEFAULT_REPO, DEFAULT_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q5_K_M.gguf",
        6_467_972_064, "c7bb6c588557e9d7c07ec7161f6f6875ff08fd62632cebe6b026a98a38c05016",
    ),
    "Q5_K_S": _spec(
        DEFAULT_REPO, DEFAULT_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q5_K_S.gguf",
        6_305_311_712, "4e01b5257e74442e36353c9374c98bbc43e330c43f4c3f0e228deefa9decda4d",
    ),
    "Q4_K_M": _spec(
        DEFAULT_REPO, DEFAULT_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_M.gguf",
        5_629_111_264, "a692de9520e4c0bb1512deb91e950f38e14854650aff43419112f87e6052c44b",
    ),
    "Q4_K_S": _spec(
        DEFAULT_REPO, DEFAULT_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.Q4_K_S.gguf",
        5_351_631_840, "70b98f788618673e7fced4e3b4fbd0e0af8aa33167b450f4a9fcae4db90bd85e",
    ),
    "IQ4_XS": _spec(
        DEFAULT_REPO, DEFAULT_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.IQ4_XS.gguf",
        5_227_899_872, "0927462a1491c6b6e3cc1d9037df27b9a07ba574ba501ff432224e6c1205ea5c",
    ),
    # Imatrix-calibrated GGUF repo (`i1-Q4_K_M` auto default on T4)
    "i1-Q6_K": _spec(
        IMATRIX_REPO, IMATRIX_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q6_K.gguf",
        7_359_261_984, "c189016a38e7a27552bc72746982a3969a12f0eae321fd98400391e0cc10a969",
    ),
    "i1-Q5_K_M": _spec(
        IMATRIX_REPO, IMATRIX_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q5_K_M.gguf",
        6_467_972_384, "f892fa454357cbc76226a597e7095da3708556937d5c11e42475ca0314f1574d",
    ),
    "i1-Q5_K_S": _spec(
        IMATRIX_REPO, IMATRIX_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q5_K_S.gguf",
        6_305_312_032, "9f22cdd1fde1f7f83eccade347fec18c09683fe431a7e53b207e7e1b4c8ef89f",
    ),
    "i1-Q4_K_M": _spec(
        IMATRIX_REPO, IMATRIX_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf",
        5_629_111_584, "8cf793a36e62fb078ac05c2e1bdb2121ff94736dccbede7cf81b76346e868949",
    ),
    "i1-Q4_K_S": _spec(
        IMATRIX_REPO, IMATRIX_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_S.gguf",
        5_351_632_160, "2a348bbc633cbc628d8e526340b6f88f4bef53bc2e5bc11919f16a861bfc4065",
    ),
    "i1-IQ4_NL": _spec(
        IMATRIX_REPO, IMATRIX_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-IQ4_NL.gguf",
        5_418_216_736, "1ba41c2b6e49d50a4b00bcbf4146227d2d8f5d57e669ebcaa6401a556e0ab87b",
    ),
    "i1-IQ4_XS": _spec(
        IMATRIX_REPO, IMATRIX_REVISION,
        "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-IQ4_XS.gguf",
        5_196_442_912, "55f51ef810a801849ab84f2ae029d1e6142bf83f563983f550d3ce4898750312",
    ),
}

_CANONICAL_QUANTS = {k.lower(): k for k in PINNED_QUANT_SPECS}

_OOM_PATTERNS = (
    "out of memory",
    "cuda_malloc failed",
    "cudamalloc failed",
    "failed to allocate",
    "ggml_gallocr_reserve_n: failed to allocate",
    "unable to allocate cuda",
    "insufficient memory",
)


def resolve_quant_choice(quant: str | None, vram_mib: int) -> str:
    """Resolve `'auto'` (`i1-Q4_K_M` imatrix 4-bit default across L4/T4/A100/H100/G4) or normalize quant name."""
    raw = (quant or "auto").strip()
    if not raw or raw.lower() == "auto":
        return L4_DEFAULT_QUANT if int(vram_mib) >= Q8_MIN_VRAM_MIB else T4_DEFAULT_QUANT
    return _CANONICAL_QUANTS.get(raw.lower(), raw)


def normalize_mtp_mode(raw: str | bool | None) -> str:
    if isinstance(raw, bool):
        return "on" if raw else "off"
    s = str(raw if raw is not None else DEFAULT_MTP_MODE).strip().lower()
    if s in ("", "auto"):
        return "auto"
    if s in ("1", "true", "yes", "on"):
        return "on"
    if s in ("0", "false", "no", "off"):
        return "off"
    raise ValueError(f"Unsupported MTP mode {raw!r}; choose from {MTP_MODES}")


def estimate_quant_weights_mib(quant: str) -> int:
    spec = PINNED_QUANT_SPECS.get(quant)
    if spec is not None:
        return int(spec["weights_mib"])
    upper = quant.upper()
    if "F16" in upper or "BF16" in upper:
        return 17_091
    if "Q8" in upper:
        return 9_087
    if "Q6" in upper:
        return 7_019
    if "Q5" in upper:
        return 6_169
    if "Q3" in upper:
        return 4_410
    if "Q2" in upper:
        return 3_650
    return 5_369


def _ggml_bytes_per_elem(cache_type: str) -> float:
    ct = (cache_type or "q8_0").strip().lower()
    if ct in ("f16", "bf16"):
        return 2.0
    if ct == "q8_0":
        return 34.0 / 32.0  # 1.0625 B/elem (32 int8 + fp16 scale)
    if ct in ("q4_0", "q4_1", "iq4_nl"):
        return 18.0 / 32.0  # 0.5625 B/elem
    raise ValueError(f"Unsupported KV cache type {cache_type!r}; choose from {SUPPORTED_CACHE_TYPES}")


def qwen35_kv_bytes_per_token(
    cache_type: str = "q8_0",
    parallel: int = 1,
    *,
    enable_mtp: bool = False,
) -> float:
    """Exact per-token GPU KV cache bytes across the 8 trunk full-attn layers (+ 1 MTP draft full-attn layer).

    From `qwen35` metadata (`block_count=32`, `full_attention_interval=4`, `nextn_predict_layers=1`):
      - Trunk: 8 full-attention layers * 4 KV heads * (256 K + 256 V) = 16,384 elements/token.
      - MTP draft (`LLAMA_CONTEXT_TYPE_MTP`): 1 full-attention `nextn` layer = +2,048 elements/token when enabled.
      - The other 24 Gated DeltaNet SSM layers keep a fixed O(1) recurrent state (~52 MiB/slot).
    """
    bpe = _ggml_bytes_per_elem(cache_type)
    attn_layers = FULL_ATTN_LAYERS + (MTP_ATTN_LAYERS if enable_mtp else 0)
    elems_per_token = attn_layers * ATTN_KV_HEADS * (ATTN_KEY_LEN + ATTN_VAL_LEN)
    return float(elems_per_token) * bpe * max(1, int(parallel))


def estimate_kv_cache_mib(
    context: int,
    cache_type: str = "q8_0",
    parallel: int = 1,
    *,
    enable_mtp: bool = False,
) -> float:
    kv_mib = (int(context) * qwen35_kv_bytes_per_token(cache_type, parallel, enable_mtp=enable_mtp)) / (1024.0 * 1024.0)
    return round(kv_mib + SSM_RECURRENT_STATE_MIB * max(1, int(parallel)), 1)


def parse_context_spec(raw: str | int | None) -> int | None:
    """Parse context token spec ('auto', '256k', '262144', '128k', '64k', '512k', '1m'); None means 'auto'."""
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("_", "").replace(",", "")
    if not s or s == "auto":
        return None
    if s in ("1m", "1024k"):
        val = 1_048_576
    elif s.endswith("k") and s[:-1].replace(".", "", 1).isdigit():
        val = int(round(float(s[:-1]) * 1024))
    elif s.isdigit():
        val = int(s)
    else:
        raise ValueError(f"Invalid context value {raw!r}; expected integer, '<N>k', '1m', or 'auto'")
    if val < MIN_CONTEXT or val > MODEL_MAX_CONTEXT:
        raise ValueError(f"Context {val} out of supported range [{MIN_CONTEXT}, {MODEL_MAX_CONTEXT}]")
    return val


def compute_yarn_params(context: int, orig_ctx: int = NATIVE_ORIG_CONTEXT) -> tuple[str, float, int]:
    """Return `(rope_scaling, rope_scale, yarn_orig_ctx)` when extending past native 262,144 tokens."""
    if int(context) <= int(orig_ctx):
        return ("", 1.0, int(orig_ctx))
    scale = round(float(context) / float(orig_ctx), 6)
    return ("yarn", scale, int(orig_ctx))


def plan_hardware_config(
    *,
    vram_mib: int,
    ram_mib: int,
    quant: str = DEFAULT_QUANT,
    requested_context: str | int | None = "auto",
    cache_type: str = "q8_0",
    enable_vision: bool = True,
    mtp: str | bool = DEFAULT_MTP_MODE,
    mtp_draft_max: int = DEFAULT_MTP_DRAFT_MAX,
    mtp_p_min: float = DEFAULT_MTP_P_MIN,
    parallel: int = 1,
    batch: int = 2048,
    ubatch: int = 512,
) -> dict[str, Any]:
    """Compute VRAM-safe quantization (`i1-Q4_K_M` default), MTP draft budget, and native 256K context plan."""
    ct = (cache_type or "q8_0").strip().lower()
    if ct not in SUPPORTED_CACHE_TYPES:
        raise ValueError(f"Unsupported cache_type {cache_type!r}; choose from {SUPPORTED_CACHE_TYPES}")
    mtp_mode = normalize_mtp_mode(mtp)
    if int(mtp_draft_max) < 1:
        raise ValueError("mtp_draft_max must be >= 1")
    if not (0.0 <= float(mtp_p_min) <= 1.0):
        raise ValueError("mtp_p_min must be in [0.0, 1.0]")

    explicit_ctx = parse_context_spec(requested_context)
    resolved_quant = resolve_quant_choice(quant, vram_mib)
    quant_auto = (quant or "auto").strip().lower() in ("", "auto")

    model_mib = estimate_quant_weights_mib(resolved_quant)
    mmproj_mib = VISION_MMPROJ_MIB if enable_vision else 0
    base_weights_mib = model_mib + mmproj_mib

    if int(vram_mib) <= 0:
        mtp_enabled = mtp_mode == "on"
        weights_mib = base_weights_mib + (MTP_WEIGHTS_MIB if mtp_enabled else 0)
        target_ctx = explicit_ctx if explicit_ctx is not None else CPU_AUTO_CONTEXT
        kv_mib = estimate_kv_cache_mib(target_ctx, ct, parallel, enable_mtp=mtp_enabled)
        rope_scaling, rope_scale, yarn_orig_ctx = compute_yarn_params(target_ctx)
        spec_repo = PINNED_QUANT_SPECS.get(resolved_quant, {}).get("repo", DEFAULT_REPO)
        return {
            "mode": "cpu-only",
            "quant": resolved_quant,
            "quant_auto": quant_auto,
            "resolved_repo": spec_repo,
            "gpu_layers": "0",
            "context": int(target_ctx),
            "context_auto": explicit_ctx is None,
            "rope_scaling": rope_scaling,
            "rope_scale": rope_scale,
            "yarn_orig_ctx": yarn_orig_ctx,
            "cache_type": ct,
            "parallel": int(parallel),
            "batch": int(batch),
            "ubatch": int(ubatch),
            "enable_vision": bool(enable_vision),
            "enable_mtp": bool(mtp_enabled),
            "mtp_mode": mtp_mode,
            "mtp_draft_max": int(mtp_draft_max),
            "mtp_p_min": float(mtp_p_min),
            "vram_mib": 0,
            "ram_mib": int(ram_mib),
            "estimated_gpu_weights_mib": 0,
            "estimated_cpu_weights_mib": int(weights_mib),
            "kv_cache_mib": kv_mib,
            "estimated_vram_headroom_mib": 0.0,
        }

    if mtp_mode == "off":
        mtp_enabled = False
    elif mtp_mode == "on":
        mtp_enabled = True
    else:
        mtp_kv_room = (
            float(vram_mib) - (base_weights_mib + MTP_WEIGHTS_MIB) - FIXED_GPU_RUNTIME_RESERVE_MIB
        )
        mtp_enabled = mtp_kv_room >= MIN_MTP_AUTO_KV_BUDGET_MIB

    weights_mib = base_weights_mib + (MTP_WEIGHTS_MIB if mtp_enabled else 0)

    if explicit_ctx is not None:
        target_ctx = explicit_ctx
    else:
        vram_for_kv = max(
            0.0,
            float(vram_mib) - weights_mib - FIXED_GPU_RUNTIME_RESERVE_MIB - SSM_RECURRENT_STATE_MIB * max(1, int(parallel)),
        )
        max_tokens = int(
            (vram_for_kv * 1024.0 * 1024.0) / qwen35_kv_bytes_per_token(ct, parallel, enable_mtp=mtp_enabled)
        )
        if max_tokens >= DEFAULT_CONTEXT:
            target_ctx = DEFAULT_CONTEXT
        else:
            aligned = (max_tokens // CONTEXT_ALIGN) * CONTEXT_ALIGN
            target_ctx = max(MIN_CONTEXT, aligned)

    kv_mib = estimate_kv_cache_mib(target_ctx, ct, parallel, enable_mtp=mtp_enabled)
    rope_scaling, rope_scale, yarn_orig_ctx = compute_yarn_params(target_ctx)
    free_vram_margin_mib = round(float(vram_mib) - weights_mib - kv_mib, 1)
    spec_repo = PINNED_QUANT_SPECS.get(resolved_quant, {}).get("repo", DEFAULT_REPO)

    return {
        "mode": "full-gpu",
        "quant": resolved_quant,
        "quant_auto": quant_auto,
        "resolved_repo": spec_repo,
        "gpu_layers": "all",
        "context": int(target_ctx),
        "context_auto": explicit_ctx is None,
        "rope_scaling": rope_scaling,
        "rope_scale": rope_scale,
        "yarn_orig_ctx": yarn_orig_ctx,
        "cache_type": ct,
        "parallel": int(parallel),
        "batch": int(batch),
        "ubatch": int(ubatch),
        "enable_vision": bool(enable_vision),
        "enable_mtp": bool(mtp_enabled),
        "mtp_mode": mtp_mode,
        "mtp_draft_max": int(mtp_draft_max),
        "mtp_p_min": float(mtp_p_min),
        "vram_mib": int(vram_mib),
        "ram_mib": int(ram_mib),
        "estimated_gpu_weights_mib": int(weights_mib),
        "estimated_cpu_weights_mib": 0,
        "kv_cache_mib": kv_mib,
        "estimated_vram_headroom_mib": free_vram_margin_mib,
    }


def detect_host_ram_mib() -> int:
    """Read total system RAM (or cgroup memory.max cap when lower) in MiB."""
    total_bytes = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2 and parts[1].isdigit():
                    total_bytes = int(parts[1]) * 1024
                break
    if total_bytes <= 0:
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
                total_bytes = pages * page_size
        except (AttributeError, ValueError, OSError):
            total_bytes = 0
    for cg_path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        if cg_path.is_file():
            raw = cg_path.read_text(encoding="utf-8", errors="replace").strip()
            if raw.isdigit():
                cg_val = int(raw)
                if 0 < cg_val < (1 << 50):
                    total_bytes = min(total_bytes, cg_val) if total_bytes > 0 else cg_val
    return total_bytes // (1024 * 1024)


def detect_gpu_info() -> dict[str, Any]:
    """Query primary NVIDIA GPU name, compute capability, CUDA arch token, and total VRAM in MiB."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return {"present": False, "name": "cpu", "compute_cap": "", "cuda_arch": "", "vram_mib": 0}
    first = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return {"present": False, "name": "cpu", "compute_cap": "", "cuda_arch": "", "vram_mib": 0}
    name, cap, mem_str = parts[0], parts[1], parts[2]
    cuda_arch = re.sub(r"[^0-9]", "", cap)
    try:
        vram_mib = int(float(mem_str))
    except ValueError:
        vram_mib = 0
    return {
        "present": vram_mib > 0 and bool(cuda_arch),
        "name": name,
        "compute_cap": cap,
        "cuda_arch": cuda_arch,
        "vram_mib": vram_mib,
    }


def default_cpu_threads() -> tuple[int, int]:
    cpus = os.cpu_count() or 8
    phys = max(2, cpus // 2)
    return (min(8, phys), min(12, max(4, cpus - 1)))


def build_server_args(
    *,
    server_bin: Path | str,
    model_path: Path | str,
    alias: str,
    port: int,
    key_file: Path | str,
    plan: dict[str, Any],
    projector_path: Path | str | None = None,
    mtp_path: Path | str | None = None,
    reasoning: str = "on",
    threads: int | None = None,
    threads_batch: int | None = None,
) -> list[str]:
    """Construct exact loopback `llama-server` CLI arguments for DeepSeek-V4-Pro-Qwen3.5-9B."""
    t_dec_default, t_bat_default = default_cpu_threads()
    t_dec = int(threads) if threads is not None else t_dec_default
    t_bat = int(threads_batch) if threads_batch is not None else t_bat_default

    args = [
        str(server_bin),
        "-m", str(model_path),
        "--alias", str(alias),
        "--host", "127.0.0.1",
        "--port", str(int(port)),
        "-ngl", str(plan["gpu_layers"]),
        "--fit", "off",
        "-c", str(int(plan["context"])),
        "-np", str(int(plan["parallel"])),
        "-t", str(t_dec),
        "-tb", str(t_bat),
        "-ctk", str(plan["cache_type"]),
        "-ctv", str(plan["cache_type"]),
        "-fa", "on",
        "-b", str(int(plan["batch"])),
        "-ub", str(int(plan["ubatch"])),
        "--jinja",
        "--api-key-file", str(key_file),
        "--no-context-shift",
        "--ctx-checkpoints", "32",
        "--cache-ram", "4096",
    ]
    if projector_path is not None and plan.get("enable_vision", True):
        args += ["--mmproj", str(projector_path)]
    if mtp_path is not None and plan.get("enable_mtp", False):
        args += [
            "-md", str(mtp_path),
            "--spec-type", "draft-mtp",
            "-ngld", str(plan["gpu_layers"]),
            "-ctkd", str(plan["cache_type"]),
            "-ctvd", str(plan["cache_type"]),
            "--spec-draft-n-max", str(int(plan.get("mtp_draft_max", DEFAULT_MTP_DRAFT_MAX))),
            "--spec-draft-p-min", str(plan.get("mtp_p_min", DEFAULT_MTP_P_MIN)),
        ]
    if plan.get("rope_scaling"):
        args += [
            "--rope-scaling", str(plan["rope_scaling"]),
            "--rope-scale", str(plan["rope_scale"]),
            "--yarn-orig-ctx", str(int(plan["yarn_orig_ctx"])),
        ]
    reasoning_mode = (reasoning or "").strip().lower()
    if reasoning_mode in REASONING_MODES:
        args += ["--reasoning", reasoning_mode]
    return args


def looks_like_oom(log_tail: str) -> bool:
    lower = (log_tail or "").lower()
    return any(pat in lower for pat in _OOM_PATTERNS)


def write_json_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_env(env: dict[str, str] | None = None) -> dict[str, str]:
    with _ENV_SPAWN_LOCK:
        return dict(env) if env is not None else dict(os.environ)


def _cmd_out(command: list[str], **kwargs: Any) -> str:
    explicit_env = kwargs.pop("env", None)
    for attempt in range(3):
        spawn_env = _snapshot_env(explicit_env)
        try:
            return subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.STDOUT,
                env=spawn_env,
                **kwargs,
            ).strip()
        except OSError as err:
            if err.errno == errno.EFAULT and attempt < 2:
                time.sleep(0.05 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Unreachable _cmd_out exhaustion for {command!r}")


def _run_logged(command: list[str], log_file: Path, env: dict[str, str] | None = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    res: subprocess.CompletedProcess[Any] | None = None
    for attempt in range(3):
        spawn_env = _snapshot_env(env)
        try:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"\n$ {' '.join(str(x) for x in command)}\n")
                handle.flush()
                res = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=spawn_env)
            break
        except OSError as err:
            if err.errno == errno.EFAULT and attempt < 2:
                time.sleep(0.05 * (attempt + 1))
                continue
            raise
    assert res is not None
    if res.returncode != 0:
        tail = "".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-40:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(command[:4])}\n{tail}")


def _restore_valid_bin(bin_dir: Path, env: dict[str, str], require_cuda: bool = True) -> bool:
    server = bin_dir / "llama-server"
    if not server.is_file():
        return False
    try:
        server.chmod(0o755)
        _cmd_out([str(server), "--version"], env=env, timeout=20)
        if require_cuda:
            devices = _cmd_out([str(server), "--list-devices"], env=env, timeout=20)
            if "CUDA" not in devices:
                return False
        help_text = _cmd_out([str(server), "--help"], env=env, timeout=20)
        return all(flag in help_text for flag in ("--api-key-file", "--ctx-checkpoints", "--reasoning", "--spec-type"))
    except (OSError, subprocess.SubprocessError):
        return False


def prepare_llama_runtime(
    *,
    work_dir: Path,
    cache_root: Path,
    llama_repo: str,
    llama_commit: str,
    gpu_info: dict[str, Any],
    log: Callable[[str], None],
) -> tuple[Path, dict[str, str], str]:
    """Build or restore cached `llama-server` for the current GPU architecture and pinned `llama_commit`."""
    commit = (llama_commit or "").strip() or _cmd_out(["git", "ls-remote", llama_repo, "HEAD"]).split()[0]
    cuda_arch = gpu_info.get("cuda_arch") or "80"
    require_cuda = bool(gpu_info.get("present"))
    nvcc_ver = _cmd_out(["nvcc", "--version"]) if shutil.which("nvcc") else "no-nvcc"
    os_rel = Path("/etc/os-release").read_text(encoding="utf-8", errors="replace") if Path("/etc/os-release").is_file() else ""
    signature = {
        "llama_commit": commit,
        "gpu": gpu_info.get("name", ""),
        "cuda_arch": cuda_arch,
        "cuda": nvcc_ver,
        "os_release": os_rel,
        "arch": platform.machine(),
        "glibc": list(platform.libc_ver()),
        "ggml_native": False,
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    cache_dir = Path(cache_root) / f"sm{cuda_arch}-{key}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    bin_dir = Path(work_dir) / "runtime" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    local_manifest = Path(work_dir) / "runtime-manifest.json"

    if any(bin_dir.iterdir()):
        try:
            cached_sig = json.loads(local_manifest.read_text(encoding="utf-8"))
        except Exception:
            cached_sig = None
        if cached_sig != signature:
            shutil.rmtree(bin_dir)
            bin_dir.mkdir(parents=True, exist_ok=True)

    archive = cache_dir / "runtime.tar"
    checksum = cache_dir / "runtime.sha256"
    if not _restore_valid_bin(bin_dir, env, require_cuda) and archive.is_file() and checksum.is_file():
        log(f"Restoring cached llama.cpp runtime ({commit[:10]}) from Drive: {cache_dir}")
        local_tar = Path(work_dir) / "runtime.tar"
        shutil.copyfile(archive, local_tar)
        if digest_file(local_tar) == checksum.read_text(encoding="utf-8").strip():
            with tarfile.open(local_tar) as bundle:
                bundle.extractall(Path(work_dir) / "runtime", filter="data")
        else:
            log("Warning: Drive build cache checksum mismatch; rebuilding from source")
        local_tar.unlink(missing_ok=True)

    if not _restore_valid_bin(bin_dir, env, require_cuda):
        build_log = Path(work_dir) / "build.log"
        log(f"Building llama.cpp ({commit[:10]}) for CUDA sm_{cuda_arch} (log: {build_log})...")
        source = Path(work_dir) / "llama.cpp"
        if not (source / ".git").is_dir():
            shutil.rmtree(source, ignore_errors=True)
            _run_logged(["git", "clone", "--no-checkout", "--filter=blob:none", llama_repo, str(source)], build_log)
        _run_logged(["git", "-C", str(source), "fetch", "--depth", "1", "origin", commit], build_log)
        _run_logged(["git", "-C", str(source), "checkout", "--detach", commit], build_log)
        cmake_args = [
            "cmake", "-S", str(source), "-B", str(source / "build"),
            f"-DGGML_CUDA={'ON' if require_cuda else 'OFF'}",
            f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
            "-DGGML_NATIVE=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_OPENSSL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
        ]
        _run_logged(cmake_args, build_log)
        _run_logged(
            ["cmake", "--build", str(source / "build"), "--config", "Release", "--target", "llama-server",
             "-j", str(min(12, os.cpu_count() or 4))],
            build_log,
        )
        shutil.copytree(source / "build" / "bin", bin_dir, dirs_exist_ok=True, symlinks=False)
        shutil.rmtree(source / "build", ignore_errors=True)
        if not _restore_valid_bin(bin_dir, env, require_cuda):
            raise RuntimeError("Built llama-server failed validation or did not detect CUDA")
        local_tar = Path(work_dir) / "runtime.tar"
        with tarfile.open(local_tar, "w", dereference=True) as bundle:
            bundle.add(bin_dir, arcname="bin")
        sha = digest_file(local_tar)
        try:
            partial = cache_dir / f"runtime.tar.{os.getpid()}.partial"
            shutil.copyfile(local_tar, partial)
            partial.replace(archive)
            checksum.write_text(sha + "\n", encoding="utf-8")
            write_json_atomic(cache_dir / "manifest.json", signature)
            log(f"Cached compiled llama.cpp runtime on Drive: {cache_dir}")
        except OSError as err:
            log(f"Runtime Drive cache save warning (non-fatal): {err}")
        finally:
            local_tar.unlink(missing_ok=True)

    write_json_atomic(local_manifest, signature)
    return bin_dir, env, commit


def _hf_client(work_dir: Path) -> tuple[Any, Any]:
    with _ENV_SPAWN_LOCK:
        os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        import hf_xet  # noqa: F401
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        _run_logged(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_xet"],
            Path(work_dir) / "download.log",
        )
        from huggingface_hub import HfApi, hf_hub_download
    return HfApi, hf_hub_download


def _xet_available() -> bool:
    try:
        import hf_xet  # noqa: F401
    except ImportError:
        return False
    return True


def _cleanup_hf_scratch(local_dir: Path) -> None:
    shutil.rmtree(local_dir / ".cache", ignore_errors=True)
    for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
        shutil.rmtree(p, ignore_errors=True)


def _valid_file(path: Path, item: dict[str, Any], *, check_sha256: bool = True) -> bool:
    if not path.is_file() or path.stat().st_size != int(item["size"]):
        return False
    if not check_sha256:
        return True
    expected_sha = (item.get("sha256") or "").strip().lower()
    return (not expected_sha) or (digest_file(path) == expected_sha)


def _stage_file(
    *,
    item: dict[str, Any],
    repo: str,
    revision: str,
    local_dir: Path,
    drive_backup_root: Path | None,
    download_fn: Any,
    log: Callable[[str], None],
) -> Path:
    target = local_dir / item["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    strict_env = os.environ.get("CT_DS9B_VERIFY_SHA256", "0") == "1"
    if _valid_file(target, item, check_sha256=False) and (not strict_env or _valid_file(target, item, check_sha256=True)):
        log(f"Reusing local SSD file: {item['name']} ({item['size'] / 1e9:.2f} GB)")
        return target

    if download_fn is not None:
        file_tag = hashlib.sha256(item["name"].encode("utf-8")).hexdigest()[:12]
        scratch_dir = local_dir / f".hf-stage-{os.getpid()}-{file_tag}"
        shutil.rmtree(scratch_dir, ignore_errors=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        try:
            log(f"Downloading {repo}/{item['name']} ({item['size'] / 1e9:.2f} GB) -> local NVMe SSD...")
            os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
            downloaded = Path(
                download_fn(
                    repo_id=repo,
                    filename=item["name"],
                    revision=revision,
                    local_dir=str(scratch_dir),
                )
            )
            strict_sha = strict_env or not _xet_available()
            if not _valid_file(downloaded, item, check_sha256=strict_sha):
                downloaded.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded GGUF size/SHA256 mismatch: {item['name']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(target)
            mode_label = "SHA256" if strict_sha else "xet Merkle + size"
            log(f"Verified {mode_label} ({item['size'] / 1e9:.2f} GB): {target.name}")
            return target
        except Exception as err:
            log(f"HF download failed for {item['name']} ({type(err).__name__}: {err}); checking Drive backup...")
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    if drive_backup_root is not None:
        backup = Path(drive_backup_root) / repo / item["name"]
        if backup.is_file() and backup.stat().st_size == int(item["size"]):
            log(f"Copying Drive backup -> local SSD: {item['name']}")
            partial = target.with_suffix(".gguf.drive.partial")
            h = hashlib.sha256()
            with backup.open("rb") as src, partial.open("wb") as dst:
                while chunk := src.read(16 * 1024 * 1024):
                    dst.write(chunk)
                    h.update(chunk)
            expected_sha = (item.get("sha256") or "").strip().lower()
            if expected_sha and h.hexdigest() != expected_sha:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Drive backup SHA256 mismatch: {item['name']}")
            partial.replace(target)
            log(f"Verified Drive backup SHA256: {target.name}")
            return target

    raise RuntimeError(f"Unable to stage {item['name']} from Hugging Face ({repo}) or Drive backup")


def resolve_model_lock(
    *,
    repo: str,
    quant: str,
    service_dir: Path,
    work_dir: Path,
) -> tuple[dict[str, Any], Any]:
    """Return `(lock_dict, hf_hub_download_or_None)` for `(repo, quant)`."""
    canonical = _CANONICAL_QUANTS.get(quant.strip().lower(), quant.strip())
    if repo in (DEFAULT_REPO, IMATRIX_REPO) and canonical in PINNED_QUANT_SPECS:
        spec = PINNED_QUANT_SPECS[canonical]
        return {
            "repo": spec["repo"],
            "quant": canonical,
            "revision": spec["revision"],
            "files": [{"name": spec["name"], "size": spec["size"], "sha256": spec["sha256"]}],
        }, None

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{repo}--{canonical}")
    lock_path = Path(service_dir) / f"model-lock-{slug}.json"
    if lock_path.is_file():
        cached = json.loads(lock_path.read_text(encoding="utf-8"))
        if cached.get("repo") == repo and cached.get("quant") == canonical and cached.get("files"):
            return cached, None

    HfApi, download_fn = _hf_client(work_dir)
    info = HfApi().model_info(repo, files_metadata=True)
    needle = canonical.lower()
    files: list[dict[str, Any]] = []
    for entry in info.siblings:
        rfn = entry.rfilename
        base_lower = Path(rfn).name.lower()
        if (
            rfn.endswith(".gguf")
            and "imatrix" not in base_lower
            and "mmproj" not in base_lower
            and (
                f".{needle}.gguf" in base_lower
                or f"-{needle}.gguf" in base_lower
                or rfn.lower().startswith(needle + "/")
            )
        ):
            lfs = getattr(entry, "lfs", None)
            sha256 = getattr(lfs, "sha256", None) or (lfs.get("oid") if isinstance(lfs, dict) else "")
            if not sha256:
                raise RuntimeError(f"Missing LFS SHA256 metadata for {rfn}")
            files.append({"name": rfn, "size": int(entry.size), "sha256": sha256})
    if not files:
        raise RuntimeError(f"No GGUF model files matching quant {canonical!r} found in {repo}")
    lock = {
        "repo": repo,
        "quant": canonical,
        "revision": info.sha,
        "files": sorted(files, key=lambda x: x["name"]),
    }
    write_json_atomic(lock_path, lock)
    return lock, download_fn


def stage_model_and_projector(
    *,
    repo: str,
    quant: str,
    enable_vision: bool,
    service_dir: Path,
    work_dir: Path,
    drive_backup_root: Path | None,
    log: Callable[[str], None],
) -> tuple[Path, Path | None]:
    """Download & SHA256-verify GGUF model + optional F16 vision projector onto local NVMe SSD."""
    lock, download_fn = resolve_model_lock(repo=repo, quant=quant, service_dir=service_dir, work_dir=work_dir)
    actual_repo = lock["repo"]
    local_dir = Path(work_dir) / "model"
    local_dir.mkdir(parents=True, exist_ok=True)
    marker = local_dir / f"verified-{lock['quant']}.json"
    expected_map = {item["name"]: item["sha256"] for item in lock["files"]}

    all_ready = (
        marker.is_file()
        and json.loads(marker.read_text(encoding="utf-8")) == expected_map
        and all(
            (local_dir / item["name"]).is_file() and (local_dir / item["name"]).stat().st_size == int(item["size"])
            for item in lock["files"]
        )
    )
    if all_ready:
        log(f"Reusing {len(lock['files'])} verified local SSD model file(s) ({lock['quant']})")
    else:
        pending = [item for item in lock["files"] if not _valid_file(local_dir / item["name"], item)]
        if pending:
            needed_bytes = sum(int(item["size"]) for item in pending) + 4 * 1024**3
            free_bytes = shutil.disk_usage(work_dir).free
            if free_bytes < needed_bytes:
                raise RuntimeError(
                    f"Insufficient local SSD space at {work_dir}: need {needed_bytes / 1e9:.1f} GB, "
                    f"free {free_bytes / 1e9:.1f} GB"
                )
            if download_fn is None:
                try:
                    _, download_fn = _hf_client(work_dir)
                except Exception as err:
                    log(f"HF client unavailable ({type(err).__name__}); falling back to Drive backups")
            with ThreadPoolExecutor(max_workers=2) as pool:
                futs = [
                    pool.submit(
                        _stage_file,
                        item=item,
                        repo=actual_repo,
                        revision=lock["revision"],
                        local_dir=local_dir,
                        drive_backup_root=drive_backup_root,
                        download_fn=download_fn,
                        log=log,
                    )
                    for item in pending
                ]
                for fut in futs:
                    fut.result()
        write_json_atomic(marker, expected_map)

    first_shard = local_dir / lock["files"][0]["name"]
    projector_path: Path | None = None
    if enable_vision:
        if repo in (DEFAULT_REPO, IMATRIX_REPO):
            proj_item = dict(VISION_PROJECTOR_SPEC)
            proj_repo = VISION_PROJECTOR_SPEC["repo"]
            proj_rev = VISION_PROJECTOR_SPEC["revision"]
        else:
            if download_fn is None:
                HfApi, download_fn = _hf_client(work_dir)
            else:
                from huggingface_hub import HfApi
            info = HfApi().model_info(repo, files_metadata=True)
            proj_repo = repo
            proj_rev = info.sha
            mmproj_entries = [
                e for e in info.siblings
                if e.rfilename.endswith(".gguf") and "mmproj" in Path(e.rfilename).name.lower()
            ]
            if not mmproj_entries:
                log(f"No mmproj GGUF found in {repo}; falling back to default F16 mmproj")
                proj_item = dict(VISION_PROJECTOR_SPEC)
                proj_repo = VISION_PROJECTOR_SPEC["repo"]
                proj_rev = VISION_PROJECTOR_SPEC["revision"]
            else:
                chosen = sorted(
                    mmproj_entries,
                    key=lambda e: (0 if "f16" in e.rfilename.lower() or "bf16" in e.rfilename.lower() else 1, e.rfilename),
                )[0]
                lfs = getattr(chosen, "lfs", None)
                sha = getattr(lfs, "sha256", None) or (lfs.get("oid") if isinstance(lfs, dict) else "")
                proj_item = {"name": chosen.rfilename, "size": int(chosen.size), "sha256": sha}
        if proj_item is not None:
            if download_fn is None and not _valid_file(local_dir / proj_item["name"], proj_item):
                try:
                    _, download_fn = _hf_client(work_dir)
                except Exception:
                    download_fn = None
            projector_path = _stage_file(
                item=proj_item,
                repo=proj_repo,
                revision=proj_rev,
                local_dir=local_dir,
                drive_backup_root=drive_backup_root,
                download_fn=download_fn,
                log=log,
            )

    _cleanup_hf_scratch(local_dir)
    return first_shard, projector_path


def _valid_gguf_sidecar(path: Path, sha_path: Path | None = None) -> bool:
    if not path.is_file() or path.stat().st_size < 500_000_000:
        return False
    try:
        with path.open("rb") as handle:
            if handle.read(4) != b"GGUF":
                return False
    except OSError:
        return False
    if sha_path is not None and sha_path.is_file():
        expected = sha_path.read_text(encoding="utf-8", errors="replace").strip().split()[0].lower()
        if expected and digest_file(path) != expected:
            return False
    return True


def _ensure_mtp_converter(
    *,
    work_dir: Path,
    llama_repo: str,
    llama_commit: str,
    conv_log: Path,
) -> Path:
    commit = (llama_commit or "").strip() or DEFAULT_LLAMA_COMMIT
    full_source = Path(work_dir) / "llama.cpp"
    if (full_source / "convert_hf_to_gguf.py").is_file() and (full_source / "gguf-py").is_dir():
        return full_source

    conv_dir = Path(work_dir) / "llama-conv"
    stamp = conv_dir / ".ct_conv_commit"
    if (
        (conv_dir / "convert_hf_to_gguf.py").is_file()
        and (conv_dir / "conversion").is_dir()
        and (conv_dir / "gguf-py").is_dir()
        and stamp.is_file()
        and stamp.read_text(encoding="utf-8", errors="replace").strip() == commit
    ):
        return conv_dir

    shutil.rmtree(conv_dir, ignore_errors=True)
    conv_dir.mkdir(parents=True, exist_ok=True)
    _run_logged(["git", "-C", str(conv_dir), "init"], conv_log)
    _run_logged(["git", "-C", str(conv_dir), "remote", "add", "origin", llama_repo], conv_log)
    _run_logged(["git", "-C", str(conv_dir), "sparse-checkout", "init", "--no-cone"], conv_log)
    _run_logged(
        [
            "git", "-C", str(conv_dir), "sparse-checkout", "set", "--no-cone",
            "/convert_hf_to_gguf.py", "/conversion", "/gguf-py",
        ],
        conv_log,
    )
    _run_logged(
        ["git", "-C", str(conv_dir), "fetch", "--depth", "1", "--filter=blob:none", "origin", commit],
        conv_log,
    )
    _run_logged(["git", "-C", str(conv_dir), "checkout", "--detach", commit], conv_log)
    stamp.write_text(commit + "\n", encoding="utf-8")
    return conv_dir


def _ensure_mtp_python_deps(conv_log: Path) -> None:
    missing: list[str] = []
    for mod_name, pkg_name in (
        ("numpy", "numpy"),
        ("requests", "requests"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
        ("safetensors", "safetensors"),
        ("huggingface_hub", "huggingface_hub"),
        ("sentencepiece", "sentencepiece"),
        ("torch", "torch"),
        ("transformers", "transformers"),
    ):
        try:
            __import__(mod_name)
        except ImportError:
            missing.append(pkg_name)
    if missing:
        _run_logged([sys.executable, "-m", "pip", "install", "-q", "-U", *missing], conv_log)


def stage_mtp_sidecar(
    *,
    upstream_repo: str = UPSTREAM_HF_REPO,
    llama_repo: str = DEFAULT_LLAMA_REPO,
    llama_commit: str = DEFAULT_LLAMA_COMMIT,
    work_dir: Path,
    drive_backup_root: Path | None,
    mtp_mode: str = DEFAULT_MTP_MODE,
    log: Callable[[str], None],
) -> Path | None:
    """Stage or export standalone Q8_0 MTP draft GGUF (`~2.32 GiB`) onto local NVMe SSD + Google Drive cache."""
    mode = normalize_mtp_mode(mtp_mode)
    if mode == "off":
        return None

    local_dir = Path(work_dir) / "model"
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / MTP_SIDECAR_FILENAME
    local_sha_file = local_dir / f"{MTP_SIDECAR_FILENAME}.sha256"
    strict_env = os.environ.get("CT_DS9B_VERIFY_SHA256", "0") == "1"

    if _valid_gguf_sidecar(target, local_sha_file if strict_env else None):
        log(f"Reusing local SSD MTP sidecar: {target.name} ({target.stat().st_size / 1e9:.2f} GB)")
        return target

    drive_dir = (Path(drive_backup_root) / upstream_repo) if drive_backup_root is not None else None
    if drive_dir is not None:
        drive_file = drive_dir / MTP_SIDECAR_FILENAME
        drive_sha_file = drive_dir / f"{MTP_SIDECAR_FILENAME}.sha256"
        if drive_file.is_file() and drive_file.stat().st_size >= 500_000_000:
            try:
                log(f"Restoring cached MTP sidecar from Drive -> local SSD: {MTP_SIDECAR_FILENAME}")
                partial = target.with_name(f"{target.name}.{os.getpid()}.drive.partial")
                h = hashlib.sha256()
                with drive_file.open("rb") as src, partial.open("wb") as dst:
                    while chunk := src.read(16 * 1024 * 1024):
                        dst.write(chunk)
                        h.update(chunk)
                actual_sha = h.hexdigest()
                if drive_sha_file.is_file():
                    expected_sha = drive_sha_file.read_text(encoding="utf-8", errors="replace").strip().split()[0].lower()
                    if expected_sha and actual_sha != expected_sha:
                        partial.unlink(missing_ok=True)
                        raise RuntimeError(f"Drive MTP sidecar SHA256 mismatch for {MTP_SIDECAR_FILENAME}")
                if not _valid_gguf_sidecar(partial):
                    partial.unlink(missing_ok=True)
                    raise RuntimeError(f"Drive MTP sidecar failed GGUF header check: {MTP_SIDECAR_FILENAME}")
                partial.replace(target)
                local_sha_file.write_text(actual_sha + "\n", encoding="utf-8")
                log(f"Restored MTP sidecar ({target.stat().st_size / 1e9:.2f} GB) from Drive cache")
                return target
            except Exception as err:
                log(f"Drive MTP cache restore warning ({type(err).__name__}: {err}); exporting remotely from HF...")

    conv_log = Path(work_dir) / "mtp-convert.log"
    mtp_hf_cache = Path(work_dir) / "mtp-hf-cache"
    try:
        conv_dir = _ensure_mtp_converter(
            work_dir=work_dir,
            llama_repo=llama_repo,
            llama_commit=llama_commit,
            conv_log=conv_log,
        )
        _ensure_mtp_python_deps(conv_log)
        conv_env = _snapshot_env()
        conv_env["PYTHONPATH"] = f"{conv_dir / 'gguf-py'}:{conv_env.get('PYTHONPATH', '')}".rstrip(":")
        conv_env["HF_HUB_DISABLE_TELEMETRY"] = "1"
        conv_env["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
        conv_env["HF_HUB_CACHE"] = str(mtp_hf_cache)

        partial_out = local_dir / f"{MTP_SIDECAR_FILENAME}.{os.getpid()}.partial"
        last_err: Exception | None = None
        for attempt in (1, 2):
            partial_out.unlink(missing_ok=True)
            log(
                f"Exporting native MTP draft sidecar from {upstream_repo} via HTTP Range "
                f"(attempt {attempt}/2, outtype=q8_0; log: {conv_log})..."
            )
            try:
                _run_logged(
                    [
                        sys.executable,
                        str(conv_dir / "convert_hf_to_gguf.py"),
                        "--remote",
                        upstream_repo,
                        "--mtp",
                        "--outtype",
                        "q8_0",
                        "--outfile",
                        str(partial_out),
                    ],
                    conv_log,
                    env=conv_env,
                )
                if not _valid_gguf_sidecar(partial_out):
                    raise RuntimeError("Exported MTP GGUF is truncated or missing GGUF magic")
                last_err = None
                break
            except Exception as err:
                last_err = err
                partial_out.unlink(missing_ok=True)
                if attempt == 1:
                    log(f"MTP remote export attempt 1 failed ({type(err).__name__}); retrying once...")
        if last_err is not None:
            raise last_err

        partial_out.replace(target)
        sha = digest_file(target)
        local_sha_file.write_text(sha + "\n", encoding="utf-8")
        log(f"Exported native MTP sidecar ({target.stat().st_size / 1e9:.2f} GB, sha256={sha[:12]}): {target.name}")

        if drive_dir is not None:
            try:
                drive_dir.mkdir(parents=True, exist_ok=True)
                drive_partial = drive_dir / f"{MTP_SIDECAR_FILENAME}.{os.getpid()}.partial"
                shutil.copyfile(target, drive_partial)
                drive_partial.replace(drive_dir / MTP_SIDECAR_FILENAME)
                (drive_dir / f"{MTP_SIDECAR_FILENAME}.sha256").write_text(sha + "\n", encoding="utf-8")
                log(f"Cached MTP sidecar on Google Drive for instant future boots: {drive_dir / MTP_SIDECAR_FILENAME}")
            except OSError as err:
                log(f"MTP Drive cache save warning (non-fatal): {err}")

        return target
    except Exception as err:
        if mode == "auto":
            log(f"Warning: MTP sidecar export failed ({type(err).__name__}: {err}); continuing without MTP draft")
            return None
        raise
    finally:
        shutil.rmtree(mtp_hf_cache, ignore_errors=True)
