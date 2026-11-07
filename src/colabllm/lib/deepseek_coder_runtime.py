"""Hardware-adaptive planning, model staging, and llama.cpp runtime helpers for DeepSeek-V4.1-Flash Coder.

Targets `Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-GGUF` (50% expert-pruned 192-expert
`IQ2_XXS` MoE trunk + `IQ4_NL` Engram conditional memory lookup table + `BF16` `deepseek41v` vision
projector) served via `llama-server` with native `--api-key-file` Bearer authentication.

Imported both locally (CLI `plan` / validation / unit tests, standard library only) and inside the
generated Colab notebook (`src/colabllm/services/deepseek-coder.ipynb`).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
from typing import Any

DEFAULT_REPO = "Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-GGUF"
DEFAULT_REVISION = "cfe5da6df1b4e22baf864e33f5132a6202c821e3"
DEFAULT_QUANT = "IQ2_XXS"
DEFAULT_ALIAS = "deepseek-v4.1-flash-coder"
DEFAULT_LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
DEFAULT_LLAMA_COMMIT = "3259ab6e5dd86fad1931962d784702cb53c7877b"

MODEL_MAX_CONTEXT = 1_048_576
MIN_CONTEXT = 16_384
CONTEXT_ALIGN = 16_384
MODEL_NUM_LAYERS = 40

# Exact weight footprints of the 192-expert IQ2_XXS Coder build (MiB = 1024**2 bytes):
#   - 5 trunk shards (00001..00005): 72,123,567,456 B = 68,782.4 MiB
#       * Dense layers (tok/out embd, MLA/DSA attention, HC, shared experts, engram gates): ~15,000 MiB
#       * Routed MoE experts (40 layers x 192 experts): ~53,783 MiB (~1,345 MiB / layer)
#   - BF16 deepseek41v mmproj: 972,208,096 B = 927.2 MiB
#   - IQ4_NL Engram table (shard 6): 110,595,290,816 B = 105,471.9 MiB
#       * On high-RAM hosts (>=128 GiB free after CPU MoE, e.g. G4 176 GiB / H100 167 GiB), loaded 100%
#         resident into DDR5 host RAM (`--load-mode none --lazy-mode off`) to eliminate cloud-disk page faults.
#       * On smaller-RAM hosts (e.g. A100-40G 83.5 GiB), streamed via `--load-mode mmap --lazy-mode on`.
TRUNK_TOTAL_MIB = 68_783
DENSE_WEIGHTS_MIB = 15_000
MOE_LAYER_MIB = 1_345
VISION_MMPROJ_MIB = 928
ENGRAM_FILE_MIB = 105_472
ENGRAM_RAM_HEADROOM_MIB = 25_600
FIXED_GPU_RUNTIME_RESERVE_MIB = 5_632
SYSTEM_RAM_RESERVE_MIB = 10_240
FULL_GPU_MIN_VRAM_MIB = 76_800
REASONING_EFFORT_LEVELS: tuple[str, ...] = ("minimal", "low", "medium", "high", "xhigh", "max")

ENGRAM_FILENAME = "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"
SHARD6_LINK_NAME = "DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-IQ2_XXS-00006-of-00006.gguf"
VISION_PROJECTOR_SPEC: dict[str, Any] = {
    "name": "mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf",
    "revision": DEFAULT_REVISION,
    "size": 972_208_096,
    "sha256": "0fdc4b7693f287e9665940cf42bd39c546ffe4f0def89c569d7d0e45d60d2df1",
}

DEFAULT_MODEL_LOCK: dict[str, Any] = {
    "repo": DEFAULT_REPO,
    "revision": DEFAULT_REVISION,
    "shard6_link": SHARD6_LINK_NAME,
    "engram_file": ENGRAM_FILENAME,
    "files": [
        {
            "name": "DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf",
            "size": 5_780_704,
            "sha256": "108e63d27240da2d99846250727a7b282e05a7fdf2448435e81904b9a98d8a4a",
        },
        {
            "name": "DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-IQ2_XXS-00002-of-00006.gguf",
            "size": 19_776_064_960,
            "sha256": "a503855d07b92bb9cf87583791788019cc5dcf57e46c9f80b99ae597b69fc880",
        },
        {
            "name": "DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-IQ2_XXS-00003-of-00006.gguf",
            "size": 19_939_989_696,
            "sha256": "a402c6d3ff9ebe5213cef3e59c6e7bb95f00f749d07f5491276acc35d01bf0db",
        },
        {
            "name": "DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-IQ2_XXS-00004-of-00006.gguf",
            "size": 19_682_533_472,
            "sha256": "ca5e46dada8e6dd915532ddb0b5dd779d7cc59402ef8cb08fc0030671c559f43",
        },
        {
            "name": "DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-IQ2_XXS-00005-of-00006.gguf",
            "size": 12_719_198_624,
            "sha256": "f375be5b2b08d0f4cacbc09462b97f98e674c5a91e7104815d3b4b9139204f74",
        },
        {
            "name": ENGRAM_FILENAME,
            "size": 110_595_290_816,
            "sha256": "3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3",
        },
    ],
}

CACHE_TYPE_BYTES: dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 1.0625,
    "q5_1": 0.75,
    "q5_0": 0.6875,
    "q4_1": 0.625,
    "q4_0": 0.5625,
    "iq4_nl": 0.5625,
}

STANDARD_CONTEXT_RUNGS: tuple[int, ...] = (
    1_048_576,
    917_504,
    786_432,
    655_360,
    524_288,
    393_216,
    262_144,
    196_608,
    131_072,
    65_536,
    32_768,
    16_384,
)

OOM_LOG_MARKERS: tuple[str, ...] = (
    "out of memory",
    "cuda_malloc failed",
    "failed to allocate",
    "ggml_backend_cuda_buffer_type_alloc_buffer",
    "unable to allocate",
    "not enough memory",
)

_SHARD_FIRST_RE = re.compile(r"^(.*)-0*1-of-(0*\d+)\.gguf$", re.IGNORECASE)


def dsv41_kv_bytes_per_token(cache_type: str = "f16", parallel: int = 1) -> float:
    """Exact persistent DSV4.1 compressed KV + lightning-indexer K cache bytes per token.

    In `llama_kv_cache_dsv4` (`LLM_ARCH_DEEPSEEK41`):
      - `kv_raw` uses sliding-window attention (`n_swa = 128`) across all 40 layers (~40 MiB fixed).
      - Only `kv_source_layers = [2, 8, 14, 20]` own context-proportional `[kv(512) | index_k(128)]` rows:
          * 3 CSA source layers (`ratio = 2`): `3 * 0.5 * 640 = 960` elements/token
          * 1 HCA source layer (`ratio = 1`):  `1 * 1.0 * 640 = 640` elements/token
        Total = `1,600` elements/token (`3,200 B/tok` at `f16`, i.e. `3,200 MiB` at 1M context).
    """
    elem_bytes = CACHE_TYPE_BYTES.get(cache_type.strip().lower(), 2.0)
    slots = max(1, int(parallel))
    return 1600.0 * elem_bytes * slots


def dsv41_total_vram_bytes_per_token(
    ubatch: int = 128,
    cache_type: str = "f16",
    parallel: int = 1,
) -> float:
    """Conservative per-token GPU budget including persistent KV cache + `sched_reserve` graph workspace.

    During worst-case reservation (`nt = n_ubatch > 32`), `deepseek41::build_attention41` allocates
    contiguous `comp_k` / `k_all` views plus `[n_comp, ubatch]` masks and candidate block scores
    (source analysis ~14 B/tok * ubatch; budgeted conservatively at 28 B/tok * ubatch + 15% safety margin).
    """
    kv_bpt = dsv41_kv_bytes_per_token(cache_type=cache_type, parallel=parallel)
    ub = max(32, int(ubatch))
    # quantized KV caches dequantize `ik` and `comp_k` to F32 during attention, adding temporary workspace
    is_quant_kv = cache_type.strip().lower() not in ("f16", "bf16", "f32")
    comp_k_bpt = 4096.0 if is_quant_kv else 2048.0
    mask_and_score_bpt = 28.0 * ub
    return (kv_bpt + comp_k_bpt + mask_and_score_bpt) * 1.15


def align_context_tokens(tokens: int, *, minimum: int = MIN_CONTEXT, maximum: int = MODEL_MAX_CONTEXT) -> int:
    """Clamp and floor-align token count to a clean 16K boundary (`1,048,576` max)."""
    if tokens >= maximum:
        return maximum
    if tokens <= minimum:
        return minimum
    aligned = (int(tokens) // CONTEXT_ALIGN) * CONTEXT_ALIGN
    return max(minimum, min(maximum, aligned))


def parse_context_spec(raw: str | int | None) -> int | None:
    """Parse `auto` -> `None`, or a positive integer token count (`16384..1048576`, supports `256k`/`1m`)."""
    if raw is None:
        return None
    text = str(raw).strip().lower().replace("_", "")
    if not text or text == "auto":
        return None
    mult = 1
    if text.endswith("k"):
        mult = 1024
        text = text[:-1]
    elif text.endswith("m"):
        mult = 1024 * 1024
        text = text[:-1]
    try:
        value = int(text) * mult
    except ValueError as err:
        raise ValueError(f"invalid --context value {raw!r}; expected 'auto' or integer in {MIN_CONTEXT}..{MODEL_MAX_CONTEXT}") from err
    if value < MIN_CONTEXT or value > MODEL_MAX_CONTEXT:
        raise ValueError(f"--context must be between {MIN_CONTEXT} and {MODEL_MAX_CONTEXT} (got {value})")
    return value


def parse_cpu_moe_spec(raw: str | int | None) -> int | None:
    """Parse `auto` -> `None`, or `0..40` MoE layers to offload to CPU RAM (`--n-cpu-moe`)."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or text == "auto":
        return None
    try:
        value = int(text)
    except ValueError as err:
        raise ValueError(f"invalid --cpu-moe-layers value {raw!r}; expected 'auto' or integer 0..{MODEL_NUM_LAYERS}") from err
    if value < 0 or value > MODEL_NUM_LAYERS:
        raise ValueError(f"--cpu-moe-layers must be between 0 and {MODEL_NUM_LAYERS} (got {value})")
    return value


def parse_ubatch_spec(raw: str | int | None) -> int | None:
    """Parse `auto` -> `None`, or a positive integer micro-batch size (`32..4096`)."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or text == "auto":
        return None
    try:
        value = int(text)
    except ValueError as err:
        raise ValueError(f"invalid --ubatch value {raw!r}; expected 'auto' or integer 32..4096") from err
    if value < 32 or value > 4096:
        raise ValueError(f"--ubatch must be between 32 and 4096 (got {value})")
    return value


def build_context_fallback_ladder(planned_context: int, *, allow_stepdown: bool = True) -> list[int]:
    """Ordered context candidates starting at `planned_context`, stepping down on startup OOM."""
    planned = max(MIN_CONTEXT, min(MODEL_MAX_CONTEXT, int(planned_context)))
    if not allow_stepdown:
        return [planned]
    rungs = [planned]
    for rung in STANDARD_CONTEXT_RUNGS:
        if MIN_CONTEXT <= rung < planned and rung not in rungs:
            rungs.append(rung)
    return rungs


def plan_hardware_config(
    *,
    vram_mib: int,
    ram_mib: int,
    requested_context: str | int | None = "auto",
    parallel: int = 1,
    cache_type: str = "f16",
    enable_vision: bool = True,
    batch: int = 4096,
    requested_ubatch: str | int | None = "auto",
    requested_cpu_moe: str | int | None = "auto",
) -> dict[str, Any]:
    """Compute maximum safe context window, GPU/CPU MoE placement, and Engram RAM residency."""
    vram_mib = max(0, int(vram_mib))
    ram_mib = max(0, int(ram_mib))
    parallel = max(1, int(parallel))
    batch = max(64, int(batch))
    cache_type = (cache_type or "f16").strip().lower()
    if cache_type not in CACHE_TYPE_BYTES:
        raise ValueError(f"unsupported KV cache type {cache_type!r}; choose from {', '.join(CACHE_TYPE_BYTES)}")

    explicit_ctx = parse_context_spec(requested_context)
    explicit_moe = parse_cpu_moe_spec(requested_cpu_moe)
    explicit_ubatch = parse_ubatch_spec(requested_ubatch)

    vision_mib = VISION_MMPROJ_MIB if enable_vision else 0
    full_weights_mib = TRUNK_TOTAL_MIB + vision_mib
    max_ram_for_moe_mib = max(0, ram_mib - SYSTEM_RAM_RESERVE_MIB)

    # CPU-only host (`--gpu none` or no CUDA device)
    if vram_mib <= 0:
        min_cpu_ram = full_weights_mib + SYSTEM_RAM_RESERVE_MIB + 2_048
        if ram_mib < min_cpu_ram:
            raise ValueError(
                f"Insufficient system RAM ({ram_mib / 1024:.1f} GiB) for CPU-only loading of the "
                f"72.1 GB Coder trunk (requires >= {min_cpu_ram / 1024:.1f} GiB RAM; Engram stays on SSD)."
            )
        ubatch = explicit_ubatch or min(batch, 128)
        bpt = dsv41_total_vram_bytes_per_token(ubatch=ubatch, cache_type=cache_type, parallel=parallel)
        budget_mib = max(0, ram_mib - full_weights_mib - SYSTEM_RAM_RESERVE_MIB)
        max_tokens = int((budget_mib * 1024 * 1024) // bpt)
        if explicit_ctx is not None:
            context = explicit_ctx
        else:
            context = align_context_tokens(max_tokens)
        engram_in_ram = (ram_mib - full_weights_mib) >= (ENGRAM_FILE_MIB + ENGRAM_RAM_HEADROOM_MIB)
        return {
            "mode": "cpu-only",
            "gpu_layers": "0",
            "n_cpu_moe": 0,
            "engram_in_ram": engram_in_ram,
            "context": context,
            "context_auto": explicit_ctx is None,
            "context_ladder": build_context_fallback_ladder(context, allow_stepdown=explicit_ctx is None),
            "batch": max(batch, ubatch),
            "ubatch": ubatch,
            "parallel": parallel,
            "cache_type": cache_type,
            "enable_vision": enable_vision,
            "vram_mib": vram_mib,
            "ram_mib": ram_mib,
            "estimated_gpu_weights_mib": 0,
            "estimated_cpu_moe_mib": full_weights_mib,
            "context_vram_budget_mib": budget_mib,
            "kv_cache_mib": round(context * dsv41_kv_bytes_per_token(cache_type, parallel) / (1024 * 1024), 1),
        }

    # GPU host: determine `ubatch` and `n_cpu_moe`
    if explicit_ubatch is not None:
        ubatch = min(batch, explicit_ubatch)
    elif vram_mib >= 90_000:
        # 96 GB G4 has ~22 GiB free above 100% GPU weights -> 512 ubatch doubles prefill speed (~82.5 GiB total at 1M ctx)
        ubatch = min(batch, 512)
    else:
        # 80 GB H100/A100 or smaller GPUs: 128 ubatch minimizes attention mask workspace to maximize n_ctx
        ubatch = min(batch, 128)

    bpt = dsv41_total_vram_bytes_per_token(ubatch=ubatch, cache_type=cache_type, parallel=parallel)
    min_ctx_vram_mib = math.ceil((MIN_CONTEXT * bpt) / (1024 * 1024))

    if explicit_moe is not None:
        n_cpu_moe = explicit_moe
    elif vram_mib >= FULL_GPU_MIN_VRAM_MIB:
        # 80 GB+ GPUs (G4 96 GB, H100/A100 80 GB): keep 100% of weights on GPU by default for maximum tok/s.
        n_cpu_moe = 0
        if explicit_ctx is not None:
            needed_ctx_mib = math.ceil((explicit_ctx * bpt) / (1024 * 1024))
            shortfall_mib = (full_weights_mib + FIXED_GPU_RUNTIME_RESERVE_MIB + needed_ctx_mib) - vram_mib
            if shortfall_mib > 0:
                # Offload just enough MoE layers so an explicitly requested 1M context fits on 80 GB VRAM.
                n_cpu_moe = min(MODEL_NUM_LAYERS, math.ceil(shortfall_mib / MOE_LAYER_MIB))
    else:
        # Sub-75 GiB GPU (e.g. 40 GB A100): keep dense attention + KV cache on GPU and offload MoE layers to RAM.
        target_ctx_tokens = explicit_ctx if explicit_ctx is not None else 262_144
        target_ctx_mib = math.ceil((target_ctx_tokens * bpt) / (1024 * 1024))
        desired_gpu_weights = vram_mib - FIXED_GPU_RUNTIME_RESERVE_MIB - target_ctx_mib
        min_gpu_weights = DENSE_WEIGHTS_MIB + vision_mib
        if desired_gpu_weights < min_gpu_weights:
            # Squeeze target context down to MIN_CONTEXT before giving up
            desired_gpu_weights = vram_mib - FIXED_GPU_RUNTIME_RESERVE_MIB - min_ctx_vram_mib
        if desired_gpu_weights < min_gpu_weights:
            raise ValueError(
                f"GPU VRAM ({vram_mib / 1024:.1f} GiB) is too small to hold dense attention weights + "
                f"runtime buffers (~{(min_gpu_weights + FIXED_GPU_RUNTIME_RESERVE_MIB) / 1024:.1f} GiB minimum; "
                f"use G4 96 GB, H100 80 GB, or A100 40/80 GB)."
            )
        shortfall_mib = max(0, full_weights_mib - desired_gpu_weights)
        n_cpu_moe = min(MODEL_NUM_LAYERS, math.ceil(shortfall_mib / MOE_LAYER_MIB))

    cpu_moe_mib = min(TRUNK_TOTAL_MIB - DENSE_WEIGHTS_MIB, n_cpu_moe * MOE_LAYER_MIB)
    if cpu_moe_mib > max_ram_for_moe_mib:
        raise ValueError(
            f"Hardware requires offloading {n_cpu_moe} MoE layers (~{cpu_moe_mib / 1024:.1f} GiB) to CPU RAM, "
            f"but only {max_ram_for_moe_mib / 1024:.1f} GiB usable RAM is available "
            f"(VRAM={vram_mib / 1024:.1f} GiB, RAM={ram_mib / 1024:.1f} GiB). Use `--gpu G4` or `--gpu H100`/`A100`."
        )

    gpu_weights_mib = full_weights_mib - cpu_moe_mib
    vram_for_ctx_mib = vram_mib - gpu_weights_mib - FIXED_GPU_RUNTIME_RESERVE_MIB
    if vram_for_ctx_mib < min_ctx_vram_mib:
        raise ValueError(
            f"Insufficient free GPU VRAM ({vram_for_ctx_mib} MiB left after {gpu_weights_mib} MiB GPU weights "
            f"and {FIXED_GPU_RUNTIME_RESERVE_MIB} MiB reserve); need at least {min_ctx_vram_mib} MiB for "
            f"{MIN_CONTEXT} context tokens."
        )

    max_auto_tokens = int((vram_for_ctx_mib * 1024 * 1024) // bpt)
    if explicit_ctx is not None:
        needed_mib = math.ceil((explicit_ctx * bpt) / (1024 * 1024))
        if needed_mib > vram_for_ctx_mib:
            raise ValueError(
                f"Requested --context {explicit_ctx} needs ~{needed_mib} MiB VRAM, exceeding available "
                f"budget ({vram_for_ctx_mib} MiB). Use `--context auto` (selects {align_context_tokens(max_auto_tokens)}) "
                f"or increase `--cpu-moe-layers`."
            )
        context = explicit_ctx
    else:
        context = align_context_tokens(max_auto_tokens)

    engram_in_ram = (ram_mib - cpu_moe_mib) >= (ENGRAM_FILE_MIB + ENGRAM_RAM_HEADROOM_MIB)
    # On startup OOM, step ubatch down (512 -> 256) before sacrificing context window
    ubatch_ladder = [ubatch]
    if explicit_ubatch is None and ubatch > 256:
        ubatch_ladder.append(256)
    mode = "full-gpu" if n_cpu_moe == 0 else f"hybrid-cpu-moe-{n_cpu_moe}"
    return {
        "mode": mode,
        "gpu_layers": "all",
        "n_cpu_moe": n_cpu_moe,
        "engram_in_ram": engram_in_ram,
        "context": context,
        "context_auto": explicit_ctx is None,
        "context_ladder": build_context_fallback_ladder(context, allow_stepdown=explicit_ctx is None),
        "batch": max(batch, ubatch),
        "ubatch": ubatch,
        "ubatch_ladder": ubatch_ladder,
        "parallel": parallel,
        "cache_type": cache_type,
        "enable_vision": enable_vision,
        "vram_mib": vram_mib,
        "ram_mib": ram_mib,
        "estimated_gpu_weights_mib": gpu_weights_mib,
        "estimated_cpu_moe_mib": cpu_moe_mib,
        "context_vram_budget_mib": vram_for_ctx_mib,
        "kv_cache_mib": round(context * dsv41_kv_bytes_per_token(cache_type, parallel) / (1024 * 1024), 1),
    }


def detect_host_ram_mib() -> int:
    """Read total system RAM (or cgroup memory.max cap when lower) in MiB on Linux/macOS."""
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
                # Ignore unlimited sentinels (e.g. 2^63 - page_size)
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


def derive_shard6_link_name(first_shard_relpath: str) -> str:
    """Given `.../Model-00001-of-00006.gguf`, return `.../Model-00006-of-00006.gguf`."""
    match = _SHARD_FIRST_RE.match(first_shard_relpath)
    if not match:
        raise ValueError(f"Expected split GGUF shard 1 matching '*-00001-of-0000N.gguf', got {first_shard_relpath!r}")
    prefix, total_str = match.group(1), match.group(2)
    width = len(total_str)
    total = int(total_str)
    return f"{prefix}-{total:0{width}d}-of-{total:0{width}d}.gguf"


def ensure_engram_shard_symlink(local_dir: Path, shard6_relpath: str, engram_relpath: str) -> Path:
    """Idempotently link the standalone Engram GGUF file to shard N so `llama.cpp` loads all shards."""
    link_path = local_dir / shard6_relpath
    engram_path = local_dir / engram_relpath
    if not engram_path.is_file():
        raise FileNotFoundError(f"Engram shard file does not exist: {engram_path}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    rel_target = os.path.relpath(engram_path, start=link_path.parent)
    if link_path.is_symlink():
        if os.readlink(link_path) == rel_target and link_path.is_file():
            return link_path
        link_path.unlink()
    elif link_path.exists():
        if link_path.stat().st_size == engram_path.stat().st_size:
            return link_path
        link_path.unlink()
    link_path.symlink_to(rel_target)
    return link_path


def default_cpu_threads(cpu_count: int | None = None) -> tuple[int, int]:
    """Return `(threads_decode, threads_batch)` tuned for CPU-side `engram_hash` + `IQ4_NL` row gathers."""
    logical = max(2, int(cpu_count or os.cpu_count() or 8))
    threads_decode = max(4, min(16, logical // 2)) if logical >= 8 else logical
    threads_batch = max(threads_decode, min(32, logical))
    return threads_decode, threads_batch


def normalize_reasoning_effort(raw: str | int | None) -> str | None:
    """Validate optional CLI/API `reasoning_effort` (`minimal|low|medium|high|xhigh|max` or `1..100`)."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or text == "default":
        return None
    if text in REASONING_EFFORT_LEVELS:
        return text
    try:
        val = int(text)
    except ValueError as err:
        allowed = ", ".join(REASONING_EFFORT_LEVELS)
        raise ValueError(f"reasoning_effort must be in 1..100 or one of {allowed} (got {raw!r})") from err
    if val < 1 or val > 100:
        raise ValueError(f"reasoning_effort integer must be in 1..100, got {val}")
    return str(val)


def build_server_args(
    *,
    server_bin: Path | str,
    model_path: Path | str,
    alias: str,
    port: int,
    key_file: Path | str,
    plan: dict[str, Any],
    projector_path: Path | str | None = None,
    chat_template_file: Path | str | None = None,
    reasoning: str = "on",
    reasoning_effort: str | int | None = None,
    threads: int | None = None,
    threads_batch: int | None = None,
) -> list[str]:
    """Construct exact loopback `llama-server` command arguments for DeepSeek-V4.1-Flash Coder."""
    engram_in_ram = bool(plan.get("engram_in_ram", False))
    load_mode = "none" if engram_in_ram else "mmap"
    lazy_mode = "off" if engram_in_ram else "on"
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
        "-c", str(int(plan["context"])),
        "-np", str(int(plan["parallel"])),
        "--load-mode", load_mode,
        "--lazy-mode", lazy_mode,
        "--override-tensor", "per_layer_token_embd.weight=CPU",
        "-t", str(t_dec),
        "-tb", str(t_bat),
        "-ctk", str(plan["cache_type"]),
        "-ctv", str(plan["cache_type"]),
        "-fa", "on",
        "-b", str(int(plan["batch"])),
        "-ub", str(int(plan["ubatch"])),
        "--jinja",
    ]
    if chat_template_file is not None:
        args += ["--chat-template-file", str(chat_template_file)]
    args += [
        "--reasoning-format", "deepseek",
        "--api-key-file", str(key_file),
        "--no-context-shift",
        # Multi-turn agent prefix reuse: more SWA/compressed-KV checkpoints per slot + host prompt cache
        "--ctx-checkpoints", "32",
        "--cache-ram", "16384" if engram_in_ram else "4096",
    ]
    if engram_in_ram:
        # Without this, CPU-resident tensors land in a CUDA_Host (cudaMallocHost) pinned buffer, i.e. the
        # 103 GiB Engram table would be page-locked; plain malloc'd CPU buffers are faster to load and safer.
        args += ["--no-host"]
    if int(plan.get("n_cpu_moe") or 0) > 0:
        args += ["--n-cpu-moe", str(int(plan["n_cpu_moe"]))]
    if projector_path is not None and plan.get("enable_vision", True):
        args += ["--mmproj", str(projector_path)]
    reasoning_mode = (reasoning or "").strip().lower()
    if reasoning_mode in ("on", "off", "auto"):
        args += ["--reasoning", reasoning_mode]
    effort_norm = normalize_reasoning_effort(reasoning_effort)
    if effort_norm is not None:
        args += ["--reasoning-effort", effort_norm]
    return args


_GGUF_SCALAR_SIZES: dict[int, int] = {
    0: 1,   # UINT8
    1: 1,   # INT8
    2: 2,   # UINT16
    3: 2,   # INT16
    4: 4,   # UINT32
    5: 4,   # INT32
    6: 4,   # FLOAT32
    7: 1,   # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


def extract_gguf_chat_template(gguf_path: Path | str) -> str | None:
    """Read `tokenizer.chat_template` directly from a GGUF v2/v3 header using stdlib `struct`."""
    path = Path(gguf_path)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            hdr = f.read(24)
            if len(hdr) < 24 or hdr[:4] != b"GGUF":
                return None
            version, _n_tensors, n_kv = struct.unpack("<IQQ", hdr[4:24])
            if version not in (2, 3) or n_kv > 100_000:
                return None

            def _read_exact(n: int) -> bytes:
                buf = f.read(n)
                if len(buf) != n:
                    raise EOFError("unexpected EOF in GGUF metadata")
                return buf

            def _skip_value(vtype: int) -> None:
                fixed = _GGUF_SCALAR_SIZES.get(vtype)
                if fixed is not None:
                    f.seek(fixed, os.SEEK_CUR)
                    return
                if vtype == 8:  # STRING
                    (slen,) = struct.unpack("<Q", _read_exact(8))
                    f.seek(slen, os.SEEK_CUR)
                    return
                if vtype == 9:  # ARRAY
                    elem_type, count = struct.unpack("<IQ", _read_exact(12))
                    elem_fixed = _GGUF_SCALAR_SIZES.get(elem_type)
                    if elem_fixed is not None:
                        f.seek(elem_fixed * count, os.SEEK_CUR)
                    else:
                        for _ in range(count):
                            _skip_value(elem_type)
                    return
                raise ValueError(f"unknown GGUF value type {vtype}")

            for _ in range(n_kv):
                (klen,) = struct.unpack("<Q", _read_exact(8))
                key = _read_exact(klen).decode("utf-8", errors="replace")
                (vtype,) = struct.unpack("<I", _read_exact(4))
                if key == "tokenizer.chat_template" and vtype == 8:
                    (slen,) = struct.unpack("<Q", _read_exact(8))
                    return _read_exact(slen).decode("utf-8")
                _skip_value(vtype)
    except (OSError, EOFError, ValueError, struct.error):
        return None
    return None


def patch_deepseek_chat_template(template: str) -> str:
    """Expand DeepSeek-V4.1 `effort_map` for full OpenAI `reasoning_effort` compatibility & calibrated default `50`.

    Stock template only accepts `"low"`, `"high"`, `"max"` or `1..100` (raising a Jinja exception when
    OpenAI SDKs / Coding Agents pass `"medium"`, `"minimal"`, or `"xhigh"`), and defaults to `"high"` (75)
    instead of GSQ-RCO-Coder's calibrated `50`. Keeps `enable_thinking` and the full-width-bar `DSML` token structure intact so
    `llama-server`'s differential autoparser and DeepSeek V4.1 tool parser work identically.
    """
    patched = template.replace(
        '{%- if reasoning_effort is not defined or reasoning_effort is none -%}{%- set reasoning_effort = "high" -%}{%- endif -%}',
        '{%- if reasoning_effort is not defined or reasoning_effort is none -%}{%- set reasoning_effort = 50 -%}{%- endif -%}',
    )
    patched = patched.replace(
        '{%- set effort_map = {"low": 50, "high": 75, "max": 100} -%}',
        '{%- set effort_map = {"minimal": 25, "low": 50, "medium": 60, "high": 75, "xhigh": 90, "max": 100} -%}',
    )
    patched = patched.replace(
        'one of \\"low\\", \\"high\\", \\"max\\"',
        'one of \\"minimal\\", \\"low\\", \\"medium\\", \\"high\\", \\"xhigh\\", \\"max\\"',
    )
    return patched


def prepare_patched_chat_template(gguf_path: Path | str, out_path: Path | str) -> Path | None:
    """Extract & patch `tokenizer.chat_template` from `gguf_path`, writing `out_path` if present."""
    raw = extract_gguf_chat_template(gguf_path)
    if not raw:
        return None
    patched = patch_deepseek_chat_template(raw)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(patched, encoding="utf-8")
    return target


def should_evict_page_cache(plan: dict[str, Any]) -> bool:
    """Only safe when no weights are served from mmap'd trunk shards (CPU MoE layers would page-fault)."""
    return int(plan.get("n_cpu_moe") or 0) == 0 and plan.get("gpu_layers") == "all"


def evict_gguf_page_cache(
    model_dir: Path | str,
    *,
    keep_engram: bool = False,
    engram_filename: str = ENGRAM_FILENAME,
    log=print,
) -> int:
    """Drop reclaimable Linux Page Cache for `.gguf` files via `POSIX_FADV_DONTNEED` after weight load.

    When `engram_in_ram=True` (`--load-mode none --lazy-mode off`), all GPU trunk weights reside in VRAM
    and Engram resides in anonymous CPU RAM (`folio_mapped == false`), so evicting `.gguf` Page Cache
    reclaims ~68–170 GiB of duplicate file cache cleanly.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    root = Path(model_dir)
    if fadvise is None or dontneed is None or not root.is_dir():
        return 0
    engram_resolved = (root / engram_filename).resolve() if keep_engram else None
    reclaimed_files = 0
    reclaimed_bytes = 0
    for candidate in sorted(root.rglob("*.gguf")):
        if candidate.is_symlink() or not candidate.is_file():
            continue
        if engram_resolved is not None and candidate.resolve() == engram_resolved:
            continue
        try:
            fd = os.open(candidate, os.O_RDONLY)
            try:
                size = os.fstat(fd).st_size
                fadvise(fd, 0, 0, dontneed)
                reclaimed_files += 1
                reclaimed_bytes += size
            finally:
                os.close(fd)
        except OSError:
            continue
    if reclaimed_files:
        log(f"Released Linux Page Cache hints on {reclaimed_files} GGUF shard(s) ({reclaimed_bytes / (1024**3):.1f} GiB)")
    return reclaimed_files


def looks_like_oom(log_text: str) -> bool:
    """Return True when `llama-server` stderr/stdout indicates a CUDA/host memory allocation failure."""
    lowered = (log_text or "").lower()
    return any(marker in lowered for marker in OOM_LOG_MARKERS)


# --------------------------------------------------------------------------- Colab VM staging helpers
def write_json_atomic(path: Path | str, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)


def digest_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def require_space(path: Path | str, needed_bytes: int) -> None:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target).free
    if free < needed_bytes:
        raise RuntimeError(f"{target}: need {needed_bytes / 1e9:.1f} GB free on local SSD, have {free / 1e9:.1f} GB")


def _cmd_out(command: list[str], *, env: dict[str, str] | None = None, timeout: int = 60) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, env=env, timeout=timeout).strip()


def _run_logged(command: list[str], log_file: Path, *, env: dict[str, str] | None = None, timeout: int = 3600) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        res = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    if res.returncode:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(command[:4])} (see {log_file})")


def _restore_valid_bin(bin_dir: Path, env: dict[str, str], *, expect_cuda: bool) -> bool:
    server = bin_dir / "llama-server"
    if not server.is_file():
        return False
    try:
        server.chmod(0o755)
        _cmd_out([str(server), "--version"], env=env, timeout=30)
        if expect_cuda:
            devices = _cmd_out([str(server), "--list-devices"], env=env, timeout=30)
            if "CUDA" not in devices:
                return False
        help_text = _cmd_out([str(server), "--help"], env=env, timeout=30)
        return all(flag in help_text for flag in ("--lazy-mode", "--load-mode", "--api-key-file", "--n-cpu-moe"))
    except (OSError, subprocess.SubprocessError):
        return False


def prepare_llama_runtime(
    *,
    work_dir: Path,
    cache_root: Path,
    llama_repo: str = DEFAULT_LLAMA_REPO,
    llama_commit: str = DEFAULT_LLAMA_COMMIT,
    gpu_info: dict[str, Any] | None = None,
    log=print,
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    """Restore or compile pinned `llama-server` for the active GPU's CUDA architecture."""
    gpu = gpu_info or detect_gpu_info()
    has_cuda = bool(gpu.get("present") and gpu.get("cuda_arch"))
    cuda_arch = gpu["cuda_arch"] if has_cuda else "cpu"
    nvcc_ver = _cmd_out(["nvcc", "--version"], timeout=30) if has_cuda else "no-cuda"
    os_rel = Path("/etc/os-release").read_text(encoding="utf-8", errors="replace") if Path("/etc/os-release").is_file() else platform.platform()
    signature = {
        "llama_commit": llama_commit,
        "cuda_arch": cuda_arch,
        "cuda": nvcc_ver,
        "os_release": os_rel,
        "arch": platform.machine(),
        "glibc": list(platform.libc_ver()),
        "targets": ["llama-server"],
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
    cache_dir = cache_root / f"sm{cuda_arch}-{key}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    bin_dir = work_dir / "runtime" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=str(bin_dir) + ":" + os.environ.get("LD_LIBRARY_PATH", ""))
    local_manifest = work_dir / "runtime-manifest.json"
    if bin_dir.exists() and any(bin_dir.iterdir()):
        try:
            cached_sig = json.loads(local_manifest.read_text(encoding="utf-8")) if local_manifest.is_file() else None
        except (OSError, ValueError):
            cached_sig = None
        if cached_sig != signature:
            shutil.rmtree(bin_dir)
            bin_dir.mkdir(parents=True, exist_ok=True)

    archive = cache_dir / "runtime.tar"
    checksum = cache_dir / "runtime.sha256"
    if not _restore_valid_bin(bin_dir, env, expect_cuda=has_cuda) and archive.is_file() and checksum.is_file():
        log(f"Restoring prebuilt llama.cpp runtime archive from Drive: {cache_dir}")
        local_tar = work_dir / "runtime.tar"
        shutil.copyfile(archive, local_tar)
        if digest_file(local_tar) != checksum.read_text(encoding="utf-8").strip():
            local_tar.unlink(missing_ok=True)
            raise RuntimeError("Drive runtime cache SHA256 mismatch; refusing corrupt archive")
        with tarfile.open(local_tar) as bundle:
            bundle.extractall(work_dir / "runtime", filter="data")
        local_tar.unlink(missing_ok=True)

    if not _restore_valid_bin(bin_dir, env, expect_cuda=has_cuda):
        build_log = work_dir / "build.log"
        log(f"Building llama.cpp ({llama_commit[:10]}) for target={cuda_arch}; log={build_log}")
        source = work_dir / "llama.cpp-src"
        if (source / ".git").exists():
            shutil.rmtree(source)
        _run_logged(["git", "clone", "--no-checkout", "--filter=blob:none", llama_repo, str(source)], build_log)
        _run_logged(["git", "-C", str(source), "fetch", "--depth", "1", "origin", llama_commit], build_log)
        _run_logged(["git", "-C", str(source), "checkout", "--detach", llama_commit], build_log)
        cmake_args = [
            "cmake", "-S", str(source), "-B", str(source / "build"),
            "-DGGML_NATIVE=OFF", "-DCMAKE_BUILD_TYPE=Release",
            "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=OFF",
        ]
        if has_cuda:
            cmake_args += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}"]
        else:
            cmake_args += ["-DGGML_CUDA=OFF"]
        _run_logged(cmake_args, build_log)
        jobs = str(min(12, os.cpu_count() or 4))
        _run_logged(
            ["cmake", "--build", str(source / "build"), "--config", "Release", "-j", jobs, "--target", "llama-server"],
            build_log,
        )
        shutil.copytree(source / "build" / "bin", bin_dir, dirs_exist_ok=True, symlinks=False)
        for lib_candidate in (source / "build").rglob("*.so*"):
            if lib_candidate.is_file() and not (bin_dir / lib_candidate.name).exists():
                shutil.copy2(lib_candidate, bin_dir / lib_candidate.name)
        # Reclaim ~4-6 GB CUDA build artifacts immediately so local SSD has full room for 183.7 GB weights
        shutil.rmtree(source, ignore_errors=True)
        if not _restore_valid_bin(bin_dir, env, expect_cuda=has_cuda):
            raise RuntimeError(f"Compiled llama-server failed validation (expect_cuda={has_cuda})")
        local_tar = work_dir / "runtime.tar"
        with tarfile.open(local_tar, "w", dereference=True) as bundle:
            bundle.add(bin_dir, arcname="bin")
        sha = digest_file(local_tar)
        partial = cache_dir / f".runtime.tar.{os.getpid()}.partial"
        shutil.copyfile(local_tar, partial)
        partial.replace(archive)
        checksum.write_text(sha + "\n", encoding="utf-8")
        write_json_atomic(cache_dir / "manifest.json", signature)
        local_tar.unlink(missing_ok=True)
        log(f"Archived llama-server runtime to {cache_dir}")

    write_json_atomic(local_manifest, signature)
    return bin_dir, env, gpu


def _hf_client(work_dir: Path):
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    try:
        import hf_xet  # noqa: F401
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        _run_logged([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub", "hf_xet"], work_dir / "download.log", timeout=600)
        from huggingface_hub import HfApi, hf_hub_download
    return HfApi, hf_hub_download


def _resolve_model_lock(repo: str, quant: str, service_dir: Path, work_dir: Path) -> dict[str, Any]:
    safe_repo = re.sub(r"[^A-Za-z0-9._-]+", "_", repo)
    lock_path = service_dir / f"model-lock-{safe_repo}-{quant}.json"
    if lock_path.is_file():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("repo") == repo:
            return lock
    if repo == DEFAULT_REPO and quant.upper() == DEFAULT_QUANT:
        write_json_atomic(lock_path, DEFAULT_MODEL_LOCK)
        return DEFAULT_MODEL_LOCK

    HfApi, _ = _hf_client(work_dir)
    info = HfApi().model_info(repo, files_metadata=True)
    trunk_files = []
    engram_entry = None
    mmproj_entry = None
    q_upper = quant.upper()
    for entry in info.siblings:
        rfn = entry.rfilename
        if not rfn.lower().endswith(".gguf"):
            continue
        if not entry.lfs:
            raise RuntimeError(f"Expected LFS SHA256 metadata for {rfn}")
        rec = {"name": rfn, "size": entry.size, "sha256": entry.lfs.sha256}
        lower = rfn.lower()
        if "mmproj" in lower:
            mmproj_entry = rec
        elif "engram" in lower:
            engram_entry = rec
        elif q_upper in rfn.upper():
            trunk_files.append(rec)
    trunk_files.sort(key=lambda item: item["name"])
    if not trunk_files:
        raise RuntimeError(f"No GGUF shards matching quant {quant!r} found in {repo}")
    shard6_link = None
    first_name = trunk_files[0]["name"]
    if _SHARD_FIRST_RE.match(first_name):
        expected_last = derive_shard6_link_name(first_name)
        if not any(item["name"] == expected_last for item in trunk_files):
            if engram_entry is None:
                raise RuntimeError(f"Split GGUF {first_name} is missing final shard {expected_last} and no Engram GGUF found")
            trunk_files.append(engram_entry)
            shard6_link = expected_last
    lock = {
        "repo": repo,
        "revision": info.sha,
        "shard6_link": shard6_link,
        "engram_file": engram_entry["name"] if engram_entry else None,
        "files": trunk_files,
        "mmproj": mmproj_entry,
    }
    write_json_atomic(lock_path, lock)
    return lock


def _xet_available() -> bool:
    try:
        import hf_xet  # noqa: F401
    except ImportError:
        return False
    return True


def _stage_one_file(
    item: dict[str, Any],
    *,
    repo: str,
    revision: str,
    local_dir: Path,
    drive_backup_root: Path | None,
    download_fn,
    verified_records: dict[str, dict[str, Any]],
    log=print,
) -> Path:
    target = local_dir / item["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    prev = verified_records.get(item["name"])
    if (
        target.is_file()
        and target.stat().st_size == item["size"]
        and prev
        and prev.get("sha256") == item["sha256"]
        and prev.get("mtime_ns") == target.stat().st_mtime_ns
    ):
        return target
    if target.is_file() and target.stat().st_size == item["size"] and digest_file(target) == item["sha256"]:
        return target

    if download_fn is not None:
        log(f"Downloading {item['name']} ({item['size'] / 1e9:.2f} GB) -> local SSD")
        file_tag = hashlib.sha256(item["name"].encode("utf-8")).hexdigest()[:12]
        scratch_dir = local_dir / f".hf-stage-{os.getpid()}-{file_tag}"
        shutil.rmtree(scratch_dir, ignore_errors=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = Path(
                download_fn(
                    repo_id=repo,
                    filename=item["name"],
                    revision=revision,
                    local_dir=str(scratch_dir),
                )
            )
            if not downloaded.is_file() or downloaded.stat().st_size != item["size"]:
                downloaded.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded file size mismatch: {item['name']}")
            # hf_xet already verifies every chunk against its content-addressed Merkle hash during reconstruction;
            # a second full-file SHA256 pass costs ~5 min on the 110.6 GB Engram file, so it is opt-in.
            strict = os.environ.get("CT_DS_VERIFY_SHA256", "0") == "1" or not _xet_available()
            if strict and digest_file(downloaded) != item["sha256"]:
                downloaded.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded file SHA256 mismatch: {item['name']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            downloaded.replace(target)
        finally:
            shutil.rmtree(scratch_dir, ignore_errors=True)
        log(f"Verified {'SHA256' if strict else 'xet chunk hashes + size'} ({item['size'] / 1e9:.2f} GB): {item['name']}")
        return target

    if drive_backup_root is not None:
        backup = drive_backup_root / repo / item["name"]
        if backup.is_file() and backup.stat().st_size == item["size"]:
            log(f"Copying Drive backup -> local SSD: {item['name']}")
            partial = target.with_name(f".{target.name}.{os.getpid()}.partial")
            shutil.copyfile(backup, partial)
            if partial.stat().st_size != item["size"] or digest_file(partial) != item["sha256"]:
                partial.unlink(missing_ok=True)
                raise RuntimeError(f"Drive backup SHA256 mismatch: {item['name']}")
            partial.replace(target)
            return target
    raise RuntimeError(f"Unable to stage {item['name']} from Hugging Face or Drive backup")


def stage_model_and_projector(
    *,
    repo: str,
    quant: str,
    enable_vision: bool,
    service_dir: Path,
    work_dir: Path,
    drive_backup_root: Path | None = None,
    log=print,
) -> tuple[Path, Path | None]:
    """Download/verify all trunk shards + Engram shard-6 symlink + optional BF16 mmproj on local NVMe SSD."""
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    lock = _resolve_model_lock(repo, quant, service_dir, work_dir)
    local_dir = work_dir / "model"
    local_dir.mkdir(parents=True, exist_ok=True)
    verified_path = local_dir / "verified.json"
    try:
        verified_records: dict[str, dict[str, Any]] = (
            json.loads(verified_path.read_text(encoding="utf-8")) if verified_path.is_file() else {}
        )
    except (OSError, ValueError):
        verified_records = {}

    items = list(lock["files"])
    mmproj_spec: dict[str, Any] | None = None
    if enable_vision:
        if repo == DEFAULT_REPO and lock["revision"] == DEFAULT_REVISION:
            mmproj_spec = VISION_PROJECTOR_SPEC
        else:
            mmproj_spec = lock.get("mmproj")
        if mmproj_spec is not None:
            items.append(mmproj_spec)

    missing_bytes = 0
    for item in items:
        target = local_dir / item["name"]
        if not (target.is_file() and target.stat().st_size == item["size"]):
            missing_bytes += int(item["size"])
    if missing_bytes > 0:
        # Reserve 6 GiB headroom above missing bytes while respecting Colab's ~253 GB local NVMe SSD
        require_space(work_dir, missing_bytes + 6 * 1024**3)

    download_fn = None
    if missing_bytes > 0 or not verified_records:
        try:
            _, download_fn = _hf_client(work_dir)
        except Exception as err:
            log(f"HF client init warning ({type(err).__name__}: {err}); falling back to local/Drive files")

    # Order largest-first across 2 isolated workers so the 110.6 GB Engram shard overlaps the 73.1 GB trunk shards
    ordered_items = sorted(items, key=lambda it: int(it["size"]), reverse=True)
    if ordered_items:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    _stage_one_file,
                    it,
                    repo=repo,
                    revision=lock["revision"],
                    local_dir=local_dir,
                    drive_backup_root=drive_backup_root,
                    download_fn=download_fn,
                    verified_records=verified_records,
                    log=log,
                )
                for it in ordered_items
            ]
            for fut in futures:
                fut.result()

    # Purge transient HF cache directories so local NVMe SSD retains maximum free space
    shutil.rmtree(local_dir / ".cache", ignore_errors=True)
    shutil.rmtree(Path.home() / ".cache" / "huggingface" / "xet", ignore_errors=True)

    if lock.get("shard6_link") and lock.get("engram_file"):
        link = ensure_engram_shard_symlink(local_dir, lock["shard6_link"], lock["engram_file"])
        log(f"Engram shard symlink ready: {link.name} -> {lock['engram_file']}")

    updated_records = {}
    for it in items:
        p = local_dir / it["name"]
        updated_records[it["name"]] = {
            "size": it["size"],
            "sha256": it["sha256"],
            "mtime_ns": p.stat().st_mtime_ns,
        }
    write_json_atomic(verified_path, updated_records)

    first_shard = local_dir / lock["files"][0]["name"]
    projector_path = (local_dir / mmproj_spec["name"]) if mmproj_spec is not None else None
    return first_shard, projector_path
