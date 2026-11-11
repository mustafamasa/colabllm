"""Hardware-adaptive 1M-context planning, GGUF staging, and llama.cpp runtime helpers for Qwen3.8-Flash-Next.

Targets `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` (`unsloth-UD-Q4_K_XL-V2`, `imatrix_micro3`
calibrated 4-bit MoE + 26.82 GiB 3-gram PLE table + `mmproj-Qwen3.8-Flash-Next-BF16.gguf` vision projector)
served via `llama-server` (`b11540`) with native `--api-key-file` Bearer authentication on a Colab G4 GPU.

Imported both locally (`src/colabllm/services/qwen-flash.py` CLI `plan` / validation / unit tests, stdlib only)
and inside the generated Colab notebook (`src/colabllm/services/qwen-flash.ipynb`).
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
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable

DEFAULT_REPO = "huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF"
DEFAULT_REVISION = "6e298f56319a5735f88ae952dd27222369f2f44a"
DEFAULT_QUANT = "unsloth-UD-Q4_K_XL-V2"
DEFAULT_ALIAS = "qwen3.8-flash-next"
DEFAULT_LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"
# ggml-org/llama.cpp tag b11540 (upstream-tested release for qwen4exp UD-Q4_K_XL-V2 + QSA k-pool indexer)
DEFAULT_LLAMA_COMMIT = "1e6f04a75e91da4bb66b5248f3fd4bdeff169d41"

NATIVE_ORIG_CONTEXT = 262_144
DEFAULT_CONTEXT = 1_048_576
MODEL_MAX_CONTEXT = 1_048_576
MIN_CONTEXT = 16_384
CONTEXT_ALIGN = 16_384
MODEL_NUM_LAYERS = 48
FULL_ATTN_LAYERS = 12  # block_count=48, full_attention_interval=4 -> 12 QSA full-attn + 36 SSM linear-attn

# Exact GGUF tensor footprint of `unsloth-UD-Q4_K_XL-V2` (measured from shard headers, in MiB = 1024**2 B):
#   - 4 shards total on disk: 111,334,654,784 B = 106,177.0 MiB (103.69 GiB)
#   - `per_layer_token_embd.weight` ([160, 320001536] 3-gram PLE table): 28,800,138,240 B = 27,466.0 MiB (26.82 GiB)
#     -> Kept in host DDR5 RAM (`--override-tensor per_layer_token_embd.weight=CPU`), 0 MiB GPU VRAM.
#   - GPU-eligible LLM trunk (embeddings + output + HC + 36 SSM + 12 QSA + 48x512 experts): 78,711 MiB (76.87 GiB)
#     * Routed MoE experts per layer (`ffn_{down,gate,up}_exps.weight`): 1,572,864,000 B = 1,500.0 MiB / layer
#     * Non-routed trunk (tok/out embd, HC, SSM, QSA, shared experts, routers): ~6,711 MiB
#   - BF16 vision projector (`mmproj-Qwen3.8-Flash-Next-BF16.gguf`): 907,543,008 B = 865.5 MiB (0.85 GiB)
PLE_TABLE_MIB = 27_466
TRUNK_GPU_WEIGHTS_MIB = 78_725
MOE_EXPERT_LAYER_MIB = 1_500
DENSE_FLOOR_MIB = TRUNK_GPU_WEIGHTS_MIB - MODEL_NUM_LAYERS * MOE_EXPERT_LAYER_MIB
VISION_MMPROJ_MIB = 866
SSM_RECURRENT_STATE_MIB = 116  # 36 linear-attention layers, constant O(1) w.r.t. context length
FIXED_GPU_RUNTIME_RESERVE_MIB = 5_632  # CUDA context + FlashAttn/QSA graph workspace + vision encoder headroom
PLE_RAM_HEADROOM_MIB = 20_480
SYSTEM_RAM_RESERVE_MIB = 10_240

QUANT_TRUNK_PROFILES_MIB: dict[str, tuple[int, int]] = {
    # quant_dir -> (gpu_trunk_mib_excluding_ple, per_layer_routed_moe_mib)
    "unsloth-UD-Q4_K_XL-V2": (78_725, 1_500),
    "unsloth-UD-Q4_K_XL": (78_725, 1_500),
    "unsloth-UD-IQ4_XS": (61_880, 1_150),
    "swift-IQ3_S": (42_830, 760),
    "swift-IQ3_XXS": (37_950, 660),
    "swift-IQ2_XS": (37_950, 660),
}

SUPPORTED_CACHE_TYPES = ("q8_0", "f16", "q4_0")
REASONING_MODES = ("on", "off", "auto")

VISION_PROJECTOR_SPEC: dict[str, Any] = {
    "name": "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
    "revision": DEFAULT_REVISION,
    "size": 907_543_008,
    "sha256": "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0",
}

DEFAULT_MODEL_LOCK: dict[str, Any] = {
    "repo": DEFAULT_REPO,
    "quant": DEFAULT_QUANT,
    "revision": DEFAULT_REVISION,
    "files": [
        {
            "name": "unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
            "size": 10_946_624,
            "sha256": "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082",
        },
        {
            "name": "unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
            "size": 49_859_583_136,
            "sha256": "ab283aba22daee321944dad65b09180810140fc75502c0791200cf01b85f9c75",
        },
        {
            "name": "unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
            "size": 49_376_141_504,
            "sha256": "2a8d2e69ba47fef6b42c953c07a24b5d4c59c1c2cdc89b2bc17e13e000a62fa0",
        },
        {
            "name": "unsloth-UD-Q4_K_XL-V2/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
            "size": 12_087_983_520,
            "sha256": "0d626450754c554235f93653b8f7c874c7c2d35380b9efab5b0852d616b84702",
        },
    ],
}

_OOM_PATTERNS = (
    "out of memory",
    "cuda_malloc failed",
    "cudamalloc failed",
    "failed to allocate",
    "ggml_gallocr_reserve_n: failed to allocate",
    "unable to allocate cuda",
    "insufficient memory",
)


def _ggml_bytes_per_elem(cache_type: str) -> float:
    ct = (cache_type or "q8_0").strip().lower()
    if ct in ("f16", "bf16"):
        return 2.0
    if ct == "q8_0":
        return 34.0 / 32.0  # 1.0625 B/elem (32 int8 + fp16 scale)
    if ct in ("q4_0", "q4_1", "iq4_nl"):
        return 18.0 / 32.0  # 0.5625 B/elem
    raise ValueError(f"Unsupported KV cache type {cache_type!r}; choose from {SUPPORTED_CACHE_TYPES}")


def qwen4exp_kv_bytes_per_token(cache_type: str = "q8_0", parallel: int = 1) -> float:
    """Exact per-token GPU KV + QSA k-pool indexer cache bytes across all 12 full-attention layers.

    From `llama.cpp` `b11540` (`models/qwen4exp.cpp` + `llama-memory-hybrid-idx.cpp`):
      - 12 full-attention layers (`block_count=48`, `full_attention_interval=4`):
        * Attention K + V: `head_count_kv=2 * (key_length=256 + value_length=256)` = 1,024 elems/token/layer
        * QSA Indexer K (`mem_idx`, MLA K-only, `kpool_row=2` raw|pooled, `head_size=128`): 256 elems/token/layer
      - Total: `12 * 1,280 = 15,360` elements/token.
    """
    bpe = _ggml_bytes_per_elem(cache_type)
    elems_per_token = FULL_ATTN_LAYERS * (2 * (256 + 256) + 256)
    return float(elems_per_token) * bpe * max(1, int(parallel))


def estimate_kv_cache_mib(context: int, cache_type: str = "q8_0", parallel: int = 1) -> float:
    kv_mib = (int(context) * qwen4exp_kv_bytes_per_token(cache_type, parallel)) / (1024.0 * 1024.0)
    return round(kv_mib + SSM_RECURRENT_STATE_MIB * max(1, int(parallel)), 1)


def parse_context_spec(raw: str | int | None) -> int | None:
    """Parse context token spec ('auto', '1m', '1048576', '800k', '512k', '256k'); None means 'auto'."""
    if raw is None:
        return None
    s = str(raw).strip().lower().replace("_", "").replace(",", "")
    if not s or s == "auto":
        return None
    if s in ("1m", "1024k"):
        val = 1_048_576
    elif s == "800k":
        val = 819_200
    elif s.endswith("k") and s[:-1].replace(".", "", 1).isdigit():
        val = int(round(float(s[:-1]) * 1024))
    elif s.isdigit():
        val = int(s)
    else:
        raise ValueError(f"Invalid context value {raw!r}; expected integer, '<N>k', '1m', or 'auto'")
    if val < MIN_CONTEXT or val > MODEL_MAX_CONTEXT:
        raise ValueError(f"Context {val} out of supported range [{MIN_CONTEXT}, {MODEL_MAX_CONTEXT}]")
    return val


def parse_cpu_moe_spec(raw: str | int | None) -> int | None:
    """Parse `--cpu-moe` ('auto' -> None, or integer in 0..48)."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s or s == "auto":
        return None
    if not s.isdigit():
        raise ValueError(f"Invalid --cpu-moe value {raw!r}; expected 'auto' or integer 0..{MODEL_NUM_LAYERS}")
    val = int(s)
    if val < 0 or val > MODEL_NUM_LAYERS:
        raise ValueError(f"--cpu-moe must be between 0 and {MODEL_NUM_LAYERS}, got {val}")
    return val


def compute_yarn_params(context: int, orig_ctx: int = NATIVE_ORIG_CONTEXT) -> tuple[str, float, int]:
    """Return `(rope_scaling, rope_scale, yarn_orig_ctx)` for a target `context`."""
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
    requested_cpu_moe: str | int | None = "auto",
    enable_vision: bool = True,
    parallel: int = 1,
    batch: int = 512,
    ubatch: int = 128,
) -> dict[str, Any]:
    """Compute VRAM/RAM-safe 1M context, YaRN scaling, PLE host-RAM residency, and `--n-cpu-moe` split."""
    ct = (cache_type or "q8_0").strip().lower()
    if ct not in SUPPORTED_CACHE_TYPES:
        raise ValueError(f"Unsupported cache_type {cache_type!r}; choose from {SUPPORTED_CACHE_TYPES}")
    explicit_ctx = parse_context_spec(requested_context)
    explicit_cpu_moe = parse_cpu_moe_spec(requested_cpu_moe)

    trunk_mib, moe_layer_mib = QUANT_TRUNK_PROFILES_MIB.get(
        quant, (TRUNK_GPU_WEIGHTS_MIB, MOE_EXPERT_LAYER_MIB)
    )
    mmproj_mib = VISION_MMPROJ_MIB if enable_vision else 0
    target_ctx = explicit_ctx if explicit_ctx is not None else MODEL_MAX_CONTEXT

    usable_ram_for_cpu_moe = max(0, int(ram_mib) - SYSTEM_RAM_RESERVE_MIB - PLE_TABLE_MIB - PLE_RAM_HEADROOM_MIB)
    max_cpu_moe_by_ram = min(MODEL_NUM_LAYERS, usable_ram_for_cpu_moe // max(1, moe_layer_mib))

    kv_mib = estimate_kv_cache_mib(target_ctx, ct, parallel)
    if explicit_cpu_moe is not None:
        n_cpu_moe = explicit_cpu_moe
    else:
        required_vram = trunk_mib + mmproj_mib + kv_mib + FIXED_GPU_RUNTIME_RESERVE_MIB
        deficit_mib = max(0.0, required_vram - float(vram_mib))
        needed_layers = int(math.ceil(deficit_mib / float(moe_layer_mib))) if deficit_mib > 0 else 0
        n_cpu_moe = min(MODEL_NUM_LAYERS, needed_layers)
        # If host RAM cannot hold that many CPU MoE layers and context is auto, shrink context to fit
        if n_cpu_moe > max_cpu_moe_by_ram and explicit_ctx is None:
            n_cpu_moe = max_cpu_moe_by_ram
            gpu_weights = trunk_mib - n_cpu_moe * moe_layer_mib + mmproj_mib
            vram_for_kv = max(
                0.0,
                float(vram_mib) - gpu_weights - FIXED_GPU_RUNTIME_RESERVE_MIB - SSM_RECURRENT_STATE_MIB * parallel,
            )
            raw_tokens = int((vram_for_kv * 1024 * 1024) / qwen4exp_kv_bytes_per_token(ct, parallel))
            aligned = (raw_tokens // CONTEXT_ALIGN) * CONTEXT_ALIGN
            target_ctx = max(MIN_CONTEXT, min(MODEL_MAX_CONTEXT, aligned))
            kv_mib = estimate_kv_cache_mib(target_ctx, ct, parallel)

    cpu_moe_mib = n_cpu_moe * moe_layer_mib
    gpu_weights_mib = max(DENSE_FLOOR_MIB, trunk_mib - cpu_moe_mib) + mmproj_mib
    ple_in_ram = int(ram_mib) >= (PLE_TABLE_MIB + cpu_moe_mib + PLE_RAM_HEADROOM_MIB)
    rope_scaling, rope_scale, yarn_orig_ctx = compute_yarn_params(target_ctx)
    free_vram_margin_mib = round(float(vram_mib) - gpu_weights_mib - kv_mib, 1)

    return {
        "mode": "full-gpu" if n_cpu_moe == 0 else f"hybrid-cpu-moe-{n_cpu_moe}",
        "quant": quant,
        "gpu_layers": "all",
        "n_cpu_moe": int(n_cpu_moe),
        "ple_in_ram": bool(ple_in_ram),
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
        "vram_mib": int(vram_mib),
        "ram_mib": int(ram_mib),
        "estimated_gpu_weights_mib": int(gpu_weights_mib),
        "estimated_cpu_moe_mib": int(cpu_moe_mib),
        "estimated_ple_ram_mib": PLE_TABLE_MIB,
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
    phys = max(4, cpus // 2)
    return (min(12, phys), min(16, max(8, cpus - 2)))


def build_server_args(
    *,
    server_bin: Path | str,
    model_path: Path | str,
    alias: str,
    port: int,
    key_file: Path | str,
    plan: dict[str, Any],
    projector_path: Path | str | None = None,
    reasoning: str = "on",
    threads: int | None = None,
    threads_batch: int | None = None,
) -> list[str]:
    """Construct exact loopback `llama-server` CLI arguments for Qwen3.8-Flash-Next."""
    ple_in_ram = bool(plan.get("ple_in_ram", True))
    load_mode = "none" if ple_in_ram else "mmap"
    lazy_mode = "off" if ple_in_ram else "on"
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
        "--api-key-file", str(key_file),
        "--no-context-shift",
        "--ctx-checkpoints", "32",
        "--cache-ram", "16384" if ple_in_ram else "4096",
    ]
    if ple_in_ram:
        # Allocate the 26.82 GiB PLE table + any CPU MoE layers in plain malloc DDR5 buffers instead of cudaMallocHost.
        args += ["--no-host"]
    if int(plan.get("n_cpu_moe") or 0) > 0:
        args += ["--n-cpu-moe", str(int(plan["n_cpu_moe"]))]
    if projector_path is not None and plan.get("enable_vision", True):
        args += ["--mmproj", str(projector_path)]
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


def evict_gguf_page_cache(model_dir: Path, log: Callable[[str], None] | None = None) -> int:
    """Advise Linux kernel (`POSIX_FADV_DONTNEED`) to drop GGUF file page cache before `--load-mode none`."""
    fadvise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if fadvise is None or dontneed is None or not Path(model_dir).is_dir():
        return 0
    evicted = 0
    for gguf in sorted(Path(model_dir).rglob("*.gguf")):
        if gguf.is_symlink() or not gguf.is_file():
            continue
        try:
            size = gguf.stat().st_size
            fd = os.open(str(gguf), os.O_RDONLY)
            try:
                fadvise(fd, 0, 0, dontneed)
            finally:
                os.close(fd)
            evicted += size
        except OSError:
            continue
    if log is not None and evicted > 0:
        log(f"Released GGUF OS page cache hint across {evicted / (1024**3):.1f} GiB before host-RAM load")
    return evicted


def _cmd_out(command: list[str], **kwargs: Any) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, **kwargs).strip()


def _run_logged(command: list[str], log_file: Path, env: dict[str, str] | None = None) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(str(x) for x in command)}\n")
        handle.flush()
        res = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
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
        return all(
            flag in help_text
            for flag in ("--lazy-mode", "--load-mode", "--api-key-file", "--n-cpu-moe", "--override-tensor")
        )
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
        # Reclaim ~3 GB build tree on local SSD right after archiving binaries
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
    if _valid_file(target, item, check_sha256=False) and (
        os.environ.get("CT_QWEN_VERIFY_SHA256", "0") != "1" or _valid_file(target, item, check_sha256=True)
    ):
        log(f"Reusing local SSD shard: {item['name']} ({item['size'] / 1e9:.2f} GB)")
        return target

    if download_fn is not None:
        file_tag = hashlib.sha256(item["name"].encode("utf-8")).hexdigest()[:12]
        scratch_dir = local_dir / f".hf-stage-{os.getpid()}-{file_tag}"
        shutil.rmtree(scratch_dir, ignore_errors=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)
        try:
            log(f"Downloading {item['name']} ({item['size'] / 1e9:.2f} GB) -> local NVMe SSD...")
            os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
            downloaded = Path(
                download_fn(
                    repo_id=repo,
                    filename=item["name"],
                    revision=revision,
                    local_dir=str(scratch_dir),
                )
            )
            strict_sha = os.environ.get("CT_QWEN_VERIFY_SHA256", "0") == "1" or not _xet_available()
            if not _valid_file(downloaded, item, check_sha256=strict_sha):
                downloaded.unlink(missing_ok=True)
                raise RuntimeError(f"Downloaded shard size/SHA256 mismatch: {item['name']}")
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

    raise RuntimeError(f"Unable to stage {item['name']} from Hugging Face or Drive backup")


def resolve_model_lock(
    *,
    repo: str,
    quant: str,
    service_dir: Path,
    work_dir: Path,
) -> tuple[dict[str, Any], Any]:
    """Return `(lock_dict, hf_hub_download_or_None)` for `(repo, quant)`."""
    if repo == DEFAULT_MODEL_LOCK["repo"] and quant == DEFAULT_MODEL_LOCK["quant"]:
        return DEFAULT_MODEL_LOCK, None

    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", f"{repo}--{quant}")
    lock_path = Path(service_dir) / f"model-lock-{slug}.json"
    if lock_path.is_file():
        cached = json.loads(lock_path.read_text(encoding="utf-8"))
        if cached.get("repo") == repo and cached.get("quant") == quant and cached.get("files"):
            return cached, None

    HfApi, download_fn = _hf_client(work_dir)
    info = HfApi().model_info(repo, files_metadata=True)
    prefix = quant.strip("/") + "/"
    files: list[dict[str, Any]] = []
    for entry in info.siblings:
        rfn = entry.rfilename
        base_lower = Path(rfn).name.lower()
        if (
            rfn.startswith(prefix)
            and rfn.endswith(".gguf")
            and "imatrix" not in base_lower
            and "mmproj" not in base_lower
            and "patch" not in base_lower
        ):
            lfs = getattr(entry, "lfs", None)
            sha256 = getattr(lfs, "sha256", None) or (lfs.get("oid") if isinstance(lfs, dict) else "")
            if not sha256:
                raise RuntimeError(f"Missing LFS SHA256 metadata for {rfn}")
            files.append({"name": rfn, "size": int(entry.size), "sha256": sha256})
    if not files:
        raise RuntimeError(f"No GGUF model shards found under {prefix!r} in {repo}")
    lock = {
        "repo": repo,
        "quant": quant,
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
    """Download & SHA256-verify all GGUF shards + optional BF16 vision projector onto local NVMe SSD."""
    lock, download_fn = resolve_model_lock(repo=repo, quant=quant, service_dir=service_dir, work_dir=work_dir)
    local_dir = Path(work_dir) / "model"
    local_dir.mkdir(parents=True, exist_ok=True)
    marker = local_dir / "verified.json"
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
        log(f"Reusing {len(lock['files'])} verified local SSD shards ({quant})")
    else:
        pending = [item for item in lock["files"] if not _valid_file(local_dir / item["name"], item)]
        if pending:
            needed_bytes = sum(int(item["size"]) for item in pending) + 8 * 1024**3
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
            # Download up to 2 shards concurrently with zero Xet chunk cache bloat
            with ThreadPoolExecutor(max_workers=2) as pool:
                futs = [
                    pool.submit(
                        _stage_file,
                        item=item,
                        repo=repo,
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
        if repo == DEFAULT_REPO:
            proj_item = dict(VISION_PROJECTOR_SPEC)
            proj_rev = VISION_PROJECTOR_SPEC["revision"]
        else:
            if download_fn is None:
                HfApi, download_fn = _hf_client(work_dir)
            else:
                from huggingface_hub import HfApi
            info = HfApi().model_info(repo, files_metadata=True)
            proj_rev = info.sha
            mmproj_entries = [
                e for e in info.siblings
                if e.rfilename.endswith(".gguf") and "mmproj" in Path(e.rfilename).name.lower()
            ]
            if not mmproj_entries:
                log(f"No mmproj GGUF found in {repo}; continuing text-only")
                proj_item = None
            else:
                chosen = sorted(
                    mmproj_entries,
                    key=lambda e: (0 if "swift" not in e.rfilename.lower() else 1, e.rfilename),
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
                repo=repo,
                revision=proj_rev,
                local_dir=local_dir,
                drive_backup_root=drive_backup_root,
                download_fn=download_fn,
                log=log,
            )

    _cleanup_hf_scratch(local_dir)
    return first_shard, projector_path
