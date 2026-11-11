#!/usr/bin/env python3
"""Qwen3.8-Flash-Next (UD-Q4_K_XL-V2, 1M YaRN 4.0x, q8_0 KV, BF16 Vision) on a Colab G4 GPU.

Deploys `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` (`unsloth-UD-Q4_K_XL-V2` calibrated
4-bit MoE + 26.82 GiB 3-gram PLE lookup table + `mmproj-Qwen3.8-Flash-Next-BF16.gguf` vision projector)
via `llama-server` (`b11540`) with native `--api-key-file` Bearer authentication.

Key properties:
  * Hardware: `G4` (`RTX PRO 6000 Blackwell` `sm_120`, `97,887 MiB ≈ 95.6 GiB` VRAM, `176 GiB` DDR5 RAM)
    requested WITHOUT `--high-mem`.
  * 1M (`1,048,576`) context on G4:
      - `per_layer_token_embd.weight` (`26.82 GiB` 3-gram PLE table) pinned in DDR5 host RAM via
        `--override-tensor per_layer_token_embd.weight=CPU --load-mode none --lazy-mode off --no-host`
        (0 MiB GPU VRAM, zero disk page faults).
      - `-ctk q8_0 -ctv q8_0 -fa on` compresses the 1M QSA attention + k-pool indexer cache from
        `30.1 GiB` (`f16`) to `16.05 GiB` (`q8_0`).
      - Hardware-adaptive `--cpu-moe auto` offloads just `3/48` routed expert layers (`4.39 GiB`) to
        DDR5 host RAM while keeping `45/48` layers + all attention/SSM/shared-expert weights + BF16
        vision projector on GPU (`73.3 GiB` VRAM), leaving `~6.2 GiB` clean VRAM headroom.
  * Runtime cache: pinned `llama.cpp` (`b11540`) built for CUDA `sm_120` and cached at
    `/content/drive/MyDrive/models/.runtimes/llama.cpp/sm120-<fingerprint>/runtime.tar` (+ `sha256`).
  * Model staging: concurrent resumable HF download (`HF_XET_CHUNK_CACHE_SIZE_BYTES=0`, excluding
    `imatrix_micro3.gguf`) to `/content/qwen-flash/model` with SHA256 verification, Drive backup
    fallback, and `POSIX_FADV_DONTNEED` page-cache release before host-RAM load.
  * Auth: native `llama-server --api-key-file` (`auth="builtin"`).
  * Default domain when `--domain` is passed without a value: `qwen.<base-domain>`.

Usage:
  colabllm qwen-flash up --domain                 # G4 + UD-Q4_K_XL-V2 + 1M context + fixed domain
  colabllm qwen-flash up                          # G4 + global transport / quick tunnel
  colabllm qwen-flash up --context 512k           # 512K context (0 CPU MoE layers, 100% GPU trunk)
  colabllm qwen-flash up --quant unsloth-UD-IQ4_XS  # Smaller 4-bit quant (1M fits 100% on GPU, n_cpu_moe=0)
  colabllm qwen-flash plan                        # Preview 1M VRAM/RAM & MoE placement across GPU presets
  colabllm qwen-flash models                      # Query live server alias, allocated n_ctx & vision
  colabllm qwen-flash ps | info [-i ID] | logs [-i ID] -f | status [-i ID] | down [-i ID | --all]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cli_support import UsageError, http_timeout, resolve_endpoint, run_cli, safe_urlopen  # noqa: E402
from colabsvc import Service, State, main  # noqa: E402
from qwen_flash_runtime import (  # noqa: E402
    DEFAULT_ALIAS,
    DEFAULT_CONTEXT,
    DEFAULT_LLAMA_COMMIT,
    DEFAULT_LLAMA_REPO,
    DEFAULT_QUANT,
    DEFAULT_REPO,
    MIN_CONTEXT,
    MODEL_MAX_CONTEXT,
    MODEL_NUM_LAYERS,
    REASONING_MODES,
    SUPPORTED_CACHE_TYPES,
    parse_context_spec,
    parse_cpu_moe_spec,
    plan_hardware_config,
)

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

HARDWARE_PRESETS: dict[str, tuple[int, int]] = {
    "G4": (97_887, 180_224),
    "H100": (81_559, 171_008),
    "A100-80G": (81_920, 171_008),
    "A100-40G": (40_960, 85_504),
}

SERVICE = Service(
    name="qwen-flash",
    notebook="qwen-flash.ipynb",
    service_port=8000,
    auth="builtin",                   # llama-server --api-key-file handles Bearer auth natively
    public_port=8000,
    gpu="G4",                         # G4 without --high-mem (high_mem=False)
    high_mem=False,
    default_hostname="qwen",
    ready_timeout=5400,               # first-time 104 GiB staging + llama.cpp sm_120 build + 1M load
    env={
        "CT_QWEN_REPO": DEFAULT_REPO,
        "CT_QWEN_QUANT": DEFAULT_QUANT,
        "CT_QWEN_ALIAS": DEFAULT_ALIAS,
        "CT_QWEN_CONTEXT": str(DEFAULT_CONTEXT),
        "CT_QWEN_CACHE_TYPE": "q8_0",
        "CT_QWEN_CPU_MOE": "auto",
        "CT_QWEN_PARALLEL": "1",
        "CT_QWEN_BATCH": "512",
        "CT_QWEN_UBATCH": "128",
        "CT_QWEN_REASONING": "on",
        "CT_QWEN_VISION": "1",
        "CT_LLAMA_REPO": DEFAULT_LLAMA_REPO,
        "CT_LLAMA_COMMIT": DEFAULT_LLAMA_COMMIT,
    },
)

UP_ENV_FLAGS = {
    "--repo": "CT_QWEN_REPO",
    "--quant": "CT_QWEN_QUANT",
    "--alias": "CT_QWEN_ALIAS",
    "--context": "CT_QWEN_CONTEXT",
    "--cache-type": "CT_QWEN_CACHE_TYPE",
    "--cpu-moe": "CT_QWEN_CPU_MOE",
    "--parallel": "CT_QWEN_PARALLEL",
    "--batch": "CT_QWEN_BATCH",
    "--ubatch": "CT_QWEN_UBATCH",
    "--reasoning": "CT_QWEN_REASONING",
    "--llama-commit": "CT_LLAMA_COMMIT",
}
_BASE_ENV = dict(SERVICE.env)


def _validate_positive_int(raw: str) -> int:
    try:
        val = int(str(raw).strip())
    except ValueError as err:
        raise argparse.ArgumentTypeError("expected a positive integer") from err
    if val < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return val


def _validate_context_arg(raw: str) -> str:
    try:
        parsed = parse_context_spec(raw)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err
    return "auto" if parsed is None else str(parsed)


def _validate_cpu_moe_arg(raw: str) -> str:
    try:
        parsed = parse_cpu_moe_spec(raw)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err
    return "auto" if parsed is None else str(parsed)


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(
        State(SERVICE).load(),
        url_override,
        token_override,
        url_env="QWEN_FLASH_URL",
        token_env="QWEN_FLASH_TOKEN",
    )


def _api_get_json(base_url: str, path: str, token: str, timeout: float) -> Any:
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": BROWSER_UA,
        },
    )
    try:
        with safe_urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")[:300]
        raise UsageError(f"Remote request failed ({path} -> HTTP {err.code}): {detail}") from err
    except urllib.error.URLError as err:
        raise UsageError(f"Cannot connect to {base_url}{path}: {err.reason}") from err


def cmd_models(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.token)
    timeout = http_timeout(30.0)
    models_payload = _api_get_json(base_url, "/v1/models", token, timeout)
    props_payload: dict[str, Any] = {}
    try:
        props_payload = _api_get_json(base_url, "/props", token, timeout)
    except UsageError:
        props_payload = {}
    default_gen = props_payload.get("default_generation_settings") or {}
    result = {
        "service": SERVICE.name,
        "base_url": base_url.rstrip("/") + "/v1",
        "models": [m.get("id") for m in (models_payload.get("data") or []) if isinstance(m, dict)],
        "context_allocated": default_gen.get("n_ctx"),
        "total_slots": props_payload.get("total_slots"),
        "modalities": props_payload.get("modalities") or {},
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    common_kwargs = dict(
        quant=args.quant,
        requested_context=args.context,
        cache_type=args.cache_type,
        requested_cpu_moe=args.cpu_moe,
        enable_vision=not args.no_vision,
        parallel=args.parallel,
        batch=args.batch,
        ubatch=args.ubatch,
    )
    if args.vram_gb is not None or args.ram_gb is not None:
        vram_mib = int(round((args.vram_gb or 0.0) * 1024))
        ram_mib = int(round((args.ram_gb if args.ram_gb is not None else 176.0) * 1024))
        try:
            planned = plan_hardware_config(vram_mib=vram_mib, ram_mib=ram_mib, **common_kwargs)
        except ValueError as err:
            raise UsageError(str(err)) from err
        print(json.dumps(planned, indent=2, ensure_ascii=False))
        return 0

    matrix: dict[str, Any] = {}
    for preset_name, (vram_mib, ram_mib) in HARDWARE_PRESETS.items():
        if args.preset and args.preset != preset_name:
            continue
        try:
            matrix[preset_name] = plan_hardware_config(vram_mib=vram_mib, ram_mib=ram_mib, **common_kwargs)
        except ValueError as err:
            matrix[preset_name] = {"error": str(err), "vram_mib": vram_mib, "ram_mib": ram_mib}
    output = matrix[args.preset] if args.preset else matrix
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def _add_planning_options(parser: argparse.ArgumentParser, *, prefix: str = "") -> None:
    def dest(key: str) -> str:
        return f"{prefix}{key}" if prefix else key

    parser.add_argument(
        "--context",
        dest=dest("CT_QWEN_CONTEXT") if prefix else "context",
        default=None if prefix else str(DEFAULT_CONTEXT),
        type=_validate_context_arg,
        help=(
            f"Context window tokens: {DEFAULT_CONTEXT} ('1m', default) or integer in "
            f"{MIN_CONTEXT}..{MODEL_MAX_CONTEXT} (accepts k/m suffixes, e.g. 512k, 800k, 1m, auto)"
        ),
    )
    parser.add_argument(
        "--cache-type",
        dest=dest("CT_QWEN_CACHE_TYPE") if prefix else "cache_type",
        default=None if prefix else "q8_0",
        choices=SUPPORTED_CACHE_TYPES,
        help="KV + QSA k-pool indexer cache precision (-ctk/-ctv, default: q8_0 -> 16.05 GiB at 1M)",
    )
    parser.add_argument(
        "--cpu-moe",
        "--cpu-moe-layers",
        dest=dest("CT_QWEN_CPU_MOE") if prefix else "cpu_moe",
        default=None if prefix else "auto",
        type=_validate_cpu_moe_arg,
        help=f"Routed MoE expert layers offloaded to DDR5 RAM (--n-cpu-moe: auto -> 3 on G4 1M, or 0..{MODEL_NUM_LAYERS})",
    )
    parser.add_argument(
        "--parallel",
        dest=dest("CT_QWEN_PARALLEL") if prefix else "parallel",
        default=None if prefix else 1,
        type=_validate_positive_int,
        help="Concurrent server slots (-np, default: 1 to dedicate full 1M context)",
    )
    parser.add_argument(
        "--batch",
        dest=dest("CT_QWEN_BATCH") if prefix else "batch",
        default=None if prefix else 512,
        type=_validate_positive_int,
        help="Logical prompt batch size (-b, default: 512)",
    )
    parser.add_argument(
        "--ubatch",
        dest=dest("CT_QWEN_UBATCH") if prefix else "ubatch",
        default=None if prefix else 128,
        type=_validate_positive_int,
        help="Physical micro-batch size (-ub, default: 128)",
    )


def configure_cli(
    _root: argparse.ArgumentParser,
    sub: argparse._SubParsersAction,
    up: argparse.ArgumentParser,
) -> None:
    up.add_argument(
        "--repo",
        dest="up_CT_QWEN_REPO",
        help=f"Hugging Face GGUF repository (default: {DEFAULT_REPO})",
    )
    up.add_argument(
        "--quant",
        dest="up_CT_QWEN_QUANT",
        help=f"GGUF quantization subfolder (default: {DEFAULT_QUANT})",
    )
    up.add_argument(
        "--alias",
        dest="up_CT_QWEN_ALIAS",
        help=f"OpenAI model alias advertised by llama-server (default: {DEFAULT_ALIAS})",
    )
    _add_planning_options(up, prefix="up_")
    up.add_argument(
        "--reasoning",
        dest="up_CT_QWEN_REASONING",
        choices=REASONING_MODES,
        help="Default thinking mode when chat_template_kwargs.enable_thinking is omitted (default: on)",
    )
    up.add_argument(
        "--llama-commit",
        dest="up_CT_LLAMA_COMMIT",
        help=f"Pinned ggml-org/llama.cpp commit or tag SHA (default: {DEFAULT_LLAMA_COMMIT[:12]} = b11540)",
    )
    up.add_argument(
        "--no-vision",
        dest="up_no_vision",
        action="store_true",
        help="Skip loading mmproj-Qwen3.8-Flash-Next-BF16.gguf (~0.85 GiB)",
    )

    models_p = sub.add_parser("models", help="query served model alias, allocated context window, and vision status")
    models_p.add_argument("--url", help="override service base URL (or set QWEN_FLASH_URL)")
    models_p.add_argument("--token", help="override Bearer API token (or set QWEN_FLASH_TOKEN)")

    plan_p = sub.add_parser("plan", help="preview 1M context window, q8_0 KV budget, and GPU/CPU MoE split offline")
    plan_p.add_argument(
        "--preset",
        choices=tuple(HARDWARE_PRESETS),
        help="inspect a single Colab GPU preset (G4, H100, A100-80G, A100-40G)",
    )
    plan_p.add_argument("--quant", default=DEFAULT_QUANT, help=f"quantization folder (default: {DEFAULT_QUANT})")
    plan_p.add_argument("--vram-gb", type=float, help="custom GPU VRAM in GiB (0 for CPU-only)")
    plan_p.add_argument("--ram-gb", type=float, help="custom system RAM in GiB (default: 176 when --vram-gb is set)")
    _add_planning_options(plan_p)
    plan_p.add_argument("--no-vision", action="store_true", help="plan without the 0.85 GiB BF16 vision projector")
    plan_p.add_argument("--json", action="store_true", help="output machine-readable JSON (default)")


def prepare_up(args: argparse.Namespace) -> None:
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    for _flag, env_key in UP_ENV_FLAGS.items():
        val = getattr(args, "up_" + env_key, None)
        if val is not None:
            SERVICE.env[env_key] = str(val)
    if getattr(args, "up_no_vision", False):
        SERVICE.env["CT_QWEN_VISION"] = "0"


if __name__ == "__main__":
    run_cli(
        lambda: main(
            SERVICE,
            configure=configure_cli,
            prepare_up=prepare_up,
            handlers={
                "models": cmd_models,
                "plan": cmd_plan,
            },
        )
    )
