#!/usr/bin/env python3
"""DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic (i1-Q4_K_M + Native MTP Speculative, 256K, F16 Vision) on Colab L4.

Deploys `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` GGUF conversions
(`mradermacher/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic-i1-GGUF` `i1-Q4_K_M` imatrix 4-bit by default,
native Multi-Token Prediction `mtp-Q8_0.gguf` sidecar via `--spec-type draft-mtp`, plus
`DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.mmproj-f16.gguf` multimodal vision projector)
via `llama-server` (`b11540`) with native `--api-key-file` Bearer authentication.

Key properties:
  * Hardware: `L4` (`NVIDIA L4` `sm_89`, `23,034 MiB ≈ 22.5 GiB` VRAM, `~53 GiB` RAM) by default;
    also compatible with `--gpu T4` (`16 GB` VRAM -> `i1-Q4_K_M` + MTP + `208K` context, or `256K` with `--no-mtp`),
    `--gpu A100`, `--gpu H100`, and `--gpu G4`.
  * High-throughput L4 default (`i1-Q4_K_M` + native `MTP` + `256K` context + `F16` vision):
      - `Qwen3.5-9B` hybrid architecture uses 24 Gated DeltaNet linear-attention layers (`O(1)` ~52 MiB state)
        and only 8 GQA full-attention layers (+ 1 `nextn` MTP draft attention layer), so a full `262,144` `q8_0`
        KV cache across both trunk and MTP draft takes just `4.83 GiB` VRAM (`4,948 MiB`).
      - On `L4` (`23,034 MiB` VRAM): `i1-Q4_K_M` (`5.24 GiB`) + `mmproj-f16` (`0.86 GiB`) + `mtp-Q8_0` (`2.27 GiB`)
        + `256K` `q8_0` KV (`4.83 GiB`) = `13.20 GiB` total, leaving `~9.30 GiB` clean VRAM headroom at 100% GPU offload.
      - Users wanting maximum 8-bit static precision can pass `--quant Q8_0` (`Q8_0` + `MTP` + `256K` also fits L4!).
  * Runtime & MTP caches: pinned `llama.cpp` (`b11540`) built for the active GPU (`sm_89` on L4, `sm_75` on T4) and
    cached at `/content/drive/MyDrive/models/.runtimes/llama.cpp/sm<arch>-<fingerprint>/runtime.tar`; standalone
    `mtp-Q8_0.gguf` exported once via HTTP Range and cached on Google Drive for ~3s restore on subsequent boots.
  * Auth: native `llama-server --api-key-file` (`auth="builtin"`).
  * Default domain when `--domain` is passed without a value: `deepseek-9b.<base-domain>`.

Usage:
  colabllm deepseek-qwen-9b up --domain          # L4 + i1-Q4_K_M + native MTP + 256K context + F16 vision
  colabllm deepseek-qwen-9b up                   # L4 + global transport / quick tunnel
  colabllm deepseek-qwen-9b up --quant Q8_0      # L4 + near-lossless Q8_0 + native MTP + 256K context
  colabllm deepseek-qwen-9b up --gpu T4          # T4 + i1-Q4_K_M + native MTP + auto 208K context + F16 vision
  colabllm deepseek-qwen-9b plan                 # Preview VRAM/KV/MTP budget across L4, T4, A100, H100 & G4
  colabllm deepseek-qwen-9b models               # Query live server alias, allocated n_ctx & vision
  colabllm deepseek-qwen-9b ps | info [-i ID] | logs [-i ID] -f | status [-i ID] | down [-i ID | --all]
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
from deepseek_qwen_9b_runtime import (  # noqa: E402
    DEFAULT_ALIAS,
    DEFAULT_CONTEXT,
    DEFAULT_LLAMA_COMMIT,
    DEFAULT_LLAMA_REPO,
    DEFAULT_MTP_DRAFT_MAX,
    DEFAULT_MTP_MODE,
    DEFAULT_QUANT,
    DEFAULT_REPO,
    MIN_CONTEXT,
    MODEL_MAX_CONTEXT,
    MTP_MODES,
    PINNED_QUANT_SPECS,
    REASONING_MODES,
    SUPPORTED_CACHE_TYPES,
    UPSTREAM_HF_REPO,
    parse_context_spec,
    plan_hardware_config,
)

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

HARDWARE_PRESETS: dict[str, tuple[int, int]] = {
    "L4": (23_034, 54_272),
    "T4": (15_360, 12_980),
    "A100-40G": (40_960, 85_504),
    "A100-80G": (81_920, 171_008),
    "H100": (81_559, 171_008),
    "G4": (97_887, 180_224),
}

SERVICE = Service(
    name="deepseek-qwen-9b",
    notebook="deepseek-qwen-9b.ipynb",
    service_port=8000,
    auth="builtin",                   # llama-server --api-key-file handles Bearer auth natively
    public_port=8000,
    gpu="L4",                         # L4 (24 GB) default: i1-Q4_K_M + MTP + 256K q8_0 KV + F16 vision (~9.3 GiB headroom)
    high_mem=False,
    default_hostname="deepseek-9b",
    ready_timeout=2400,
    env={
        "CT_DS9B_REPO": DEFAULT_REPO,
        "CT_DS9B_UPSTREAM_REPO": UPSTREAM_HF_REPO,
        "CT_DS9B_QUANT": DEFAULT_QUANT,
        "CT_DS9B_ALIAS": DEFAULT_ALIAS,
        "CT_DS9B_CONTEXT": "auto",
        "CT_DS9B_CACHE_TYPE": "q8_0",
        "CT_DS9B_PARALLEL": "1",
        "CT_DS9B_BATCH": "2048",
        "CT_DS9B_UBATCH": "512",
        "CT_DS9B_REASONING": "on",
        "CT_DS9B_VISION": "1",
        "CT_DS9B_MTP": DEFAULT_MTP_MODE,
        "CT_DS9B_MTP_DRAFT": str(DEFAULT_MTP_DRAFT_MAX),
        "CT_LLAMA_REPO": DEFAULT_LLAMA_REPO,
        "CT_LLAMA_COMMIT": DEFAULT_LLAMA_COMMIT,
    },
)

UP_ENV_FLAGS = {
    "--repo": "CT_DS9B_REPO",
    "--quant": "CT_DS9B_QUANT",
    "--alias": "CT_DS9B_ALIAS",
    "--context": "CT_DS9B_CONTEXT",
    "--cache-type": "CT_DS9B_CACHE_TYPE",
    "--parallel": "CT_DS9B_PARALLEL",
    "--batch": "CT_DS9B_BATCH",
    "--ubatch": "CT_DS9B_UBATCH",
    "--reasoning": "CT_DS9B_REASONING",
    "--mtp": "CT_DS9B_MTP",
    "--mtp-draft": "CT_DS9B_MTP_DRAFT",
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


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(
        State(SERVICE).load(),
        url_override,
        token_override,
        url_env="DEEPSEEK_QWEN_9B_URL",
        token_env="DEEPSEEK_QWEN_9B_TOKEN",
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
    mtp_arg = "off" if getattr(args, "no_mtp", False) else args.mtp
    common_kwargs = dict(
        quant=args.quant,
        requested_context=args.context,
        cache_type=args.cache_type,
        enable_vision=not args.no_vision,
        mtp=mtp_arg,
        mtp_draft_max=args.mtp_draft,
        parallel=args.parallel,
        batch=args.batch,
        ubatch=args.ubatch,
    )
    if args.vram_gb is not None or args.ram_gb is not None:
        vram_mib = int(round((args.vram_gb or 0.0) * 1024))
        ram_mib = int(round((args.ram_gb if args.ram_gb is not None else 53.0) * 1024))
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
        dest=dest("CT_DS9B_CONTEXT") if prefix else "context",
        default=None if prefix else "auto",
        type=_validate_context_arg,
        help=(
            f"Context window tokens: 'auto' (default -> native {DEFAULT_CONTEXT} = 256K on L4) or integer in "
            f"{MIN_CONTEXT}..{MODEL_MAX_CONTEXT} (accepts k/m suffixes, e.g. 64k, 128k, 256k, 512k, 1m)"
        ),
    )
    parser.add_argument(
        "--cache-type",
        dest=dest("CT_DS9B_CACHE_TYPE") if prefix else "cache_type",
        default=None if prefix else "q8_0",
        choices=SUPPORTED_CACHE_TYPES,
        help="KV cache precision across trunk + MTP full-attention layers (-ctk/-ctv, default: q8_0 -> 4.83 GiB at 256K)",
    )
    parser.add_argument(
        "--mtp",
        dest=dest("CT_DS9B_MTP") if prefix else "mtp",
        default=None if prefix else DEFAULT_MTP_MODE,
        choices=MTP_MODES,
        help="Multi-Token Prediction speculative decoding ('auto' default, 'on', or 'off')",
    )
    parser.add_argument(
        "--mtp-draft",
        dest=dest("CT_DS9B_MTP_DRAFT") if prefix else "mtp_draft",
        default=None if prefix else DEFAULT_MTP_DRAFT_MAX,
        type=_validate_positive_int,
        help=f"Maximum tokens drafted per MTP step (--spec-draft-n-max, default: {DEFAULT_MTP_DRAFT_MAX})",
    )
    parser.add_argument(
        "--parallel",
        dest=dest("CT_DS9B_PARALLEL") if prefix else "parallel",
        default=None if prefix else 1,
        type=_validate_positive_int,
        help="Concurrent server slots (-np, default: 1)",
    )
    parser.add_argument(
        "--batch",
        dest=dest("CT_DS9B_BATCH") if prefix else "batch",
        default=None if prefix else 2048,
        type=_validate_positive_int,
        help="Logical prompt batch size (-b, default: 2048)",
    )
    parser.add_argument(
        "--ubatch",
        dest=dest("CT_DS9B_UBATCH") if prefix else "ubatch",
        default=None if prefix else 512,
        type=_validate_positive_int,
        help="Physical micro-batch size (-ub, default: 512)",
    )


def configure_cli(
    _root: argparse.ArgumentParser,
    sub: argparse._SubParsersAction,
    up: argparse.ArgumentParser,
) -> None:
    quant_list = ", ".join(("auto", "i1-Q4_K_M", "i1-Q6_K", "Q8_0", "Q6_K", "f16"))
    up.add_argument(
        "--repo",
        dest="up_CT_DS9B_REPO",
        help=f"Hugging Face GGUF repository (default: {DEFAULT_REPO})",
    )
    up.add_argument(
        "--quant",
        dest="up_CT_DS9B_QUANT",
        help=f"GGUF quantization ('auto' -> i1-Q4_K_M imatrix default; e.g. {quant_list})",
    )
    up.add_argument(
        "--alias",
        dest="up_CT_DS9B_ALIAS",
        help=f"OpenAI model alias advertised by llama-server (default: {DEFAULT_ALIAS})",
    )
    _add_planning_options(up, prefix="up_")
    up.add_argument(
        "--reasoning",
        dest="up_CT_DS9B_REASONING",
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
        help="Skip loading mmproj-f16.gguf (~0.86 GiB)",
    )
    up.add_argument(
        "--no-mtp",
        dest="up_no_mtp",
        action="store_true",
        help="Disable native Multi-Token Prediction (MTP) speculative draft sidecar (~2.27 GiB)",
    )

    models_p = sub.add_parser("models", help="query served model alias, allocated context window, and vision status")
    models_p.add_argument("--url", help="override service base URL (or set DEEPSEEK_QWEN_9B_URL)")
    models_p.add_argument("--token", help="override Bearer API token (or set DEEPSEEK_QWEN_9B_TOKEN)")

    plan_p = sub.add_parser("plan", help="preview 256K context window, quant selection, MTP draft, and KV VRAM budget offline")
    plan_p.add_argument(
        "--preset",
        choices=tuple(HARDWARE_PRESETS),
        help="inspect a single Colab GPU preset (L4, T4, A100-40G, A100-80G, H100, G4)",
    )
    plan_p.add_argument("--quant", default=DEFAULT_QUANT, help="quantization ('auto' -> i1-Q4_K_M, 'Q8_0', 'i1-Q6_K', 'f16', ...)")
    plan_p.add_argument("--vram-gb", type=float, help="custom GPU VRAM in GiB (0 for CPU-only)")
    plan_p.add_argument("--ram-gb", type=float, help="custom system RAM in GiB (default: 53 when --vram-gb is set)")
    _add_planning_options(plan_p)
    plan_p.add_argument("--no-vision", action="store_true", help="plan without the 0.86 GiB F16 vision projector")
    plan_p.add_argument("--no-mtp", action="store_true", help="plan without the 2.27 GiB Q8_0 MTP draft sidecar")
    plan_p.add_argument("--json", action="store_true", help="output machine-readable JSON (default)")


def prepare_up(args: argparse.Namespace) -> None:
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    for _flag, env_key in UP_ENV_FLAGS.items():
        val = getattr(args, "up_" + env_key, None)
        if val is not None:
            SERVICE.env[env_key] = str(val)
    if getattr(args, "up_no_vision", False):
        SERVICE.env["CT_DS9B_VISION"] = "0"
    if getattr(args, "up_no_mtp", False):
        SERVICE.env["CT_DS9B_MTP"] = "off"


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
