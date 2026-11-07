#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash-1017 Uncensored GSQ-RCO Coder (GGUF + Engram + Vision) via llama-server.

Deploys `Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-GGUF` (50% expert-pruned 192-expert
`IQ2_XXS` MoE trunk, 72.1 GB + 110.6 GB `IQ4_NL` Engram lookup table + BF16 `deepseek41v` vision
projector) on Google Colab via `llama-server` with native `--api-key-file` Bearer authentication and
hardware-adaptive maximum context window sizing (`CT_DS_CONTEXT=auto`, up to the full 1,048,576 native
1M context on a 96 GB G4 GPU).

Key properties:
  * Hardware default: `G4` (RTX PRO 6000 Blackwell `sm_120`, ~96 GB VRAM) -> 100% GPU layers (`n_cpu_moe=0`)
    + full `1,048,576` (1M) native context window (`f16` DSV4.1 compressed KV + fused lightning indexer).
  * Hardware-adaptive context (`--context auto`, default): inspects actual GPU VRAM & host/cgroup RAM at
    startup, computes exact `deepseek41` KV + graph workspace budget (`~640K` on 80 GB `H100`/`A100`,
    hybrid `--n-cpu-moe` + `~256K` on 40 GB `A100`), and steps down automatically on startup OOM.
  * Engram host-RAM residency & SSD fallback: automatically symlinks `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf`
    as shard 6 (`...-00006-of-00006.gguf`) with `--override-tensor per_layer_token_embd.weight=CPU`
    (`HF_XET_CHUNK_CACHE_SIZE_BYTES=0`). On high-RAM hosts (`G4` 176 GiB / `H100` 167 GiB), loads the
    103 GiB Engram table 100% into DDR5 RAM (`--load-mode none --lazy-mode off`) for zero disk page faults;
    on lower-RAM hosts (`A100-40G`), falls back to `--load-mode mmap --lazy-mode on`.
  * Thinking mode: `--reasoning on` by default; clients/agents decide per API request via
    `chat_template_kwargs: {"enable_thinking": false}` (fast direct answers) or `true`, and tune it with
    plus `reasoning_effort` (`minimal|low|medium|high|xhigh|max` or `1..100`) and `reasoning_budget_tokens`.
  * Auth: native `llama-server --api-key-file` (`auth="builtin"`, OpenAI-compatible `/v1`).

Usage:
  colabllm deepseek-coder up --domain                       # G4 (96 GB) + auto 1M context + fixed domain
  colabllm deepseek-coder up                                # G4 + global transport / quick tunnel
  colabllm deepseek-coder up --gpu H100 --domain            # H100 (80 GB) + auto max context (~640K)
  colabllm deepseek-coder up --gpu H100 --context 1m        # H100 + 1M context (auto offloads ~3 MoE layers)
  colabllm deepseek-coder up --gpu A100 --domain            # A100 (80 GB or 40 GB hybrid MoE offload)
  colabllm deepseek-coder plan                              # Preview auto context window across GPU presets
  colabllm deepseek-coder models                            # Query live server alias, allocated n_ctx & vision
  colabllm deepseek-coder down
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
from cli_support import CliError, UsageError, http_timeout, resolve_endpoint, run_cli, safe_urlopen  # noqa: E402
from colabsvc import Service, State, main  # noqa: E402
from deepseek_coder_runtime import (  # noqa: E402
    CACHE_TYPE_BYTES,
    DEFAULT_ALIAS,
    DEFAULT_LLAMA_COMMIT,
    DEFAULT_LLAMA_REPO,
    DEFAULT_QUANT,
    DEFAULT_REPO,
    MODEL_MAX_CONTEXT,
    MIN_CONTEXT,
    normalize_reasoning_effort,
    parse_context_spec,
    parse_cpu_moe_spec,
    parse_ubatch_spec,
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
    name="deepseek-coder",
    notebook="deepseek-coder.ipynb",
    service_port=8001,
    auth="builtin",
    public_port=8001,
    gpu="G4",
    high_mem=False,
    default_hostname="deepseek",
    ready_timeout=5400,
    env={
        "CT_DS_REPO": DEFAULT_REPO,
        "CT_DS_QUANT": DEFAULT_QUANT,
        "CT_DS_ALIAS": DEFAULT_ALIAS,
        "CT_DS_CONTEXT": "auto",
        "CT_DS_PARALLEL": "1",
        "CT_DS_CACHE_TYPE": "f16",
        "CT_DS_BATCH": "4096",
        "CT_DS_UBATCH": "auto",
        "CT_DS_CPU_MOE": "auto",
        "CT_DS_REASONING": "on",
        "CT_DS_REASONING_EFFORT": "",
        "CT_DS_VISION": "1",
        "CT_LLAMA_REPO": DEFAULT_LLAMA_REPO,
        "CT_LLAMA_COMMIT": DEFAULT_LLAMA_COMMIT,
    },
)

UP_ENV_FLAGS = {
    "--repo": "CT_DS_REPO",
    "--quant": "CT_DS_QUANT",
    "--alias": "CT_DS_ALIAS",
    "--context": "CT_DS_CONTEXT",
    "--parallel": "CT_DS_PARALLEL",
    "--cache-type": "CT_DS_CACHE_TYPE",
    "--batch": "CT_DS_BATCH",
    "--ubatch": "CT_DS_UBATCH",
    "--cpu-moe-layers": "CT_DS_CPU_MOE",
    "--reasoning": "CT_DS_REASONING",
    "--reasoning-effort": "CT_DS_REASONING_EFFORT",
}
_BASE_ENV = dict(SERVICE.env)


def _validate_reasoning_effort(raw: str) -> str:
    try:
        parsed = normalize_reasoning_effort(raw)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err
    return "" if parsed is None else parsed


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


def _validate_ubatch_arg(raw: str) -> str:
    try:
        parsed = parse_ubatch_spec(raw)
    except ValueError as err:
        raise argparse.ArgumentTypeError(str(err)) from err
    return "auto" if parsed is None else str(parsed)


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(
        State(SERVICE).load(),
        url_override,
        token_override,
        url_env="DEEPSEEK_CODER_URL",
        token_env="DEEPSEEK_CODER_TOKEN",
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
        requested_context=args.context,
        parallel=args.parallel,
        cache_type=args.cache_type,
        enable_vision=not args.no_vision,
        batch=args.batch,
        requested_ubatch=args.ubatch,
        requested_cpu_moe=args.cpu_moe_layers,
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
        dest=dest("CT_DS_CONTEXT") if prefix else "context",
        default=None if prefix else "auto",
        type=_validate_context_arg,
        help=(
            f"Context window tokens: 'auto' (default, maximizes for detected GPU/RAM up to {MODEL_MAX_CONTEXT}) "
            f"or integer in {MIN_CONTEXT}..{MODEL_MAX_CONTEXT} (accepts k/m suffixes, e.g. 512k, 1m)"
        ),
    )
    parser.add_argument(
        "--parallel",
        dest=dest("CT_DS_PARALLEL") if prefix else "parallel",
        default=None if prefix else 1,
        type=_validate_positive_int,
        help="Concurrent server slots (-np, default: 1 to maximize per-request context)",
    )
    parser.add_argument(
        "--cache-type",
        dest=dest("CT_DS_CACHE_TYPE") if prefix else "cache_type",
        default=None if prefix else "f16",
        choices=tuple(CACHE_TYPE_BYTES),
        help="KV cache precision (-ctk/-ctv, default: f16 for fused DSV4.1 lightning indexer)",
    )
    parser.add_argument(
        "--batch",
        dest=dest("CT_DS_BATCH") if prefix else "batch",
        default=None if prefix else 4096,
        type=_validate_positive_int,
        help="Logical prompt batch size (-b, default: 4096)",
    )
    parser.add_argument(
        "--ubatch",
        dest=dest("CT_DS_UBATCH") if prefix else "ubatch",
        default=None if prefix else "auto",
        type=_validate_ubatch_arg,
        help="Physical micro-batch size (-ub, default: auto -> 512 on >=90 GB VRAM, 128 otherwise)",
    )
    parser.add_argument(
        "--cpu-moe-layers",
        dest=dest("CT_DS_CPU_MOE") if prefix else "cpu_moe_layers",
        default=None if prefix else "auto",
        type=_validate_cpu_moe_arg,
        help="MoE expert layers offloaded to CPU RAM (--n-cpu-moe: auto, or 0..40)",
    )


def configure_cli(
    _root: argparse.ArgumentParser,
    sub: argparse._SubParsersAction,
    up: argparse.ArgumentParser,
) -> None:
    up.add_argument(
        "--repo",
        dest="up_CT_DS_REPO",
        help=f"Hugging Face GGUF repository (default: {DEFAULT_REPO})",
    )
    up.add_argument(
        "--quant",
        dest="up_CT_DS_QUANT",
        help=f"GGUF quantization identifier (default: {DEFAULT_QUANT})",
    )
    up.add_argument(
        "--alias",
        dest="up_CT_DS_ALIAS",
        help=f"OpenAI model alias advertised by llama-server (default: {DEFAULT_ALIAS})",
    )
    _add_planning_options(up, prefix="up_")
    up.add_argument(
        "--reasoning",
        dest="up_CT_DS_REASONING",
        choices=("on", "off", "auto"),
        help="Default thinking mode when a request does not set chat_template_kwargs.enable_thinking (--reasoning, default: on)",
    )
    up.add_argument(
        "--reasoning-effort",
        dest="up_CT_DS_REASONING_EFFORT",
        type=_validate_reasoning_effort,
        help="Default thinking effort when thinking is active: minimal|low|medium|high|xhigh|max or 1..100 (template default: 50)",
    )
    up.add_argument(
        "--no-vision",
        dest="up_no_vision",
        action="store_true",
        help="Skip loading the BF16 deepseek41v multimodal vision projector (~0.97 GB)",
    )

    models_p = sub.add_parser("models", help="query served model alias, allocated context window, and vision status")
    models_p.add_argument("--url", help="override service base URL (or set DEEPSEEK_CODER_URL)")
    models_p.add_argument("--token", help="override Bearer API token (or set DEEPSEEK_CODER_TOKEN)")

    plan_p = sub.add_parser("plan", help="preview hardware-adaptive context window and GPU/CPU MoE placement offline")
    plan_p.add_argument(
        "--preset",
        choices=tuple(HARDWARE_PRESETS),
        help="inspect a single Colab GPU preset (G4, H100, A100-80G, A100-40G)",
    )
    plan_p.add_argument("--vram-gb", type=float, help="custom GPU VRAM in GiB (0 for CPU-only)")
    plan_p.add_argument("--ram-gb", type=float, help="custom system RAM in GiB (default: 176 when --vram-gb is set)")
    _add_planning_options(plan_p)
    plan_p.add_argument("--no-vision", action="store_true", help="plan without the 0.97 GB vision projector")


def prepare_up(args: argparse.Namespace) -> None:
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    for _flag, env_key in UP_ENV_FLAGS.items():
        val = getattr(args, "up_" + env_key, None)
        if val is not None:
            SERVICE.env[env_key] = str(val)
    if getattr(args, "up_no_vision", False):
        SERVICE.env["CT_DS_VISION"] = "0"


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
