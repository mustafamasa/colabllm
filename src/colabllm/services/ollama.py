#!/usr/bin/env python3
"""Ollama LLM server on Google Colab behind Caddy auth (notebook: ollama.ipynb).

Grounded in https://docs.ollama.com/:
  * Automatic model download & GPU VRAM warmup (`--models`, `--alias`, `--preload` / `--preload-all`).
    Supports Ollama library tags (`qwen3:8b`, `gemma4:e4b`, `nomic-embed-text`) and Hugging Face GGUF
    models (`hf.co/<org>/<repo>:<quant>`).
  * Selectable or full-open compatible APIs (`--compat all` by default, or any combination of
    `ollama`, `openai`, `anthropic`):
      - `ollama`: Native REST API (`/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/ps`, ...)
      - `openai`: OpenAI Chat Completions (`/v1/chat/completions`), Responses (`/v1/responses`),
        Completions (`/v1/completions`), Embeddings (`/v1/embeddings`), Models (`/v1/models`),
        and Audio (`/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations`)
      - `anthropic`: Anthropic Messages API (`/v1/messages`, `/v1/models`) for Claude Code & Anthropic SDKs,
        supporting both `Authorization: Bearer <token>` and `x-api-key: <token>` via Caddy!
  * Startup server tuning (`OLLAMA_*`): `--profile`, `--context-length`, `--flash-attention`,
    `--kv-cache-type` (`f16`/`q8_0`/`q4_0`), `--num-parallel`, `--max-loaded-models`, `--max-queue`,
    `--keep-alive`, `--load-timeout`, `--gpu-overhead`, `--sched-spread`, `--max-transfer-streams`,
    `--origins`, `--no-cloud`, `--models-storage` (`ssd`/`drive`), and `--ollama-env KEY=VALUE`.

Usage:
  colabllm ollama up --models qwen3:8b
  colabllm ollama up --models qwen2.5-coder:7b --alias claude-3-5-sonnet=qwen2.5-coder:7b --compat openai,anthropic --profile coding
  colabllm ollama up --gpu L4 --models gemma4:e4b,nomic-embed-text --context-length 32768 --kv-cache-type q8_0 --keep-alive -1
  colabllm ollama endpoints [--json] [--show-secrets]
  colabllm ollama models [--json]
  colabllm ollama pull [-i ID | -A] qwen2.5:7b [--alias gpt-4o] [--preload]
  colabllm ollama run qwen2.5:0.5b "Introduce yourself in one sentence" [--api ollama|openai|anthropic]
  colabllm ollama preload qwen2.5:7b --keep-alive -1
  colabllm ollama unload qwen2.5:7b
  colabllm ollama cp qwen2.5:7b gpt-4o-mini
  colabllm ollama show qwen2.5:7b
  colabllm ollama rm qwen2.5:0.5b
  colabllm ollama ps | info [-i ID] | logs [-i ID] -f | status [-i ID] | down [-i ID | --all] [--delete-tunnel]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cli_support import (  # noqa: E402
    CliError,
    Deadline,
    UsageError,
    diagnostic,
    http_timeout,
    nonnegative_seconds,
    resolve_endpoint,
    safe_urlopen,
)
from colabsvc import (  # noqa: E402
    Service,
    State,
    apply_instance,
    die,
    list_service_instances,
    log,
    main,
)
from http_client import BROWSER_UA, http_json, http_request  # noqa: E402
from ollama_runtime import (  # noqa: E402
    COMPAT_APIS,
    DEFAULT_MODEL,
    KV_CACHE_TYPES,
    MODELS_STORAGE_MODES,
    OPTIMIZATION_PROFILES,
    build_ollama_server_env,
    compat_endpoints_info,
    copy_model_alias,
    normalize_api_keep_alive,
    parse_compat_apis,
    parse_model_aliases,
    parse_model_specs,
    preload_or_unload_model,
    pull_model_stream,
    validate_duration,
    validate_model_spec,
    validate_ollama_version,
)

SERVICE = Service(
    name="ollama",
    notebook="ollama.ipynb",
    service_port=11434,
    auth="caddy",            # no built-in auth -> Caddy on public_port (Basic Auth + Bearer + X-Api-Key)
    public_port=18080,
    gpu="T4",
    ready_timeout=3600,
    env={"CT_MODELS": DEFAULT_MODEL},
)

_BASE_ENV = dict(SERVICE.env)


def _add_endpoint_args(parser: argparse.ArgumentParser, *, include_timeout: bool = True) -> None:
    parser.add_argument("--url", help="service URL override (default: saved instance URL or $OLLAMA_URL)")
    parser.add_argument("--api-token", help="API token override (default: saved token or $OLLAMA_API_TOKEN)")
    if include_timeout:
        parser.add_argument("--timeout", type=nonnegative_seconds, default=120,
                            help="request timeout in seconds; 0 means unlimited (default: 120)")


def _resolve_service_endpoint(
    svc: Service | None = None,
    url_override: str | None = None,
    token_override: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    target = svc or SERVICE
    st = State(target).load()
    url, token = resolve_endpoint(st, url_override, token_override,
                                  url_env="OLLAMA_URL", token_env="OLLAMA_API_TOKEN")
    return url, token, st


def _resolve_target_services(all_instances: bool = False, instances_csv: str = "") -> list[Service]:
    """Return target Service instance(s) for multi-instance operations like `pull`."""
    base_name = getattr(SERVICE, "base_name", SERVICE.name)
    if instances_csv:
        targets: list[Service] = []
        for raw_inst in instances_csv.split(","):
            inst = raw_inst.strip()
            if not inst:
                continue
            s = copy.deepcopy(SERVICE)
            s.name = base_name
            s.base_name = base_name
            if inst not in ("default", "0", base_name):
                apply_instance(s, inst)
            else:
                s.default_hostname = getattr(SERVICE, "base_default_hostname", None) or SERVICE.default_hostname
            targets.append(s)
        if targets:
            return targets
    if all_instances:
        targets = []
        for rec in list_service_instances(base_name):
            if rec["running"] and rec["url"] and rec["api_token"]:
                s = copy.deepcopy(SERVICE)
                s.name = rec["name"]
                s.base_name = base_name
                targets.append(s)
        if not targets:
            die(f"No running {base_name} instance; run `colabllm ollama ps` or `colabllm ollama up` first")
        return targets
    return [SERVICE]


def _auth_headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "User-Agent": BROWSER_UA,
    }


def _fetch_installed_model_names(url: str, token: str, timeout: float | None = 30) -> set[str] | None:
    """Return installed model tags (both exact and `:latest`-stripped), or None if model listing is unavailable."""
    try:
        tags = http_json(f"{url}/api/tags", token=token, timeout=timeout)
        names: set[str] = set()
        for item in tags.get("models") or ():
            name = str(item.get("name") or item.get("model") or "").strip()
            if name:
                names.add(name)
                if name.endswith(":latest"):
                    names.add(name[:-7])
        return names
    except CliError:
        pass
    try:
        v1_models = http_json(f"{url}/v1/models", token=token, timeout=timeout)
        names = set()
        for item in v1_models.get("data") or ():
            mid = str(item.get("id") or "").strip()
            if mid:
                names.add(mid)
                if mid.endswith(":latest"):
                    names.add(mid[:-7])
        return names
    except CliError:
        return None


def _model_is_installed(model: str, installed: set[str]) -> bool:
    if model in installed:
        return True
    if ":" not in model and f"{model}:latest" in installed:
        return True
    return False


def _cmd_endpoints(args: argparse.Namespace) -> None:
    st = State(SERVICE).load()
    saved_env = st.get("env") if isinstance(st.get("env"), dict) else {}
    compat = parse_compat_apis(saved_env.get("CT_OLLAMA_COMPAT"))
    read_only = saved_env.get("CT_OLLAMA_READ_ONLY_MODELS") == "1"

    url = (args.url or st.get("url") or "").rstrip("/")
    token: str | None = None
    if args.show_secrets:
        if args.url or args.api_token:
            url, token = resolve_endpoint(st, args.url, args.api_token,
                                          url_env="OLLAMA_URL", token_env="OLLAMA_API_TOKEN")
        else:
            token = st.get("api_token")

    info = compat_endpoints_info(url, compat, api_token=token, read_only_models=read_only)
    info["service"] = SERVICE.name
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    lines = [
        f"service          : {SERVICE.name}",
        f"URL              : {url or '(not started)'}",
        f"compat APIs      : {', '.join(compat)}" + (" (read-only models)" if read_only else ""),
    ]
    for api_name, detail in info["interfaces"].items():
        lines.append(f"[{api_name}] base_url : {detail['base_url']}")
        lines.append(f"[{api_name}] routes   : {', '.join(detail['endpoints'])}")
    if args.show_secrets and token:
        lines.append(f"api_token        : {token}")
    print("\n".join(lines))


def _cmd_models(args: argparse.Namespace) -> None:
    url, token, _ = _resolve_service_endpoint(url_override=args.url, token_override=args.api_token)
    source_api = "ollama"
    downloaded: list[dict[str, Any]] = []
    running: list[dict[str, Any]] = []

    try:
        tags_resp = http_json(f"{url}/api/tags", token=token, timeout=args.timeout)
        downloaded = list(tags_resp.get("models") or [])
        try:
            ps_resp = http_json(f"{url}/api/ps", token=token, timeout=args.timeout)
            running = list(ps_resp.get("models") or [])
        except CliError:
            running = []
    except CliError as err:
        # Fallback to /v1/models when `--compat openai` or `--compat anthropic` disabled `/api/*`
        if "HTTP 403" not in str(err) and "HTTP 404" not in str(err):
            raise
        source_api = "v1"
        v1_resp = http_json(f"{url}/v1/models", token=token, timeout=args.timeout)
        downloaded = [{"name": m.get("id"), "modified_at": m.get("created"), "owned_by": m.get("owned_by")}
                      for m in (v1_resp.get("data") or [])]

    result = {
        "service": SERVICE.name,
        "source": source_api,
        "models": downloaded,
        "running": running,
    }
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not downloaded:
        print("No models downloaded yet. Use `colabllm ollama pull <model>` to download one.")
    else:
        print(f"{'MODEL':<36} {'SIZE (GB)':<12} {'QUANT / FAMILY'}")
        for m in downloaded:
            name = str(m.get("name") or m.get("model") or "-")
            size_b = m.get("size")
            size_str = f"{size_b / 1e9:.2f}" if isinstance(size_b, (int, float)) else "-"
            details = m.get("details") if isinstance(m.get("details"), dict) else {}
            quant_family = " / ".join(
                str(x) for x in (details.get("quantization_level"), details.get("family")) if x
            ) or str(m.get("owned_by") or "-")
            print(f"{name:<36} {size_str:<12} {quant_family}")

    if running:
        print("\nLoaded in VRAM / Memory (/api/ps):")
        print(f"{'MODEL':<36} {'VRAM (GB)':<12} {'CONTEXT':<10} {'EXPIRES'}")
        for r in running:
            rname = str(r.get("name") or r.get("model") or "-")
            vram_b = r.get("size_vram")
            total_b = r.get("size")
            if isinstance(vram_b, (int, float)) and isinstance(total_b, (int, float)) and total_b > 0:
                pct = int(round(vram_b * 100.0 / total_b))
                vram_str = f"{vram_b / 1e9:.2f} ({pct}%)"
            else:
                vram_str = "-"
            ctx_str = str(r.get("context_length") or "-")
            exp_str = str(r.get("expires_at") or "-")
            print(f"{rname:<36} {vram_str:<12} {ctx_str:<10} {exp_str}")


def _cmd_pull(args: argparse.Namespace) -> None:
    models = [validate_model_spec(m) for raw in args.models for m in raw.split(",") if m.strip()]
    if not models:
        raise UsageError("Specify at least one model name to pull")
    if args.alias and len(models) != 1:
        raise UsageError("--alias only works when pulling a single model; for several models use `colabllm ollama cp <src> <dst>`")
    alias_name = validate_model_spec(args.alias) if args.alias else None
    if args.keep_alive is not None:
        validate_duration(args.keep_alive, "--keep-alive")

    targets = _resolve_target_services(all_instances=args.all_instances, instances_csv=args.instances)
    if args.url and len(targets) > 1:
        raise UsageError("--url targets a single endpoint and cannot be combined with -A / --instances")

    deadline = Deadline(args.timeout, "Timed out pulling the model; the remote download may still be running. Check logs or retry later")
    results: list[dict[str, Any]] = []
    for svc in targets:
        url, token, _ = _resolve_service_endpoint(svc, args.url, args.api_token)
        headers = _auth_headers(token)
        for spec in models:
            log(f"[{svc.name}] pulling model {spec} ...")
            pull_res = pull_model_stream(
                url,
                spec,
                headers=headers,
                opener=safe_urlopen,
                log_fn=lambda msg, sname=svc.name: diagnostic(f"[{sname}] {msg}"),
                timeout=http_timeout(deadline.remaining()),
                retries=2,
            )
            entry: dict[str, Any] = {"instance": svc.name, **pull_res}
            if alias_name and alias_name != spec:
                copy_model_alias(
                    url, spec, alias_name,
                    headers=headers, opener=safe_urlopen, timeout=http_timeout(deadline.remaining(60)),
                )
                entry["alias"] = alias_name
            if args.preload:
                diagnostic(f"[{svc.name}] preloading {spec} into GPU memory ...")
                preload_res = preload_or_unload_model(
                    url, spec, keep_alive=args.keep_alive,
                    headers=headers, opener=safe_urlopen, timeout=http_timeout(deadline.remaining(600)),
                )
                entry["preloaded"] = preload_res["endpoint"]
            results.append(entry)
    print(json.dumps(results[0] if len(results) == 1 else {"results": results}, ensure_ascii=False, indent=2))


def _cmd_preload(args: argparse.Namespace) -> None:
    spec = validate_model_spec(args.model)
    if args.keep_alive is not None:
        validate_duration(args.keep_alive, "--keep-alive")
    url, token, _ = _resolve_service_endpoint(url_override=args.url, token_override=args.api_token)
    headers = _auth_headers(token)
    deadline = Deadline(args.timeout, "Timed out preloading the model")

    if args.auto_pull:
        installed = _fetch_installed_model_names(url, token, timeout=deadline.remaining(30))
        if installed is not None and not _model_is_installed(spec, installed):
            diagnostic(f"[{SERVICE.name}] Model {spec} is not downloaded yet; pulling it automatically ...")
            pull_model_stream(
                url, spec, headers=headers, opener=safe_urlopen,
                log_fn=lambda msg: diagnostic(f"[{SERVICE.name}] {msg}"),
                timeout=http_timeout(deadline.remaining()),
            )

    res = preload_or_unload_model(
        url, spec, keep_alive=args.keep_alive,
        headers=headers, opener=safe_urlopen, timeout=http_timeout(deadline.remaining()),
    )
    try:
        ps_resp = http_json(f"{url}/api/ps", token=token, timeout=deadline.remaining(30))
        res["running"] = ps_resp.get("models") or []
    except CliError:
        pass
    print(json.dumps(res, ensure_ascii=False, indent=2))


def _cmd_unload(args: argparse.Namespace) -> None:
    spec = validate_model_spec(args.model)
    url, token, _ = _resolve_service_endpoint(url_override=args.url, token_override=args.api_token)
    res = preload_or_unload_model(
        url, spec, keep_alive=0,
        headers=_auth_headers(token), opener=safe_urlopen, timeout=http_timeout(args.timeout),
    )
    res["status"] = "unloaded"
    print(json.dumps(res, ensure_ascii=False, indent=2))


def _cmd_cp(args: argparse.Namespace) -> None:
    url, token, _ = _resolve_service_endpoint(url_override=args.url, token_override=args.api_token)
    res = copy_model_alias(
        url, args.source, args.destination,
        headers=_auth_headers(token), opener=safe_urlopen, timeout=http_timeout(args.timeout),
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


def _cmd_rm(args: argparse.Namespace) -> None:
    models = [validate_model_spec(m) for raw in args.models for m in raw.split(",") if m.strip()]
    if not models:
        raise UsageError("Specify at least one model name to delete")
    url, token, _ = _resolve_service_endpoint(url_override=args.url, token_override=args.api_token)
    deleted: list[str] = []
    for spec in models:
        http_request(
            f"{url}/api/delete",
            token=token,
            method="DELETE",
            payload={"model": spec},
            timeout=args.timeout,
        )
        deleted.append(spec)
    print(json.dumps({"deleted": deleted}, ensure_ascii=False, indent=2))


def _cmd_show(args: argparse.Namespace) -> None:
    spec = validate_model_spec(args.model)
    url, token, _ = _resolve_service_endpoint(url_override=args.url, token_override=args.api_token)
    payload: dict[str, Any] = {"model": spec}
    if args.verbose:
        payload["verbose"] = True
    res = http_json(f"{url}/api/show", token=token, method="POST", payload=payload, timeout=args.timeout)
    if args.json:
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    details = res.get("details") if isinstance(res.get("details"), dict) else {}
    summary = {
        "model": spec,
        "family": details.get("family"),
        "parameter_size": details.get("parameter_size"),
        "quantization_level": details.get("quantization_level"),
        "capabilities": res.get("capabilities"),
        "parameters": res.get("parameters"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _cmd_run(args: argparse.Namespace) -> None:
    spec = validate_model_spec(args.model)
    prompt = args.prompt
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        if not prompt:
            raise UsageError("Provide a prompt argument or pipe text through standard input")

    if args.keep_alive is not None:
        validate_duration(args.keep_alive, "--keep-alive")

    url, token, _ = _resolve_service_endpoint(url_override=args.url, token_override=args.api_token)
    deadline = Deadline(args.timeout, "Inference request timed out")

    if args.auto_pull:
        installed = _fetch_installed_model_names(url, token, timeout=deadline.remaining(30))
        if installed is not None and not _model_is_installed(spec, installed):
            diagnostic(f"[{SERVICE.name}] Model {spec} is not downloaded yet; pulling it automatically ...")
            pull_model_stream(
                url, spec,
                headers=_auth_headers(token),
                opener=safe_urlopen,
                log_fn=lambda msg: diagnostic(f"[{SERVICE.name}] {msg}"),
                timeout=http_timeout(deadline.remaining()),
            )

    api_mode = args.api
    req_timeout = deadline.remaining()
    text_out = ""
    raw_resp: dict[str, Any] = {}

    if api_mode == "ollama":
        messages: list[dict[str, str]] = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        body: dict[str, Any] = {"model": spec, "messages": messages, "stream": False}
        options: dict[str, Any] = {}
        if args.num_ctx is not None:
            options["num_ctx"] = args.num_ctx
        if args.temperature is not None:
            options["temperature"] = args.temperature
        if args.max_tokens is not None:
            options["num_predict"] = args.max_tokens
        if options:
            body["options"] = options
        if args.think is not None:
            body["think"] = args.think
        if args.keep_alive is not None:
            body["keep_alive"] = normalize_api_keep_alive(args.keep_alive)
        raw_resp = http_json(f"{url}/api/chat", token=token, method="POST", payload=body, timeout=req_timeout)
        msg = raw_resp.get("message") if isinstance(raw_resp.get("message"), dict) else {}
        text_out = str(msg.get("content") or "")

    elif api_mode == "openai":
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt})
        body = {"model": spec, "messages": messages, "stream": False}
        if args.temperature is not None:
            body["temperature"] = args.temperature
        if args.max_tokens is not None:
            body["max_tokens"] = args.max_tokens
        raw_resp = http_json(f"{url}/v1/chat/completions", token=token, method="POST", payload=body, timeout=req_timeout)
        choices = raw_resp.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            text_out = str(msg.get("content") or "")

    elif api_mode == "anthropic":
        body = {
            "model": spec,
            "max_tokens": args.max_tokens or 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        if args.system:
            body["system"] = args.system
        if args.temperature is not None:
            body["temperature"] = args.temperature
        if args.think is True:
            body["thinking"] = {"type": "enabled", "budget_tokens": max(256, (args.max_tokens or 1024) // 2)}
        elif args.think is False:
            body["thinking"] = {"type": "disabled"}
        # Exercise native Anthropic `x-api-key` + `anthropic-version` header authentication through Caddy
        _, _, raw_bytes = http_request(
            f"{url}/v1/messages",
            method="POST",
            payload=body,
            headers={"x-api-key": token, "anthropic-version": "2023-06-01"},
            timeout=req_timeout,
        )
        raw_resp = json.loads(raw_bytes.decode("utf-8", errors="replace") or "{}")
        blocks = [
            str(b.get("text") or "")
            for b in (raw_resp.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text_out = "".join(blocks)

    if args.json:
        print(json.dumps(raw_resp, ensure_ascii=False, indent=2))
    else:
        print(text_out)


def configure_cli(parser: argparse.ArgumentParser, commands: Any, up: argparse.ArgumentParser) -> None:
    # 1. Model download, aliases, storage & warmup options on `up`
    model_grp = up.add_argument_group("Ollama model & storage options")
    model_grp.add_argument(
        "--models", "--model", "-m", action="append",
        help=f"models to auto-download at startup (default: {DEFAULT_MODEL}); repeat or comma-separate, supports hf.co/<org>/<repo>:<quant>",
    )
    model_grp.add_argument(
        "--no-models", action="store_true",
        help="start Ollama server without pre-downloading any model at startup",
    )
    model_grp.add_argument(
        "--alias", action="append", metavar="ALIAS=MODEL",
        help="create model alias via `ollama cp` (auto-downloads MODEL if needed), e.g. claude-3-5-sonnet=qwen2.5-coder:7b",
    )
    preload_grp = model_grp.add_mutually_exclusive_group()
    preload_grp.add_argument(
        "--preload", action=argparse.BooleanOptionalAction, default=None,
        help="preload primary model into GPU VRAM before CT_READY (default: enabled)",
    )
    preload_grp.add_argument(
        "--preload-all", action="store_true",
        help="preload all startup models into GPU VRAM before CT_READY",
    )
    model_grp.add_argument(
        "--models-storage", choices=MODELS_STORAGE_MODES, default=None,
        help="where to store Ollama model blobs: 'ssd' (/content/ollama/models, fast default) or 'drive' (Google Drive persistent)",
    )
    model_grp.add_argument(
        "--ollama-version", metavar="VER", default=None,
        help="install a specific Ollama release version (e.g. 0.17.7)",
    )

    # 2. Compatible API exposure options on `up`
    compat_grp = up.add_argument_group("Compatible API interface options")
    compat_grp.add_argument(
        "--compat", "--apis", action="append", metavar="APIS",
        help="compatible APIs to expose: 'all' (default) or comma-separated subset of ollama,openai,anthropic",
    )
    compat_grp.add_argument(
        "--read-only-models", action="store_true",
        help="block public model-mutating endpoints (/api/pull, /api/delete, /api/create, /api/copy, /api/push)",
    )

    # 3. Ollama server optimization & tuning options on `up`
    tune_grp = up.add_argument_group("Ollama server optimization parameters")
    tune_grp.add_argument(
        "--profile", choices=tuple(OPTIMIZATION_PROFILES), default=None,
        help="server optimization preset (explicit flags below override preset values)",
    )
    tune_grp.add_argument(
        "--context-length", "--ctx-size", dest="context_length", type=int, default=None, metavar="TOKENS",
        help="default context length in tokens (OLLAMA_CONTEXT_LENGTH, e.g. 8192, 32768, 65536)",
    )
    tune_grp.add_argument(
        "--flash-attention", action=argparse.BooleanOptionalAction, default=None,
        help="force-enable or disable Flash Attention (OLLAMA_FLASH_ATTENTION)",
    )
    tune_grp.add_argument(
        "--kv-cache-type", choices=KV_CACHE_TYPES, default=None,
        help="K/V cache quantization precision (OLLAMA_KV_CACHE_TYPE: f16, q8_0, q4_0; q8_0/q4_0 auto-enables Flash Attention)",
    )
    tune_grp.add_argument(
        "--num-parallel", type=int, default=None, metavar="N",
        help="max parallel requests per loaded model (OLLAMA_NUM_PARALLEL)",
    )
    tune_grp.add_argument(
        "--max-loaded-models", type=int, default=None, metavar="N",
        help="max concurrently loaded models in VRAM (OLLAMA_MAX_LOADED_MODELS; 0 = auto)",
    )
    tune_grp.add_argument(
        "--max-queue", type=int, default=None, metavar="N",
        help="max queued requests when busy before HTTP 503 (OLLAMA_MAX_QUEUE, default 512)",
    )
    tune_grp.add_argument(
        "--keep-alive", default=None, metavar="DURATION",
        help="how long models remain loaded in VRAM (OLLAMA_KEEP_ALIVE: e.g. 30m, 24h, -1 permanent, 0 unload immediately)",
    )
    tune_grp.add_argument(
        "--load-timeout", default=None, metavar="DURATION",
        help="max wait time for model load stall detection (OLLAMA_LOAD_TIMEOUT: e.g. 5m, 15m, 0 unlimited)",
    )
    tune_grp.add_argument(
        "--gpu-overhead", default=None, metavar="SIZE",
        help="VRAM bytes reserved per GPU (OLLAMA_GPU_OVERHEAD: integer bytes or 512M, 1G)",
    )
    tune_grp.add_argument(
        "--sched-spread", action=argparse.BooleanOptionalAction, default=None,
        help="spread model layers across multiple GPUs (OLLAMA_SCHED_SPREAD)",
    )
    tune_grp.add_argument(
        "--max-transfer-streams", type=int, default=None, metavar="N",
        help="concurrent transfer streams for Safetensors pulls (OLLAMA_MAX_TRANSFER_STREAMS)",
    )
    tune_grp.add_argument(
        "--origins", default=None, metavar="ORIGINS",
        help="comma-separated allowed CORS origins (OLLAMA_ORIGINS)",
    )
    cloud_grp = tune_grp.add_mutually_exclusive_group()
    cloud_grp.add_argument(
        "--no-cloud", dest="no_cloud", action="store_true", default=None,
        help="disable Ollama cloud features (OLLAMA_NO_CLOUD=1)",
    )
    cloud_grp.add_argument(
        "--cloud", dest="no_cloud", action="store_false",
        help="explicitly allow Ollama cloud features (OLLAMA_NO_CLOUD=0)",
    )
    tune_grp.add_argument(
        "--ollama-debug", choices=("0", "1", "2"), default=None,
        help="Ollama server log verbosity: 0=info, 1=debug, 2=trace (OLLAMA_DEBUG)",
    )
    tune_grp.add_argument(
        "--ollama-env", action="append", metavar="KEY=VALUE",
        help="additional OLLAMA_* / GPU / HTTPS_PROXY environment variable (repeatable)",
    )

    # 4. Runtime management & inference subcommands
    ep = commands.add_parser("endpoints", help="show active compatible APIs (ollama/openai/anthropic) and Base URLs")
    _add_endpoint_args(ep, include_timeout=False)
    ep.add_argument("--json", action="store_true", help="output machine-readable JSON")
    ep.add_argument("--show-secrets", action="store_true", help="include API token in output")

    md = commands.add_parser("models", help="list downloaded models and VRAM-resident models")
    _add_endpoint_args(md)
    md.add_argument("--json", action="store_true", help="output machine-readable JSON")

    pl = commands.add_parser("pull", help="download model(s) into the running Colab Ollama instance")
    _add_endpoint_args(pl, include_timeout=False)
    pl.add_argument("--timeout", type=nonnegative_seconds, default=0,
                    help="total wait seconds; 0 means unlimited (default: 0)")
    pl.add_argument("--alias", help="optional alias name (`ollama cp <model> <alias>`) when pulling a single model")
    pl.add_argument("--preload", action=argparse.BooleanOptionalAction, default=False,
                    help="also preload downloaded model(s) into GPU VRAM")
    pl.add_argument("--keep-alive", default=None, metavar="DURATION",
                    help="keep-alive duration when --preload is enabled (e.g. 30m, -1)")
    targets = pl.add_mutually_exclusive_group()
    targets.add_argument("-A", "--all-instances", action="store_true", help="pull on all running instances")
    targets.add_argument("--instances", default="", help="comma-separated instance IDs")
    pl.add_argument("models", nargs="+", help="Ollama or hf.co model tag(s) to pull")

    rn = commands.add_parser("run", help="run a prompt on a model (auto-downloads model if missing)")
    _add_endpoint_args(rn, include_timeout=False)
    rn.add_argument("--timeout", type=nonnegative_seconds, default=600,
                    help="total wait seconds including auto-pull; 0 means unlimited (default: 600)")
    rn.add_argument("--api", choices=COMPAT_APIS, default="ollama",
                    help="which compatible API endpoint to call: ollama (/api/chat), openai (/v1/chat/completions), anthropic (/v1/messages)")
    rn.add_argument("--system", default=None, help="optional system prompt")
    rn.add_argument("--num-ctx", type=int, default=None, help="per-request context size (native ollama API)")
    rn.add_argument("--temperature", type=float, default=None, help="sampling temperature")
    rn.add_argument("--max-tokens", type=int, default=None, help="maximum generated tokens")
    rn.add_argument("--think", action=argparse.BooleanOptionalAction, default=None,
                    help="enable or disable model thinking/reasoning mode")
    rn.add_argument("--keep-alive", default=None, metavar="DURATION", help="keep-alive duration (native ollama API)")
    rn.add_argument("--auto-pull", action=argparse.BooleanOptionalAction, default=True,
                    help="automatically pull the model first if not yet downloaded (default: enabled)")
    rn.add_argument("--json", action="store_true", help="output raw API response JSON")
    rn.add_argument("model", help="model name (e.g. qwen3:8b)")
    rn.add_argument("prompt", nargs="?", default=None, help="user prompt (or pipe via stdin)")

    pr = commands.add_parser("preload", help="preload a model into GPU VRAM (auto-downloads if missing)")
    _add_endpoint_args(pr, include_timeout=False)
    pr.add_argument("--timeout", type=nonnegative_seconds, default=900,
                    help="total wait seconds; 0 means unlimited (default: 900)")
    pr.add_argument("--keep-alive", default=None, metavar="DURATION",
                    help="keep-alive duration in VRAM (e.g. -1 for permanent, 30m)")
    pr.add_argument("--auto-pull", action=argparse.BooleanOptionalAction, default=True,
                    help="automatically pull the model first if not yet downloaded (default: enabled)")
    pr.add_argument("model", help="model name to preload into VRAM")

    ul = commands.add_parser("unload", help="immediately unload a model from GPU VRAM (keep_alive=0)")
    _add_endpoint_args(ul)
    ul.add_argument("model", help="model name to unload")

    cp_p = commands.add_parser("cp", help="copy/alias an existing model (`ollama cp <source> <destination>`)")
    _add_endpoint_args(cp_p)
    cp_p.add_argument("source", help="existing source model name")
    cp_p.add_argument("destination", help="new alias model name (e.g. gpt-4o or claude-3-5-sonnet)")

    rm_p = commands.add_parser("rm", help="remove model(s) from the remote Ollama instance")
    _add_endpoint_args(rm_p)
    rm_p.add_argument("models", nargs="+", help="model name(s) to delete")

    sh_p = commands.add_parser("show", help="show remote model metadata, capabilities, and parameters")
    _add_endpoint_args(sh_p)
    sh_p.add_argument("--verbose", action="store_true", help="include full verbose tensor/tokenizer info")
    sh_p.add_argument("--json", action="store_true", help="output full raw /api/show JSON")
    sh_p.add_argument("model", help="model name")


def prepare_up(args: Any) -> None:
    # Reset to baseline defaults on every call so overrides never leak across invocations.
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)

    raw_models = getattr(args, "models", None)
    no_models = bool(getattr(args, "no_models", False))
    if no_models or raw_models is not None:
        parsed_models = parse_model_specs(raw_models, no_models=no_models)
        SERVICE.env["CT_MODELS"] = ",".join(parsed_models)
        if not parsed_models:
            SERVICE.env["CT_OLLAMA_NO_MODELS"] = "1"

    raw_aliases = getattr(args, "alias", None)
    if raw_aliases:
        aliases = parse_model_aliases(raw_aliases)
        SERVICE.env["CT_OLLAMA_ALIASES"] = ",".join(f"{alias}={src}" for alias, src in aliases)

    preload_all = bool(getattr(args, "preload_all", False))
    preload = getattr(args, "preload", None)
    if preload_all:
        SERVICE.env["CT_OLLAMA_PRELOAD"] = "all"
    elif preload is False:
        SERVICE.env["CT_OLLAMA_PRELOAD"] = "none"
    elif preload is True:
        SERVICE.env["CT_OLLAMA_PRELOAD"] = "primary"

    storage = getattr(args, "models_storage", None)
    if storage:
        if storage not in MODELS_STORAGE_MODES:
            raise ValueError(f"--models-storage must be one of {', '.join(MODELS_STORAGE_MODES)}")
        SERVICE.env["CT_OLLAMA_MODELS_STORAGE"] = storage

    ollama_ver = getattr(args, "ollama_version", None)
    if ollama_ver:
        SERVICE.env["CT_OLLAMA_VERSION"] = validate_ollama_version(ollama_ver)

    raw_compat = getattr(args, "compat", None)
    if raw_compat:
        compat = parse_compat_apis(raw_compat)
        SERVICE.env["CT_OLLAMA_COMPAT"] = ",".join(compat)

    if getattr(args, "read_only_models", False):
        SERVICE.env["CT_OLLAMA_READ_ONLY_MODELS"] = "1"

    tuning_kwargs = {
        "profile": getattr(args, "profile", None),
        "context_length": getattr(args, "context_length", None),
        "flash_attention": getattr(args, "flash_attention", None),
        "kv_cache_type": getattr(args, "kv_cache_type", None),
        "num_parallel": getattr(args, "num_parallel", None),
        "max_loaded_models": getattr(args, "max_loaded_models", None),
        "max_queue": getattr(args, "max_queue", None),
        "keep_alive": getattr(args, "keep_alive", None),
        "load_timeout": getattr(args, "load_timeout", None),
        "gpu_overhead": getattr(args, "gpu_overhead", None),
        "sched_spread": getattr(args, "sched_spread", None),
        "max_transfer_streams": getattr(args, "max_transfer_streams", None),
        "origins": getattr(args, "origins", None),
        "no_cloud": getattr(args, "no_cloud", None),
        "ollama_debug": getattr(args, "ollama_debug", None),
        "extra_env": getattr(args, "ollama_env", None),
    }
    if any(val is not None and val != [] for val in tuning_kwargs.values()):
        server_env = build_ollama_server_env(**tuning_kwargs)
        SERVICE.env["CT_OLLAMA_SERVER_ENV"] = json.dumps(server_env, sort_keys=True)


HANDLERS = {
    "endpoints": _cmd_endpoints,
    "models": _cmd_models,
    "pull": _cmd_pull,
    "run": _cmd_run,
    "preload": _cmd_preload,
    "unload": _cmd_unload,
    "cp": _cmd_cp,
    "rm": _cmd_rm,
    "show": _cmd_show,
}

if __name__ == "__main__":
    from cli_support import run_cli
    run_cli(lambda: main(SERVICE, configure=configure_cli, prepare_up=prepare_up, handlers=HANDLERS))
