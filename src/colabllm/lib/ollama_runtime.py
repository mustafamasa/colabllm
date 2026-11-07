"""Shared configuration, Caddy routing, and model management helpers for the Ollama Colab service.

Imported both by ``src/colabllm/services/ollama.py`` on the local CLI side (standard library only)
and embedded into ``src/colabllm/services/ollama.ipynb`` on the Colab VM.
"""
from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable

DEFAULT_MODEL = "qwen2.5:0.5b"
DEFAULT_KEEP_ALIVE = "30m"

COMPAT_APIS = ("ollama", "openai", "anthropic")
COMPAT_ALIASES = {
    "ollama": "ollama",
    "native": "ollama",
    "api": "ollama",
    "openai": "openai",
    "oai": "openai",
    "v1": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
}

KV_CACHE_TYPES = ("f16", "q8_0", "q4_0")
MODELS_STORAGE_MODES = ("ssd", "drive")

# Presets grounded in https://docs.ollama.com/faq and https://docs.ollama.com/context-length
OPTIMIZATION_PROFILES: dict[str, dict[str, str]] = {
    "balanced": {
        "OLLAMA_KEEP_ALIVE": DEFAULT_KEEP_ALIVE,
    },
    "low-vram": {
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_KEEP_ALIVE": DEFAULT_KEEP_ALIVE,
    },
    "coding": {
        "OLLAMA_CONTEXT_LENGTH": "65536",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_KEEP_ALIVE": "-1",
    },
    "agent": {
        "OLLAMA_CONTEXT_LENGTH": "65536",
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_KV_CACHE_TYPE": "q8_0",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
        "OLLAMA_KEEP_ALIVE": "-1",
    },
    "throughput": {
        "OLLAMA_FLASH_ATTENTION": "1",
        "OLLAMA_NUM_PARALLEL": "4",
        "OLLAMA_MAX_QUEUE": "512",
        "OLLAMA_KEEP_ALIVE": "-1",
    },
}

ALLOWED_EXTRA_ENV_EXACT = {
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "GGML_VK_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_OVERRIDE_GFX_VERSION",
    "HTTPS_PROXY",
    "https_proxy",
    "NO_PROXY",
    "no_proxy",
    "JETSON_JETPACK",
}
FORBIDDEN_ENV_KEYS = {
    "OLLAMA_HOST",   # Must always bind to 127.0.0.1:<SERVICE_PORT> behind Caddy
    "HTTP_PROXY",    # Ollama docs warn HTTP_PROXY breaks local client connections; use HTTPS_PROXY
    "http_proxy",
}

MODEL_SPEC_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,254}$")
DURATION_RE = re.compile(r"^-?(?:\d+(?:\.\d+)?(?:ns|us|µs|ms|s|m|h))+$|^-?\d+$")
BYTE_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]i?b?)?\s*$", re.IGNORECASE)
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[A-Za-z0-9._-]+)?$")

# Caddy path patterns per API family (zero collision across /api/*, OpenAI /v1/*, Anthropic /v1/messages*)
API_FAMILY_PATHS: dict[str, tuple[str, ...]] = {
    "ollama": ("/api/*",),
    "openai": (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/responses",
        "/v1/embeddings",
        "/v1/models",
        "/v1/models/*",
        "/v1/audio/*",
    ),
    "anthropic": (
        "/v1/messages",
        "/v1/messages/*",
        "/v1/models",
        "/v1/models/*",
    ),
}
MODEL_MUTATION_PATHS = (
    "/api/pull",
    "/api/push",
    "/api/create",
    "/api/copy",
    "/api/delete",
    "/api/blobs/*",
)


def validate_model_spec(spec: str) -> str:
    """Validate an Ollama library or Hugging Face model tag, rejecting traversal and CLI flags."""
    cleaned = (spec or "").strip()
    if not cleaned or ".." in cleaned or not MODEL_SPEC_RE.fullmatch(cleaned):
        raise ValueError(
            f"Invalid model name {spec!r}: use a valid Ollama model tag (e.g. qwen3:8b) or a Hugging Face GGUF reference (e.g. hf.co/org/repo:Q4_K_M)"
        )
    return cleaned


def parse_model_specs(
    raw_values: list[str] | str | None,
    *,
    no_models: bool = False,
    default: str | None = DEFAULT_MODEL,
) -> list[str]:
    """Parse repeatable/comma-separated model specs into an ordered deduplicated list."""
    if isinstance(raw_values, str):
        items = [raw_values]
    elif raw_values is None:
        items = []
    else:
        items = list(raw_values)

    if no_models:
        if any(part.strip() for item in items for part in str(item).split(",")):
            raise ValueError("--no-models cannot be combined with a non-empty --models")
        return []

    if raw_values is None:
        return [validate_model_spec(default)] if default else []

    tokens: list[str] = []
    explicit_none = False
    for item in items:
        for part in str(item).split(","):
            tok = part.strip()
            if not tok:
                continue
            if tok.lower() in ("none", "off"):
                explicit_none = True
                continue
            tokens.append(validate_model_spec(tok))

    if explicit_none and tokens:
        raise ValueError("The model list cannot contain both 'none' and specific model names")
    if explicit_none or not tokens:
        return []
    return list(dict.fromkeys(tokens))


def parse_model_aliases(raw_values: list[str] | str | None) -> list[tuple[str, str]]:
    """Parse repeatable/comma-separated ``ALIAS=SOURCE_MODEL`` mappings."""
    if not raw_values:
        return []
    items = [raw_values] if isinstance(raw_values, str) else list(raw_values)
    mapping: dict[str, str] = {}
    for item in items:
        for part in str(item).split(","):
            entry = part.strip()
            if not entry:
                continue
            alias_raw, sep, source_raw = entry.partition("=")
            if not sep:
                raise ValueError(f"--alias must have the form ALIAS=MODEL (e.g. gpt-4o=qwen3:8b), got: {entry!r}")
            alias = validate_model_spec(alias_raw)
            source = validate_model_spec(source_raw)
            if alias == source:
                raise ValueError(f"A model alias cannot be the same as its source model: {alias!r}")
            if alias in mapping and mapping[alias] != source:
                raise ValueError(f"Model alias {alias!r} is mapped to different models: {mapping[alias]!r} and {source!r}")
            mapping[alias] = source
    return list(mapping.items())


def models_to_pull(models: list[str], aliases: list[tuple[str, str]]) -> list[str]:
    """Return ordered deduplicated list of models to download, including alias source models."""
    combined = list(models)
    alias_names = {alias for alias, _ in aliases}
    for _, source in aliases:
        if source not in combined and source not in alias_names:
            combined.append(source)
    return list(dict.fromkeys(combined))


def parse_compat_apis(raw_values: list[str] | str | None) -> tuple[str, ...]:
    """Normalize compatible API selection into canonical order ``('ollama', 'openai', 'anthropic')``."""
    if not raw_values:
        return COMPAT_APIS
    items = [raw_values] if isinstance(raw_values, str) else list(raw_values)
    selected: set[str] = set()
    for item in items:
        for part in str(item).split(","):
            tok = part.strip().lower()
            if not tok:
                continue
            if tok in ("all", "*"):
                return COMPAT_APIS
            canonical = COMPAT_ALIASES.get(tok)
            if not canonical:
                raise ValueError(
                    f"Unsupported compatibility API {part.strip()!r}; choices: all, ollama, openai, anthropic (comma-separated)"
                )
            selected.add(canonical)
    if not selected:
        raise ValueError("--compat needs at least one API (all, ollama, openai, anthropic)")
    return tuple(api for api in COMPAT_APIS if api in selected)


def validate_duration(value: str, flag_name: str = "duration") -> str:
    """Validate Go duration strings or integer seconds accepted by Ollama (e.g. 30m, 24h, -1, 0)."""
    cleaned = str(value).strip()
    if not cleaned or not DURATION_RE.fullmatch(cleaned):
        raise ValueError(
            f"Invalid {flag_name} ({value!r}): use a Go duration (e.g. 30m, 24h, 5m) or integer seconds (-1 keeps the model loaded forever, 0 unloads immediately / no limit)"
        )
    return cleaned


def normalize_api_keep_alive(value: str | int | None) -> str | int | None:
    """Convert unitless integer strings (e.g. '-1', '0', '3600') to int for Ollama JSON Duration unmarshaling."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    cleaned = validate_duration(str(value), "keep_alive")
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    return cleaned


def parse_byte_size(value: str | int, flag_name: str = "--gpu-overhead") -> int:
    """Parse integer bytes or human size strings (e.g. 512M, 1.5G, 1073741824) into non-negative bytes."""
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"{flag_name} cannot be negative")
        return value
    raw = str(value).strip()
    match = BYTE_SIZE_RE.fullmatch(raw)
    if not match:
        raise ValueError(f"Invalid {flag_name} ({value!r}): use a non-negative byte count or a size with a unit (e.g. 512M, 1G)")
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1024, "kb": 1024, "kib": 1024,
        "m": 1024**2, "mb": 1024**2, "mib": 1024**2,
        "g": 1024**3, "gb": 1024**3, "gib": 1024**3,
        "t": 1024**4, "tb": 1024**4, "tib": 1024**4,
    }
    if unit not in multipliers or not math.isfinite(number):
        raise ValueError(f"Invalid unit for {flag_name} ({value!r})")
    return int(round(number * multipliers[unit]))


def validate_ollama_version(value: str) -> str:
    cleaned = str(value).strip().lstrip("v")
    if not VERSION_RE.fullmatch(cleaned):
        raise ValueError(f"Invalid Ollama version {value!r}: e.g. 0.17.7 or 0.18.0-rc1")
    return cleaned


def parse_extra_ollama_env(entries: list[str] | None) -> dict[str, str]:
    """Validate repeatable ``KEY=VALUE`` overrides for ``ollama serve``."""
    out: dict[str, str] = {}
    for raw in entries or ():
        key, sep, val = str(raw).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"--ollama-env must have the form KEY=VALUE, got: {raw!r}")
        if key in FORBIDDEN_ENV_KEYS or key.upper() in FORBIDDEN_ENV_KEYS:
            raise ValueError(
                f"{key} cannot be overridden with --ollama-env: OLLAMA_HOST must stay bound to 127.0.0.1, and proxies are configured with HTTPS_PROXY"
            )
        if not (key.startswith("OLLAMA_") or key in ALLOWED_EXTRA_ENV_EXACT or key.startswith("HSA_OVERRIDE_GFX_VERSION_")):
            raise ValueError(
                f"--ollama-env only accepts OLLAMA_* or supported GPU/proxy environment variables, got: {key!r}"
            )
        if "\x00" in val or "\n" in val or "\r" in val:
            raise ValueError(f"The value of --ollama-env {key} cannot contain newlines or null bytes")
        out[key] = val
    return out


def build_ollama_server_env(
    *,
    profile: str | None = None,
    context_length: int | None = None,
    flash_attention: bool | None = None,
    kv_cache_type: str | None = None,
    num_parallel: int | None = None,
    max_loaded_models: int | None = None,
    max_queue: int | None = None,
    keep_alive: str | None = None,
    load_timeout: str | None = None,
    gpu_overhead: str | int | None = None,
    sched_spread: bool | None = None,
    max_transfer_streams: int | None = None,
    origins: str | None = None,
    no_cloud: bool | None = None,
    ollama_debug: str | None = None,
    extra_env: list[str] | None = None,
) -> dict[str, str]:
    """Build validated ``OLLAMA_*`` server environment variables (excluding host/models path)."""
    env: dict[str, str] = {"OLLAMA_KEEP_ALIVE": DEFAULT_KEEP_ALIVE}
    if profile is not None:
        prof = profile.strip().lower()
        if prof not in OPTIMIZATION_PROFILES:
            raise ValueError(f"Unsupported optimization profile {profile!r}; choices: {', '.join(OPTIMIZATION_PROFILES)}")
        env.update(OPTIMIZATION_PROFILES[prof])

    if context_length is not None:
        if int(context_length) < 256:
            raise ValueError("--context-length must be >= 256")
        env["OLLAMA_CONTEXT_LENGTH"] = str(int(context_length))

    if kv_cache_type is not None:
        kvt = kv_cache_type.strip().lower()
        if kvt not in KV_CACHE_TYPES:
            raise ValueError(f"--kv-cache-type must be one of {', '.join(KV_CACHE_TYPES)}")
        if kvt in ("q8_0", "q4_0") and flash_attention is False:
            raise ValueError(f"KV cache quantization ({kvt}) requires Flash Attention; it cannot be combined with --no-flash-attention")
        env["OLLAMA_KV_CACHE_TYPE"] = kvt
        if kvt in ("q8_0", "q4_0") and flash_attention is None:
            # Ollama requires Flash Attention when quantizing the K/V cache.
            env["OLLAMA_FLASH_ATTENTION"] = "1"

    if flash_attention is not None:
        env["OLLAMA_FLASH_ATTENTION"] = "1" if flash_attention else "0"
        if not flash_attention and env.get("OLLAMA_KV_CACHE_TYPE") in ("q8_0", "q4_0"):
            # Explicit --no-flash-attention overrides a preset's q8_0 KV cache back to f16 unless user also forced kv_cache_type
            env["OLLAMA_KV_CACHE_TYPE"] = "f16"

    if num_parallel is not None:
        if int(num_parallel) < 1:
            raise ValueError("--num-parallel must be >= 1")
        env["OLLAMA_NUM_PARALLEL"] = str(int(num_parallel))

    if max_loaded_models is not None:
        if int(max_loaded_models) < 0:
            raise ValueError("--max-loaded-models must be >= 0 (0 means automatic)")
        env["OLLAMA_MAX_LOADED_MODELS"] = str(int(max_loaded_models))

    if max_queue is not None:
        if int(max_queue) < 1:
            raise ValueError("--max-queue must be >= 1")
        env["OLLAMA_MAX_QUEUE"] = str(int(max_queue))

    if keep_alive is not None:
        env["OLLAMA_KEEP_ALIVE"] = validate_duration(keep_alive, "--keep-alive")

    if load_timeout is not None:
        env["OLLAMA_LOAD_TIMEOUT"] = validate_duration(load_timeout, "--load-timeout")

    if gpu_overhead is not None:
        env["OLLAMA_GPU_OVERHEAD"] = str(parse_byte_size(gpu_overhead, "--gpu-overhead"))

    if sched_spread is not None:
        env["OLLAMA_SCHED_SPREAD"] = "1" if sched_spread else "0"

    if max_transfer_streams is not None:
        if int(max_transfer_streams) < 1:
            raise ValueError("--max-transfer-streams must be >= 1")
        env["OLLAMA_MAX_TRANSFER_STREAMS"] = str(int(max_transfer_streams))

    if origins is not None:
        cleaned_origins = origins.strip()
        if "\n" in cleaned_origins or "\r" in cleaned_origins or "\x00" in cleaned_origins:
            raise ValueError("--origins cannot contain newlines or null bytes")
        env["OLLAMA_ORIGINS"] = cleaned_origins

    if no_cloud is not None:
        env["OLLAMA_NO_CLOUD"] = "1" if no_cloud else "0"

    if ollama_debug is not None:
        dbg = str(ollama_debug).strip().lower()
        if dbg in ("0", "false", "off"):
            env["OLLAMA_DEBUG"] = "0"
        elif dbg in ("1", "true", "on", "debug"):
            env["OLLAMA_DEBUG"] = "1"
        elif dbg in ("2", "trace"):
            env["OLLAMA_DEBUG"] = "2"
        else:
            raise ValueError("--ollama-debug accepts 0, 1 (debug) or 2 (trace)")

    env.update(parse_extra_ollama_env(extra_env))
    return env


def allowed_caddy_paths(compat_apis: tuple[str, ...]) -> list[str]:
    """Return deduplicated Caddy path matchers for the selected API families (always keeping `/` health)."""
    paths: list[str] = ["/"]
    for api in compat_apis:
        for pat in API_FAMILY_PATHS.get(api, ()):
            if pat not in paths:
                paths.append(pat)
    return paths


def _escape_caddy_quoted(value: str, label: str) -> str:
    if any(ch in value for ch in ("\r", "\n", "\x00", '"', "\\")):
        raise ValueError(f"Unsafe character in Caddy {label}")
    return value


def render_ollama_caddyfile(
    *,
    public_port: int,
    service_port: int,
    auth_user: str,
    hashed_pass: str,
    api_token: str,
    cookie_tok: str,
    compat_apis: tuple[str, ...] = COMPAT_APIS,
    read_only_models: bool = False,
) -> str:
    """Render Caddyfile supporting Bearer, Anthropic X-Api-Key, browser Basic Auth, and API family filtering."""
    safe_user = _escape_caddy_quoted(auth_user, "auth_user")
    safe_token = _escape_caddy_quoted(api_token, "api_token")
    safe_cookie = _escape_caddy_quoted(cookie_tok, "cookie_tok")
    compat = parse_compat_apis(list(compat_apis))

    matcher_defs: list[str] = [
        f'    @bearer header Authorization "Bearer {safe_token}"',
        f'    @apikey header X-Api-Key "{safe_token}"',
        f"    @cookie header Cookie *ct_session={safe_cookie}*",
    ]
    guard_lines: list[str] = []

    if read_only_models:
        matcher_defs.append(f"    @blocked_mgmt path {' '.join(MODEL_MUTATION_PATHS)}")
        guard_lines.extend([
            '            header @blocked_mgmt Content-Type "application/json"',
            '            respond @blocked_mgmt "{\\"error\\":\\"model management endpoints are disabled (--read-only-models)\\"}" 403',
        ])

    if set(compat) != set(COMPAT_APIS):
        allowed = " ".join(allowed_caddy_paths(compat))
        matcher_defs.append(f"    @disallowed_api not path {allowed}")
        enabled_str = ",".join(compat)
        guard_lines.extend([
            '            header @disallowed_api Content-Type "application/json"',
            f'            respond @disallowed_api "{{\\"error\\":\\"API endpoint is not enabled (active compat: {enabled_str})\\"}}" 403',
        ])

    proxy_block = (
        f"            reverse_proxy 127.0.0.1:{int(service_port)} {{\n"
        f"                header_up Host {{upstream_hostport}}\n"
        f"                header_up Origin http://{{upstream_hostport}}\n"
        f"                header_up -Authorization\n"
        f"                header_up -X-Api-Key\n"
        f"                flush_interval -1\n"
        f"            }}"
    )
    route_inner = "\n".join([*guard_lines, proxy_block])
    authenticated_route = f"        route {{\n{route_inner}\n        }}"
    matchers_text = "\n".join(matcher_defs)

    return f"""{{
    admin off
    auto_https off
}}
:{int(public_port)} {{
    encode zstd gzip
{matchers_text}
    handle @bearer {{
{authenticated_route}
    }}
    handle @apikey {{
{authenticated_route}
    }}
    handle @cookie {{
{authenticated_route}
    }}
    handle {{
        route {{
            basic_auth {{
                "{safe_user}" {hashed_pass}
            }}
            header +Set-Cookie "ct_session={safe_cookie}; Path=/; HttpOnly; SameSite=Lax"
{route_inner}
        }}
    }}
}}
"""


def compat_endpoints_info(
    base_url: str,
    compat_apis: tuple[str, ...] = COMPAT_APIS,
    *,
    api_token: str | None = None,
    read_only_models: bool = False,
) -> dict[str, Any]:
    """Build structured metadata showing active compatible APIs, Base URLs, and SDK settings."""
    clean_url = (base_url or "").rstrip("/")
    compat = parse_compat_apis(list(compat_apis))
    token_display = api_token if api_token else "<API_TOKEN>"
    interfaces: dict[str, Any] = {}
    if "ollama" in compat:
        endpoints = ["/api/chat", "/api/generate", "/api/embed", "/api/tags", "/api/ps", "/api/show", "/api/version"]
        if not read_only_models:
            endpoints.extend(["/api/pull", "/api/copy", "/api/create", "/api/delete"])
        interfaces["ollama"] = {
            "base_url": clean_url,
            "auth_header": f"Authorization: Bearer {token_display}",
            "read_only_models": read_only_models,
            "endpoints": endpoints,
        }
    if "openai" in compat:
        interfaces["openai"] = {
            "base_url": f"{clean_url}/v1" if clean_url else "/v1",
            "api_key": token_display,
            "auth_header": f"Authorization: Bearer {token_display}",
            "endpoints": [
                "/v1/chat/completions",
                "/v1/responses",
                "/v1/completions",
                "/v1/embeddings",
                "/v1/models",
                "/v1/audio/speech",
                "/v1/audio/transcriptions",
                "/v1/audio/translations",
            ],
        }
    if "anthropic" in compat:
        interfaces["anthropic"] = {
            "base_url": clean_url,
            "api_key": token_display,
            "auth_headers": [
                f"Authorization: Bearer {token_display}",
                f"x-api-key: {token_display}",
            ],
            "env": {
                "ANTHROPIC_BASE_URL": clean_url,
                "ANTHROPIC_API_KEY": token_display,
            },
            "endpoints": ["/v1/messages", "/v1/models"],
        }
    return {
        "url": clean_url or None,
        "compat": list(compat),
        "read_only_models": read_only_models,
        "interfaces": interfaces,
    }


def pull_model_stream(
    api_base: str,
    model: str,
    *,
    headers: dict[str, str] | None = None,
    opener: Callable[..., Any] | None = None,
    log_fn: Callable[[str], None] | None = None,
    timeout: float | None = 1800,
    retries: int = 3,
) -> dict[str, Any]:
    """Stream ``POST /api/pull`` with throttled progress logging and automatic resume retries."""
    spec = validate_model_spec(model)
    open_url = opener or urllib.request.urlopen
    req_headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    if headers:
        req_headers.update(headers)
    payload = json.dumps({"model": spec, "stream": True}).encode("utf-8")
    url = f"{api_base.rstrip('/')}/api/pull"

    last_err: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        req = urllib.request.Request(url, data=payload, headers=req_headers, method="POST")
        last_log_t = 0.0
        last_status = ""
        final_event: dict[str, Any] = {}
        try:
            with open_url(req, timeout=timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("error"):
                        raise RuntimeError(f"Pulling model {spec} failed: {event['error']}")
                    final_event = event
                    status = str(event.get("status") or "")
                    total = event.get("total")
                    completed = event.get("completed")
                    now = time.monotonic()
                    if log_fn:
                        if isinstance(total, int) and total > 0 and isinstance(completed, int):
                            pct = completed * 100.0 / total
                            if now - last_log_t >= 5.0 or (status != last_status and pct in (0.0, 100.0)):
                                log_fn(
                                    f"pulling {spec}: {status} {pct:.1f}% "
                                    f"({completed / 1e9:.2f}/{total / 1e9:.2f} GB)"
                                )
                                last_log_t = now
                                last_status = status
                        elif status and status != last_status:
                            log_fn(f"pulling {spec}: {status}")
                            last_status = status
                            last_log_t = now
            return {"model": spec, "status": final_event.get("status", "success")}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as err:
            # HTTP 4xx errors (e.g. 401/403/404 invalid model) should fail fast without retrying
            if isinstance(err, urllib.error.HTTPError) and 400 <= err.code < 500 and err.code != 429:
                body = err.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {err.code} pulling {spec}: {body}") from None
            last_err = err
            if attempt < retries:
                if log_fn:
                    log_fn(f"pull {spec} interrupted ({err}); retrying ({attempt + 1}/{retries})...")
                time.sleep(min(2 * attempt, 6))
                continue
            raise RuntimeError(f"Pulling model {spec} failed (after {retries} retries): {err}") from None
    raise RuntimeError(f"Pulling model {spec} failed: {last_err}")


def copy_model_alias(
    api_base: str,
    source: str,
    destination: str,
    *,
    headers: dict[str, str] | None = None,
    opener: Callable[..., Any] | None = None,
    timeout: float | None = 60,
) -> dict[str, str]:
    """Create a model alias via ``POST /api/copy``."""
    src = validate_model_spec(source)
    dst = validate_model_spec(destination)
    open_url = opener or urllib.request.urlopen
    req_headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    if headers:
        req_headers.update(headers)
    payload = json.dumps({"source": src, "destination": dst}).encode("utf-8")
    req = urllib.request.Request(f"{api_base.rstrip('/')}/api/copy", data=payload, headers=req_headers, method="POST")
    try:
        with open_url(req, timeout=timeout) as resp:
            resp.read()
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {err.code} copying model {src} -> {dst}: {body}") from None
    return {"source": src, "alias": dst, "status": "copied"}


def preload_or_unload_model(
    api_base: str,
    model: str,
    *,
    keep_alive: str | int | None = None,
    headers: dict[str, str] | None = None,
    opener: Callable[..., Any] | None = None,
    timeout: float | None = 600,
) -> dict[str, Any]:
    """Preload a model into VRAM (or unload with ``keep_alive=0``), falling back to ``/api/embed`` for embedding models."""
    spec = validate_model_spec(model)
    open_url = opener or urllib.request.urlopen
    req_headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    if headers:
        req_headers.update(headers)

    gen_body: dict[str, Any] = {"model": spec}
    normalized_ka = normalize_api_keep_alive(keep_alive)
    if normalized_ka is not None:
        gen_body["keep_alive"] = normalized_ka

    gen_req = urllib.request.Request(
        f"{api_base.rstrip('/')}/api/generate",
        data=json.dumps(gen_body).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    try:
        with open_url(gen_req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
            data = json.loads(raw) if raw else {}
            return {"model": spec, "endpoint": "/api/generate", "done_reason": data.get("done_reason", "load")}
    except urllib.error.HTTPError as err:
        body = err.read().decode("utf-8", errors="replace")
        # Embedding-only models return HTTP 400 "does not support generate" -> preload via /api/embed
        if err.code == 400 and ("embed" in body.lower() or "does not support generate" in body.lower()):
            emb_body: dict[str, Any] = {"model": spec, "input": "" if normalized_ka == 0 else "warmup"}
            if normalized_ka is not None:
                emb_body["keep_alive"] = normalized_ka
            emb_req = urllib.request.Request(
                f"{api_base.rstrip('/')}/api/embed",
                data=json.dumps(emb_body).encode("utf-8"),
                headers=req_headers,
                method="POST",
            )
            try:
                with open_url(emb_req, timeout=timeout) as resp2:
                    resp2.read()
                    return {"model": spec, "endpoint": "/api/embed", "done_reason": "load"}
            except urllib.error.HTTPError as err2:
                body2 = err2.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {err2.code} preloading embedding model {spec}: {body2}") from None
        raise RuntimeError(f"HTTP {err.code} preloading model {spec}: {body}") from None
