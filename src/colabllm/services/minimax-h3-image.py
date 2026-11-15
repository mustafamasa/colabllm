#!/usr/bin/env python3
"""MiniMax-H3-Image on Colab G4 / H100 / A100 GPUs via SGLang Diffusion (notebook: minimax-h3-image.ipynb).

Deploys `MiniMaxAI/MiniMax-H3-Image` (33.1B DiT + 33.4B Qwen3-VL encoder + MiniMaxH3ImageVAE,
6-step fused PDD schedule, native 4K 16 MP text-to-image & 1–16 `<Picture 1>..<Picture 16>` multi-reference
image editing):
  - Defaults to Colab `G4` (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 96 GB GDDR7, `sm_120`) with
    `--quantization fp8-all` (`--component-residency all=resident --quantization fp8
    --component-quantizations.text_encoder fp8` + online AdaLN precompute): ~50.4 GiB peak VRAM,
    ~2.34s at 2K (`2048x2048`), ~14.45s at 4K (`4736x3552`), leaving ~45 GiB VRAM headroom for multi-ref edits.
  - Also supports lossless native BF16 on G4 (`colabllm minimax-h3-image up --quantization bf16`, ~92.2 GiB
    peak VRAM, ~18.54s at 4K) and H100/A100 (`--gpu H100` / `--gpu A100`).
  - Network-friendly internal CLI protocol (`/api/ct/*`) by default:
      * Content-addressed raw binary upload (`HEAD/POST /api/ct/uploads`) — avoids +33% Base64 bloat and
        deduplicates repeated edits on the same local reference image to 0 upload bytes.
      * Async job submission + lightweight JSON polling (`POST/GET /api/ct/jobs`) — never holds long idle
        HTTP connections over FRP / Cloudflare Tunnel / proxies and survives transient disconnects.
      * Resumable raw binary `Range` download (`GET /api/ct/jobs/{id}/content`) with server-side PNG
        compression or WebP/JPEG transcoding (`-o out.webp` shrinks 40 MB 4K outputs to ~2–4 MB), plus
        zero-transfer Google Drive I/O (`--drive-image`, `--save-drive`, `--drive-only`).
  - Also exposes standard OpenAI-compatible `/v1/images/generations`, `/v1/images/edits`, `/v1/models`,
    and `/health` protected fail-closed by Bearer / `X-API-Key` authentication.

Usage:
  colabllm minimax-h3-image up                                      # G4 + FP8 Resident (default)
  colabllm minimax-h3-image up --quantization bf16                  # G4 + Native lossless BF16 Resident
  colabllm minimax-h3-image status
  colabllm minimax-h3-image models
  colabllm minimax-h3-image generate "Rain-soaked Tokyo alley at dusk, neon reflections" -o alley.webp
  colabllm minimax-h3-image generate "Editorial watch macro shot" --aspect 16:9 --tier 4k --save-drive
  colabllm minimax-h3-image edit "Place the person in <Picture 1> into the cafe in <Picture 2>" \\
      -r person.png -r cafe.jpg -o composite.webp
  colabllm minimax-h3-image job                                     # List recent jobs
  colabllm minimax-h3-image download <job_id> -o result.png         # Download / transcode finished job
  colabllm minimax-h3-image down
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cli_support import (  # noqa: E402
    CliError,
    Deadline,
    UsageError,
    http_timeout,
    nonnegative_seconds,
    resolve_endpoint,
    safe_urlopen,
)
from colabsvc import Service, State, die, log, main  # noqa: E402

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
MAX_REFERENCE_IMAGES = 16
MAX_IMAGES_PER_JOB = 8
DIM_ALIGNMENT = 32
DRIVE_PREFIXES = ("drive://", "gdrive://", "MyDrive/", "/content/drive/")
TRANSIENT_HTTP_CODES = frozenset({408, 429, 502, 503, 504, 520, 522, 524})

MIME_BY_EXT = {
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}
EXT_BY_FORMAT = {
    "png": ".png",
    "webp": ".webp",
    "jpeg": ".jpg",
    "jpg": ".jpg",
}
FORMAT_BY_EXT = {
    ".png": "png",
    ".webp": "webp",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
}
ASPECT_CHOICES = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "9:21")
TIER_CHOICES = ("4k", "2k", "1k", "16mp", "4mp", "1mp")

SERVICE = Service(
    name="minimax-h3-image",
    notebook="minimax-h3-image.ipynb",
    service_port=8092,
    auth="builtin",
    public_port=8092,
    gpu="G4",
    high_mem=False,
    default_hostname="minimax-h3-image",
    ready_timeout=5400,
    env={
        "CT_H3_IMAGE_MODEL": "MiniMaxAI/MiniMax-H3-Image",
        "CT_H3_IMAGE_ALIAS": "minimax-h3-image",
        "CT_H3_IMAGE_QUANTIZATION": "fp8-all",
        "CT_H3_IMAGE_PLACEMENT": "auto",
        "CT_H3_IMAGE_WARMUP": "server",
    },
)

UP_ENV_FLAGS = {
    "--model": "CT_H3_IMAGE_MODEL",
    "--alias": "CT_H3_IMAGE_ALIAS",
    "--quantization": "CT_H3_IMAGE_QUANTIZATION",
    "--placement": "CT_H3_IMAGE_PLACEMENT",
}
_BASE_ENV = dict(SERVICE.env)


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(
        State(SERVICE).load(),
        url_override,
        token_override,
        url_env="MINIMAX_H3_IMAGE_URL",
        token_env="MINIMAX_H3_IMAGE_API_TOKEN",
    )


def _http_json(
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    *,
    method: str | None = None,
    timeout: int | float | None = 60,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": BROWSER_UA,
        **(extra_headers or {}),
    }
    body = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers=headers,
        method=method or ("POST" if body is not None else "GET"),
    )
    eff_timeout = http_timeout(timeout) if isinstance(timeout, (int, float)) else timeout
    try:
        with safe_urlopen(req, timeout=eff_timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            parsed = json.loads(raw) if raw.strip().startswith(("{", "[")) else {"raw": raw}
            return resp.status, parsed, hdrs
    except urllib.error.HTTPError as err:
        try:
            raw_err = err.read().decode("utf-8", errors="replace")
            hdrs = {k.lower(): v for k, v in err.headers.items()} if err.headers else {}
        finally:
            err.close()
        try:
            return err.code, json.loads(raw_err), hdrs
        except Exception:
            return err.code, {"error": raw_err}, hdrs


def _format_error(code: int, obj: dict[str, Any]) -> str:
    err = obj.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or json.dumps(err, ensure_ascii=False)
    elif err:
        msg = str(err)
    else:
        msg = json.dumps(obj, ensure_ascii=False)
    return f"HTTP {code}: {msg}"


def _is_drive_target(spec: str | None) -> bool:
    return bool(spec and spec.strip().startswith(DRIVE_PREFIXES))


def _normalize_drive_spec(spec: str) -> str:
    s = spec.strip()
    if s.startswith(("drive://", "gdrive://", "/content/drive/")):
        return s
    return f"drive://{s}"


def _upload_local_reference_dedup(
    base_url: str,
    token: str,
    local_path: Path,
    *,
    timeout: float = 180,
) -> tuple[str, bool]:
    """Upload a local reference image as raw binary with SHA-256 server-side deduplication.

    Returns `(upload_uri, cache_hit)`.
    """
    raw = local_path.read_bytes()
    if not raw:
        raise UsageError(f"Local reference image is empty: {local_path}")
    digest = hashlib.sha256(raw).hexdigest()

    # 1. Zero-byte deduplication probe
    code_chk, _, _ = _http_json(
        f"{base_url}/api/ct/uploads/{digest}",
        token,
        method="HEAD",
        timeout=min(30.0, timeout or 30.0),
    )
    if code_chk == 200:
        return f"upload://{digest}", True

    # 2. Raw binary stream upload (no +33% Base64 overhead)
    ext = local_path.suffix.lower()
    mime = MIME_BY_EXT.get(ext) or mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
    safe_fname = local_path.name.replace("\r", "").replace("\n", "")
    req = urllib.request.Request(
        f"{base_url}/api/ct/uploads?" + urllib.parse.urlencode({"filename": safe_fname}),
        data=raw,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": BROWSER_UA,
            "Content-Type": mime,
            "X-Filename": safe_fname,
        },
        method="POST",
    )
    try:
        with safe_urlopen(req, timeout=http_timeout(timeout)) as resp:
            meta = json.loads(resp.read().decode("utf-8"))
            return str(meta["uri"]), False
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        try:
            die(_format_error(err.code, json.loads(detail)))
        except Exception:
            die(f"HTTP {err.code}: {detail}")
        raise AssertionError("unreachable")


def _stage_reference_image(
    spec: str,
    *,
    force_drive: bool,
    base_url: str,
    token: str,
    use_internal_upload: bool,
    timeout: float,
) -> tuple[str, str]:
    """Resolve one CLI reference argument into an internal URI (`drive://`, `upload://`, `https://`, or `data:`)."""
    s = (spec or "").strip()
    if not s:
        raise UsageError("Reference image path must not be empty")
    if force_drive or _is_drive_target(s):
        d = _normalize_drive_spec(s)
        return d, f"drive ({d})"
    if s.startswith(("http://", "https://", "upload://", "data:image/")):
        label = "data-uri" if s.startswith("data:") else s
        return s, label

    local_p = Path(s).expanduser()
    if local_p.is_file():
        if use_internal_upload:
            uri, hit = _upload_local_reference_dedup(base_url, token, local_p, timeout=timeout)
            tag = "cached 0B" if hit else f"binary {local_p.stat().st_size / 1024:.1f} KB"
            return uri, f"local ({local_p}, {tag})"
        raw = local_p.read_bytes()
        if not raw:
            raise UsageError(f"Local reference image is empty: {local_p}")
        mime = MIME_BY_EXT.get(local_p.suffix.lower(), "image/png")
        return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}", f"local-b64 ({local_p})"

    d = _normalize_drive_spec(s)
    return d, f"drive ({d})"


def _download_binary_resumable(
    url: str,
    token: str,
    dest_path: Path,
    *,
    expected_sha256: str | None = None,
    timeout: float = 300,
    max_retries: int = 5,
) -> Path:
    """Stream raw binary image bytes directly to disk with automatic HTTP `Range` resume on interruption."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(f".{dest_path.name}.{os.getpid()}.part")
    part_path.unlink(missing_ok=True)

    attempts = 0
    try:
        while True:
            offset = part_path.stat().st_size if part_path.is_file() else 0
            headers = {
                "Authorization": f"Bearer {token}",
                "User-Agent": BROWSER_UA,
            }
            if offset > 0:
                headers["Range"] = f"bytes={offset}-"
            req = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with safe_urlopen(req, timeout=http_timeout(min(120.0, timeout) if timeout else 120.0)) as resp:
                    mode = "ab" if (offset > 0 and resp.status == 206) else "wb"
                    with open(part_path, mode) as out_f:
                        while True:
                            chunk = resp.read(65536)
                            if not chunk:
                                break
                            out_f.write(chunk)
                break
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                if isinstance(err, urllib.error.HTTPError) and err.code not in TRANSIENT_HTTP_CODES:
                    raw_err = err.read().decode("utf-8", errors="replace")
                    die(f"HTTP {err.code} downloading image: {raw_err}")
                attempts += 1
                if attempts > max_retries:
                    raise CliError(f"Download interrupted after {max_retries} retries: {err}") from err
                cur_bytes = part_path.stat().st_size if part_path.is_file() else 0
                log(f"network hiccup during binary download ({cur_bytes / 1e6:.2f} MB received); resuming...")
                time.sleep(min(6.0, 1.2 * attempts))

        if expected_sha256:
            digest = hashlib.sha256(part_path.read_bytes()).hexdigest()
            if digest.lower() != expected_sha256.lower():
                raise CliError(
                    f"Downloaded image SHA-256 mismatch (expected {expected_sha256[:12]}..., got {digest[:12]}...)"
                )
        os.replace(part_path, dest_path)
        return dest_path
    finally:
        part_path.unlink(missing_ok=True)


def _poll_job_until_done(
    base_url: str,
    token: str,
    job_id: str,
    *,
    timeout: float = 600,
    poll_interval: float = 1.5,
) -> dict[str, Any]:
    """Poll lightweight `/api/ct/jobs/{job_id}` JSON until terminal status, surviving transient network drops."""
    deadline = Deadline(
        timeout,
        message=(
            f"Timed out waiting; generation job {job_id} continues in the background on Colab. "
            f"Use `colabllm minimax-h3-image job {job_id}` or `download {job_id}` later to retrieve the result"
        ),
    )
    transient_failures = 0
    last_state = ""
    last_notice = time.monotonic()

    try:
        while True:
            req_timeout = min(25.0, deadline.remaining(25.0) or 25.0)
            try:
                code, rec, _ = _http_json(
                    f"{base_url}/api/ct/jobs/{job_id}",
                    token,
                    timeout=req_timeout,
                )
            except (urllib.error.URLError, TimeoutError, OSError) as net_err:
                transient_failures += 1
                if transient_failures > 8:
                    raise CliError(
                        f"Network repeatedly interrupted ({net_err}); background job {job_id} is still running. "
                        f"After reconnecting run `colabllm minimax-h3-image download {job_id}`"
                    ) from net_err
                deadline.sleep(min(5.0, 1.0 + transient_failures))
                continue

            if code in TRANSIENT_HTTP_CODES:
                transient_failures += 1
                if transient_failures > 8:
                    die(_format_error(code, rec))
                deadline.sleep(min(5.0, 1.0 + transient_failures))
                continue

            if code != 200:
                die(_format_error(code, rec))

            transient_failures = 0
            status = str(rec.get("status") or "")
            done_cnt = int(rec.get("completed_images") or 0)
            total_cnt = int(rec.get("total_images") or 1)
            elapsed = float(rec.get("elapsed_sec") or 0.0)
            state_sig = f"{status}:{done_cnt}/{total_cnt}"

            if status == "succeeded":
                return rec
            if status in ("failed", "cancelled"):
                err_msg = rec.get("error") or f"Job {job_id} {status}"
                die(f"Job {job_id} {status}: {err_msg}")

            now = time.monotonic()
            if state_sig != last_state or (now - last_notice) >= 8.0:
                log(f"[{job_id}] status={status} progress={done_cnt}/{total_cnt} size={rec.get('size')} elapsed={elapsed:.1f}s")
                last_state = state_sig
                last_notice = now

            deadline.sleep(poll_interval)
    except KeyboardInterrupt:
        log(f"Local wait interrupted; remote job {job_id} is still running. Check it later with `colabllm minimax-h3-image job {job_id}`.")
        raise


class _OrderedImageAction(argparse.Action):
    """Preserve exact CLI argument order (`<Picture 1>..<Picture 16>`) across `-r/--image` and `--drive-image`."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        force_drive = option_string == "--drive-image"
        items = values if isinstance(values, list) else [values]
        ordered: list[tuple[str, bool]] = list(getattr(namespace, "ordered_images", None) or [])
        for raw in items:
            if raw and str(raw).strip():
                ordered.append((str(raw).strip(), force_drive))
        setattr(namespace, "ordered_images", ordered)


def _resolve_prompt(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if getattr(args, "prompt_opt", None):
        parts.append(str(args.prompt_opt).strip())
    if getattr(args, "prompt", None):
        parts.append(str(args.prompt).strip())
    if getattr(args, "prompt_file", None):
        pf = Path(args.prompt_file).expanduser()
        if not pf.is_file():
            raise UsageError(f"Prompt file not found: {pf}")
        parts.append(pf.read_text(encoding="utf-8").strip())
    text = "\n".join(p for p in parts if p).strip()
    if not text:
        raise UsageError("Missing prompt: pass a prompt argument, `-p/--prompt`, or `--prompt-file FILE`")
    return text


def _infer_output_targets(
    args: argparse.Namespace,
    *,
    prefix: str,
) -> tuple[Path | None, str | None, str]:
    """Return `(local_output_path, save_to_drive_spec, effective_output_format)`."""
    raw_out = getattr(args, "output", None)
    save_drive = getattr(args, "save_drive", None)
    drive_only = bool(getattr(args, "drive_only", False))
    explicit_fmt = getattr(args, "output_format", None)

    local_out: Path | None = None
    if raw_out:
        if _is_drive_target(raw_out):
            save_drive = _normalize_drive_spec(raw_out)
        else:
            local_out = Path(raw_out).expanduser()

    if drive_only and not save_drive:
        save_drive = "auto"
    if drive_only:
        local_out = None
    elif local_out is None and not save_drive:
        ext = EXT_BY_FORMAT.get(explicit_fmt or "png", ".png")
        local_out = Path(f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}{ext}")

    if explicit_fmt:
        eff_fmt = explicit_fmt.lower()
    elif local_out is not None and local_out.suffix.lower() in FORMAT_BY_EXT:
        eff_fmt = FORMAT_BY_EXT[local_out.suffix.lower()]
    elif save_drive and Path(save_drive).suffix.lower() in FORMAT_BY_EXT:
        eff_fmt = FORMAT_BY_EXT[Path(save_drive).suffix.lower()]
    else:
        eff_fmt = "png"

    return local_out, save_drive, eff_fmt


def _resolve_local_dest_for_index(local_out: Path, *, index: int, total: int, fmt: str) -> Path:
    ext = EXT_BY_FORMAT.get(fmt, ".png")
    if local_out.is_dir() or str(local_out).endswith(("/", os.sep)):
        local_out.mkdir(parents=True, exist_ok=True)
        suffix = f"-{index + 1:02d}" if total > 1 else ""
        return local_out / f"minimax-h3-{time.strftime('%Y%m%d-%H%M%S')}{suffix}{ext}"
    target = local_out if local_out.suffix else local_out.with_suffix(ext)
    if total > 1:
        return target.with_name(f"{target.stem}-{index + 1:02d}{target.suffix}")
    return target


def _validate_size_flags(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if getattr(args, "size", None):
        s = str(args.size).strip()
        if s.lower() != "auto" and not re.fullmatch(r"\d+\s*[xX:]\s*\d+", s):
            raise UsageError(f"Invalid --size '{s}': expected WIDTHxHEIGHT (e.g. 4096x4096, 2048x2048) or 'auto'")
        out["size"] = s
    if getattr(args, "width", None) is not None or getattr(args, "height", None) is not None:
        if getattr(args, "width", None) is None or getattr(args, "height", None) is None:
            raise UsageError("Pass both --width and --height together (or use --size / --aspect)")
        w, h = int(args.width), int(args.height)
        if w < 64 or h < 64 or w > 8192 or h > 8192:
            raise UsageError(f"Resolution {w}x{h} out of allowed range [64, 8192]")
        out["width"] = w
        out["height"] = h
    if getattr(args, "aspect", None):
        out["aspect"] = args.aspect
    if getattr(args, "tier", None):
        out["tier"] = args.tier
    if getattr(args, "min_area", None) is not None:
        area = int(args.min_area)
        if area < 1024 or area > 4096 * 4096:
            raise UsageError(f"--min-area must be between 1024 and {4096 * 4096}")
        out["min_generation_area"] = area
    if getattr(args, "fast_preview", False):
        out["fast_preview"] = True
    return out


def _run_generation_or_edit(args: argparse.Namespace, *, is_edit: bool) -> None:
    prompt = _resolve_prompt(args)
    if args.num_images < 1 or args.num_images > MAX_IMAGES_PER_JOB:
        raise UsageError(f"-n / --num-images must be between 1 and {MAX_IMAGES_PER_JOB}")
    size_fields = _validate_size_flags(args)
    prefix = "h3-edit" if is_edit else "h3-gen"
    local_out, save_drive, eff_fmt = _infer_output_targets(args, prefix=prefix)

    ordered_raw: list[tuple[str, bool]] = list(getattr(args, "ordered_images", None) or [])
    if is_edit and not ordered_raw:
        raise UsageError("Edit requires 1–16 reference images via `-r/--image` or `--drive-image`")
    if len(ordered_raw) > MAX_REFERENCE_IMAGES:
        raise UsageError(f"MiniMax-H3-Image supports at most {MAX_REFERENCE_IMAGES} reference images (got {len(ordered_raw)})")

    base_url, token = _get_endpoint(args.url, args.api_token)
    use_internal = getattr(args, "transport", "internal") == "internal"

    staged_refs: list[str] = []
    for idx, (spec, force_drive) in enumerate(ordered_raw):
        uri, label = _stage_reference_image(
            spec,
            force_drive=force_drive,
            base_url=base_url,
            token=token,
            use_internal_upload=use_internal,
            timeout=args.timeout or 300,
        )
        staged_refs.append(uri)
        log(f"  <Picture {idx + 1}> = {label}")

    payload: dict[str, Any] = {
        "model": args.model,
        "task": "edit" if is_edit else "t2i",
        "prompt": prompt,
        "n": args.num_images,
        "generator_device": args.generator_device,
        "output_format": eff_fmt,
        "output_quality": args.quality,
        **size_fields,
    }
    if args.seed is not None:
        payload["seed"] = args.seed
    if staged_refs:
        payload["images"] = staged_refs
    if save_drive:
        payload["save_to_drive"] = save_drive
    if local_out is None and save_drive:
        payload["drive_only"] = True

    t0 = time.monotonic()

    # Optional OpenAI synchronous compatibility transport (--transport openai)
    if not use_internal:
        route = "/v1/images/edits" if is_edit else "/v1/images/generations"
        code, res, _ = _http_json(f"{base_url}{route}", token, payload=payload, timeout=args.timeout)
        if code != 200:
            die(_format_error(code, res))
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        if local_out is not None:
            items = res.get("data") or []
            for idx, item in enumerate(items):
                b64 = item.get("b64_json") if isinstance(item, dict) else None
                if b64:
                    dest = _resolve_local_dest_for_index(local_out, index=idx, total=len(items), fmt=eff_fmt)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(base64.b64decode(b64))
                    log(f"saved local image: {dest} ({dest.stat().st_size / 1024:.1f} KB)")
        for dp in res.get("rel_drive_paths") or []:
            log(f"saved Google Drive image: {dp}")
        return

    # Default network-friendly internal async flow: POST /api/ct/jobs -> poll -> binary Range stream
    code, job_view, _ = _http_json(
        f"{base_url}/api/ct/jobs",
        token,
        payload=payload,
        timeout=min(60.0, args.timeout or 60.0),
    )
    if code not in (200, 202):
        die(_format_error(code, job_view))

    job_id = str(job_view["job_id"])
    if getattr(args, "no_wait", False):
        if args.json:
            print(json.dumps(job_view, indent=2, ensure_ascii=False))
        else:
            print(job_id)
            log(f"submitted async job {job_id} (size={job_view.get('size')}); poll with `colabllm minimax-h3-image job {job_id}`")
        return

    final_rec = _poll_job_until_done(base_url, token, job_id, timeout=args.timeout)
    elapsed = time.monotonic() - t0

    outputs = final_rec.get("outputs") or []
    local_saved: list[str] = []
    if local_out is not None:
        for out_meta in outputs:
            idx = int(out_meta["index"])
            actual_fmt = str(out_meta.get("format") or eff_fmt)
            dest = _resolve_local_dest_for_index(local_out, index=idx, total=len(outputs), fmt=actual_fmt)
            dl_url = f"{base_url}/api/ct/jobs/{job_id}/content?index={idx}"
            _download_binary_resumable(
                dl_url,
                token,
                dest,
                expected_sha256=out_meta.get("sha256"),
                timeout=args.timeout or 300,
            )
            local_saved.append(str(dest))
            log(
                f"saved local image: {dest} "
                f"({out_meta.get('width')}x{out_meta.get('height')} {actual_fmt}, "
                f"{dest.stat().st_size / 1024:.1f} KB, {elapsed:.2f}s)"
            )

    for dp in final_rec.get("rel_drive_paths") or []:
        log(f"saved Google Drive image: {dp} ({elapsed:.2f}s)")

    if args.json:
        print(json.dumps({**final_rec, "local_paths": local_saved}, indent=2, ensure_ascii=False))


def cmd_generate(args: argparse.Namespace) -> None:
    _run_generation_or_edit(args, is_edit=False)


def cmd_edit(args: argparse.Namespace) -> None:
    _run_generation_or_edit(args, is_edit=True)


def cmd_job(args: argparse.Namespace) -> None:
    base_url, token = _get_endpoint(args.url, args.api_token)
    if not args.job_id:
        code, res, _ = _http_json(f"{base_url}/api/ct/jobs?limit={args.limit}", token, timeout=30)
        if code != 200:
            die(_format_error(code, res))
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return
        jobs = res.get("jobs") or []
        if not jobs:
            print("No recent jobs.")
            return
        for j in jobs:
            print(
                f"{j.get('job_id')}  [{j.get('status')}]  task={j.get('task')}  "
                f"size={j.get('size')}  progress={j.get('completed_images')}/{j.get('total_images')}  "
                f"elapsed={j.get('elapsed_sec', 0):.1f}s"
            )
        return

    if args.cancel:
        code, res, _ = _http_json(f"{base_url}/api/ct/jobs/{args.job_id}", token, method="DELETE", timeout=30)
        if code != 200:
            die(_format_error(code, res))
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(f"{res.get('job_id')}: {res.get('status')}")
        return

    if args.wait:
        rec = _poll_job_until_done(base_url, token, args.job_id, timeout=args.timeout)
    else:
        code, rec, _ = _http_json(f"{base_url}/api/ct/jobs/{args.job_id}", token, timeout=30)
        if code != 200:
            die(_format_error(code, rec))

    if args.json:
        print(json.dumps(rec, indent=2, ensure_ascii=False))
        return
    print(f"Job ID:       {rec.get('job_id')}")
    print(f"Status:       {rec.get('status')}")
    print(f"Task / Size:  {rec.get('task')} / {rec.get('size')} (seed={rec.get('seed')})")
    print(f"Progress:     {rec.get('completed_images')}/{rec.get('total_images')} ({rec.get('elapsed_sec', 0):.2f}s)")
    if rec.get("rel_drive_paths"):
        print(f"Drive outputs:{', '.join(rec['rel_drive_paths'])}")
    if rec.get("error"):
        print(f"Error:        {rec['error']}")


def cmd_download(args: argparse.Namespace) -> None:
    base_url, token = _get_endpoint(args.url, args.api_token)
    rec = _poll_job_until_done(base_url, token, args.job_id, timeout=args.timeout) if args.wait else None
    if rec is None:
        code, rec, _ = _http_json(f"{base_url}/api/ct/jobs/{args.job_id}", token, timeout=30)
        if code != 200:
            die(_format_error(code, rec))
        if rec.get("status") != "succeeded":
            die(f"Job '{args.job_id}' is '{rec.get('status')}' (pass --wait to wait for completion)")

    outputs = rec.get("outputs") or []
    if not outputs:
        die(f"Job '{args.job_id}' has no output artifacts")
    indices = [args.index] if args.index is not None else [int(o["index"]) for o in outputs]

    raw_out = args.output
    req_fmt = args.output_format
    if not req_fmt and raw_out and Path(raw_out).suffix.lower() in FORMAT_BY_EXT:
        req_fmt = FORMAT_BY_EXT[Path(raw_out).suffix.lower()]

    base_dest = Path(raw_out).expanduser() if raw_out else Path(f"{args.job_id}{EXT_BY_FORMAT.get(req_fmt or 'png', '.png')}")
    saved: list[str] = []
    for idx in indices:
        if idx < 0 or idx >= len(outputs):
            raise UsageError(f"--index {idx} out of range (0..{len(outputs) - 1})")
        out_meta = outputs[idx]
        target_fmt = (req_fmt or out_meta.get("format") or "png").lower()
        dest = _resolve_local_dest_for_index(base_dest, index=idx, total=len(indices), fmt=target_fmt)
        qs: dict[str, str] = {"index": str(idx)}
        expected_sha = out_meta.get("sha256")
        if req_fmt and req_fmt.lower() != str(out_meta.get("format") or "").lower():
            qs["format"] = req_fmt.lower()
            qs["quality"] = str(args.quality)
            expected_sha = None
        dl_url = f"{base_url}/api/ct/jobs/{args.job_id}/content?" + urllib.parse.urlencode(qs)
        _download_binary_resumable(dl_url, token, dest, expected_sha256=expected_sha, timeout=args.timeout)
        saved.append(str(dest))
        log(f"downloaded {dest} ({dest.stat().st_size / 1024:.1f} KB)")

    if args.json:
        print(json.dumps({"job_id": args.job_id, "local_paths": saved}, indent=2, ensure_ascii=False))


def cmd_ls_drive(args: argparse.Namespace) -> None:
    base_url, token = _get_endpoint(args.url, args.api_token)
    kind = "outputs" if args.outputs else ("inputs" if args.inputs else args.kind)
    qs = urllib.parse.urlencode({
        "path": args.path or "",
        "kind": kind,
        "recursive": "0" if args.no_recursive else "1",
    })
    code, res, _ = _http_json(f"{base_url}/api/ct/drive-list?{qs}", token, timeout=60)
    if code != 200:
        die(_format_error(code, res))
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    files = res.get("files") or []
    log(f"Google Drive mounted={res.get('drive_mounted')} kind={kind} matched={len(files)}")
    for item in files:
        print(f"  {item.get('rel_path', item.get('drive_path'))}  [{item.get('type', 'image')}, {item.get('size_mb', 0):.2f} MB]")


def cmd_models(args: argparse.Namespace) -> None:
    base_url, token = _get_endpoint(args.url, args.api_token)
    code_i, info_res, _ = _http_json(f"{base_url}/api/ct/info", token, timeout=30)
    code_m, mod_res, _ = _http_json(f"{base_url}/v1/models", token, timeout=30)
    if code_i != 200 and code_m != 200:
        die(_format_error(code_i, info_res))
    combined = {
        "info": info_res,
        "models": [m.get("id") for m in (mod_res.get("data") or []) if isinstance(m, dict)],
    }
    if args.json:
        print(json.dumps(combined, indent=2, ensure_ascii=False))
        return
    print(f"Service:            {info_res.get('service', 'minimax-h3-image')}")
    print(f"Engine:             {info_res.get('engine', 'sglang-diffusion')}")
    print(f"Model repo:         {info_res.get('model_repo')}")
    print(f"Served names:       {', '.join(combined['models'] or info_res.get('served_models') or [])}")
    print(
        f"Deployment config:  quantization={info_res.get('quantization')} "
        f"placement={info_res.get('placement')} warmup={info_res.get('warmup_mode')}"
    )
    print(
        f"Sampling contract:  pdd_steps={info_res.get('pdd_steps')} "
        f"min_generation_area={info_res.get('default_min_generation_area')} "
        f"max_refs={info_res.get('max_reference_images')}"
    )
    presets_4k = (info_res.get("aspect_presets") or {}).get("4k") or {}
    if presets_4k:
        summary = ", ".join(f"{k}={v}" for k, v in presets_4k.items())
        print(f"Native 4K presets:  {summary}")
    print(f"Drive inputs dir:   {info_res.get('drive_inputs_dir')}")
    print(f"Drive outputs dir:  {info_res.get('drive_outputs_dir')}")


def _add_endpoint_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", help="Override endpoint URL (or set MINIMAX_H3_IMAGE_URL)")
    p.add_argument("--api-token", help="Override Bearer API token (or set MINIMAX_H3_IMAGE_API_TOKEN)")


def _add_generation_common_flags(p: argparse.ArgumentParser, *, is_edit: bool) -> None:
    prompt_help = (
        "Editing prompt referring to ordered references as <Picture 1>, <Picture 2>, ... (<image1> also accepted)"
        if is_edit
        else "Detailed visual generation prompt (up to 4096 tokens)"
    )
    p.add_argument("prompt", nargs="?", help=prompt_help)
    p.add_argument("-p", "--prompt", dest="prompt_opt", help="Prompt text (alternative to positional argument)")
    p.add_argument("--prompt-file", help="Read prompt text from a UTF-8 file")
    size_help = (
        "Output resolution WIDTHxHEIGHT (32px aligned) or 'auto' to match <Picture 1> aspect ratio (default: auto)"
        if is_edit
        else "Output resolution WIDTHxHEIGHT (32px aligned, default: 4096x4096 at --tier 4k or 2048x2048 at --tier 2k)"
    )
    p.add_argument("--size", help=size_help)
    p.add_argument("--width", type=int, help="Output width in pixels (aligned to multiple of 32)")
    p.add_argument("--height", type=int, help="Output height in pixels (aligned to multiple of 32)")
    p.add_argument(
        "--aspect", "-a",
        choices=ASPECT_CHOICES,
        help="Official aspect ratio preset (e.g. 1:1=4096x4096, 4:3=4736x3552, 16:9=5472x3072, 21:9=6272x2688 at 4K)",
    )
    p.add_argument(
        "--tier", "-t",
        choices=TIER_CHOICES,
        default="4k",
        help="Resolution tier when --size is omitted or 'auto': 4k/16mp (native default), 2k/4mp (~2.3s fast), or 1k/1mp",
    )
    p.add_argument(
        "--min-area",
        type=int,
        help="Explicit min_generation_area integer in [1024, 16777216] (default: 4194304 = 2048x2048)",
    )
    p.add_argument(
        "--fast-preview",
        action="store_true",
        help="Set min_generation_area=width*height on <2K requests so 1K previews skip 2048x2048 internal upscaling",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument(
        "--generator-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="RNG device: 'cpu' (default, reproducible across GPUs) or 'cuda'",
    )
    p.add_argument(
        "--output-format", "-f",
        choices=("png", "webp", "jpeg"),
        help="Output image format (inferred from -o extension when omitted; use webp/jpeg for 10x smaller transfers)",
    )
    p.add_argument(
        "--quality", "-q",
        type=int,
        default=92,
        help="WebP/JPEG compression quality 1–100 (100 = lossless WebP; default: 92)",
    )
    p.add_argument(
        "-o", "--output",
        help="Local output path (file or directory) OR Google Drive path (drive://outputs/out.png)",
    )
    p.add_argument(
        "--save-drive",
        nargs="?",
        const="auto",
        help="Save output directly on mounted Google Drive (optional path, default: colab-llm/minimax-h3-image/outputs/)",
    )
    p.add_argument(
        "--drive-only", "--no-local",
        dest="drive_only",
        action="store_true",
        help="Save only on Google Drive and transfer zero output image bytes over the network",
    )
    p.add_argument(
        "--no-wait", "--async",
        dest="no_wait",
        action="store_true",
        help="Submit job via /api/ct/jobs and exit immediately with job_id without waiting",
    )
    p.add_argument(
        "--transport",
        choices=("internal", "openai"),
        default="internal",
        help="Wire protocol: 'internal' (/api/ct/jobs + binary Range download, default) or 'openai' (/v1/images/*)",
    )
    p.add_argument(
        "--model",
        default="MiniMaxAI/MiniMax-H3-Image",
        help="Served model name (default: MiniMaxAI/MiniMax-H3-Image)",
    )
    p.add_argument(
        "--timeout",
        type=nonnegative_seconds,
        default=600,
        help="Total wait timeout in seconds; 0 = unlimited (default: 600)",
    )
    p.add_argument("--json", action="store_true", help="Print structured JSON summary")
    _add_endpoint_flags(p)


def _build_custom_parser(_parser: argparse.ArgumentParser, sub: Any) -> None:
    for cmd_name, help_msg in (
        ("generate", "Generate native 4K / 2K image(s) via internal async job & binary streaming"),
        ("t2i", "Alias for 'generate' (Text-to-Image)"),
    ):
        gp = sub.add_parser(cmd_name, help=help_msg)
        gp.add_argument("-n", "--num-images", type=int, default=1, help="Number of images to generate (1–8, default: 1)")
        _add_generation_common_flags(gp, is_edit=False)

    for cmd_name, help_msg in (
        ("edit", "Edit or combine 1–16 reference images (<Picture 1>..<Picture 16>) with deduped binary upload"),
        ("i2i", "Alias for 'edit' (Image-to-Image / multi-reference editing)"),
    ):
        ep = sub.add_parser(cmd_name, help=help_msg)
        ep.add_argument("-n", "--num-images", type=int, default=1, help="Number of edited variations (1–8, default: 1)")
        ep.add_argument(
            "-r", "--image", "--ref",
            dest="ordered_images",
            action=_OrderedImageAction,
            nargs="+",
            help="Reference image(s) in order (1–16, bound as <Picture 1>..<Picture 16>): local file, https://, upload://, or drive://",
        )
        ep.add_argument(
            "--drive-image",
            dest="ordered_images",
            action=_OrderedImageAction,
            nargs="+",
            help="Reference image(s) read directly from mounted Google Drive (zero client upload)",
        )
        _add_generation_common_flags(ep, is_edit=True)

    for jcmd in ("job", "jobs"):
        jp = sub.add_parser(jcmd, help="Check, wait on, cancel, or list async /api/ct/jobs tasks")
        jp.add_argument("job_id", nargs="?", help="Job ID (omit to list recent jobs)")
        jp.add_argument("--wait", action="store_true", help="Wait until the job completes")
        jp.add_argument("--cancel", action="store_true", help="Cancel a queued/running job")
        jp.add_argument("--limit", type=int, default=20, help="Max jobs to list when job_id is omitted (default: 20)")
        jp.add_argument("--timeout", type=nonnegative_seconds, default=600, help="Wait timeout in seconds (default: 600)")
        jp.add_argument("--json", action="store_true", help="Print raw JSON response")
        _add_endpoint_flags(jp)

    dlp = sub.add_parser("download", help="Download (and optionally transcode) output image(s) of a completed job")
    dlp.add_argument("job_id", help="Completed (or running with --wait) job ID")
    dlp.add_argument("-o", "--output", help="Local destination file or directory")
    dlp.add_argument("--index", type=int, help="Specific output image index (default: download all outputs in job)")
    dlp.add_argument(
        "--output-format", "-f",
        choices=("png", "webp", "jpeg"),
        help="Transcode on the server before downloading (e.g. webp/jpeg for smaller network payload)",
    )
    dlp.add_argument("--quality", "-q", type=int, default=92, help="WebP/JPEG quality 1–100 (default: 92)")
    dlp.add_argument("--wait", action="store_true", help="Wait if the job is still queued/running")
    dlp.add_argument("--timeout", type=nonnegative_seconds, default=600, help="Download/wait timeout in seconds")
    dlp.add_argument("--json", action="store_true", help="Print JSON summary")
    _add_endpoint_flags(dlp)

    lsd = sub.add_parser("ls-drive", help="List reference images or generated outputs on mounted Google Drive")
    lsd.add_argument("path", nargs="?", default="", help="Drive folder or file path (e.g. inputs, outputs)")
    lsd.add_argument("--kind", choices=("images", "inputs", "outputs"), default="images", help="Category (default: images)")
    lsd.add_argument("--inputs", action="store_true", help="Shortcut for --kind inputs")
    lsd.add_argument("--outputs", action="store_true", help="Shortcut for --kind outputs")
    lsd.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    lsd.add_argument("--json", action="store_true", help="Print raw JSON response")
    _add_endpoint_flags(lsd)

    mp = sub.add_parser("models", help="Show active MiniMax-H3-Image SGLang config, 4K presets & Google Drive paths")
    mp.add_argument("--json", action="store_true", help="Print raw JSON response")
    _add_endpoint_flags(mp)


def configure_cli(parser: argparse.ArgumentParser, commands: Any, up: argparse.ArgumentParser) -> None:
    _build_custom_parser(parser, commands)
    up.add_argument(
        "--model",
        dest="up_CT_H3_IMAGE_MODEL",
        help="Hugging Face repo (default: MiniMaxAI/MiniMax-H3-Image)",
    )
    up.add_argument(
        "--alias",
        dest="up_CT_H3_IMAGE_ALIAS",
        help="Served model alias (default: minimax-h3-image)",
    )
    up.add_argument(
        "--quantization",
        dest="up_CT_H3_IMAGE_QUANTIZATION",
        choices=("fp8-all", "bf16", "none", "fp8-dit", "mxfp8-all", "mxfp8-dit"),
        help=(
            "Weight precision: fp8-all (default on G4, DiT + Qwen3-VL in FP8, ~50.4 GiB peak VRAM, 2.34s@2K / 14.45s@4K), "
            "bf16 / none (native lossless BF16 + online AdaLN, ~92.2 GiB peak VRAM on G4, 18.54s@4K), "
            "fp8-dit, mxfp8-all, or mxfp8-dit"
        ),
    )
    up.add_argument(
        "--placement",
        dest="up_CT_H3_IMAGE_PLACEMENT",
        choices=("auto", "resident", "offload", "layerwise"),
        help=(
            "Component placement: auto (default: all=resident on G4 96 GB; text_encoder offload on 80 GB BF16), "
            "resident, offload (component-offload), or layerwise (layerwise-offload)"
        ),
    )
    up.add_argument(
        "--no-warmup",
        dest="up_no_warmup",
        action="store_true",
        help="Skip SGLang startup PDD grid warmup (--warmup-mode off) for faster initial boot",
    )


def prepare_up(args: argparse.Namespace) -> None:
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    for _flag, env_key in UP_ENV_FLAGS.items():
        val = getattr(args, "up_" + env_key, None)
        if val is not None:
            SERVICE.env[env_key] = str(val)
    if getattr(args, "up_no_warmup", False):
        SERVICE.env["CT_H3_IMAGE_WARMUP"] = "off"


if __name__ == "__main__":
    from cli_support import run_cli

    run_cli(
        lambda: main(
            SERVICE,
            configure=configure_cli,
            prepare_up=prepare_up,
            handlers={
                "generate": cmd_generate,
                "t2i": cmd_generate,
                "edit": cmd_edit,
                "i2i": cmd_edit,
                "job": cmd_job,
                "jobs": cmd_job,
                "download": cmd_download,
                "ls-drive": cmd_ls_drive,
                "models": cmd_models,
            },
        )
    )
