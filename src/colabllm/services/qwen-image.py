#!/usr/bin/env python3
"""Qwen-Image-2.1 on Colab G4 / H100 / A100 GPUs (notebook: qwen-image.ipynb).

Follows https://recipes.vllm.ai/Qwen/Qwen-Image-2.1:
  - Defaults to Colab `G4` (RTX PRO 6000 Blackwell, 96 GB VRAM) for full BF16 precision with ~55 GB
    headroom; also supports `--gpu H100` (80 GB HBM3, fastest latency) and `--gpu A100 --quantization fp8-all`
    (40 GB / 80 GB). Note: `L4` (24 GB) cannot fit the 33.1 GB checkpoint without severe CPU offload.
  - Serves `Qwen/Qwen-Image-2.1` (7.1B single-stream DiT with exact cross-step prefix KV cache +
    Qwen3-VL-8B vision-language text encoder + 16x 4-channel RGBA autoencoder) via `vllm serve --omni`
    on loopback `:8091` with `--vae-use-tiling` enabled by default.
  - Enforces fail-closed Bearer / `X-API-Key` authentication across every HTTP route (with stateless
    CORS `OPTIONS` preflight headers for browser studio integrations such as `open-pose-costume`).
  - Automatically applies Qwen-Image-2.1's recommended 40-step sampling (`num_inference_steps=40`) and
    True CFG defaults (`true_cfg_scale=1.0`, or `4.0` when `negative_prompt` is supplied).
  - Exposes complete OpenAI-compatible & Google Drive-native APIs:
      * `POST /v1/images/generations` (Text-to-Image + RGBA + Drive output saving + LoRA)
      * `POST /v1/images/edits` & `POST /api/ct/edit` (1–4 `<image1>..<image4>` reference images via
                                       multipart/form-data OR JSON, preserving 4-channel RGBA alpha,
                                       supporting `size="auto"`, `drive://...`, `data:image/...`, URLs)
      * `POST /v1/chat/completions`   (Multimodal `image_url` I2I + Drive input/output rewriting)
      * `GET  /api/ct/drive-list`     (List images, outputs, and LoRA adapters on Google Drive)
      * `GET  /api/ct/info` & `GET /v1/models` & `GET /health`
  - CLI (`generate`/`t2i`, `edit`/`i2i`, `ls-drive`, `models`) reads reference images directly from
    mounted Google Drive without client-side re-upload and saves generated/edited images directly to
    Google Drive (`--save-drive` or `-o drive://...`) and/or local disk.

Usage:
  colabllm qwen-image up                                    # Default G4 (96 GB, BF16) + quick tunnel
  colabllm qwen-image up --domain                           # G4 + fixed Cloudflare domain
  colabllm qwen-image up --gpu H100 --domain --step-execution  # H100 (80 GB HBM3) + step batching
  colabllm qwen-image up --gpu A100 --quantization fp8-all  # A100 (40 GB+) online FP8 (~28 GB VRAM)
  colabllm qwen-image ls-drive                              # List images on Google Drive
  colabllm qwen-image generate "A ceramic teapot on a wooden table" --save-drive outputs/teapot.png
  colabllm qwen-image edit "Make the person in image 1 <image1> strike the pose from image 2 <image2> and wear the outfit from image 3 <image3>" \\
      --drive-image inputs/person.jpg inputs/pose.jpg inputs/outfit.jpg --save-drive outputs/swap.png -o swap.png
  colabllm qwen-image edit "Extract the complete outfit the person is wearing, laid flat on a transparent background" -r outfit.jpg --rgba -o garment_rgba.png
  colabllm qwen-image down
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cli_support import UsageError, http_timeout, nonnegative_seconds, resolve_endpoint, safe_urlopen  # noqa: E402
from colabsvc import (  # noqa: E402
    Service,
    State,
    die,
    log,
    main,
)

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
MAX_REFERENCE_IMAGES = 4
DIM_ALIGNMENT = 32
DRIVE_PREFIXES = ("drive://", "gdrive://", "MyDrive/", "/content/drive/")

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

SERVICE = Service(
    name="qwen-image",
    notebook="qwen-image.ipynb",
    service_port=8091,
    auth="builtin",
    public_port=8091,
    gpu="G4",
    high_mem=False,
    default_hostname="qwen-image",
    ready_timeout=3600,
    env={
        "CT_QWEN_IMAGE_MODEL": "Qwen/Qwen-Image-2.1",
        "CT_QWEN_IMAGE_ALIAS": "qwen-image-2.1",
        "CT_QWEN_IMAGE_QUANTIZATION": "none",
        "CT_QWEN_IMAGE_PREFIX_KV_DTYPE": "auto",
        "CT_QWEN_IMAGE_STEP_EXECUTION": "0",
        "CT_QWEN_IMAGE_VAE_TILING": "1",
        "CT_QWEN_IMAGE_MAX_NUM_SEQS": "",
        "CT_QWEN_IMAGE_LORA_PATH": "",
        "CT_QWEN_IMAGE_LORA_SCALE": "1.0",
    },
)

UP_ENV_FLAGS = {
    "--model": "CT_QWEN_IMAGE_MODEL",
    "--alias": "CT_QWEN_IMAGE_ALIAS",
    "--quantization": "CT_QWEN_IMAGE_QUANTIZATION",
    "--prefix-kv-dtype": "CT_QWEN_IMAGE_PREFIX_KV_DTYPE",
    "--max-num-seqs": "CT_QWEN_IMAGE_MAX_NUM_SEQS",
    "--lora-path": "CT_QWEN_IMAGE_LORA_PATH",
    "--lora-scale": "CT_QWEN_IMAGE_LORA_SCALE",
}
_BASE_ENV = dict(SERVICE.env)


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(
        State(SERVICE).load(),
        url_override,
        token_override,
        url_env="QWEN_IMAGE_URL",
        token_env="QWEN_IMAGE_API_TOKEN",
    )


def _http_json(
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    method: str | None = None,
    timeout: int | float | None = 300,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any], dict[str, str]]:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": BROWSER_UA,
        **(extra_headers or {}),
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method or ("POST" if data is not None else "GET"),
    )
    eff_timeout = http_timeout(timeout) if isinstance(timeout, (int, float)) else timeout
    try:
        with safe_urlopen(req, timeout=eff_timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            resp_hdrs = {k.lower(): v for k, v in resp.headers.items()}
            parsed = json.loads(raw) if raw.strip().startswith(("{", "[")) else {"raw": raw}
            return resp.status, parsed, resp_hdrs
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        resp_hdrs = {k.lower(): v for k, v in err.headers.items()} if err.headers else {}
        try:
            return err.code, json.loads(raw_err), resp_hdrs
        except Exception:
            return err.code, {"error": raw_err}, resp_hdrs


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
    if not spec:
        return False
    return spec.strip().startswith(DRIVE_PREFIXES)


def _normalize_drive_spec(spec: str) -> str:
    s = spec.strip()
    if s.startswith(("drive://", "gdrive://", "/content/drive/")):
        return s
    return f"drive://{s}"


def _encode_local_image_to_data_url(path: Path) -> str:
    raw = path.read_bytes()
    if not raw:
        raise UsageError(f"Local image file is empty: {path}")
    ext = path.suffix.lower()
    mime = MIME_BY_EXT.get(ext) or mimetypes.guess_type(str(path))[0] or "image/png"
    if not mime.startswith("image/"):
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _resolve_reference_image_spec(spec: str, *, force_drive: bool = False) -> tuple[str, str]:
    """Return (resolved_url_or_data_uri, source_label) for a local, URL, or Google Drive image."""
    s = (spec or "").strip()
    if not s:
        raise UsageError("Reference image path must not be empty")
    if force_drive or _is_drive_target(s):
        d = _normalize_drive_spec(s)
        return d, f"drive ({d})"
    if s.startswith(("http://", "https://", "data:image/")):
        label = "data-uri" if s.startswith("data:") else s
        return s, label
    local_p = Path(s).expanduser()
    if local_p.is_file():
        return _encode_local_image_to_data_url(local_p), f"local ({local_p})"
    d = _normalize_drive_spec(s)
    return d, f"drive ({d})"


def _flatten_groups(groups: list[list[str]] | None) -> list[str]:
    if not groups:
        return []
    return [item for grp in groups for item in grp if item and item.strip()]


class _OrderedImageAction(argparse.Action):
    """Preserve exact CLI argument order (<image1>..<image4>) across --image and --drive-image flags."""

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


def _collect_ordered_images(args: argparse.Namespace) -> list[tuple[str, bool]]:
    ordered = getattr(args, "ordered_images", None)
    if ordered:
        return list(ordered)
    out: list[tuple[str, bool]] = []
    for s in _flatten_groups(getattr(args, "image", None)):
        out.append((s, False))
    for s in _flatten_groups(getattr(args, "drive_image", None)):
        out.append((s, True))
    return out


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
        raise UsageError("Missing prompt: pass a prompt argument, `--prompt`, or `--prompt-file FILE`")
    return text


def _parse_dimensions(args: argparse.Namespace, *, default_size: str | None) -> tuple[int | None, int | None]:
    size_val = getattr(args, "size", None)
    width_val = getattr(args, "width", None)
    height_val = getattr(args, "height", None)
    if size_val:
        if str(size_val).strip().lower() == "auto":
            return None, None
        m = re.fullmatch(r"\s*(\d+)\s*[xX:]\s*(\d+)\s*", str(size_val))
        if not m:
            raise UsageError(f"Invalid --size '{size_val}': expected WIDTHxHEIGHT (e.g. 1024x1024) or 'auto'")
        w, h = int(m.group(1)), int(m.group(2))
    elif width_val is not None or height_val is not None:
        w = int(width_val if width_val is not None else 1024)
        h = int(height_val if height_val is not None else 1024)
    elif default_size:
        w_s, h_s = default_size.split("x", 1)
        w, h = int(w_s), int(h_s)
    else:
        return None, None

    if w < 64 or h < 64 or w > 4096 or h > 4096:
        raise UsageError(f"Invalid resolution {w}x{h}: width and height must be in [64, 4096]")
    if w % DIM_ALIGNMENT != 0 or h % DIM_ALIGNMENT != 0:
        w_aligned = max(64, (w // DIM_ALIGNMENT) * DIM_ALIGNMENT)
        h_aligned = max(64, (h // DIM_ALIGNMENT) * DIM_ALIGNMENT)
        log(f"aligning {w}x{h} -> {w_aligned}x{h_aligned} (Qwen-Image-2.1 32px patch grid)")
        w, h = w_aligned, h_aligned
    return w, h


def _resolve_output_targets(args: argparse.Namespace, *, prefix: str, ext: str = ".png") -> tuple[Path | None, str, bool]:
    """Return (local_output_path_or_None, drive_save_spec, drive_only_flag)."""
    raw_out = getattr(args, "output", None)
    raw_save_drive = getattr(args, "save_drive", None)
    explicit_drive_only = bool(getattr(args, "drive_only", False))

    drive_spec = ""
    local_out: Path | None = None

    if raw_out and _is_drive_target(raw_out):
        drive_spec = raw_out.strip()
        if raw_save_drive and raw_save_drive not in ("auto",):
            drive_spec = raw_save_drive.strip()
    else:
        if raw_save_drive:
            drive_spec = raw_save_drive.strip()
        if raw_out:
            local_out = Path(raw_out).expanduser()

    if explicit_drive_only:
        if not drive_spec:
            drive_spec = "auto"
        return None, drive_spec, True

    if drive_spec and local_out is None:
        return None, drive_spec, True

    if not drive_spec and local_out is None:
        stamp = f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}{ext}"
        local_out = Path(stamp)

    return local_out, drive_spec, False


def _write_local_images(images_b64: list[str], target: Path, prefix: str, default_ext: str = ".png") -> list[Path]:
    written: list[Path] = []
    is_dir = target.is_dir() or str(target).endswith("/") or (len(images_b64) > 1 and target.suffix == "")
    if is_dir:
        target.mkdir(parents=True, exist_ok=True)
        stamp = f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"
        for idx, b64 in enumerate(images_b64, 1):
            suffix = f"-{idx:02d}{default_ext}" if len(images_b64) > 1 else default_ext
            out_p = target / f"{stamp}{suffix}"
            out_p.write_bytes(base64.b64decode(b64))
            written.append(out_p)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        ext = target.suffix or default_ext
        stem = target.stem if target.suffix else target.name
        for idx, b64 in enumerate(images_b64, 1):
            out_p = target.parent / (f"{stem}-{idx:02d}{ext}" if len(images_b64) > 1 else f"{stem}{ext}")
            out_p.write_bytes(base64.b64decode(b64))
            written.append(out_p)
    return written


def _build_lora_dict(args: argparse.Namespace) -> dict[str, Any] | None:
    lpath = getattr(args, "lora", None)
    lname = getattr(args, "lora_name", None)
    lscale = getattr(args, "lora_scale", 1.0)
    if lname and not lpath:
        raise UsageError("--lora-name requires --lora")
    if not lpath:
        return None
    spec = lpath.strip()
    return {
        "path": spec,
        "name": lname or Path(spec.removeprefix("drive://").removeprefix("gdrive://")).stem,
        "scale": float(lscale),
    }


def _apply_common_sampling(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.seed is not None:
        payload["seed"] = int(args.seed)
    steps = args.steps if args.steps is not None else 40
    payload["num_inference_steps"] = int(steps)

    neg = (args.negative_prompt or "").strip()
    if neg:
        payload["negative_prompt"] = neg

    if args.true_cfg_scale is not None:
        payload["true_cfg_scale"] = float(args.true_cfg_scale)
    else:
        payload["true_cfg_scale"] = 4.0 if neg else 1.0

    if getattr(args, "rgba", False):
        payload["color_format"] = "RGBA"
        payload["background"] = "transparent"
        payload["output_format"] = "png"
    else:
        if getattr(args, "color_format", None):
            payload["color_format"] = str(args.color_format).upper()
        if getattr(args, "background", None):
            payload["background"] = str(args.background)
        if getattr(args, "output_format", None):
            payload["output_format"] = str(args.output_format).lower()

    lora_cfg = _build_lora_dict(args)
    if lora_cfg:
        payload["lora"] = lora_cfg


def cmd_generate(args: argparse.Namespace) -> None:
    prompt = _resolve_prompt(args)
    w, h = _parse_dimensions(args, default_size="1024x1024")
    out_fmt = "png" if getattr(args, "rgba", False) else (getattr(args, "output_format", None) or "png")
    ext = ".jpg" if out_fmt in ("jpg", "jpeg") else f".{out_fmt}"
    local_out, drive_spec, drive_only = _resolve_output_targets(args, prefix="qwen-t2i", ext=ext)
    base_url, token = _get_endpoint(args.url, args.api_token)

    payload: dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "size": f"{w}x{h}" if (w and h) else "1024x1024",
        "n": int(args.num_images),
        "response_format": "b64_json",
    }
    _apply_common_sampling(payload, args)
    if drive_spec:
        payload["save_to_drive"] = drive_spec
    if drive_only:
        payload["drive_only"] = True

    t0 = time.monotonic()
    code, res, hdrs = _http_json(
        f"{base_url}/v1/images/generations",
        token,
        payload=payload,
        timeout=args.timeout,
    )
    elapsed = time.monotonic() - t0
    if code != 200:
        die(_format_error(code, res))

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    data_items = res.get("data") or []
    b64_list = [it["b64_json"] for it in data_items if isinstance(it, dict) and it.get("b64_json")]
    local_written: list[Path] = []
    if local_out is not None and b64_list:
        local_written = _write_local_images(b64_list, local_out, prefix="qwen-t2i", default_ext=ext)

    saved_drive = res.get("saved_drive_paths") or (
        [hdrs["x-saved-drive-path"]] if hdrs.get("x-saved-drive-path") else []
    )
    rel_drive = res.get("rel_drive_paths") or saved_drive

    for lp in local_written:
        log(f"saved local image: {lp} ({lp.stat().st_size / 1024:.1f} KB, {elapsed:.2f}s)")
    for dp in rel_drive:
        log(f"saved Google Drive image: {dp} ({elapsed:.2f}s)")


def cmd_edit(args: argparse.Namespace) -> None:
    prompt = _resolve_prompt(args)
    ordered_specs = _collect_ordered_images(args)
    total_refs = len(ordered_specs)
    if total_refs == 0:
        raise UsageError("At least 1 reference image is required: pass `--image PATH` or `--drive-image DRIVE_PATH`")
    if total_refs > MAX_REFERENCE_IMAGES:
        raise UsageError(
            f"Qwen-Image-2.1 supports at most {MAX_REFERENCE_IMAGES} reference images (<image1>..<image4>), "
            f"got {total_refs}"
        )

    resolved_urls: list[str] = []
    source_labels: list[str] = []
    for idx, (spec, force_drive) in enumerate(ordered_specs, 1):
        u, lbl = _resolve_reference_image_spec(spec, force_drive=force_drive)
        resolved_urls.append(u)
        source_labels.append(f"<image{idx}>={lbl}")

    w, h = _parse_dimensions(args, default_size=None)
    out_fmt = "png" if getattr(args, "rgba", False) else (getattr(args, "output_format", None) or "png")
    ext = ".jpg" if out_fmt in ("jpg", "jpeg") else f".{out_fmt}"
    local_out, drive_spec, drive_only = _resolve_output_targets(args, prefix="qwen-edit", ext=ext)
    base_url, token = _get_endpoint(args.url, args.api_token)

    if args.endpoint == "chat":
        payload: dict[str, Any] = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image_url", "image_url": {"url": u}} for u in resolved_urls],
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        if w is not None and h is not None:
            payload["width"] = w
            payload["height"] = h
        target_path = "/v1/chat/completions"
    else:
        payload = {
            "model": args.model,
            "prompt": prompt,
            "images": resolved_urls,
            "size": f"{w}x{h}" if (w is not None and h is not None) else "auto",
            "n": int(args.num_images),
        }
        target_path = "/v1/images/edits" if args.endpoint == "edits" else "/api/ct/edit"

    _apply_common_sampling(payload, args)
    if drive_spec:
        payload["save_to_drive"] = drive_spec
    if drive_only:
        payload["drive_only"] = True

    if not args.json:
        log(f"editing with {len(resolved_urls)} reference image(s): {', '.join(source_labels)}")

    t0 = time.monotonic()
    code, res, hdrs = _http_json(
        f"{base_url}{target_path}",
        token,
        payload=payload,
        timeout=args.timeout,
    )
    elapsed = time.monotonic() - t0
    if code != 200:
        die(_format_error(code, res))

    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    b64_list: list[str] = []
    if args.endpoint == "chat":
        for ch in res.get("choices") or []:
            content = (ch.get("message") or {}).get("content") or []
            for part in content if isinstance(content, list) else []:
                u = ((part.get("image_url") or {}).get("url") or "") if isinstance(part, dict) else ""
                if u.startswith("data:image/") and "," in u:
                    b64_list.append(u.split(",", 1)[1])
    else:
        for it in res.get("data") or []:
            if isinstance(it, dict) and it.get("b64_json"):
                b64_list.append(it["b64_json"])

    local_written: list[Path] = []
    if local_out is not None and b64_list:
        local_written = _write_local_images(b64_list, local_out, prefix="qwen-edit", default_ext=ext)

    saved_drive = res.get("saved_drive_paths") or (
        [hdrs["x-saved-drive-path"]] if hdrs.get("x-saved-drive-path") else []
    )
    rel_drive = res.get("rel_drive_paths") or saved_drive

    for lp in local_written:
        log(f"saved local edited image: {lp} ({lp.stat().st_size / 1024:.1f} KB, {elapsed:.2f}s)")
    for dp in rel_drive:
        log(f"saved Google Drive edited image: {dp} ({elapsed:.2f}s)")


def cmd_ls_drive(args: argparse.Namespace) -> None:
    base_url, token = _get_endpoint(args.url, args.api_token)
    kind = args.kind
    if args.loras:
        kind = "loras"
    elif args.outputs:
        kind = "outputs"
    elif args.all:
        kind = "all"

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
    combined = {"info": info_res, "models": [m.get("id") for m in (mod_res.get("data") or []) if isinstance(m, dict)]}
    if args.json:
        print(json.dumps(combined, indent=2, ensure_ascii=False))
        return
    print(f"Service:            {info_res.get('service', 'qwen-image')}")
    print(f"Model repo:         {info_res.get('model_repo')}")
    print(f"Served names:       {', '.join(combined['models'] or info_res.get('served_models') or [])}")
    print(
        f"Engine config:      quant={info_res.get('quantization')} "
        f"prefix_kv={info_res.get('prefix_kv_cache_dtype')} "
        f"step_execution={info_res.get('step_execution')} "
        f"max_num_seqs={info_res.get('max_num_seqs')} "
        f"vae_tiling={info_res.get('vae_use_tiling')}"
    )
    print(
        f"Default sampling:   steps={info_res.get('default_steps')} "
        f"true_cfg_scale={info_res.get('default_true_cfg_scale')} "
        f"(with neg prompt: {info_res.get('default_true_cfg_scale_with_negative_prompt')}) "
        f"rgba_vae={info_res.get('rgba_vae')} max_refs={info_res.get('max_reference_images')}"
    )
    print(f"Preloaded LoRA:     {info_res.get('preloaded_lora') or 'none'}")
    print(f"Drive inputs dir:   {info_res.get('drive_inputs_dir')}")
    print(f"Drive outputs dir:  {info_res.get('drive_outputs_dir')}")
    print(f"Drive loras dir:    {info_res.get('drive_loras_dir')}")


def _add_endpoint_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--url", help="Override endpoint URL (or set QWEN_IMAGE_URL)")
    p.add_argument("--api-token", help="Override Bearer API token (or set QWEN_IMAGE_API_TOKEN)")


def _add_generation_common_flags(p: argparse.ArgumentParser, *, is_edit: bool) -> None:
    p.add_argument("prompt", nargs="?", help="Generation / editing prompt (use <image1>..<image4> to bind reference images)")
    p.add_argument("-p", "--prompt", dest="prompt_opt", help="Prompt text (alternative to positional argument)")
    p.add_argument("--prompt-file", help="Read prompt text from a UTF-8 file")
    size_help = (
        "Output resolution WIDTHxHEIGHT (32px aligned) or 'auto' to preserve reference image aspect ratio (default: auto)"
        if is_edit
        else "Output resolution WIDTHxHEIGHT (32px aligned, default: 1024x1024)"
    )
    p.add_argument("--size", help=size_help)
    p.add_argument("--width", type=int, help="Output width in pixels (aligned to multiple of 32)")
    p.add_argument("--height", type=int, help="Output height in pixels (aligned to multiple of 32)")
    p.add_argument("--steps", type=int, default=40, help="Sampling steps (Qwen-Image-2.1 default: 40)")
    p.add_argument(
        "--true-cfg-scale", "--cfg-scale",
        dest="true_cfg_scale",
        type=float,
        help="True CFG scale (default: 1.0 when --negative-prompt is empty, 4.0 when --negative-prompt is given)",
    )
    p.add_argument("--negative-prompt", default="", help="Negative prompt (engages True CFG when non-empty)")
    p.add_argument("--rgba", action="store_true", help="Preserve / generate 4-channel RGBA PNG transparency")
    p.add_argument("--color-format", choices=("RGBA", "RGB"), help="Output color channels: RGBA (4ch default) or RGB (3ch)")
    p.add_argument("--background", choices=("auto", "transparent", "opaque"), help="OpenAI Images background hint")
    p.add_argument("--output-format", choices=("png", "jpeg", "webp"), help="Output image format (default: png)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--lora", dest="lora", help="LoRA adapter on Google Drive (e.g. drive://loras/style.safetensors) or staged server path")
    p.add_argument("--lora-name", help="Optional adapter label (defaults to LoRA filename stem)")
    p.add_argument("--lora-scale", type=float, default=1.0, help="Per-request LoRA scale (default: 1.0)")
    p.add_argument(
        "-o", "--output",
        help="Local output path (file or directory) OR Google Drive path (drive://outputs/out.png)",
    )
    p.add_argument(
        "--save-drive",
        nargs="?",
        const="auto",
        help="Save output directly on mounted Google Drive (optional path, default: colab-llm/qwen-image/outputs/)",
    )
    p.add_argument(
        "--drive-only", "--no-local",
        dest="drive_only",
        action="store_true",
        help="Save only on Google Drive and skip downloading image bytes locally",
    )
    p.add_argument(
        "--model",
        default="Qwen/Qwen-Image-2.1",
        help="Served model name (default: Qwen/Qwen-Image-2.1)",
    )
    p.add_argument(
        "--timeout",
        type=nonnegative_seconds,
        default=300,
        help="HTTP request timeout in seconds; 0 = unlimited (default: 300)",
    )
    p.add_argument("--json", action="store_true", help="Print raw JSON response")
    _add_endpoint_flags(p)


def _build_custom_parser(_parser: argparse.ArgumentParser, sub: Any) -> None:
    for cmd_name, help_msg in (
        ("generate", "Generate image(s) from text (/v1/images/generations) with RGBA & Google Drive output"),
        ("t2i", "Alias for 'generate' (Text-to-Image)"),
    ):
        gp = sub.add_parser(cmd_name, help=help_msg)
        gp.add_argument("-n", "--num-images", type=int, default=1, help="Number of images to generate (default: 1)")
        _add_generation_common_flags(gp, is_edit=False)

    for cmd_name, help_msg in (
        ("edit", "Edit or combine 1-4 local/URL/Google Drive reference images (<image1>..<image4>, RGBA preserved)"),
        ("i2i", "Alias for 'edit' (Image-to-Image / multi-reference editing)"),
    ):
        ep = sub.add_parser(cmd_name, help=help_msg)
        ep.add_argument("-n", "--num-images", type=int, default=1, help="Number of edited variations to generate (default: 1)")
        ep.add_argument(
            "-r", "--image", "--ref",
            dest="ordered_images",
            action=_OrderedImageAction,
            nargs="+",
            help="Reference image(s) in order (1-4, bound as <image1>..<image4>): local file, https:// URL, or drive:// path",
        )
        ep.add_argument(
            "--drive-image",
            dest="ordered_images",
            action=_OrderedImageAction,
            nargs="+",
            help="Reference image(s) resolved strictly on mounted Google Drive (zero client upload)",
        )
        ep.add_argument(
            "--endpoint",
            choices=("edits", "edit", "chat"),
            default="edits",
            help="Target API route: 'edits' (/v1/images/edits, preserves 4ch RGBA), 'edit' (/api/ct/edit), or 'chat' (/v1/chat/completions)",
        )
        _add_generation_common_flags(ep, is_edit=True)

    lsd = sub.add_parser("ls-drive", help="List images, generated outputs, or LoRA adapters on mounted Google Drive")
    lsd.add_argument("path", nargs="?", default="", help="Drive folder or glob (e.g. inputs/*.png, MyDrive/photos)")
    lsd.add_argument("--kind", choices=("images", "outputs", "loras", "all"), default="images", help="File category to list (default: images)")
    lsd.add_argument("--loras", action="store_true", help="Shortcut for --kind loras (list .safetensors & PEFT dirs)")
    lsd.add_argument("--outputs", action="store_true", help="Shortcut for --kind outputs (list generated images)")
    lsd.add_argument("--all", action="store_true", help="Shortcut for --kind all")
    lsd.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    lsd.add_argument("--json", action="store_true", help="Print raw JSON response")
    _add_endpoint_flags(lsd)

    mp = sub.add_parser("models", help="Show active Qwen-Image-2.1 server config, sampling defaults & Google Drive paths")
    mp.add_argument("--json", action="store_true", help="Print raw JSON response")
    _add_endpoint_flags(mp)


def configure_cli(parser: argparse.ArgumentParser, commands: Any, up: argparse.ArgumentParser) -> None:
    _build_custom_parser(parser, commands)
    up.add_argument(
        "--model",
        dest="up_CT_QWEN_IMAGE_MODEL",
        help="Hugging Face repo (default: Qwen/Qwen-Image-2.1)",
    )
    up.add_argument(
        "--alias",
        dest="up_CT_QWEN_IMAGE_ALIAS",
        help="Served model alias (default: qwen-image-2.1)",
    )
    up.add_argument(
        "--quantization",
        dest="up_CT_QWEN_IMAGE_QUANTIZATION",
        choices=("none", "fp8", "fp8-text", "fp8-all"),
        help=(
            "Online FP8 quantization: none (BF16 default, ~34-40 GB VRAM), "
            "fp8 (DiT FP8 keeping img_mlp BF16, ~32 GB), fp8-text (Qwen3-VL-8B FP8, ~28 GB), "
            "fp8-all (DiT + Qwen3-VL-8B FP8, ~28 GB, recommended for 40 GB A100)"
        ),
    )
    up.add_argument(
        "--prefix-kv-dtype",
        dest="up_CT_QWEN_IMAGE_PREFIX_KV_DTYPE",
        choices=("auto", "fp8_v", "fp8"),
        help="Cross-step prefix KV cache precision: auto (BF16 + CUDA graphs), fp8_v (V-only FP8), or fp8 (K+V FP8)",
    )
    up.add_argument(
        "--step-execution",
        dest="up_step_execution",
        action="store_true",
        help="Enable step-level continuous batching across concurrent requests (--step-execution, default max-num-seqs=8)",
    )
    up.add_argument(
        "--max-num-seqs",
        dest="up_CT_QWEN_IMAGE_MAX_NUM_SEQS",
        type=int,
        help="Max concurrent sequences in the diffusion engine (default: 4 in request mode, 8 with --step-execution)",
    )
    up.add_argument(
        "--no-vae-tiling",
        dest="up_no_vae_tiling",
        action="store_true",
        help="Disable default --vae-use-tiling (512px tile / 384px stride adaptive OOM protection)",
    )
    up.add_argument(
        "--lora-path",
        dest="up_CT_QWEN_IMAGE_LORA_PATH",
        help="Preload a LoRA adapter at server startup (supports drive://loras/adapter.safetensors)",
    )
    up.add_argument(
        "--lora-scale",
        dest="up_CT_QWEN_IMAGE_LORA_SCALE",
        help="Default scale for preloaded LoRA adapter (default: 1.0)",
    )


def prepare_up(args: argparse.Namespace) -> None:
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    for _flag, env_key in UP_ENV_FLAGS.items():
        val = getattr(args, "up_" + env_key, None)
        if val is not None:
            SERVICE.env[env_key] = str(val)
    if getattr(args, "up_step_execution", False):
        SERVICE.env["CT_QWEN_IMAGE_STEP_EXECUTION"] = "1"
    if getattr(args, "up_no_vae_tiling", False):
        SERVICE.env["CT_QWEN_IMAGE_VAE_TILING"] = "0"


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
                "ls-drive": cmd_ls_drive,
                "models": cmd_models,
            },
        )
    )
