"""Authenticated ASGI gateway, async job queue, binary upload/download & Drive I/O for MiniMax-H3-Image.

Sits on `127.0.0.1:<CT_SERVICE_PORT>` (default `8092`) in front of `sglang serve` on loopback
`127.0.0.1:<CT_H3_SGLANG_PORT>` (default `30000`).

Network-friendly design:
  1. Async job submission & tiny JSON polling (`POST /api/ct/jobs`, `GET /api/ct/jobs/<id>`) so CLI
     calls never hold long idle HTTP connections over Cloudflare Tunnel / FRP / proxies.
  2. Content-addressed raw binary reference upload (`HEAD/GET /api/ct/uploads/<sha256>` +
     `POST /api/ct/uploads`) avoiding +33% Base64 bloat and deduplicating repeated edits to 0 bytes.
  3. Zero-Base64 loopback file handoff with SGLang (`response_format="url"` + `output_compression=0`
     into `/content/minimax-h3-image/sglang-outputs`), plus server-side PNG compression or WebP/JPEG
     transcoding and resumable `Range` binary streaming (`GET /api/ct/jobs/<id>/content`).
  4. Zero-upload / zero-download Google Drive I/O (`drive://...` inputs & `save_to_drive` outputs).
  5. Full OpenAI `/v1/images/generations`, `/v1/images/edits`, `/v1/models`, and `/health` compatibility,
     all protected fail-closed by `verify_bearer_auth`.

Imports cleanly with only the Python standard library so local unit tests can exercise all security,
normalization, job queue, upload deduplication, and binary range-streaming paths without GPU deps.
"""
from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from email.parser import BytesParser
from email.policy import default as email_default_policy
import hashlib
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import secrets
import shutil
import struct
import time
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request

from colab_asgi_common import (
    allowed_drive_roots,
    build_multipart_form,
    rel_to_mydrive,
    resolve_drive_path,
    sanitize_header_param,
    tail_service_log,
    verify_bearer_auth,
)


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


SERVICE_NAME = os.environ.get("CT_SERVICE") or "minimax-h3-image"
SERVICE_PORT = _env_int("CT_SERVICE_PORT", 8092)
SGLANG_PORT = _env_int("CT_H3_SGLANG_PORT", 30000)
SGLANG_BASE_URL = (os.environ.get("CT_H3_SGLANG_URL") or f"http://127.0.0.1:{SGLANG_PORT}").rstrip("/")

API_TOKEN = os.environ.get("CT_API_TOKEN", "")
MODEL_REPO = (os.environ.get("CT_H3_IMAGE_MODEL") or "MiniMaxAI/MiniMax-H3-Image").strip()
MODEL_ALIAS = (os.environ.get("CT_H3_IMAGE_ALIAS") or "minimax-h3-image").strip()
QUANTIZATION = (os.environ.get("CT_H3_IMAGE_QUANTIZATION") or "fp8-all").strip()
PLACEMENT = (os.environ.get("CT_H3_IMAGE_PLACEMENT") or "auto").strip()
WARMUP_MODE = (os.environ.get("CT_H3_IMAGE_WARMUP") or "server").strip()

DRIVE_ROOT = Path("/content/drive")
MYDRIVE_ROOT = DRIVE_ROOT / "MyDrive"
DATA_DIR = Path(os.environ.get("CT_DATA_DIR") or f"/content/drive/MyDrive/colab-llm/{SERVICE_NAME}")
DRIVE_INPUTS_DIR = DATA_DIR / "inputs"
DRIVE_OUTPUTS_DIR = DATA_DIR / "outputs"

LOCAL_WORK_DIR = Path(os.environ.get("CT_H3_LOCAL_WORK_DIR") or "/content/minimax-h3-image")
SGLANG_OUTPUT_DIR = LOCAL_WORK_DIR / "sglang-outputs"
SGLANG_INPUT_DIR = LOCAL_WORK_DIR / "sglang-inputs"
UPLOADS_DIR = LOCAL_WORK_DIR / "uploads"
JOBS_DIR = LOCAL_WORK_DIR / "jobs"
LOG_FILE = Path("/content/ct/service.log")
SGLANG_LOG_FILE = Path("/content/ct/sglang.log")

DIM_ALIGNMENT = 32
MIN_DIM = 64
MAX_DIM = 8192
FIXED_PDD_STEPS = 6
DEFAULT_MIN_GENERATION_AREA = 2048 * 2048  # 4,194,304
MIN_GENERATION_AREA_BOUNDS = (1024, 4096 * 4096)  # [1024, 16,777,216]
MAX_REFERENCE_IMAGES = 16
MAX_IMAGES_PER_JOB = 8
MAX_JSON_BODY_BYTES = 96 * 1024 * 1024
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_RETAINED_JOBS = 48
MAX_UPLOADS_CACHE_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB
STREAM_CHUNK_BYTES = 65536

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif")
MIME_BY_FORMAT = {
    "png": "image/png",
    "webp": "image/webp",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
}
EXT_BY_FORMAT = {
    "png": ".png",
    "webp": ".webp",
    "jpeg": ".jpg",
    "jpg": ".jpg",
}

# Official MiniMax-H3-Image native 4K (16 MP) resolutions and aligned 2K (4 MP) / 1K (1 MP) counterparts.
ASPECT_PRESETS: dict[str, dict[str, tuple[int, int]]] = {
    "4k": {
        "1:1": (4096, 4096),
        "4:3": (4736, 3552),
        "3:4": (3552, 4736),
        "3:2": (5024, 3360),
        "2:3": (3360, 5024),
        "16:9": (5472, 3072),
        "9:16": (3072, 5472),
        "21:9": (6272, 2688),
        "9:21": (2688, 6272),
    },
    "2k": {
        "1:1": (2048, 2048),
        "4:3": (2368, 1760),
        "3:4": (1760, 2368),
        "3:2": (2496, 1664),
        "2:3": (1664, 2496),
        "16:9": (2752, 1536),
        "9:16": (1536, 2752),
        "21:9": (3136, 1344),
        "9:21": (1344, 3136),
    },
    "1k": {
        "1:1": (1024, 1024),
        "4:3": (1184, 896),
        "3:4": (896, 1184),
        "3:2": (1248, 832),
        "2:3": (832, 1248),
        "16:9": (1376, 768),
        "9:16": (768, 1376),
        "21:9": (1568, 672),
        "9:21": (672, 1568),
    },
}
TIER_TARGET_AREAS = {
    "4k": 4096 * 4096,
    "2k": 2048 * 2048,
    "1k": 1024 * 1024,
}

MODEL_ALIASES = {
    "",
    "minimax-h3-image",
    "h3-image",
    "minimaxai/minimax-h3-image",
    "gpt-image-1",
}

UPLOAD_ID_RE = re.compile(r"^[0-9a-f]{64}$")
JOB_ID_RE = re.compile(r"^job_[0-9a-f]{16,32}$")
QWEN_IMAGE_TAG_RE = re.compile(r"<image\s*(\d+)>", re.IGNORECASE)

CORS_HEADERS: list[tuple[bytes, bytes]] = [
    (b"access-control-allow-origin", b"*"),
    (b"access-control-allow-methods", b"GET, HEAD, POST, DELETE, OPTIONS"),
    (b"access-control-allow-headers", b"Authorization, Content-Type, X-API-Key, X-Filename, Range"),
    (
        b"access-control-expose-headers",
        b"Content-Length, Content-Range, Accept-Ranges, ETag, X-Job-Id, "
        b"X-Image-Width, X-Image-Height, X-Image-Format, X-Saved-Drive-Path",
    ),
]


def _inject_cors_headers(headers: list[tuple[bytes, bytes]]) -> list[tuple[bytes, bytes]]:
    existing = {k.lower() for k, _ in headers}
    merged = list(headers)
    for k, v in CORS_HEADERS:
        if k.lower() not in existing:
            merged.append((k, v))
    return merged


def normalize_model_name(raw_model: str | None) -> str:
    cleaned = (raw_model or "").strip()
    if cleaned.lower() in MODEL_ALIASES or cleaned.lower() == MODEL_ALIAS.lower():
        return MODEL_REPO
    return cleaned


def normalize_edit_prompt_tags(prompt: str) -> str:
    """Translate `<image1>`/`<image 1>` tags to MiniMax-H3-Image `<Picture 1>` when no `<Picture` tag is present."""
    if not prompt or "<picture" in prompt.lower():
        return prompt
    return QWEN_IMAGE_TAG_RE.sub(lambda m: f"<Picture {int(m.group(1))}>", prompt)


def normalize_tier(raw_tier: str | None, *, default: str = "4k") -> str:
    t = (raw_tier or default).strip().lower()
    aliases = {
        "4k": "4k",
        "16mp": "4k",
        "native": "4k",
        "2k": "2k",
        "4mp": "2k",
        "1k": "1k",
        "1mp": "1k",
        "preview": "1k",
    }
    if t not in aliases:
        raise ValueError(f"Unsupported resolution tier '{raw_tier}': expected 4k (16mp), 2k (4mp), or 1k (1mp)")
    return aliases[t]


def align_dim(value: int) -> int:
    aligned = int(round(value / DIM_ALIGNMENT)) * DIM_ALIGNMENT
    return max(MIN_DIM, min(MAX_DIM, aligned))


def resolve_dimensions(
    *,
    size: str | None = None,
    width: int | None = None,
    height: int | None = None,
    aspect: str | None = None,
    tier: str | None = None,
    is_edit: bool = False,
    ref_dimensions: tuple[int, int] | None = None,
) -> tuple[int, int]:
    """Resolve output `(width, height)` aligned to 32 pixels."""
    eff_tier = normalize_tier(tier, default="4k")
    size_str = (size or "").strip()

    if size_str and size_str.lower() != "auto":
        m = re.fullmatch(r"(\d+)\s*[xX:]\s*(\d+)", size_str)
        if not m:
            raise ValueError(f"Invalid size '{size}': expected WIDTHxHEIGHT (e.g. 4096x4096, 2048x2048) or 'auto'")
        w, h = int(m.group(1)), int(m.group(2))
    elif width is not None or height is not None:
        if width is None or height is None:
            raise ValueError("Both width and height must be provided together when size is omitted")
        if isinstance(width, bool) or isinstance(height, bool):
            raise ValueError("width and height must be integers")
        w, h = int(width), int(height)
    elif aspect:
        asp = aspect.strip()
        preset_map = ASPECT_PRESETS[eff_tier]
        if asp not in preset_map:
            valid = ", ".join(preset_map.keys())
            raise ValueError(f"Unsupported aspect ratio '{aspect}': choose from {valid}")
        return preset_map[asp]
    elif is_edit and ref_dimensions and ref_dimensions[0] > 0 and ref_dimensions[1] > 0:
        rw, rh = ref_dimensions
        target_area = TIER_TARGET_AREAS[eff_tier]
        ratio = rw / rh
        # Snap to exact official preset if within 3% aspect ratio for native 4K/2K grid quality.
        best_preset: tuple[int, int] | None = None
        best_rel_diff = 0.03
        for pw, ph in ASPECT_PRESETS[eff_tier].values():
            rel_diff = abs((pw / ph) - ratio) / max(ratio, 1e-6)
            if rel_diff < best_rel_diff:
                best_rel_diff = rel_diff
                best_preset = (pw, ph)
        if best_preset is not None:
            return best_preset
        w = align_dim(int(math.sqrt(target_area * ratio)))
        h = align_dim(int(math.sqrt(target_area / ratio)))
        return w, h
    else:
        return ASPECT_PRESETS[eff_tier]["1:1"]

    if w < MIN_DIM or h < MIN_DIM or w > MAX_DIM or h > MAX_DIM:
        raise ValueError(f"Resolution {w}x{h} out of range [{MIN_DIM}, {MAX_DIM}]")
    w = align_dim(w)
    h = align_dim(h)
    return w, h


def validate_min_generation_area(
    value: Any,
    *,
    width: int,
    height: int,
    fast_preview: bool = False,
) -> int | None:
    if value is None:
        if fast_preview and (width * height) < DEFAULT_MIN_GENERATION_AREA:
            return max(MIN_GENERATION_AREA_BOUNDS[0], min(MIN_GENERATION_AREA_BOUNDS[1], width * height))
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("min_generation_area must be an integer between 1024 and 16777216")
    lo, hi = MIN_GENERATION_AREA_BOUNDS
    if value < lo or value > hi:
        raise ValueError(f"min_generation_area={value} out of allowed range [{lo}, {hi}]")
    return value


def validate_h3_sampling_contract(payload: dict[str, Any]) -> None:
    """Enforce MiniMax-H3-Image's strict 6-step PDD / single-positive-branch contract."""
    steps = payload.get("num_inference_steps", payload.get("steps"))
    if steps is not None and steps != "":
        if isinstance(steps, bool) or not isinstance(steps, int) or steps != FIXED_PDD_STEPS:
            raise ValueError(
                f"MiniMax-H3-Image uses a fused {FIXED_PDD_STEPS}-step PDD schedule; "
                f"num_inference_steps must be {FIXED_PDD_STEPS} or omitted (got {steps!r})"
            )

    guidance = payload.get("guidance_scale")
    if guidance is not None and guidance != "":
        if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or float(guidance) != 1.0:
            raise ValueError(
                f"MiniMax-H3-Image is CFG-distilled (single positive branch); "
                f"guidance_scale must be 1 or omitted (got {guidance!r})"
            )

    neg = payload.get("negative_prompt")
    if neg is not None and str(neg).strip() != "":
        raise ValueError("MiniMax-H3-Image does not support negative_prompt (single positive branch)")

    if payload.get("lora") or payload.get("lora_path"):
        raise ValueError("MiniMax-H3-Image SGLang pipeline does not support LoRA adapters")


def inspect_image_bytes(data: bytes) -> tuple[int, int, str]:
    """Return `(width, height, format_lower)` using Pillow (with EXIF transpose) or stdlib header parser."""
    if not data:
        raise ValueError("Empty image data")
    try:
        from PIL import Image, ImageOps  # type: ignore[import-not-found]

        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "PNG").lower()
            im_oriented = ImageOps.exif_transpose(im)
            w, h = im_oriented.size
            if fmt == "jpg":
                fmt = "jpeg"
            return int(w), int(h), fmt
    except ImportError:
        pass

    # Pure-stdlib fallback for PNG / WebP / JPEG headers in unit tests without Pillow.
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        w, h = struct.unpack(">II", data[16:24])
        return int(w), int(h), "png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP" and len(data) >= 30:
        chunk = data[12:16]
        if chunk == b"VP8X" and len(data) >= 30:
            w = 1 + int.from_bytes(data[24:27], "little")
            h = 1 + int.from_bytes(data[27:30], "little")
            return w, h, "webp"
    if data.startswith(b"\xff\xd8"):
        idx = 2
        n = len(data)
        while idx + 4 < n:
            if data[idx] != 0xFF:
                idx += 1
                continue
            marker = data[idx + 1]
            while marker == 0xFF and idx + 2 < n:
                idx += 1
                marker = data[idx + 1]
            if marker in (0xD8, 0xD9):
                idx += 2
                continue
            if idx + 4 > n:
                break
            seg_len = struct.unpack(">H", data[idx + 2 : idx + 4])[0]
            if marker in (0xC0, 0xC1, 0xC2) and idx + 9 <= n:
                h, w = struct.unpack(">HH", data[idx + 5 : idx + 9])
                return int(w), int(h), "jpeg"
            idx += 2 + max(2, seg_len)
    raise ValueError("Unsupported or corrupted image data")


def transcode_image_bytes(
    src_bytes: bytes,
    *,
    output_format: str = "png",
    output_quality: int = 92,
    png_compress_level: int = 6,
) -> tuple[bytes, int, int, str]:
    """Encode/compress image bytes into `png`, `webp`, or `jpeg`, returning `(bytes, width, height, fmt)`."""
    fmt = (output_format or "png").strip().lower()
    if fmt == "jpg":
        fmt = "jpeg"
    if fmt not in ("png", "webp", "jpeg"):
        raise ValueError(f"Unsupported output_format '{output_format}': expected png, webp, or jpeg")
    quality = max(1, min(100, int(output_quality)))
    compress_level = max(0, min(9, int(png_compress_level)))

    try:
        from PIL import Image  # type: ignore[import-not-found]

        with Image.open(io.BytesIO(src_bytes)) as im:
            w, h = im.size
            src_fmt = (im.format or "").lower()
            # Fast path: already PNG and caller doesn't need format change or recompression
            if fmt == "png" and src_fmt == "png" and compress_level == 0:
                return src_bytes, int(w), int(h), "png"
            buf = io.BytesIO()
            if fmt == "png":
                im.save(buf, format="PNG", compress_level=compress_level)
            elif fmt == "webp":
                lossless = quality >= 100
                im.save(buf, format="WEBP", quality=quality, lossless=lossless, method=4)
            else:
                rgb = im.convert("RGB") if im.mode not in ("RGB", "L") else im
                rgb.save(buf, format="JPEG", quality=quality, optimize=True)
            return buf.getvalue(), int(w), int(h), fmt
    except ImportError:
        w, h, detected = inspect_image_bytes(src_bytes)
        if fmt != detected and not (fmt == "jpeg" and detected == "jpg"):
            raise ValueError(f"Pillow is required to transcode {detected} -> {fmt}")
        return src_bytes, w, h, fmt


def save_output_image_to_drive(
    image_bytes: bytes,
    spec: str,
    *,
    index: int = 0,
    total: int = 1,
    fmt: str = "png",
    prefix: str = "h3",
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    mydrive_root: Path | None = None,
    drive_root: Path | None = None,
) -> tuple[str, str]:
    """Persist generated image bytes strictly inside allowed Drive/data roots."""
    eff_data = data_dir or DATA_DIR
    eff_out = output_dir or DRIVE_OUTPUTS_DIR
    eff_mydrive = mydrive_root or MYDRIVE_ROOT
    eff_drive = drive_root or DRIVE_ROOT
    eff_out.mkdir(parents=True, exist_ok=True)

    ext = EXT_BY_FORMAT.get(fmt.lower(), ".png")
    stamp = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    clean_spec = (spec or "").strip()

    if clean_spec and clean_spec.lower() not in ("auto", "true", "1"):
        dest = resolve_drive_path(
            clean_spec,
            data_dir=eff_data,
            default_subdir="outputs",
            mydrive_root=eff_mydrive,
            drive_root=eff_drive,
        )
        if dest.is_dir() or clean_spec.endswith("/"):
            dest.mkdir(parents=True, exist_ok=True)
            suffix_idx = f"-{index + 1:02d}" if total > 1 else ""
            dest = dest / f"{prefix}-{stamp}{suffix_idx}{ext}"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.suffix:
                dest = dest.with_suffix(ext)
            if total > 1:
                dest = dest.with_name(f"{dest.stem}-{index + 1:02d}{dest.suffix}")
    else:
        suffix_idx = f"-{index + 1:02d}" if total > 1 else ""
        dest = (eff_out / f"{prefix}-{stamp}{suffix_idx}{ext}").resolve()
        roots = allowed_drive_roots(eff_data, eff_mydrive)
        if not any(dest.is_relative_to(r) for r in roots):
            raise ValueError("Resolved output path escaped allowed Drive roots")

    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    try:
        tmp.write_bytes(image_bytes)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    return str(dest), rel_to_mydrive(dest, eff_mydrive)


def list_drive_image_files(
    spec: str = "",
    *,
    kind: str = "images",
    recursive: bool = True,
    data_dir: Path | None = None,
    mydrive_root: Path | None = None,
    drive_root: Path | None = None,
) -> list[dict[str, Any]]:
    """List image files on mounted Google Drive strictly inside allowed roots."""
    eff_data = data_dir or DATA_DIR
    eff_mydrive = mydrive_root or MYDRIVE_ROOT
    eff_drive = drive_root or DRIVE_ROOT
    roots = allowed_drive_roots(eff_data, eff_mydrive)

    default_sub: str | None = None
    if not (spec or "").strip():
        if kind == "outputs":
            default_sub = "outputs"
            (eff_data / "outputs").mkdir(parents=True, exist_ok=True)
        elif kind == "inputs":
            default_sub = "inputs"
            (eff_data / "inputs").mkdir(parents=True, exist_ok=True)

    target = resolve_drive_path(
        spec or "",
        data_dir=eff_data,
        default_subdir=default_sub,
        mydrive_root=eff_mydrive,
        drive_root=eff_drive,
    )
    matched: list[Path] = []
    if target.is_file():
        if target.suffix.lower() in IMAGE_EXTS:
            matched.append(target)
    elif target.is_dir():
        if not recursive:
            for item in sorted(target.iterdir()):
                rp = item.resolve()
                if any(rp.is_relative_to(r) for r in roots) and rp.is_file() and rp.suffix.lower() in IMAGE_EXTS:
                    matched.append(rp)
        else:
            skip = {".git", "__pycache__", "cache-archives", "kernel-archives", ".Trash", ".ipynb_checkpoints"}
            for root_dir, dirnames, filenames in os.walk(target):
                dirnames[:] = sorted(d for d in dirnames if d not in skip and not d.startswith("."))
                for fn in sorted(filenames):
                    rp = (Path(root_dir) / fn).resolve()
                    if any(rp.is_relative_to(r) for r in roots) and rp.is_file() and rp.suffix.lower() in IMAGE_EXTS:
                        matched.append(rp)

    out: list[dict[str, Any]] = []
    for p in matched:
        st = p.stat()
        out.append({
            "drive_path": str(p),
            "rel_path": rel_to_mydrive(p, eff_mydrive),
            "name": p.name,
            "type": "output" if "outputs" in p.parts else "image",
            "size_bytes": st.st_size,
            "size_mb": round(st.st_size / 1e6, 2),
            "mtime": int(st.st_mtime),
        })
    return out


def store_upload_bytes(
    data: bytes,
    *,
    filename: str = "image.png",
    uploads_dir: Path | None = None,
) -> dict[str, Any]:
    """Content-address and persist a reference image upload on local SSD."""
    if not data:
        raise ValueError("Uploaded image body is empty")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"Uploaded image exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB limit")
    w, h, fmt = inspect_image_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    eff_uploads = uploads_dir or UPLOADS_DIR
    eff_uploads.mkdir(parents=True, exist_ok=True)
    ext = EXT_BY_FORMAT.get(fmt, ".png")
    dest = (eff_uploads / f"{digest}{ext}").resolve()
    if not dest.is_relative_to(eff_uploads.resolve()):
        raise ValueError("Invalid upload destination")
    if not dest.is_file():
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, dest)
        finally:
            tmp.unlink(missing_ok=True)
    else:
        os.utime(dest, None)
    _prune_uploads_dir(eff_uploads)
    return {
        "upload_id": digest,
        "uri": f"upload://{digest}",
        "format": fmt,
        "width": w,
        "height": h,
        "size_bytes": len(data),
        "filename": Path(filename or "image.png").name,
    }


def lookup_upload(upload_id: str, *, uploads_dir: Path | None = None) -> Path | None:
    uid = (upload_id or "").strip().lower()
    if not UPLOAD_ID_RE.fullmatch(uid):
        raise ValueError(f"Invalid upload_id '{upload_id}': expected 64-char hex SHA-256")
    eff_uploads = (uploads_dir or UPLOADS_DIR).resolve()
    if not eff_uploads.is_dir():
        return None
    for ext in (".png", ".webp", ".jpg", ".jpeg"):
        cand = (eff_uploads / f"{uid}{ext}").resolve()
        if cand.is_relative_to(eff_uploads) and cand.is_file():
            return cand
    return None


def _prune_uploads_dir(uploads_dir: Path) -> None:
    try:
        files = [p for p in uploads_dir.iterdir() if p.is_file() and not p.name.startswith(".")]
        total = sum(p.stat().st_size for p in files)
        if total <= MAX_UPLOADS_CACHE_BYTES:
            return
        files.sort(key=lambda p: p.stat().st_mtime)
        for p in files:
            if total <= (MAX_UPLOADS_CACHE_BYTES * 3) // 4:
                break
            sz = p.stat().st_size
            p.unlink(missing_ok=True)
            total -= sz
    except OSError:
        pass


def resolve_reference_input(
    spec: str,
    *,
    index: int,
    data_dir: Path | None = None,
    mydrive_root: Path | None = None,
    drive_root: Path | None = None,
    uploads_dir: Path | None = None,
) -> tuple[str, bytes, str]:
    """Resolve one reference image (`upload://`, `drive://`, `data:image/`, `https://`) -> `(filename, bytes, mime)`."""
    raw = (spec or "").strip()
    if not raw:
        raise ValueError(f"Reference image #{index + 1} is empty")

    if raw.startswith("upload://"):
        uid = raw[len("upload://") :].strip()
        up_path = lookup_upload(uid, uploads_dir=uploads_dir)
        if up_path is None:
            raise FileNotFoundError(f"Uploaded reference upload://{uid[:12]}... not found on server; re-upload required")
        img_bytes = up_path.read_bytes()
        _, _, fmt = inspect_image_bytes(img_bytes)
        return up_path.name, img_bytes, MIME_BY_FORMAT.get(fmt, "image/png")

    if raw.startswith("data:image/"):
        if "," not in raw:
            raise ValueError(f"Malformed data URI for reference image #{index + 1}")
        header, b64_part = raw.split(",", 1)
        img_bytes = base64.b64decode(b64_part)
        _, _, fmt = inspect_image_bytes(img_bytes)
        ext = EXT_BY_FORMAT.get(fmt, ".png")
        mime = header[5:].split(";", 1)[0] or MIME_BY_FORMAT.get(fmt, "image/png")
        return f"ref_{index + 1}{ext}", img_bytes, mime

    if raw.startswith(("http://", "https://")):
        parsed = urllib.parse.urlsplit(raw)
        if parsed.hostname in ("127.0.0.1", "localhost", "::1", "169.254.169.254", "metadata.google.internal"):
            raise ValueError(f"Refusing loopback/metadata URL in reference image #{index + 1}")
        req = urllib.request.Request(raw, headers={"User-Agent": "colabllm-minimax-h3-image/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            img_bytes = resp.read(MAX_UPLOAD_BYTES + 1)
        if len(img_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError(f"Remote reference image #{index + 1} exceeds size limit")
        _, _, fmt = inspect_image_bytes(img_bytes)
        ext = EXT_BY_FORMAT.get(fmt, ".png")
        return f"ref_{index + 1}{ext}", img_bytes, MIME_BY_FORMAT.get(fmt, "image/png")

    # Otherwise resolve strictly inside allowed Google Drive / DATA_DIR roots (rejects /etc/passwd, traversal, etc.)
    drive_file = resolve_drive_path(
        raw,
        data_dir=data_dir or DATA_DIR,
        default_subdir="inputs",
        mydrive_root=mydrive_root or MYDRIVE_ROOT,
        drive_root=drive_root or DRIVE_ROOT,
    )
    if not drive_file.is_file():
        raise FileNotFoundError(f"Google Drive reference image not found: {raw} (resolved: {drive_file})")
    img_bytes = drive_file.read_bytes()
    _, _, fmt = inspect_image_bytes(img_bytes)
    mime = MIME_BY_FORMAT.get(fmt) or mimetypes.guess_type(str(drive_file))[0] or "image/png"
    return drive_file.name, img_bytes, mime


def _parse_multipart_edits_request(content_type: str, body: bytes) -> dict[str, Any]:
    """Parse standard OpenAI `multipart/form-data` `/v1/images/edits` into a canonical dict."""
    if not content_type.lower().startswith("multipart/form-data"):
        raise ValueError(f"Unsupported Content-Type '{content_type}' for /v1/images/edits")
    raw_msg = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
    msg = BytesParser(policy=email_default_policy).parsebytes(raw_msg)
    if not msg.is_multipart():
        raise ValueError("Invalid multipart/form-data payload")

    fields: dict[str, Any] = {}
    raw_files: list[tuple[str, bytes, str]] = []
    url_refs: list[str] = []

    for part in msg.iter_parts():
        disp = part.get("Content-Disposition", "")
        if "form-data" not in disp:
            continue
        name = part.get_param("name", header="content-disposition")
        filename = part.get_param("filename", header="content-disposition")
        payload_bytes = part.get_payload(decode=True) or b""
        if not name:
            continue
        if filename is not None or name in ("image", "image[]"):
            if filename is not None:
                raw_files.append((str(filename or "image.png"), payload_bytes, part.get_content_type()))
            else:
                text_val = payload_bytes.decode("utf-8", errors="replace").strip()
                if text_val:
                    url_refs.append(text_val)
        elif name in ("url", "url[]"):
            text_val = payload_bytes.decode("utf-8", errors="replace").strip()
            if text_val:
                url_refs.append(text_val)
        elif name == "mask":
            raise ValueError("MiniMax-H3-Image does not support masked inpainting ('mask' field)")
        else:
            fields[name] = payload_bytes.decode("utf-8", errors="replace").strip()

    for key in ("n", "seed", "num_inference_steps", "min_generation_area", "output_compression", "width", "height"):
        if key in fields and fields[key] != "":
            try:
                fields[key] = int(fields[key])
            except ValueError as exc:
                raise ValueError(f"Field '{key}' must be an integer, got {fields[key]!r}") from exc
    if "guidance_scale" in fields and fields["guidance_scale"] != "":
        try:
            fields["guidance_scale"] = float(fields["guidance_scale"])
        except ValueError as exc:
            raise ValueError(f"Field 'guidance_scale' must be numeric, got {fields['guidance_scale']!r}") from exc

    fields["_multipart_files"] = raw_files
    if url_refs:
        fields["images"] = url_refs
    return fields


def parse_byte_range(range_header: str | None, total_size: int) -> tuple[int, int] | None:
    """Parse single HTTP `Range: bytes=start-end` header -> `(start, end_inclusive)` or `None`."""
    if not range_header or total_size <= 0:
        return None
    raw = range_header.strip()
    if not raw.lower().startswith("bytes=") or "," in raw:
        return None
    spec = raw[6:].strip()
    if "-" not in spec:
        return None
    start_s, end_s = spec.split("-", 1)
    start_s, end_s = start_s.strip(), end_s.strip()
    if not start_s:
        if not end_s.isdigit():
            return None
        suffix = int(end_s)
        if suffix <= 0:
            return None
        start = max(0, total_size - suffix)
        return start, total_size - 1
    if not start_s.isdigit():
        return None
    start = int(start_s)
    end = int(end_s) if end_s.isdigit() else (total_size - 1)
    if start >= total_size or start < 0 or end < start:
        raise ValueError(f"Unsatisfiable byte range {range_header} for size {total_size}")
    return start, min(end, total_size - 1)


def default_sglang_caller(
    task: str,
    *,
    prompt: str,
    width: int,
    height: int,
    seed: int,
    generator_device: str = "cpu",
    min_generation_area: int | None = None,
    quality: str | None = None,
    references: list[tuple[str, bytes, str]] | None = None,
    sglang_base_url: str | None = None,
    sglang_output_dir: Path | None = None,
    timeout: float = 900.0,
) -> bytes:
    """Invoke SGLang Diffusion on loopback using zero-Base64 File URL handoff when `--output-path` is active."""
    base = (sglang_base_url or SGLANG_BASE_URL).rstrip("/")
    out_root = (sglang_output_dir or SGLANG_OUTPUT_DIR).resolve()
    use_url_handoff = out_root.is_dir()
    resp_fmt = "url" if use_url_handoff else "b64_json"
    size_str = f"{width}x{height}"

    if task == "edit":
        if not references:
            raise ValueError("Edit task requires at least 1 reference image (<Picture 1>)")
        form_fields: list[tuple[str, str]] = [
            ("model", MODEL_REPO),
            ("prompt", prompt),
            ("size", size_str),
            ("seed", str(int(seed))),
            ("generator_device", generator_device),
            ("num_inference_steps", str(FIXED_PDD_STEPS)),
            ("response_format", resp_fmt),
            ("output_format", "png"),
            ("output_compression", "0"),
        ]
        if quality and quality != "auto":
            form_fields.append(("quality", quality))
        if min_generation_area is not None:
            form_fields.append(("min_generation_area", str(int(min_generation_area))))
        form_files = [("image", fname, mime, fbytes) for fname, fbytes, mime in references]
        ct_hdr, raw_body = build_multipart_form(form_fields, form_files)
        req = urllib.request.Request(
            f"{base}/v1/images/edits",
            data=raw_body,
            headers={"Content-Type": ct_hdr},
            method="POST",
        )
    else:
        json_payload: dict[str, Any] = {
            "model": MODEL_REPO,
            "prompt": prompt,
            "size": size_str,
            "width": width,
            "height": height,
            "n": 1,
            "seed": int(seed),
            "generator_device": generator_device,
            "num_inference_steps": FIXED_PDD_STEPS,
            "response_format": resp_fmt,
            "output_format": "png",
            "output_compression": 0,
        }
        if quality and quality != "auto":
            json_payload["quality"] = quality
        if min_generation_area is not None:
            json_payload["min_generation_area"] = int(min_generation_area)
        req = urllib.request.Request(
            f"{base}/v1/images/generations",
            data=json.dumps(json_payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            resp_obj = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        detail = err.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"SGLang upstream HTTP {err.code}: {detail}") from err

    items = resp_obj.get("data") or []
    if not items or not isinstance(items[0], dict):
        raise RuntimeError(f"SGLang returned empty image data: {resp_obj}")
    first = items[0]

    # Preferred fast path: read raw uncompressed PNG directly from shared local NVMe SSD and unlink scratch copy!
    file_path = first.get("file_path")
    if file_path:
        fp = Path(str(file_path)).resolve()
        if fp.is_relative_to(out_root) and fp.is_file():
            try:
                return fp.read_bytes()
            finally:
                fp.unlink(missing_ok=True)

    if first.get("url"):
        content_url = urllib.parse.urljoin(f"{base}/", str(first["url"]))
        with opener.open(content_url, timeout=120) as c_resp:
            return c_resp.read()

    if first.get("b64_json"):
        return base64.b64decode(first["b64_json"])

    raise RuntimeError(f"SGLang response missing file_path, url, and b64_json: {first}")


class MiniMaxH3JobStore:
    """Thread-safe / asyncio-friendly single-GPU job queue with local NVMe output retention."""

    def __init__(
        self,
        *,
        jobs_dir: Path | None = None,
        uploads_dir: Path | None = None,
        data_dir: Path | None = None,
        drive_inputs_dir: Path | None = None,
        drive_outputs_dir: Path | None = None,
        mydrive_root: Path | None = None,
        drive_root: Path | None = None,
        sglang_caller: Callable[..., bytes] | None = None,
    ) -> None:
        self.jobs_dir = jobs_dir or JOBS_DIR
        self.uploads_dir = uploads_dir or UPLOADS_DIR
        self.data_dir = data_dir or DATA_DIR
        self.drive_inputs_dir = drive_inputs_dir or (self.data_dir / "inputs" if data_dir else DRIVE_INPUTS_DIR)
        self.drive_outputs_dir = drive_outputs_dir or (self.data_dir / "outputs" if data_dir else DRIVE_OUTPUTS_DIR)
        self.mydrive_root = mydrive_root or MYDRIVE_ROOT
        self.drive_root = drive_root or DRIVE_ROOT
        self.sglang_caller = sglang_caller or default_sglang_caller
        self.jobs: dict[str, dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="h3-gpu-worker")

    def prepare_canonical_job(self, payload: dict[str, Any], *, default_task: str = "t2i") -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("Request body must be a JSON object")
        validate_h3_sampling_contract(payload)

        raw_task = (payload.get("task") or payload.get("task_type") or default_task).strip().lower()
        if raw_task in ("t2i", "text2image", "generation", "generations", "image"):
            task = "t2i"
        elif raw_task in ("edit", "edits", "i2i", "image_edit"):
            task = "edit"
        else:
            raise ValueError(f"Unsupported task '{raw_task}': expected 't2i' or 'edit'")

        prompt = str(payload.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Field 'prompt' must be a non-empty string")

        multipart_files: list[tuple[str, bytes, str]] = list(payload.get("_multipart_files") or [])
        raw_images = payload.get("images")
        if raw_images is None and payload.get("image") is not None:
            raw_images = [payload["image"]] if isinstance(payload["image"], str) else payload["image"]
        image_specs: list[str] = []
        if raw_images is not None:
            if not isinstance(raw_images, list) or not all(isinstance(x, str) for x in raw_images):
                raise ValueError("Field 'images' must be a list of strings (drive://, upload://, https://, or data:)")
            image_specs = [x.strip() for x in raw_images if x and x.strip()]

        total_refs = len(multipart_files) + len(image_specs)
        if task == "edit" and total_refs == 0:
            raise ValueError("Edit task requires 1-16 reference images via 'images' or multipart 'image'")
        if total_refs > MAX_REFERENCE_IMAGES:
            raise ValueError(f"MiniMax-H3-Image supports at most {MAX_REFERENCE_IMAGES} reference images (got {total_refs})")
        if total_refs > 0 and "task" not in payload and "task_type" not in payload:
            task = "edit"

        resolved_refs: list[tuple[str, bytes, str]] = []
        for fname, fbytes, fmime in multipart_files:
            inspect_image_bytes(fbytes)
            resolved_refs.append((fname, fbytes, fmime or "image/png"))
        for idx, spec in enumerate(image_specs):
            resolved_refs.append(
                resolve_reference_input(
                    spec,
                    index=len(resolved_refs) + idx,
                    data_dir=self.data_dir,
                    mydrive_root=self.mydrive_root,
                    drive_root=self.drive_root,
                    uploads_dir=self.uploads_dir,
                )
            )

        if task == "edit":
            prompt = normalize_edit_prompt_tags(prompt)

        first_ref_dims: tuple[int, int] | None = None
        if resolved_refs:
            rw, rh, _ = inspect_image_bytes(resolved_refs[0][1])
            first_ref_dims = (rw, rh)

        width, height = resolve_dimensions(
            size=payload.get("size"),
            width=payload.get("width"),
            height=payload.get("height"),
            aspect=payload.get("aspect"),
            tier=payload.get("tier"),
            is_edit=(task == "edit"),
            ref_dimensions=first_ref_dims,
        )

        min_area = validate_min_generation_area(
            payload.get("min_generation_area"),
            width=width,
            height=height,
            fast_preview=bool(payload.get("fast_preview", False)),
        )

        n_raw = payload.get("n", 1)
        if n_raw is None:
            n_raw = 1
        if isinstance(n_raw, bool) or not isinstance(n_raw, int) or n_raw < 1 or n_raw > MAX_IMAGES_PER_JOB:
            raise ValueError(f"Field 'n' must be an integer between 1 and {MAX_IMAGES_PER_JOB}")

        seed_raw = payload.get("seed")
        if seed_raw is None or seed_raw == "":
            seed = secrets.randbelow(2**31 - 1)
        else:
            if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
                raise ValueError("Field 'seed' must be an integer")
            seed = int(seed_raw)

        gen_dev = str(payload.get("generator_device") or "cpu").strip().lower()
        if gen_dev not in ("cpu", "cuda"):
            raise ValueError("generator_device must be 'cpu' or 'cuda'")

        out_fmt = str(payload.get("output_format") or "png").strip().lower()
        if out_fmt == "jpg":
            out_fmt = "jpeg"
        if out_fmt not in ("png", "webp", "jpeg"):
            raise ValueError(f"Unsupported output_format '{out_fmt}': expected png, webp, or jpeg")

        out_quality_raw = payload.get("output_quality", 92)
        if isinstance(out_quality_raw, str) and out_quality_raw.lower() == "default":
            out_quality = 92
        else:
            out_quality = max(1, min(100, int(out_quality_raw)))

        png_level_raw = payload.get("png_compress_level", payload.get("output_compression", 6))
        png_compress_level = 6 if png_level_raw is None else max(0, min(9, int(png_level_raw)))

        save_spec = payload.get("save_to_drive")
        if save_spec is not None and save_spec is not False and str(save_spec).strip() != "":
            save_str = "auto" if save_spec is True else str(save_spec).strip()
            # Pre-validate Drive destination path BEFORE running expensive GPU inference!
            if save_str.lower() not in ("auto", "true", "1"):
                resolve_drive_path(
                    save_str,
                    data_dir=self.data_dir,
                    default_subdir="outputs",
                    mydrive_root=self.mydrive_root,
                    drive_root=self.drive_root,
                )
        else:
            save_str = ""

        quality_raw = str(payload.get("quality") or "").strip().lower()
        if not quality_raw or quality_raw in ("auto", "lossless", "hd", "standard", "high", "medium", "low"):
            quality: str | None = "lossless"
        elif quality_raw == "exact":
            quality = "exact"
        else:
            raise ValueError(f"Unsupported quality '{quality_raw}': expected 'lossless' or 'exact'")
        return {
            "model": normalize_model_name(payload.get("model")),
            "task": task,
            "prompt": prompt,
            "width": width,
            "height": height,
            "size": f"{width}x{height}",
            "n": n_raw,
            "seed": seed,
            "generator_device": gen_dev,
            "min_generation_area": min_area,
            "quality": quality,
            "output_format": out_fmt,
            "output_quality": out_quality,
            "png_compress_level": png_compress_level,
            "save_to_drive": save_str,
            "drive_only": bool(payload.get("drive_only", False)),
            "reference_count": len(resolved_refs),
            "_references": resolved_refs,
        }

    def create_job(self, canonical: dict[str, Any]) -> dict[str, Any]:
        job_id = f"job_{int(time.time())}_{secrets.token_hex(6)}"
        # Ensure strict JOB_ID_RE compatibility: hex only after `job_`
        job_id = f"job_{int(time.time()):08x}{secrets.token_hex(8)}"
        now = time.time()
        record: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "elapsed_sec": 0.0,
            "task": canonical["task"],
            "model": canonical["model"],
            "prompt": canonical["prompt"],
            "size": canonical["size"],
            "width": canonical["width"],
            "height": canonical["height"],
            "n": canonical["n"],
            "seed": canonical["seed"],
            "generator_device": canonical["generator_device"],
            "min_generation_area": canonical["min_generation_area"],
            "output_format": canonical["output_format"],
            "output_quality": canonical["output_quality"],
            "reference_count": canonical["reference_count"],
            "save_to_drive": canonical["save_to_drive"] or None,
            "completed_images": 0,
            "total_images": canonical["n"],
            "outputs": [],
            "saved_drive_paths": [],
            "rel_drive_paths": [],
            "error": None,
            "_cancelled": False,
        }
        self.jobs[job_id] = record
        self._prune_jobs()
        self._executor.submit(self._run_job_sync, job_id, canonical)
        return self.public_view(record)

    def run_sync(self, canonical: dict[str, Any]) -> tuple[dict[str, Any], list[bytes]]:
        """Execute synchronously via the single-GPU worker and return `(job_view, encoded_images)`."""
        job_id = f"job_{int(time.time()):08x}{secrets.token_hex(8)}"
        now = time.time()
        record: dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "started_at": None,
            "completed_at": None,
            "elapsed_sec": 0.0,
            "task": canonical["task"],
            "model": canonical["model"],
            "prompt": canonical["prompt"],
            "size": canonical["size"],
            "width": canonical["width"],
            "height": canonical["height"],
            "n": canonical["n"],
            "seed": canonical["seed"],
            "generator_device": canonical["generator_device"],
            "min_generation_area": canonical["min_generation_area"],
            "output_format": canonical["output_format"],
            "output_quality": canonical["output_quality"],
            "reference_count": canonical["reference_count"],
            "save_to_drive": canonical["save_to_drive"] or None,
            "completed_images": 0,
            "total_images": canonical["n"],
            "outputs": [],
            "saved_drive_paths": [],
            "rel_drive_paths": [],
            "error": None,
            "_cancelled": False,
        }
        self.jobs[job_id] = record
        self._prune_jobs()
        fut = self._executor.submit(self._run_job_sync, job_id, canonical)
        encoded = fut.result()
        if record["status"] != "succeeded":
            raise RuntimeError(record.get("error") or "Image generation failed")
        return self.public_view(record), encoded

    def _run_job_sync(self, job_id: str, canonical: dict[str, Any]) -> list[bytes]:
        record = self.jobs.get(job_id)
        if record is None or record.get("_cancelled"):
            if record is not None:
                record["status"] = "cancelled"
                record["completed_at"] = time.time()
            return []

        t0 = time.monotonic()
        record["status"] = "running"
        record["started_at"] = time.time()
        job_dir = (self.jobs_dir / job_id).resolve()
        job_dir.mkdir(parents=True, exist_ok=True)
        encoded_list: list[bytes] = []

        try:
            total = canonical["n"]
            for idx in range(total):
                if record.get("_cancelled"):
                    record["status"] = "cancelled"
                    record["completed_at"] = time.time()
                    record["elapsed_sec"] = round(time.monotonic() - t0, 3)
                    return encoded_list

                cur_seed = canonical["seed"] + idx
                raw_png = self.sglang_caller(
                    canonical["task"],
                    prompt=canonical["prompt"],
                    width=canonical["width"],
                    height=canonical["height"],
                    seed=cur_seed,
                    generator_device=canonical["generator_device"],
                    min_generation_area=canonical["min_generation_area"],
                    quality=canonical["quality"],
                    references=canonical["_references"] if canonical["task"] == "edit" else None,
                )
                out_bytes, actual_w, actual_h, actual_fmt = transcode_image_bytes(
                    raw_png,
                    output_format=canonical["output_format"],
                    output_quality=canonical["output_quality"],
                    png_compress_level=canonical["png_compress_level"],
                )
                ext = EXT_BY_FORMAT.get(actual_fmt, ".png")
                local_file = (job_dir / f"output_{idx:02d}{ext}").resolve()
                local_file.write_bytes(out_bytes)
                sha = hashlib.sha256(out_bytes).hexdigest()

                saved_drive = None
                rel_drive = None
                if canonical["save_to_drive"]:
                    prefix = "h3-edit" if canonical["task"] == "edit" else "h3-gen"
                    saved_drive, rel_drive = save_output_image_to_drive(
                        out_bytes,
                        canonical["save_to_drive"],
                        index=idx,
                        total=total,
                        fmt=actual_fmt,
                        prefix=prefix,
                        data_dir=self.data_dir,
                        output_dir=self.drive_outputs_dir,
                        mydrive_root=self.mydrive_root,
                        drive_root=self.drive_root,
                    )
                    record["saved_drive_paths"].append(saved_drive)
                    record["rel_drive_paths"].append(rel_drive)

                record["outputs"].append({
                    "index": idx,
                    "seed": cur_seed,
                    "format": actual_fmt,
                    "mime_type": MIME_BY_FORMAT.get(actual_fmt, "image/png"),
                    "width": actual_w,
                    "height": actual_h,
                    "size_bytes": len(out_bytes),
                    "sha256": sha,
                    "content_path": f"/api/ct/jobs/{job_id}/content?index={idx}",
                    "saved_drive_path": saved_drive,
                    "rel_drive_path": rel_drive,
                    "_local_file": str(local_file),
                })
                record["completed_images"] = idx + 1
                record["elapsed_sec"] = round(time.monotonic() - t0, 3)
                encoded_list.append(out_bytes)

            record["status"] = "succeeded"
            record["completed_at"] = time.time()
            record["elapsed_sec"] = round(time.monotonic() - t0, 3)
            return encoded_list
        except Exception as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            record["completed_at"] = time.time()
            record["elapsed_sec"] = round(time.monotonic() - t0, 3)
            return encoded_list
        finally:
            # Release reference image bytes from memory as soon as inference completes.
            canonical["_references"] = []

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        if not JOB_ID_RE.fullmatch(job_id or ""):
            raise ValueError(f"Invalid job_id '{job_id}'")
        rec = self.jobs.get(job_id)
        if rec is None:
            return None
        if rec["status"] == "running" and rec.get("started_at"):
            rec["elapsed_sec"] = round(max(0.0, time.time() - float(rec["started_at"])), 3)
        return rec

    def cancel_job(self, job_id: str) -> dict[str, Any] | None:
        rec = self.get_job(job_id)
        if rec is None:
            return None
        if rec["status"] == "queued":
            rec["_cancelled"] = True
            rec["status"] = "cancelled"
            rec["completed_at"] = time.time()
        elif rec["status"] == "running":
            rec["_cancelled"] = True
        return self.public_view(rec)

    def get_output_bytes(
        self,
        job_id: str,
        *,
        index: int = 0,
        output_format: str | None = None,
        output_quality: int | None = None,
    ) -> tuple[bytes, dict[str, Any]]:
        rec = self.get_job(job_id)
        if rec is None:
            raise FileNotFoundError(f"Job '{job_id}' not found")
        if rec["status"] != "succeeded":
            raise ValueError(f"Job '{job_id}' is in state '{rec['status']}' (not 'succeeded')")
        outputs = rec.get("outputs") or []
        if index < 0 or index >= len(outputs):
            raise ValueError(f"Output index {index} out of range (0..{max(0, len(outputs) - 1)})")
        meta = dict(outputs[index])
        local_p = Path(meta["_local_file"]).resolve()
        if not local_p.is_relative_to(self.jobs_dir.resolve()) or not local_p.is_file():
            raise FileNotFoundError(f"Output artifact for {job_id}[{index}] expired from local SSD")
        raw = local_p.read_bytes()
        req_fmt = (output_format or "").strip().lower()
        if req_fmt == "jpg":
            req_fmt = "jpeg"
        if req_fmt and req_fmt != meta["format"]:
            raw, w, h, fmt = transcode_image_bytes(
                raw,
                output_format=req_fmt,
                output_quality=output_quality if output_quality is not None else 92,
            )
            meta = {
                **meta,
                "format": fmt,
                "mime_type": MIME_BY_FORMAT.get(fmt, "image/png"),
                "width": w,
                "height": h,
                "size_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        meta.pop("_local_file", None)
        return raw, meta

    @staticmethod
    def public_view(rec: dict[str, Any]) -> dict[str, Any]:
        public_outputs = []
        for item in rec.get("outputs") or []:
            clean_item = {k: v for k, v in item.items() if not k.startswith("_")}
            public_outputs.append(clean_item)
        return {
            k: (public_outputs if k == "outputs" else v)
            for k, v in rec.items()
            if not k.startswith("_")
        }

    def list_jobs(self, *, limit: int = 20) -> list[dict[str, Any]]:
        ordered = sorted(self.jobs.values(), key=lambda r: float(r.get("created_at") or 0), reverse=True)
        return [self.public_view(r) for r in ordered[: max(1, min(100, limit))]]

    def _prune_jobs(self) -> None:
        if len(self.jobs) <= MAX_RETAINED_JOBS:
            return
        terminal = [
            r for r in self.jobs.values() if r.get("status") in ("succeeded", "failed", "cancelled")
        ]
        terminal.sort(key=lambda r: float(r.get("created_at") or 0))
        while len(self.jobs) > MAX_RETAINED_JOBS and terminal:
            victim = terminal.pop(0)
            jid = victim["job_id"]
            self.jobs.pop(jid, None)
            jdir = (self.jobs_dir / jid).resolve()
            if jdir.is_relative_to(self.jobs_dir.resolve()) and jdir.is_dir():
                shutil.rmtree(jdir, ignore_errors=True)


def check_sglang_upstream_live(sglang_base_url: str | None = None, timeout: float = 5.0) -> bool:
    base = (sglang_base_url or SGLANG_BASE_URL).rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for route in ("/health", "/liveness", "/v1/models"):
        try:
            with opener.open(f"{base}{route}", timeout=timeout) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            continue
    return False


class MiniMaxH3ImageGateway:
    """Pure-ASGI authenticated front-gateway for MiniMax-H3-Image."""

    def __init__(
        self,
        *,
        job_store: MiniMaxH3JobStore | None = None,
        upstream_health_fn: Callable[[], bool] | None = None,
    ) -> None:
        self.store = job_store or MiniMaxH3JobStore()
        self.upstream_health_fn = upstream_health_fn or check_sglang_upstream_live

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope.get("type") != "http":
            return

        method = (scope.get("method") or "GET").upper()
        path = scope.get("path") or "/"
        raw_qs = (scope.get("query_string") or b"").decode("utf-8", errors="replace")
        query = urllib.parse.parse_qs(raw_qs, keep_blank_values=True)
        headers = {
            k.decode("latin1").lower(): v.decode("latin1")
            for k, v in (scope.get("headers") or [])
        }

        if method == "OPTIONS":
            await self._send_bytes(send, 204, b"", content_type="text/plain")
            return

        if not verify_bearer_auth(headers, API_TOKEN):
            await self._send_json(
                send,
                401,
                {"error": {"message": "Unauthorized: valid Bearer token or X-API-Key required", "type": "auth_error"}},
                extra_headers=[(b"www-authenticate", b'Bearer realm="colabllm"')],
            )
            return

        try:
            if path == "/health" and method in ("GET", "HEAD"):
                ok = await asyncio.to_thread(self.upstream_health_fn)
                status = 200 if ok else 503
                await self._send_json(
                    send,
                    status,
                    {"status": "ok" if ok else "starting", "service": SERVICE_NAME, "model": MODEL_REPO},
                    head_only=(method == "HEAD"),
                )
                return

            if path == "/v1/models" and method == "GET":
                created = int(time.time())
                models = [
                    {"id": MODEL_REPO, "object": "model", "created": created, "owned_by": "MiniMaxAI"},
                ]
                if MODEL_ALIAS and MODEL_ALIAS != MODEL_REPO:
                    models.append({"id": MODEL_ALIAS, "object": "model", "created": created, "owned_by": "MiniMaxAI"})
                await self._send_json(send, 200, {"object": "list", "data": models})
                return

            if path == "/api/ct/info" and method == "GET":
                drive_mounted = self.store.mydrive_root.is_dir()
                await self._send_json(
                    send,
                    200,
                    {
                        "service": SERVICE_NAME,
                        "engine": "sglang-diffusion",
                        "model_repo": MODEL_REPO,
                        "served_models": [MODEL_REPO, *([MODEL_ALIAS] if MODEL_ALIAS != MODEL_REPO else [])],
                        "quantization": QUANTIZATION,
                        "placement": PLACEMENT,
                        "warmup_mode": WARMUP_MODE,
                        "pdd_steps": FIXED_PDD_STEPS,
                        "default_min_generation_area": DEFAULT_MIN_GENERATION_AREA,
                        "max_reference_images": MAX_REFERENCE_IMAGES,
                        "max_images_per_job": MAX_IMAGES_PER_JOB,
                        "aspect_presets": {
                            tier: {asp: f"{w}x{h}" for asp, (w, h) in presets.items()}
                            for tier, presets in ASPECT_PRESETS.items()
                        },
                        "drive_mounted": drive_mounted,
                        "data_dir": str(self.store.data_dir),
                        "drive_inputs_dir": str(self.store.drive_inputs_dir),
                        "drive_outputs_dir": str(self.store.drive_outputs_dir),
                        "internal_endpoints": [
                            "POST /api/ct/uploads",
                            "HEAD /api/ct/uploads/{sha256}",
                            "POST /api/ct/jobs",
                            "GET /api/ct/jobs",
                            "GET /api/ct/jobs/{job_id}",
                            "GET /api/ct/jobs/{job_id}/content",
                            "DELETE /api/ct/jobs/{job_id}",
                            "GET /api/ct/drive-list",
                            "GET /api/ct/info",
                        ],
                    },
                )
                return

            if path == "/api/ct/drive-list" and method == "GET":
                spec = (query.get("path") or [""])[0]
                kind = (query.get("kind") or ["images"])[0]
                recursive = (query.get("recursive") or ["1"])[0] not in ("0", "false", "False")
                files = await asyncio.to_thread(
                    list_drive_image_files,
                    spec,
                    kind=kind,
                    recursive=recursive,
                    data_dir=self.store.data_dir,
                    mydrive_root=self.store.mydrive_root,
                    drive_root=self.store.drive_root,
                )
                await self._send_json(
                    send,
                    200,
                    {
                        "drive_mounted": self.store.mydrive_root.is_dir(),
                        "data_dir": str(self.store.data_dir),
                        "kind": kind,
                        "files": files,
                    },
                )
                return

            # Content-addressed upload deduplication check & binary upload
            if path.startswith("/api/ct/uploads/") and method in ("GET", "HEAD"):
                uid = path[len("/api/ct/uploads/") :].strip("/")
                found = lookup_upload(uid, uploads_dir=self.store.uploads_dir)
                if found is None:
                    await self._send_json(
                        send,
                        404,
                        {"exists": False, "upload_id": uid},
                        head_only=(method == "HEAD"),
                    )
                else:
                    st = found.stat()
                    await self._send_json(
                        send,
                        200,
                        {
                            "exists": True,
                            "upload_id": uid,
                            "uri": f"upload://{uid}",
                            "size_bytes": st.st_size,
                        },
                        head_only=(method == "HEAD"),
                    )
                return

            if path == "/api/ct/uploads" and method == "POST":
                raw_body = await self._read_body(receive, max_bytes=MAX_UPLOAD_BYTES)
                fname = headers.get("x-filename") or (query.get("filename") or ["image.png"])[0]
                meta = await asyncio.to_thread(
                    store_upload_bytes,
                    raw_body,
                    filename=fname,
                    uploads_dir=self.store.uploads_dir,
                )
                await self._send_json(send, 200, meta)
                return

            # Lightweight async job API
            if path == "/api/ct/jobs" and method == "GET":
                limit = int((query.get("limit") or ["20"])[0])
                await self._send_json(send, 200, {"jobs": self.store.list_jobs(limit=limit)})
                return

            if path == "/api/ct/jobs" and method == "POST":
                raw_body = await self._read_body(receive, max_bytes=MAX_JSON_BODY_BYTES)
                payload = json.loads(raw_body.decode("utf-8"))
                canonical = await asyncio.to_thread(self.store.prepare_canonical_job, payload, default_task="t2i")
                job_view = self.store.create_job(canonical)
                await self._send_json(
                    send,
                    202,
                    job_view,
                    extra_headers=[(b"x-job-id", job_view["job_id"].encode("ascii"))],
                )
                return

            if path.startswith("/api/ct/jobs/"):
                sub = path[len("/api/ct/jobs/") :].strip("/")
                parts = sub.split("/")
                job_id = parts[0]

                if len(parts) == 1 and method == "GET":
                    rec = self.store.get_job(job_id)
                    if rec is None:
                        await self._send_json(send, 404, {"error": {"message": f"Job '{job_id}' not found"}})
                        return
                    await self._send_json(send, 200, self.store.public_view(rec))
                    return

                if len(parts) == 1 and method == "DELETE":
                    cancelled = self.store.cancel_job(job_id)
                    if cancelled is None:
                        await self._send_json(send, 404, {"error": {"message": f"Job '{job_id}' not found"}})
                        return
                    await self._send_json(send, 200, cancelled)
                    return

                if len(parts) == 2 and parts[1] == "cancel" and method == "POST":
                    cancelled = self.store.cancel_job(job_id)
                    if cancelled is None:
                        await self._send_json(send, 404, {"error": {"message": f"Job '{job_id}' not found"}})
                        return
                    await self._send_json(send, 200, cancelled)
                    return

                if len(parts) == 2 and parts[1] == "content" and method in ("GET", "HEAD"):
                    idx = int((query.get("index") or ["0"])[0])
                    req_fmt = (query.get("format") or [""])[0] or None
                    req_qual_s = (query.get("quality") or [""])[0]
                    req_qual = int(req_qual_s) if req_qual_s.isdigit() else None
                    img_bytes, meta = await asyncio.to_thread(
                        self.store.get_output_bytes,
                        job_id,
                        index=idx,
                        output_format=req_fmt,
                        output_quality=req_qual,
                    )
                    await self._stream_image_response(
                        send,
                        img_bytes,
                        meta=meta,
                        job_id=job_id,
                        range_header=headers.get("range"),
                        if_none_match=headers.get("if-none-match"),
                        head_only=(method == "HEAD"),
                    )
                    return

            # OpenAI-compatible synchronous endpoints (/v1/images/generations, /v1/images/edits, /api/ct/edit)
            if path in ("/v1/images/generations", "/v1/images/edits", "/api/ct/edit") and method == "POST":
                raw_body = await self._read_body(receive, max_bytes=MAX_JSON_BODY_BYTES)
                ct = headers.get("content-type", "")
                default_task = "t2i" if path == "/v1/images/generations" else "edit"
                if ct.lower().startswith("multipart/form-data"):
                    payload = await asyncio.to_thread(_parse_multipart_edits_request, ct, raw_body)
                else:
                    payload = json.loads(raw_body.decode("utf-8"))
                resp_format = str(payload.get("response_format") or "b64_json").strip().lower()
                if resp_format not in ("b64_json", "url"):
                    raise ValueError(f"Unsupported response_format '{resp_format}': expected 'b64_json' or 'url'")

                canonical = await asyncio.to_thread(
                    self.store.prepare_canonical_job,
                    payload,
                    default_task=default_task,
                )
                job_view, encoded_list = await asyncio.to_thread(self.store.run_sync, canonical)
                data_items: list[dict[str, Any]] = []
                for idx, out_meta in enumerate(job_view.get("outputs") or []):
                    item: dict[str, Any] = {"revised_prompt": canonical["prompt"]}
                    if resp_format == "url":
                        item["url"] = out_meta["content_path"]
                    elif not canonical["drive_only"]:
                        item["b64_json"] = base64.b64encode(encoded_list[idx]).decode("ascii")
                    if out_meta.get("saved_drive_path"):
                        item["saved_drive_path"] = out_meta["saved_drive_path"]
                        item["rel_drive_path"] = out_meta["rel_drive_path"]
                    data_items.append(item)

                extra_hdrs: list[tuple[bytes, bytes]] = [(b"x-job-id", job_view["job_id"].encode("ascii"))]
                if job_view.get("rel_drive_paths"):
                    safe_dp = sanitize_header_param(job_view["rel_drive_paths"][0])
                    extra_hdrs.append((b"x-saved-drive-path", safe_dp.encode("utf-8")))

                await self._send_json(
                    send,
                    200,
                    {
                        "created": int(job_view["created_at"]),
                        "job_id": job_view["job_id"],
                        "size": job_view["size"],
                        "output_format": job_view["output_format"],
                        "elapsed_sec": job_view["elapsed_sec"],
                        "saved_drive_paths": job_view["saved_drive_paths"],
                        "rel_drive_paths": job_view["rel_drive_paths"],
                        "data": data_items,
                    },
                    extra_headers=extra_hdrs,
                )
                return

            if path == "/api/ct/logs" and method == "GET":
                lines = max(1, min(200, int((query.get("lines") or ["40"])[0])))
                await self._send_json(
                    send,
                    200,
                    {
                        "gateway_log": tail_service_log(LOG_FILE, lines=lines),
                        "sglang_log": tail_service_log(SGLANG_LOG_FILE, lines=lines),
                    },
                )
                return

            await self._send_json(send, 404, {"error": {"message": f"Not found: {method} {path}"}})

        except (ValueError, json.JSONDecodeError) as bad_req:
            await self._send_json(send, 400, {"error": {"message": str(bad_req), "type": "invalid_request_error"}})
        except FileNotFoundError as not_found:
            await self._send_json(send, 404, {"error": {"message": str(not_found), "type": "not_found_error"}})
        except Exception as err:
            await self._send_json(send, 500, {"error": {"message": str(err), "type": "server_error"}})

    @staticmethod
    async def _read_body(receive: Callable, *, max_bytes: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            msg = await receive()
            mtype = msg.get("type")
            if mtype == "http.request":
                chunk = msg.get("body") or b""
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"Request body exceeds {max_bytes // (1024 * 1024)} MiB limit")
                chunks.append(chunk)
                if not msg.get("more_body", False):
                    break
            elif mtype == "http.disconnect":
                break
        return b"".join(chunks)

    @staticmethod
    async def _send_bytes(
        send: Callable,
        status: int,
        body: bytes,
        *,
        content_type: str = "application/octet-stream",
        extra_headers: list[tuple[bytes, bytes]] | None = None,
        head_only: bool = False,
    ) -> None:
        hdrs: list[tuple[bytes, bytes]] = [
            (b"content-type", content_type.encode("utf-8")),
            (b"content-length", str(len(body)).encode("ascii")),
            *(extra_headers or []),
        ]
        await send({"type": "http.response.start", "status": status, "headers": _inject_cors_headers(hdrs)})
        await send({"type": "http.response.body", "body": b"" if head_only else body, "more_body": False})

    async def _send_json(
        self,
        send: Callable,
        status: int,
        payload: dict[str, Any],
        *,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
        head_only: bool = False,
    ) -> None:
        raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await self._send_bytes(
            send,
            status,
            raw,
            content_type="application/json; charset=utf-8",
            extra_headers=extra_headers,
            head_only=head_only,
        )

    async def _stream_image_response(
        self,
        send: Callable,
        img_bytes: bytes,
        *,
        meta: dict[str, Any],
        job_id: str,
        range_header: str | None,
        if_none_match: str | None,
        head_only: bool = False,
    ) -> None:
        etag = f'"{meta["sha256"]}"'
        if if_none_match and if_none_match.strip() == etag:
            await self._send_bytes(send, 304, b"", extra_headers=[(b"etag", etag.encode("ascii"))], head_only=True)
            return

        total = len(img_bytes)
        ext = EXT_BY_FORMAT.get(meta["format"], ".png")
        fname = sanitize_header_param(f"{job_id}_{meta['index']:02d}{ext}")
        base_headers: list[tuple[bytes, bytes]] = [
            (b"accept-ranges", b"bytes"),
            (b"etag", etag.encode("ascii")),
            (b"x-job-id", job_id.encode("ascii")),
            (b"x-image-width", str(meta["width"]).encode("ascii")),
            (b"x-image-height", str(meta["height"]).encode("ascii")),
            (b"x-image-format", str(meta["format"]).encode("ascii")),
            (b"content-disposition", f'inline; filename="{fname}"'.encode("utf-8")),
        ]
        if meta.get("rel_drive_path"):
            safe_dp = sanitize_header_param(str(meta["rel_drive_path"]))
            base_headers.append((b"x-saved-drive-path", safe_dp.encode("utf-8")))

        try:
            byte_range = parse_byte_range(range_header, total)
        except ValueError:
            base_headers.append((b"content-range", f"bytes */{total}".encode("ascii")))
            await self._send_bytes(send, 416, b"", extra_headers=base_headers, head_only=True)
            return

        if byte_range is not None:
            start, end = byte_range
            view = img_bytes[start : end + 1]
            status = 206
            base_headers.append((b"content-range", f"bytes {start}-{end}/{total}".encode("ascii")))
        else:
            view = img_bytes
            status = 200

        hdrs: list[tuple[bytes, bytes]] = [
            (b"content-type", str(meta["mime_type"]).encode("ascii")),
            (b"content-length", str(len(view)).encode("ascii")),
            *base_headers,
        ]
        await send({"type": "http.response.start", "status": status, "headers": _inject_cors_headers(hdrs)})
        if head_only or not view:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        offset = 0
        n = len(view)
        while offset < n:
            nxt = min(n, offset + STREAM_CHUNK_BYTES)
            await send({
                "type": "http.response.body",
                "body": view[offset:nxt],
                "more_body": nxt < n,
            })
            offset = nxt


app = MiniMaxH3ImageGateway()


def main() -> None:
    import uvicorn  # type: ignore[import-not-found]

    for d in (LOCAL_WORK_DIR, SGLANG_OUTPUT_DIR, SGLANG_INPUT_DIR, UPLOADS_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=SERVICE_PORT,
        log_level="info",
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    main()
