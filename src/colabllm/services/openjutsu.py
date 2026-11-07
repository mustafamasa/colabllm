#!/usr/bin/env python3
"""Openjutsu (MiniMax H3 motion transfer, object edit, face/colour post-processing & FLUX.3 picture prep) on Colab G4.

Co-locates all three loopback tiers inside one Colab G4 (RTX PRO 6000 Blackwell, 96 GB VRAM) VM behind Caddy auth:
  1. ComfyUI (`127.0.0.1:8188`) + all 13 Openjutsu custom node packs + local NVMe SSD models
  2. Patched `comfy-api-proxy` (`127.0.0.1:8189`) for live progress & multi-clip resume
  3. Openjutsu Next.js 16 production web UI & API (`127.0.0.1:3000`)

Usage:
  colabllm openjutsu up --domain
  colabllm openjutsu up --domain --profile core
  colabllm openjutsu check [--mode all|simple|advanced|object|face|color|montage|people|mask|fit]
  colabllm openjutsu swap --video input.mp4 -p person.png --steps 4 --clip-seconds 7.5 -o result.mp4
  colabllm openjutsu object --video input.mp4 --target jacket --change-to "leather jacket" -o edited.mp4
  colabllm openjutsu mask --video input.mp4 -o mask.mp4
  colabllm openjutsu face --video result.mp4 -p person.png -o fixed_face.mp4
  colabllm openjutsu color --video result.mp4 -o graded.mp4
  colabllm openjutsu job JOB_ID --wait -o result.mp4
  colabllm openjutsu download JOB_ID -o result.mp4
  colabllm openjutsu cancel JOB_ID
  colabllm openjutsu info | status | logs -f | down
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cli_support import (  # noqa: E402
    CliError,
    Deadline,
    SameOriginRedirect,
    UsageError,
    diagnostic,
    nonnegative_seconds,
    positive_seconds,
    resolve_endpoint,
)
from colabsvc import Service, State, main  # noqa: E402

SERVICE = Service(
    name="openjutsu",
    notebook="openjutsu.ipynb",
    service_port=3000,
    auth="caddy",
    public_port=18080,
    gpu="G4",
    ready_timeout=3600,
    env={
        "CT_OPENJUTSU_PROFILE": "full",
        "CT_OPENJUTSU_CHAINS_STORAGE": "ssd",
        "CT_COMFY_VRAM_MODE": "auto",
    },
)

DEFAULT_LORA = ("minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors", 1.0)

DEFAULT_MODEL_SETTINGS: dict[str, object] = {
    "model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "loras": [{"name": DEFAULT_LORA[0], "strength": 1}],
    "steps": 6,
    "textEncoder": "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
    "videoVae": "minimax_h3_video_vae_int8_convrot.safetensors",
    "audioVae": "minimax_h3_audio_vae_fp32.safetensors",
    "acceleration": "none",
}

DEFAULT_TONE: dict[str, object] = {
    "mode": "Reinhard",
    "exposure": 0.09,
    "contrast": 0.84,
    "pivot": 0.69,
    "highlight_rolloff": 0.27,
    "shadow_lift": 0.15,
    "black_floor": 0.03,
    "saturation": 1.05,
    "strength": 0.2,
}

TONE_MODES = ("Filmic Soft", "Filmic Strong", "ACES Approx", "Reinhard", "Cineon-ish")
TONE_RANGES: dict[str, tuple[float, float]] = {
    "exposure": (-5.0, 5.0),
    "contrast": (0.0, 3.0),
    "pivot": (0.01, 1.0),
    "highlight_rolloff": (0.0, 2.0),
    "shadow_lift": (-1.0, 1.0),
    "black_floor": (0.0, 0.25),
    "saturation": (0.0, 3.0),
    "strength": (0.0, 2.0),
}

SETUP_PAYLOADS: dict[str, dict[str, object]] = {
    "mask": {"kind": "mask", "hatFix": True, "margin": "normal", "target": "person"},
    "people": {"kind": "people", "resolution": "736p"},
    "fit": {"kind": "fit", "resolution": "736p"},
    "color": {"kind": "color", "settings": {"detail": -0.2, "tone": dict(DEFAULT_TONE)}},
    "simple": {
        "kind": "simple",
        "settings": {
            "role": "",
            "resolution": "736p",
            "motion": 768,
            "hatFix": True,
            "noLipSync": True,
            "margin": "normal",
            "target": "person",
            "seed": 0,
            "clipSeconds": 6,
        },
        "model": DEFAULT_MODEL_SETTINGS,
    },
    "advanced": {
        "kind": "advanced",
        "pictures": 2,
        "settings": {
            "roles": ["", ""],
            "resolution": "736p",
            "motion": 768,
            "hatFix": True,
            "noLipSync": True,
            "margin": "normal",
            "seed": 0,
            "clipSeconds": 6,
        },
        "model": DEFAULT_MODEL_SETTINGS,
    },
    "object": {
        "kind": "object",
        "picture": True,
        "settings": {
            "mode": "change",
            "target": "jacket",
            "changeTo": "leather jacket",
            "biggestOnly": True,
            "keepOutline": False,
            "resolution": "736p",
            "motion": 768,
            "margin": "normal",
            "seed": 0,
            "clipSeconds": 6,
        },
        "model": DEFAULT_MODEL_SETTINGS,
    },
    "montage": {
        "kind": "montage",
        "pictures": 2,
        "review": None,
        "settings": {
            "roles": ["", ""],
            "resolution": "736p",
            "motion": 768,
            "hatFix": True,
            "noLipSync": True,
            "margin": "normal",
            "seed": 0,
            "clipSeconds": 6,
        },
        "model": DEFAULT_MODEL_SETTINGS,
    },
    "face": {
        "kind": "face",
        "settings": {"style": "mouth", "seed": 0, "chunkSeconds": 6.5},
        "model": DEFAULT_MODEL_SETTINGS,
    },
}

MODE_CHOICES = ("all", *SETUP_PAYLOADS.keys())
RESOLUTION_PRESETS = ("480p", "576p", "736p", "768p", "1080p")
MOTION_CHOICES = (512, 768, 1376)
TERMINAL_STATUSES = frozenset(("succeeded", "failed", "canceled", "expired"))
TRANSIENT_HTTP_CODES = frozenset((429, 502, 503, 504, 530))

_SPEED_LORA_RE = re.compile(
    r"dmad|light[ _-]?x2v|turbo|(?:^|[/ _-])(?:[12468][ _-]?steps?|lightning)(?:[_. -]|$)",
    re.IGNORECASE,
)
_VIDEO_EXT_MIME = {
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".avi": "video/x-msvideo",
}
_IMAGE_EXT_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def is_speed_lora(name: str) -> bool:
    return bool(_SPEED_LORA_RE.search(name.replace("\\", "/")))


def validate_resolution(value: str) -> str:
    clean = (value or "").strip()
    if clean in RESOLUTION_PRESETS:
        return clean
    match = re.fullmatch(r"(\d+)x(\d+)", clean)
    if match:
        w = round(int(match.group(1)) / 32) * 32
        h = round(int(match.group(2)) / 32) * 32
        if min(w, h) >= 256 and max(w, h) <= 2048:
            return clean
    raise UsageError(f"Unsupported resolution {value!r}; choose one of {RESOLUTION_PRESETS} or WIDTHxHEIGHT within 256-2048 (e.g. 736x1312)")


def validate_section(start: float, length: float) -> tuple[float, float]:
    if start < 0:
        raise UsageError("--start must be >= 0 seconds")
    if length < 0 or (0 < length < 2.0):
        raise UsageError("--length must be 0 (full video) or >= 2.0 seconds")
    return start, length


def validate_clip_seconds(seconds: float, *, flag: str = "--clip-seconds") -> float:
    if not (2.5 <= seconds <= 15.0):
        raise UsageError(f"{flag} must be between 2.5 and 15.0 seconds")
    return seconds


def validate_seed(seed: int) -> int:
    if not (0 <= seed <= 0xFFFFFFFF):
        raise UsageError("--seed must be an integer between 0 and 4294967295")
    return seed


def validate_steps(steps: int) -> int:
    if not (1 <= steps <= 100):
        raise UsageError("--steps must be an integer between 1 and 100")
    return steps


def parse_lora_spec(spec: str) -> tuple[str, float]:
    raw = (spec or "").strip()
    if not raw:
        raise UsageError("The LoRA name cannot be empty")
    if ":" in raw:
        name, strength_str = raw.rsplit(":", 1)
        try:
            strength = float(strength_str)
        except ValueError:
            name, strength = raw, 1.0
    else:
        name, strength = raw, 1.0
    name = name.strip()
    if not name:
        raise UsageError(f"Invalid LoRA argument: {spec!r}")
    if not (-4.0 <= strength <= 4.0):
        raise UsageError(f"LoRA weight must be between -4 and 4: {spec!r}")
    return name, strength


def resolve_loras(
    *,
    loras: list[str] | None,
    extra_loras: list[str] | None,
    no_lora: bool,
    acceleration: str,
) -> list[tuple[str, float]]:
    if no_lora:
        if loras or extra_loras:
            raise UsageError("--no-lora cannot be combined with --lora or --extra-lora")
        resolved: list[tuple[str, float]] = []
    elif loras:
        resolved = [parse_lora_spec(item) for item in loras]
        if extra_loras:
            resolved.extend(parse_lora_spec(item) for item in extra_loras)
    else:
        resolved = [] if acceleration == "spectrum" else [DEFAULT_LORA]
        if extra_loras:
            resolved.extend(parse_lora_spec(item) for item in extra_loras)
    if len(resolved) > 4:
        raise UsageError("At most 4 LoRAs can be loaded at once")
    if acceleration == "spectrum" and any(strength != 0 and is_speed_lora(name) for name, strength in resolved):
        raise UsageError("--acceleration spectrum cannot be combined with acceleration LoRAs (such as DMAD); use --no-lora or remove the acceleration LoRA")
    return resolved


def validate_media_file(path_str: str, *, kind: str) -> tuple[Path, str]:
    path = Path(path_str).expanduser().resolve()
    if not path.is_file():
        raise UsageError(f"File not found: {path}")
    size = path.stat().st_size
    if size <= 0:
        raise UsageError(f"File is empty: {path}")
    if size > 500 * 1024 * 1024:
        raise UsageError(f"File exceeds the 500 MB upload limit: {path}")
    with path.open("rb") as fh:
        head = fh.read(12)
    ext = path.suffix.lower()
    if kind == "video":
        is_mp4_mov = len(head) >= 8 and head[4:8] == b"ftyp"
        is_mkv_webm = head.startswith(b"\x1a\x45\xdf\xa3")
        is_avi = len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"AVI "
        if not (is_mp4_mov or is_mkv_webm or is_avi):
            raise UsageError(f"Unsupported video format (MP4/MOV/MKV/WebM/AVI required): {path}")
        mime = _VIDEO_EXT_MIME.get(ext) or mimetypes.guess_type(path.name)[0]
        if not mime or not mime.startswith("video/"):
            mime = "video/mp4"
        return path, mime
    if kind == "image":
        is_png = head.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpeg = head.startswith(b"\xff\xd8\xff")
        is_webp = len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP"
        if not (is_png or is_jpeg or is_webp):
            raise UsageError(f"Unsupported image format (PNG/JPEG/WebP required): {path}")
        if is_png:
            mime = "image/png"
        elif is_jpeg:
            mime = "image/jpeg"
        else:
            mime = _IMAGE_EXT_MIME.get(ext, "image/webp")
        return path, mime
    raise ValueError(f"Unsupported media kind: {kind}")


def _safe_part_filename(name: str) -> str:
    return name.replace("\\", "_").replace('"', "_").replace("\r", "_").replace("\n", "_")


class OpenjutsuClient:
    """Authenticated HTTP/multipart/SSE client for Openjutsu's Next.js API behind Caddy."""

    def __init__(self, url: str, token: str):
        self.url = url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": "Mozilla/5.0",
        }
        self.opener = urllib.request.build_opener(SameOriginRedirect())

    def open(
        self,
        path: str,
        *,
        method: str = "GET",
        data: object = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = 60.0,
    ):
        req = urllib.request.Request(
            self.url + path,
            data=data,
            headers={**self.headers, **(headers or {})},
            method=method,
        )
        try:
            return self.opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as err:
            raw = err.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw)
                detail = parsed.get("error") or raw
            except json.JSONDecodeError:
                detail = raw.strip() or err.reason
            err_out = CliError(f"[{SERVICE.name}] HTTP {err.code} ({path}): {detail}")
            err_out.http_status = err.code  # type: ignore[attr-defined]
            err_out.retry_after = err.headers.get("Retry-After")  # type: ignore[attr-defined]
            raise err_out from None

    def request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        body: dict[str, object] | None = None,
        timeout: float | None = 60.0,
    ) -> dict[str, object]:
        headers = {"Accept": "application/json"}
        payload = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            payload = json.dumps(body).encode("utf-8")
        with self.open(path, method=method, data=payload, headers=headers, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def submit_form(
        self,
        fields: list[tuple[str, str]],
        files: list[tuple[str, Path, str]],
        *,
        timeout: float | None = 300.0,
    ) -> dict[str, object]:
        boundary = "openjutsu-" + uuid.uuid4().hex
        total_bytes = sum(path.stat().st_size for _, path, _ in files)
        if total_bytes > 512 * 1024 * 1024:
            raise UsageError("The total size of all uploads in one request cannot exceed 512 MB")

        for attempt in range(4):
            with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as body:
                for key, val in fields:
                    safe_key = _safe_part_filename(key)
                    body.write(
                        (
                            f"--{boundary}\r\n"
                            f'Content-Disposition: form-data; name="{safe_key}"\r\n\r\n'
                            f"{val}\r\n"
                        ).encode("utf-8")
                    )
                for field_name, file_path, mime in files:
                    safe_key = _safe_part_filename(field_name)
                    safe_name = _safe_part_filename(file_path.name)
                    body.write(
                        (
                            f"--{boundary}\r\n"
                            f'Content-Disposition: form-data; name="{safe_key}"; filename="{safe_name}"\r\n'
                            f"Content-Type: {mime}\r\n\r\n"
                        ).encode("utf-8")
                    )
                    with file_path.open("rb") as src:
                        shutil.copyfileobj(src, body)
                    body.write(b"\r\n")
                body.write(f"--{boundary}--\r\n".encode("utf-8"))
                length = body.tell()
                body.seek(0)
                try:
                    with self.open(
                        "/api/jobs",
                        method="POST",
                        data=body,
                        headers={
                            "Accept": "application/json",
                            "Content-Type": f"multipart/form-data; boundary={boundary}",
                            "Content-Length": str(length),
                        },
                        timeout=timeout,
                    ) as resp:
                        result = json.loads(resp.read().decode("utf-8"))
                        if not isinstance(result, dict) or not isinstance(result.get("id"), str):
                            raise CliError("The server returned an invalid job response")
                        return result
                except CliError as err:
                    status = getattr(err, "http_status", None)
                    if status == 429 and attempt < 3:
                        wait_s = min(int(getattr(err, "retry_after", None) or 5), 15)
                        diagnostic(f"[{SERVICE.name}] The server is receiving another upload; retrying in {wait_s}s...")
                        Deadline(wait_s + 1).sleep(wait_s)
                        continue
                    raise
        raise CliError("Job submission failed: maximum retries exceeded")

    def job(self, job_id: str, *, timeout: float | None = 30.0) -> dict[str, object]:
        safe_id = urllib.parse.quote(job_id, safe="")
        return self.request_json(f"/api/jobs/{safe_id}", timeout=timeout)

    def cancel(self, job_id: str, *, timeout: float | None = 30.0) -> dict[str, object]:
        safe_id = urllib.parse.quote(job_id, safe="")
        return self.request_json(f"/api/jobs/{safe_id}", method="DELETE", timeout=timeout)

    def _consume_events_once(self, job_id: str, kind: str, deadline: Deadline, last_msg: list[str | None]) -> dict[str, object] | None:
        safe_id = urllib.parse.quote(job_id, safe="")
        query = urllib.parse.urlencode({"kind": kind})
        with self.open(
            f"/api/jobs/{safe_id}/events?{query}",
            headers={"Accept": "text/event-stream"},
            timeout=deadline.remaining(90),
        ) as resp:
            while True:
                raw_line = resp.readline()
                if not raw_line:
                    return None
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "progress":
                    node = event.get("node") or {}
                    title = node.get("title") or "Running"
                    clip = node.get("clip")
                    step, steps = event.get("step"), event.get("steps")
                    prefix = f"clip {clip} | " if clip else ""
                    step_txt = f" ({step}/{steps})" if step is not None and steps else ""
                    msg = f"[{SERVICE.name}] {prefix}{title}{step_txt}"
                    if msg != last_msg[0]:
                        diagnostic(msg)
                        last_msg[0] = msg
                elif etype == "status":
                    st = event.get("status")
                    qpos = event.get("queuePosition")
                    msg = f"[{SERVICE.name}] Status: {st}" + (f" (queued #{qpos})" if qpos is not None else "")
                    if msg != last_msg[0]:
                        diagnostic(msg)
                        last_msg[0] = msg
                elif etype == "job" and isinstance(event.get("job"), dict):
                    job_obj = event["job"]
                    if job_obj.get("status") in TERMINAL_STATUSES:
                        return job_obj
        return None

    def wait(
        self,
        job_id: str,
        *,
        kind: str = "simple",
        poll: float = 3.0,
        timeout: float = 0.0,
        use_events: bool = True,
    ) -> dict[str, object]:
        deadline = Deadline(timeout, f"Timed out waiting; the remote job keeps running: {job_id}")
        last_msg: list[str | None] = [None]
        sse_supported = use_events
        while True:
            state: dict[str, object] | None = None
            if sse_supported:
                try:
                    state = self._consume_events_once(job_id, kind, deadline, last_msg)
                except CliError as err:
                    code = getattr(err, "http_status", None)
                    if code == 404:
                        sse_supported = False
                    elif code not in TRANSIENT_HTTP_CODES:
                        raise
                except (urllib.error.URLError, TimeoutError, OSError):
                    pass

            if state is None:
                try:
                    state = self.job(job_id, timeout=deadline.remaining(30))
                except CliError as err:
                    if getattr(err, "http_status", None) not in TRANSIENT_HTTP_CODES:
                        raise
                    msg = f"[{SERVICE.name}] Connection unstable, reconnecting (the background job is not affected)..."
                    if msg != last_msg[0]:
                        diagnostic(msg)
                        last_msg[0] = msg
                    deadline.sleep(poll)
                    continue
                except (urllib.error.URLError, TimeoutError, OSError):
                    msg = f"[{SERVICE.name}] Connection unstable, reconnecting (the background job is not affected)..."
                    if msg != last_msg[0]:
                        diagnostic(msg)
                        last_msg[0] = msg
                    deadline.sleep(poll)
                    continue

            status = str(state.get("status") or "")
            if status == "succeeded":
                return state
            if status in ("failed", "canceled", "expired"):
                reason = state.get("error") or status
                raise CliError(f"Job {job_id} ended with status {status}: {reason}")

            qpos = state.get("queuePosition")
            msg = f"[{SERVICE.name}] Status: {status}" + (f" (queued #{qpos})" if qpos is not None else "")
            if msg != last_msg[0]:
                diagnostic(msg)
                last_msg[0] = msg
            if not sse_supported:
                deadline.sleep(poll)

    def download(self, state: dict[str, object], destination: str) -> str:
        if state.get("status") != "succeeded":
            raise UsageError("Job has not completed successfully yet; add --wait to wait for completion before downloading")
        video_info = state.get("video")
        data_url = state.get("data")
        rel_url: str | None = None
        if isinstance(video_info, dict) and isinstance(video_info.get("url"), str):
            rel_url = video_info["url"]
        elif isinstance(data_url, str):
            rel_url = data_url
        if not rel_url:
            raise CliError(f"Job {state.get('id')} returned no downloadable output file")
        sep = "&" if "?" in rel_url else "?"
        dl_path = f"{rel_url}{sep}download=1"
        output = Path(destination).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix="." + output.name + ".",
                suffix=".part",
                delete=False,
            ) as target:
                temporary = Path(target.name)
                with self.open(dl_path, timeout=300) as resp:
                    shutil.copyfileobj(resp, target)
            temporary.replace(output)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return str(output)


def _resolve_client(args: argparse.Namespace) -> OpenjutsuClient:
    explicit_token = getattr(args, "api_token", None) or os.environ.get("OPENJUTSU_API_TOKEN") or os.environ.get("OPENJUTSU_TOKEN")
    url, token = resolve_endpoint(
        State(SERVICE).load(),
        getattr(args, "url", None),
        explicit_token,
        url_env="OPENJUTSU_URL",
    )
    return OpenjutsuClient(url, token)


def _add_endpoint_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--url", help="override the service URL (default: current instance state or $OPENJUTSU_URL)")
    parser.add_argument("--api-token", "--token", dest="api_token",
                        help="Bearer API token (default: current instance state or $OPENJUTSU_API_TOKEN)")


def _add_wait_download_flags(parser: argparse.ArgumentParser, *, include_output: bool = True) -> None:
    parser.add_argument("--wait", action="store_true",
                        help="wait for the job to finish and show live segment/sampling-step progress (Ctrl+C only stops waiting locally)")
    if include_output:
        parser.add_argument("-o", "--output", help="download the finished video to a local path (implies waiting)")
    parser.add_argument("--poll", type=positive_seconds, default=3.0, help="fallback polling interval in seconds (default: 3)")
    parser.add_argument("--timeout", type=nonnegative_seconds, default=0.0,
                        help="maximum wait in seconds; 0 waits until completion (default: 0)")
    parser.add_argument("--no-events", action="store_true", help="disable the live SSE progress stream and use polling only")


def _add_section_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=nonnegative_seconds, default=0.0, help="start offset in the source video, in seconds (default: 0)")
    parser.add_argument("--length", type=nonnegative_seconds, default=0.0,
                        help="duration to take from the source video, in seconds; 0 processes to the end (default: 0; if non-zero must be >= 2)")


def _add_model_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=6, help="diffusion sampling steps 1-100 (default: 6; the DMAD 4-step LoRA is loaded by default, 4 is recommended for speed)")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_SETTINGS["model"]), help="H3 diffusion base model file name")
    parser.add_argument("--lora", action="append", dest="loras", metavar="NAME[:STRENGTH]",
                        help="replace the default LoRA list (repeatable up to 4 times, e.g. drive/my.safetensors:0.8)")
    parser.add_argument("--extra-lora", action="append", dest="extra_loras", metavar="NAME[:STRENGTH]",
                        help="add an extra LoRA on top of the default DMAD acceleration LoRA (e.g. drive/style.safetensors:0.7)")
    parser.add_argument("--no-lora", action="store_true", help="do not load any LoRA")
    parser.add_argument("--text-encoder", default=str(DEFAULT_MODEL_SETTINGS["textEncoder"]), help="H3 text encoder file name")
    parser.add_argument("--describer", help="optional visual describer model file name (models/text_encoders)")
    parser.add_argument("--video-vae", default=str(DEFAULT_MODEL_SETTINGS["videoVae"]), help="video VAE file name")
    parser.add_argument("--audio-vae", default=str(DEFAULT_MODEL_SETTINGS["audioVae"]), help="audio VAE file name")
    parser.add_argument("--acceleration", choices=("none", "spectrum"), default="none",
                        help="inference acceleration option (default: none; spectrum requires --no-lora)")


def _add_h3_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resolution", default="736p",
                        help="output resolution preset (480p|576p|736p|768p|1080p; ideally not above the source's short side) or WIDTHxHEIGHT (default: 736p)")
    parser.add_argument("--motion", type=int, choices=MOTION_CHOICES, default=512,
                        help="long-side pixels of the motion reference (default: 512; choices 512 / 768 / 1376)")
    parser.add_argument("--clip-seconds", type=float, default=6.0,
                        help="segment window in seconds 2.5-15 (default: 6.0; sweet spot 7-8s, >9s makes 3D full-attention cost grow quadratically)")
    parser.add_argument("--seed", type=int, default=0, help="random seed 0-4294967295 (default: 0)")


def _append_section_fields(fields: list[tuple[str, str]], args: argparse.Namespace) -> None:
    start, length = validate_section(args.start, args.length)
    if start > 0 or length > 0:
        fields.append(("start", f"{start:.2f}"))
        fields.append(("length", f"{length:.2f}"))


def _append_model_fields(fields: list[tuple[str, str]], args: argparse.Namespace) -> None:
    steps = validate_steps(args.steps)
    loras = resolve_loras(
        loras=args.loras,
        extra_loras=args.extra_loras,
        no_lora=args.no_lora,
        acceleration=args.acceleration,
    )
    fields.extend([
        ("steps", str(steps)),
        ("model", args.model),
        ("text_encoder", args.text_encoder),
        ("video_vae", args.video_vae),
        ("audio_vae", args.audio_vae),
        ("acceleration", args.acceleration),
    ])
    if args.describer:
        fields.append(("describer", args.describer))
    for name, strength in loras:
        fields.append(("lora", name))
        fields.append(("lora_strength", str(strength)))
        fields.append(("lora_speed", "true" if is_speed_lora(name) else "false"))


def build_submission(args: argparse.Namespace) -> tuple[str, list[tuple[str, str]], list[tuple[str, Path, str]]]:
    """Validate CLI arguments locally and build `(sse_kind, form_fields, form_files)` before network upload."""
    cmd = args.cmd
    video_path, video_mime = validate_media_file(args.video, kind="video")
    files: list[tuple[str, Path, str]] = [("video", video_path, video_mime)]
    fields: list[tuple[str, str]] = []

    if cmd == "swap":
        if not args.pictures or len(args.pictures) > 4:
            raise UsageError("Provide 1 to 4 character reference images with -p/--picture")
        pic_entries = [validate_media_file(p, kind="image") for p in args.pictures]
        roles = list(args.roles or [])
        if len(roles) > len(pic_entries):
            raise UsageError(f"The number of --role values ({len(roles)}) cannot exceed the number of reference images ({len(pic_entries)})")
        while len(roles) < len(pic_entries):
            roles.append("")

        if args.montage:
            if any(not r.strip() for r in roles):
                raise UsageError("--montage mode requires a non-empty --role for every reference image (e.g. 'dancer - white shirt')")
            if args.max_quality:
                raise UsageError("--max-quality only applies to the two-person advanced mode and cannot be combined with --montage")
            mode, sse_kind = "montage", "montage"
        elif len(pic_entries) == 1 and not args.max_quality:
            mode, sse_kind = "simple", "simple"
        else:
            if args.max_quality and len(pic_entries) != 2:
                raise UsageError("--max-quality requires exactly 2 character reference images")
            mode = "advanced"
            sse_kind = "advanced_maxq" if args.max_quality else "advanced"

        resolution = validate_resolution(args.resolution)
        clip_seconds = validate_clip_seconds(args.clip_seconds)
        seed = validate_seed(args.seed)

        fields.append(("mode", mode))
        _append_section_fields(fields, args)
        fields.extend([
            ("hat_fix", "true" if args.hat_fix else "false"),
            ("no_lip_sync", "true" if args.no_lip_sync else "false"),
            ("margin", args.margin),
            ("target", args.target),
            ("resolution", resolution),
            ("motion", str(args.motion)),
            ("seed", str(seed)),
            ("clip_seconds", str(clip_seconds)),
        ])
        if mode == "simple":
            files.append(("picture", pic_entries[0][0], pic_entries[0][1]))
            fields.append(("role", roles[0]))
        else:
            for (pic_path, pic_mime), role in zip(pic_entries, roles):
                files.append(("picture", pic_path, pic_mime))
                fields.append(("role", role))
            if mode == "advanced":
                fields.append(("max_quality", "true" if args.max_quality else "false"))
        _append_model_fields(fields, args)
        return sse_kind, fields, files

    if cmd == "object":
        target = (args.target or "").strip()
        if not target:
            raise UsageError("--target cannot be empty")
        change_to = (args.change_to or "").strip()
        pic_entry = validate_media_file(args.picture, kind="image") if args.picture else None
        if args.edit_mode != "remove" and not change_to and not pic_entry:
            raise UsageError(f"Edit mode {args.edit_mode!r} requires a --change-to description or a -p/--picture reference image (or both)")
        if args.edit_mode == "add" and not args.placement:
            raise UsageError("--edit-mode add requires a placement-region JSON via --placement, e.g. '[{\"frame\":0,\"x\":0.25,\"y\":0.2,\"width\":0.5,\"height\":0.6}]'")
        if args.placement:
            try:
                parsed_pl = json.loads(args.placement)
            except json.JSONDecodeError as exc:
                raise UsageError(f"--placement is not valid JSON: {exc}") from None
            if not isinstance(parsed_pl, list) or not parsed_pl:
                raise UsageError("--placement must be a non-empty array of keyframes")

        resolution = validate_resolution(args.resolution)
        clip_seconds = validate_clip_seconds(args.clip_seconds)
        seed = validate_seed(args.seed)

        fields.append(("mode", "object"))
        _append_section_fields(fields, args)
        fields.extend([
            ("edit_mode", args.edit_mode),
            ("target", target),
            ("change_to", change_to),
            ("margin", args.margin),
            ("biggest_only", "true" if args.biggest_only else "false"),
            ("keep_outline", "true" if args.keep_outline else "false"),
            ("resolution", resolution),
            ("motion", str(args.motion)),
            ("seed", str(seed)),
            ("clip_seconds", str(clip_seconds)),
        ])
        if pic_entry and args.edit_mode != "remove":
            files.append(("picture", pic_entry[0], pic_entry[1]))
        if args.edit_mode == "add":
            fields.append(("placement", args.placement))
            fields.append(("action", (args.action or "").strip()))
        _append_model_fields(fields, args)
        return "object", fields, files

    if cmd == "mask":
        fields.append(("mode", "mask"))
        _append_section_fields(fields, args)
        fields.extend([
            ("hat_fix", "true" if args.hat_fix else "false"),
            ("margin", args.margin),
            ("target", args.target),
        ])
        return "mask", fields, files

    if cmd == "face":
        pic_path, pic_mime = validate_media_file(args.picture, kind="image")
        files.append(("picture", pic_path, pic_mime))
        chunk_seconds = validate_clip_seconds(args.clip_seconds, flag="--clip-seconds")
        seed = validate_seed(args.seed)
        fields.append(("mode", "face"))
        _append_section_fields(fields, args)
        fields.extend([
            ("style", args.style),
            ("seed", str(seed)),
            ("clip_seconds", str(chunk_seconds)),
            ("motion", "512"),
        ])
        _append_model_fields(fields, args)
        return "face", fields, files

    if cmd == "color":
        if args.no_detail and args.no_tone:
            raise UsageError("--no-detail and --no-tone cannot both be set (enable at least one grading step)")
        fields.append(("mode", "color"))
        _append_section_fields(fields, args)
        if not args.no_detail:
            if not (-1.0 <= args.detail <= 1.0):
                raise UsageError("--detail must be between -1.0 and 1.0")
            fields.append(("detail", str(args.detail)))
        if not args.no_tone:
            fields.append(("tone_mode", args.tone_mode))
            for key, (low, high) in TONE_RANGES.items():
                val = getattr(args, f"tone_{key}")
                if not (low <= val <= high):
                    raise UsageError(f"--tone-{key.replace('_', '-')} must be between {low} and {high}")
                fields.append((f"tone_{key}", str(val)))
        return "color", fields, files

    raise ValueError(f"Unsupported submission command: {cmd}")


def check_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(prog="colabllm openjutsu check")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="all",
                        help="Openjutsu workflow mode to verify via /api/setup (default: all)")
    parser.add_argument("--profile", choices=("full", "core"), default="full",
                        help="expected model profile when --mode=all (default: full)")
    _add_endpoint_flags(parser)
    parser.add_argument("--timeout", type=positive_seconds, default=60, help="HTTP timeout seconds (default: 60)")
    return parser


def custom_parser(
    parser: argparse.ArgumentParser | None = None,
    commands: argparse._SubParsersAction | None = None,
) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        prog="colabllm openjutsu",
        description="Openjutsu: submit motion transfer, local video edits, mask previews, face/color fixes, and query/download jobs from the command line",
    )
    commands = commands or parser.add_subparsers(dest="cmd", required=True)

    # 1. swap (simple / advanced / montage)
    swap = commands.add_parser("swap", help="character motion transfer (1 reference image uses simple, 2-4 use advanced, --montage uses multi-person editing)")
    swap.add_argument("--video", required=True, help="local source video path (MP4/MOV/MKV/WebM/AVI, <= 500 MB)")
    swap.add_argument("-p", "--picture", action="append", dest="pictures", required=True,
                      help="local character reference image path (PNG/JPEG/WebP; repeat for 1-4 images)")
    swap.add_argument("--role", action="append", dest="roles",
                      help="description of the person each reference image replaces (e.g. 'left dancer'; required in --montage mode)")
    swap.add_argument("--montage", action="store_true", help="enable Montage multi-shot automatic person tracking and matching")
    swap.add_argument("--max-quality", action="store_true", help="in two-person mode, render each person in a separate pass (highest quality)")
    swap.add_argument("--target", default="person", help="SAM 3.1 tracking target class (default: person)")
    swap.add_argument("--margin", choices=("tight", "normal", "loose"), default="normal",
                      help="mask edge expansion width (default: normal)")
    swap.add_argument("--hat-fix", action="store_true", help="enable mask protection for hat/headwear regions")
    swap.add_argument("--no-lip-sync", action="store_true", help="disable lip sync (the person does not sing along / mouth the audio)")
    _add_section_flags(swap)
    _add_h3_run_flags(swap)
    _add_model_flags(swap)
    _add_wait_download_flags(swap)
    _add_endpoint_flags(swap)

    # 2. object (change / remove / background / add)
    obj = commands.add_parser("object", help="local video editing (replace clothing/props: change / remove objects: remove / swap background: background / add objects: add)")
    obj.add_argument("--video", required=True, help="local source video path")
    obj.add_argument("--edit-mode", choices=("change", "remove", "background", "add"), default="change",
                     help="local edit mode (default: change)")
    obj.add_argument("--target", required=True,
                     help="tracking target (change/remove: the object to modify such as 'jacket'; background: the subject to keep such as 'person'; add: the location such as 'on the table')")
    obj.add_argument("--change-to", default="", help="what the target becomes or what to add (e.g. 'black leather jacket')")
    obj.add_argument("-p", "--picture", help="optional appearance reference image path (PNG/JPEG/WebP)")
    obj.add_argument("--margin", choices=("tight", "normal", "loose"), default="normal", help="mask edge width (default: normal)")
    obj.add_argument("--biggest-only", action="store_true", help="when several targets of the same class appear, edit only the largest one")
    outline_grp = obj.add_mutually_exclusive_group()
    outline_grp.add_argument("--keep-outline", dest="keep_outline", action="store_true", default=True,
                             help="keep the original object's outline as motion guidance (default: on)")
    outline_grp.add_argument("--no-keep-outline", dest="keep_outline", action="store_false",
                             help="do not keep the original object's outline (suits outfit swaps with very different cuts)")
    obj.add_argument("--placement", help="required in add mode: JSON array of normalized placement-box keyframes")
    obj.add_argument("--action", default="", help="optional in add mode: description of the added object's action")
    _add_section_flags(obj)
    _add_h3_run_flags(obj)
    _add_model_flags(obj)
    _add_wait_download_flags(obj)
    _add_endpoint_flags(obj)

    # 3. mask (fast SAM 3.1 mask preview)
    mask = commands.add_parser("mask", help="quickly generate a SAM 3.1 tracking mask preview video")
    mask.add_argument("--video", required=True, help="local source video path")
    mask.add_argument("--target", default="person", help="tracking target (default: person)")
    mask.add_argument("--margin", choices=("tight", "normal", "loose"), default="normal", help="mask edge width (default: normal)")
    mask.add_argument("--hat-fix", action="store_true", help="enable hat/headwear mask protection")
    _add_section_flags(mask)
    _add_wait_download_flags(mask)
    _add_endpoint_flags(mask)

    # 4. face (Fix faces post-processing)
    face = commands.add_parser("face", help="face refinement repaint post-processing for a finished video (requires the full model tier)")
    face.add_argument("--video", required=True, help="path of the video whose faces should be fixed")
    face.add_argument("-p", "--picture", required=True, help="clear frontal face reference image path (PNG/JPEG/WebP)")
    face.add_argument("--style", choices=("photo", "mouth"), default="photo",
                      help="face blending mode: photo matches the photo's features (default), mouth strictly follows the source video's mouth shapes")
    face.add_argument("--clip-seconds", type=float, default=6.5, help="refinement chunk length in seconds 2.5-15 (default: 6.5)")
    face.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    _add_section_flags(face)
    _add_model_flags(face)
    _add_wait_download_flags(face)
    _add_endpoint_flags(face)

    # 5. color (Fix colours post-processing)
    color = commands.add_parser("color", help="detail softening and Orion4D DCTL color grading post-processing for a finished video")
    color.add_argument("--video", required=True, help="path of the video to grade")
    color.add_argument("--detail", type=float, default=-0.2, help="detail contrast -1.0 to 1.0 (default: -0.2 softens high-frequency noise)")
    color.add_argument("--no-detail", action="store_true", help="disable detail contrast adjustment")
    color.add_argument("--tone-mode", choices=TONE_MODES, default=str(DEFAULT_TONE["mode"]),
                       help="DCTL tone-mapping curve (default: Reinhard)")
    color.add_argument("--no-tone", action="store_true", help="disable DCTL tone mapping")
    for key, val in DEFAULT_TONE.items():
        if key == "mode":
            continue
        color.add_argument(f"--tone-{key.replace('_', '-')}", dest=f"tone_{key}", type=float, default=float(val),
                           help=f"tone parameter {key} (default: {val})")
    _add_section_flags(color)
    _add_wait_download_flags(color)
    _add_endpoint_flags(color)

    # 6. job / download / cancel
    job = commands.add_parser("job", help="query job status; add --wait to follow progress live, -o to download the result")
    job.add_argument("job_id", help="job ID")
    job.add_argument("--kind", default="simple", help="workflow type for parsing SSE node titles (default: simple)")
    _add_wait_download_flags(job)
    _add_endpoint_flags(job)

    dl = commands.add_parser("download", help="download the output of a finished job (add --wait to wait for completion first)")
    dl.add_argument("job_id", help="job ID")
    dl.add_argument("--kind", default="simple", help="workflow type while waiting (default: simple)")
    _add_wait_download_flags(dl)
    _add_endpoint_flags(dl)

    cancel = commands.add_parser("cancel", help="cancel a queued or running job")
    cancel.add_argument("job_id", help="job ID")
    _add_endpoint_flags(cancel)

    check_parser(commands.add_parser("check", help="verify /api/models and /api/setup readiness"))
    return parser


def _cmd_check(raw_args: list[str] | argparse.Namespace) -> None:
    """Verify Openjutsu model inventory (`/api/models`) and ComfyUI custom-node/model readiness (`/api/setup`)."""
    args = raw_args if isinstance(raw_args, argparse.Namespace) else check_parser().parse_args(raw_args)
    deadline = Deadline(args.timeout)
    client = _resolve_client(args)
    models = client.request_json("/api/models", timeout=deadline.remaining(30))
    if args.mode == "all":
        modes = [m for m in SETUP_PAYLOADS if args.profile == "full" or m != "face"]
    else:
        modes = [args.mode]
    mode_results: dict[str, dict[str, object]] = {}
    all_ok = not bool(models.get("error"))
    for mode in modes:
        res = client.request_json(
            "/api/setup",
            method="POST",
            body=SETUP_PAYLOADS[mode],
            timeout=deadline.remaining(30),
        )
        mode_results[mode] = res
        if not res.get("ok"):
            all_ok = False
    report = {
        "ok": all_ok,
        "url": client.url,
        "models": models,
        "modes": mode_results,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not all_ok:
        raise CliError(f"[{SERVICE.name}] setup check reported missing models or custom node packs")


def _run_business_command(raw_args: list[str] | argparse.Namespace, cmd_name: str) -> None:
    if isinstance(raw_args, argparse.Namespace):
        args = raw_args
    else:
        args = custom_parser().parse_args([cmd_name, *raw_args])

    if args.cmd == "cancel":
        client = _resolve_client(args)
        print(json.dumps(client.cancel(args.job_id), indent=2, ensure_ascii=False))
        return

    if args.cmd in ("swap", "object", "mask", "face", "color"):
        sse_kind, fields, files = build_submission(args)
        client = _resolve_client(args)
        diagnostic(f"[{SERVICE.name}] Uploading assets and submitting the {args.cmd} job...")
        submitted = client.submit_form(fields, files)
        job_id = str(submitted["id"])
        diagnostic(f"[{SERVICE.name}] Job submitted: {job_id}")
        if not args.wait and not args.output:
            print(json.dumps(submitted, indent=2, ensure_ascii=False))
            return
    else:
        client = _resolve_client(args)
        job_id = args.job_id
        sse_kind = getattr(args, "kind", "simple") or "simple"

    should_wait = bool(args.wait or (args.output and args.cmd != "download"))
    state = (
        client.wait(
            job_id,
            kind=sse_kind,
            poll=args.poll,
            timeout=args.timeout,
            use_events=not args.no_events,
        )
        if should_wait
        else client.job(job_id)
    )
    if args.output or args.cmd == "download":
        default_ext = ".mp4" if state.get("video") or not state.get("data") else ".json"
        dest = args.output or f"{job_id}{default_ext}"
        saved = client.download(state, dest)
        state = {**state, "local_output": saved}
        diagnostic(f"[{SERVICE.name}] Saved locally: {saved}")
    print(json.dumps(state, indent=2, ensure_ascii=False))


def _make_handler(cmd_name: str):
    return lambda raw_args: _run_business_command(raw_args, cmd_name)


BUSINESS_COMMANDS = ("swap", "object", "mask", "face", "color", "job", "download", "cancel")


def configure_cli(parser, commands, up) -> None:
    up.add_argument(
        "--profile",
        choices=("full", "core"),
        default="full",
        help="model bundle to stage onto local SSD: full (~80.7 GB, all modes) or core (~50.8 GB, H3 swap/edit/montage/color)",
    )
    up.add_argument(
        "--chains-storage",
        choices=("ssd", "drive"),
        default="ssd",
        help="where to store H3 Context-Loop per-clip checkpoints (~450 MB/clip): ssd (default, fast bind-mount) or drive",
    )
    up.add_argument(
        "--vram-mode",
        choices=("auto", "highvram", "dynamic", "gpu-only"),
        default="auto",
        help="ComfyUI VRAM policy: auto enables --highvram on >=76 GiB GPUs such as G4 (default: auto)",
    )
    custom_parser(parser, commands)


_BASE_ENV = dict(SERVICE.env)


def prepare_up(args) -> None:
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    SERVICE.env["CT_OPENJUTSU_PROFILE"] = getattr(args, "profile", None) or "full"
    SERVICE.env["CT_OPENJUTSU_CHAINS_STORAGE"] = getattr(args, "chains_storage", None) or "ssd"
    SERVICE.env["CT_COMFY_VRAM_MODE"] = getattr(args, "vram_mode", None) or "auto"


HANDLERS = {
    "check": _cmd_check,
    **{name: _make_handler(name) for name in BUSINESS_COMMANDS},
}


if __name__ == "__main__":
    from cli_support import run_cli
    run_cli(lambda: main(SERVICE, configure=configure_cli, prepare_up=prepare_up, handlers=HANDLERS))
