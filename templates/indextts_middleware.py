"""Authenticated ASGI middleware and Google Drive voice endpoints for IndexTTS-2.5."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

from colab_asgi_common import (
    audio_path_to_wav_data_url,
    build_multipart_form,
    list_drive_audio_files,
    resolve_drive_path,
    sanitize_header_param,
    save_output_audio_to_drive,
    tail_service_log,
    validate_voice_name,
    verify_bearer_auth,
)

try:
    from starlette.datastructures import Headers
    from starlette.responses import JSONResponse
    from starlette.types import ASGIApp, Message, Receive, Scope, Send
except ImportError:  # pragma: no cover - local unit-test fallback without starlette
    Headers = dict  # type: ignore[assignment,misc]
    JSONResponse = None  # type: ignore[assignment,misc]
    ASGIApp = Message = Receive = Scope = Send = Any  # type: ignore[assignment,misc]

API_TOKEN = os.environ.get("CT_API_TOKEN", "")
SERVICE_PORT = int(os.environ.get("CT_SERVICE_PORT", "8000"))
DATA_DIR = Path(os.environ.get("CT_DATA_DIR", "/content/drive/MyDrive/colab-llm/indextts"))
SPEAKERS_DIR = Path(os.environ.get("SPEAKER_SAMPLES_DIR", "/content/indextts/speakers"))
DEFAULT_REF_WAV = Path(os.environ.get("CT_DEFAULT_REF_WAV", "/content/indextts/default_ref.wav"))
DRIVE_ROOT = Path("/content/drive")
MYDRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_VOICES_DIR = DATA_DIR / "voices"
DRIVE_OUTPUT_DIR = DATA_DIR / "outputs"

SUPPORTED_LANGS = {
    "zh": "Chinese (Mandarin)",
    "en": "English",
    "zhen": "Chinese-English code-switching",
    "ja": "Japanese",
    "es": "Spanish",
    "ar": "Arabic",
    "yue": "Cantonese",
}
EMOTION_LABELS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]


def _verify_auth(headers: Any) -> bool:
    return verify_bearer_auth(headers, API_TOKEN)


def _resolve_path(raw_spec: str, default_subdir: str | None = "voices") -> Path:
    return resolve_drive_path(
        raw_spec,
        data_dir=DATA_DIR,
        default_subdir=default_subdir,
        mydrive_root=MYDRIVE_ROOT,
        drive_root=DRIVE_ROOT,
    )


def _sync_speakers_to_drive() -> None:
    try:
        DRIVE_VOICES_DIR.mkdir(parents=True, exist_ok=True)
        if SPEAKERS_DIR.is_dir():
            for sf in SPEAKERS_DIR.glob("*.safetensors"):
                dst = DRIVE_VOICES_DIR / sf.name
                if not dst.is_file() or sf.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(sf, dst)
    except Exception:
        pass


def _rewrite_speech_request_body(raw_body: bytes) -> tuple[bytes, str | None]:
    if not raw_body:
        return raw_body, None
    try:
        payload = json.loads(raw_body)
    except Exception:
        return raw_body, None
    if not isinstance(payload, dict):
        return raw_body, None

    changed = False
    save_drive_spec: str | None = None

    extras = payload.get("extra_params")
    if not isinstance(extras, dict):
        extras = {}

    for top_key in ("lang", "text_normalization", "emo_audio", "emo_vector", "emo_alpha", "emo_text", "use_emo_text", "use_random"):
        if top_key in payload and top_key not in extras:
            extras[top_key] = payload.pop(top_key)
            changed = True

    for save_key in ("save_to_drive", "save_drive_path"):
        val = payload.pop(save_key, None)
        if val is None and save_key in extras:
            val = extras.pop(save_key, None)
            changed = True
        elif val is not None:
            changed = True
        if val:
            save_drive_spec = str(val) if not isinstance(val, bool) else "auto"

    # Resolve drive://, file:// (Drive-only), or bare Drive paths in ref_audio
    ref_audio = payload.get("ref_audio")
    if isinstance(ref_audio, str) and ref_audio.strip():
        ra = ref_audio.strip()
        if not ra.startswith(("http://", "https://", "data:")):
            ap = _resolve_path(ra)
            payload["ref_audio"] = audio_path_to_wav_data_url(ap)
            changed = True
    elif ref_audio is None and (not payload.get("voice") or str(payload.get("voice")).strip() == "default"):
        if DEFAULT_REF_WAV.is_file():
            payload["ref_audio"] = f"file://{DEFAULT_REF_WAV}"
            payload.pop("voice", None)
            changed = True

    # Resolve drive://, file:// (Drive-only), or bare Drive paths in extra_params.emo_audio
    emo_audio = extras.get("emo_audio")
    if isinstance(emo_audio, str) and emo_audio.strip():
        ea = emo_audio.strip()
        if not ea.startswith(("http://", "https://", "data:")):
            ap = _resolve_path(ea)
            extras["emo_audio"] = audio_path_to_wav_data_url(ap)
            changed = True

    if extras:
        payload["extra_params"] = extras
    elif "extra_params" in payload and not extras:
        payload.pop("extra_params", None)

    if not changed:
        return raw_body, save_drive_spec
    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), save_drive_spec


def _upload_drive_voice_sync(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    raw_name = str(body.get("name") or "").strip()
    use_default_ref = bool(body.get("use_default_ref", False))
    raw_path = str(body.get("drive_path") or body.get("audio") or body.get("path") or "").strip()
    consent = str(body.get("consent") or "user-consent").strip()
    ref_text = body.get("ref_text")
    speaker_desc = body.get("speaker_description")
    if not raw_name or (not raw_path and not use_default_ref):
        return 400, {"error": "Both 'name' and 'drive_path' are required"}
    try:
        name = validate_voice_name(raw_name)
    except ValueError as err:
        return 400, {"error": str(err)}

    if use_default_ref:
        ap = DEFAULT_REF_WAV
        if not ap.is_file():
            return 404, {"error": f"Default reference audio not found: {ap}"}
    else:
        try:
            ap = _resolve_path(raw_path)
        except ValueError as err:
            return 400, {"error": str(err)}
        if not ap.is_file():
            return 404, {"error": f"Audio file not found on Google Drive: {raw_path} (resolved to {ap})"}

    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-t", "24.0",
        "-i", str(ap), "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
        wav_bytes = res.stdout if (res.returncode == 0 and res.stdout) else ap.read_bytes()
    except OSError:
        wav_bytes = ap.read_bytes()

    fields: list[tuple[str, str]] = [("name", name), ("consent", consent)]
    if ref_text:
        fields.append(("ref_text", str(ref_text)))
    if speaker_desc:
        fields.append(("speaker_description", str(speaker_desc)))
    content_type, body_bytes = build_multipart_form(
        fields,
        [("audio_sample", f"{name}.wav", "audio/wav", wav_bytes)],
    )
    req = urllib.request.Request(
        f"http://127.0.0.1:{SERVICE_PORT}/v1/audio/voices",
        data=body_bytes,
        headers={
            "Content-Type": content_type,
            "Authorization": f"Bearer {API_TOKEN}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            out = json.loads(resp.read().decode("utf-8", errors="replace"))
            _sync_speakers_to_drive()
            return resp.status, out
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        try:
            return err.code, json.loads(raw_err)
        except Exception:
            return err.code, {"error": raw_err}


class IndexTTSMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        if scope.get("method") == "OPTIONS":
            await self.app(scope, receive, send)
            return

        root_path = scope.get("root_path", "")
        path = scope["path"].removeprefix(root_path)
        headers = Headers(scope=scope)

        if not _verify_auth(headers):
            resp = JSONResponse(
                {"error": {"message": "Unauthorized: invalid or missing Bearer token", "type": "AuthenticationError", "code": 401}},
                status_code=HTTPStatus.UNAUTHORIZED,
            )
            await resp(scope, receive, send)
            return

        if scope["type"] == "http":
            method = scope.get("method", "GET")

            if method == "GET" and path == "/api/ct/info":
                resp = JSONResponse({
                    "service": "indextts",
                    "model": "IndexTeam/IndexTTS-2.5",
                    "languages": SUPPORTED_LANGS,
                    "emotion_labels": EMOTION_LABELS,
                    "speed_range": [0.5, 2.0],
                    "default_ref_wav": str(DEFAULT_REF_WAV),
                    "drive_voices_dir": str(DRIVE_VOICES_DIR),
                    "drive_output_dir": str(DRIVE_OUTPUT_DIR),
                })
                await resp(scope, receive, send)
                return

            if method == "GET" and path == "/api/ct/logs":
                qs = parse_qs((scope.get("query_string") or b"").decode("utf-8", errors="replace"))
                try:
                    n_lines = int((qs.get("lines") or ["120"])[0])
                except ValueError:
                    n_lines = 120
                resp = JSONResponse(tail_service_log(lines=n_lines))
                await resp(scope, receive, send)
                return

            if method == "GET" and path == "/api/ct/drive-list":
                qs = parse_qs((scope.get("query_string") or b"").decode("utf-8", errors="replace"))
                spec = (qs.get("path") or [""])[0]
                recursive = (qs.get("recursive") or ["1"])[0] not in ("0", "false", "False")
                try:
                    files = await asyncio.to_thread(
                        list_drive_audio_files,
                        spec,
                        data_dir=DATA_DIR,
                        default_subdir="voices",
                        recursive=recursive,
                        mydrive_root=MYDRIVE_ROOT,
                        drive_root=DRIVE_ROOT,
                    )
                    resp = JSONResponse({
                        "drive_mounted": MYDRIVE_ROOT.is_dir(),
                        "query": spec,
                        "count": len(files),
                        "files": files,
                    })
                except Exception as e:
                    resp = JSONResponse({"error": str(e)}, status_code=400)
                await resp(scope, receive, send)
                return

            if method == "POST" and path == "/api/ct/drive-upload-voice":
                chunks: list[bytes] = []
                while True:
                    msg = await receive()
                    if msg["type"] == "http.request":
                        chunks.append(msg.get("body", b""))
                        if not msg.get("more_body", False):
                            break
                    elif msg["type"] == "http.disconnect":
                        return
                try:
                    body = json.loads(b"".join(chunks))
                    code, out = await asyncio.to_thread(_upload_drive_voice_sync, body)
                    resp = JSONResponse(out, status_code=code)
                except ValueError as e:
                    resp = JSONResponse({"error": str(e)}, status_code=400)
                except Exception as e:
                    resp = JSONResponse({"error": str(e)}, status_code=500)
                await resp(scope, receive, send)
                return

            if method == "POST" and path == "/v1/audio/speech":
                chunks = []
                while True:
                    msg = await receive()
                    if msg["type"] == "http.request":
                        chunks.append(msg.get("body", b""))
                        if not msg.get("more_body", False):
                            break
                    elif msg["type"] == "http.disconnect":
                        return
                raw_body = b"".join(chunks)
                try:
                    rewritten_body, save_drive_spec = await asyncio.to_thread(_rewrite_speech_request_body, raw_body)
                except Exception as e:
                    resp = JSONResponse(
                        {"error": {"message": str(e), "type": "BadRequestError", "code": 400}},
                        status_code=400,
                    )
                    await resp(scope, receive, send)
                    return

                header_save = headers.get("x-save-drive")
                if header_save and not save_drive_spec:
                    save_drive_spec = header_save

                body_sent = False

                async def replay_receive() -> Message:
                    nonlocal body_sent
                    if not body_sent:
                        body_sent = True
                        return {"type": "http.request", "body": rewritten_body, "more_body": False}
                    return await receive()

                if not save_drive_spec:
                    await self.app(scope, replay_receive, send)
                    return

                resp_status = 200
                resp_headers: list[tuple[bytes, bytes]] = []
                resp_chunks: list[bytes] = []
                start_msg: Message | None = None

                async def tee_send(message: Message) -> None:
                    nonlocal resp_status, resp_headers, start_msg
                    if message["type"] == "http.response.start":
                        resp_status = message.get("status", 200)
                        resp_headers = list(message.get("headers", []))
                        start_msg = message
                        if resp_status != 200:
                            await send(message)
                    elif message["type"] == "http.response.body":
                        chunk = message.get("body", b"")
                        more = message.get("more_body", False)
                        if resp_status != 200:
                            await send(message)
                            return
                        if chunk:
                            resp_chunks.append(chunk)
                        if not more and start_msg is not None:
                            full_audio = b"".join(resp_chunks)
                            saved_path = await asyncio.to_thread(
                                save_output_audio_to_drive,
                                full_audio,
                                save_drive_spec,
                                data_dir=DATA_DIR,
                                output_dir=DRIVE_OUTPUT_DIR,
                                prefix="indextts",
                                fmt="wav",
                                mydrive_root=MYDRIVE_ROOT,
                                drive_root=DRIVE_ROOT,
                            )
                            safe_hdr = sanitize_header_param(saved_path).encode("utf-8")
                            new_headers = [
                                (k, v) for (k, v) in resp_headers if k.lower() != b"x-saved-drive-path"
                            ]
                            new_headers.append((b"x-saved-drive-path", safe_hdr))
                            start_msg["headers"] = new_headers
                            await send(start_msg)
                            await send({"type": "http.response.body", "body": full_audio, "more_body": False})
                    else:
                        await send(message)

                await self.app(scope, replay_receive, tee_send)
                return

            if method in ("POST", "DELETE") and path.startswith("/v1/audio/voices"):
                await self.app(scope, receive, send)
                await asyncio.to_thread(_sync_speakers_to_drive)
                return

        await self.app(scope, receive, send)
