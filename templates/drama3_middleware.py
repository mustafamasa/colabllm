"""Authenticated ASGI middleware, Google Drive voice/output I/O, and launcher wrapper for Fish Audio Drama 3."""
from __future__ import annotations

import asyncio
import base64
from http import HTTPStatus
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any
import urllib.error
from urllib.parse import parse_qs, unquote
import urllib.request

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
    from starlette.responses import JSONResponse, Response
    from starlette.types import ASGIApp, Message, Receive, Scope, Send
except ImportError:  # pragma: no cover - local unit-test fallback without starlette
    ASGIApp = Message = Receive = Scope = Send = Any  # type: ignore[assignment,misc]

    class Headers:  # type: ignore[no-redef]
        def __init__(self, scope: dict[str, Any]) -> None:
            self._d = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}

        def get(self, key: str, default: str = "") -> str:
            return self._d.get(key.lower(), default)

    class Response:  # type: ignore[no-redef]
        def __init__(self, content: bytes = b"", status_code: int = 200, media_type: str | None = None) -> None:
            self.body = content
            self.status_code = int(status_code)
            self.media_type = media_type

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            hdrs = [(b"content-length", str(len(self.body)).encode("ascii"))]
            if self.media_type:
                hdrs.append((b"content-type", self.media_type.encode("ascii")))
            await send({"type": "http.response.start", "status": self.status_code, "headers": hdrs})
            await send({"type": "http.response.body", "body": self.body, "more_body": False})

    class JSONResponse(Response):  # type: ignore[no-redef]
        def __init__(self, content: Any, status_code: int = 200) -> None:
            super().__init__(json.dumps(content).encode("utf-8"), status_code=status_code, media_type="application/json")

API_TOKEN = os.environ.get("CT_API_TOKEN", "")
SERVICE_PORT = int((os.environ.get("CT_SERVICE_PORT") or "").strip() or "8000")
SERVED_MODEL = (os.environ.get("CT_DRAMA3_MODEL") or "fishaudio/drama-3").strip()
MODEL_ALIAS = (os.environ.get("CT_DRAMA3_ALIAS") or "drama3").strip()
MAX_REF_SEC = float((os.environ.get("CT_DRAMA3_MAX_REF_SEC") or "").strip() or "60.0")
SAMPLE_RATE = 44100

DATA_DIR = Path(os.environ.get("CT_DATA_DIR", "/content/drive/MyDrive/colab-llm/drama3"))
SPEAKERS_DIR = Path(os.environ.get("SPEAKER_SAMPLES_DIR", "/content/drama3/speakers"))
DEFAULT_REF_WAV = Path(os.environ.get("CT_DEFAULT_REF_WAV", "/content/drama3/default_ref.wav"))
DEFAULT_REF_TEXT = (os.environ.get("CT_DEFAULT_REF_TEXT") or "").strip()
DRIVE_ROOT = Path("/content/drive")
MYDRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_VOICES_DIR = DATA_DIR / "voices"
DRIVE_OUTPUT_DIR = DATA_DIR / "outputs"

MODEL_ALIASES = frozenset({
    "",
    "default",
    "drama3",
    "drama-3",
    "drama-3-bf16",
    "drama-3-int4",
    "fishaudio/drama-3",
    "fishaudio/drama-3-int4",
    "tts-1",
    "tts-1-hd",
    "gpt-4o-mini-tts",
})

UNSUPPORTED_SPEECH_KEYS = (
    "language",
    "instructions",
    "seed",
    "extra_params",
    "x_vector_only_mode",
    "stream_codec_output",
    "suppress_bootstrap_silence",
    "token_count",
    "duration_tokens",
    "initial_codec_chunk_frames",
)

BLOCKED_ADMIN_PREFIXES = (
    "/pause_generation",
    "/continue_generation",
    "/update_weights",
    "/update_weights_from_disk",
    "/update_weights_from_tensor",
    "/update_weights_from_distributed",
    "/init_weights_update_group",
    "/destroy_weights_update_group",
    "/weights_checker",
)

CORS_HEADERS: tuple[tuple[bytes, bytes], ...] = (
    (b"access-control-allow-origin", b"*"),
    (b"access-control-allow-methods", b"GET, POST, DELETE, OPTIONS"),
    (
        b"access-control-allow-headers",
        b"Authorization, Content-Type, X-API-Key, X-Save-Drive",
    ),
    (b"access-control-expose-headers", b"X-Saved-Drive-Path, X-Audio-Sample-Rate"),
    (b"access-control-max-age", b"86400"),
)


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


def _validate_save_drive_spec(spec: str) -> str:
    cleaned = (spec or "").strip()
    if not cleaned:
        return ""
    if cleaned not in ("auto", "true", "True", "1"):
        resolve_drive_path(
            cleaned,
            data_dir=DATA_DIR,
            mydrive_root=MYDRIVE_ROOT,
            drive_root=DRIVE_ROOT,
        )
    return cleaned


def _sync_speakers_to_drive(deleted_voice: str | None = None) -> None:
    try:
        DRIVE_VOICES_DIR.mkdir(parents=True, exist_ok=True)
        if deleted_voice:
            try:
                vname = validate_voice_name(deleted_voice)
                if not (SPEAKERS_DIR / f"{vname}.safetensors").is_file():
                    (DRIVE_VOICES_DIR / f"{vname}.safetensors").unlink(missing_ok=True)
            except ValueError:
                pass
        if SPEAKERS_DIR.is_dir():
            for sf in SPEAKERS_DIR.glob("*.safetensors"):
                dst = DRIVE_VOICES_DIR / sf.name
                if not dst.is_file() or sf.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(sf, dst)
    except Exception:
        pass


def _resolve_ref_audio_value(raw_val: str) -> str:
    ra = raw_val.strip()
    if ra.startswith(("http://", "https://", "data:")):
        return ra
    ap = _resolve_path(ra)
    return audio_path_to_wav_data_url(ap, max_duration_sec=MAX_REF_SEC, sample_rate=SAMPLE_RATE)


def _normalize_speech_item(item: dict[str, Any], *, is_batch_item: bool = False) -> bool:
    changed = False

    model_val = item.get("model")
    if isinstance(model_val, str):
        m_clean = model_val.strip()
        if m_clean.lower() in MODEL_ALIASES or m_clean == MODEL_ALIAS:
            if model_val != SERVED_MODEL:
                item["model"] = SERVED_MODEL
                changed = True
    elif model_val is None and not is_batch_item:
        item["model"] = SERVED_MODEL
        changed = True

    for unsupported_key in UNSUPPORTED_SPEECH_KEYS:
        if unsupported_key in item:
            item.pop(unsupported_key, None)
            changed = True

    ref_audio = item.get("ref_audio")
    if isinstance(ref_audio, str) and ref_audio.strip():
        rewritten = _resolve_ref_audio_value(ref_audio)
        if rewritten != ref_audio:
            item["ref_audio"] = rewritten
            changed = True

    refs = item.get("references")
    if isinstance(refs, list):
        for ref_entry in refs:
            if not isinstance(ref_entry, dict):
                continue
            for key_name in ("audio_path", "ref_audio", "audio"):
                raw_ref = ref_entry.get(key_name)
                if isinstance(raw_ref, str) and raw_ref.strip():
                    rewritten = _resolve_ref_audio_value(raw_ref)
                    if key_name != "audio_path":
                        ref_entry.pop(key_name, None)
                    ref_entry["audio_path"] = rewritten
                    changed = True
                    break

    return changed


def _prepare_speech_request(raw_body: bytes, *, is_batch: bool = False) -> tuple[bytes, str | None, str]:
    if not raw_body:
        return raw_body, None, "wav"
    try:
        payload = json.loads(raw_body)
    except Exception:
        return raw_body, None, "wav"
    if not isinstance(payload, dict):
        return raw_body, None, "wav"

    changed = False
    save_drive_spec: str | None = None
    resp_fmt = str(payload.get("response_format") or "wav").strip().lower() or "wav"

    for save_key in ("save_to_drive", "save_drive_path"):
        if save_key in payload:
            val = payload.pop(save_key)
            changed = True
            if val:
                raw_spec = str(val) if not isinstance(val, bool) else "auto"
                save_drive_spec = _validate_save_drive_spec(raw_spec)

    if _normalize_speech_item(payload, is_batch_item=False):
        changed = True

    if is_batch and isinstance(payload.get("items"), list):
        for entry in payload["items"]:
            if isinstance(entry, dict) and _normalize_speech_item(entry, is_batch_item=True):
                changed = True

    if not changed:
        return raw_body, save_drive_spec, resp_fmt
    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), save_drive_spec, resp_fmt


def _decode_base64_audio(raw_b64: str) -> bytes:
    cleaned = raw_b64.strip()
    if cleaned.startswith("data:") and "," in cleaned:
        cleaned = cleaned.split(",", 1)[1]
    return base64.b64decode(cleaned)


def _upload_drive_voice_sync(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    raw_name = str(body.get("name") or "").strip()
    use_default_ref = bool(body.get("use_default_ref", False))
    raw_path = str(body.get("drive_path") or body.get("audio") or body.get("path") or "").strip()
    raw_b64 = str(body.get("audio_base64") or "").strip()
    consent = str(body.get("consent") or "owner-consent").strip()
    ref_text = body.get("ref_text")
    speaker_desc = body.get("speaker_description")

    if not raw_name:
        return 400, {"error": "Field 'name' is required"}
    try:
        name = validate_voice_name(raw_name)
    except ValueError as err:
        return 400, {"error": str(err)}

    if not raw_path and not raw_b64 and not use_default_ref:
        return 400, {"error": "Provide 'drive_path', 'audio_base64', or 'use_default_ref'"}

    temp_input: Path | None = None
    try:
        if use_default_ref:
            ap = DEFAULT_REF_WAV
            if not ap.is_file():
                return 404, {"error": f"Default reference audio not found: {ap}"}
            if not ref_text and DEFAULT_REF_TEXT:
                ref_text = DEFAULT_REF_TEXT
        elif raw_b64:
            try:
                raw_bytes = _decode_base64_audio(raw_b64)
            except Exception as err:
                return 400, {"error": f"Invalid audio_base64: {err}"}
            with tempfile.NamedTemporaryFile(prefix="drama3-voice-", suffix=".bin", delete=False) as tf:
                tf.write(raw_bytes)
                temp_input = Path(tf.name)
            ap = temp_input
        else:
            try:
                ap = _resolve_path(raw_path)
            except ValueError as err:
                return 400, {"error": str(err)}
            if not ap.is_file():
                return 404, {"error": f"Audio file not found on Google Drive: {raw_path} (resolved to {ap})"}

        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            "-t", f"{MAX_REF_SEC:.1f}",
            "-i", str(ap),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
        ]
        try:
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
            wav_bytes = res.stdout if (res.returncode == 0 and res.stdout) else ap.read_bytes()
        except OSError:
            wav_bytes = ap.read_bytes()
    finally:
        if temp_input is not None:
            temp_input.unlink(missing_ok=True)

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
        with urllib.request.urlopen(req, timeout=90) as resp:
            out = json.loads(resp.read().decode("utf-8", errors="replace"))
            _sync_speakers_to_drive()
            return resp.status, out
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        try:
            return err.code, json.loads(raw_err)
        except Exception:
            return err.code, {"error": raw_err}


def _with_cors_send(send: Send) -> Send:
    existing_keys = {k.lower() for k, _ in CORS_HEADERS}

    async def wrapped_send(message: Message) -> None:
        if message["type"] == "http.response.start":
            hdrs = [
                (k, v) for (k, v) in message.get("headers", []) if k.lower() not in existing_keys
            ]
            hdrs.extend(CORS_HEADERS)
            message = {**message, "headers": hdrs}
        await send(message)

    return wrapped_send


class Drama3Middleware:
    """Fail-closed Bearer authentication, CORS, and Google Drive I/O middleware for SGLang-Omni Drama 3."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        scope_type = scope["type"]
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        if scope_type == "http" and scope.get("method") == "OPTIONS":
            resp = Response(b"", status_code=HTTPStatus.NO_CONTENT)
            await resp(scope, receive, _with_cors_send(send))
            return

        send_out = _with_cors_send(send) if scope_type == "http" else send
        root_path = scope.get("root_path", "")
        path = scope["path"].removeprefix(root_path)
        headers = Headers(scope=scope)

        if not _verify_auth(headers):
            if scope_type == "websocket":
                await send({"type": "websocket.close", "code": 4401})
                return
            resp = JSONResponse(
                {
                    "error": {
                        "message": "Unauthorized: invalid or missing Bearer token",
                        "type": "AuthenticationError",
                        "code": 401,
                    }
                },
                status_code=HTTPStatus.UNAUTHORIZED,
            )
            await resp(scope, receive, send_out)
            return

        if any(path == p or path.startswith(p + "/") for p in BLOCKED_ADMIN_PREFIXES):
            if scope_type == "websocket":
                await send({"type": "websocket.close", "code": 4403})
                return
            resp = JSONResponse(
                {
                    "error": {
                        "message": f"Endpoint {path} is disabled on public inference runtime",
                        "type": "PermissionError",
                        "code": 403,
                    }
                },
                status_code=HTTPStatus.FORBIDDEN,
            )
            await resp(scope, receive, send_out)
            return

        if scope_type == "http":
            method = scope.get("method", "GET")

            if method == "GET" and path == "/api/ct/info":
                resp = JSONResponse({
                    "service": "drama3",
                    "engine": "sglang-omni",
                    "model": SERVED_MODEL,
                    "alias": MODEL_ALIAS,
                    "sample_rate": SAMPLE_RATE,
                    "max_codec_frames": 2048,
                    "max_chunk_seconds": 95.1,
                    "max_reference_seconds": MAX_REF_SEC,
                    "speed_range": [0.25, 4.0],
                    "formats": ["wav", "mp3", "flac", "pcm", "aac", "opus"],
                    "supports_batch": True,
                    "supports_multi_reference": True,
                    "speaker_tags": ["<|speaker:0|>"],
                    "default_ref_wav": str(DEFAULT_REF_WAV),
                    "drive_voices_dir": str(DRIVE_VOICES_DIR),
                    "drive_output_dir": str(DRIVE_OUTPUT_DIR),
                })
                await resp(scope, receive, send_out)
                return

            if method == "GET" and path == "/api/ct/logs":
                qs = parse_qs((scope.get("query_string") or b"").decode("utf-8", errors="replace"))
                try:
                    n_lines = int((qs.get("lines") or ["120"])[0])
                except ValueError:
                    n_lines = 120
                resp = JSONResponse(tail_service_log(lines=n_lines))
                await resp(scope, receive, send_out)
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
                except Exception as err:
                    resp = JSONResponse({"error": str(err)}, status_code=400)
                await resp(scope, receive, send_out)
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
                    if not isinstance(body, dict):
                        raise ValueError("Expected JSON object")
                    code, out = await asyncio.to_thread(_upload_drive_voice_sync, body)
                    resp = JSONResponse(out, status_code=code)
                except ValueError as err:
                    resp = JSONResponse({"error": str(err)}, status_code=400)
                except Exception as err:
                    resp = JSONResponse({"error": str(err)}, status_code=500)
                await resp(scope, receive, send_out)
                return

            if method == "POST" and path in ("/v1/audio/speech", "/v1/audio/speech/batch"):
                is_batch = path.endswith("/batch")
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
                    rewritten_body, save_drive_spec, resp_fmt = await asyncio.to_thread(
                        _prepare_speech_request, raw_body, is_batch=is_batch
                    )
                    header_save = headers.get("x-save-drive")
                    if header_save and not save_drive_spec and not is_batch:
                        save_drive_spec = await asyncio.to_thread(_validate_save_drive_spec, header_save)
                except Exception as err:
                    resp = JSONResponse(
                        {"error": {"message": str(err), "type": "BadRequestError", "code": 400}},
                        status_code=400,
                    )
                    await resp(scope, receive, send_out)
                    return

                # Update content-length header for downstream ASGI parser
                new_headers = [
                    (k, v) for (k, v) in scope.get("headers", []) if k.lower() != b"content-length"
                ]
                new_headers.append((b"content-length", str(len(rewritten_body)).encode("ascii")))
                patched_scope = {**scope, "headers": new_headers}

                body_sent = False

                async def replay_receive() -> Message:
                    nonlocal body_sent
                    if not body_sent:
                        body_sent = True
                        return {"type": "http.request", "body": rewritten_body, "more_body": False}
                    return await receive()

                if not save_drive_spec or is_batch:
                    await self.app(patched_scope, replay_receive, send_out)
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
                            await send_out(message)
                    elif message["type"] == "http.response.body":
                        chunk = message.get("body", b"")
                        more = message.get("more_body", False)
                        if resp_status != 200:
                            await send_out(message)
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
                                prefix="drama3",
                                fmt=resp_fmt,
                                mydrive_root=MYDRIVE_ROOT,
                                drive_root=DRIVE_ROOT,
                            )
                            safe_hdr = sanitize_header_param(saved_path).encode("utf-8")
                            out_headers = [
                                (k, v) for (k, v) in resp_headers if k.lower() != b"x-saved-drive-path"
                            ]
                            out_headers.append((b"x-saved-drive-path", safe_hdr))
                            await send_out({**start_msg, "headers": out_headers})
                            await send_out({"type": "http.response.body", "body": full_audio, "more_body": False})
                    else:
                        await send_out(message)

                await self.app(patched_scope, replay_receive, tee_send)
                return

            if method in ("POST", "DELETE") and path.startswith("/v1/audio/voices"):
                deleted_name: str | None = None
                if method == "DELETE" and path.startswith("/v1/audio/voices/"):
                    deleted_name = unquote(path.removeprefix("/v1/audio/voices/").strip("/"))
                await self.app(scope, receive, send_out)
                await asyncio.to_thread(_sync_speakers_to_drive, deleted_name)
                return

        await self.app(scope, receive, send_out)


def run_sgl_omni_server(argv: list[str] | None = None) -> None:
    """Wrap sglang-omni's FastAPI app with Drama3Middleware and launch sgl-omni serve."""
    import sglang_omni.cli as sgl_cli
    import sglang_omni.serve.launcher as launcher

    orig_server_init = launcher.PipelineUvicornServer.__init__

    def _wrapped_server_init(self: Any, config: Any, *args: Any, **kwargs: Any) -> None:
        if not isinstance(config.app, Drama3Middleware):
            config.app = Drama3Middleware(config.app)
            if getattr(config, "loaded_app", None) is not None:
                config.loaded_app = config.app
        orig_server_init(self, config, *args, **kwargs)

    launcher.PipelineUvicornServer.__init__ = _wrapped_server_init  # type: ignore[method-assign]

    orig_async_client = launcher.httpx.AsyncClient

    def _authed_async_client(*args: Any, **kwargs: Any) -> Any:
        hdrs = dict(kwargs.get("headers") or {})
        if API_TOKEN:
            hdrs.setdefault("Authorization", f"Bearer {API_TOKEN}")
        kwargs["headers"] = hdrs
        return orig_async_client(*args, **kwargs)

    launcher.httpx.AsyncClient = _authed_async_client  # type: ignore[assignment]

    if argv is not None:
        sys.argv = [sys.argv[0], *argv]
    sgl_cli.app()


if __name__ == "__main__":
    run_sgl_omni_server()
