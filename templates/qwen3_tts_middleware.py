"""Authenticated ASGI middleware and multi-task routing for Qwen3-TTS."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import subprocess
import time
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
    from starlette.responses import JSONResponse, Response
    from starlette.types import ASGIApp, Message, Receive, Scope, Send
except ImportError:  # pragma: no cover - local unit-test fallback without starlette
    Headers = dict  # type: ignore[assignment,misc]
    JSONResponse = Response = None  # type: ignore[assignment,misc]
    ASGIApp = Message = Receive = Scope = Send = Any  # type: ignore[assignment,misc]

API_TOKEN = os.environ.get("CT_API_TOKEN", "")
SERVICE_PORT = int(os.environ.get("CT_SERVICE_PORT", "8000"))
AUX_PORT = int(os.environ.get("CT_QWEN3_TTS_AUX_PORT", "8001"))
DATA_DIR = Path(os.environ.get("CT_DATA_DIR", "/content/drive/MyDrive/colab-llm/qwen3-tts"))
SPEAKERS_DIR = Path(os.environ.get("SPEAKER_SAMPLES_DIR", "/content/qwen3-tts/speakers"))
DEFAULT_REF_WAV = Path(os.environ.get("CT_DEFAULT_REF_WAV", "/content/qwen3-tts/default_ref.wav"))
DEFAULT_REF_TEXT = os.environ.get(
    "CT_DEFAULT_REF_TEXT",
    "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
)
PRIMARY_MODEL_REPO = os.environ.get("CT_QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DEFAULT_SPEAKER = os.environ.get("CT_QWEN3_DEFAULT_SPEAKER", "vivian").strip().lower()

DRIVE_ROOT = Path("/content/drive")
MYDRIVE_ROOT = Path("/content/drive/MyDrive")
DRIVE_VOICES_DIR = DATA_DIR / "voices"
DRIVE_OUTPUT_DIR = DATA_DIR / "outputs"
CUSTOM_VOICES_META_FILE = SPEAKERS_DIR / "custom_voices_meta.json"

BUILTIN_SPEAKERS = {
    "vivian": {"description": "Bright, slightly edgy young female voice", "native_language": "Chinese"},
    "serena": {"description": "Warm, gentle young female voice", "native_language": "Chinese"},
    "uncle_fu": {"description": "Seasoned male voice with a low, mellow timbre", "native_language": "Chinese"},
    "dylan": {"description": "Youthful Beijing male voice with a clear, natural timbre", "native_language": "Chinese (Beijing Dialect)"},
    "eric": {"description": "Lively Chengdu male voice with a slightly husky brightness", "native_language": "Chinese (Sichuan Dialect)"},
    "ryan": {"description": "Dynamic male voice with strong rhythmic drive", "native_language": "English"},
    "aiden": {"description": "Sunny American male voice with a clear midrange", "native_language": "English"},
    "ono_anna": {"description": "Playful Japanese female voice with a light, nimble timbre", "native_language": "Japanese"},
    "sohee": {"description": "Warm Korean female voice with rich emotion", "native_language": "Korean"},
}

LANG_MAP = {
    "auto": "Auto",
    "zhen": "Auto",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "en": "English",
    "english": "English",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "ru": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "italian": "Italian",
}


def _detect_primary_task_type() -> str:
    low = PRIMARY_MODEL_REPO.lower()
    if "voicedesign" in low or "voice-design" in low or "voice_design" in low:
        return "VoiceDesign"
    if "base" in low:
        return "Base"
    return "CustomVoice"


PRIMARY_TASK_TYPE = _detect_primary_task_type()


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


def _load_voices_meta() -> dict[str, dict[str, Any]]:
    for p in (CUSTOM_VOICES_META_FILE, DRIVE_VOICES_DIR / "custom_voices_meta.json"):
        if p.is_file():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def _save_voices_meta(meta: dict[str, dict[str, Any]]) -> None:
    try:
        SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
        DRIVE_VOICES_DIR.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(meta, indent=2, ensure_ascii=False)
        CUSTOM_VOICES_META_FILE.write_text(raw, encoding="utf-8")
        (DRIVE_VOICES_DIR / "custom_voices_meta.json").write_text(raw, encoding="utf-8")
    except Exception:
        pass


def _sync_speakers_to_drive() -> None:
    try:
        DRIVE_VOICES_DIR.mkdir(parents=True, exist_ok=True)
        if SPEAKERS_DIR.is_dir():
            for sf in SPEAKERS_DIR.iterdir():
                if sf.is_file():
                    dst = DRIVE_VOICES_DIR / sf.name
                    if not dst.is_file() or sf.stat().st_mtime > dst.stat().st_mtime:
                        shutil.copy2(sf, dst)
    except Exception:
        pass


def _register_custom_voice_bytes(
    name: str,
    wav_bytes: bytes,
    ref_text: str | None = None,
    speaker_desc: str | None = None,
    consent: str = "user-consent",
) -> tuple[int, dict[str, Any]]:
    vname = validate_voice_name(name)
    SPEAKERS_DIR.mkdir(parents=True, exist_ok=True)
    DRIVE_VOICES_DIR.mkdir(parents=True, exist_ok=True)
    wav_file = SPEAKERS_DIR / f"{vname}.wav"
    wav_file.write_bytes(wav_bytes)
    shutil.copy2(wav_file, DRIVE_VOICES_DIR / f"{vname}.wav")

    meta = _load_voices_meta()
    meta[vname] = {
        "id": vname,
        "wav_path": str(wav_file),
        "drive_wav_path": str(DRIVE_VOICES_DIR / f"{vname}.wav"),
        "ref_text": ref_text,
        "speaker_description": speaker_desc or "Custom cloned voice for Qwen3-TTS",
        "file_size": len(wav_bytes),
        "mime_type": "audio/wav",
        "created_at": int(time.time()),
    }
    _save_voices_meta(meta)

    fields: list[tuple[str, str]] = [("name", vname), ("consent", consent)]
    if ref_text:
        fields.append(("ref_text", str(ref_text)))
    if speaker_desc:
        fields.append(("speaker_description", str(speaker_desc)))
    content_type, body_bytes = build_multipart_form(
        fields,
        [("audio_sample", f"{vname}.wav", "audio/wav", wav_bytes)],
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
    except Exception:
        return 200, {"voice": meta[vname]}


def _upload_drive_voice_sync(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    raw_name = str(body.get("name") or "").strip()
    raw_path = str(body.get("drive_path") or body.get("audio") or body.get("path") or "").strip()
    audio_b64 = str(body.get("audio_base64") or "").strip()
    consent = str(body.get("consent") or "user-consent").strip()
    ref_text = body.get("ref_text")
    speaker_desc = body.get("speaker_description")
    if not raw_name or (not raw_path and not audio_b64):
        return 400, {"error": "'name' and either 'drive_path' or 'audio_base64' are required"}
    try:
        name = validate_voice_name(raw_name)
    except ValueError as err:
        return 400, {"error": str(err)}

    if audio_b64:
        if "," in audio_b64 and audio_b64.startswith("data:"):
            audio_b64 = audio_b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(audio_b64)
        cmd = [
            "ffmpeg", "-nostdin", "-v", "error", "-t", "24.0",
            "-i", "pipe:0", "-vn", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
        ]
        res = subprocess.run(cmd, input=raw_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
        wav_bytes = res.stdout if (res.returncode == 0 and res.stdout) else raw_bytes
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
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=False)
        wav_bytes = res.stdout if (res.returncode == 0 and res.stdout) else ap.read_bytes()

    return _register_custom_voice_bytes(name, wav_bytes, ref_text=ref_text, speaker_desc=speaker_desc, consent=consent)


def _prepare_speech_request(raw_body: bytes) -> tuple[bytes, dict[str, Any], str, str | None]:
    payload = json.loads(raw_body) if raw_body else {}
    if not isinstance(payload, dict):
        payload = {}

    save_drive_spec: str | None = None
    extras = payload.get("extra_params")
    if not isinstance(extras, dict):
        extras = {}

    for save_key in ("save_to_drive", "save_drive_path"):
        val = payload.pop(save_key, None)
        if val is None and save_key in extras:
            val = extras.pop(save_key, None)
        if val:
            save_drive_spec = str(val) if not isinstance(val, bool) else "auto"

    for k in ("language", "lang", "task_type", "instructions", "instruct", "ref_audio", "ref_text", "x_vector_only_mode"):
        if k in extras and k not in payload:
            payload[k] = extras.pop(k)

    if "instruct" in payload and not payload.get("instructions"):
        payload["instructions"] = payload.pop("instruct")
    else:
        payload.pop("instruct", None)

    raw_lang = str(payload.pop("lang", None) or payload.get("language") or "Auto").strip()
    payload["language"] = LANG_MAP.get(raw_lang.lower(), raw_lang.title() if raw_lang else "Auto")

    # Resolve ref_audio if present (reject arbitrary external file:// outside Drive)
    ref_audio = payload.get("ref_audio")
    if isinstance(ref_audio, str) and ref_audio.strip():
        ra = ref_audio.strip()
        if not ra.startswith(("http://", "https://", "data:")):
            ap = _resolve_path(ra)
            payload["ref_audio"] = audio_path_to_wav_data_url(ap)

    meta = _load_voices_meta()
    raw_voice = str(payload.get("voice") or payload.get("speaker") or "").strip().lower()
    payload.pop("speaker", None)
    is_custom_uploaded_voice = bool(raw_voice and raw_voice not in BUILTIN_SPEAKERS and raw_voice not in ("default", "alloy"))
    if raw_voice in meta and not payload.get("ref_audio"):
        vinfo = meta[raw_voice]
        wpath = Path(vinfo.get("wav_path") or "")
        if not wpath.is_file():
            wpath = Path(vinfo.get("drive_wav_path") or "")
        if wpath.is_file():
            payload["ref_audio"] = f"file://{wpath}"
            if vinfo.get("ref_text") and not payload.get("ref_text"):
                payload["ref_text"] = vinfo["ref_text"]
            is_custom_uploaded_voice = True

    req_task = str(payload.get("task_type") or "").strip()
    if req_task.lower() in ("voicedesign", "voice_design", "voice-design"):
        task_type = "VoiceDesign"
    elif req_task.lower() in ("base", "voiceclone", "voice_clone", "voice-clone") or payload.get("ref_audio") or is_custom_uploaded_voice:
        task_type = "Base"
    elif req_task.lower() in ("customvoice", "custom_voice", "custom-voice"):
        task_type = "CustomVoice"
    else:
        task_type = PRIMARY_TASK_TYPE

    payload["task_type"] = task_type

    if task_type == "CustomVoice":
        if not raw_voice or raw_voice in ("default", "alloy"):
            raw_voice = "ryan" if payload.get("language") == "English" else DEFAULT_SPEAKER
        payload["voice"] = raw_voice
        payload.pop("ref_audio", None)
        payload.pop("ref_text", None)
        payload.pop("x_vector_only_mode", None)
    elif task_type == "VoiceDesign":
        payload.pop("voice", None)
        payload.pop("ref_audio", None)
        payload.pop("ref_text", None)
        payload.pop("x_vector_only_mode", None)
        if not payload.get("instructions"):
            payload["instructions"] = "Natural, expressive human voice."
    elif task_type == "Base":
        if not payload.get("ref_audio") and (not raw_voice or raw_voice in ("default", "alloy") or raw_voice in BUILTIN_SPEAKERS):
            payload["ref_audio"] = f"file://{DEFAULT_REF_WAV}"
            if not payload.get("ref_text") and payload.get("x_vector_only_mode") is None:
                payload["ref_text"] = DEFAULT_REF_TEXT
            payload.pop("voice", None)
        elif payload.get("ref_audio"):
            payload.pop("voice", None)
        if payload.get("x_vector_only_mode") is None:
            payload["x_vector_only_mode"] = not bool(payload.get("ref_text") and str(payload.get("ref_text")).strip())

    if extras:
        payload["extra_params"] = extras
    else:
        payload.pop("extra_params", None)

    return json.dumps(payload, ensure_ascii=False).encode("utf-8"), payload, task_type, save_drive_spec


def _call_aux_worker_sync(payload: dict[str, Any]) -> tuple[int, bytes, dict[str, str]]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{AUX_PORT}/synthesize",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            headers = {
                "content-type": resp.headers.get("Content-Type") or "audio/wav",
                "x-audio-sample-rate": resp.headers.get("X-Audio-Sample-Rate") or "24000",
                "x-qwen3-tts-engine": resp.headers.get("X-Qwen3-TTS-Engine") or "aux",
            }
            return resp.status, resp.read(), headers
    except urllib.error.HTTPError as err:
        return err.code, err.read(), {"content-type": "application/json"}


def _design_and_register_voice_sync(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    instruct = str(body.get("instructions") or body.get("instruct") or "").strip()
    sample_text = str(body.get("sample_text") or body.get("text") or body.get("input") or "").strip()
    language = LANG_MAP.get(str(body.get("language") or body.get("lang") or "Auto").strip().lower(), "Auto")
    raw_voice_name = str(body.get("name") or "").strip()
    voice_name = ""
    if raw_voice_name:
        try:
            voice_name = validate_voice_name(raw_voice_name)
        except ValueError as err:
            return 400, {"error": str(err)}
    if not instruct:
        return 400, {"error": "'instructions' (voice persona description) is required for Voice Design"}
    if not sample_text:
        sample_text = (
            "Hello! This is a sample of my designed voice, ready for speech synthesis."
            if language == "English"
            else "Hello! This is a sample of the custom voice generated from your description."
        )

    status, audio_bytes, headers = _call_aux_worker_sync({
        "task_type": "VoiceDesign",
        "input": sample_text,
        "language": language,
        "instructions": instruct,
        "response_format": "wav",
    })
    if status != 200:
        try:
            return status, json.loads(audio_bytes.decode("utf-8", errors="replace"))
        except Exception:
            return status, {"error": audio_bytes.decode("utf-8", errors="replace")}

    result: dict[str, Any] = {
        "status": "ok",
        "task_type": "VoiceDesign",
        "instructions": instruct,
        "sample_text": sample_text,
        "language": language,
        "sample_rate": int(headers.get("x-audio-sample-rate") or 24000),
        "audio_base64": base64.b64encode(audio_bytes).decode("ascii"),
    }
    if voice_name:
        _, reg_out = _register_custom_voice_bytes(
            voice_name,
            audio_bytes,
            ref_text=sample_text,
            speaker_desc=f"VoiceDesign: {instruct}",
            consent="voice-design",
        )
        result["registered_voice"] = voice_name
        result["registration"] = reg_out
    if body.get("save_to_drive"):
        try:
            result["saved_drive_path"] = save_output_audio_to_drive(
                audio_bytes,
                str(body.get("save_to_drive")),
                data_dir=DATA_DIR,
                output_dir=DRIVE_OUTPUT_DIR,
                prefix="qwen3-tts",
                fmt="wav",
                mydrive_root=MYDRIVE_ROOT,
                drive_root=DRIVE_ROOT,
            )
        except ValueError as err:
            return 400, {"error": str(err)}
    return 200, result


class Qwen3TTSMiddleware:
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
                meta = _load_voices_meta()
                resp = JSONResponse({
                    "service": "qwen3-tts",
                    "primary_model": PRIMARY_MODEL_REPO,
                    "primary_task_type": PRIMARY_TASK_TYPE,
                    "supported_tasks": ["CustomVoice", "VoiceDesign", "Base"],
                    "builtin_speakers": BUILTIN_SPEAKERS,
                    "custom_voices": list(meta.keys()),
                    "languages": sorted(set(LANG_MAP.values())),
                    "default_speaker": DEFAULT_SPEAKER,
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

            if method == "POST" and path == "/api/ct/design-voice":
                chunks = []
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
                    code, out = await asyncio.to_thread(_design_and_register_voice_sync, body)
                    resp = JSONResponse(out, status_code=code)
                except ValueError as e:
                    resp = JSONResponse({"error": str(e)}, status_code=400)
                except Exception as e:
                    resp = JSONResponse({"error": str(e)}, status_code=500)
                await resp(scope, receive, send)
                return

            if method == "GET" and path == "/v1/audio/voices":
                meta = _load_voices_meta()
                all_voices = sorted(set(BUILTIN_SPEAKERS.keys()) | set(meta.keys()))
                resp = JSONResponse({
                    "voices": all_voices,
                    "builtin_speakers": BUILTIN_SPEAKERS,
                    "uploaded_voices": list(meta.values()),
                })
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
                    rewritten_body, payload_dict, task_type, save_drive_spec = await asyncio.to_thread(
                        _prepare_speech_request, raw_body
                    )
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

                if task_type != PRIMARY_TASK_TYPE:
                    status, audio_bytes, aux_headers = await asyncio.to_thread(_call_aux_worker_sync, payload_dict)
                    if status == 200 and save_drive_spec:
                        try:
                            saved_path = await asyncio.to_thread(
                                save_output_audio_to_drive,
                                audio_bytes,
                                save_drive_spec,
                                data_dir=DATA_DIR,
                                output_dir=DRIVE_OUTPUT_DIR,
                                prefix="qwen3-tts",
                                fmt=str(payload_dict.get("response_format") or "wav"),
                                mydrive_root=MYDRIVE_ROOT,
                                drive_root=DRIVE_ROOT,
                            )
                        except ValueError as e:
                            resp = JSONResponse({"error": str(e)}, status_code=400)
                            await resp(scope, receive, send)
                            return
                        aux_headers["x-saved-drive-path"] = sanitize_header_param(saved_path)
                    resp = Response(content=audio_bytes, status_code=status, headers=aux_headers)
                    await resp(scope, receive, send)
                    return

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
                                prefix="qwen3-tts",
                                fmt=str(payload_dict.get("response_format") or "wav"),
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
                if method == "DELETE":
                    raw_vname = path.rstrip("/").split("/")[-1]
                    try:
                        vname = validate_voice_name(raw_vname)
                    except ValueError as err:
                        resp = JSONResponse({"error": str(err)}, status_code=400)
                        await resp(scope, receive, send)
                        return
                    meta = _load_voices_meta()
                    if vname in meta:
                        meta.pop(vname, None)
                        _save_voices_meta(meta)
                        for d in (SPEAKERS_DIR, DRIVE_VOICES_DIR):
                            (d / f"{vname}.wav").unlink(missing_ok=True)
                await self.app(scope, receive, send)
                await asyncio.to_thread(_sync_speakers_to_drive)
                return

        await self.app(scope, receive, send)
