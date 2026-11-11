#!/usr/bin/env python3
"""Fish Audio Drama 3 (SGLang-Omni 0.2.0 zero-shot multi-ref TTS & novel narration on Colab G4/H100/A100).

Serves `fishaudio/drama-3` (BF16 on >=72 GiB GPUs such as G4 / H100 / A100-80GB) or
`fishaudio/drama-3-int4` (auto-selected on <72 GiB GPUs or via `--variant int4`) using
`sglang-omni[fishaudio]==0.2.0`:
  - 3-stage pipeline: `preprocessing` -> `tts_engine` (Dual-AR Slow+Fast Transformer + CUDA graphs)
    -> `vocoder` (S2-Pro DAC 44.1 kHz mono decoder).
  - Zero-shot single or multi-reference (`--ref-audio ... --ref-audio ...`) voice cloning,
    persistent named voices (`/v1/audio/voices`) synced to Google Drive, inline expressive tags
    (`[sighs]`, `[laughs]`, `[whispers]`), and high-throughput `/v1/audio/speech/batch` synthesis.
  - Built-in `narrate` subcommand for full-length novels / audiobooks: chapter detection,
    sentence-aligned chunking below Drama 3's ~95s (2048-frame) single-generation limit,
    dynamic `max_new_tokens` estimation, trailing silence trimming, resumable `manifest.json`,
    and seamless 44.1 kHz chapter stitching.

Usage:
  colabllm drama3 up                                    # G4 GPU (default) + configured tunnel
  colabllm drama3 up --gpu H100                         # H100 80GB (BF16)
  colabllm drama3 up --gpu A100 --variant int4          # A100 40GB (INT4)
  colabllm drama3 status
  colabllm drama3 voices
  colabllm drama3 upload-voice --name narrator --audio narrator.wav --ref-text "Transcript of the reference audio."
  colabllm drama3 tts "What a lovely day. [sighs] Finally some quiet time to read." --voice narrator -o out.wav
  colabllm drama3 narrate novel.txt --voice narrator -O ./novel-audio
  colabllm drama3 down
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cli_support import http_timeout, nonnegative_seconds, origin, resolve_endpoint, safe_urlopen  # noqa: E402
from colabsvc import Service, State, die, log, main, validate_instance, warn  # noqa: E402
from drama3_narration import (  # noqa: E402
    DEFAULT_CHUNK_MAX_CHARS,
    DRAMA3_SAMPLE_RATE,
    build_llm_cast_messages,
    concat_wav_segments,
    estimate_max_new_tokens,
    extract_cast_name_map,
    parse_and_verify_llm_cast_output,
    parse_chapter_selection,
    pcm16_to_wav_bytes,
    resolve_chunk_voice_spec,
    sanitize_drama3_text,
    split_into_chapters,
    split_into_llm_windows,
    split_text_chunks,
    trim_trailing_silence_wav,
    wav_duration_seconds,
)
from paths import user_home  # noqa: E402

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

SERVICE = Service(
    name="drama3",
    notebook="drama3.ipynb",
    service_port=8000,
    auth="builtin",
    public_port=8000,
    gpu="G4",
    high_mem=False,
    default_hostname="drama3",
    ready_timeout=3600,
    env={
        "CT_DRAMA3_VARIANT": "auto",
        "CT_DRAMA3_MODEL": "",
        "CT_DRAMA3_ALIAS": "drama3",
        "CT_DRAMA3_ENGINE_GPU_MEM": "0.70",
        "CT_DRAMA3_VOCODER_GPU_MEM": "0.10",
        "CT_DRAMA3_MAX_RUNNING_REQS": "16",
        "CT_DRAMA3_CONTEXT_LENGTH": "16384",
        "CT_DRAMA3_BATCH_MAX_ITEMS": "32",
        "CT_DRAMA3_MAX_REF_SEC": "60.0",
    },
)

UP_ENV_FLAGS = {
    "--variant": "CT_DRAMA3_VARIANT",
    "--model": "CT_DRAMA3_MODEL",
    "--alias": "CT_DRAMA3_ALIAS",
    "--engine-gpu-mem": "CT_DRAMA3_ENGINE_GPU_MEM",
    "--vocoder-gpu-mem": "CT_DRAMA3_VOCODER_GPU_MEM",
    "--max-running-reqs": "CT_DRAMA3_MAX_RUNNING_REQS",
    "--context-length": "CT_DRAMA3_CONTEXT_LENGTH",
    "--batch-max-items": "CT_DRAMA3_BATCH_MAX_ITEMS",
    "--max-ref-sec": "CT_DRAMA3_MAX_REF_SEC",
}


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(
        State(SERVICE).load(),
        url_override,
        token_override,
        url_env="DRAMA3_URL",
        token_env="DRAMA3_API_TOKEN",
    )


def _http_json(
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    method: str | None = None,
    timeout: int | float | None = 120,
) -> tuple[int, dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": BROWSER_UA,
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
    try:
        with safe_urlopen(req, timeout=http_timeout(timeout)) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else {"raw": raw})
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        try:
            return err.code, json.loads(raw_err)
        except Exception:
            return err.code, {"error": raw_err}


def _http_speech_bytes(
    base_url: str,
    token: str,
    payload: dict[str, Any],
    timeout: int | float | None = 300,
) -> tuple[bytes, dict[str, str]]:
    req = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": BROWSER_UA,
        },
        method="POST",
    )
    with safe_urlopen(req, timeout=http_timeout(timeout)) as resp:
        chunks: list[bytes] = []
        while True:
            blk = resp.read(65536)
            if not blk:
                break
            chunks.append(blk)
        hdrs = {k.lower(): v for k, v in resp.headers.items()}
        return b"".join(chunks), hdrs


def _encode_local_audio_to_data_url(path_str: str) -> str:
    p = Path(path_str).expanduser()
    if not p.is_file():
        die(f"Local audio file not found: {p}")
    mime, _ = mimetypes.guess_type(str(p))
    if not mime or not mime.startswith("audio/"):
        mime = "audio/wav"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _resolve_audio_spec(spec: str | None, is_drive: bool = False) -> str | None:
    if not spec:
        return None
    s = spec.strip()
    if is_drive:
        return s if s.startswith(("drive://", "gdrive://", "/content/drive/")) else f"drive://{s}"
    if s.startswith(("http://", "https://", "data:", "file://", "drive://", "gdrive://", "/content/")):
        return s
    p = Path(s).expanduser()
    if p.is_file():
        return _encode_local_audio_to_data_url(str(p))
    return f"drive://{s}"


def _build_voice_cloning_fields(
    ref_audios: list[str] | None,
    ref_texts: list[str] | None,
    *,
    is_drive: bool = False,
) -> dict[str, Any]:
    audios = [a for a in (ref_audios or []) if a and a.strip()]
    texts = list(ref_texts or [])
    if not audios:
        return {}
    if len(texts) > len(audios):
        die(f"Got {len(texts)} --ref-text values for {len(audios)} --ref-audio values.")
    resolved = [_resolve_audio_spec(a, is_drive=is_drive) for a in audios]
    if len(resolved) == 1:
        out: dict[str, Any] = {"ref_audio": resolved[0]}
        if texts and texts[0].strip():
            out["ref_text"] = texts[0].strip()
        return out
    refs: list[dict[str, Any]] = []
    for idx, r_audio in enumerate(resolved):
        entry: dict[str, Any] = {"audio_path": r_audio}
        if idx < len(texts) and texts[idx].strip():
            entry["text"] = texts[idx].strip()
        refs.append(entry)
    return {"references": refs}


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _convert_wav_file(wav_path: Path, target_path: Path, fmt: str) -> None:
    if fmt == "wav":
        if wav_path != target_path:
            shutil.copyfile(wav_path, target_path)
        return
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        die(f"Local `ffmpeg` is required to transcode stitched chapters to --format {fmt} (or use --format wav).")
    codec_args: list[str]
    if fmt == "mp3":
        codec_args = ["-c:a", "libmp3lame", "-q:a", "2"]
    elif fmt == "opus":
        codec_args = ["-c:a", "libopus", "-b:a", "96k"]
    elif fmt == "flac":
        codec_args = ["-c:a", "flac"]
    else:
        codec_args = []
    res = subprocess.run(
        [ffmpeg_bin, "-y", "-nostdin", "-v", "error", "-i", str(wav_path), *codec_args, str(target_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        die(f"ffmpeg failed converting {wav_path.name} -> {target_path.name}: {res.stderr.strip()}")


def cmd_ls_drive(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.api_token)
    qs = urllib.parse.urlencode({
        "path": args.path or "",
        "recursive": "0" if args.no_recursive else "1",
    })
    code, data = _http_json(f"{base_url}/api/ct/drive-list?{qs}", token, timeout=60)
    if code != 200:
        print(json.dumps(data, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    files = data.get("files") or []
    log(f"Google Drive audio files ({len(files)} matched):")
    for f in files:
        print(f"  {f['size_mb']:>7.2f} MB  {f['rel_path']}")
    return 0


def cmd_voices(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.api_token)
    code, data = _http_json(f"{base_url}/v1/audio/voices", token, timeout=30)
    if code != 200:
        print(json.dumps(data, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    voices = data.get("voices") or []
    uploaded = data.get("uploaded_voices") or []
    log(f"Available voices ({len(voices)}): {', '.join(voices) if voices else '(none)'}")
    if uploaded:
        print("\nUploaded voices details:")
        for v in uploaded:
            vname = v.get("name") or v.get("id") or "?"
            desc = f" — {v.get('speaker_description')}" if v.get("speaker_description") else ""
            rtxt = f" [ref_text={v.get('ref_text')[:40]!r}]" if v.get("ref_text") else ""
            print(f"  - {vname} (size={v.get('file_size', 0)} bytes, mime={v.get('mime_type', '')}){desc}{rtxt}")
    return 0


def cmd_upload_voice(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.api_token)
    if not args.audio and not args.drive:
        die("Provide either `--audio <local_file>` or `--drive <drive_path>`.")

    if args.drive or (args.audio and str(args.audio).startswith(("drive://", "gdrive://", "/content/drive/"))):
        drive_spec = args.drive or args.audio
        payload: dict[str, Any] = {
            "name": args.name,
            "drive_path": drive_spec,
            "consent": args.consent,
        }
        if args.ref_text:
            payload["ref_text"] = args.ref_text
        if args.description:
            payload["speaker_description"] = args.description
        code, out = _http_json(f"{base_url}/api/ct/drive-upload-voice", token, payload=payload, timeout=120)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if code in (200, 201) else 1

    audio_path = Path(args.audio).expanduser()
    if not audio_path.is_file():
        die(f"Audio file not found: {audio_path}")

    # Route local uploads through /api/ct/drive-upload-voice (base64) when non-WAV or when ffmpeg normalization helps,
    # or multipart to /v1/audio/voices for direct WAV/MP3/FLAC/OGG uploads.
    if audio_path.suffix.lower() not in (".wav", ".mp3", ".flac", ".ogg"):
        payload = {
            "name": args.name,
            "audio_base64": base64.b64encode(audio_path.read_bytes()).decode("ascii"),
            "consent": args.consent,
        }
        if args.ref_text:
            payload["ref_text"] = args.ref_text
        if args.description:
            payload["speaker_description"] = args.description
        code, out = _http_json(f"{base_url}/api/ct/drive-upload-voice", token, payload=payload, timeout=120)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if code in (200, 201) else 1

    mime, _ = mimetypes.guess_type(str(audio_path))
    if not mime or not mime.startswith("audio/"):
        mime = "audio/wav"

    boundary = f"----drama3voice{int(time.time() * 1000)}"
    parts: list[bytes] = []

    def add_field(k: str, v: str) -> None:
        safe_k = k.replace('"', "").replace("\r", "").replace("\n", "")
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{safe_k}"\r\n\r\n{v}\r\n'.encode("utf-8")
        )

    add_field("name", args.name)
    add_field("consent", args.consent)
    if args.ref_text:
        add_field("ref_text", args.ref_text)
    if args.description:
        add_field("speaker_description", args.description)

    safe_fn = audio_path.name.replace('"', "_").replace("\r", "").replace("\n", "")
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio_sample"; filename="{safe_fn}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
        + audio_path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )
    req = urllib.request.Request(
        f"{base_url}/v1/audio/voices",
        data=b"".join(parts),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": BROWSER_UA,
        },
        method="POST",
    )
    try:
        with safe_urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode("utf-8", errors="replace"))
            print(json.dumps(out, indent=2, ensure_ascii=False))
            return 0
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        print(raw_err, file=sys.stderr)
        return 1


def cmd_delete_voice(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.api_token)
    vname = urllib.parse.quote(args.name.strip())
    code, out = _http_json(f"{base_url}/v1/audio/voices/{vname}", token, method="DELETE", timeout=30)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if code == 200 else 1


def cmd_tts(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.api_token)
    text = (args.text or args.text_opt or "").strip()
    if args.text_file:
        tf = Path(args.text_file).expanduser()
        if not tf.is_file():
            die(f"Text file not found: {tf}")
        text = tf.read_text(encoding="utf-8").strip()
    text = sanitize_drama3_text(text)
    if not text:
        die("Provide text to synthesize (positional `text`, `--text`, or `--text-file`).")

    out_fmt = args.format.lower()
    wire_fmt = "pcm" if args.stream else out_fmt
    payload: dict[str, Any] = {
        "model": args.model,
        "input": text,
        "response_format": wire_fmt,
        "speed": float(args.speed),
    }
    if args.stream:
        payload["stream"] = True

    clone_fields = _build_voice_cloning_fields(args.ref_audio, args.ref_text, is_drive=args.ref_drive)
    if clone_fields:
        payload.update(clone_fields)
    elif args.voice:
        payload["voice"] = args.voice

    if args.max_new_tokens is not None:
        payload["max_new_tokens"] = int(args.max_new_tokens)
    elif not args.no_auto_max_tokens:
        payload["max_new_tokens"] = estimate_max_new_tokens(text, speed=float(args.speed))

    if args.temperature is not None:
        payload["temperature"] = float(args.temperature)
    if args.top_p is not None:
        payload["top_p"] = float(args.top_p)
    if args.top_k is not None:
        payload["top_k"] = int(args.top_k)
    if args.save_drive and not args.stream:
        payload["save_to_drive"] = args.save_drive

    t0 = time.monotonic()
    try:
        raw_audio, hdrs = _http_speech_bytes(base_url, token, payload, timeout=args.timeout)
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        print(f"[drama3] HTTP {err.code} from /v1/audio/speech:\n{raw_err}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - t0
    sample_rate = int(hdrs.get("x-audio-sample-rate") or DRAMA3_SAMPLE_RATE)
    saved_drive = hdrs.get("x-saved-drive-path")

    if args.stream and out_fmt == "wav":
        audio_bytes = pcm16_to_wav_bytes(raw_audio, sample_rate=sample_rate, channels=1)
    elif args.stream and out_fmt != "pcm":
        die("--stream only supports `--format wav` or `--format pcm`.")
    else:
        audio_bytes = raw_audio

    if out_fmt == "wav" and not args.no_trim_silence:
        audio_bytes = trim_trailing_silence_wav(audio_bytes)

    out_str = args.output or f"drama3-{time.strftime('%Y%m%d-%H%M%S')}.{out_fmt}"
    out_path = Path(out_str).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)

    dur_info = ""
    if out_fmt == "wav":
        try:
            dur = wav_duration_seconds(audio_bytes)
            rtf = elapsed / dur if dur > 0 else 0.0
            dur_info = f", duration={dur:.2f}s, RTF={rtf:.3f}"
        except Exception:
            pass

    msg = (
        f"Synthesized {len(audio_bytes):,} bytes ({out_fmt}, {sample_rate} Hz, "
        f"speed={args.speed}x{dur_info}) in {elapsed:.2f}s -> {out_path}"
    )
    if saved_drive:
        msg += f" (Drive: {saved_drive})"
    log(msg)
    return 0


def _normalize_openai_v1_base(raw_url: str) -> str:
    u = raw_url.rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if u.endswith(suffix):
            u = u[: -len(suffix)].rstrip("/")
    if not u.endswith("/v1"):
        u = f"{u}/v1"
    return u


def _resolve_llm_client(args: argparse.Namespace) -> dict[str, Any] | None:
    llm_svc = (getattr(args, "llm_service", None) or "").strip()
    llm_url = (getattr(args, "llm_url", None) or "").strip()
    llm_tok = (getattr(args, "llm_token", None) or "").strip()
    if not llm_svc and not llm_url:
        return None

    auth_header: str | None = None
    if llm_svc:
        svc_id = validate_instance(llm_svc)
        st_path = user_home() / "services" / svc_id / "state.json"
        if not st_path.is_file():
            die(f"LLM service '{svc_id}' has no state file at {st_path}; start it first or pass --llm-url.")
        try:
            st = json.loads(st_path.read_text(encoding="utf-8"))
        except Exception as err:
            die(f"Invalid state file for LLM service '{svc_id}': {err}")
        if not isinstance(st, dict):
            die(f"Invalid state object for LLM service '{svc_id}'.")

        if llm_tok or st.get("api_token") or os.environ.get("DRAMA3_LLM_API_TOKEN"):
            resolved_url, resolved_tok = resolve_endpoint(
                st,
                llm_url or None,
                llm_tok or None,
                url_env="DRAMA3_LLM_URL",
                token_env="DRAMA3_LLM_API_TOKEN",
            )
            auth_header = f"Bearer {resolved_tok}"
        elif st.get("url") and st.get("auth_user") and st.get("auth_pass"):
            resolved_url = (llm_url or str(st["url"])).rstrip("/")
            if origin(resolved_url) != origin(str(st["url"])):
                die("LLM URL origin differs from stored service state; pass --llm-token explicitly.")
            b64_cred = base64.b64encode(f"{st['auth_user']}:{st['auth_pass']}".encode("utf-8")).decode("ascii")
            auth_header = f"Basic {b64_cred}"
        else:
            die(f"LLM service '{svc_id}' has no stored URL/credentials; start it first or pass --llm-url / --llm-token.")
    else:
        _, host, _ = origin(llm_url)
        resolved_url = llm_url.rstrip("/")
        tok = llm_tok or os.environ.get("DRAMA3_LLM_API_TOKEN") or os.environ.get("OPENAI_API_KEY") or ""
        if tok.strip():
            auth_header = f"Bearer {tok.strip()}"
        elif host not in ("127.0.0.1", "localhost", "::1"):
            die("Provide --llm-token (or DRAMA3_LLM_API_TOKEN / OPENAI_API_KEY) when using a remote --llm-url.")

    v1_base = _normalize_openai_v1_base(resolved_url)
    model_name = (getattr(args, "llm_model", None) or "").strip()
    if not model_name:
        headers = {"User-Agent": BROWSER_UA}
        if auth_header:
            headers["Authorization"] = auth_header
        req = urllib.request.Request(f"{v1_base}/models", headers=headers, method="GET")
        try:
            with safe_urlopen(req, timeout=http_timeout(30)) as resp:
                mdata = json.loads(resp.read().decode("utf-8", errors="replace"))
                entries = mdata.get("data") if isinstance(mdata, dict) else None
                if isinstance(entries, list) and entries and isinstance(entries[0], dict):
                    model_name = str(entries[0].get("id") or "").strip()
        except Exception as err:
            warn(f"Could not auto-discover LLM model from {v1_base}/models ({err}); falling back to 'default'.")
        if not model_name:
            model_name = "default"

    return {
        "v1_base": v1_base,
        "auth_header": auth_header,
        "model": model_name,
        "is_colab_service": bool(llm_svc),
    }


def _call_llm_chat(
    llm_client: dict[str, Any],
    messages: list[dict[str, str]],
    *,
    timeout: int | float | None = 180,
) -> str:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": BROWSER_UA,
    }
    if llm_client.get("auth_header"):
        headers["Authorization"] = llm_client["auth_header"]

    base_payload: dict[str, Any] = {
        "model": llm_client["model"],
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 4096,
    }
    attempts = [
        {**base_payload, "chat_template_kwargs": {"enable_thinking": False}},
        base_payload,
    ]
    last_err: Exception | None = None
    for attempt_idx, payload in enumerate(attempts):
        req = urllib.request.Request(
            f"{llm_client['v1_base']}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with safe_urlopen(req, timeout=http_timeout(timeout)) as resp:
                body = json.loads(resp.read().decode("utf-8", errors="replace"))
                choices = body.get("choices") if isinstance(body, dict) else None
                if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content
                raise RuntimeError("Empty choices in /chat/completions response")
        except urllib.error.HTTPError as err:
            last_err = err
            if attempt_idx == 0 and err.code in (400, 422):
                continue
            break
        except Exception as err:
            last_err = err
            break
    raise RuntimeError(f"LLM /chat/completions failed: {last_err}")


def _annotate_chapter_via_llm(
    chapter_text: str,
    slug: str,
    out_dir: Path,
    llm_client: dict[str, Any],
    cast_cfg: dict[str, Any] | None,
    *,
    add_tags: bool = False,
    force: bool = False,
    timeout: int | float | None = 180,
) -> str:
    cache_dir = out_dir / "annotated"
    cache_file = cache_dir / f"{slug}.txt"
    if not force and cache_file.is_file() and cache_file.stat().st_size > 0:
        log(f"  Using cached LLM role script: {cache_file.name}")
        return cache_file.read_text(encoding="utf-8")

    char_names: list[str] = []
    for _, canon in extract_cast_name_map(cast_cfg):
        if canon not in char_names:
            char_names.append(canon)

    windows = split_into_llm_windows(chapter_text, max_window_chars=1600)
    out_windows: list[str] = []
    ok_count = 0
    for w_idx, win in enumerate(windows, 1):
        msgs = build_llm_cast_messages(win, char_names, add_emotion_tags=add_tags)
        try:
            raw_reply = _call_llm_chat(llm_client, msgs, timeout=timeout)
            verified = parse_and_verify_llm_cast_output(raw_reply, win)
            if verified:
                out_windows.append(verified)
                ok_count += 1
            else:
                warn(
                    f"LLM script window {w_idx}/{len(windows)} in {slug} failed verbatim check; "
                    "falling back to rule-based quote splitting for that window."
                )
                out_windows.append(win)
        except Exception as err:
            warn(
                f"LLM script window {w_idx}/{len(windows)} in {slug} failed ({err}); "
                "falling back to rule-based quote splitting for that window."
            )
            out_windows.append(win)

    assembled = "\n\n".join(out_windows)
    if ok_count > 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(assembled + "\n", encoding="utf-8")
        log(f"  LLM role annotation verified ({ok_count}/{len(windows)} window(s)) -> {cache_file.name}")
    return assembled


def _synthesize_narration_batch(
    base_url: str,
    token: str,
    batch_chunks: list[dict[str, Any]],
    *,
    model: str,
    default_voice: str | None,
    cast: dict[str, Any] | None,
    clone_fields: dict[str, Any],
    ref_drive: bool = False,
    speed: float,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    trim_silence: bool,
    use_batch_endpoint: bool,
    timeout: int | float | None,
) -> list[bytes]:
    """Synthesize a batch of chunks via /v1/audio/speech/batch with per-item fallback retry."""
    sampling_common: dict[str, Any] = {}
    if temperature is not None:
        sampling_common["temperature"] = float(temperature)
    if top_p is not None:
        sampling_common["top_p"] = float(top_p)
    if top_k is not None:
        sampling_common["top_k"] = int(top_k)

    cast_clone_cache: dict[tuple[str, str, bool], dict[str, Any]] = {}

    def get_cast_clone_fields(ref_audio: str, ref_text: str | None) -> dict[str, Any]:
        key = (ref_audio, ref_text or "", ref_drive)
        if key not in cast_clone_cache:
            cast_clone_cache[key] = _build_voice_cloning_fields(
                [ref_audio],
                [ref_text] if ref_text else None,
                is_drive=ref_drive,
            )
        return cast_clone_cache[key]

    def build_single_payload(ch: dict[str, Any]) -> dict[str, Any]:
        vspec = ch.get("voice_spec") or resolve_chunk_voice_spec(ch, default_voice=default_voice, cast=cast)
        item_speed = float(vspec.get("speed") if vspec.get("speed") is not None else speed)
        p: dict[str, Any] = {
            "model": model,
            "input": ch["text"],
            "response_format": "wav",
            "speed": item_speed,
            "max_new_tokens": ch["max_new_tokens"],
            **sampling_common,
        }
        if vspec.get("ref_audio"):
            p.update(get_cast_clone_fields(vspec["ref_audio"], vspec.get("ref_text")))
        elif clone_fields and not ch.get("role_hint") and not (isinstance(cast, dict) and cast.get("narrator")):
            p.update(clone_fields)
        elif vspec.get("voice"):
            p["voice"] = vspec["voice"]
        elif clone_fields:
            p.update(clone_fields)
        return p

    outputs: list[bytes | None] = [None] * len(batch_chunks)

    if use_batch_endpoint and len(batch_chunks) > 1:
        items = []
        for ch in batch_chunks:
            p = build_single_payload(ch)
            p.pop("model", None)
            items.append(p)
        batch_payload: dict[str, Any] = {
            "model": model,
            "response_format": "wav",
            "speed": float(speed),
            "items": items,
        }
        code, data = _http_json(
            f"{base_url}/v1/audio/speech/batch",
            token,
            payload=batch_payload,
            timeout=timeout,
        )
        if code == 200 and isinstance(data.get("results"), list):
            for res_item in data["results"]:
                if not isinstance(res_item, dict):
                    continue
                idx = res_item.get("index")
                if isinstance(idx, int) and 0 <= idx < len(outputs) and res_item.get("status") == "success":
                    b64 = res_item.get("audio_data")
                    if isinstance(b64, str) and b64:
                        wav_data = base64.b64decode(b64)
                        outputs[idx] = trim_trailing_silence_wav(wav_data) if trim_silence else wav_data

    # Sequential fallback for batch_size==1 or any failed batch item
    for idx, ch in enumerate(batch_chunks):
        if outputs[idx] is not None:
            continue
        single_payload = build_single_payload(ch)
        last_err: Exception | None = None
        for attempt in range(1, 3):
            try:
                wav_data, _ = _http_speech_bytes(base_url, token, single_payload, timeout=timeout)
                outputs[idx] = trim_trailing_silence_wav(wav_data) if trim_silence else wav_data
                last_err = None
                break
            except Exception as err:
                last_err = err
                if attempt < 2:
                    time.sleep(1.5)
        if outputs[idx] is None:
            raise RuntimeError(f"Chunk {ch['index']} failed synthesis: {last_err}")

    return [b for b in outputs if b is not None]


def cmd_narrate(args: argparse.Namespace) -> int:
    in_path = Path(args.input_file).expanduser()
    if not in_path.is_file():
        die(f"Manuscript file not found: {in_path}")
    raw_text = in_path.read_text(encoding="utf-8")
    if not raw_text.strip():
        die(f"Manuscript file is empty: {in_path}")

    if args.batch_size < 1 or args.batch_size > 32:
        die("--batch-size must be between 1 and 32.")

    cast_cfg: dict[str, Any] | None = None
    if args.cast:
        cast_path = Path(args.cast).expanduser()
        if not cast_path.is_file():
            die(f"Cast JSON file not found: {cast_path}")
        try:
            cast_cfg = json.loads(cast_path.read_text(encoding="utf-8"))
        except Exception as err:
            die(f"Invalid --cast JSON in {cast_path}: {err}")

    llm_client = _resolve_llm_client(args)
    split_dialogue = bool(
        args.split_dialogue
        or llm_client is not None
        or (cast_cfg is not None and not args.no_split_dialogue)
    )
    clone_fields = _build_voice_cloning_fields(args.ref_audio, args.ref_text, is_drive=args.ref_drive)

    if args.single_chapter:
        chapters = [{
            "index": 1,
            "title": in_path.stem,
            "slug": f"001-{in_path.stem[:48]}",
            "text": raw_text,
        }]
    else:
        chapters = split_into_chapters(raw_text)
    if not chapters:
        die(f"No readable text found in {in_path}")

    try:
        selected_indices = parse_chapter_selection(args.chapters, len(chapters))
    except ValueError as err:
        die(str(err))
    target_chapters = [ch for ch in chapters if ch["index"] in selected_indices]
    if not target_chapters:
        die(f"No chapters matched `--chapters {args.chapters}` (manuscript has {len(chapters)} chapters).")

    base_url, token = _get_endpoint(args.url, args.api_token)
    out_dir = Path(args.output_dir or f"./{in_path.stem}-audiobook").expanduser()
    chunks_root = out_dir / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_root.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    manifest: dict[str, Any] = {
        "source_file": str(in_path),
        "format": args.format,
        "max_chars": args.max_chars,
        "batch_size": args.batch_size,
        "speed": args.speed,
        "voice": args.voice,
        "split_dialogue": split_dialogue,
        "total_chapters": len(chapters),
        "chapters": {},
    }
    if manifest_path.is_file() and not args.force:
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and isinstance(prev.get("chapters"), dict):
                manifest["chapters"] = prev["chapters"]
        except Exception:
            pass

    out_fmt = args.format.lower()
    trim_silence = not bool(args.no_trim_silence)
    t_start = time.monotonic()
    total_audio_sec = 0.0
    total_synth_chunks = 0
    completed_outputs: list[dict[str, Any]] = []

    mode_tags = []
    if split_dialogue:
        mode_tags.append("split_dialogue=on")
    if llm_client:
        mode_tags.append(f"llm={llm_client['model']}")
    mode_suffix = f", {', '.join(mode_tags)}" if mode_tags else ""
    log(
        f"Narrating {len(target_chapters)}/{len(chapters)} chapter(s) from {in_path.name} -> {out_dir} "
        f"(max_chars={args.max_chars}, batch_size={args.batch_size}, format={out_fmt}{mode_suffix})"
    )

    for ch in target_chapters:
        ch_idx = ch["index"]
        slug = ch["slug"]
        final_file = out_dir / f"{slug}.{out_fmt}"
        ch_key = str(ch_idx)
        prev_rec = manifest["chapters"].get(ch_key) or {}

        if not args.force and prev_rec.get("status") == "completed" and final_file.is_file():
            dur_s = float(prev_rec.get("duration_sec") or 0.0)
            total_audio_sec += dur_s
            completed_outputs.append({
                "chapter": ch_idx,
                "title": ch["title"],
                "output": str(final_file),
                "duration_sec": round(dur_s, 2),
                "voice_chunks": prev_rec.get("voice_chunks") or {},
                "resumed": True,
            })
            log(f"[Chapter {ch_idx}/{len(chapters)}] Skipping completed {final_file.name} ({dur_s:.1f}s)")
            continue

        chapter_source_text = ch["text"]
        if llm_client is not None:
            chapter_source_text = _annotate_chapter_via_llm(
                chapter_source_text,
                slug,
                out_dir,
                llm_client,
                cast_cfg,
                add_tags=bool(args.llm_tags),
                force=bool(args.force),
                timeout=args.timeout,
            )

        try:
            chunks = split_text_chunks(
                chapter_source_text,
                max_chars=args.max_chars,
                split_dialogue=split_dialogue,
                cast=cast_cfg,
            )
        except ValueError as err:
            die(str(err))
        if not chunks:
            continue

        voice_counts: dict[str, int] = {}
        for c in chunks:
            vspec = resolve_chunk_voice_spec(c, default_voice=args.voice, cast=cast_cfg)
            c["voice_spec"] = vspec
            eff_speed = float(vspec.get("speed") if vspec.get("speed") is not None else args.speed)
            c["max_new_tokens"] = estimate_max_new_tokens(c["text"], speed=eff_speed)
            v_label = vspec.get("voice") or (Path(vspec["ref_audio"]).stem if vspec.get("ref_audio") else (c.get("role_hint") or "default"))
            voice_counts[v_label] = voice_counts.get(v_label, 0) + 1

        ch_chunk_dir = chunks_root / slug
        ch_chunk_dir.mkdir(parents=True, exist_ok=True)

        pending_chunks: list[dict[str, Any]] = []
        for c in chunks:
            c_wav = ch_chunk_dir / f"chunk_{c['index']:04d}.wav"
            c["wav_path"] = c_wav
            if args.force or not (c_wav.is_file() and c_wav.stat().st_size > 128):
                pending_chunks.append(c)

        cast_summary = ", ".join(f"{k}={v}" for k, v in voice_counts.items())
        log(
            f"[Chapter {ch_idx}/{len(chapters)}] {ch['title']} — {len(chunks)} chunks "
            f"({len(chunks) - len(pending_chunks)} cached, {len(pending_chunks)} to synthesize; voices: {cast_summary})"
        )

        for b_off in range(0, len(pending_chunks), args.batch_size):
            batch_slice = pending_chunks[b_off : b_off + args.batch_size]
            b_t0 = time.monotonic()
            try:
                wav_list = _synthesize_narration_batch(
                    base_url,
                    token,
                    batch_slice,
                    model=args.model,
                    default_voice=args.voice,
                    cast=cast_cfg,
                    clone_fields=clone_fields,
                    ref_drive=bool(args.ref_drive),
                    speed=float(args.speed),
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    trim_silence=trim_silence,
                    use_batch_endpoint=args.batch_size > 1,
                    timeout=args.timeout,
                )
            except Exception as err:
                manifest["chapters"][ch_key] = {
                    "index": ch_idx,
                    "title": ch["title"],
                    "slug": slug,
                    "status": "partial",
                    "chunks_total": len(chunks),
                    "voice_chunks": voice_counts,
                    "error": str(err),
                }
                _write_json_atomic(manifest_path, manifest)
                die(f"Narration interrupted in chapter {ch_idx} ({ch['title']}): {err}\nRe-run the same command to resume.")

            b_dur = 0.0
            for c_meta, wav_bytes in zip(batch_slice, wav_list):
                c_meta["wav_path"].write_bytes(wav_bytes)
                b_dur += wav_duration_seconds(wav_bytes)
            b_elapsed = time.monotonic() - b_t0
            total_synth_chunks += len(batch_slice)
            done_now = (len(chunks) - len(pending_chunks)) + b_off + len(batch_slice)
            speed_x = (b_dur / b_elapsed) if b_elapsed > 0 else 0.0
            log(
                f"  [{done_now}/{len(chunks)}] batch={len(batch_slice)} audio={b_dur:.1f}s "
                f"wall={b_elapsed:.2f}s ({speed_x:.1f}x realtime)"
            )

        segments: list[tuple[bytes, int]] = []
        for c in chunks:
            if c["paragraph_end"]:
                pause = args.para_pause_ms
            elif c.get("turn_transition"):
                pause = args.turn_pause_ms
            else:
                pause = args.pause_ms
            segments.append((c["wav_path"].read_bytes(), pause))

        stitched_wav = concat_wav_segments(segments, default_sample_rate=DRAMA3_SAMPLE_RATE)
        ch_dur_sec = wav_duration_seconds(stitched_wav)
        total_audio_sec += ch_dur_sec

        stitched_wav_path = out_dir / f"{slug}.wav"
        stitched_wav_path.write_bytes(stitched_wav)
        if out_fmt != "wav":
            _convert_wav_file(stitched_wav_path, final_file, out_fmt)
            stitched_wav_path.unlink(missing_ok=True)

        if args.clean_chunks:
            shutil.rmtree(ch_chunk_dir, ignore_errors=True)

        manifest["chapters"][ch_key] = {
            "index": ch_idx,
            "title": ch["title"],
            "slug": slug,
            "status": "completed",
            "chunks_total": len(chunks),
            "voice_chunks": voice_counts,
            "duration_sec": round(ch_dur_sec, 2),
            "output": final_file.name,
            "updated_at": time.strftime("%FT%T%z"),
        }
        _write_json_atomic(manifest_path, manifest)
        completed_outputs.append({
            "chapter": ch_idx,
            "title": ch["title"],
            "output": str(final_file),
            "chunks": len(chunks),
            "voice_chunks": voice_counts,
            "duration_sec": round(ch_dur_sec, 2),
            "resumed": False,
        })
        log(f"[Chapter {ch_idx}/{len(chapters)}] Saved {final_file} ({ch_dur_sec / 60.0:.1f} min)")

    if args.clean_chunks:
        shutil.rmtree(chunks_root, ignore_errors=True)

    wall_total = time.monotonic() - t_start
    summary = {
        "input_file": str(in_path),
        "output_dir": str(out_dir),
        "manifest": str(manifest_path),
        "chapters_completed": len(completed_outputs),
        "chunks_synthesized": total_synth_chunks,
        "total_duration_sec": round(total_audio_sec, 2),
        "elapsed_sec": round(wall_total, 2),
        "outputs": completed_outputs,
    }
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        log(
            f"Narration complete: {len(completed_outputs)} chapter(s), "
            f"{total_audio_sec / 60.0:.2f} min total audio in {wall_total:.1f}s -> {out_dir}"
        )
    return 0


def _add_endpoint_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--url", help="Override public service URL")
    sp.add_argument("--api-token", help="Override Bearer API token")


def _build_custom_parser(parser=None, commands=None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(prog="colabllm drama3")
    sub = commands or p.add_subparsers(dest="cmd", required=True)

    for tts_name in ("tts", "speak"):
        sp = sub.add_parser(tts_name, help="Synthesize speech via /v1/audio/speech (zero-shot, multi-ref, or named voice)")
        sp.add_argument("text", nargs="?", help="Text to synthesize (supports inline tags like [sighs], [laughs])")
        sp.add_argument("--text", dest="text_opt", help="Text to synthesize (alternative to positional arg)")
        sp.add_argument("--text-file", help="Read text to synthesize from a local UTF-8 file")
        sp.add_argument("-o", "--output", default=None, help="Local output audio file path (default: ./drama3-<timestamp>.<format>)")
        sp.add_argument("--format", default="wav", choices=["wav", "mp3", "flac", "pcm", "aac", "opus"], help="Output audio format (default: wav)")
        sp.add_argument("--voice", help="Registered voice name from /v1/audio/voices (e.g. narrator, alloy)")
        sp.add_argument("--ref-audio", action="append", help="Reference audio for zero-shot cloning (repeatable for multi-reference; local file, drive://, or URL)")
        sp.add_argument("--ref-text", action="append", help="Transcript matching each --ref-audio (repeatable)")
        sp.add_argument("--ref-drive", action="store_true", help="Interpret --ref-audio paths as Google Drive paths")
        sp.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier in [0.25, 4.0] (default: 1.0)")
        sp.add_argument("--temperature", type=float, help="Sampling temperature override")
        sp.add_argument("--top-p", type=float, help="Nucleus sampling top_p override")
        sp.add_argument("--top-k", type=int, help="Top-k sampling override")
        sp.add_argument("--max-new-tokens", type=int, help="Explicit codec frame cap (1..2048; auto-estimated from text length by default)")
        sp.add_argument("--no-auto-max-tokens", action="store_true", help="Do not auto-estimate max_new_tokens when --max-new-tokens is omitted")
        sp.add_argument("--no-trim-silence", action="store_true", help="Disable trailing silence trimming on WAV output")
        sp.add_argument("--stream", action="store_true", help="Stream raw PCM chunks and wrap into WAV/PCM locally")
        sp.add_argument("--save-drive", nargs="?", const="auto", help="Also save output audio to Google Drive (optional path or 'auto' for colab-llm/drama3/outputs/)")
        sp.add_argument("--model", default="drama3", help="Model name or alias (default: drama3)")
        sp.add_argument("--timeout", type=nonnegative_seconds, default=300, help="HTTP request timeout in seconds; 0 = unlimited (default: 300)")
        _add_endpoint_flags(sp)

    np = sub.add_parser("narrate", help="Narrate a full novel or long text into chapter audio files using batch synthesis")
    np.add_argument("input_file", help="Local UTF-8 .txt or .md manuscript to narrate")
    np.add_argument("-O", "--output-dir", default=None, help="Output directory for chapter audio and manifest.json (default: ./<book>-audiobook)")
    np.add_argument("--voice", help="Registered voice name to use (recommended for fast batch narration)")
    np.add_argument("--ref-audio", action="append", help="Zero-shot reference audio (repeatable; local file, drive://, or URL)")
    np.add_argument("--ref-text", action="append", help="Transcript matching each --ref-audio (repeatable)")
    np.add_argument("--ref-drive", action="store_true", help="Interpret --ref-audio paths as Google Drive paths")
    np.add_argument("--cast", help="Optional JSON file mapping narrator/dialogue/characters to voices, speeds, or ref_audio")
    np.add_argument("--split-dialogue", action="store_true", help="Split quoted dialogue and narration inside paragraphs (auto-enabled with --cast or --llm-*)")
    np.add_argument("--no-split-dialogue", action="store_true", help="Disable automatic intra-paragraph dialogue splitting when --cast is set")
    np.add_argument("--llm-service", help="Optional running colabllm LLM service (e.g. qwen-flash, deepseek-coder) for speaker script annotation")
    np.add_argument("--llm-url", help="Optional OpenAI-compatible base URL (e.g. https://.../v1 or http://127.0.0.1:11434/v1) for speaker annotation")
    np.add_argument("--llm-token", help="Bearer API token for --llm-url (or set DRAMA3_LLM_API_TOKEN / OPENAI_API_KEY)")
    np.add_argument("--llm-model", help="Model ID for --llm-service / --llm-url (auto-discovered from /v1/models when omitted)")
    np.add_argument("--llm-tags", action="store_true", help="Allow LLM preprocessor to insert sparse expressive tags ([sighs], [laughs], [gasp])")
    np.add_argument("--chapters", help="1-based chapter filter, e.g. '1', '1-5', or '1,3,8-10' (default: all)")
    np.add_argument("--single-chapter", action="store_true", help="Treat the entire file as a single chapter without heading detection")
    np.add_argument("--max-chars", type=int, default=DEFAULT_CHUNK_MAX_CHARS, help=f"Max characters per synthesis chunk in [20, 500] (default: {DEFAULT_CHUNK_MAX_CHARS})")
    np.add_argument("--batch-size", type=int, default=8, help="Chunks per /v1/audio/speech/batch call in [1, 32] (default: 8)")
    np.add_argument("--turn-pause-ms", type=int, default=180, help="Silence between narration attribution and quoted dialogue within a paragraph in ms (default: 180)")
    np.add_argument("--pause-ms", type=int, default=280, help="Silence inserted between chunks in the same turn in ms (default: 280)")
    np.add_argument("--para-pause-ms", type=int, default=650, help="Silence inserted between paragraphs in ms (default: 650)")
    np.add_argument("--format", default="wav", choices=["wav", "mp3", "flac", "opus"], help="Final chapter output format (default: wav)")
    np.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier in [0.25, 4.0] (default: 1.0)")
    np.add_argument("--temperature", type=float, help="Sampling temperature override")
    np.add_argument("--top-p", type=float, help="Nucleus sampling top_p override")
    np.add_argument("--top-k", type=int, help="Top-k sampling override")
    np.add_argument("--no-trim-silence", action="store_true", help="Disable trailing silence trimming on chunk WAVs")
    np.add_argument("--clean-chunks", action="store_true", help="Delete intermediate per-chunk WAV files once each chapter is stitched")
    np.add_argument("--force", action="store_true", help="Re-synthesize chunks and chapters even if already completed in manifest.json")
    np.add_argument("--json", action="store_true", help="Output final narration summary as JSON on stdout")
    np.add_argument("--model", default="drama3", help="Model name or alias (default: drama3)")
    np.add_argument("--timeout", type=nonnegative_seconds, default=600, help="HTTP timeout per batch request in seconds; 0 = unlimited (default: 600)")
    _add_endpoint_flags(np)

    vp = sub.add_parser("voices", help="List registered voices (/v1/audio/voices)")
    vp.add_argument("--json", action="store_true", help="Output raw JSON")
    _add_endpoint_flags(vp)

    upv = sub.add_parser("upload-voice", help="Register a persistent named voice from local or Google Drive audio")
    upv.add_argument("--name", required=True, help="Voice identifier (1-64 lowercase [a-z0-9._-] chars, e.g. narrator)")
    upv.add_argument("--audio", help="Local audio file path (or drive:// path)")
    upv.add_argument("--drive", help="Google Drive audio path (relative to colab-llm/drama3/voices/ or MyDrive)")
    upv.add_argument("--consent", default="owner-consent", help="Consent tag (default: owner-consent)")
    upv.add_argument("--ref-text", help="Transcript of the reference audio (improves prosody & alignment)")
    upv.add_argument("--description", help="Optional speaker description")
    _add_endpoint_flags(upv)

    delv = sub.add_parser("delete-voice", help="Delete a registered voice (/v1/audio/voices/{name})")
    delv.add_argument("name", help="Voice name to delete")
    _add_endpoint_flags(delv)

    lsd = sub.add_parser("ls-drive", help="List audio files on Google Drive")
    lsd.add_argument("path", nargs="?", default="", help="Drive folder or glob pattern (default: colab-llm/drama3/voices)")
    lsd.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    lsd.add_argument("--json", action="store_true", help="Output raw JSON")
    _add_endpoint_flags(lsd)

    return p


CUSTOM_CMDS = {"tts", "speak", "narrate", "voices", "upload-voice", "delete-voice", "ls-drive"}


def configure_cli(parser, commands, up):
    _build_custom_parser(parser, commands)
    for flag, env in UP_ENV_FLAGS.items():
        up.add_argument(flag, dest="up_" + env, help=f"override {env}")


_BASE_ENV = dict(SERVICE.env)


def prepare_up(args):
    # Reset baseline defaults on every invocation so overrides never leak across calls.
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    for env in UP_ENV_FLAGS.values():
        value = getattr(args, "up_" + env, None)
        if value is not None:
            SERVICE.env[env] = str(value)


if __name__ == "__main__":
    from cli_support import run_cli

    run_cli(
        lambda: main(
            SERVICE,
            configure=configure_cli,
            prepare_up=prepare_up,
            handlers={
                "tts": cmd_tts,
                "speak": cmd_tts,
                "narrate": cmd_narrate,
                "voices": cmd_voices,
                "upload-voice": cmd_upload_voice,
                "delete-voice": cmd_delete_voice,
                "ls-drive": cmd_ls_drive,
            },
        )
    )
