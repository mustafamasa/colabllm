#!/usr/bin/env python3
"""IndexTTS-2.5 (vLLM-Omni 2-stage zero-shot voice cloning & emotion TTS on Colab L4 GPU).

Follows https://recipes.vllm.ai/IndexTeam/IndexTTS-2.5:
  - Stage 0 (`indextts2_5_talker`): GPT-2 24-layer 1280-dim AR talker + Wav2Vec2-BERT 2.0 +
    CAMPPlus speaker encoder + Qwen-0.6B emotion analyzer (TRITON_ATTN).
  - Stage 1 (`indextts2_5_s2mel_decoder`): EnhancedCodec + S2Mel DiT (BF16, torch.compile,
    CFM batch size 4) + BigVGAN v2 22kHz vocoder.
  - Native speed / duration control (`0.5` - `2.0`), 7 language modes (`zh`, `en`, `zhen`,
    `ja`, `es`, `ar`, `yue`), 8-dim emotion vector / emotion text / emotion audio control,
    zero-shot `ref_audio` cloning, and persistent uploaded voices (`/v1/audio/voices`) backed
    by Google Drive.

Usage:
  colabllm indextts up                              # L4 GPU + quick tunnel
  colabllm indextts up --domain                     # fixed domain indextts.example.com
  colabllm indextts status
  colabllm indextts voices                          # list registered voices
  colabllm indextts upload-voice --name narrator --audio sample.wav
  colabllm indextts upload-voice --name narrator --drive voices/sample.wav
  colabllm indextts tts "The weather is lovely today, let's take a walk in the park." -o out.wav
  colabllm indextts tts "Hello, this is zero-shot voice cloning!" --ref-audio sample.wav -o clone.wav
  colabllm indextts tts "What a wonderful surprise!" --emo-preset happy --emo-alpha 0.8 -o happy.wav
  colabllm indextts down
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cli_support import http_timeout, nonnegative_seconds, resolve_endpoint, safe_urlopen
from colabsvc import (  # noqa: E402
    Service,
    State,
    die,
    log,
    main,
)

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

SERVICE = Service(
    name="indextts",
    notebook="indextts.ipynb",
    service_port=8000,
    auth="builtin",
    public_port=8000,
    gpu="L4",
    high_mem=False,
    default_hostname="indextts",
    ready_timeout=3600,
    env={
        "CT_INDEXTTS_MODEL": "IndexTeam/IndexTTS-2.5",
        "CT_INDEXTTS_ALIAS": "indextts-2.5",
        "CT_INDEXTTS_GPU_MEM_STAGE0": "0.4",
        "CT_INDEXTTS_GPU_MEM_STAGE1": "0.4",
        "CT_INDEXTTS_MAX_MODEL_LEN": "2560",
        "CT_INDEXTTS_MAX_NUM_SEQS": "4",
    },
)

EMO_PRESETS: dict[str, list[float]] = {
    "happy":       [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "angry":       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "sad":         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "afraid":      [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    "disgusted":   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "melancholic": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    "surprised":   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    "calm":        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
}


UP_ENV_FLAGS = {'--model': 'CT_INDEXTTS_MODEL',
 '--alias': 'CT_INDEXTTS_ALIAS',
 '--stage0-gpu-mem': 'CT_INDEXTTS_GPU_MEM_STAGE0',
 '--stage1-gpu-mem': 'CT_INDEXTTS_GPU_MEM_STAGE1',
 '--max-model-len': 'CT_INDEXTTS_MAX_MODEL_LEN',
 '--max-seqs': 'CT_INDEXTTS_MAX_NUM_SEQS'}


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(State(SERVICE).load(), url_override, token_override,
                            url_env="INDEXTTS_URL", token_env="INDEXTTS_API_TOKEN")


def _http_json(
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    method: str | None = None,
    timeout: int = 120,
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
        with safe_urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else {"raw": raw})
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        try:
            return err.code, json.loads(raw_err)
        except Exception:
            return err.code, {"error": raw_err}


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


def _auto_detect_lang(text: str) -> str:
    has_zh = bool(re.search(r"[\u4e00-\u9fff]", text))
    has_ja = bool(re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text))
    has_ar = bool(re.search(r"[\u0600-\u06ff]", text))
    has_en = bool(re.search(r"[A-Za-z]", text))
    if has_ja:
        return "ja"
    if has_ar:
        return "ar"
    if has_zh and has_en:
        return "zhen"
    if has_zh:
        return "zh"
    if has_en:
        return "en"
    return "zh"


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
            desc = f" — {v.get('speaker_description')}" if v.get("speaker_description") else ""
            print(f"  - {v.get('id')} (size={v.get('file_size', 0)} bytes, mime={v.get('mime_type', '')}){desc}")
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
        code, out = _http_json(f"{base_url}/api/ct/drive-upload-voice", token, payload=payload, timeout=90)
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if code == 200 else 1

    audio_path = Path(args.audio).expanduser()
    if not audio_path.is_file():
        die(f"Audio file not found: {audio_path}")

    mime, _ = mimetypes.guess_type(str(audio_path))
    if not mime or not mime.startswith("audio/"):
        mime = "audio/wav"

    boundary = f"----indexttsvoice{int(time.time() * 1000)}"
    parts: list[bytes] = []

    def add_field(k: str, v: str) -> None:
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode("utf-8")
        )

    add_field("name", args.name)
    add_field("consent", args.consent)
    if args.ref_text:
        add_field("ref_text", args.ref_text)
    if args.description:
        add_field("speaker_description", args.description)

    parts.append(
        (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"audio_sample\"; filename=\"{audio_path.name}\"\r\n"
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
        + audio_path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode("utf-8")
    )
    body_bytes = b"".join(parts)
    req = urllib.request.Request(
        f"{base_url}/v1/audio/voices",
        data=body_bytes,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": BROWSER_UA,
        },
        method="POST",
    )
    try:
        with safe_urlopen(req, timeout=90) as resp:
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
    if not text:
        die("Provide text to synthesize (positional `text`, `--text`, or `--text-file`).")

    lang = (args.lang or "").strip() or _auto_detect_lang(text)
    extra_params: dict[str, Any] = {
        "lang": lang,
        "text_normalization": not bool(args.no_text_norm),
    }

    if args.emo_preset:
        extra_params["emo_vector"] = list(EMO_PRESETS[args.emo_preset])
    if args.emo_vector:
        vals = [float(x) for x in re.split(r"[\s,]+", args.emo_vector.strip()) if x]
        if len(vals) != 8:
            die(
                "--emo-vector requires 8 numbers in [0, 1.2]: "
                "[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]."
            )
        extra_params["emo_vector"] = vals
    if args.emo_text:
        extra_params["use_emo_text"] = True
        extra_params["emo_text"] = args.emo_text
    if args.emo_audio:
        extra_params["emo_audio"] = _resolve_audio_spec(args.emo_audio, is_drive=args.emo_audio_drive)
    if args.emo_alpha is not None:
        extra_params["emo_alpha"] = float(args.emo_alpha)
    elif any(k in extra_params for k in ("emo_vector", "emo_text", "emo_audio")):
        extra_params["emo_alpha"] = 0.6
    if args.use_random:
        extra_params["use_random"] = True

    payload: dict[str, Any] = {
        "model": args.model,
        "input": text,
        "response_format": args.format,
        "speed": float(args.speed),
        "extra_params": extra_params,
    }
    if args.ref_audio:
        payload["ref_audio"] = _resolve_audio_spec(args.ref_audio, is_drive=args.ref_drive)
    elif args.voice:
        payload["voice"] = args.voice
    else:
        payload["voice"] = "default"

    if args.max_new_tokens:
        payload["max_new_tokens"] = int(args.max_new_tokens)
    if args.seed is not None:
        payload["seed"] = int(args.seed)
    if args.save_drive:
        payload["save_to_drive"] = args.save_drive

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": BROWSER_UA,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    t0 = time.monotonic()
    try:
        with safe_urlopen(req, timeout=http_timeout(args.timeout)) as resp:
            audio_bytes = resp.read()
            elapsed = time.monotonic() - t0
            saved_drive = resp.headers.get("X-Saved-Drive-Path")
            sample_rate = resp.headers.get("X-Audio-Sample-Rate") or "22050"
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        print(f"[indextts] HTTP {err.code} from /v1/audio/speech:\n{raw_err}", file=sys.stderr)
        return 1

    out_str = args.output or f"indextts-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    out_path = Path(out_str).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    msg = (
        f"Synthesized {len(audio_bytes):,} bytes ({args.format}, {sample_rate} Hz, "
        f"lang={lang}, speed={args.speed}x) in {elapsed:.2f}s -> {out_path}"
    )
    if saved_drive:
        msg += f" (Drive: {saved_drive})"
    log(msg)
    return 0


def _add_endpoint_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--url", help="Override public service URL")
    sp.add_argument("--api-token", help="Override Bearer API token")


def _build_custom_parser(parser=None, commands=None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(prog="colabllm indextts")
    sub = commands or p.add_subparsers(dest="cmd", required=True)

    for tts_name in ("tts", "speak"):
        sp = sub.add_parser(tts_name, help="Synthesize speech via /v1/audio/speech (zero-shot or named voice)")
        sp.add_argument("text", nargs="?", help="Text to synthesize (supports <word|PRONUNCIATION> annotations)")
        sp.add_argument("--text", dest="text_opt", help="Text to synthesize (alternative to positional arg)")
        sp.add_argument("--text-file", help="Read text to synthesize from a local UTF-8 file")
        sp.add_argument("-o", "--output", default=None, help="Local output audio file path (default: ./indextts-<timestamp>.wav)")
        sp.add_argument("--format", default="wav", choices=["wav", "mp3", "flac", "opus", "pcm"], help="Output audio format (default: wav)")
        sp.add_argument("--voice", help="Uploaded/preset voice name (default: 'default' when --ref-audio is omitted)")
        sp.add_argument("--ref-audio", help="Reference audio for zero-shot voice cloning (local file, drive:// path, or URL)")
        sp.add_argument("--ref-drive", action="store_true", help="Interpret --ref-audio as a Google Drive path")
        sp.add_argument("--lang", choices=["zh", "en", "zhen", "ja", "es", "ar", "yue"], help="Language code (auto-detected from text if omitted)")
        sp.add_argument("--speed", type=float, default=1.0, help="Native speech speed multiplier in [0.5, 2.0] (default: 1.0)")
        sp.add_argument("--emo-preset", choices=sorted(EMO_PRESETS.keys()), help="Named emotion preset (happy, angry, sad, afraid, disgusted, melancholic, surprised, calm)")
        sp.add_argument("--emo-vector", help="8 comma-separated emotion weights in [0, 1.2]: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm")
        sp.add_argument("--emo-text", help="Emotion text description analyzed by Qwen-0.6B emotion module")
        sp.add_argument("--emo-audio", help="Emotion reference audio (local file, drive:// path, or URL)")
        sp.add_argument("--emo-audio-drive", action="store_true", help="Interpret --emo-audio as a Google Drive path")
        sp.add_argument("--emo-alpha", type=float, help="Emotion blending alpha in [0.0, 1.0] (default: 0.6 when emotion is set)")
        sp.add_argument("--no-text-norm", action="store_true", help="Disable WeTextProcessing number/date normalization")
        sp.add_argument("--use-random", action="store_true", help="Enable stochastic speaker/emotion sampling")
        sp.add_argument("--max-new-tokens", type=int, help="Maximum codec tokens to generate")
        sp.add_argument("--seed", type=int, help="Random seed for reproducible generation")
        sp.add_argument("--save-drive", nargs="?", const="auto", help="Also save output audio to Google Drive (optional path or 'auto' for colab-llm/indextts/outputs/)")
        sp.add_argument("--model", default="IndexTeam/IndexTTS-2.5", help="Model name (default: IndexTeam/IndexTTS-2.5)")
        sp.add_argument("--timeout", type=nonnegative_seconds, default=300, help="HTTP request timeout in seconds; 0 = unlimited (default: 300)")
        _add_endpoint_flags(sp)

    vp = sub.add_parser("voices", help="List available and uploaded voices (/v1/audio/voices)")
    vp.add_argument("--json", action="store_true", help="Output raw JSON")
    _add_endpoint_flags(vp)

    upv = sub.add_parser("upload-voice", help="Register a persistent named voice from local or Google Drive audio")
    upv.add_argument("--name", required=True, help="Voice identifier (e.g. narrator, alice)")
    upv.add_argument("--audio", help="Local audio file path (or drive:// path)")
    upv.add_argument("--drive", help="Google Drive audio path (relative to colab-llm/indextts/voices/ or MyDrive)")
    upv.add_argument("--consent", default="owner-consent", help="Consent tag (default: owner-consent)")
    upv.add_argument("--ref-text", help="Optional transcript of the reference audio")
    upv.add_argument("--description", help="Optional description of the speaker voice")
    _add_endpoint_flags(upv)

    delv = sub.add_parser("delete-voice", help="Delete an uploaded voice (/v1/audio/voices/{name})")
    delv.add_argument("name", help="Voice name to delete")
    _add_endpoint_flags(delv)

    lsd = sub.add_parser("ls-drive", help="List audio files on Google Drive")
    lsd.add_argument("path", nargs="?", default="", help="Drive folder or glob pattern (default: colab-llm/indextts/voices)")
    lsd.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    lsd.add_argument("--json", action="store_true", help="Output raw JSON")
    _add_endpoint_flags(lsd)

    return p


CUSTOM_CMDS = {"tts", "speak", "voices", "upload-voice", "delete-voice", "ls-drive"}



def configure_cli(parser, commands, up):
    _build_custom_parser(parser, commands)
    for flag, env in UP_ENV_FLAGS.items():
        up.add_argument(flag, dest="up_" + env, help=f"override {env}")


_BASE_ENV = dict(SERVICE.env)


def prepare_up(args):
    # Start from the baseline defaults on every call so overrides never leak between invocations.
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    for flag, env in UP_ENV_FLAGS.items():
        value = getattr(args, "up_" + env)
        if value is not None:
            SERVICE.env[env] = value


if __name__ == "__main__":
    from cli_support import run_cli
    run_cli(lambda: main(SERVICE, configure=configure_cli, prepare_up=prepare_up,
                        handlers={"tts": cmd_tts, "speak": cmd_tts, "voices": cmd_voices,
                "upload-voice": cmd_upload_voice, "delete-voice": cmd_delete_voice, "ls-drive": cmd_ls_drive}))
