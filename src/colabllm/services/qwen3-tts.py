#!/usr/bin/env python3
"""Qwen3-TTS (vLLM-Omni 2-stage CustomVoice, VoiceDesign & 3s Zero-Shot Voice Cloning on Colab L4 GPU).

Follows https://github.com/QwenLM/Qwen3-TTS and vLLM-Omni's Qwen3-TTS pipeline:
  - Stage 0 (`qwen3_tts`): Qwen3-TTS-12Hz 1.7B / 0.6B Multi-Codebook Talker + CodePredictor + ECAPA-TDNN SpeakerEncoder.
  - Stage 1 (`code2wav`): Qwen3-TTS-Tokenizer-12Hz (16 codebooks @ 12.5 Hz -> 24 kHz waveform).
  - Auxiliary multi-task GPU worker (`127.0.0.1:8001`) so all 4 official Qwen3-TTS workflows work
    seamlessly on a single deployment without restarting:
      1. CustomVoice (`Qwen3-TTS-12Hz-1.7B-CustomVoice`): 9 built-in premium timbres (`Vivian`,
         `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`) + natural-language
         emotion/style/rate `instructions`.
      2. VoiceDesign (`Qwen3-TTS-12Hz-1.7B-VoiceDesign`): create brand-new voices from natural-language
         persona descriptions (`--voice-design` / `--task VoiceDesign --instruct "..."`).
      3. VoiceClone (`Qwen3-TTS-12Hz-1.7B-Base`): 3-second rapid zero-shot voice cloning (`--ref-audio`
         + optional `--ref-text` for ICL mode or automatic X-Vector mode) + persistent uploaded voices.
      4. Voice Design then Clone (`design-voice`): design a persona voice with `VoiceDesign`, save the
         reference clip to Google Drive, and register it as a reusable cloned voice (`--name <voice>`).
  - Supports 10 languages (`Chinese`/`zh`, `English`/`en`, `Japanese`/`ja`, `Korean`/`ko`,
    `German`/`de`, `French`/`fr`, `Russian`/`ru`, `Portuguese`/`pt`, `Spanish`/`es`, `Italian`/`it`, `Auto`).

Usage:
  colabllm qwen3-tts up                              # L4 GPU + quick tunnel
  colabllm qwen3-tts up --domain                     # fixed domain qwen3-tts.example.com
  colabllm qwen3-tts status
  colabllm qwen3-tts voices                          # list 9 built-in + uploaded/designed voices
  colabllm qwen3-tts tts "I've noticed that I'm someone who is really good at reading other people's emotions." --voice vivian --instruct "Say it in a very happy tone" -o out.wav
  colabllm qwen3-tts tts "You're back! I've been waiting for you for so long!" --voice-design "A cute, youthful girl's voice with a high pitch and lively intonation" -o design.wav
  colabllm qwen3-tts tts "Hello, this is 3-second zero-shot voice cloning!" --ref-audio sample.wav -o clone.wav
  colabllm qwen3-tts design-voice --instruct "A calm, magnetic middle-aged male announcer voice" --name announcer -o announcer_ref.wav
  colabllm qwen3-tts tts "Welcome to today's news program." --voice announcer -o news.wav
  colabllm qwen3-tts down
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
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
    name="qwen3-tts",
    notebook="qwen3-tts.ipynb",
    service_port=8000,
    auth="builtin",
    public_port=8000,
    gpu="L4",
    high_mem=False,
    default_hostname="qwen3-tts",
    ready_timeout=3600,
    env={
        "CT_QWEN3_TTS_VARIANT": "customvoice",
        "CT_QWEN3_TTS_MODEL": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "CT_QWEN3_TTS_ALIAS": "qwen3-tts",
        "CT_QWEN3_TTS_PRELOAD_ALL": "1",
        "CT_QWEN3_TTS_GPU_MEM_STAGE0": "0.30",
        "CT_QWEN3_TTS_GPU_MEM_STAGE1": "0.20",
        "CT_QWEN3_TTS_MAX_MODEL_LEN": "4096",
        "CT_QWEN3_TTS_MAX_NUM_SEQS": "16",
    },
)

BUILTIN_SPEAKERS = {
    "vivian":   "Chinese — Bright, slightly edgy young female voice",
    "serena":   "Chinese — Warm, gentle young female voice",
    "uncle_fu": "Chinese — Seasoned male voice with a low, mellow timbre",
    "dylan":    "Chinese (Beijing Dialect) — Youthful Beijing male voice",
    "eric":     "Chinese (Sichuan Dialect) — Lively Chengdu male voice",
    "ryan":     "English — Dynamic male voice with strong rhythmic drive",
    "aiden":    "English — Sunny American male voice with a clear midrange",
    "ono_anna": "Japanese — Playful Japanese female voice with a light, nimble timbre",
    "sohee":    "Korean — Warm Korean female voice with rich emotion",
}


UP_ENV_FLAGS = {'--variant': 'CT_QWEN3_TTS_VARIANT',
 '--model': 'CT_QWEN3_TTS_MODEL',
 '--alias': 'CT_QWEN3_TTS_ALIAS',
 '--preload-all': 'CT_QWEN3_TTS_PRELOAD_ALL',
 '--stage0-gpu-mem': 'CT_QWEN3_TTS_GPU_MEM_STAGE0',
 '--stage1-gpu-mem': 'CT_QWEN3_TTS_GPU_MEM_STAGE1',
 '--max-model-len': 'CT_QWEN3_TTS_MAX_MODEL_LEN',
 '--max-seqs': 'CT_QWEN3_TTS_MAX_NUM_SEQS'}
VARIANT_REPOS = {'customvoice': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
 '1.7b-customvoice': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
 'voicedesign': 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
 '1.7b-voicedesign': 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
 'base': 'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
 '1.7b-base': 'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
 '0.6b-customvoice': 'Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice',
 '0.6b-base': 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'}


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(State(SERVICE).load(), url_override, token_override,
                            url_env="QWEN3_TTS_URL", token_env="QWEN3_TTS_API_TOKEN")


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
    log(f"Available voices ({len(voices)}): {', '.join(voices)}")
    print("\nBuilt-in CustomVoice speakers (9):")
    for spk, desc in BUILTIN_SPEAKERS.items():
        print(f"  - {spk:<10} : {desc}")
    if uploaded:
        print(f"\nUploaded / VoiceDesign custom voices ({len(uploaded)}):")
        for v in uploaded:
            desc = f" — {v.get('speaker_description')}" if v.get("speaker_description") else ""
            rt = f" [ICL ref_text={v.get('ref_text')!r}]" if v.get("ref_text") else " [x-vector mode]"
            print(f"  - {v.get('id')} (size={v.get('file_size', 0)} bytes){rt}{desc}")
    return 0


def cmd_upload_voice(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.api_token)
    if not args.audio and not args.drive:
        die("Provide either `--audio <local_file>` or `--drive <drive_path>`.")

    payload: dict[str, Any] = {
        "name": args.name.strip().lower(),
        "consent": args.consent,
    }
    if args.ref_text:
        payload["ref_text"] = args.ref_text
    if args.description:
        payload["speaker_description"] = args.description

    if args.drive or (args.audio and str(args.audio).startswith(("drive://", "gdrive://", "/content/drive/"))):
        payload["drive_path"] = args.drive or args.audio
    else:
        audio_path = Path(args.audio).expanduser()
        if not audio_path.is_file():
            die(f"Audio file not found: {audio_path}")
        payload["audio_base64"] = base64.b64encode(audio_path.read_bytes()).decode("ascii")

    code, out = _http_json(f"{base_url}/api/ct/drive-upload-voice", token, payload=payload, timeout=120)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if code == 200 else 1


def cmd_delete_voice(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.api_token)
    vname = urllib.parse.quote(args.name.strip().lower())
    code, out = _http_json(f"{base_url}/v1/audio/voices/{vname}", token, method="DELETE", timeout=30)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0 if code == 200 else 1


def cmd_design_voice(args: argparse.Namespace) -> int:
    """Voice Design then Clone: generate a persona reference clip via VoiceDesign and optionally register it."""
    base_url, token = _get_endpoint(args.url, args.api_token)
    payload: dict[str, Any] = {
        "instructions": args.instruct,
        "language": args.lang or "Auto",
    }
    if args.text:
        payload["sample_text"] = args.text
    if args.name:
        payload["name"] = args.name.strip().lower()
    if args.save_drive:
        payload["save_to_drive"] = args.save_drive

    t0 = time.monotonic()
    code, data = _http_json(f"{base_url}/api/ct/design-voice", token, payload=payload, timeout=http_timeout(args.timeout))
    elapsed = time.monotonic() - t0
    if code != 200:
        print(json.dumps(data, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    audio_b64 = data.pop("audio_base64", None)
    if audio_b64:
        audio_bytes = base64.b64decode(audio_b64)
        out_str = args.output or (
            f"qwen3-tts-design-{args.name.strip().lower()}.wav"
            if args.name
            else f"qwen3-tts-design-{time.strftime('%Y%m%d-%H%M%S')}.wav"
        )
        out_path = Path(out_str).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio_bytes)
        msg = f"Designed voice ({len(audio_bytes):,} bytes WAV, {data.get('sample_rate', 24000)} Hz) in {elapsed:.2f}s -> {out_path}"
        if data.get("registered_voice"):
            msg += f" | registered reusable voice: '{data['registered_voice']}'"
        if data.get("saved_drive_path"):
            msg += f" (Drive: {data['saved_drive_path']})"
        log(msg)
    return 0


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

    payload: dict[str, Any] = {
        "model": args.model,
        "input": text,
        "response_format": args.format,
        "speed": float(args.speed),
        "language": args.lang or "Auto",
    }

    if args.voice_design:
        payload["task_type"] = "VoiceDesign"
        payload["instructions"] = args.voice_design
    elif args.task:
        payload["task_type"] = args.task

    if args.instruct and not payload.get("instructions"):
        payload["instructions"] = args.instruct

    if args.ref_audio:
        payload["task_type"] = "Base"
        payload["ref_audio"] = _resolve_audio_spec(args.ref_audio, is_drive=args.ref_drive)
        if args.ref_text:
            payload["ref_text"] = args.ref_text
        if args.x_vector_only:
            payload["x_vector_only_mode"] = True
    elif args.voice:
        payload["voice"] = args.voice.strip().lower()

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
            sample_rate = resp.headers.get("X-Audio-Sample-Rate") or "24000"
            engine_tag = resp.headers.get("X-Qwen3-TTS-Engine") or "vllm-omni"
    except urllib.error.HTTPError as err:
        raw_err = err.read().decode("utf-8", errors="replace")
        print(f"[qwen3-tts] HTTP {err.code} from /v1/audio/speech:\n{raw_err}", file=sys.stderr)
        return 1

    out_str = args.output or f"qwen3-tts-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    out_path = Path(out_str).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(audio_bytes)
    msg = (
        f"Synthesized {len(audio_bytes):,} bytes ({args.format}, {sample_rate} Hz, "
        f"engine={engine_tag}, speed={args.speed}x) in {elapsed:.2f}s -> {out_path}"
    )
    if saved_drive:
        msg += f" (Drive: {saved_drive})"
    log(msg)
    return 0


def _add_endpoint_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--url", help="Override public service URL")
    sp.add_argument("--api-token", help="Override Bearer API token")


def _build_custom_parser(parser=None, commands=None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(prog="colabllm qwen3-tts")
    sub = commands or p.add_subparsers(dest="cmd", required=True)

    for tts_name in ("tts", "speak"):
        sp = sub.add_parser(
            tts_name,
            help="Synthesize speech via /v1/audio/speech (CustomVoice, VoiceDesign, or 3s Base Voice Clone)",
        )
        sp.add_argument("text", nargs="?", help="Text to synthesize")
        sp.add_argument("--text", dest="text_opt", help="Text to synthesize (alternative to positional arg)")
        sp.add_argument("--text-file", help="Read text to synthesize from a local UTF-8 file")
        sp.add_argument("-o", "--output", default=None, help="Local output audio file path (default: ./qwen3-tts-<timestamp>.wav)")
        sp.add_argument("--format", default="wav", choices=["wav", "mp3", "flac", "opus", "pcm"], help="Output audio format (default: wav)")
        sp.add_argument(
            "--voice",
            help="Speaker name: built-in (vivian, serena, uncle_fu, dylan, eric, ryan, aiden, ono_anna, sohee) or uploaded/designed custom voice",
        )
        sp.add_argument(
            "--task",
            choices=["CustomVoice", "VoiceDesign", "Base"],
            help="Explicit Qwen3-TTS task type (auto-inferred from --voice-design / --ref-audio / --voice if omitted)",
        )
        sp.add_argument(
            "--instruct",
            help="Natural-language style/emotion/tone instruction (for CustomVoice) or voice persona description (for VoiceDesign)",
        )
        sp.add_argument(
            "--voice-design",
            help="Shortcut for `--task VoiceDesign --instruct <description>` to synthesize with a brand-new designed voice",
        )
        sp.add_argument("--ref-audio", help="Reference audio for 3-second Base voice cloning (local file, drive:// path, or URL)")
        sp.add_argument("--ref-drive", action="store_true", help="Interpret --ref-audio as a Google Drive path")
        sp.add_argument("--ref-text", help="Transcript of --ref-audio for full In-Context Learning (ICL) voice cloning")
        sp.add_argument("--x-vector-only", action="store_true", help="Use speaker embedding only without --ref-text (auto-enabled if --ref-text is omitted)")
        sp.add_argument(
            "--lang",
            help="Language: Auto, Chinese/zh, English/en, Japanese/ja, Korean/ko, German/de, French/fr, Russian/ru, Portuguese/pt, Spanish/es, Italian/it",
        )
        sp.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier in [0.25, 4.0] (default: 1.0)")
        sp.add_argument("--max-new-tokens", type=int, help="Maximum codec frames to generate (default: 2048)")
        sp.add_argument("--seed", type=int, help="Random seed for reproducible generation")
        sp.add_argument("--save-drive", nargs="?", const="auto", help="Also save output audio to Google Drive (optional path or 'auto' for colab-llm/qwen3-tts/outputs/)")
        sp.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", help="Model name")
        sp.add_argument("--timeout", type=nonnegative_seconds, default=300, help="HTTP request timeout in seconds; 0 = unlimited (default: 300)")
        _add_endpoint_flags(sp)

    dvp = sub.add_parser(
        "design-voice",
        help="Voice Design then Clone: create a voice from natural-language description and optionally register it for reuse",
    )
    dvp.add_argument("--instruct", required=True, help="Natural-language voice persona description")
    dvp.add_argument("--text", help="Sample text to synthesize for the reference audio clip")
    dvp.add_argument("--lang", default="Auto", help="Language of sample text (default: Auto)")
    dvp.add_argument("--name", help="Register the designed voice under this name for reuse with `tts --voice <name>`")
    dvp.add_argument("-o", "--output", default=None, help="Local output path for the designed reference WAV")
    dvp.add_argument("--save-drive", nargs="?", const="auto", help="Also save designed audio to Google Drive")
    dvp.add_argument("--timeout", type=nonnegative_seconds, default=300, help="HTTP request timeout in seconds; 0 = unlimited (default: 300)")
    _add_endpoint_flags(dvp)

    vp = sub.add_parser("voices", help="List built-in CustomVoice speakers and uploaded/designed voices (/v1/audio/voices)")
    vp.add_argument("--json", action="store_true", help="Output raw JSON")
    _add_endpoint_flags(vp)

    upv = sub.add_parser("upload-voice", help="Register a persistent named voice from local or Google Drive audio")
    upv.add_argument("--name", required=True, help="Voice identifier (e.g. narrator, alice)")
    upv.add_argument("--audio", help="Local audio file path (or drive:// path)")
    upv.add_argument("--drive", help="Google Drive audio path (relative to colab-llm/qwen3-tts/voices/ or MyDrive)")
    upv.add_argument("--consent", default="owner-consent", help="Consent tag (default: owner-consent)")
    upv.add_argument("--ref-text", help="Transcript of the reference audio (enables high-fidelity ICL voice cloning)")
    upv.add_argument("--description", help="Optional description of the speaker voice")
    _add_endpoint_flags(upv)

    delv = sub.add_parser("delete-voice", help="Delete an uploaded voice (/v1/audio/voices/{name})")
    delv.add_argument("name", help="Voice name to delete")
    _add_endpoint_flags(delv)

    lsd = sub.add_parser("ls-drive", help="List audio files on Google Drive")
    lsd.add_argument("path", nargs="?", default="", help="Drive folder or glob pattern (default: colab-llm/qwen3-tts/voices)")
    lsd.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    lsd.add_argument("--json", action="store_true", help="Output raw JSON")
    _add_endpoint_flags(lsd)

    return p


CUSTOM_CMDS = {"tts", "speak", "design-voice", "voices", "upload-voice", "delete-voice", "ls-drive"}



def configure_cli(parser, commands, up):
    _build_custom_parser(parser, commands)
    for flag, env in UP_ENV_FLAGS.items():
        if flag == "--variant":
            up.add_argument(flag, dest="up_" + env, type=str.lower, choices=tuple(VARIANT_REPOS),
                            help=f"override {env}")
        elif flag == "--preload-all":
            up.add_argument(flag, dest="up_" + env, nargs="?", const="1", choices=("0", "1"),
                            help=f"override {env} (bare flag = 1)")
        else:
            up.add_argument(flag, dest="up_" + env, help=f"override {env}")


_BASE_ENV = dict(SERVICE.env)


def prepare_up(args):
    # Start from the baseline defaults on every call so overrides never leak between invocations.
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    variant = getattr(args, "up_CT_QWEN3_TTS_VARIANT")
    if variant:
        if variant.lower() not in VARIANT_REPOS:
            raise ValueError("Unknown --variant; choices: " + ", ".join(VARIANT_REPOS))
        SERVICE.env["CT_QWEN3_TTS_MODEL"] = VARIANT_REPOS[variant.lower()]
    for flag, env in UP_ENV_FLAGS.items():
        value = getattr(args, "up_" + env)
        if value is not None:
            SERVICE.env[env] = value


if __name__ == "__main__":
    from cli_support import run_cli
    run_cli(lambda: main(SERVICE, configure=configure_cli, prepare_up=prepare_up,
                        handlers={"tts": cmd_tts, "speak": cmd_tts, "voices": cmd_voices,
                "upload-voice": cmd_upload_voice, "delete-voice": cmd_delete_voice, "ls-drive": cmd_ls_drive, "design-voice": cmd_design_voice}))
