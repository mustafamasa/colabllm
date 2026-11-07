#!/usr/bin/env python3
"""NVIDIA-Nemotron-3.5-Transcribe-30B-A4B on a Colab G4 GPU (notebook: nemotron-transcribe.ipynb).

Serves `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (BF16 reference weights by default; optional FP8 / NVFP4)
via `vllm serve` (0.32.0) + `NVIDIA-NeMo/Speech` (`VLLM_PLUGINS=nemo_speechlm` + `ctc_timestamp_middleware`)
on a Colab G4 GPU (NVIDIA RTX PRO 6000 Blackwell, ~96 GB VRAM), exposed through Cloudflare Tunnel with
Bearer token authentication (`auth="builtin"`, default fixed domain `https://transcribe.example.com`).

Capabilities:
  1. Remote OpenAI-compatible API (`/v1/models`, `/v1/chat/completions`):
     - Accepts standard `data:audio/<ext>;base64,...` or `https://...` audio URLs, plus direct Google Drive
       paths (`drive://path/to/audio.wav` or `/content/drive/MyDrive/path/to/audio.wav`).
     - Supports custom instructions (`prompt` / `system_prompt`), reasoning mode (`enable_thinking=True`),
       and CTC word-level timestamps + speaker diarization (`mm_processor_kwargs={"capture_ctc_timestamps": True}`).
  2. Remote Google Drive endpoints (`/api/ct/drive-list`, `/api/ct/drive-transcribe`, `/api/ct/prompts`):
     - Reads audio files directly from Google Drive inside the Colab VM (avoiding local upload & Cloudflare's
       100 MB body limit even for 1-hour recordings) and optionally writes `.txt` / `.srt` / `.nemotron.json`
       back to Google Drive.
  3. CLI single & batch execution (`transcribe` / `run`, `ls-drive`, `prompts`):
     - Specify one or multiple Google Drive audio files, directories, globs (`meetings/*.wav`), or a batch
       manifest (`--batch-file tasks.jsonl` / `.csv` / `.txt` with per-file prompts), or local audio files.
     - Supports concurrent batch execution (`-j / --concurrency N`), custom `--prompt` or NVIDIA `--preset`,
       word timestamps (`--timestamps`), reasoning (`--reasoning`), saving to Google Drive (`--save-drive`),
       and saving locally (`--output` / `--output-dir`).

Usage:
  # 1. Lifecycle
  colabllm nemotron-transcribe up --domain                  # https://transcribe.example.com (G4, BF16)
  colabllm nemotron-transcribe up --domain --precision fp8  # optional FP8 / NVFP4 variant
  colabllm nemotron-transcribe info | logs -f | status | down [--delete-tunnel]

  # 2. Browse audio files on Google Drive & inspect prompt presets
  colabllm nemotron-transcribe ls-drive [MyDrive/path/or/*.wav]
  colabllm nemotron-transcribe prompts

  # 3. Single or batch transcription / audio QA on Google Drive, media URLs (yt-dlp), or local files
  colabllm nemotron-transcribe transcribe "audio/sample.wav" --prompt "Transcribe the following audio."
  colabllm nemotron-transcribe transcribe "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --preset verbatim
  colabllm nemotron-transcribe transcribe "https://x.com/user/status/123456" --download-dir "video/voices" --preset speaker -t
  colabllm nemotron-transcribe transcribe "meetings/*.mp3" --preset speaker --timestamps --save-drive -O ./out -j 2
  colabllm nemotron-transcribe transcribe --batch-file tasks.jsonl --save-drive -O ./out
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import csv
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cli_support import CliError, UsageError, command_timeout, diagnostic, http_timeout, nonnegative_seconds, resolve_endpoint, safe_urlopen
from colabsvc import (  # noqa: E402
    Service,
    State,
    apply_instance,
    die,
    list_service_instances,
    log,
    main,
    warn,
)

PRECISION_REPOS = {
    "bf16": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
    "fp8": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8",
    "nvfp4": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
}

LOCAL_PRESETS: dict[str, dict[str, Any]] = {
    "verbatim": {
        "description": "Verbatim transcription (keeps fillers, repetitions, punctuation and casing; recommended multilingual default)",
        "user_prompt": (
            "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. "
            "Formatting: punctuation and capitalization. Output: transcript only."
        ),
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "clean": {
        "description": "Clean written transcription (removes fillers and accidental repetitions; keeps meaning, punctuation and casing)",
        "user_prompt": (
            "Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; "
            "keep the remaining spoken wording unchanged. Formatting: punctuation and capitalization. "
            "Output: transcript only."
        ),
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "speaker": {
        "description": "Speaker-diarized transcription + word-level timestamps (<spk:0> <spk:1> tags; timestamps enabled automatically)",
        "user_prompt": (
            "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. "
            "Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
        ),
        "system_prompt": "",
        "reasoning": False,
        "timestamps": True,
    },
    "multispeaker": {
        "description": "Multi-speaker plain-text transcription (no punctuation, lowercase, <spk:N> before each turn)",
        "user_prompt": (
            "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. "
            "Use no punctuation and no capitalization. Start every speaker turn with a speaker tag such as "
            "<spk:0>, <spk:1>, <spk:2>, etc.; assign one stable tag per speaker and keep speaker identities "
            "consistent. Output only the speaker-tagged transcript."
        ),
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "rich": {
        "description": "Rich transcription (includes non-speech events <ev:*> and speaking-style <sty:*> tags)",
        "user_prompt": (
            "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. "
            "Write numbers in their standard written form. Use <ev:*> and <sty:*> special tokens to recognize all "
            "non-speech verbatims and speaking style."
        ),
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "near-field": {
        "description": "Transcribe only the speaker closest to the microphone (filters far-field background voices)",
        "user_prompt": (
            "Transcribe this audio. Include all spoken words and verbal fillers. Write numbers as spoken. "
            "No punctuation or capitalization. Transcribe only the speaker nearest to the microphone. "
            "Do not transcribe any other voices."
        ),
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "structured": {
        "description": "Structured-content transcription (precisely keeps emails, URLs, file paths, commands, numbers and punctuation separators)",
        "user_prompt": (
            "Transcribe as a clean written transcript. Preserve exact values and punctuation/separators in emails, "
            "URLs, file paths, commands, and identifiers. Use conventional written forms."
        ),
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "context": {
        "description": "Context-biased transcription with domain vocabulary / proper-noun hints (use with --context)",
        "user_prompt": "Transcribe with punctuation and capitalization. Context terms: {context}.",
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "translate": {
        "description": "Direct speech translation into target-language text (use with --lang, e.g. Chinese / English / Japanese)",
        "user_prompt": "Please translate the given speech to {lang}.",
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "lid": {
        "description": "Language identification (outputs the language label of the recording)",
        "user_prompt": (
            "Identify the language spoken in this audio. Respond with exactly one of the following locale labels "
            "and nothing else: ar_eg, bg_bg, cmn_hans_cn, hr_hr, cs_cz, da_dk, nl_nl, en_us, et_ee, fil_ph, fi_fi, "
            "fr_fr, de_de, el_gr, he_il, hi_in, hu_hu, id_id, it_it, ja_jp, ko_kr, lv_lv, lt_lt, mt_mt, pl_pl, "
            "pt_br, ro_ro, ru_ru, sk_sk, sl_si, es_419, sv_se, th_th, tr_tr, uk_ua, vi_vn."
        ),
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "qa": {
        "description": "Audio understanding and Q&A (reasoning mode --reasoning enabled automatically; answers questions about the recording)",
        "user_prompt": "What is happening in this audio?",
        "system_prompt": (
            "Answer the question directly using evidence from every provided audio clip in order. "
            "Answer every requested part, preserve exact names and numbers, and give a concise, specific answer first. "
            "Do not invent details."
        ),
        "reasoning": True,
        "timestamps": False,
    },
}

AUDIO_EXTS = (
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".mkv", ".wma", ".mpga", ".mpeg",
)
MIME_BY_EXT = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".opus": "audio/ogg",
    ".webm": "audio/webm",
    ".mp4": "audio/mp4",
}

CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)

SERVICE = Service(
    name="nemotron-transcribe",
    notebook="nemotron-transcribe.ipynb",
    service_port=8000,
    auth="builtin",                   # vLLM --api-key + ct_nemotron_middleware enforce Bearer token auth
    public_port=8000,
    gpu="G4",                         # Colab G4 (RTX PRO 6000 Blackwell ~96 GB VRAM), without --high-mem
    high_mem=False,
    default_hostname="transcribe",
    ready_timeout=3600,
    env={
        "CT_NEMOTRON_PRECISION": "bf16",
        "CT_NEMOTRON_REPO": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
        "CT_NEMOTRON_ALIAS": "nemotron-transcribe",
        "CT_NEMOTRON_MTP": "8",
        "CT_NEMOTRON_MAX_MODEL_LEN": "65536",
    },
)


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(State(SERVICE).load(), url_override, token_override,
                            url_env="NEMOTRON_URL", token_env="NEMOTRON_API_TOKEN")


def _get_endpoint_pool(cli_args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Return list of (instance_name, base_url, api_token) for single or multi-instance load balancing."""
    if cli_args.url or cli_args.api_token:
        u, t = _get_endpoint(cli_args.url, cli_args.api_token)
        return [(SERVICE.name, u, t)]

    base_name = SERVICE.base_name or "nemotron-transcribe"
    all_items = list_service_instances(base_name)

    if getattr(cli_args, "all_instances", False):
        pool = [
            (it["name"], it["url"].rstrip("/"), it["api_token"])
            for it in all_items
            if it.get("ready") and it.get("url") and it.get("api_token")
        ]
        if not pool:
            pool = [
                (it["name"], it["url"].rstrip("/"), it["api_token"])
                for it in all_items
                if it.get("alive") and it.get("url") and it.get("api_token")
            ]
        if not pool:
            die(f"No running {base_name} instance found in the user state directory; start at least one instance with `up` first.")
        return pool

    if getattr(cli_args, "instances", None):
        wanted = [x.strip() for x in cli_args.instances.split(",") if x.strip()]
        by_id = {it["instance"]: it for it in all_items}
        by_arg = {it["instance_arg"]: it for it in all_items if it["instance_arg"]}
        by_name = {it["name"]: it for it in all_items}
        pool = []
        for w in wanted:
            it = by_id.get(w) or by_arg.get(w) or by_name.get(w) or by_name.get(f"{base_name}-{w}")
            if not it or not it.get("url") or not it.get("api_token"):
                die(f"Instance '{w}' not found (list started instances with `colabllm nemotron-transcribe ps`)")
            pool.append((it["name"], it["url"].rstrip("/"), it["api_token"]))
        return pool

    u, t = _get_endpoint()
    return [(SERVICE.name, u, t)]


def _http_request(url: str, token: str, payload: dict[str, Any] | None = None, timeout: int = 1800) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) NemotronTranscribeCLI/1.0",
    }
    data = None
    method = "GET"
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with safe_urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            err_obj = json.loads(raw)
        except Exception:
            err_obj = raw
        raise RuntimeError(f"HTTP {e.code} from {url}: {err_obj}") from e


def _cleanup_loops(text: str) -> str:
    if not text:
        return text
    text = CHAR_RUN.sub(lambda m: m.group(1), text)
    return BLOCK_RUN.sub(lambda m: m.group(1), text)


def _fmt_srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _words_to_srt(words: list[dict[str, Any]]) -> str:
    if not words:
        return ""
    segments = []
    cur_words: list[str] = []
    cur_start = 0.0
    cur_end = 0.0
    cur_spk = None

    def flush() -> None:
        nonlocal cur_words, cur_start, cur_end, cur_spk
        if not cur_words:
            return
        prefix = f"[{cur_spk}] " if cur_spk else ""
        text = prefix + " ".join(cur_words).strip()
        segments.append((cur_start, max(cur_end, cur_start + 0.2), text))
        cur_words = []

    for w in words:
        w_text = str(w.get("word") or "").strip()
        if not w_text:
            continue
        w_start = float(w.get("start") or 0.0)
        w_end = float(w.get("end") or w_start)
        w_spk = w.get("speaker")
        if not cur_words:
            cur_start = w_start
            cur_end = w_end
            cur_spk = w_spk
            cur_words.append(w_text)
            continue
        gap = w_start - cur_end
        if (w_spk != cur_spk) or gap > 1.2 or len(cur_words) >= 14 or (
            len(cur_words) >= 6 and cur_words[-1].endswith((".", "?", "!", "\u3002", "\uff1f", "\uff01"))
        ):
            flush()
            cur_start = w_start
            cur_end = w_end
            cur_spk = w_spk
            cur_words.append(w_text)
        else:
            cur_end = max(cur_end, w_end)
            cur_words.append(w_text)
    flush()
    return "\n".join(
        f"{idx}\n{_fmt_srt_time(st_t)} --> {_fmt_srt_time(ed_t)}\n{txt}\n"
        for idx, (st_t, ed_t, txt) in enumerate(segments, 1)
    )


def _cmd_prompts() -> None:
    print("\nBuilt-in recommended NVIDIA Nemotron 3.5 Transcribe prompt presets (--preset <name>):\n")
    for name, info in LOCAL_PRESETS.items():
        flags = []
        if info["reasoning"]:
            flags.append("reasoning=True")
        if info["timestamps"]:
            flags.append("timestamps=True")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  * {name:<14} {info['description']}{flag_str}")
        print(f"    user_prompt  : {info['user_prompt']}")
        if info["system_prompt"]:
            print(f"    system_prompt: {info['system_prompt']}")
        print()


SKIP_DRIVE_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "custom_nodes",
    "models", ".runtimes", "Takeout", ".Trash", ".ipynb_checkpoints",
}


def _is_media_url(spec: str) -> bool:
    s = (spec or "").strip()
    return s.startswith(("http://", "https://", "ytdl://", "yt-dlp://"))


def _clean_media_url(spec: str) -> str:
    s = (spec or "").strip()
    for pfx in ("ytdl://", "yt-dlp://"):
        if s.startswith(pfx):
            s = s[len(pfx):]
            if not s.startswith(("http://", "https://")):
                s = "https://" + s
    return s


def _sanitize_filename(name: str, max_len: int = 96) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-\u4e00-\u9fff]+", "_", (name or "").strip()).strip("._-")
    return (cleaned[:max_len].rstrip("._-") or "media")


def _normalize_drive_dir(raw_spec: str) -> str:
    spec = (raw_spec or "").strip()
    for pfx in ("drive://", "gdrive://", "file://"):
        if spec.startswith(pfx):
            spec = spec[len(pfx):]
    spec = spec.strip("/")
    if not spec or spec == "MyDrive":
        return "/content/drive/MyDrive"
    if spec.startswith("content/drive/"):
        return "/" + spec
    if spec.startswith("drive/"):
        return "/content/" + spec
    if spec.startswith("MyDrive/"):
        return "/content/drive/" + spec
    return f"/content/drive/MyDrive/{spec}"


DEFAULT_YTDLP_DRIVE_DIR = "/content/drive/MyDrive/colab-llm/nemotron-transcribe/audio/yt-dlp"


def _download_media_locally_and_upload_to_drive(
    raw_url: str,
    session_name: str,
    download_dir: str = "",
    audio_format: str = "mp3",
    cookies_file: str | None = None,
    cookies_from_browser: str | None = None,
    playlist: bool = False,
    download_video: bool = False,
) -> list[dict[str, Any]]:
    """Run local `yt-dlp` to download audio or video, then upload to Google Drive via `colab upload`."""
    ytdlp_bin = shutil.which("yt-dlp")
    if not ytdlp_bin:
        raise RuntimeError("`yt-dlp` is not installed locally; local yt-dlp downloads are unavailable.")

    url = _clean_media_url(raw_url)
    target_drive_dir = _normalize_drive_dir(download_dir) if (download_dir or "").strip() else DEFAULT_YTDLP_DRIVE_DIR
    fmt = (audio_format or "mp3").strip().lower().lstrip(".")
    if fmt not in ("mp3", "m4a", "wav", "opus", "flac"):
        fmt = "mp3"

    # Ensure remote directory exists on mounted Google Drive. The kernel is busy (blocked in the
    # keep-alive cell) while the service runs, so this is best effort and must never hang the CLI.
    try:
        subprocess.run(
            ["colab", "exec", "-s", session_name, "-c", f"mkdir -p {shlex.quote(target_drive_dir)}"],
            capture_output=True,
            text=True,
            timeout=command_timeout(30),
            check=False,
        )
    except subprocess.TimeoutExpired:
        diagnostic(f"[nemotron] Remote mkdir timed out (the kernel may be busy); uploading directly to {target_drive_dir}")

    with tempfile.TemporaryDirectory(prefix="nemotron-ytdlp-") as stage_str:
        stage_dir = Path(stage_str)
        out_tmpl = str(stage_dir / "%(extractor)s_%(id)s.%(ext)s")
        cmd = [
            ytdlp_bin,
            "--no-progress",
            "--no-warnings",
            "--ignore-config",
            "--no-playlist" if not playlist else "--yes-playlist",
        ]
        if download_video:
            cmd.extend([
                "-f", "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
                "--merge-output-format", "mp4",
            ])
        else:
            cmd.extend([
                "-x",
                "--audio-format", fmt,
                "--audio-quality", "0",
            ])
        cmd.extend([
            "--write-info-json",
            "-o", out_tmpl,
        ])
        if cookies_file:
            cp = Path(cookies_file).expanduser().resolve()
            if not cp.is_file():
                raise FileNotFoundError(f"Cookies file not found: {cp}")
            cmd.extend(["--cookies", str(cp)])
        if cookies_from_browser:
            cmd.extend(["--cookies-from-browser", cookies_from_browser])
        cmd.append(url)

        kind_str = "video (.mp4)" if download_video else f"audio (.{fmt})"
        log(f"Downloading {kind_str} locally with yt-dlp: {url}")
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=command_timeout(900), check=False)
        audio_files = sorted(
            f for f in stage_dir.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        )
        if res.returncode != 0 and not audio_files:
            err_tail = (res.stderr or res.stdout or "").strip()[-800:]
            raise RuntimeError(f"Local yt-dlp download failed ({url}): {err_tail}")
        if not audio_files:
            raise RuntimeError(f"yt-dlp did not download any media file: {url}")

        uploaded: list[dict[str, Any]] = []
        for af in audio_files:
            stem = af.stem
            info_path = stage_dir / f"{stem}.info.json"
            meta: dict[str, Any] = {}
            if info_path.is_file():
                try:
                    meta = json.loads(info_path.read_text(encoding="utf-8", errors="replace"))
                except Exception:
                    meta = {}
            extractor = _sanitize_filename(str(meta.get("extractor_key") or meta.get("extractor") or "media"), 24).lower()
            vid_id = _sanitize_filename(str(meta.get("id") or stem), 40)
            title = _sanitize_filename(str(meta.get("title") or ""), 64)
            base_name = f"{extractor}_{vid_id}_{title}" if title and title != vid_id else f"{extractor}_{vid_id}"
            dest_name = f"{base_name}{af.suffix.lower()}"
            remote_path = f"{target_drive_dir.rstrip('/')}/{dest_name}"
            log(f"Uploading {kind_str} to Google Drive ({session_name}): {remote_path}")
            up_res = subprocess.run(
                ["colab", "upload", "-s", session_name, str(af), remote_path],
                capture_output=True,
                text=True,
                timeout=command_timeout(300),
                check=False,
            )
            if up_res.returncode != 0:
                err = (up_res.stderr or up_res.stdout or "").strip()
                raise RuntimeError(f"`colab upload` failed ({remote_path}): {err}")
            rel = remote_path.removeprefix("/content/drive/MyDrive/")
            uploaded.append({
                "drive_path": remote_path,
                "rel_path": rel,
                "name": dest_name,
                "media_type": "video" if download_video else "audio",
                "saved_to_drive": True,
                "source_url": str(meta.get("webpage_url") or url),
                "title": meta.get("title") or dest_name,
                "uploader": meta.get("uploader") or meta.get("channel"),
                "duration": meta.get("duration"),
                "extractor": meta.get("extractor_key") or meta.get("extractor"),
                "id": meta.get("id") or vid_id,
                "cached": False,
            })
        return uploaded


def _colab_ls_single(session_name: str, remote_path: str) -> tuple[list[str], list[str]]:
    """Use `colab ls -s <session> <remote_path>` directly (no HTTP tunnel roundtrip)."""
    res = subprocess.run(
        ["colab", "ls", "-s", session_name, remote_path],
        capture_output=True,
        text=True,
        timeout=command_timeout(30),
        check=False,
    )
    if res.returncode != 0:
        err = (res.stderr or res.stdout or "").strip()
        raise RuntimeError(f"`colab ls -s {session_name} {remote_path}` failed: {err}")
    dirs: list[str] = []
    files: list[str] = []
    for line in res.stdout.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("[colab]"):
            continue
        if entry.endswith("/"):
            dirs.append(entry[:-1])
        else:
            files.append(entry)
    return sorted(dirs), sorted(files)


def _list_drive_via_colab_cli(
    session_name: str,
    raw_spec: str,
    recursive: bool = False,
) -> tuple[str, list[str], list[dict[str, Any]]]:
    """List directories and audio files on the mounted Colab Drive via `colab ls`."""
    has_glob = any(ch in (raw_spec or "") for ch in ("*", "?", "["))
    if has_glob:
        full_pattern = _normalize_drive_dir(raw_spec)
        parent_dir = str(Path(full_pattern).parent)
        name_pat = Path(full_pattern).name
        dirs, files = _colab_ls_single(session_name, parent_dir)
        import fnmatch
        matched_audio = []
        for f in files:
            if fnmatch.fnmatch(f, name_pat) and Path(f).suffix.lower() in AUDIO_EXTS:
                dp = f"{parent_dir.rstrip('/')}/{f}"
                rel = dp.removeprefix("/content/drive/MyDrive/")
                matched_audio.append({"drive_path": dp, "rel_path": rel, "name": f})
        return parent_dir, [], matched_audio

    base_dir = _normalize_drive_dir(raw_spec)
    if Path(base_dir).suffix.lower() in AUDIO_EXTS:
        rel = base_dir.removeprefix("/content/drive/MyDrive/")
        return str(Path(base_dir).parent), [], [{"drive_path": base_dir, "rel_path": rel, "name": Path(base_dir).name}]

    top_dirs, top_files = _colab_ls_single(session_name, base_dir)
    audio_items: list[dict[str, Any]] = []
    for f in top_files:
        if Path(f).suffix.lower() in AUDIO_EXTS:
            dp = f"{base_dir.rstrip('/')}/{f}"
            rel = dp.removeprefix("/content/drive/MyDrive/")
            audio_items.append({"drive_path": dp, "rel_path": rel, "name": f})

    if recursive:
        queue = [f"{base_dir.rstrip('/')}/{d}" for d in top_dirs if d not in SKIP_DRIVE_DIRS and not d.startswith(".")]
        while queue:
            cur = queue.pop(0)
            try:
                sub_dirs, sub_files = _colab_ls_single(session_name, cur)
            except Exception:
                continue
            for f in sub_files:
                if Path(f).suffix.lower() in AUDIO_EXTS:
                    dp = f"{cur.rstrip('/')}/{f}"
                    rel = dp.removeprefix("/content/drive/MyDrive/")
                    audio_items.append({"drive_path": dp, "rel_path": rel, "name": f})
            for d in sub_dirs:
                if d not in SKIP_DRIVE_DIRS and not d.startswith("."):
                    queue.append(f"{cur.rstrip('/')}/{d}")

    return base_dir, top_dirs, audio_items


def _build_ls_drive_parser(parser=None):
    p = parser or argparse.ArgumentParser(
        prog=f"colabllm nemotron-transcribe ls-drive",
        description="Query Drive directories and audio files directly through Colab's mounted /content/drive (no public HTTP tunnel needed)",
    )
    p.add_argument("path", nargs="?", default="", help="Drive relative path, directory or glob (e.g. 'video' or 'meetings/*.wav'; empty means the MyDrive root)")
    p.add_argument("-r", "--recursive", action="store_true", default=False, help="search subdirectories recursively for audio files (default lists only the current directory to avoid full-drive scan timeouts)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false", help="list only the current directory level")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    return p


def _cmd_ls_drive(argv) -> None:
    p = _build_ls_drive_parser()
    a = argv if isinstance(argv, argparse.Namespace) else p.parse_args(argv)

    base_dir, sub_dirs, audio_files = _list_drive_via_colab_cli(SERVICE.name, a.path, recursive=a.recursive)
    if a.json:
        print(json.dumps({
            "session": SERVICE.name,
            "base_dir": base_dir,
            "directories": sub_dirs,
            "audio_files": audio_files,
        }, indent=2, ensure_ascii=False))
        return
    log(f"Session [{SERVICE.name}] Drive directory: {base_dir}")
    if sub_dirs and not a.recursive:
        print(f"\n  Subdirectories ({len(sub_dirs)}):")
        for d in sub_dirs:
            print(f"    📁 {d}/")
    print(f"\n  Audio files ({len(audio_files)}):")
    if not audio_files:
        print("    (no audio files at this level; specify a subdirectory such as `ls-drive video` or add `-r` to search subdirectories)")
    else:
        for item in audio_files:
            print(f"    🎵 {item['rel_path']}   ({item['drive_path']})")
    print()


def _resolve_task_prompt(task: dict[str, Any], cli_args: argparse.Namespace) -> dict[str, Any]:
    preset_name = (task.get("preset") or cli_args.preset or "").strip().lower()
    preset = LOCAL_PRESETS.get(preset_name) if preset_name else None

    prompt = task.get("prompt")
    if prompt is None:
        prompt = cli_args.prompt
    if prompt is None or str(prompt).strip() == "":
        prompt = preset["user_prompt"] if preset else LOCAL_PRESETS["verbatim"]["user_prompt"]
    prompt = str(prompt)

    ctx = (task.get("context") or cli_args.context or "").strip()
    lang = (task.get("lang") or task.get("language") or cli_args.lang or "Chinese").strip()
    if "{context}" in prompt:
        prompt = prompt.replace("{context}", ctx or "general")
    elif ctx and "Context terms:" not in prompt:
        prompt = f"{prompt.rstrip()} Context terms: {ctx}."
    if "{lang}" in prompt:
        prompt = prompt.replace("{lang}", lang)

    sys_prompt = task.get("system_prompt")
    if sys_prompt is None:
        sys_prompt = cli_args.system_prompt
    if sys_prompt is None:
        sys_prompt = preset["system_prompt"] if preset else ""

    reasoning = task.get("reasoning")
    if reasoning is None:
        reasoning = cli_args.reasoning
    if reasoning is None:
        reasoning = bool(preset["reasoning"]) if preset else False
    else:
        reasoning = bool(reasoning)

    timestamps = task.get("timestamps")
    if timestamps is None:
        timestamps = cli_args.timestamps
    if timestamps is None:
        timestamps = bool(preset["timestamps"]) if preset else False
    else:
        timestamps = bool(timestamps)

    return {
        "prompt": prompt,
        "system_prompt": str(sys_prompt),
        "preset": preset_name or None,
        "context": ctx or None,
        "lang": lang,
        "reasoning": reasoning,
        "timestamps": timestamps,
        "temperature": task.get("temperature", cli_args.temperature),
        "max_tokens": task.get("max_tokens", cli_args.max_tokens),
    }


def _load_batch_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        die(f"batch file not found: {manifest_path}")
    ext = manifest_path.suffix.lower()
    raw = manifest_path.read_text(encoding="utf-8").strip()
    if not raw:
        return []
    tasks: list[dict[str, Any]] = []
    if ext == ".jsonl":
        for idx, line in enumerate(raw.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                tasks.append({"audio": obj})
            elif isinstance(obj, dict):
                tasks.append(obj)
            else:
                die(f"invalid JSONL entry at {manifest_path}:{idx}")
    elif ext == ".json":
        data = json.loads(raw)
        if isinstance(data, dict) and isinstance(data.get("tasks"), list):
            data = data["tasks"]
        if not isinstance(data, list):
            die(f"expected a JSON list in {manifest_path}")
        for item in data:
            tasks.append({"audio": item} if isinstance(item, str) else dict(item))
    elif ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        reader = csv.DictReader(raw.splitlines(), delimiter=delim)
        for row in reader:
            cleaned = {k.strip(): v.strip() for k, v in row.items() if k and v is not None and v.strip() != ""}
            if "reasoning" in cleaned:
                cleaned["reasoning"] = cleaned["reasoning"].lower() in ("1", "true", "yes")
            if "timestamps" in cleaned:
                cleaned["timestamps"] = cleaned["timestamps"].lower() in ("1", "true", "yes")
            tasks.append(cleaned)
    else:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
                tasks.append({"audio": parts[0].strip(), "prompt": parts[1].strip()})
            else:
                tasks.append({"audio": line})
    return tasks


def _expand_tasks(cli_args: argparse.Namespace, base_url: str, token: str) -> list[dict[str, Any]]:
    raw_items: list[dict[str, Any]] = []
    for p in cli_args.paths:
        raw_items.append({"audio": p})
    if cli_args.batch_file:
        raw_items.extend(_load_batch_manifest(Path(cli_args.batch_file)))
    if not raw_items:
        die("Specify at least one media URL (YouTube/Twitter etc.), a Drive/local audio path, or a batch task list with `--batch-file <file>`.")

    expanded: list[dict[str, Any]] = []
    for item in raw_items:
        spec = (
            item.get("audio")
            or item.get("url")
            or item.get("drive_path")
            or item.get("path")
            or item.get("file")
            or ""
        ).strip()
        if not spec:
            die(f"batch item is missing 'audio' / 'url' / 'drive_path': {item}")

        # 1. Media URL (YouTube, Twitter/X, Bilibili, TikTok, direct HTTP audio/video...)
        if _is_media_url(spec):
            use_local_ytdlp = bool(
                getattr(cli_args, "local_ytdlp", False)
                or getattr(cli_args, "cookies", None)
                or getattr(cli_args, "cookies_from_browser", None)
            )
            is_playlist = bool(item.get("playlist", getattr(cli_args, "playlist", False)))
            dl_dir = str(item.get("download_dir") or getattr(cli_args, "download_dir", "") or "").strip()
            audio_fmt = str(item.get("audio_format") or getattr(cli_args, "audio_format", "mp3") or "mp3").strip()
            cookies_drive = str(item.get("cookies_drive") or getattr(cli_args, "cookies_drive", "") or "").strip()
            force_dl = bool(item.get("force_download", getattr(cli_args, "force_download", False)))
            dl_video = bool(item.get("download_video", getattr(cli_args, "download_video", False)))
            save_media_drive = bool(item.get("save_media_to_drive", getattr(cli_args, "save_media_drive", True)))

            if use_local_ytdlp or is_playlist:
                downloaded_files: list[dict[str, Any]] = []
                if not use_local_ytdlp:
                    try:
                        log(f"Downloading media list to Google Drive with remote yt-dlp: {spec}")
                        dl_res = _http_request(
                            f"{base_url}/api/ct/ytdlp-download",
                            token,
                            payload={
                                "url": spec,
                                "download_dir": dl_dir,
                                "audio_format": audio_fmt,
                                "cookies_drive": cookies_drive,
                                "playlist": is_playlist,
                                "force_download": force_dl,
                                "download_video": dl_video,
                                "save_media_to_drive": save_media_drive,
                            },
                            timeout=None if not cli_args.timeout else max(cli_args.timeout, 900),
                        )
                        downloaded_files = dl_res.get("files") or []
                    except Exception as e:
                        warn(f"Remote Colab yt-dlp download failed ({e}); switching to local yt-dlp and uploading to Google Drive...")
                        downloaded_files = _download_media_locally_and_upload_to_drive(
                            raw_url=spec,
                            session_name=SERVICE.name,
                            download_dir=dl_dir,
                            audio_format=audio_fmt,
                            cookies_file=getattr(cli_args, "cookies", None),
                            cookies_from_browser=getattr(cli_args, "cookies_from_browser", None),
                            playlist=is_playlist,
                            download_video=dl_video,
                        )
                else:
                    downloaded_files = _download_media_locally_and_upload_to_drive(
                        raw_url=spec,
                        session_name=SERVICE.name,
                        download_dir=dl_dir,
                        audio_format=audio_fmt,
                        cookies_file=getattr(cli_args, "cookies", None),
                        cookies_from_browser=getattr(cli_args, "cookies_from_browser", None),
                        playlist=is_playlist,
                        download_video=dl_video,
                    )
                for df in downloaded_files:
                    expanded.append({
                        **item,
                        "source_type": "drive",
                        "drive_path": df["drive_path"],
                        "source_url": df.get("source_url") or spec,
                        "media_info": df,
                        "label": df.get("rel_path") or df.get("name") or spec,
                    })
            else:
                expanded.append({
                    **item,
                    "source_type": "url",
                    "url": spec,
                    "download_dir": dl_dir,
                    "audio_format": audio_fmt,
                    "cookies_drive": cookies_drive,
                    "force_download": force_dl,
                    "download_video": dl_video,
                    "save_media_to_drive": save_media_drive,
                    "label": spec,
                })
            continue

        is_local = False
        if cli_args.local and not spec.startswith(("drive://", "gdrive://", "/content/drive/")):
            is_local = True
        elif not cli_args.drive and not spec.startswith(("drive://", "gdrive://", "/content/drive/")):
            local_candidate = Path(spec)
            if local_candidate.exists():
                is_local = True

        if is_local:
            lp = Path(spec)
            if lp.is_file():
                expanded.append({**item, "source_type": "local", "local_path": str(lp.resolve()), "label": lp.name})
            elif lp.is_dir():
                it = lp.rglob("*") if cli_args.recursive else lp.glob("*")
                matched = [f.resolve() for f in sorted(it) if f.is_file() and f.suffix.lower() in AUDIO_EXTS]
                if not matched:
                    warn(f"No supported audio files found in local directory: {lp}")
                for f in matched:
                    expanded.append({**item, "source_type": "local", "local_path": str(f), "label": str(f.relative_to(lp.resolve()))})
            else:
                die(f"local path not found: {lp}")
        else:
            # Check if it's a directory or glob on Google Drive via `colab ls`
            clean_spec = spec
            for pfx in ("drive://", "gdrive://"):
                if clean_spec.startswith(pfx):
                    clean_spec = clean_spec[len(pfx):]
            needs_list = any(ch in clean_spec for ch in ("*", "?", "[")) or clean_spec.endswith("/") or (
                Path(clean_spec).suffix.lower() not in AUDIO_EXTS
            )
            if needs_list:
                try:
                    _, _, files = _list_drive_via_colab_cli(SERVICE.name, spec, recursive=cli_args.recursive)
                except Exception:
                    qs = urllib.parse.urlencode({"path": spec, "recursive": "1" if cli_args.recursive else "0"})
                    res = _http_request(f"{base_url}/api/ct/drive-list?{qs}", token, timeout=60)
                    files = res.get("files") or []
                if not files:
                    die(f"No matching audio file found on Google Drive: '{spec}'")
                for f in files:
                    expanded.append({
                        **item,
                        "source_type": "drive",
                        "drive_path": f["drive_path"],
                        "label": f.get("rel_path") or f["name"],
                    })
            else:
                expanded.append({
                    **item,
                    "source_type": "drive",
                    "drive_path": spec,
                    "label": clean_spec,
                })
    return expanded


def _probe_local_audio_duration_sec(audio_path: Path) -> float:
    """Return local audio/video duration in seconds using ffprobe (0.0 if unknown)."""
    try:
        res = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=command_timeout(30),
            check=False,
        )
        val = (res.stdout or "").strip()
        return float(val) if val and val != "N/A" else 0.0
    except Exception:
        return 0.0


def _find_local_silence_near(audio_path: Path, target_sec: float, window_sec: float = 45.0) -> float:
    win_start = max(0.0, target_sec - window_sec)
    win_dur = window_sec * 2.0
    try:
        res = subprocess.run(
            [
                "ffmpeg", "-nostdin", "-hide_banner",
                "-ss", f"{win_start:.3f}", "-t", f"{win_dur:.3f}",
                "-i", str(audio_path),
                "-vn", "-af", "silencedetect=n=-35dB:d=0.35",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=command_timeout(25),
            check=False,
        )
        log_out = res.stderr or ""
        starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", log_out)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", log_out)]
        candidates = [
            win_start + (st + (ends[i] if i < len(ends) else st + 0.35)) / 2.0
            for i, st in enumerate(starts)
        ]
        if candidates:
            return min(candidates, key=lambda c: abs(c - target_sec))
    except Exception:
        pass
    return target_sec


def _compute_local_chunk_ranges(
    audio_path: Path,
    total_duration: float,
    max_chunk_sec: float = 2400.0,
) -> list[tuple[float, float]]:
    if max_chunk_sec <= 0 or total_duration <= 0 or total_duration <= max_chunk_sec:
        return [(0.0, max(total_duration, 0.0))]
    import math
    num_chunks = max(2, int(math.ceil(total_duration / max_chunk_sec)))
    ideal_len = total_duration / num_chunks
    boundaries = [0.0]
    for k in range(1, num_chunks):
        target = k * ideal_len
        split_pt = _find_local_silence_near(audio_path, target, window_sec=min(45.0, ideal_len * 0.15))
        split_pt = max(boundaries[-1] + 30.0, min(split_pt, total_duration - 30.0))
        boundaries.append(round(split_pt, 3))
    boundaries.append(round(total_duration, 3))
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def _load_local_audio_data_url(
    lp: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    normalize_audio: bool = False,
) -> str:
    ext = lp.suffix.lower()
    is_slice = (start_sec is not None and start_sec > 0.0) or (end_sec is not None and end_sec > 0.0)
    if not is_slice and not normalize_audio and ext in MIME_BY_EXT and lp.stat().st_size <= 70 * 1024 * 1024:
        b64 = base64.b64encode(lp.read_bytes()).decode("ascii")
        return f"data:{MIME_BY_EXT[ext]};base64,{b64}"
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start_sec is not None and start_sec > 0.0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    if end_sec is not None and end_sec > 0.0:
        dur = max(0.1, end_sec - (start_sec or 0.0))
        cmd.extend(["-t", f"{dur:.3f}"])
    cmd.extend([
        "-i", str(lp),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
    ])
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if res.returncode != 0 or not res.stdout:
        err = res.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed decoding {lp.name}: {err}")
    b64 = base64.b64encode(res.stdout).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def _shift_ctc_timestamps(ctc_ts: dict[str, Any] | None, offset_sec: float) -> dict[str, Any] | None:
    if not isinstance(ctc_ts, dict) or offset_sec <= 0.0:
        return ctc_ts
    shifted: dict[str, Any] = dict(ctc_ts)
    for key in ("words", "segments", "diarization"):
        items = ctc_ts.get(key)
        if isinstance(items, list):
            new_items = []
            for item in items:
                if isinstance(item, dict):
                    cp = dict(item)
                    if cp.get("start") is not None:
                        cp["start"] = round(float(cp["start"]) + offset_sec, 3)
                    if cp.get("end") is not None:
                        cp["end"] = round(float(cp["end"]) + offset_sec, 3)
                    new_items.append(cp)
                else:
                    new_items.append(item)
            shifted[key] = new_items
    return shifted


def _run_one_task(
    idx: int,
    total: int,
    task: dict[str, Any],
    cli_args: argparse.Namespace,
    base_url: str,
    token: str,
) -> dict[str, Any]:
    cfg = _resolve_task_prompt(task, cli_args)
    label = (
        task.get("id")
        or task.get("label")
        or task.get("url")
        or task.get("drive_path")
        or task.get("local_path")
        or f"task-{idx}"
    )
    t0 = time.monotonic()
    chunk_minutes = float(task.get("chunk_minutes", getattr(cli_args, "chunk_minutes", 40.0)))

    if task["source_type"] in ("drive", "url"):
        is_url_task = task["source_type"] == "url"
        target_spec = task["url"] if is_url_task else task["drive_path"]
        body: dict[str, Any] = {
            "drive_path": target_spec,
            "prompt": cfg["prompt"],
            "system_prompt": cfg["system_prompt"],
            "preset": cfg["preset"],
            "reasoning": cfg["reasoning"],
            "timestamps": cfg["timestamps"],
            "normalize_audio": cli_args.normalize_audio,
            "chunk_minutes": chunk_minutes,
            "save_to_drive": bool(cli_args.save_drive is not None),
            "save_drive_dir": (
                cli_args.save_drive if isinstance(cli_args.save_drive, str) and cli_args.save_drive != "@same" else ""
            ),
        }
        if is_url_task:
            body["url"] = task["url"]
            body["download_dir"] = task.get("download_dir") or ""
            body["audio_format"] = task.get("audio_format") or "mp3"
            body["cookies_drive"] = task.get("cookies_drive") or ""
            body["force_download"] = bool(task.get("force_download", False))
            body["download_video"] = bool(task.get("download_video", False))
            body["save_media_to_drive"] = bool(task.get("save_media_to_drive", True))
        if cfg["temperature"] is not None:
            body["temperature"] = cfg["temperature"]
        if cfg["max_tokens"] is not None:
            body["max_tokens"] = cfg["max_tokens"]
        try:
            res = _http_request(f"{base_url}/api/ct/drive-transcribe", token, payload=body, timeout=http_timeout(cli_args.timeout))
        except Exception as e:
            # Automatic fallback: if remote Colab yt-dlp was blocked or endpoint not updated yet,
            # download locally via yt-dlp, upload to Google Drive via `colab upload`, and transcribe!
            if is_url_task and ("yt-dlp" in str(e) or "HTTP 502" in str(e) or "HTTP 400" in str(e)):
                warn(f"Remote yt-dlp download did not succeed ({e}); switching to local yt-dlp and uploading to Google Drive...")
                uploaded = _download_media_locally_and_upload_to_drive(
                    raw_url=task["url"],
                    session_name=SERVICE.name,
                    download_dir=task.get("download_dir") or "",
                    audio_format=task.get("audio_format") or "mp3",
                    cookies_file=getattr(cli_args, "cookies", None),
                    cookies_from_browser=getattr(cli_args, "cookies_from_browser", None),
                    playlist=False,
                    download_video=bool(task.get("download_video", False)),
                )
                first_up = uploaded[0]
                body.pop("url", None)
                body["drive_path"] = first_up["drive_path"]
                res = _http_request(f"{base_url}/api/ct/drive-transcribe", token, payload=body, timeout=http_timeout(cli_args.timeout))
                res["source_url"] = task["url"]
                res["media_info"] = first_up
            else:
                raise
        if task.get("source_url") and not res.get("source_url"):
            res["source_url"] = task["source_url"]
        if task.get("media_info") and not res.get("media_info"):
            res["media_info"] = task["media_info"]
        res["index"] = idx
        res["label"] = res.get("rel_path") or label
        return res

    # Local file -> call /v1/chat/completions (with auto-chunking if duration > chunk_minutes)
    lp = Path(task["local_path"])
    total_duration = _probe_local_audio_duration_sec(lp)
    chunk_ranges = _compute_local_chunk_ranges(lp, total_duration, max_chunk_sec=chunk_minutes * 60.0)
    is_multi_chunk = len(chunk_ranges) > 1
    if is_multi_chunk:
        log(f"Local long audio {lp.name} is {total_duration / 60:.1f} minutes; splitting at silences into {len(chunk_ranges)} chunks and transcribing them in order...")

    reasoning = cfg["reasoning"]
    timestamps = cfg["timestamps"]
    extra_body: dict[str, Any] = {
        "chat_template_kwargs": {"enable_thinking": reasoning},
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "seed": 0,
    }
    if reasoning:
        extra_body.update(top_k=-1, thinking_token_budget=17408)
        extra_body["chat_template_kwargs"]["reasoning_budget"] = 16384
    if timestamps:
        extra_body["mm_processor_kwargs"] = {"capture_ctc_timestamps": True}

    texts: list[str] = []
    reasonings: list[str] = []
    merged_words: list[dict[str, Any]] = []
    merged_segments: list[dict[str, Any]] = []
    merged_diarization: list[dict[str, Any]] = []
    last_ctc_ts: dict[str, Any] | None = None
    chunk_meta_list: list[dict[str, Any]] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for c_idx, (c_start, c_end) in enumerate(chunk_ranges, 1):
        c_t0 = time.monotonic()
        data_url = _load_local_audio_data_url(
            lp,
            start_sec=c_start if is_multi_chunk else None,
            end_sec=c_end if is_multi_chunk else None,
            normalize_audio=cli_args.normalize_audio or is_multi_chunk,
        )
        messages: list[dict[str, Any]] = []
        if cfg["system_prompt"]:
            messages.append({"role": "system", "content": cfg["system_prompt"]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": cfg["prompt"]},
                {"type": "audio_url", "audio_url": {"url": data_url}},
            ],
        })
        payload = {
            "model": SERVICE.env.get("CT_NEMOTRON_ALIAS", "nemotron-transcribe"),
            "messages": messages,
            "temperature": float(cfg["temperature"]) if cfg["temperature"] is not None else (0.6 if reasoning else 0.0),
            "top_p": 0.95 if reasoning else 1.0,
            "max_tokens": int(cfg["max_tokens"]) if cfg["max_tokens"] is not None else (20480 if reasoning else 16384),
            **extra_body,
        }
        completion = _http_request(f"{base_url}/v1/chat/completions", token, payload=payload, timeout=http_timeout(cli_args.timeout))
        choice = (completion.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        c_text = (msg.get("content") or "").strip()
        c_reasoning = msg.get("reasoning_content") or msg.get("reasoning") or None
        if not reasoning:
            c_text = _cleanup_loops(c_text)
        if c_text:
            if is_multi_chunk and reasoning:
                texts.append(f"[{_fmt_srt_time(c_start)[:8]} - {_fmt_srt_time(c_end)[:8]}]\n{c_text}")
            else:
                texts.append(c_text)
        if c_reasoning:
            if is_multi_chunk:
                reasonings.append(f"[Chunk {c_idx}/{len(chunk_ranges)}: {_fmt_srt_time(c_start)[:8]} - {_fmt_srt_time(c_end)[:8]}]\n{c_reasoning}")
            else:
                reasonings.append(str(c_reasoning))

        c_ctc = _shift_ctc_timestamps(completion.get("ctc_timestamps"), c_start)
        if isinstance(c_ctc, dict):
            last_ctc_ts = c_ctc
            if isinstance(c_ctc.get("words"), list):
                merged_words.extend(c_ctc["words"])
            if isinstance(c_ctc.get("segments"), list):
                merged_segments.extend(c_ctc["segments"])
            if isinstance(c_ctc.get("diarization"), list):
                merged_diarization.extend(c_ctc["diarization"])

        u = completion.get("usage")
        if isinstance(u, dict):
            for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                total_usage[k] += int(u.get(k) or 0)

        chunk_meta_list.append({
            "chunk": c_idx,
            "start_sec": round(c_start, 2),
            "end_sec": round(c_end, 2),
            "duration_sec": round(max(0.0, c_end - c_start), 2),
            "elapsed_sec": round(time.monotonic() - c_t0, 2),
        })

    text = ("\n\n" if is_multi_chunk else "").join(texts)
    if last_ctc_ts is not None:
        ctc_ts: dict[str, Any] | None = dict(last_ctc_ts)
        if merged_words:
            ctc_ts["words"] = merged_words
        if merged_segments:
            ctc_ts["segments"] = merged_segments
        if merged_diarization:
            ctc_ts["diarization"] = merged_diarization
    else:
        ctc_ts = None
    srt_text = _words_to_srt(merged_words) if merged_words else None
    return {
        "index": idx,
        "label": label,
        "local_path": str(lp),
        "duration_sec": round(total_duration, 2) if total_duration > 0 else None,
        "chunk_count": len(chunk_ranges),
        "chunks": chunk_meta_list if is_multi_chunk else None,
        "prompt": cfg["prompt"],
        "system_prompt": cfg["system_prompt"],
        "preset": cfg["preset"],
        "reasoning": reasoning,
        "timestamps": timestamps,
        "text": text,
        "reasoning_content": "\n\n".join(reasonings) if reasonings else None,
        "ctc_timestamps": ctc_ts,
        "srt": srt_text,
        "elapsed_sec": round(time.monotonic() - t0, 2),
        "usage": total_usage if total_usage["total_tokens"] > 0 else None,
    }


def _save_local_result(res: dict[str, Any], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_name = Path(res.get("label") or res.get("drive_path") or res.get("local_path") or f"item-{res['index']}").stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-\u4e00-\u9fff]+", "_", raw_name).strip("_") or f"item_{res['index']}"
    saved = []

    txt_path = out_dir / f"{safe_stem}.txt"
    text = res.get("text") or ""
    txt_path.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    saved.append(str(txt_path))

    json_path = out_dir / f"{safe_stem}.nemotron.json"
    json_path.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    saved.append(str(json_path))

    if res.get("srt"):
        srt_path = out_dir / f"{safe_stem}.srt"
        srt_path.write_text(res["srt"], encoding="utf-8")
        saved.append(str(srt_path))
    return saved


def _build_download_parser(parser=None):
    p = parser or argparse.ArgumentParser(
        prog=f"colabllm nemotron-transcribe download",
        description="Use yt-dlp to download audio or full video from media URLs (YouTube, Twitter/X, Bilibili, TikTok etc.) and save it to Google Drive",
    )
    p.add_argument("urls", nargs="+", metavar="URL", help="one or more media URLs (YouTube, Twitter/X, Bilibili etc.)")
    p.add_argument(
        "--download-dir",
        default="",
        metavar="DRIVE_DIR",
        help="Drive save directory (e.g. 'video/voices'; default /content/drive/MyDrive/colab-llm/nemotron-transcribe/audio/yt-dlp)",
    )
    p.add_argument(
        "--download-video",
        dest="download_video",
        action="store_true",
        default=False,
        help="download and save the full video (.mp4) to Drive (by default only the audio is extracted)",
    )
    p.add_argument(
        "--no-download-video",
        dest="download_video",
        action="store_false",
        help="extract and save audio only (default)",
    )
    p.add_argument(
        "--audio-format",
        default="mp3",
        choices=["mp3", "m4a", "wav", "opus", "flac"],
        help="save format when extracting audio only (default: mp3)",
    )
    p.add_argument("--cookies-drive", metavar="DRIVE_PATH", help="path to cookies.txt on Google Drive (for remote Colab yt-dlp)")
    p.add_argument("--cookies", metavar="LOCAL_FILE", help="local cookies.txt path (enables local yt-dlp download, then upload to Drive)")
    p.add_argument("--cookies-from-browser", metavar="BROWSER", help="extract cookies from a local browser (e.g. chrome / firefox)")
    p.add_argument("--playlist", action="store_true", help="if the URL is a playlist, download every item in it (default: single item only)")
    p.add_argument("--force-download", action="store_true", help="force a re-download, ignoring cached files with the same name on Drive")
    p.add_argument("--local-ytdlp", action="store_true", help="force a local yt-dlp download and upload to Drive with `colab upload`")
    p.add_argument("-i", "--instance", metavar="ID", help="use a specific instance (e.g. -i 1)")
    p.add_argument("--url", help="override the public service URL")
    p.add_argument("--api-token", help="override the Bearer API token")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    return p


def _cmd_download(argv) -> None:
    p = _build_download_parser()
    a = argv if isinstance(argv, argparse.Namespace) else p.parse_args(argv)

    if a.instance:
        apply_instance(SERVICE, a.instance)

    use_local = bool(a.local_ytdlp or a.cookies or a.cookies_from_browser)
    all_saved: list[dict[str, Any]] = []
    base_url, token = ("", "")
    if not use_local:
        try:
            base_url, token = _get_endpoint(a.url, a.api_token)
        except CliError:
            use_local = True

    for u in a.urls:
        if not use_local and base_url:
            try:
                log(f"Downloading to Google Drive with remote Colab yt-dlp: {u}")
                res = _http_request(
                    f"{base_url}/api/ct/ytdlp-download",
                    token,
                    payload={
                        "url": u,
                        "download_dir": a.download_dir,
                        "audio_format": a.audio_format,
                        "cookies_drive": a.cookies_drive or "",
                        "playlist": a.playlist,
                        "force_download": a.force_download,
                        "download_video": a.download_video,
                    },
                    timeout=900,
                )
                items = res.get("files") or []
                all_saved.extend(items)
                continue
            except Exception as e:
                warn(f"Remote yt-dlp download failed ({e}); switching to local yt-dlp and uploading to Google Drive...")

        items = _download_media_locally_and_upload_to_drive(
            raw_url=u,
            session_name=SERVICE.name,
            download_dir=a.download_dir,
            audio_format=a.audio_format,
            cookies_file=a.cookies,
            cookies_from_browser=a.cookies_from_browser,
            playlist=a.playlist,
            download_video=a.download_video,
        )
        all_saved.extend(items)

    if a.json:
        print(json.dumps({"count": len(all_saved), "files": all_saved}, indent=2, ensure_ascii=False))
        return
    log(f"Downloaded and saved {len(all_saved)} media files to Google Drive:")
    for item in all_saved:
        cached_tag = " (cached)" if item.get("cached") else ""
        icon = "🎬" if item.get("media_type") == "video" else "🎵"
        print(f"  {icon} {item.get('rel_path') or item['drive_path']}{cached_tag}")
        print(f"     drive_path: {item['drive_path']}")
        if item.get("title"):
            print(f"     title     : {item['title']}")


def _build_transcribe_parser(parser=None):
    p = parser or argparse.ArgumentParser(
        prog=f"colabllm nemotron-transcribe transcribe",
        description="Run Nemotron-3.5-Transcribe-30B-A4B on Colab G4 against media URLs (YouTube/Twitter etc. via yt-dlp), Google Drive, or single/batch local audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "paths",
        nargs="*",
        metavar="AUDIO_OR_URL",
        help="media URL (YouTube/Twitter/Bilibili etc., downloaded to Drive with yt-dlp), Google Drive audio path/directory/glob (e.g. 'video/voices'), or a local path",
    )
    p.add_argument("-p", "--prompt", help="custom user prompt (defaults to the official recommended prompt for --preset)")
    p.add_argument("--prompt-file", help="read the user prompt from a local text file")
    p.add_argument(
        "--preset",
        choices=list(LOCAL_PRESETS.keys()),
        help="use an official NVIDIA task preset (verbatim|clean|speaker|multispeaker|rich|near-field|structured|context|translate|lid|qa)",
    )
    p.add_argument("--context", help="domain vocabulary / names and terms hint (context terms for biased transcription)")
    p.add_argument("--lang", default="Chinese", help="target language for --preset translate (default: Chinese)")
    p.add_argument("-s", "--system-prompt", help="custom system prompt (leave empty for transcription/translation; useful for audio Q&A / assistant tasks)")
    p.add_argument("--reasoning", dest="reasoning", action="store_true", default=None, help="enable reasoning mode (enable_thinking=True)")
    p.add_argument("--no-reasoning", dest="reasoning", action="store_false", help="disable reasoning mode (enable_thinking=False)")
    p.add_argument("-t", "--timestamps", action="store_true", default=None, help="enable CTC word-level timestamps with speaker alignment (also writes .srt subtitles)")
    p.add_argument("-b", "--batch-file", metavar="FILE", help="batch task list (.jsonl / .json / .csv / .tsv / .txt; each line may carry its own audio/url and prompt)")
    p.add_argument("-r", "--recursive", action="store_true", default=False, help="search subdirectories for audio when a directory is given (default: current directory only)")
    p.add_argument("--no-recursive", dest="recursive", action="store_false", help="do not recurse into subdirectories when a directory is given")
    p.add_argument("-j", "--concurrency", type=int, default=1, help="concurrent requests in batch mode (default: 1)")
    p.add_argument(
        "--download-dir",
        default="",
        metavar="DRIVE_DIR",
        help="Drive directory where yt-dlp saves media when the input is a media URL (e.g. 'video/voices'; default colab-llm/nemotron-transcribe/audio/yt-dlp)",
    )
    p.add_argument(
        "--download-video",
        dest="download_video",
        action="store_true",
        default=False,
        help="when the input is a media URL, download the full video (.mp4) to Google Drive (by default only the audio is extracted)",
    )
    p.add_argument(
        "--no-download-video",
        dest="download_video",
        action="store_false",
        help="when the input is a media URL, extract and save audio only (default)",
    )
    p.add_argument(
        "--save-media-drive",
        dest="save_media_drive",
        action="store_true",
        default=True,
        help="keep the media downloaded by yt-dlp on Google Drive (default: on)",
    )
    p.add_argument(
        "--no-save-media-drive",
        dest="save_media_drive",
        action="store_false",
        help="do not keep the media downloaded by yt-dlp on Google Drive (only staged on the Colab local SSD and cleaned up after recognition)",
    )
    p.add_argument(
        "--audio-format",
        default="mp3",
        choices=["mp3", "m4a", "wav", "opus", "flac"],
        help="audio format for yt-dlp extraction (default: mp3)",
    )
    p.add_argument("--cookies-drive", metavar="DRIVE_PATH", help="path to cookies.txt on Google Drive (lets remote Colab yt-dlp download videos that require login)")
    p.add_argument("--cookies", metavar="LOCAL_FILE", help="local cookies.txt path (downloads with local yt-dlp and uploads to Drive automatically)")
    p.add_argument("--cookies-from-browser", metavar="BROWSER", help="extract cookies from a local browser (e.g. chrome / firefox; downloads with local yt-dlp and uploads to Drive)")
    p.add_argument("--playlist", action="store_true", help="if the input URL is a playlist, download and transcribe every entry (default: single item only)")
    p.add_argument("--force-download", action="store_true", help="force yt-dlp to re-download the URL, ignoring cached audio with the same name on Drive")
    p.add_argument("--local-ytdlp", action="store_true", help="force yt-dlp to run on this machine and upload to Google Drive with `colab upload` before transcribing")
    p.add_argument(
        "--save-drive",
        nargs="?",
        const="@same",
        metavar="DRIVE_DIR",
        help="save transcription results (.txt / .nemotron.json / .srt) directly to Google Drive (without a value: next to the source audio; or give a Drive target directory)",
    )
    p.add_argument("-o", "--output", metavar="FILE", help="save results to a local file (text/JSON/SRT for a single file, .jsonl/.json for several)")
    p.add_argument("-O", "--output-dir", metavar="DIR", help="save each audio's .txt / .nemotron.json / .srt plus batch_summary.jsonl to a local directory")
    p.add_argument("--drive", action="store_true", help="force input paths to be resolved as Google Drive paths")
    p.add_argument("--local", action="store_true", help="force input paths to be resolved as local paths")
    p.add_argument("--normalize-audio", action="store_true", help="resample to 16 kHz mono WAV with ffmpeg on the remote side before sending to the model")
    p.add_argument(
        "--chunk-minutes",
        type=float,
        default=40.0,
        metavar="MINUTES",
        help="silence-aligned chunking threshold for very long audio (minutes, default: 40; longer audio is split at silences and text/timestamps are stitched seamlessly; 0 disables chunking)",
    )
    p.add_argument("--temperature", type=float, help="override the sampling temperature (default: 0.0 for transcription / 0.6 for reasoning)")
    p.add_argument("--max-tokens", type=int, help="override the maximum generated tokens (default: 16384 for transcription / 20480 for reasoning)")
    p.add_argument("--timeout", type=nonnegative_seconds, default=3600, help="request timeout per audio task (seconds, 0 for unlimited, default: 3600)")
    p.add_argument("-i", "--instance", metavar="ID", help="use a specific instance (e.g. -i 1 or -i 2)")
    p.add_argument("--instances", metavar="ID1,ID2", help="spread batch tasks across several instances in parallel (e.g. --instances 1,2,3)")
    p.add_argument("-A", "--all-instances", action="store_true", help="discover all running instances and spread batch tasks across them in parallel")
    p.add_argument("--url", help="override the public service URL")
    p.add_argument("--api-token", help="override the Bearer API token")
    p.add_argument("--json", action="store_true", help="print the full JSON result to standard output")
    return p


def _cmd_transcribe(argv) -> None:
    p = _build_transcribe_parser()
    a = argv if isinstance(argv, argparse.Namespace) else p.parse_args(argv)

    if a.instance:
        apply_instance(SERVICE, a.instance)
    if a.prompt_file:
        a.prompt = Path(a.prompt_file).read_text(encoding="utf-8").strip()
    if a.output and Path(a.output).suffix.lower() == ".srt" and not a.timestamps:
        if a.timestamps is False:
            raise UsageError("Exporting .srt subtitles requires --timestamps (or --preset speaker)")
        a.timestamps = True
        diagnostic("[nemotron] Output is .srt; word-level --timestamps enabled automatically")

    endpoints = _get_endpoint_pool(a)
    first_url, first_token = endpoints[0][1], endpoints[0][2]
    tasks = _expand_tasks(a, first_url, first_token)
    total = len(tasks)
    workers = max(a.concurrency, len(endpoints)) if len(endpoints) > 1 and a.concurrency == 1 else max(1, a.concurrency)
    ep_desc = ", ".join(f"{name}({url})" for name, url, _ in endpoints)
    log(f"Preparing {total} audio tasks (concurrency={workers}, instances={len(endpoints)}: {ep_desc})")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    out_dir = Path(a.output_dir) if a.output_dir else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        fut_map = {}
        for idx, task in enumerate(tasks, 1):
            inst_name, ep_url, ep_tok = endpoints[(idx - 1) % len(endpoints)]
            fut = pool.submit(_run_one_task, idx, total, task, a, ep_url, ep_tok)
            fut_map[fut] = (idx, task, inst_name)
        for fut in concurrent.futures.as_completed(fut_map):
            idx, task, inst_name = fut_map[fut]
            label = task.get("id") or task.get("label") or task.get("url") or task.get("drive_path") or task.get("local_path")
            try:
                res = fut.result()
                res["instance"] = inst_name
                if out_dir:
                    res["saved_local_files"] = _save_local_result(res, out_dir)
                results.append(res)
                if not a.json:
                    inst_tag = f" [{inst_name}]" if len(endpoints) > 1 else ""
                    hdr = f"[{idx}/{total}]{inst_tag} {res['label']} ({res.get('elapsed_sec', 0):.2f}s)"
                    print(f"\n=== {hdr} ===", flush=True)
                    if res.get("source_url") and res.get("drive_path"):
                        log(f"Media URL downloaded to Drive: {res['drive_path']} (source: {res['source_url']})")
                    if (res.get("chunk_count") or 1) > 1:
                        dur_m = (res.get("duration_sec") or 0) / 60.0
                        log(f"Long audio ({dur_m:.1f} minutes) was split at silences into {res['chunk_count']} chunks and text/timestamps were stitched together")
                    print(res.get("text") or "", flush=True)
                    if res.get("saved_drive_files"):
                        log(f"Written to Drive: {', '.join(res['saved_drive_files'])}")
                    if res.get("saved_local_files"):
                        log(f"Written locally: {', '.join(res['saved_local_files'])}")
            except Exception as e:
                err_rec = {"index": idx, "instance": inst_name, "label": label, "error": str(e), "task": task}
                errors.append(err_rec)
                warn(f"[{idx}/{total}] [{inst_name}] {label} failed: {e}")

    results.sort(key=lambda r: r.get("index", 0))

    if out_dir:
        summary_path = out_dir / "batch_summary.jsonl"
        with summary_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log(f"Batch summary saved to: {summary_path}")

    if a.output:
        out_file = Path(a.output)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if len(results) == 1 and out_file.suffix.lower() in (".txt", ".md"):
            txt = results[0].get("text") or ""
            out_file.write_text(txt + ("\n" if not txt.endswith("\n") else ""), encoding="utf-8")
        elif out_file.suffix.lower() == ".srt":
            if len(results) != 1 or not results[0].get("srt"):
                die("Cannot write .srt: only a single audio is supported, and the server must return word-level timestamps (--timestamps)")
            out_file.write_text(results[0]["srt"], encoding="utf-8")
        elif out_file.suffix.lower() == ".jsonl":
            with out_file.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            payload_out = results[0] if len(results) == 1 else {"results": results, "errors": errors}
            out_file.write_text(json.dumps(payload_out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        log(f"Output saved to: {out_file}")

    if a.json:
        out_obj = results[0] if (len(results) == 1 and not errors) else {"results": results, "errors": errors}
        print(json.dumps(out_obj, indent=2, ensure_ascii=False))

    if errors:
        die(f"Finished {len(results)}/{total} tasks; {len(errors)} tasks failed.")



def configure_cli(parser, commands, up):
    _build_transcribe_parser(commands.add_parser("transcribe", aliases=["run"], help="transcribe audio"))
    _build_download_parser(commands.add_parser("download", aliases=["yt-dlp", "ytdlp"], help="download media to Drive"))
    _build_ls_drive_parser(commands.add_parser("ls-drive", help="list Drive audio"))
    commands.add_parser("prompts", help="list transcription prompt presets")
    up.add_argument("--precision", choices=tuple(PRECISION_REPOS))
    up.add_argument("--repo", help="override model repository")
    up.add_argument("--mtp", type=int, help="override MTP value")


_BASE_ENV = dict(SERVICE.env)


def prepare_up(args):
    # Start from the baseline defaults on every call so overrides never leak between invocations.
    SERVICE.env.clear()
    SERVICE.env.update(_BASE_ENV)
    if args.precision:
        SERVICE.env.update(CT_NEMOTRON_PRECISION=args.precision, CT_NEMOTRON_REPO=PRECISION_REPOS[args.precision])
    if args.repo:
        SERVICE.env["CT_NEMOTRON_REPO"] = args.repo
    if args.mtp is not None:
        SERVICE.env["CT_NEMOTRON_MTP"] = str(args.mtp)


if __name__ == "__main__":
    from cli_support import run_cli
    run_cli(lambda: main(SERVICE, configure=configure_cli, prepare_up=prepare_up,
        handlers={"transcribe": _cmd_transcribe, "run": _cmd_transcribe, "download": _cmd_download,
                  "yt-dlp": _cmd_download, "ytdlp": _cmd_download, "ls-drive": _cmd_ls_drive,
                  "prompts": lambda args: _cmd_prompts()}))
