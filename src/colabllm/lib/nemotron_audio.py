"""Shared audio chunking, SRT formatting, and prompt presets for Nemotron Transcribe."""
from __future__ import annotations

import base64
import math
import re
import subprocess
from pathlib import Path
from typing import Any

AUDIO_EXTS = (
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".mkv", ".wma", ".mpga", ".mpeg",
)
DIRECT_MIMES = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}

PRESETS: dict[str, dict[str, Any]] = {
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
        "description": "Speaker-diarized transcription + word-level timestamps (<spk:0> <spk:1> tags; use with timestamps=True)",
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
        "description": "Context-biased transcription with domain vocabulary / proper-noun hints (use with the context parameter)",
        "user_prompt": "Transcribe with punctuation and capitalization. Context terms: {context}.",
        "system_prompt": "",
        "reasoning": False,
        "timestamps": False,
    },
    "translate": {
        "description": "Direct speech translation into target-language text (use with the lang parameter, e.g. Chinese / English / Japanese)",
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
        "description": "Audio understanding and Q&A (reasoning mode enable_thinking=True; answers questions about the recording)",
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

YTDLP_DOMAIN_MARKERS = (
    "youtube.com/", "youtu.be/", "twitter.com/", "x.com/", "t.co/",
    "bilibili.com/", "b23.tv/", "tiktok.com/", "soundcloud.com/",
    "vimeo.com/", "twitch.tv/", "xiaoyuzhoufm.com/", "podcasts.apple.com/",
    "ytdl://", "yt-dlp://",
)

CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    if not text:
        return text
    text = CHAR_RUN.sub(lambda m: m.group(1), text)
    return BLOCK_RUN.sub(lambda m: m.group(1), text)


def probe_audio_duration_sec(audio_path: Path) -> float:
    """Return audio/video duration in seconds using ffprobe (0.0 if unknown)."""
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
            timeout=30,
            check=False,
        )
        val = (res.stdout or "").strip()
        return float(val) if val and val != "N/A" else 0.0
    except Exception:
        return 0.0


def find_silence_near(audio_path: Path, target_sec: float, window_sec: float = 45.0) -> float:
    """Search for a natural speech pause within [target_sec - window_sec, target_sec + window_sec]."""
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
            timeout=25,
            check=False,
        )
        log_out = res.stderr or ""
        starts = [float(x) for x in re.findall(r"silence_start:\s*([0-9.]+)", log_out)]
        ends = [float(x) for x in re.findall(r"silence_end:\s*([0-9.]+)", log_out)]
        candidates: list[float] = []
        for idx, st in enumerate(starts):
            ed = ends[idx] if idx < len(ends) else (st + 0.35)
            mid = win_start + (st + ed) / 2.0
            candidates.append(mid)
        if candidates:
            return min(candidates, key=lambda c: abs(c - target_sec))
    except Exception:
        pass
    return target_sec


def compute_chunk_ranges(
    audio_path: Path,
    total_duration: float,
    max_chunk_sec: float = 2400.0,
) -> list[tuple[float, float]]:
    """Split audio longer than max_chunk_sec into balanced, silence-aligned (start_sec, end_sec) chunks."""
    if max_chunk_sec <= 0 or total_duration <= 0 or total_duration <= max_chunk_sec:
        return [(0.0, max(total_duration, 0.0))]

    num_chunks = max(2, int(math.ceil(total_duration / max_chunk_sec)))
    ideal_len = total_duration / num_chunks
    boundaries = [0.0]
    for k in range(1, num_chunks):
        target = k * ideal_len
        split_pt = find_silence_near(audio_path, target, window_sec=min(45.0, ideal_len * 0.15))
        split_pt = max(boundaries[-1] + 30.0, min(split_pt, total_duration - 30.0))
        boundaries.append(round(split_pt, 3))
    boundaries.append(round(total_duration, 3))
    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def load_audio_as_data_url(
    audio_path: Path,
    *,
    force_ffmpeg_16k: bool = False,
    start_sec: float | None = None,
    end_sec: float | None = None,
    direct_max_bytes: int = 140 * 1024 * 1024,
) -> str:
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    ext = audio_path.suffix.lower()
    sz = audio_path.stat().st_size
    is_slice = (start_sec is not None and start_sec > 0.0) or (end_sec is not None and end_sec > 0.0)
    if not is_slice and not force_ffmpeg_16k and ext in DIRECT_MIMES and sz <= direct_max_bytes:
        b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        return f"data:{DIRECT_MIMES[ext]};base64,{b64}"
    cmd = ["ffmpeg", "-nostdin", "-v", "error"]
    if start_sec is not None and start_sec > 0.0:
        cmd.extend(["-ss", f"{start_sec:.3f}"])
    if end_sec is not None and end_sec > 0.0:
        dur = max(0.1, end_sec - (start_sec or 0.0))
        cmd.extend(["-t", f"{dur:.3f}"])
    cmd.extend([
        "-i", str(audio_path),
        "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
    ])
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300, check=False)
    if res.returncode != 0 or not res.stdout:
        err = res.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed decoding {audio_path.name}: {err}")
    b64 = base64.b64encode(res.stdout).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def shift_ctc_timestamps(ctc_ts: dict[str, Any] | None, offset_sec: float) -> dict[str, Any] | None:
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


def fmt_srt_time(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def words_to_srt(words: list[dict[str, Any]]) -> str:
    if not words:
        return ""
    segments = []
    cur_words: list[str] = []
    cur_start = 0.0
    cur_end = 0.0
    cur_spk = None

    def flush():
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
        spk_changed = w_spk != cur_spk
        if (
            spk_changed
            or gap > 1.2
            or len(cur_words) >= 14
            or (len(cur_words) >= 6 and cur_words[-1].endswith((".", "?", "!", "\u3002", "\uff1f", "\uff01")))
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

    lines = []
    for idx, (st_t, ed_t, txt) in enumerate(segments, 1):
        lines.append(f"{idx}\n{fmt_srt_time(st_t)} --> {fmt_srt_time(ed_t)}\n{txt}\n")
    return "\n".join(lines)


def build_prompt_and_settings(body: dict[str, Any]) -> tuple[str, str, bool, bool]:
    preset_name = (body.get("preset") or "").strip().lower()
    preset = PRESETS.get(preset_name) if preset_name else None

    user_prompt = body.get("prompt")
    if user_prompt is None or str(user_prompt).strip() == "":
        user_prompt = preset["user_prompt"] if preset else PRESETS["verbatim"]["user_prompt"]
    user_prompt = str(user_prompt)

    context_terms = (body.get("context") or "").strip()
    target_lang = (body.get("lang") or body.get("language") or "Chinese").strip()
    if "{context}" in user_prompt:
        user_prompt = user_prompt.replace("{context}", context_terms or "general")
    elif context_terms and "Context terms:" not in user_prompt:
        user_prompt = f"{user_prompt.rstrip()} Context terms: {context_terms}."
    if "{lang}" in user_prompt:
        user_prompt = user_prompt.replace("{lang}", target_lang)

    sys_prompt = body.get("system_prompt")
    if sys_prompt is None:
        sys_prompt = preset["system_prompt"] if preset else ""
    sys_prompt = str(sys_prompt)

    reasoning = body.get("reasoning")
    if reasoning is None:
        reasoning = bool(preset["reasoning"]) if preset else False
    else:
        reasoning = bool(reasoning)

    timestamps = body.get("timestamps")
    if timestamps is None:
        timestamps = bool(preset["timestamps"]) if preset else False
    else:
        timestamps = bool(timestamps)

    return user_prompt, sys_prompt, reasoning, timestamps
