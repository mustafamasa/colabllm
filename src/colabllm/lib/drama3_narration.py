"""Pure-stdlib novel chunking, multi-character dialogue attribution, token estimation, and WAV stitching for Drama 3."""
from __future__ import annotations

from array import array
import io
import json
import math
import re
import sys
from typing import Any
import wave

DRAMA3_SAMPLE_RATE = 44100
DRAMA3_FRAME_RATE = 44100.0 / 2048.0  # ~21.5332 codec frames/sec (2048 frames ~= 95.11s ceiling)
DRAMA3_MAX_CODEC_FRAMES = 2048
DEFAULT_CHUNK_MAX_CHARS = 180

_CHAPTER_HEADING_RE = re.compile(
    r"^\s*(?:"
    r"#{1,3}\s+\S[^\n]{0,100}"
    r"|第[零一二三四五六七八九十百千万两0-9]+[章节回卷部集篇幕](?:[\s:：、.\-—]+[^\n]{0,80}|\s*)"
    r"|(?:楔子|序章|序幕|引子|尾声|后记|番外)(?:[\s:：、.\-—]+[^\n]{0,80}|\s*)"
    r"|(?:chapter|part|book|act|scene)\s+[0-9ivxlcdm]+\b[^\n]{0,90}"
    r"|(?:prologue|epilogue)\b[^\n]{0,80}"
    r")\s*$",
    re.IGNORECASE,
)

_UNSUPPORTED_SPEAKER_TAG_RE = re.compile(r"<\|speaker:([1-9]\d*)\|>")
_INLINE_ROLE_RE = re.compile(r"\[(?:role|speaker|voice)\s*[:=]\s*([^\]\r\n]{1,64})\]", re.IGNORECASE)
_INLINE_TAG_RE = re.compile(r"\[[A-Za-z][A-Za-z0-9 _-]{1,30}\]")
_CJK_CHAR_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
_MAJOR_PUNCT_RE = re.compile(r"[。！？!?；;…]+")
_MINOR_PUNCT_RE = re.compile(r"[，,、：:—]+")
_SENTENCE_SPLIT_RE = re.compile(r"([^。！？!?；;…\n]+[。！？!?；;…]+[\"'”’』」）】》]*|[^\n]+)")
_CLAUSE_SPLIT_RE = re.compile(r"([^，,、：:—\n]+[，,、：:—]+[\"'”’』」）】》]*|[^\n]+)")
_DIALOGUE_ONLY_RE = re.compile(
    r"^\s*(?:“[^”]+”|「[^」]+」|『[^』]+』|\"[^\"]+\")\s*$"
)
_QUOTE_PAIR_RE = re.compile(
    r"(“[^”\n]+”|「[^」\n]+」|『[^』\n]+』|(?<![A-Za-z0-9])\"[^\"\n]{2,}\"(?![A-Za-z0-9]))"
)
_SPEECH_CUE_RE = re.compile(
    r"(?:说道|问道|答道|笑道|叹道|喊道|叫道|喝道|冷道|怒道|低声道|轻声道|喃喃道|回答道|追问道|开口道|"
    r"低语|喃喃|开口|回应|反问|追问|轻笑|冷笑|苦笑|叹息|叮嘱|吩咐|解释|"
    r"说|问|答|道|笑|叹|喊|叫|喝|吼|骂)\s*[：:，,]?\s*$"
)
_NARRATOR_ROLES = frozenset({"narrator", "旁白", "叙述", "叙述者", "default"})


def sanitize_drama3_text(text: str) -> str:
    """Normalize line endings and strip unsupported multi-speaker tags (<|speaker:1..15|>)."""
    cleaned = (text or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _UNSUPPORTED_SPEAKER_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def slugify_title(title: str, index: int) -> str:
    """Create a filesystem-safe chapter stem preserving ASCII/CJK characters."""
    raw = re.sub(r"^#{1,6}\s*", "", (title or "").strip())
    safe = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", raw, flags=re.UNICODE).strip("-_")
    if not safe:
        safe = f"chapter-{index:03d}"
    return f"{index:03d}-{safe[:48]}"


def split_into_chapters(text: str) -> list[dict[str, Any]]:
    """Split a full novel/book manuscript into ordered chapters."""
    normalized = (text or "").replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    lines = normalized.split("\n")
    raw_sections: list[tuple[str, list[str]]] = []
    current_title: str | None = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) <= 120 and _CHAPTER_HEADING_RE.match(stripped):
            if current_lines and any(x.strip() for x in current_lines):
                title = current_title or "序"
                raw_sections.append((title, current_lines))
            current_title = re.sub(r"^#{1,6}\s*", "", stripped).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines and any(x.strip() for x in current_lines):
        if current_title is None and not raw_sections:
            return [{
                "index": 1,
                "title": "Chapter 1",
                "slug": "001-chapter-1",
                "text": normalized,
            }]
        raw_sections.append((current_title or f"Chapter {len(raw_sections) + 1}", current_lines))
    elif current_title is not None:
        raw_sections.append((current_title, [current_title]))

    chapters: list[dict[str, Any]] = []
    for idx, (title, body_lines) in enumerate(raw_sections, 1):
        body = "\n".join(body_lines).strip()
        if body and not body.startswith(title):
            spoken_text = f"[role:旁白] {title}。\n\n{body}" if _INLINE_ROLE_RE.search(body) else f"{title}。\n\n{body}"
        else:
            spoken_text = body or title
        chapters.append({
            "index": idx,
            "title": title,
            "slug": slugify_title(title, idx),
            "text": spoken_text,
        })
    return chapters


def extract_cast_name_map(cast: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Return (alias_or_name, canonical_role) pairs sorted longest-first for speaker matching."""
    if not isinstance(cast, dict):
        return []
    chars = cast.get("characters") or cast.get("roles") or {}
    pairs: dict[str, str] = {}
    if isinstance(chars, dict):
        for role_name, cfg in chars.items():
            r_clean = str(role_name).strip()
            if not r_clean or r_clean.lower() in _NARRATOR_ROLES:
                continue
            pairs[r_clean] = r_clean
            if isinstance(cfg, dict):
                for alias in cfg.get("aliases") or []:
                    a_clean = str(alias).strip()
                    if a_clean:
                        pairs[a_clean] = r_clean
    aliases_map = cast.get("aliases")
    if isinstance(aliases_map, dict):
        for alias, canonical in aliases_map.items():
            a_clean = str(alias).strip()
            c_clean = str(canonical).strip()
            if a_clean and c_clean:
                pairs[a_clean] = c_clean
    return sorted(pairs.items(), key=lambda kv: (-len(kv[0]), kv[0]))


def _infer_speaker_from_context(
    context_text: str,
    name_map: list[tuple[str, str]],
    *,
    is_preceding: bool,
    quote_text: str = "",
) -> str | None:
    """Infer which character in `name_map` spoke an adjacent quotation."""
    if not context_text or not name_map:
        return None

    raw_ctx = context_text.strip()
    if is_preceding:
        # Focus first on the sentence immediately preceding the opening quote
        sentences = [s.strip() for s in re.split(r"[。！？!?；;\n]+", raw_ctx[-120:]) if s.strip()]
        candidates = [sentences[-1], raw_ctx[-80:]] if sentences else [raw_ctx[-80:]]
    else:
        # Post-quote attribution ("...”林晚说道。) must come from the immediate first sentence after the quote.
        first_sent = re.split(r"(?<=[。！？!?；;\n])", raw_ctx[:120])[0].strip()
        if not first_sent:
            return None
        # If the single clause right after a full-stop quote ends with ':' / '：', it introduces the *next* quote.
        q_inner = quote_text[1:-1].strip() if len(quote_text) >= 2 else ""
        if first_sent.endswith(("：", ":")) and not q_inner.endswith(("，", ",")):
            return None
        candidates = [first_sent]

    escaped = [re.escape(alias) for alias, _ in name_map]
    alias_to_role = dict(name_map)
    names_union = "|".join(escaped)

    prep_re = re.compile(
        rf"(?P<subj>{names_union})[^。！？；\n]{{0,8}}?(?:对|向|朝|冲|望着|看着|盯着|朝着|转向)"
        rf"(?:{names_union}|[^，。！？；：\s]{{1,8}})[^。！？；\n]{{0,14}}?"
        rf"(?:说|问|答|道|笑|叹|喊|叫|开口|低语|喃喃)"
    )
    verb_re = re.compile(
        rf"(?P<subj>{names_union})[^，。！？；：“”\n]{{0,12}}?"
        rf"(?:说道|问道|答道|笑道|叹道|喊道|叫道|喝道|低声道|轻声道|喃喃道|"
        rf"低语|喃喃|开口|回应|反问|追问|轻笑|冷笑|苦笑|说|问|答|道|笑|叹|喊|叫|喝)"
    )

    for snippet in candidates:
        # 1. Prepositional addressee pattern: "林晚看着陈默，轻声问道：" -> speaker is 林晚 (subject), not 陈默!
        prep_hits = list(prep_re.finditer(snippet))
        if prep_hits:
            chosen = prep_hits[-1] if is_preceding else prep_hits[0]
            return alias_to_role.get(chosen.group("subj"))

        # 2. Direct speech-verb subject pattern: "陈默抬头笑了笑：" / "”林晚轻声道"
        verb_hits = list(verb_re.finditer(snippet))
        if verb_hits:
            chosen = verb_hits[-1] if is_preceding else verb_hits[0]
            return alias_to_role.get(chosen.group("subj"))

        # 3. Single-mention fallback within the immediate sentence
        hits = [(m.start(), alias_to_role[m.group(0)]) for m in re.finditer(names_union, snippet)]
        if len({r for _, r in hits}) == 1:
            return hits[0][1]
        if hits:
            return hits[-1][1] if is_preceding else hits[0][1]

    return None


def _is_spoken_dialogue_quote(para: str, match: re.Match[str]) -> bool:
    """Distinguish spoken character lines from short inline scare-quotes (e.g. 所谓的“规矩”)."""
    quote_text = match.group(0)
    inner = quote_text[1:-1].strip()
    if not inner:
        return False
    start, end = match.start(), match.end()
    before = para[:start].rstrip()
    after = para[end:].lstrip()

    # Standalone or starts at paragraph/sentence boundary
    if not before or before[-1] in "：:。！？!?；;\n":
        return True
    # Preceded by speech cue verb ("说道，“..." or "问道：“...")
    if _SPEECH_CUE_RE.search(before[-16:]):
        return True
    # Ends with terminal/dialogue punctuation inside the quote
    if inner[-1] in "。！？!?…~～，,——":
        return True
    # Followed by post-quote speech attribution ("...”林晚说道")
    if after and re.match(r"^[^。！？\n]{0,10}(?:道|说|问|答|笑|叹|喊)", after):
        return True
    # Long quoted utterance (>= 10 chars)
    if len(inner) >= 10:
        return True
    return False


def _split_paragraph_into_turns(
    para: str,
    *,
    split_dialogue: bool,
    name_map: list[tuple[str, str]],
    recent_speakers: list[str],
    allow_alternation: bool = True,
) -> list[dict[str, Any]]:
    """Split a paragraph into role-attributed narration and dialogue turns."""
    # First split on explicit [role:...] markers anywhere in the paragraph
    role_matches = list(_INLINE_ROLE_RE.finditer(para))
    if role_matches:
        turns: list[dict[str, Any]] = []
        if role_matches[0].start() > 0:
            prefix = para[: role_matches[0].start()].strip()
            if prefix:
                turns.append({
                    "text": prefix,
                    "role_hint": None,
                    "is_dialogue": bool(_DIALOGUE_ONLY_RE.match(prefix)),
                })
        for idx, m in enumerate(role_matches):
            role_name = m.group(1).strip()
            seg_end = role_matches[idx + 1].start() if idx + 1 < len(role_matches) else len(para)
            seg_text = para[m.end() : seg_end].strip()
            if not seg_text:
                continue
            is_narr = role_name.lower() in _NARRATOR_ROLES
            if not is_narr and role_name not in recent_speakers[-1:]:
                recent_speakers.append(role_name)
            turns.append({
                "text": seg_text,
                "role_hint": None if is_narr else role_name,
                "is_dialogue": (not is_narr) or bool(_DIALOGUE_ONLY_RE.match(seg_text)),
            })
        return turns

    if not split_dialogue:
        is_diag = bool(_DIALOGUE_ONLY_RE.match(para))
        return [{"text": para, "role_hint": None, "is_dialogue": is_diag}]

    # Detect spoken dialogue quotes inside an unannotated paragraph
    spoken_matches = [m for m in _QUOTE_PAIR_RE.finditer(para) if _is_spoken_dialogue_quote(para, m)]
    if not spoken_matches:
        return [{"text": para, "role_hint": None, "is_dialogue": False}]

    raw_spans: list[dict[str, Any]] = []
    pos = 0
    for m in spoken_matches:
        if m.start() > pos:
            narr_seg = para[pos : m.start()].strip()
            if narr_seg:
                raw_spans.append({"text": narr_seg, "is_dialogue": False, "role_hint": None})
        q_text = m.group(0).strip()
        if q_text:
            raw_spans.append({"text": q_text, "is_dialogue": True, "role_hint": None})
        pos = m.end()
    if pos < len(para):
        tail_seg = para[pos:].strip()
        if tail_seg:
            # Pure punctuation tail like "。" stays attached to previous span
            if raw_spans and not re.search(r"[\w\u4e00-\u9fff]", tail_seg):
                raw_spans[-1]["text"] += tail_seg
            else:
                raw_spans.append({"text": tail_seg, "is_dialogue": False, "role_hint": None})

    # Attribute each dialogue span using preceding narration, then following narration, then 2-speaker alternation
    inferred_in_para: str | None = None
    for i, sp in enumerate(raw_spans):
        if not sp["is_dialogue"]:
            continue
        speaker: str | None = None
        if i > 0 and not raw_spans[i - 1]["is_dialogue"]:
            speaker = _infer_speaker_from_context(
                raw_spans[i - 1]["text"], name_map, is_preceding=True, quote_text=sp["text"]
            )
        if not speaker and i + 1 < len(raw_spans) and not raw_spans[i + 1]["is_dialogue"]:
            speaker = _infer_speaker_from_context(
                raw_spans[i + 1]["text"], name_map, is_preceding=False, quote_text=sp["text"]
            )
        if not speaker and inferred_in_para and i >= 2 and raw_spans[i - 2]["is_dialogue"]:
            # Interrupted quote by the same speaker: “快进来，”陈默笑道，“外面风大。”
            mid_narr = raw_spans[i - 1]["text"] if not raw_spans[i - 1]["is_dialogue"] else ""
            if len(mid_narr) <= 24:
                speaker = raw_spans[i - 2].get("role_hint") or inferred_in_para
        if (
            not speaker
            and allow_alternation
            and len(raw_spans) == 1
            and len(recent_speakers) >= 2
            and recent_speakers[-1] != recent_speakers[-2]
        ):
            speaker = recent_speakers[-2]

        sp["role_hint"] = speaker
        if speaker:
            inferred_in_para = speaker
            if not recent_speakers or recent_speakers[-1] != speaker:
                recent_speakers.append(speaker)

    return raw_spans


def _hard_slice_piece(piece: str, max_chars: int) -> list[str]:
    text = piece.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    out: list[str] = []
    rem = text
    while len(rem) > max_chars:
        window = rem[:max_chars]
        sp = window.rfind(" ")
        cut = sp if sp >= max_chars // 2 else max_chars
        chunk = rem[:cut].strip()
        if chunk:
            out.append(chunk)
        rem = rem[cut:].strip()
    if rem:
        out.append(rem)
    return out


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    s = sentence.strip()
    if not s:
        return []
    if len(s) <= max_chars:
        return [s]
    clauses = [m.group(0).strip() for m in _CLAUSE_SPLIT_RE.finditer(s) if m.group(0).strip()]
    packed: list[str] = []
    cur = ""
    for cl in clauses:
        if len(cl) > max_chars:
            if cur:
                packed.append(cur)
                cur = ""
            packed.extend(_hard_slice_piece(cl, max_chars))
            continue
        candidate = f"{cur}{cl}" if not cur or not (cur[-1].isascii() and cl[0].isascii()) else f"{cur} {cl}"
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                packed.append(cur)
            cur = cl
    if cur:
        packed.append(cur)
    return packed


def split_text_chunks(
    text: str,
    max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    *,
    split_dialogue: bool = False,
    cast: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Split text into sentence-aligned narration/dialogue chunks bounded below Drama 3's 95s limit."""
    if max_chars < 20 or max_chars > 500:
        raise ValueError("max_chars must be between 20 and 500 (Drama 3 caps each generation at ~95s / 2048 frames)")
    cleaned = sanitize_drama3_text(text)
    if not cleaned:
        return []

    name_map = extract_cast_name_map(cast)
    allow_alt = bool(cast.get("alternate_unattributed", True)) if isinstance(cast, dict) else True
    recent_speakers: list[str] = []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+|\n", cleaned) if p.strip()]
    chunks: list[dict[str, Any]] = []

    for p_idx, para in enumerate(paragraphs):
        turns = _split_paragraph_into_turns(
            para,
            split_dialogue=split_dialogue,
            name_map=name_map,
            recent_speakers=recent_speakers,
            allow_alternation=allow_alt,
        )
        if not turns:
            continue

        for t_idx, turn in enumerate(turns):
            t_text = turn["text"].strip()
            if not t_text:
                continue
            raw_sentences = [m.group(0).strip() for m in _SENTENCE_SPLIT_RE.finditer(t_text) if m.group(0).strip()]
            units: list[str] = []
            for sent in raw_sentences:
                units.extend(_split_long_sentence(sent, max_chars))

            turn_chunks: list[str] = []
            cur = ""
            for u in units:
                sep = " " if (cur and cur[-1].isascii() and u[0].isascii()) else ""
                cand = f"{cur}{sep}{u}" if cur else u
                if len(cand) <= max_chars:
                    cur = cand
                else:
                    if cur:
                        turn_chunks.append(cur)
                    cur = u
            if cur:
                turn_chunks.append(cur)

            is_last_turn_in_para = t_idx == (len(turns) - 1)
            for c_i, c_text in enumerate(turn_chunks):
                is_last_chunk_in_turn = c_i == (len(turn_chunks) - 1)
                chunks.append({
                    "index": len(chunks),
                    "paragraph_index": p_idx,
                    "text": c_text,
                    "char_count": len(c_text),
                    "paragraph_end": is_last_turn_in_para and is_last_chunk_in_turn,
                    "turn_transition": (not is_last_turn_in_para) and is_last_chunk_in_turn,
                    "role_hint": turn.get("role_hint"),
                    "is_dialogue": bool(turn.get("is_dialogue")),
                })
    return chunks


def resolve_chunk_voice_spec(
    chunk: dict[str, Any],
    default_voice: str | None = None,
    cast: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve {'voice': ..., 'speed': ..., 'ref_audio': ..., 'ref_text': ...} for a chunk."""
    role = chunk.get("role_hint")

    def _unpack(spec: Any) -> dict[str, Any]:
        if isinstance(spec, str) and spec.strip():
            return {"voice": spec.strip()}
        if isinstance(spec, dict):
            out: dict[str, Any] = {}
            if spec.get("voice"):
                out["voice"] = str(spec["voice"]).strip()
            if spec.get("speed") is not None:
                out["speed"] = float(spec["speed"])
            if spec.get("ref_audio"):
                out["ref_audio"] = str(spec["ref_audio"]).strip()
            if spec.get("ref_text"):
                out["ref_text"] = str(spec["ref_text"]).strip()
            return out
        return {}

    if isinstance(cast, dict):
        chars = cast.get("characters") or cast.get("roles") or {}
        aliases = cast.get("aliases") or {}
        if role:
            canon = aliases.get(role, role) if isinstance(aliases, dict) else role
            if isinstance(chars, dict) and canon in chars:
                res = _unpack(chars[canon])
                if res:
                    return res
            if canon in cast and canon not in ("characters", "roles", "aliases", "alternate_unattributed"):
                res = _unpack(cast[canon])
                if res:
                    return res
        if chunk.get("is_dialogue") and cast.get("dialogue"):
            res = _unpack(cast["dialogue"])
            if res:
                return res
        if cast.get("narrator"):
            res = _unpack(cast["narrator"])
            if res:
                return res

    if role and role.lower() not in _NARRATOR_ROLES:
        return {"voice": str(role)}
    if default_voice:
        return {"voice": default_voice}
    return {}


def assign_chunk_voice(
    chunk: dict[str, Any],
    default_voice: str | None = None,
    cast: dict[str, Any] | None = None,
) -> str | None:
    """Resolve the voice name string for a chunk using optional cast configuration."""
    return resolve_chunk_voice_spec(chunk, default_voice=default_voice, cast=cast).get("voice")


def _content_skeleton(text: str) -> str:
    """Extract alphanumeric + CJK skeleton ignoring tags, quotes, and punctuation for fidelity checks."""
    no_roles = _INLINE_ROLE_RE.sub("", text or "")
    no_tags = _INLINE_TAG_RE.sub("", no_roles)
    return "".join(re.findall(r"[\w\u4e00-\u9fff]", no_tags))


def build_llm_cast_messages(
    window_text: str,
    character_names: list[str],
    *,
    add_emotion_tags: bool = False,
) -> list[dict[str, str]]:
    """Construct chat completion messages to convert raw novel prose into [role:Name] turns."""
    known_str = "、".join(["旁白", *[c for c in character_names if c != "旁白"]]) if character_names else "旁白、对白角色"
    tag_rule = (
        "可在角色台词情感极其明显处少量插入英文方括号副语言标签（仅限 [sighs]、[laughs]、[gasp]），严禁滥用；"
        if add_emotion_tags
        else "不要添加任何原文没有的词语或标签；"
    )
    sys_prompt = (
        "你是有声书剧本分角色标注器。请将输入的小说片段拆分为按顺序排列的朗读片段 JSON 数组。\n"
        f"已知角色列表：{known_str}（若遇到未列出的说话角色，使用『对白』或原文明确姓名；所有叙述文字一律使用『旁白』）。\n"
        "严格规则：\n"
        "1. 必须将段落内的叙述文字（如『林晚推开门问道：』）与引号内的角色台词（如『“你来了？”』）拆成相邻的独立元素，分别标为『旁白』和对应说话人。\n"
        f"2. 100% 逐字保留原文汉字、词语和顺序，严禁删减、缩写或改写任何原文句子！{tag_rule}\n"
        "3. 只输出合法的 JSON 数组，格式为：[{\"role\": \"旁白\", \"text\": \"...\"}, {\"role\": \"林晚\", \"text\": \"...\"}]，不要输出 Markdown 代码块说明。"
    )
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": window_text},
    ]


def parse_and_verify_llm_cast_output(raw_reply: str, original_window: str) -> str | None:
    """Parse LLM JSON turn array into `[role:Name] text` script and verify verbatim fidelity."""
    cleaned = (raw_reply or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start_bracket = cleaned.find("[")
    end_bracket = cleaned.rfind("]")
    if start_bracket < 0 or end_bracket <= start_bracket:
        return None
    try:
        items = json.loads(cleaned[start_bracket : end_bracket + 1])
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None

    lines: list[str] = []
    reconstructed_parts: list[str] = []
    for entry in items:
        if not isinstance(entry, dict):
            return None
        role = str(entry.get("role") or "旁白").strip().replace("]", "")
        txt = str(entry.get("text") or "").strip()
        if not txt:
            continue
        lines.append(f"[role:{role}] {txt}")
        reconstructed_parts.append(txt)

    if not lines:
        return None

    orig_skel = _content_skeleton(original_window)
    new_skel = _content_skeleton("".join(reconstructed_parts))
    if not orig_skel:
        return "\n\n".join(lines)
    # Require tight length conservation so truncated or hallucinated outputs are rejected
    ratio = len(new_skel) / float(len(orig_skel))
    if ratio < 0.94 or ratio > 1.06:
        return None
    return "\n\n".join(lines)


def split_into_llm_windows(text: str, max_window_chars: int = 1600) -> list[str]:
    """Group paragraphs into bounded windows for reliable LLM annotation."""
    paras = [p.strip() for p in re.split(r"\n\s*\n+|\n", (text or "").strip()) if p.strip()]
    windows: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in paras:
        if cur and cur_len + len(p) > max_window_chars:
            windows.append("\n\n".join(cur))
            cur = [p]
            cur_len = len(p)
        else:
            cur.append(p)
            cur_len += len(p)
    if cur:
        windows.append("\n\n".join(cur))
    return windows


def estimate_max_new_tokens(
    text: str,
    *,
    speed: float = 1.0,
    min_tokens: int = 192,
    max_tokens: int = DRAMA3_MAX_CODEC_FRAMES,
) -> int:
    """Estimate a safe codec-frame cap for Drama 3 so generation cannot runaway into long silence."""
    tags = _INLINE_TAG_RE.findall(text or "")
    stripped = _INLINE_TAG_RE.sub(" ", text or "")
    cjk_count = len(_CJK_CHAR_RE.findall(stripped))
    non_cjk = _CJK_CHAR_RE.sub(" ", stripped)
    word_count = len(_LATIN_WORD_RE.findall(non_cjk))
    major_pauses = len(_MAJOR_PUNCT_RE.findall(stripped))
    minor_pauses = len(_MINOR_PUNCT_RE.findall(stripped))

    est_seconds = (
        cjk_count * 0.25
        + word_count * 0.40
        + major_pauses * 0.25
        + minor_pauses * 0.10
        + len(tags) * 1.10
    )
    eff_speed = max(0.25, min(float(speed or 1.0), 4.0))
    est_seconds = max(1.5, est_seconds / eff_speed)
    raw_frames = int(math.ceil(est_seconds * DRAMA3_FRAME_RATE * 1.85 + 96))
    lo = max(32, min(min_tokens, max_tokens))
    return max(lo, min(raw_frames, max_tokens))


def parse_chapter_selection(spec: str | None, total_chapters: int) -> set[int]:
    """Parse 1-based chapter selector like '1', '1-5', or '1,3,5-7'."""
    if not spec or not spec.strip():
        return set(range(1, total_chapters + 1))
    selected: set[int] = set()
    for part in spec.split(","):
        tok = part.strip()
        if not tok:
            continue
        if "-" in tok:
            a_str, b_str = tok.split("-", 1)
            a, b = int(a_str), int(b_str)
            if a < 1 or b < a:
                raise ValueError(f"Invalid chapter range '{tok}'")
            for idx in range(a, min(b, total_chapters) + 1):
                selected.add(idx)
        else:
            idx = int(tok)
            if idx < 1:
                raise ValueError(f"Invalid chapter number '{tok}'")
            if idx <= total_chapters:
                selected.add(idx)
    return selected


def pcm16_to_wav_bytes(pcm_bytes: bytes, sample_rate: int = DRAMA3_SAMPLE_RATE, channels: int = 1) -> bytes:
    """Wrap raw 16-bit little-endian PCM bytes in a standard RIFF WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Return duration in seconds of a standard WAV byte buffer."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        rate = wf.getframerate()
        return (wf.getnframes() / float(rate)) if rate > 0 else 0.0


def _decode_pcm16_samples(raw_frames: bytes) -> array:
    if len(raw_frames) % 2:
        raw_frames = raw_frames[:-1]
    samples = array("h")
    samples.frombytes(raw_frames)
    if sys.byteorder != "little":
        samples.byteswap()
    return samples


def _encode_pcm16_samples(samples: array) -> bytes:
    if sys.byteorder != "little":
        cp = array("h", samples)
        cp.byteswap()
        return cp.tobytes()
    return samples.tobytes()


def trim_trailing_silence_wav(
    wav_bytes: bytes,
    *,
    threshold_rms: int = 100,
    keep_tail_ms: int = 260,
    max_trim_s: float = 45.0,
) -> bytes:
    """Trim anomalous trailing silence from 16-bit PCM WAV output while keeping a natural room tail."""
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            nch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            comp = wf.getcomptype()
            nframes = wf.getnframes()
            if comp != "NONE" or sw != 2 or nch < 1 or sr <= 0 or nframes == 0:
                return wav_bytes
            raw = wf.readframes(nframes)
    except Exception:
        return wav_bytes

    samples = _decode_pcm16_samples(raw)
    total_frames = len(samples) // nch
    if total_frames == 0:
        return wav_bytes

    win_frames = max(1, sr // 50)  # 20ms window
    min_scan_frame = max(0, total_frames - int(max_trim_s * sr))
    last_active_frame = total_frames

    pos = total_frames
    found_voice = False
    while pos > min_scan_frame:
        start = max(min_scan_frame, pos - win_frames)
        span = samples[start * nch : pos * nch]
        if span:
            sq_sum = sum(int(s) * int(s) for s in span)
            rms = math.sqrt(sq_sum / len(span))
            if rms >= threshold_rms:
                last_active_frame = pos
                found_voice = True
                break
        pos = start

    if not found_voice:
        last_active_frame = max(min_scan_frame, min(total_frames, int(0.25 * sr)))

    keep_frames = min(total_frames, last_active_frame + int(sr * keep_tail_ms / 1000.0))
    keep_frames = max(min(total_frames, int(0.15 * sr)), keep_frames)
    if keep_frames >= total_frames:
        return wav_bytes

    trimmed_raw = _encode_pcm16_samples(samples[: keep_frames * nch])
    return pcm16_to_wav_bytes(trimmed_raw, sample_rate=sr, channels=nch)


def concat_wav_segments(
    segments: list[tuple[bytes, int]],
    *,
    default_sample_rate: int = DRAMA3_SAMPLE_RATE,
) -> bytes:
    """Concatenate (wav_bytes, pause_after_ms) pairs into a single mono/stereo PCM16 WAV."""
    if not segments:
        return pcm16_to_wav_bytes(b"", sample_rate=default_sample_rate, channels=1)

    target_sr: int | None = None
    target_ch: int | None = None
    out = io.BytesIO()
    wf_out: wave.Wave_write | None = None
    try:
        for idx, (wav_data, pause_ms) in enumerate(segments):
            with wave.open(io.BytesIO(wav_data), "rb") as wf_in:
                nch = wf_in.getnchannels()
                sw = wf_in.getsampwidth()
                sr = wf_in.getframerate()
                comp = wf_in.getcomptype()
                if comp != "NONE" or sw != 2:
                    raise ValueError(f"Segment {idx} is not 16-bit uncompressed PCM WAV")
                frames = wf_in.readframes(wf_in.getnframes())

            if wf_out is None:
                target_sr = sr
                target_ch = nch
                wf_out = wave.open(out, "wb")
                wf_out.setnchannels(target_ch)
                wf_out.setsampwidth(2)
                wf_out.setframerate(target_sr)
            elif sr != target_sr or nch != target_ch:
                raise ValueError(
                    f"WAV format mismatch in segment {idx}: got {sr}Hz/{nch}ch, expected {target_sr}Hz/{target_ch}ch"
                )

            wf_out.writeframes(frames)
            if pause_ms > 0 and idx < len(segments) - 1 and target_sr and target_ch:
                silence_frames = int(target_sr * pause_ms / 1000.0)
                if silence_frames > 0:
                    wf_out.writeframes(b"\x00" * (silence_frames * target_ch * 2))
    finally:
        if wf_out is not None:
            wf_out.close()
    return out.getvalue()
