"""Shared security, Drive path containment, and audio helpers for Colab ASGI middlewares."""
from __future__ import annotations

import base64
import glob as glob_mod
import os
import re
import secrets
import subprocess
import time
from pathlib import Path
from typing import Any

AUDIO_EXTS = (
    ".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac", ".opus", ".webm", ".mp4", ".mkv", ".wma", ".mpga", ".mpeg",
)
DEFAULT_SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "aux-venv", "custom_nodes",
    "models", ".runtimes", "Takeout", ".Trash", ".ipynb_checkpoints",
    "flashinfer-cache", "triton-cache", "vllm-cache", "cache-archives", "kernel-archives",
}
VOICE_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def validate_voice_name(name: str) -> str:
    """Validate and normalize a custom voice identifier against path traversal."""
    vname = (name or "").strip().lower()
    if not vname or ".." in vname or not VOICE_NAME_RE.fullmatch(vname):
        raise ValueError(f"Invalid voice name '{name}': use 1-64 lowercase [a-z0-9._-] characters without '..'")
    return vname


def _get_header(headers: Any, key: str) -> str:
    val = headers.get(key)
    if val:
        return str(val)
    if isinstance(headers, dict):
        low = key.lower()
        for k, v in headers.items():
            if str(k).lower() == low and v:
                return str(v)
    return ""


def verify_bearer_auth(headers: Any, api_token: str) -> bool:
    """Fail-closed constant-time Bearer / X-API-Key authentication check."""
    if not api_token:
        return False
    expected_bearer = f"Bearer {api_token}"
    auth = _get_header(headers, "authorization")
    if auth and secrets.compare_digest(auth.encode("utf-8"), expected_bearer.encode("utf-8")):
        return True
    xkey = _get_header(headers, "x-api-key")
    if xkey and secrets.compare_digest(xkey.encode("utf-8"), api_token.encode("utf-8")):
        return True
    return False


def sanitize_header_param(value: str) -> str:
    """Strip CR/LF/NUL and escape quotes/backslashes for Content-Disposition parameters."""
    cleaned = (value or "").replace("\r", "").replace("\n", "").replace("\x00", "")
    return cleaned.replace("\\", "\\\\").replace('"', '\\"')


def build_multipart_form(
    fields: list[tuple[str, str]],
    files: list[tuple[str, str, str, bytes]] | None = None,
) -> tuple[str, bytes]:
    """Build a safe multipart/form-data body returning (content_type, body_bytes)."""
    boundary = f"----ctform{secrets.token_hex(16)}"
    parts: list[bytes] = []
    for key, val in fields:
        safe_key = sanitize_header_param(str(key))
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{safe_key}"\r\n\r\n{val}\r\n'.encode("utf-8")
        )
    for field_name, filename, content_type, data in files or []:
        safe_field = sanitize_header_param(str(field_name))
        safe_filename = sanitize_header_param(str(filename))
        safe_ct = (content_type or "application/octet-stream").replace("\r", "").replace("\n", "").replace("\x00", "")
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{safe_field}"; filename="{safe_filename}"\r\n'
                f"Content-Type: {safe_ct}\r\n\r\n"
            ).encode("utf-8")
            + data
            + b"\r\n"
        )
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


def allowed_drive_roots(
    data_dir: Path,
    mydrive_root: Path = Path("/content/drive/MyDrive"),
) -> tuple[Path, ...]:
    """Return resolved root directories that Drive/data operations may access."""
    roots: list[Path] = []
    if mydrive_root.is_dir():
        roots.append(mydrive_root.resolve())
    roots.append(data_dir.resolve())
    return tuple(dict.fromkeys(roots))


def _ensure_within_roots(candidate: Path, roots: tuple[Path, ...], raw_spec: str) -> Path:
    resolved = candidate.resolve()
    if not any(resolved.is_relative_to(root) for root in roots):
        raise ValueError(f"Path '{raw_spec}' escapes allowed Drive/data directory")
    return resolved


def resolve_drive_path(
    raw_spec: str,
    *,
    data_dir: Path,
    default_subdir: str | None = None,
    mydrive_root: Path = Path("/content/drive/MyDrive"),
    drive_root: Path = Path("/content/drive"),
) -> Path:
    """Resolve a Drive or service-data path strictly inside allowed roots."""
    if "\x00" in (raw_spec or ""):
        raise ValueError("Invalid NUL byte in path")
    roots = allowed_drive_roots(data_dir, mydrive_root)
    spec = (raw_spec or "").strip()

    is_file_scheme = spec.startswith("file://")
    for prefix in ("drive://", "gdrive://", "file://"):
        if spec.startswith(prefix):
            spec = spec[len(prefix):]
            break

    if is_file_scheme:
        candidate = Path("/" + spec.lstrip("/"))
        return _ensure_within_roots(candidate, roots, raw_spec)

    had_leading_slash = spec.startswith("/")
    clean = spec.lstrip("/")
    if not clean:
        if default_subdir and (data_dir / default_subdir).is_dir():
            return _ensure_within_roots(data_dir / default_subdir, roots, raw_spec)
        return roots[0]

    if had_leading_slash and not clean.startswith(("content/drive/", "drive/", "MyDrive/", "MyDrive")):
        # Allow explicit absolute path ONLY if it already lies inside an allowed root (e.g. DATA_DIR in --no-drive mode)
        return _ensure_within_roots(Path("/" + clean), roots, raw_spec)

    candidates: list[Path] = []
    if clean.startswith("content/drive/"):
        candidates.append(Path("/") / clean)
    elif clean.startswith("drive/"):
        candidates.append(drive_root.parent / clean)
    elif clean.startswith("MyDrive/"):
        candidates.append(drive_root / clean)
    elif clean == "MyDrive":
        candidates.append(mydrive_root)
    elif clean.startswith("colab-llm/") and mydrive_root.is_dir():
        candidates.append(mydrive_root / clean)
        candidates.append(data_dir / clean)
    else:
        first_seg = clean.split("/", 1)[0]
        already_prefixed = bool(
            default_subdir
            and (
                clean == default_subdir
                or clean.startswith(f"{default_subdir}/")
                or (first_seg != clean and (data_dir / first_seg).is_dir())
            )
        )
        if default_subdir and not already_prefixed:
            candidates.append(data_dir / default_subdir / clean)
        candidates.append(data_dir / clean)
        if default_subdir and already_prefixed:
            candidates.append(data_dir / default_subdir / clean)
        if mydrive_root.is_dir():
            candidates.append(mydrive_root / clean)

    checked: list[Path] = []
    for cand in candidates:
        resolved = _ensure_within_roots(cand, roots, raw_spec)
        checked.append(resolved)
        if resolved.exists():
            return resolved
    if not checked:
        raise ValueError(f"Path '{raw_spec}' could not be resolved inside allowed Drive/data directory")
    return checked[0]


def rel_to_mydrive(p: Path, mydrive_root: Path = Path("/content/drive/MyDrive")) -> str:
    try:
        return str(p.resolve().relative_to(mydrive_root.resolve()))
    except Exception:
        return str(p)


def list_drive_audio_files(
    spec: str,
    *,
    data_dir: Path,
    default_subdir: str | None = None,
    recursive: bool = True,
    mydrive_root: Path = Path("/content/drive/MyDrive"),
    drive_root: Path = Path("/content/drive"),
    skip_dirs: set[str] | None = None,
) -> list[dict[str, Any]]:
    """List audio files matching a Drive path or glob strictly inside allowed roots."""
    if "\x00" in (spec or ""):
        raise ValueError("Invalid NUL byte in path")
    roots = allowed_drive_roots(data_dir, mydrive_root)
    ignored = skip_dirs if skip_dirs is not None else DEFAULT_SKIP_DIRS
    raw = (spec or "").strip()
    is_file_scheme = raw.startswith("file://")
    for prefix in ("drive://", "gdrive://", "file://"):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break

    has_glob = any(ch in raw for ch in ("*", "?", "["))
    matched_paths: list[Path] = []

    if has_glob:
        if is_file_scheme:
            bases = [Path("/" + raw.lstrip("/"))]
        else:
            clean = raw.lstrip("/")
            if raw.startswith("/") and not clean.startswith(("content/drive/", "drive/", "MyDrive/")):
                bases = [Path("/" + clean)]
            elif clean.startswith("content/drive/"):
                bases = [Path("/") / clean]
            elif clean.startswith("MyDrive/"):
                bases = [drive_root / clean]
            else:
                bases = []
                if default_subdir:
                    bases.append(data_dir / default_subdir / clean)
                bases.append(data_dir / clean)
                if mydrive_root.is_dir():
                    bases.append(mydrive_root / clean)

        # Ensure static prefix before wildcard does not escape allowed roots
        for pattern_path in bases:
            prefix_parts: list[str] = []
            for part in pattern_path.parts:
                if any(ch in part for ch in ("*", "?", "[")):
                    break
                prefix_parts.append(part)
            if prefix_parts:
                _ensure_within_roots(Path(*prefix_parts), roots, spec)

        seen: set[str] = set()
        for pattern_path in bases:
            for hit in sorted(glob_mod.glob(str(pattern_path), recursive=True)):
                hp = Path(hit).resolve()
                if not any(hp.is_relative_to(r) for r in roots):
                    continue
                if hp.is_file() and hp.suffix.lower() in AUDIO_EXTS and str(hp) not in seen:
                    seen.add(str(hp))
                    matched_paths.append(hp)
    else:
        target = resolve_drive_path(
            spec,
            data_dir=data_dir,
            default_subdir=default_subdir,
            mydrive_root=mydrive_root,
            drive_root=drive_root,
        )
        if target.is_file():
            if target.suffix.lower() in AUDIO_EXTS:
                matched_paths.append(target)
        elif target.is_dir():
            if not recursive:
                for hp in sorted(target.iterdir()):
                    resolved = hp.resolve()
                    if any(resolved.is_relative_to(r) for r in roots) and resolved.is_file() and resolved.suffix.lower() in AUDIO_EXTS:
                        matched_paths.append(resolved)
            else:
                for root_dir, dirnames, filenames in os.walk(target):
                    dirnames[:] = sorted(d for d in dirnames if d not in ignored and not d.startswith("."))
                    for fn in sorted(filenames):
                        hp = (Path(root_dir) / fn).resolve()
                        if any(hp.is_relative_to(r) for r in roots) and hp.is_file() and hp.suffix.lower() in AUDIO_EXTS:
                            matched_paths.append(hp)

    items: list[dict[str, Any]] = []
    for p in matched_paths:
        st = p.stat()
        items.append({
            "drive_path": str(p),
            "rel_path": rel_to_mydrive(p, mydrive_root),
            "name": p.name,
            "size_bytes": st.st_size,
            "size_mb": round(st.st_size / 1e6, 2),
            "mtime": int(st.st_mtime),
        })
    return items


def audio_path_to_wav_data_url(
    audio_path: Path,
    max_duration_sec: float = 24.0,
    sample_rate: int = 24000,
    timeout: int = 60,
) -> str:
    """Decode a Drive audio file to a mono PCM16 WAV data URL with bounded ffmpeg timeout."""
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-t", f"{max_duration_sec:.1f}",
        "-i", str(audio_path),
        "-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", "-f", "wav", "pipe:1",
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except OSError as os_err:
        if audio_path.suffix.lower() == ".wav":
            b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            return f"data:audio/wav;base64,{b64}"
        raise RuntimeError(f"ffmpeg failed decoding {audio_path.name}: {os_err}") from os_err
    if res.returncode != 0 or not res.stdout:
        if audio_path.suffix.lower() == ".wav":
            b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
            return f"data:audio/wav;base64,{b64}"
        err = res.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed decoding {audio_path.name}: {err}")
    b64 = base64.b64encode(res.stdout).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def save_output_audio_to_drive(
    audio_bytes: bytes,
    spec: str,
    *,
    data_dir: Path,
    output_dir: Path,
    prefix: str,
    fmt: str = "wav",
    mydrive_root: Path = Path("/content/drive/MyDrive"),
    drive_root: Path = Path("/content/drive"),
) -> str:
    """Persist synthesized audio bytes safely inside Drive or service output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = "." + re.sub(r"[^a-z0-9]", "", (fmt or "wav").lower())
    if not ext or ext == ".":
        ext = ".wav"
    if spec and spec not in ("auto", "true", "True", "1"):
        dest = resolve_drive_path(
            spec,
            data_dir=data_dir,
            mydrive_root=mydrive_root,
            drive_root=drive_root,
        )
        if dest.is_dir() or spec.endswith("/"):
            dest.mkdir(parents=True, exist_ok=True)
            dest = dest / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}{ext}"
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.suffix:
                dest = dest.with_suffix(ext)
    else:
        dest = output_dir / f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}{ext}"
    dest.write_bytes(audio_bytes)
    return str(dest)


def tail_service_log(
    log_file: Path = Path("/content/ct/service.log"),
    lines: int = 120,
) -> dict[str, Any]:
    n_lines = max(1, min(int(lines), 2000))
    entries = log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-n_lines:] if log_file.is_file() else []
    return {"lines": len(entries), "log": "\n".join(entries)}
