"""Standalone ASGI server and security layer for TangoFlux (44.1 kHz stereo text-to-audio / SFX).

Imports cleanly with only the Python standard library so local unit tests can exercise all
authentication, CORS, Drive path containment, parameter validation, and WAV encoding logic.
"""
from __future__ import annotations

from array import array
import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import types
from typing import Any, Callable
import urllib.parse
import wave

from colab_asgi_common import (
    list_drive_audio_files,
    resolve_drive_path,
    sanitize_header_param,
    save_output_audio_to_drive,
    tail_service_log,
    verify_bearer_auth,
)

TANGOFLUX_SAMPLE_RATE = 44100
TANGOFLUX_CHANNELS = 2
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 30.0
DEFAULT_DURATION_SEC = 10.0
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE_SCALE = 4.5
MAX_WAVEFORMS_PER_REQUEST = 4
MAX_BATCH_ITEMS = 32
MAX_BODY_BYTES = 8 * 1024 * 1024

AUDIO_MIME_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
}
SUPPORTED_FORMATS = frozenset({*AUDIO_MIME_TYPES.keys(), "json", "b64_json"})

API_TOKEN = os.environ.get("CT_API_TOKEN", "")
SERVED_MODEL = os.environ.get("CT_TANGOFLUX_SERVED_MODEL", "declare-lab/TangoFlux")
MODEL_ALIAS = os.environ.get("CT_TANGOFLUX_ALIAS", "tangoflux")
MODEL_VARIANT = os.environ.get("CT_TANGOFLUX_VARIANT", "crpo")
MODEL_LOCAL_DIR = Path(os.environ.get("CT_TANGOFLUX_MODEL_DIR", "/content/tangoflux/models/tangoflux"))
TEXT_ENCODER_DIR = Path(os.environ.get("CT_TANGOFLUX_TEXT_ENCODER_DIR", "/content/tangoflux/models/flan-t5-large"))
DATA_DIR = Path(os.environ.get("CT_DATA_DIR", "/content/drive/MyDrive/colab-llm/tangoflux"))
DRIVE_ROOT = Path("/content/drive")
MYDRIVE_ROOT = DRIVE_ROOT / "MyDrive"
DRIVE_OUTPUT_DIR = DATA_DIR / "outputs"
LOG_FILE = Path("/content/ct/service.log")

CORS_HEADERS: list[tuple[bytes, bytes]] = [
    (b"access-control-allow-origin", b"*"),
    (b"access-control-allow-methods", b"GET, POST, OPTIONS"),
    (b"access-control-allow-headers", b"Authorization, Content-Type, X-API-Key"),
    (
        b"access-control-expose-headers",
        b"X-Saved-Drive-Path, X-Audio-Sample-Rate, X-Audio-Channels, "
        b"X-Audio-Duration, X-Audio-Seed, X-Generation-Seconds",
    ),
]


def _clamp_sample_i16(val: float) -> int:
    if not math.isfinite(val):
        return 0
    if val >= 1.0:
        return 32767
    if val <= -1.0:
        return -32768
    return int(round(val * 32767.0))


def float_stereo_to_wav_bytes(
    channels_first: Any,
    sample_rate: int = TANGOFLUX_SAMPLE_RATE,
    headroom_peak: float = 0.98,
) -> bytes:
    """Encode a [C, N] or [N] float waveform into 16-bit PCM WAV bytes with peak headroom protection.

    AutoencoderOobleck's final linear conv is unbounded and often produces peaks in [1.2, 2.5] on loud
    SFX prompts (explosions, jet engines, gunfire). Scaling down only when `peak > headroom_peak` matches
    IPython.display.Audio's peak normalization while keeping normal-level waveforms bit-accurate.
    """
    if hasattr(channels_first, "detach"):
        channels_first = channels_first.detach().cpu().float().numpy()
    try:
        import numpy as np  # type: ignore[import-not-found]

        arr = np.asarray(channels_first, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected 1D or 2D waveform, got shape {arr.shape}")
        nch = int(arr.shape[0])
        clean = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(clean))) if clean.size else 0.0
        if peak > headroom_peak:
            clean = clean * (headroom_peak / peak)
        clipped = np.clip(clean, -1.0, 1.0)
        interleaved = (clipped.T * 32767.0).round().astype("<i2")
        pcm_bytes = interleaved.tobytes(order="C")
    except ImportError:
        rows = list(channels_first)
        if not rows:
            nch = TANGOFLUX_CHANNELS
            pcm_bytes = b""
        elif isinstance(rows[0], (int, float)):
            nch = 1
            clean_1d = [0.0 if (math.isnan(float(v)) or math.isinf(float(v))) else float(v) for v in rows]
            peak = max((abs(v) for v in clean_1d), default=0.0)
            scale = (headroom_peak / peak) if peak > headroom_peak else 1.0
            samples = array("h", (_clamp_sample_i16(v * scale) for v in clean_1d))
            if sys.byteorder != "little":
                samples.byteswap()
            pcm_bytes = samples.tobytes()
        else:
            nch = len(rows)
            n_frames = len(rows[0])
            peak = 0.0
            for ch in range(nch):
                for v in rows[ch]:
                    fv = float(v)
                    if not (math.isnan(fv) or math.isinf(fv)) and abs(fv) > peak:
                        peak = abs(fv)
            scale = (headroom_peak / peak) if peak > headroom_peak else 1.0
            samples = array("h")
            for i in range(n_frames):
                for ch in range(nch):
                    fv = float(rows[ch][i])
                    if math.isnan(fv) or math.isinf(fv):
                        fv = 0.0
                    samples.append(_clamp_sample_i16(fv * scale))
            if sys.byteorder != "little":
                samples.byteswap()
            pcm_bytes = samples.tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(max(1, nch))
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def transcode_audio_bytes(wav_bytes: bytes, fmt: str, timeout: int = 60) -> bytes:
    """Convert 44.1 kHz stereo WAV bytes to compressed audio formats using ffmpeg."""
    target = (fmt or "wav").lower().strip()
    if target == "wav":
        return wav_bytes
    codec_map: dict[str, tuple[str, list[str]]] = {
        "mp3": ("mp3", ["-c:a", "libmp3lame", "-q:a", "2"]),
        "flac": ("flac", ["-c:a", "flac"]),
        "ogg": ("ogg", ["-c:a", "libvorbis", "-q:a", "5"]),
        "opus": ("ogg", ["-c:a", "libopus", "-b:a", "128k"]),
        "aac": ("adts", ["-c:a", "aac", "-b:a", "160k"]),
    }
    if target not in codec_map:
        raise ValueError(f"Unsupported output audio format '{fmt}'")
    muxer, codec_args = codec_map[target]
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-f", "wav", "-i", "pipe:0",
        "-vn", *codec_args, "-f", muxer, "pipe:1",
    ]
    res = subprocess.run(cmd, input=wav_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    if res.returncode != 0 or not res.stdout:
        err = res.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed encoding {target}: {err}")
    return res.stdout


def _validate_save_spec(save_spec: Any) -> str | None:
    if save_spec is None or save_spec is False or save_spec == "":
        return None
    spec_str = "auto" if save_spec is True else str(save_spec).strip()
    if not spec_str:
        return None
    if spec_str not in ("auto", "true", "True", "1"):
        resolve_drive_path(
            spec_str,
            data_dir=DATA_DIR,
            default_subdir="outputs",
            mydrive_root=MYDRIVE_ROOT,
            drive_root=DRIVE_ROOT,
        )
    return spec_str


def _parse_generate_params(data: dict[str, Any]) -> dict[str, Any]:
    raw_prompt = data.get("prompt")
    if raw_prompt is None:
        raw_prompt = data.get("input")
    if not isinstance(raw_prompt, str) or not raw_prompt.strip():
        raise ValueError("Provide a non-empty string in `prompt` (or `input`).")
    prompt = raw_prompt.strip()
    if len(prompt) > 2000:
        raise ValueError("`prompt` must be at most 2000 characters.")

    raw_dur = data.get("duration", DEFAULT_DURATION_SEC)
    try:
        duration = float(raw_dur)
    except (TypeError, ValueError):
        raise ValueError("`duration` must be a number between 1.0 and 30.0 seconds.") from None
    if not math.isfinite(duration) or duration < MIN_DURATION_SEC or duration > MAX_DURATION_SEC:
        raise ValueError(f"`duration` must be between {MIN_DURATION_SEC:g} and {MAX_DURATION_SEC:g} seconds (got {raw_dur!r}).")

    raw_steps = data.get("steps", data.get("num_inference_steps", DEFAULT_STEPS))
    try:
        steps = int(raw_steps)
    except (TypeError, ValueError):
        raise ValueError("`steps` must be an integer between 1 and 200.") from None
    if steps < 1 or steps > 200:
        raise ValueError(f"`steps` must be between 1 and 200 (got {steps}).")

    raw_cfg = data.get("guidance_scale", data.get("cfg_scale", DEFAULT_GUIDANCE_SCALE))
    try:
        guidance_scale = float(raw_cfg)
    except (TypeError, ValueError):
        raise ValueError("`guidance_scale` must be a number between 1.0 and 20.0.") from None
    if not math.isfinite(guidance_scale) or guidance_scale < 1.0 or guidance_scale > 20.0:
        raise ValueError(f"`guidance_scale` must be between 1.0 and 20.0 (got {raw_cfg!r}).")

    raw_n = data.get("num_waveforms", data.get("n", 1))
    try:
        num_waveforms = int(raw_n)
    except (TypeError, ValueError):
        raise ValueError(f"`num_waveforms` must be an integer between 1 and {MAX_WAVEFORMS_PER_REQUEST}.") from None
    if num_waveforms < 1 or num_waveforms > MAX_WAVEFORMS_PER_REQUEST:
        raise ValueError(f"`num_waveforms` must be between 1 and {MAX_WAVEFORMS_PER_REQUEST} (got {num_waveforms}).")

    raw_seed = data.get("seed")
    if raw_seed is None or raw_seed == "" or (isinstance(raw_seed, int) and raw_seed < 0):
        seed = secrets.randbelow(2**31 - 1)
    else:
        try:
            seed = int(raw_seed)
        except (TypeError, ValueError):
            raise ValueError("`seed` must be a non-negative integer.") from None
        if seed < 0 or seed > 2**32 - 1:
            raise ValueError("`seed` must be between 0 and 4294967295.")

    raw_fmt = str(data.get("response_format") or data.get("format") or "wav").strip().lower()
    if raw_fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported `response_format` '{raw_fmt}'; choose from {sorted(SUPPORTED_FORMATS)}.")
    return_json = bool(data.get("return_json")) or raw_fmt in ("json", "b64_json") or num_waveforms > 1
    audio_fmt = "wav" if raw_fmt in ("json", "b64_json") else raw_fmt

    save_spec = _validate_save_spec(data.get("save_to_drive"))
    return {
        "prompt": prompt,
        "duration": duration,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "num_waveforms": num_waveforms,
        "seed": seed,
        "audio_format": audio_fmt,
        "return_json": return_json,
        "save_to_drive": save_spec,
    }


def _prepare_generate_request(raw_body: bytes) -> dict[str, Any]:
    if not raw_body:
        raise ValueError("Request JSON body is required.")
    try:
        data = json.loads(raw_body.decode("utf-8"))
    except Exception as err:
        raise ValueError(f"Invalid JSON body: {err}") from None
    if not isinstance(data, dict):
        raise ValueError("Request JSON body must be an object.")
    return _parse_generate_params(data)


def _prepare_batch_request(raw_body: bytes) -> dict[str, Any]:
    if not raw_body:
        raise ValueError("Request JSON body is required.")
    try:
        data = json.loads(raw_body.decode("utf-8"))
    except Exception as err:
        raise ValueError(f"Invalid JSON body: {err}") from None
    if not isinstance(data, dict):
        raise ValueError("Batch request JSON body must be an object.")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("Batch request must provide a non-empty `items` list.")
    if len(items) > MAX_BATCH_ITEMS:
        raise ValueError(f"`items` exceeds maximum batch size of {MAX_BATCH_ITEMS}.")

    default_fmt = str(data.get("response_format") or data.get("format") or "wav").strip().lower()
    if default_fmt in ("json", "b64_json"):
        default_fmt = "wav"
    if default_fmt not in AUDIO_MIME_TYPES:
        raise ValueError(f"Unsupported batch `response_format` '{default_fmt}'.")

    top_save = _validate_save_spec(data.get("save_to_drive"))
    parsed_items: list[dict[str, Any]] = []
    for idx, entry in enumerate(items):
        if isinstance(entry, str):
            entry = {"prompt": entry}
        if not isinstance(entry, dict):
            raise ValueError(f"Batch item {idx} must be an object or prompt string.")
        merged = {
            "duration": data.get("duration", DEFAULT_DURATION_SEC),
            "steps": data.get("steps", DEFAULT_STEPS),
            "guidance_scale": data.get("guidance_scale", DEFAULT_GUIDANCE_SCALE),
            "response_format": entry.get("response_format") or entry.get("format") or default_fmt,
            **entry,
            "num_waveforms": 1,
        }
        if "save_to_drive" not in merged and top_save:
            merged["save_to_drive"] = top_save
        spec = _parse_generate_params(merged)
        spec["name"] = str(entry.get("name") or f"sfx-{idx + 1:03d}").strip()[:80]
        parsed_items.append(spec)
    return {"items": parsed_items, "response_format": default_fmt}


class TangoFluxEngine:
    """GPU inference wrapper around declare-lab/TangoFlux with local SSD loading and multi-sample decoding."""

    def __init__(self, model_dir: Path, text_encoder_dir: Path | None = None, device: str | None = None):
        for _tf_flag, _tf_val in (
            ("USE_TF", "0"),
            ("USE_FLAX", "0"),
            ("TRANSFORMERS_NO_TF", "1"),
            ("TRANSFORMERS_NO_FLAX", "1"),
        ):
            os.environ.setdefault(_tf_flag, _tf_val)

        # Upstream tangoflux/model.py imports unused `datasets` at module level; stub if absent so pyarrow isn't needed.
        try:
            import datasets  # noqa: F401
        except ImportError:
            import importlib.machinery

            stub = types.ModuleType("datasets")
            stub.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
            stub.load_dataset = None  # type: ignore[attr-defined]
            stub.Audio = None  # type: ignore[attr-defined]
            sys.modules["datasets"] = stub

        import torch  # type: ignore[import-not-found]
        from diffusers import AutoencoderOobleck  # type: ignore[import-not-found]
        from safetensors.torch import load_file  # type: ignore[import-not-found]
        from tangoflux.model import TangoFlux  # type: ignore[import-not-found]

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cuda_amp = bool(self.device.startswith("cuda") and torch.cuda.is_available())
        if self.use_cuda_amp:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.gpu_name = torch.cuda.get_device_name(0)
            bf16_ok = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
            self.amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        else:
            self.gpu_name = "cpu"
            self.amp_dtype = torch.float32

        cfg_path = model_dir / "config.json"
        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        enc_arg = str(text_encoder_dir) if (text_encoder_dir and text_encoder_dir.is_dir()) else None

        self.vae = AutoencoderOobleck()
        self.vae.load_state_dict(load_file(str(model_dir / "vae.safetensors")))
        self.model = TangoFlux(config, text_encoder_dir=enc_arg)
        self.model.load_state_dict(load_file(str(model_dir / "tangoflux.safetensors")), strict=False)

        # Keep Flan-T5-Large in full FP32 (avoids FP16 overflow NaNs) and fix upstream bug where
        # guidance_scale <= 1.0 passes unsupported `num_samples_per_prompt` to `encode_text`.
        orig_encode_text = self.model.encode_text

        def _encode_text_compat(prompt: Any, num_samples_per_prompt: int = 1):
            if self.use_cuda_amp:
                with torch.amp.autocast("cuda", enabled=False):
                    enc, mask = orig_encode_text(prompt)
            else:
                enc, mask = orig_encode_text(prompt)
            if num_samples_per_prompt > 1:
                enc = enc.repeat_interleave(num_samples_per_prompt, 0)
                mask = mask.repeat_interleave(num_samples_per_prompt, 0)
            return enc, mask

        self.model.encode_text = _encode_text_compat  # type: ignore[method-assign]

        self.vae.to(self.device).eval()
        self.model.to(self.device).eval()
        self.sample_rate = int(getattr(self.vae.config, "sampling_rate", TANGOFLUX_SAMPLE_RATE))

    def generate_waveforms(
        self,
        *,
        prompt: str,
        duration: float,
        steps: int,
        guidance_scale: float,
        seed: int,
        num_waveforms: int = 1,
    ) -> list[Any]:
        torch = self.torch
        with torch.no_grad():
            if self.use_cuda_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    latents = self.model.inference_flow(
                        prompt,
                        duration=float(duration),
                        num_inference_steps=int(steps),
                        guidance_scale=float(guidance_scale),
                        seed=int(seed),
                        disable_progress=True,
                        num_samples_per_prompt=int(num_waveforms),
                    )
                latents = latents.float()
            else:
                latents = self.model.inference_flow(
                    prompt,
                    duration=float(duration),
                    num_inference_steps=int(steps),
                    guidance_scale=float(guidance_scale),
                    seed=int(seed),
                    disable_progress=True,
                    num_samples_per_prompt=int(num_waveforms),
                )
            decoded = self.vae.decode(latents.transpose(2, 1)).sample.float().cpu()
        waveform_end = max(1, int(round(float(duration) * self.sample_rate)))
        return [decoded[i, :, :waveform_end] for i in range(int(decoded.shape[0]))]


def _headers_to_dict(raw_headers: list[tuple[bytes, bytes]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in raw_headers:
        ks = k.decode("latin1").lower()
        vs = v.decode("latin1")
        if ks not in out:
            out[ks] = vs
    return out


async def _read_body(receive: Any, max_bytes: int = MAX_BODY_BYTES) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        msg = await receive()
        if msg["type"] == "http.disconnect":
            break
        piece = msg.get("body", b"")
        if piece:
            total += len(piece)
            if total > max_bytes:
                raise ValueError(f"Request body exceeds {max_bytes // (1024 * 1024)} MiB limit")
            chunks.append(piece)
        if not msg.get("more_body", False):
            break
    return b"".join(chunks)


async def _send_json(
    send: Any,
    status: int,
    payload: Any,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    hdrs = [
        *CORS_HEADERS,
        (b"content-type", b"application/json; charset=utf-8"),
        (b"content-length", str(len(raw)).encode("ascii")),
        *(extra_headers or []),
    ]
    await send({"type": "http.response.start", "status": status, "headers": hdrs})
    await send({"type": "http.response.body", "body": raw, "more_body": False})


async def _send_bytes(
    send: Any,
    status: int,
    body: bytes,
    content_type: str,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
) -> None:
    hdrs = [
        *CORS_HEADERS,
        (b"content-type", content_type.encode("latin1")),
        (b"content-length", str(len(body)).encode("ascii")),
        *(extra_headers or []),
    ]
    await send({"type": "http.response.start", "status": status, "headers": hdrs})
    await send({"type": "http.response.body", "body": body, "more_body": False})


class TangoFluxApp:
    """Authenticated ASGI application serving TangoFlux sound-effect generation and Drive I/O."""

    def __init__(self, generate_fn: Callable[..., list[Any]], *, gpu_name: str = "cuda", sample_rate: int = TANGOFLUX_SAMPLE_RATE):
        self._generate_fn = generate_fn
        self._gpu_name = gpu_name
        self._sample_rate = sample_rate
        self._gpu_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _synthesize_encoded(self, spec: dict[str, Any], prefix: str = "tangoflux") -> dict[str, Any]:
        t0 = time.monotonic()
        waves = self._generate_fn(
            prompt=spec["prompt"],
            duration=spec["duration"],
            steps=spec["steps"],
            guidance_scale=spec["guidance_scale"],
            seed=spec["seed"],
            num_waveforms=spec["num_waveforms"],
        )
        elapsed = time.monotonic() - t0
        fmt = spec["audio_format"]
        save_spec = spec.get("save_to_drive")

        items: list[dict[str, Any]] = []
        for idx, w in enumerate(waves):
            wav_bytes = float_stereo_to_wav_bytes(w, sample_rate=self._sample_rate)
            encoded = transcode_audio_bytes(wav_bytes, fmt) if fmt != "wav" else wav_bytes
            saved_path: str | None = None
            if save_spec:
                item_save = save_spec
                if len(waves) > 1 and save_spec not in ("auto", "true", "True", "1") and not save_spec.endswith("/"):
                    p_obj = Path(save_spec)
                    stem_s = f"{p_obj.stem}_{idx + 1:02d}{p_obj.suffix}"
                    item_save = str(p_obj.with_name(stem_s))
                saved_path = save_output_audio_to_drive(
                    encoded,
                    item_save,
                    data_dir=DATA_DIR,
                    output_dir=DRIVE_OUTPUT_DIR,
                    prefix=f"{prefix}-{idx + 1:02d}" if len(waves) > 1 else prefix,
                    fmt=fmt,
                    mydrive_root=MYDRIVE_ROOT,
                    drive_root=DRIVE_ROOT,
                )
            items.append({
                "index": idx,
                "audio_bytes": encoded,
                "format": fmt,
                "saved_drive_path": saved_path,
            })
        return {
            "elapsed_sec": round(elapsed, 3),
            "items": items,
        }

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return

        method = scope.get("method", "GET").upper()
        path = scope.get("path", "/")
        if method == "OPTIONS":
            await send({"type": "http.response.start", "status": 204, "headers": list(CORS_HEADERS)})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        headers = _headers_to_dict(scope.get("headers") or [])
        if not verify_bearer_auth(headers, API_TOKEN):
            await _send_json(
                send,
                401,
                {"error": {"message": "Missing or invalid Bearer API token", "type": "authentication_error"}},
                extra_headers=[(b"www-authenticate", b'Bearer realm="tangoflux"')],
            )
            return

        qs = urllib.parse.parse_qs((scope.get("query_string") or b"").decode("latin1"))

        if method == "GET" and path == "/health":
            await _send_json(send, 200, {
                "status": "ok",
                "ready": True,
                "model": SERVED_MODEL,
                "alias": MODEL_ALIAS,
                "sample_rate": self._sample_rate,
                "channels": TANGOFLUX_CHANNELS,
            })
            return

        if method == "GET" and path == "/v1/models":
            created = int(time.time())
            models = [
                {"id": MODEL_ALIAS, "object": "model", "created": created, "owned_by": "declare-lab"},
            ]
            if SERVED_MODEL != MODEL_ALIAS:
                models.append({"id": SERVED_MODEL, "object": "model", "created": created, "owned_by": "declare-lab"})
            await _send_json(send, 200, {"object": "list", "data": models})
            return

        if method == "GET" and path == "/api/ct/info":
            await _send_json(send, 200, {
                "service": "tangoflux",
                "model": SERVED_MODEL,
                "alias": MODEL_ALIAS,
                "variant": MODEL_VARIANT,
                "gpu_name": self._gpu_name,
                "sample_rate": self._sample_rate,
                "channels": TANGOFLUX_CHANNELS,
                "duration_range_sec": [MIN_DURATION_SEC, MAX_DURATION_SEC],
                "default_duration_sec": DEFAULT_DURATION_SEC,
                "default_steps": DEFAULT_STEPS,
                "default_guidance_scale": DEFAULT_GUIDANCE_SCALE,
                "data_dir": str(DATA_DIR),
                "drive_output_dir": str(DRIVE_OUTPUT_DIR),
            })
            return

        if method == "GET" and path == "/api/ct/logs":
            n_lines = int((qs.get("lines") or ["120"])[0])
            await _send_json(send, 200, tail_service_log(LOG_FILE, n_lines))
            return

        if method == "GET" and path == "/api/ct/drive-list":
            spec = (qs.get("path") or [""])[0]
            recursive = (qs.get("recursive") or ["1"])[0] not in ("0", "false", "False")
            try:
                files = list_drive_audio_files(
                    spec,
                    data_dir=DATA_DIR,
                    default_subdir="outputs",
                    recursive=recursive,
                    mydrive_root=MYDRIVE_ROOT,
                    drive_root=DRIVE_ROOT,
                )
                await _send_json(send, 200, {"count": len(files), "files": files})
            except ValueError as err:
                await _send_json(send, 400, {"error": {"message": str(err), "type": "invalid_request_error"}})
            except Exception as err:
                await _send_json(send, 500, {"error": {"message": str(err), "type": "server_error"}})
            return

        if method == "POST" and path in ("/api/ct/generate", "/v1/audio/generations", "/v1/audio/speech"):
            try:
                raw_body = await _read_body(receive)
                spec = _prepare_generate_request(raw_body)
            except ValueError as err:
                await _send_json(send, 400, {"error": {"message": str(err), "type": "invalid_request_error"}})
                return

            try:
                loop = asyncio.get_running_loop()
                async with self._gpu_lock:
                    gen_out = await loop.run_in_executor(self._executor, self._synthesize_encoded, spec, "tangoflux")
            except ValueError as err:
                await _send_json(send, 400, {"error": {"message": str(err), "type": "invalid_request_error"}})
                return
            except Exception as err:
                await _send_json(send, 500, {"error": {"message": str(err), "type": "generation_error"}})
                return

            out_items = gen_out["items"]
            if not spec["return_json"] and len(out_items) == 1:
                item0 = out_items[0]
                resp_hdrs: list[tuple[bytes, bytes]] = [
                    (b"x-audio-sample-rate", str(self._sample_rate).encode("ascii")),
                    (b"x-audio-channels", str(TANGOFLUX_CHANNELS).encode("ascii")),
                    (b"x-audio-duration", f"{spec['duration']:.2f}".encode("ascii")),
                    (b"x-audio-seed", str(spec["seed"]).encode("ascii")),
                    (b"x-generation-seconds", str(gen_out["elapsed_sec"]).encode("ascii")),
                ]
                if item0.get("saved_drive_path"):
                    safe_dp = sanitize_header_param(str(item0["saved_drive_path"]))
                    resp_hdrs.append((b"x-saved-drive-path", safe_dp.encode("latin1", errors="replace")))
                await _send_bytes(
                    send,
                    200,
                    item0["audio_bytes"],
                    AUDIO_MIME_TYPES[item0["format"]],
                    extra_headers=resp_hdrs,
                )
                return

            payload = {
                "id": f"sfx-{secrets.token_hex(6)}",
                "model": SERVED_MODEL,
                "prompt": spec["prompt"],
                "duration": spec["duration"],
                "steps": spec["steps"],
                "guidance_scale": spec["guidance_scale"],
                "seed": spec["seed"],
                "sample_rate": self._sample_rate,
                "channels": TANGOFLUX_CHANNELS,
                "elapsed_sec": gen_out["elapsed_sec"],
                "data": [
                    {
                        "index": it["index"],
                        "format": it["format"],
                        "media_type": AUDIO_MIME_TYPES[it["format"]],
                        "b64_json": base64.b64encode(it["audio_bytes"]).decode("ascii"),
                        "saved_drive_path": it.get("saved_drive_path"),
                    }
                    for it in out_items
                ],
            }
            await _send_json(send, 200, payload)
            return

        if method == "POST" and path == "/api/ct/batch":
            try:
                raw_body = await _read_body(receive)
                batch_spec = _prepare_batch_request(raw_body)
            except ValueError as err:
                await _send_json(send, 400, {"error": {"message": str(err), "type": "invalid_request_error"}})
                return

            results: list[dict[str, Any]] = []
            t_batch = time.monotonic()
            loop = asyncio.get_running_loop()
            async with self._gpu_lock:
                for idx, item_spec in enumerate(batch_spec["items"]):
                    safe_prefix = "".join(
                        ch if (ch.isalnum() or ch in "-_") else "-"
                        for ch in item_spec.get("name", f"sfx-{idx + 1:03d}")
                    ).strip("-") or f"sfx-{idx + 1:03d}"
                    try:
                        gen_out = await loop.run_in_executor(
                            self._executor, self._synthesize_encoded, item_spec, safe_prefix[:48]
                        )
                        it0 = gen_out["items"][0]
                        results.append({
                            "index": idx,
                            "name": item_spec["name"],
                            "status": "success",
                            "prompt": item_spec["prompt"],
                            "duration": item_spec["duration"],
                            "steps": item_spec["steps"],
                            "guidance_scale": item_spec["guidance_scale"],
                            "seed": item_spec["seed"],
                            "format": it0["format"],
                            "media_type": AUDIO_MIME_TYPES[it0["format"]],
                            "elapsed_sec": gen_out["elapsed_sec"],
                            "audio_data": base64.b64encode(it0["audio_bytes"]).decode("ascii"),
                            "saved_drive_path": it0.get("saved_drive_path"),
                        })
                    except Exception as err:
                        results.append({
                            "index": idx,
                            "name": item_spec["name"],
                            "status": "error",
                            "prompt": item_spec["prompt"],
                            "error": str(err),
                        })

            await _send_json(send, 200, {
                "id": f"batch-{secrets.token_hex(6)}",
                "model": SERVED_MODEL,
                "sample_rate": self._sample_rate,
                "channels": TANGOFLUX_CHANNELS,
                "elapsed_sec": round(time.monotonic() - t_batch, 3),
                "results": results,
            })
            return

        await _send_json(send, 404, {"error": {"message": f"Unknown endpoint {method} {path}", "type": "not_found"}})


def run_tangoflux_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn  # type: ignore[import-not-found]

    DRIVE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[tangoflux] Loading {SERVED_MODEL} from {MODEL_LOCAL_DIR} ...", flush=True)
    t0 = time.monotonic()
    engine = TangoFluxEngine(model_dir=MODEL_LOCAL_DIR, text_encoder_dir=TEXT_ENCODER_DIR)
    print(
        f"[tangoflux] Model loaded on {engine.device} ({engine.gpu_name}) in {time.monotonic() - t0:.1f}s "
        f"(sample_rate={engine.sample_rate} Hz, stereo)",
        flush=True,
    )
    app = TangoFluxApp(engine.generate_waveforms, gpu_name=engine.gpu_name, sample_rate=engine.sample_rate)
    uvicorn.run(app, host=host, port=int(port), log_level="info")


if __name__ == "__main__":
    run_tangoflux_server(
        host="127.0.0.1",
        port=int(os.environ.get("CT_SERVICE_PORT", "8000")),
    )
