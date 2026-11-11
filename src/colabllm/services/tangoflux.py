#!/usr/bin/env python3
"""TangoFlux (44.1 kHz stereo text-to-audio & sound effect generation on Colab T4/L4/G4).

Serves `declare-lab/TangoFlux` (CRPO preference-optimized 515M FluxTransformer + Flan-T5-Large +
AutoencoderOobleck stereo VAE, or `declare-lab/TangoFlux-base` via `--variant base`):
  - Generates 1–30s 44.1 kHz stereo sound effects, ambient soundscapes, foley, and musical cues
    in ~3–8s per clip on T4/L4/G4 GPUs (~4.3 GB VRAM).
  - Supports single prompt generation (`generate` / `sfx`), multi-variation sampling (`-n 2..4`),
    deterministic `--seed`, resumable cue-sheet batch generation (`batch` from `.txt`/`.json`/`.jsonl`),
    compressed formats (`wav`, `mp3`, `flac`, `ogg`, `opus`, `aac`), and Google Drive output persistence.

Usage:
  colabllm tangoflux up                                 # T4 GPU (default) + configured tunnel
  colabllm tangoflux up --gpu L4                        # Faster inference on L4
  colabllm tangoflux up --variant base                  # Load declare-lab/TangoFlux-base
  colabllm tangoflux status
  colabllm tangoflux generate "Hammer slowly hitting the wooden table" -d 10 -s 50 -o hammer.wav
  colabllm tangoflux sfx "Heavy rain and distant thunder on a cabin roof" -d 15 -n 2 -o rain.wav
  colabllm tangoflux batch sfx_cues.json -O ./sfx-pack
  colabllm tangoflux down
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from cli_support import http_timeout, nonnegative_seconds, resolve_endpoint, safe_urlopen  # noqa: E402
from colabsvc import Service, State, die, log, main  # noqa: E402

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
SUPPORTED_AUDIO_FORMATS = ("wav", "mp3", "flac", "ogg", "opus", "aac")

SERVICE = Service(
    name="tangoflux",
    notebook="tangoflux.ipynb",
    service_port=8000,
    auth="builtin",
    public_port=8000,
    gpu="T4",
    high_mem=False,
    default_hostname="tangoflux",
    ready_timeout=2400,
    env={
        "CT_TANGOFLUX_VARIANT": "crpo",
        "CT_TANGOFLUX_MODEL": "",
        "CT_TANGOFLUX_ALIAS": "tangoflux",
    },
)

UP_ENV_FLAGS = {
    "--variant": "CT_TANGOFLUX_VARIANT",
    "--model": "CT_TANGOFLUX_MODEL",
    "--alias": "CT_TANGOFLUX_ALIAS",
}


def _get_endpoint(url_override: str | None = None, token_override: str | None = None) -> tuple[str, str]:
    return resolve_endpoint(
        State(SERVICE).load(),
        url_override,
        token_override,
        url_env="TANGOFLUX_URL",
        token_env="TANGOFLUX_API_TOKEN",
    )


def _http_json(
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    method: str | None = None,
    timeout: int | float | None = 180,
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


def _http_audio_bytes(
    url: str,
    token: str,
    payload: dict[str, Any],
    timeout: int | float | None = 300,
) -> tuple[bytes, dict[str, str]]:
    req = urllib.request.Request(
        url,
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


def _slugify_cue_name(raw_name: str, index: int) -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "-", (raw_name or "").strip(), flags=re.UNICODE).strip("-_")
    if not cleaned:
        cleaned = f"sfx-{index:03d}"
    return f"{index:03d}-{cleaned[:48]}" if not re.match(r"^\d{2,4}[-_]", cleaned) else cleaned[:54]


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _wav_info(wav_bytes: bytes) -> dict[str, Any]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        nch = wf.getnchannels()
        nframes = wf.getnframes()
        return {
            "sample_rate": sr,
            "channels": nch,
            "duration_sec": round(nframes / float(sr), 2) if sr > 0 else 0.0,
        }


def _validate_cli_gen_args(duration: float, steps: int, guidance_scale: float, num_waveforms: int = 1) -> None:
    if not math.isfinite(duration) or duration < 1.0 or duration > 30.0:
        die("--duration must be between 1.0 and 30.0 seconds.")
    if steps < 1 or steps > 200:
        die("--steps must be between 1 and 200.")
    if not math.isfinite(guidance_scale) or guidance_scale < 1.0 or guidance_scale > 20.0:
        die("--guidance-scale must be between 1.0 and 20.0.")
    if num_waveforms < 1 or num_waveforms > 4:
        die("--num-waveforms (-n) must be between 1 and 4.")


def _load_cue_sheet(
    path: Path,
    *,
    default_duration: float,
    default_steps: int,
    default_guidance_scale: float,
    default_seed: int | None,
) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        die(f"Cue sheet file is empty: {path}")

    entries: list[Any] = []
    suffix = path.suffix.lower()
    if suffix == ".json" or raw.startswith(("[", "{")):
        try:
            parsed = json.loads(raw)
        except Exception as err:
            if suffix == ".jsonl":
                parsed = None
            else:
                die(f"Invalid JSON in {path}: {err}")
        if suffix != ".jsonl" and parsed is not None:
            if isinstance(parsed, dict):
                items = parsed.get("items") or parsed.get("cues")
                if not isinstance(items, list):
                    die(f"JSON object in {path} must contain an `items` or `cues` list.")
                entries = items
            elif isinstance(parsed, list):
                entries = parsed
            else:
                die(f"Unsupported JSON root in {path}")
    if not entries and suffix == ".jsonl":
        for lineno, line in enumerate(raw.splitlines(), 1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            try:
                entries.append(json.loads(s))
            except Exception as err:
                die(f"Invalid JSONL at {path}:{lineno}: {err}")
    if not entries and suffix not in (".json", ".jsonl"):
        # Plain text format: one prompt per line, or `name | prompt` / `name \t prompt`
        for line in raw.splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "\t" in s:
                nm, pr = s.split("\t", 1)
                entries.append({"name": nm.strip(), "prompt": pr.strip()})
            elif "|" in s:
                nm, pr = s.split("|", 1)
                entries.append({"name": nm.strip(), "prompt": pr.strip()})
            else:
                entries.append({"prompt": s})

    cues: list[dict[str, Any]] = []
    for idx, item in enumerate(entries, 1):
        if isinstance(item, str):
            item = {"prompt": item}
        if not isinstance(item, dict):
            die(f"Cue #{idx} in {path} must be a string or object.")
        prompt = str(item.get("prompt") or item.get("input") or "").strip()
        if not prompt:
            die(f"Cue #{idx} in {path} is missing `prompt`.")
        dur = float(item.get("duration", default_duration))
        steps = int(item.get("steps", item.get("num_inference_steps", default_steps)))
        cfg = float(item.get("guidance_scale", item.get("cfg_scale", default_guidance_scale)))
        _validate_cli_gen_args(dur, steps, cfg, 1)
        seed_val = item.get("seed", default_seed)
        raw_name = str(item.get("name") or prompt[:32]).strip()
        cue: dict[str, Any] = {
            "index": idx,
            "name": raw_name,
            "slug": _slugify_cue_name(raw_name, idx),
            "prompt": prompt,
            "duration": dur,
            "steps": steps,
            "guidance_scale": cfg,
        }
        if seed_val is not None:
            cue["seed"] = int(seed_val)
        cues.append(cue)
    return cues


def cmd_models(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.api_token)
    code_m, models_data = _http_json(f"{base_url}/v1/models", token, timeout=30)
    code_i, info_data = _http_json(f"{base_url}/api/ct/info", token, timeout=30)
    if code_m != 200:
        print(json.dumps(models_data, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    combined = {
        "models": [m.get("id") for m in (models_data.get("data") or []) if isinstance(m, dict)],
        "info": info_data if code_i == 200 else {},
    }
    if args.json:
        print(json.dumps(combined, indent=2, ensure_ascii=False))
        return 0
    info = combined["info"]
    log(
        f"Models: {', '.join(combined['models'])} | GPU: {info.get('gpu_name', '?')} | "
        f"{info.get('sample_rate', 44100)} Hz, {info.get('channels', 2)}ch, "
        f"duration={info.get('duration_range_sec', [1, 30])}s"
    )
    return 0


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


def cmd_generate(args: argparse.Namespace) -> int:
    base_url, token = _get_endpoint(args.url, args.api_token)
    prompt = (args.prompt or args.prompt_opt or "").strip()
    if args.prompt_file:
        pf = Path(args.prompt_file).expanduser()
        if not pf.is_file():
            die(f"Prompt file not found: {pf}")
        prompt = pf.read_text(encoding="utf-8").strip()
    if not prompt:
        die("Provide a sound description via positional `prompt`, `--prompt`, or `--prompt-file`.")

    duration = float(args.duration)
    steps = int(args.steps)
    guidance_scale = float(args.guidance_scale)
    num_waveforms = int(args.num_waveforms)
    _validate_cli_gen_args(duration, steps, guidance_scale, num_waveforms)

    out_fmt = args.format.lower()
    payload: dict[str, Any] = {
        "prompt": prompt,
        "duration": duration,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "num_waveforms": num_waveforms,
        "response_format": out_fmt,
    }
    if args.seed is not None:
        payload["seed"] = int(args.seed)
    if args.save_drive:
        payload["save_to_drive"] = args.save_drive

    base_out = Path(
        args.output or f"tangoflux-{time.strftime('%Y%m%d-%H%M%S')}.{out_fmt}"
    ).expanduser()
    base_out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    if num_waveforms == 1:
        try:
            audio_bytes, hdrs = _http_audio_bytes(
                f"{base_url}/api/ct/generate", token, payload, timeout=args.timeout
            )
        except urllib.error.HTTPError as err:
            raw_err = err.read().decode("utf-8", errors="replace")
            print(f"[tangoflux] HTTP {err.code} from /api/ct/generate:\n{raw_err}", file=sys.stderr)
            return 1
        elapsed = time.monotonic() - t0
        base_out.write_bytes(audio_bytes)
        sample_rate = int(hdrs.get("x-audio-sample-rate") or 44100)
        channels = int(hdrs.get("x-audio-channels") or 2)
        actual_seed = int(hdrs.get("x-audio-seed") or (args.seed if args.seed is not None else 0))
        saved_drive = hdrs.get("x-saved-drive-path")

        meta = {
            "prompt": prompt,
            "output": str(base_out),
            "format": out_fmt,
            "bytes": len(audio_bytes),
            "duration_sec": duration,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": actual_seed,
            "sample_rate": sample_rate,
            "channels": channels,
            "elapsed_sec": round(elapsed, 2),
            "saved_drive_path": saved_drive,
        }
        if out_fmt == "wav":
            try:
                meta.update(_wav_info(audio_bytes))
            except Exception:
                pass
        if args.json:
            print(json.dumps(meta, indent=2, ensure_ascii=False))
        else:
            msg = (
                f"Generated {len(audio_bytes):,} bytes ({out_fmt}, {sample_rate} Hz, {channels}ch, "
                f"duration={duration:g}s, steps={steps}, cfg={guidance_scale:g}, seed={actual_seed}) "
                f"in {elapsed:.2f}s -> {base_out}"
            )
            if saved_drive:
                msg += f" (Drive: {saved_drive})"
            log(msg)
        return 0

    # Multi-waveform (-n 2..4) JSON response
    payload["return_json"] = True
    code, data = _http_json(f"{base_url}/api/ct/generate", token, payload=payload, timeout=args.timeout)
    elapsed = time.monotonic() - t0
    if code != 200:
        print(json.dumps(data, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    items = data.get("data") or []
    written_files: list[str] = []
    drive_paths: list[str] = []
    suffix = base_out.suffix or f".{out_fmt}"
    stem = base_out.stem if base_out.suffix else base_out.name
    for idx, entry in enumerate(items, 1):
        audio_raw = base64.b64decode(entry["b64_json"])
        target = base_out.with_name(f"{stem}_{idx:02d}{suffix}")
        target.write_bytes(audio_raw)
        written_files.append(str(target))
        if entry.get("saved_drive_path"):
            drive_paths.append(str(entry["saved_drive_path"]))

    summary = {
        "prompt": prompt,
        "outputs": written_files,
        "format": out_fmt,
        "count": len(written_files),
        "duration_sec": duration,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": data.get("seed"),
        "sample_rate": data.get("sample_rate", 44100),
        "channels": data.get("channels", 2),
        "elapsed_sec": round(elapsed, 2),
        "saved_drive_paths": drive_paths,
    }
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        log(
            f"Generated {len(written_files)} variations ({out_fmt}, duration={duration:g}s, "
            f"steps={steps}, seed={data.get('seed')}) in {elapsed:.2f}s -> {', '.join(written_files)}"
        )
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    in_path = Path(args.input_file).expanduser()
    if not in_path.is_file():
        die(f"Cue sheet file not found: {in_path}")
    if args.batch_size < 1 or args.batch_size > 16:
        die("--batch-size must be between 1 and 16.")

    cues = _load_cue_sheet(
        in_path,
        default_duration=float(args.duration),
        default_steps=int(args.steps),
        default_guidance_scale=float(args.guidance_scale),
        default_seed=args.seed,
    )
    if not cues:
        die(f"No sound cues found in {in_path}")

    base_url, token = _get_endpoint(args.url, args.api_token)
    out_fmt = args.format.lower()
    out_dir = Path(args.output_dir or f"./{in_path.stem}-sfx").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    manifest: dict[str, Any] = {
        "source_file": str(in_path),
        "format": out_fmt,
        "total_cues": len(cues),
        "cues": {},
    }
    if manifest_path.is_file() and not args.force:
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(prev, dict) and isinstance(prev.get("cues"), dict):
                manifest["cues"] = prev["cues"]
        except Exception:
            pass

    pending: list[dict[str, Any]] = []
    completed_outputs: list[dict[str, Any]] = []
    for cue in cues:
        ckey = str(cue["index"])
        out_file = out_dir / f"{cue['slug']}.{out_fmt}"
        cue["output_path"] = out_file
        prev_rec = manifest["cues"].get(ckey) or {}
        if not args.force and prev_rec.get("status") == "completed" and out_file.is_file() and out_file.stat().st_size > 64:
            completed_outputs.append({
                "index": cue["index"],
                "name": cue["name"],
                "output": str(out_file),
                "duration_sec": cue["duration"],
                "seed": prev_rec.get("seed"),
                "resumed": True,
            })
        else:
            pending.append(cue)

    log(
        f"Batch SFX generation: {len(cues)} cue(s) from {in_path.name} -> {out_dir} "
        f"({len(completed_outputs)} cached, {len(pending)} to generate, batch_size={args.batch_size})"
    )

    t0 = time.monotonic()
    for off in range(0, len(pending), args.batch_size):
        slice_cues = pending[off : off + args.batch_size]
        req_items: list[dict[str, Any]] = []
        for c in slice_cues:
            item_body: dict[str, Any] = {
                "name": c["slug"],
                "prompt": c["prompt"],
                "duration": c["duration"],
                "steps": c["steps"],
                "guidance_scale": c["guidance_scale"],
                "response_format": out_fmt,
            }
            if "seed" in c:
                item_body["seed"] = c["seed"]
            if args.save_drive:
                item_body["save_to_drive"] = args.save_drive
            req_items.append(item_body)

        code, data = _http_json(
            f"{base_url}/api/ct/batch",
            token,
            payload={"items": req_items, "response_format": out_fmt},
            timeout=args.timeout,
        )
        if code != 200 or not isinstance(data.get("results"), list):
            _write_json_atomic(manifest_path, manifest)
            die(f"Batch SFX HTTP {code} failed: {json.dumps(data, ensure_ascii=False)}\nRe-run to resume.")

        for c, res_item in zip(slice_cues, data["results"]):
            ckey = str(c["index"])
            if res_item.get("status") != "success" or not res_item.get("audio_data"):
                manifest["cues"][ckey] = {
                    "index": c["index"],
                    "name": c["name"],
                    "prompt": c["prompt"],
                    "status": "error",
                    "error": res_item.get("error", "unknown error"),
                }
                _write_json_atomic(manifest_path, manifest)
                die(f"Cue #{c['index']} ({c['name']}) failed: {res_item.get('error')}\nRe-run to resume.")

            audio_bytes = base64.b64decode(res_item["audio_data"])
            c["output_path"].write_bytes(audio_bytes)
            manifest["cues"][ckey] = {
                "index": c["index"],
                "name": c["name"],
                "slug": c["slug"],
                "prompt": c["prompt"],
                "duration_sec": c["duration"],
                "steps": c["steps"],
                "guidance_scale": c["guidance_scale"],
                "seed": res_item.get("seed"),
                "output": c["output_path"].name,
                "saved_drive_path": res_item.get("saved_drive_path"),
                "status": "completed",
                "updated_at": time.strftime("%FT%T%z"),
            }
            completed_outputs.append({
                "index": c["index"],
                "name": c["name"],
                "output": str(c["output_path"]),
                "duration_sec": c["duration"],
                "seed": res_item.get("seed"),
                "saved_drive_path": res_item.get("saved_drive_path"),
                "resumed": False,
            })
        _write_json_atomic(manifest_path, manifest)
        done_cnt = len(completed_outputs)
        log(f"  [{done_cnt}/{len(cues)}] saved {', '.join(c['output_path'].name for c in slice_cues)}")

    elapsed = time.monotonic() - t0
    summary = {
        "input_file": str(in_path),
        "output_dir": str(out_dir),
        "manifest": str(manifest_path),
        "cues_completed": len(completed_outputs),
        "cues_generated": len(pending),
        "elapsed_sec": round(elapsed, 2),
        "outputs": sorted(completed_outputs, key=lambda x: x["index"]),
    }
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        log(f"Batch complete: {len(completed_outputs)} SFX file(s) in {elapsed:.2f}s -> {out_dir}")
    return 0


def _add_endpoint_flags(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--url", help="Override public service URL")
    sp.add_argument("--api-token", help="Override Bearer API token")


def _build_custom_parser(parser=None, commands=None) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(prog="colabllm tangoflux")
    sub = commands or p.add_subparsers(dest="cmd", required=True)

    for gen_name in ("generate", "sfx"):
        gp = sub.add_parser(gen_name, help="Generate 44.1 kHz stereo sound effects / audio from a text prompt")
        gp.add_argument("prompt", nargs="?", help="Text description of the sound effect, ambience, or musical cue")
        gp.add_argument("-p", "--prompt", dest="prompt_opt", help="Text prompt (alternative to positional argument)")
        gp.add_argument("--prompt-file", help="Read text prompt from a local UTF-8 file")
        gp.add_argument("-o", "--output", default=None, help="Local output file path (default: ./tangoflux-<timestamp>.<format>)")
        gp.add_argument("-d", "--duration", type=float, default=10.0, help="Audio duration in seconds in [1.0, 30.0] (default: 10.0)")
        gp.add_argument("-s", "--steps", type=int, default=50, help="Flow-matching inference steps in [1, 200] (default: 50)")
        gp.add_argument("-g", "--guidance-scale", type=float, default=4.5, help="Classifier-free guidance scale in [1.0, 20.0] (default: 4.5)")
        gp.add_argument("--seed", type=int, default=None, help="Deterministic random seed (default: random per request)")
        gp.add_argument("-n", "--num-waveforms", type=int, default=1, help="Number of variations to generate in [1, 4] (default: 1)")
        gp.add_argument("--format", default="wav", choices=list(SUPPORTED_AUDIO_FORMATS), help="Output audio format (default: wav)")
        gp.add_argument("--save-drive", nargs="?", const="auto", help="Also save output audio to Google Drive (optional path or 'auto' for colab-llm/tangoflux/outputs/)")
        gp.add_argument("--json", action="store_true", help="Output generation metadata as JSON on stdout")
        gp.add_argument("--timeout", type=nonnegative_seconds, default=300, help="HTTP request timeout in seconds; 0 = unlimited (default: 300)")
        _add_endpoint_flags(gp)

    bp = sub.add_parser("batch", help="Generate a batch of sound effects from a .txt, .json, or .jsonl cue sheet")
    bp.add_argument("input_file", help="Local cue sheet (.txt with one prompt or 'name | prompt' per line, .json, or .jsonl)")
    bp.add_argument("-O", "--output-dir", default=None, help="Output directory for generated SFX and manifest.json (default: ./<sheet>-sfx)")
    bp.add_argument("-d", "--duration", type=float, default=10.0, help="Default duration in seconds when omitted per cue (default: 10.0)")
    bp.add_argument("-s", "--steps", type=int, default=50, help="Default inference steps when omitted per cue (default: 50)")
    bp.add_argument("-g", "--guidance-scale", type=float, default=4.5, help="Default guidance scale when omitted per cue (default: 4.5)")
    bp.add_argument("--seed", type=int, default=None, help="Default seed when omitted per cue")
    bp.add_argument("--batch-size", type=int, default=4, help="Cues per /api/ct/batch HTTP call in [1, 16] (default: 4)")
    bp.add_argument("--format", default="wav", choices=list(SUPPORTED_AUDIO_FORMATS), help="Output audio format (default: wav)")
    bp.add_argument("--save-drive", nargs="?", const="auto", help="Also save generated cues to Google Drive")
    bp.add_argument("--force", action="store_true", help="Re-generate cues even if already completed in manifest.json")
    bp.add_argument("--json", action="store_true", help="Output batch summary as JSON on stdout")
    bp.add_argument("--timeout", type=nonnegative_seconds, default=600, help="HTTP timeout per batch call in seconds; 0 = unlimited (default: 600)")
    _add_endpoint_flags(bp)

    mp = sub.add_parser("models", help="Show active TangoFlux model and audio runtime parameters")
    mp.add_argument("--json", action="store_true", help="Output raw JSON")
    _add_endpoint_flags(mp)

    lsd = sub.add_parser("ls-drive", help="List generated audio files on Google Drive")
    lsd.add_argument("path", nargs="?", default="", help="Drive folder or glob pattern (default: colab-llm/tangoflux/outputs)")
    lsd.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    lsd.add_argument("--json", action="store_true", help="Output raw JSON")
    _add_endpoint_flags(lsd)

    return p


CUSTOM_CMDS = {"generate", "sfx", "batch", "models", "ls-drive"}


def configure_cli(parser, commands, up):
    _build_custom_parser(parser, commands)
    for flag, env in UP_ENV_FLAGS.items():
        up.add_argument(flag, dest="up_" + env, help=f"override {env}")


_BASE_ENV = dict(SERVICE.env)


def prepare_up(args):
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
                "generate": cmd_generate,
                "sfx": cmd_generate,
                "batch": cmd_batch,
                "models": cmd_models,
                "ls-drive": cmd_ls_drive,
            },
        )
    )
