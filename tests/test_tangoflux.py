"""Unit tests for TangoFlux ASGI server security, 44.1 kHz stereo WAV encoding, and CLI workflows."""
from __future__ import annotations

import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

import tangoflux_server as tf_srv  # noqa: E402


def _load_tangoflux_service():
    path = ROOT / "src/colabllm/services/tangoflux.py"
    spec = importlib.util.spec_from_file_location("tangoflux_service_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_stereo_waveform(duration_sec: float = 0.25, sample_rate: int = 44100) -> list[list[float]]:
    n = int(round(duration_sec * sample_rate))
    left = [0.5 * math.sin(2.0 * math.pi * 220.0 * i / sample_rate) for i in range(n)]
    right = [0.5 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate) for i in range(n)]
    return [left, right]


async def _call_asgi(
    app,
    method: str,
    path: str,
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
    body: bytes = b"",
    query_string: bytes = b"",
) -> tuple[int, dict[str, str], bytes]:
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "root_path": "",
        "query_string": query_string,
        "headers": headers or [],
    }
    sent_body = False
    status_code = 500
    resp_headers: dict[str, str] = {}
    chunks: list[bytes] = []

    async def receive():
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(msg):
        nonlocal status_code
        if msg["type"] == "http.response.start":
            status_code = msg["status"]
            for k, v in msg.get("headers", []):
                resp_headers[k.decode("latin1").lower()] = v.decode("latin1")
        elif msg["type"] == "http.response.body":
            chunks.append(msg.get("body", b""))

    await app(scope, receive, send)
    return status_code, resp_headers, b"".join(chunks)


class TangoFluxServerTest(unittest.TestCase):
    def test_stereo_wav_encoder_and_param_validation(self):
        stereo = _fake_stereo_waveform(0.2, 44100)
        wav_bytes = tf_srv.float_stereo_to_wav_bytes(stereo, sample_rate=44100)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 2)
            self.assertEqual(wf.getsampwidth(), 2)
            self.assertEqual(wf.getframerate(), 44100)
            self.assertEqual(wf.getnframes(), 8820)

        # Hot Oobleck VAE output (>1.0) should headroom-normalize cleanly without hard clipping
        hot = [[v * 4.8 for v in ch] for ch in stereo]  # peak ~ 2.4
        hot_wav = tf_srv.float_stereo_to_wav_bytes(hot, sample_rate=44100)
        with wave.open(io.BytesIO(hot_wav), "rb") as wf:
            import array as _arr

            pcm = _arr.array("h")
            pcm.frombytes(wf.readframes(wf.getnframes()))
            peak_i16 = max(abs(x) for x in pcm)
            self.assertGreater(peak_i16, 31500)
            self.assertLess(peak_i16, 32500)

        for bad_payload in (
            {},
            {"prompt": "   "},
            {"prompt": "rain", "duration": 0.2},
            {"prompt": "rain", "duration": 35.0},
            {"prompt": "rain", "steps": 0},
            {"prompt": "rain", "guidance_scale": 0.5},
            {"prompt": "rain", "num_waveforms": 9},
            {"prompt": "rain", "response_format": "exe"},
        ):
            with self.assertRaises(ValueError, msg=bad_payload):
                tf_srv._parse_generate_params(bad_payload)

    def test_asgi_auth_cors_drive_save_and_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mydrive = root / "drive/MyDrive"
            data_dir = mydrive / "colab-llm/tangoflux"
            out_dir = data_dir / "outputs"
            out_dir.mkdir(parents=True)

            recorded_calls: list[dict] = []

            def fake_generate(*, prompt, duration, steps, guidance_scale, seed, num_waveforms=1):
                recorded_calls.append({
                    "prompt": prompt,
                    "duration": duration,
                    "steps": steps,
                    "guidance_scale": guidance_scale,
                    "seed": seed,
                    "num_waveforms": num_waveforms,
                })
                return [_fake_stereo_waveform(0.15, 44100) for _ in range(num_waveforms)]

            with patch.multiple(
                tf_srv,
                API_TOKEN="tf-secret-token",
                SERVED_MODEL="declare-lab/TangoFlux",
                MODEL_ALIAS="tangoflux",
                DATA_DIR=data_dir,
                DRIVE_ROOT=root / "drive",
                MYDRIVE_ROOT=mydrive,
                DRIVE_OUTPUT_DIR=out_dir,
            ):
                app = tf_srv.TangoFluxApp(fake_generate, gpu_name="Tesla T4", sample_rate=44100)

                # 1. CORS preflight requires no auth
                code, hdrs, _ = asyncio.run(_call_asgi(app, "OPTIONS", "/api/ct/generate"))
                self.assertEqual(code, 204)
                self.assertEqual(hdrs.get("access-control-allow-origin"), "*")

                # 2. Missing / wrong auth fails closed with 401
                code, _, _ = asyncio.run(_call_asgi(app, "GET", "/health"))
                self.assertEqual(code, 401)
                code, _, _ = asyncio.run(
                    _call_asgi(app, "GET", "/v1/models", headers=[(b"authorization", b"Bearer wrong")])
                )
                self.assertEqual(code, 401)

                auth_hdr = [(b"authorization", b"Bearer tf-secret-token")]

                # 3. Single generation returns 44.1 kHz stereo WAV and saves to Drive
                code, resp_hdrs, wav_body = asyncio.run(
                    _call_asgi(
                        app,
                        "POST",
                        "/api/ct/generate",
                        headers=auth_hdr,
                        body=json.dumps({
                            "prompt": "Hammer slowly hitting the wooden table",
                            "duration": 5.0,
                            "steps": 25,
                            "seed": 42,
                            "save_to_drive": "auto",
                        }).encode("utf-8"),
                    )
                )
                self.assertEqual(code, 200)
                self.assertEqual(resp_hdrs.get("x-audio-sample-rate"), "44100")
                self.assertEqual(resp_hdrs.get("x-audio-channels"), "2")
                self.assertEqual(resp_hdrs.get("x-audio-seed"), "42")
                saved_drive = resp_hdrs.get("x-saved-drive-path", "")
                self.assertTrue(saved_drive and Path(saved_drive).is_file())
                with wave.open(io.BytesIO(wav_body), "rb") as wf:
                    self.assertEqual(wf.getnchannels(), 2)
                    self.assertEqual(wf.getframerate(), 44100)

                # 4. Multi-waveform (-n 2) returns JSON with 2 base64 audio clips
                code, _, multi_raw = asyncio.run(
                    _call_asgi(
                        app,
                        "POST",
                        "/v1/audio/generations",
                        headers=auth_hdr,
                        body=json.dumps({"prompt": "Rain on window", "duration": 4.0, "num_waveforms": 2}).encode("utf-8"),
                    )
                )
                self.assertEqual(code, 200)
                multi_data = json.loads(multi_raw)
                self.assertEqual(len(multi_data["data"]), 2)

                # 5. Drive traversal rejected with 400
                code, _, _ = asyncio.run(
                    _call_asgi(
                        app,
                        "POST",
                        "/api/ct/generate",
                        headers=auth_hdr,
                        body=json.dumps({"prompt": "Rain", "save_to_drive": "../../outside.wav"}).encode("utf-8"),
                    )
                )
                self.assertEqual(code, 400)


class TangoFluxCliEndToEndTest(unittest.TestCase):
    def test_cli_generate_multi_and_resumable_batch(self):
        svc = _load_tangoflux_service()
        sample_wav = tf_srv.float_stereo_to_wav_bytes(_fake_stereo_waveform(0.2, 44100), 44100)
        sample_b64 = base64.b64encode(sample_wav).decode("ascii")
        api_calls: list[tuple[str, dict]] = []

        class MockTangoHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.headers.get("Authorization") != "Bearer tf-unit":
                    self.send_response(401)
                    self.end_headers()
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode("utf-8"))
                api_calls.append((self.path, body))
                if self.path == "/api/ct/generate":
                    if body.get("num_waveforms", 1) > 1 or body.get("return_json"):
                        out = json.dumps({
                            "seed": body.get("seed", 77),
                            "sample_rate": 44100,
                            "channels": 2,
                            "data": [
                                {"index": i, "format": "wav", "b64_json": sample_b64}
                                for i in range(body.get("num_waveforms", 1))
                            ],
                        }).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(out)))
                        self.end_headers()
                        self.wfile.write(out)
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("X-Audio-Sample-Rate", "44100")
                    self.send_header("X-Audio-Channels", "2")
                    self.send_header("X-Audio-Seed", str(body.get("seed", 123)))
                    self.send_header("Content-Length", str(len(sample_wav)))
                    self.end_headers()
                    self.wfile.write(sample_wav)
                    return
                if self.path == "/api/ct/batch":
                    results = [
                        {
                            "index": i,
                            "name": it.get("name", f"sfx-{i}"),
                            "status": "success",
                            "seed": it.get("seed", 100 + i),
                            "format": "wav",
                            "audio_data": sample_b64,
                        }
                        for i, it in enumerate(body["items"])
                    ]
                    out = json.dumps({"results": results}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                self.send_response(404)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), MockTangoHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base_url = f"http://127.0.0.1:{server.server_port}"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = svc._build_custom_parser()

            # 1. Single sfx generation
            out_single = root / "hammer.wav"
            args_gen = parser.parse_args([
                "sfx", "Hammer slowly hitting the wooden table",
                "-d", "8", "-s", "30", "--seed", "99",
                "-o", str(out_single),
                "--url", base_url, "--api-token", "tf-unit",
            ])
            self.assertEqual(svc.cmd_generate(args_gen), 0)
            self.assertTrue(out_single.is_file())

            # 2. Multi-variation (-n 2) generation
            out_multi = root / "warp.wav"
            args_multi = parser.parse_args([
                "generate", "Spacecraft engine hum",
                "-n", "2", "-d", "6",
                "-o", str(out_multi),
                "--url", base_url, "--api-token", "tf-unit",
            ])
            self.assertEqual(svc.cmd_generate(args_multi), 0)
            self.assertTrue((root / "warp_01.wav").is_file())
            self.assertTrue((root / "warp_02.wav").is_file())

            # 3. Batch cue sheet (.txt with 'name | prompt' + plain prompt) and zero-call resume
            sheet = root / "cues.txt"
            sheet.write_text(
                "# Teahouse SFX\n"
                "fire_crackle | Warm crackling charcoal fire in a wooden room\n"
                "Heavy winter wind outside a cabin door\n",
                encoding="utf-8",
            )
            batch_dir = root / "sfx-pack"
            args_batch = parser.parse_args([
                "batch", str(sheet),
                "-d", "10", "-O", str(batch_dir),
                "--url", base_url, "--api-token", "tf-unit",
            ])
            self.assertEqual(svc.cmd_batch(args_batch), 0)
            self.assertEqual(len(list(batch_dir.glob("*.wav"))), 2)

            calls_before_resume = len(api_calls)
            self.assertEqual(svc.cmd_batch(args_batch), 0)
            self.assertEqual(len(api_calls), calls_before_resume)


if __name__ == "__main__":
    unittest.main()
