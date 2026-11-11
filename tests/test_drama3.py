"""Unit tests for Fish Audio Drama 3 narration helpers, ASGI middleware, and CLI workflows."""
from __future__ import annotations

import asyncio
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

import drama3_middleware as mw  # noqa: E402
import drama3_narration as narr  # noqa: E402


def _load_drama3_service():
    path = ROOT / "src/colabllm/services/drama3.py"
    spec = importlib.util.spec_from_file_location("drama3_service_test", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_tone_wav(tone_sec: float = 0.5, silence_sec: float = 0.0, sample_rate: int = 44100) -> bytes:
    n_tone = int(sample_rate * tone_sec)
    n_sil = int(sample_rate * silence_sec)
    tone = [int(8000 * math.sin(2.0 * math.pi * 440.0 * i / sample_rate)) for i in range(n_tone)]
    pcm = struct.pack(f"<{n_tone}h", *tone) + (b"\x00\x00" * n_sil)
    return narr.pcm16_to_wav_bytes(pcm, sample_rate=sample_rate, channels=1)


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


class Drama3NarrationTest(unittest.TestCase):
    def test_chapter_splitting_and_chunking(self):
        novel = (
            "第一章 雨夜归人\n"
            "窗外的雨下了一整夜，屋檐上的水珠滴答作响。林晚推开木门，抖落肩头的寒气。\n\n"
            "[role:林晚] “你终于回来了，我等了你很久！”\n\n"
            "第二章 破晓\n"
            "天边泛起鱼肚白，<|speaker:2|>远处的钟声缓缓敲响。新的旅程即将开始。"
        )
        chapters = narr.split_into_chapters(novel)
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["title"], "第一章 雨夜归人")
        self.assertEqual(chapters[1]["title"], "第二章 破晓")

        chunks_ch1 = narr.split_text_chunks(chapters[0]["text"], max_chars=40)
        self.assertTrue(all(len(c["text"]) <= 40 for c in chunks_ch1))
        dialogue_chunks = [c for c in chunks_ch1 if c["role_hint"] == "林晚"]
        self.assertEqual(len(dialogue_chunks), 1)
        self.assertEqual(
            narr.assign_chunk_voice(dialogue_chunks[0], default_voice="narrator", cast={"characters": {"林晚": "heroine"}}),
            "heroine",
        )

        chunks_ch2 = narr.split_text_chunks(chapters[1]["text"], max_chars=80)
        self.assertTrue(all("<|speaker:" not in c["text"] for c in chunks_ch2))

    def test_token_estimation_and_silence_trimming_and_concat(self):
        short_tokens = narr.estimate_max_new_tokens("你好世界。")
        long_tokens = narr.estimate_max_new_tokens("这是一个更长的中文朗读段落，包含多个分句，[sighs] 以及叹息停顿。" * 4)
        self.assertGreaterEqual(short_tokens, 192)
        self.assertGreater(long_tokens, short_tokens)
        self.assertLessEqual(long_tokens, 2048)

        raw_wav = _make_tone_wav(tone_sec=0.5, silence_sec=4.0)
        self.assertAlmostEqual(narr.wav_duration_seconds(raw_wav), 4.5, places=2)
        trimmed = narr.trim_trailing_silence_wav(raw_wav, keep_tail_ms=200)
        trimmed_dur = narr.wav_duration_seconds(trimmed)
        self.assertGreater(trimmed_dur, 0.55)
        self.assertLess(trimmed_dur, 0.85)

        seg_a = _make_tone_wav(tone_sec=0.4, silence_sec=0.0)
        seg_b = _make_tone_wav(tone_sec=0.6, silence_sec=0.0)
        stitched = narr.concat_wav_segments([(seg_a, 500), (seg_b, 300)])
        self.assertAlmostEqual(narr.wav_duration_seconds(stitched), 1.5, places=2)

    def test_intra_paragraph_multi_character_dialogue_and_scare_quotes(self):
        cast = {
            "narrator": "narrator",
            "dialogue": "fallback_diag",
            "characters": {
                "林晚": {"voice": "linwan_v", "aliases": ["小晚"], "speed": 1.04},
                "陈默": {"voice": "chenmo_v", "aliases": ["老陈"], "speed": 0.95},
            },
        }
        prose = (
            "他一向最讨厌所谓的“规矩”二字。\n\n"
            "林晚看着陈默，轻声问道：“你等很久了吧？”陈默抬头笑了笑：“刚泡好茶，快坐下吧。”\n\n"
            "“别客气，”老陈笑道，“外面风雪太大了。”\n\n"
            "“谢谢你。”小晚双手捧起茶杯。陈默把雨伞靠在门边，低声道：“路上耽搁了一会儿。”\n\n"
            "“先暖暖手吧。”"
        )
        chunks = narr.split_text_chunks(prose, max_chars=120, split_dialogue=True, cast=cast)
        voices = [narr.assign_chunk_voice(c, default_voice="narrator", cast=cast) for c in chunks]

        # Paragraph 0: inline scare-quote "规矩" stays unsplit as a single narrator chunk
        p0 = [c for c in chunks if c["paragraph_index"] == 0]
        self.assertEqual(len(p0), 1)
        self.assertFalse(p0[0]["is_dialogue"])
        self.assertIn("“规矩”", p0[0]["text"])

        # Paragraph 1: narrator -> 林晚 -> narrator -> 陈默
        p1 = [c for c in chunks if c["paragraph_index"] == 1]
        self.assertEqual([c["role_hint"] for c in p1], [None, "林晚", None, "陈默"])
        self.assertEqual([c["turn_transition"] for c in p1], [True, True, True, False])
        self.assertEqual([c["paragraph_end"] for c in p1], [False, False, False, True])

        # Paragraph 2: interrupted quote attributed via alias 老陈 -> 陈默 on both halves
        p2 = [c for c in chunks if c["paragraph_index"] == 2]
        self.assertEqual([c["role_hint"] for c in p2], ["陈默", None, "陈默"])

        # Paragraph 3: post-quote 小晚 -> 林晚, followed by pre-quote 陈默 in same paragraph
        p3 = [c for c in chunks if c["paragraph_index"] == 3]
        self.assertEqual([c["role_hint"] for c in p3], ["林晚", None, "陈默"])

        # Paragraph 4: standalone unattributed quote alternates back to 林晚 after 陈默
        p4 = [c for c in chunks if c["paragraph_index"] == 4]
        self.assertEqual([c["role_hint"] for c in p4], ["林晚"])
        self.assertIn("linwan_v", voices)
        self.assertIn("chenmo_v", voices)

    def test_llm_cast_output_verifier(self):
        orig = "林晚推开门问道：“有人在吗？”陈默抬起头答道：“进来吧。”"
        good_reply = """```json
[
  {"role": "旁白", "text": "林晚推开门问道："},
  {"role": "林晚", "text": "“有人在吗？”"},
  {"role": "旁白", "text": "陈默抬起头答道："},
  {"role": "陈默", "text": "[sighs] “进来吧。”"}
]
```"""
        verified = narr.parse_and_verify_llm_cast_output(good_reply, orig)
        self.assertIsNotNone(verified)
        assert verified is not None
        self.assertIn("[role:林晚]", verified)
        self.assertIn("[role:陈默] [sighs]", verified)

        truncated_reply = '[{"role": "旁白", "text": "林晚推开门问道："}]'
        self.assertIsNone(narr.parse_and_verify_llm_cast_output(truncated_reply, orig))


class Drama3MiddlewareTest(unittest.TestCase):
    def test_auth_cors_admin_block_and_drive_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mydrive = root / "drive/MyDrive"
            data_dir = mydrive / "colab-llm/drama3"
            voices_dir = data_dir / "voices"
            out_dir = data_dir / "outputs"
            speakers_dir = root / "speakers"
            voices_dir.mkdir(parents=True)
            speakers_dir.mkdir(parents=True)
            sample_wav = _make_tone_wav(tone_sec=0.3)
            (voices_dir / "sample.wav").write_bytes(sample_wav)

            captured_downstream: list[dict] = []

            async def fake_downstream(scope, receive, send):
                msg = await receive()
                body = msg.get("body", b"")
                if body:
                    captured_downstream.append(json.loads(body.decode("utf-8")))
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"audio/wav")],
                })
                await send({"type": "http.response.body", "body": sample_wav, "more_body": False})

            with patch.multiple(
                mw,
                API_TOKEN="test-secret-token",
                SERVED_MODEL="fishaudio/drama-3",
                DATA_DIR=data_dir,
                DRIVE_ROOT=root / "drive",
                MYDRIVE_ROOT=mydrive,
                DRIVE_VOICES_DIR=voices_dir,
                DRIVE_OUTPUT_DIR=out_dir,
                SPEAKERS_DIR=speakers_dir,
            ):
                app = mw.Drama3Middleware(fake_downstream)

                # 1. OPTIONS preflight succeeds without auth and returns CORS headers
                code, hdrs, _ = asyncio.run(_call_asgi(app, "OPTIONS", "/v1/audio/speech"))
                self.assertEqual(code, 204)
                self.assertEqual(hdrs.get("access-control-allow-origin"), "*")

                # 2. Missing / wrong auth returns 401
                code, _, _ = asyncio.run(_call_asgi(app, "GET", "/api/ct/info"))
                self.assertEqual(code, 401)
                code, _, _ = asyncio.run(
                    _call_asgi(app, "GET", "/api/ct/info", headers=[(b"authorization", b"Bearer wrong")])
                )
                self.assertEqual(code, 401)

                # 3. Blocked admin endpoint returns 403 even with valid token
                auth_hdr = [(b"authorization", b"Bearer test-secret-token")]
                code, _, _ = asyncio.run(
                    _call_asgi(app, "POST", "/update_weights_from_disk", headers=auth_hdr, body=b"{}")
                )
                self.assertEqual(code, 403)

                # 4. /api/ct/info succeeds
                code, _, info_raw = asyncio.run(_call_asgi(app, "GET", "/api/ct/info", headers=auth_hdr))
                self.assertEqual(code, 200)
                self.assertEqual(json.loads(info_raw)["model"], "fishaudio/drama-3")

                # 5. /v1/audio/speech rewrites model alias, strips unsupported fields, resolves drive:// ref, and saves output
                req_payload = {
                    "model": "drama3",
                    "input": "测试语音。",
                    "language": "zh",
                    "seed": 42,
                    "references": [{"audio_path": "drive://voices/sample.wav", "text": "参考"}],
                    "save_to_drive": "auto",
                }
                code, resp_hdrs, resp_body = asyncio.run(
                    _call_asgi(
                        app,
                        "POST",
                        "/v1/audio/speech",
                        headers=auth_hdr,
                        body=json.dumps(req_payload).encode("utf-8"),
                    )
                )
                self.assertEqual(code, 200)
                self.assertEqual(resp_body, sample_wav)
                self.assertEqual(len(captured_downstream), 1)
                forwarded = captured_downstream[0]
                self.assertEqual(forwarded["model"], "fishaudio/drama-3")
                self.assertNotIn("language", forwarded)
                self.assertNotIn("seed", forwarded)
                self.assertTrue(forwarded["references"][0]["audio_path"].startswith("data:audio/wav;base64,"))
                saved_hdr = resp_hdrs.get("x-saved-drive-path", "")
                self.assertTrue(saved_hdr and Path(saved_hdr).is_file())

                # 6. WAV reference fallback succeeds even when ffmpeg is absent from PATH
                with patch("colab_asgi_common.subprocess.run", side_effect=FileNotFoundError(2, "No ffmpeg", "ffmpeg")):
                    code_no_ff, _, _ = asyncio.run(
                        _call_asgi(
                            app,
                            "POST",
                            "/v1/audio/speech",
                            headers=auth_hdr,
                            body=json.dumps(req_payload).encode("utf-8"),
                        )
                    )
                    self.assertEqual(code_no_ff, 200)


class Drama3CliNarrationEndToEndTest(unittest.TestCase):
    def test_narrate_batch_stitch_and_resume(self):
        svc = _load_drama3_service()
        batch_calls: list[dict] = []
        chunk_wav = _make_tone_wav(tone_sec=0.35, silence_sec=1.5)
        chunk_b64 = base64.b64encode(chunk_wav).decode("ascii")

        class MockDrama3Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                if self.headers.get("Authorization") != "Bearer unit-token":
                    self.send_response(401)
                    self.end_headers()
                    return
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = json.loads(raw.decode("utf-8"))
                if self.path == "/v1/audio/speech/batch":
                    batch_calls.append(body)
                    results = [
                        {
                            "index": i,
                            "status": "success",
                            "audio_data": chunk_b64,
                            "format": "wav",
                            "media_type": "audio/wav",
                            "finish_reason": "stop",
                        }
                        for i in range(len(body["items"]))
                    ]
                    out = json.dumps({"id": "batch-1", "results": results}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                if self.path == "/v1/audio/speech":
                    batch_calls.append(body)
                    self.send_response(200)
                    self.send_header("Content-Type", "audio/wav")
                    self.send_header("Content-Length", str(len(chunk_wav)))
                    self.end_headers()
                    self.wfile.write(chunk_wav)
                    return
                self.send_response(404)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), MockDrama3Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base_url = f"http://127.0.0.1:{server.server_port}"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            book = root / "book.txt"
            book.write_text(
                "第一章 启程\n"
                "清晨的雾气尚未散尽，山道上已经响起了马蹄声。少年背着行囊回望故乡。\n\n"
                "他深吸一口气，转身迈入群山之中。\n\n"
                "第二章 客栈\n"
                "傍晚时分，风雪骤起，前方出现了一盏昏黄的灯笼。\n",
                encoding="utf-8",
            )
            out_dir = root / "out"
            parser = svc._build_custom_parser()

            args1 = parser.parse_args([
                "narrate", str(book),
                "--voice", "narrator",
                "--max-chars", "45",
                "--batch-size", "4",
                "-O", str(out_dir),
                "--url", base_url,
                "--api-token", "unit-token",
            ])
            rc1 = svc.cmd_narrate(args1)
            self.assertEqual(rc1, 0)
            self.assertGreaterEqual(len(batch_calls), 2)

            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["chapters"]["1"]["status"], "completed")
            self.assertEqual(manifest["chapters"]["2"]["status"], "completed")
            wav_files = sorted(out_dir.glob("*.wav"))
            self.assertEqual(len(wav_files), 2)
            # Trailing 1.5s silence on each chunk was trimmed before stitching!
            self.assertLess(narr.wav_duration_seconds(wav_files[0].read_bytes()), 3.5)

            # Re-running without --force must skip all completed chapters with 0 HTTP requests!
            prev_call_count = len(batch_calls)
            rc2 = svc.cmd_narrate(args1)
            self.assertEqual(rc2, 0)
            self.assertEqual(len(batch_calls), prev_call_count)

    def test_narrate_multi_character_cast_and_llm_preprocessor(self):
        svc = _load_drama3_service()
        batch_calls: list[dict] = []
        llm_calls: list[dict] = []
        chunk_wav = _make_tone_wav(tone_sec=0.25, silence_sec=0.0)
        chunk_b64 = base64.b64encode(chunk_wav).decode("ascii")

        class MockDramaAndLlmHandler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path == "/v1/models":
                    out = json.dumps({"data": [{"id": "mock-qwen-annotator"}]}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                body = json.loads(raw.decode("utf-8"))
                if self.path == "/v1/chat/completions":
                    llm_calls.append(body)
                    annotated = json.dumps([
                        {"role": "旁白", "text": "林晚看着陈默问道："},
                        {"role": "林晚", "text": "“你等很久了吧？”"},
                        {"role": "旁白", "text": "陈默微笑着答道："},
                        {"role": "陈默", "text": "“刚泡好茶。”"},
                    ], ensure_ascii=False)
                    resp = {"choices": [{"message": {"content": annotated}}]}
                    out = json.dumps(resp, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                if self.path == "/v1/audio/speech/batch":
                    batch_calls.append(body)
                    results = [
                        {
                            "index": i,
                            "status": "success",
                            "audio_data": chunk_b64,
                            "format": "wav",
                            "media_type": "audio/wav",
                            "finish_reason": "stop",
                        }
                        for i in range(len(body["items"]))
                    ]
                    out = json.dumps({"id": "batch-multi", "results": results}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)
                    return
                self.send_response(404)
                self.end_headers()

        server = ThreadingHTTPServer(("127.0.0.1", 0), MockDramaAndLlmHandler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base_url = f"http://127.0.0.1:{server.server_port}"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "scene.txt"
            scene.write_text("林晚看着陈默问道：“你等很久了吧？”陈默微笑着答道：“刚泡好茶。”\n", encoding="utf-8")
            cast_file = root / "cast.json"
            cast_file.write_text(
                json.dumps({
                    "narrator": "narrator_voice",
                    "characters": {
                        "林晚": {"voice": "linwan_voice", "speed": 1.05},
                        "陈默": {"voice": "chenmo_voice", "speed": 0.94},
                    },
                }, ensure_ascii=False),
                encoding="utf-8",
            )
            out_dir = root / "multi-out"
            parser = svc._build_custom_parser()
            args = parser.parse_args([
                "narrate", str(scene),
                "--single-chapter",
                "--cast", str(cast_file),
                "--llm-url", f"{base_url}/v1",
                "--turn-pause-ms", "150",
                "-O", str(out_dir),
                "--url", base_url,
                "--api-token", "unit-token",
            ])
            rc = svc.cmd_narrate(args)
            self.assertEqual(rc, 0)
            self.assertEqual(len(llm_calls), 1)
            self.assertEqual(llm_calls[0]["model"], "mock-qwen-annotator")
            self.assertEqual(len(batch_calls), 1)

            items = batch_calls[0]["items"]
            self.assertEqual([it["voice"] for it in items], ["narrator_voice", "linwan_voice", "narrator_voice", "chenmo_voice"])
            self.assertAlmostEqual(items[1]["speed"], 1.05, places=2)
            self.assertAlmostEqual(items[3]["speed"], 0.94, places=2)
            self.assertTrue((out_dir / "annotated/001-scene.txt").is_file())
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["chapters"]["1"]["voice_chunks"],
                {"narrator_voice": 2, "linwan_voice": 1, "chenmo_voice": 1},
            )


if __name__ == "__main__":
    unittest.main()
