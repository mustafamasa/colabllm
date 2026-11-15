"""Unit tests for MiniMax-H3-Image service, network-friendly internal CLI, async jobs, dedup uploads & ASGI gateway."""
from __future__ import annotations

import ast
import asyncio
import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import threading
import unittest
from unittest import mock
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

from colabsvc import build_parser  # noqa: E402
import minimax_h3_image_middleware as mw  # noqa: E402


def _make_png(width: int = 64, height: int = 48) -> bytes:
    """Create a valid minimal RGB PNG with arbitrary dimensions using pure Python stdlib."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw_rows = b"".join(b"\x00" + (b"\x20\x60\xa0" * width) for _ in range(height))
    idat = zlib.compress(raw_rows, level=6)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


SAMPLE_PNG_4_3 = _make_png(128, 96)
SAMPLE_PNG_1_1 = _make_png(64, 64)


def _load_service_module():
    spec = importlib.util.spec_from_file_location(
        "minimax_h3_image_svc",
        ROOT / "src/colabllm/services/minimax-h3-image.py",
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


async def _invoke_asgi(
    app,
    method: str,
    path: str,
    *,
    body: bytes = b"",
    token: str | None = "test-secret",
    content_type: str = "application/json",
    extra_headers: list[tuple[str, str]] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    q_idx = path.find("?")
    if q_idx >= 0:
        clean_path, raw_qs = path[:q_idx], path[q_idx + 1 :].encode("utf-8")
    else:
        clean_path, raw_qs = path, b""

    hdrs: list[tuple[bytes, bytes]] = []
    if token is not None:
        hdrs.append((b"authorization", f"Bearer {token}".encode("utf-8")))
    if body:
        hdrs.append((b"content-type", content_type.encode("utf-8")))
        hdrs.append((b"content-length", str(len(body)).encode("ascii")))
    for k, v in extra_headers or []:
        hdrs.append((k.lower().encode("latin1"), v.encode("latin1")))

    sent = False

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    resp_status = 500
    resp_headers: dict[str, str] = {}
    resp_chunks: list[bytes] = []

    async def send(msg):
        nonlocal resp_status
        if msg["type"] == "http.response.start":
            resp_status = msg["status"]
            for k, v in msg.get("headers") or []:
                resp_headers[k.decode("latin1").lower()] = v.decode("latin1")
        elif msg["type"] == "http.response.body":
            resp_chunks.append(msg.get("body") or b"")

    scope = {
        "type": "http",
        "method": method.upper(),
        "path": clean_path,
        "raw_path": clean_path.encode("utf-8"),
        "query_string": raw_qs,
        "headers": hdrs,
    }
    await app(scope, receive, send)
    return resp_status, b"".join(resp_chunks), resp_headers


def _start_asgi_http_server(app) -> tuple[ThreadingHTTPServer, str, threading.Thread]:
    """Bridge ASGI `MiniMaxH3ImageGateway` onto a local `ThreadingHTTPServer` for end-to-end CLI testing."""

    class Handler(BaseHTTPRequestHandler):
        def _dispatch(self, method: str) -> None:
            length = int(self.headers.get("Content-Length") or "0")
            body = self.rfile.read(length) if length > 0 else b""
            hdr_pairs = [(k, v) for k, v in self.headers.items()]
            # Preserve exact Authorization header without duplicate
            auth_hdr = self.headers.get("Authorization")
            tok = None
            clean_extra: list[tuple[str, str]] = []
            for k, v in hdr_pairs:
                if k.lower() == "authorization" and v.startswith("Bearer "):
                    tok = v[7:]
                elif k.lower() not in ("content-type", "content-length"):
                    clean_extra.append((k, v))
            if auth_hdr and tok is None:
                clean_extra.append(("authorization", auth_hdr))
            status, out_body, out_hdrs = asyncio.run(
                _invoke_asgi(
                    app,
                    method,
                    self.path,
                    body=body,
                    token=tok,
                    content_type=self.headers.get("Content-Type") or "application/octet-stream",
                    extra_headers=clean_extra,
                )
            )
            self.send_response(status)
            for hk, hv in out_hdrs.items():
                if hk.lower() not in ("server", "date", "transfer-encoding"):
                    self.send_header(hk, hv)
            self.end_headers()
            if method != "HEAD" and out_body:
                self.wfile.write(out_body)

        def do_GET(self):
            self._dispatch("GET")

        def do_HEAD(self):
            self._dispatch("HEAD")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

        def do_OPTIONS(self):
            self._dispatch("OPTIONS")

        def log_message(self, fmt, *args):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host, port = httpd.server_address
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, f"http://{host}:{port}", t


class MiniMaxH3ImageUnitTest(unittest.TestCase):
    def test_notebook_code_cells_parse_and_include_g4_optimizations(self):
        nb = json.loads((ROOT / "src/colabllm/services/minimax-h3-image.ipynb").read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        self.assertEqual(len(code_cells), 6)
        full_src = []
        for idx, cell in enumerate(code_cells):
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            ast.parse(src, filename=f"minimax-h3-image.ipynb#cell-{idx}")
            full_src.append(src)
        joined = "\n".join(full_src)
        self.assertIn("H3_SETUP_OK = True", joined)
        self.assertIn("--minimax-h3-adaln-online", joined)
        self.assertIn("--component-quantizations.text_encoder", joined)
        self.assertIn("transformer_diffusers/*", joined)
        self.assertIn("_patch_local_model_index", joined)
        self.assertIn("POSIX_FADV_DONTNEED", joined)

    def test_contract_validation_aspect_presets_and_prompt_normalization(self):
        with mock.patch.dict("os.environ", {"CT_SERVICE_PORT": "", "CT_H3_SGLANG_PORT": ""}):
            importlib.reload(mw)
            self.assertEqual(mw.SERVICE_PORT, 8092)
            self.assertEqual(mw.SGLANG_PORT, 30000)
        importlib.reload(mw)

        # Prompt tag helper converts Qwen-style <image1> when <Picture is absent, preserves <Picture N>
        self.assertEqual(
            mw.normalize_edit_prompt_tags("Blend <image1> into <image 2>"),
            "Blend <Picture 1> into <Picture 2>",
        )
        self.assertEqual(
            mw.normalize_edit_prompt_tags("Keep <Picture 1> as is"),
            "Keep <Picture 1> as is",
        )

        # Official 4K & 2K presets and auto aspect snapping
        self.assertEqual(mw.resolve_dimensions(aspect="16:9", tier="4k"), (5472, 3072))
        self.assertEqual(mw.resolve_dimensions(aspect="4:3", tier="4k"), (4736, 3552))
        self.assertEqual(mw.resolve_dimensions(aspect="1:1", tier="2k"), (2048, 2048))
        # 4:3 reference image snaps to official 4736x3552 at 4k and 2368x1760 at 2k!
        self.assertEqual(
            mw.resolve_dimensions(size="auto", tier="4k", is_edit=True, ref_dimensions=(128, 96)),
            (4736, 3552),
        )
        self.assertEqual(
            mw.resolve_dimensions(size="auto", tier="2k", is_edit=True, ref_dimensions=(128, 96)),
            (2368, 1760),
        )

        store = mw.MiniMaxH3JobStore()
        # Empty negative_prompt is silently stripped; non-empty negative_prompt / wrong steps / wrong guidance rejected
        canon = store.prepare_canonical_job({"prompt": "sunset", "negative_prompt": "", "tier": "2k"})
        self.assertNotIn("negative_prompt", canon)
        self.assertEqual(canon["size"], "2048x2048")

        for invalid in (
            {"prompt": "x", "num_inference_steps": 20},
            {"prompt": "x", "guidance_scale": 4.5},
            {"prompt": "x", "negative_prompt": "ugly"},
            {"prompt": "x", "min_generation_area": 512},
            {"prompt": "x", "min_generation_area": 4194304.0},
            {"prompt": "x", "min_generation_area": True},
            {"prompt": "x", "lora": "style.safetensors"},
        ):
            with self.assertRaises(ValueError, msg=str(invalid)):
                store.prepare_canonical_job(invalid)

        # fast_preview lowers min_generation_area for 1K previews
        fast = store.prepare_canonical_job({"prompt": "sketch", "tier": "1k", "fast_preview": True})
        self.assertEqual(fast["size"], "1024x1024")
        self.assertEqual(fast["min_generation_area"], 1024 * 1024)

    def test_auth_fail_closed_and_sglang_file_url_loopback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sglang_out = root / "sglang-outputs"
            sglang_out.mkdir(parents=True)

            # Fake local SGLang server verifying response_format="url" + output_compression=0 + zero negative_prompt
            sglang_requests: list[dict] = []

            class FakeSGLangHandler(BaseHTTPRequestHandler):
                def do_GET(self):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"status":"ok"}')

                def do_POST(self):
                    ln = int(self.headers.get("Content-Length") or "0")
                    raw = self.rfile.read(ln)
                    ct = self.headers.get("Content-Type") or ""
                    if self.path == "/v1/images/generations":
                        obj = json.loads(raw.decode("utf-8"))
                        sglang_requests.append({"path": self.path, "json": obj})
                        assert "negative_prompt" not in obj
                        assert obj["response_format"] == "url"
                        assert obj["output_compression"] == 0
                    else:
                        parsed_mp = mw._parse_multipart_edits_request(ct, raw)
                        sglang_requests.append({"path": self.path, "form": parsed_mp})
                        assert "negative_prompt" not in parsed_mp
                        assert len(parsed_mp["_multipart_files"]) >= 1
                    out_file = sglang_out / f"gen_{len(sglang_requests)}.png"
                    out_file.write_bytes(SAMPLE_PNG_4_3)
                    resp = {
                        "data": [
                            {
                                "url": f"/v1/images/req_{len(sglang_requests)}/content",
                                "file_path": str(out_file),
                            }
                        ]
                    }
                    raw_resp = json.dumps(resp).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw_resp)))
                    self.end_headers()
                    self.wfile.write(raw_resp)

                def log_message(self, fmt, *args):
                    return

            sglang_srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeSGLangHandler)
            sglang_url = f"http://127.0.0.1:{sglang_srv.server_address[1]}"
            threading.Thread(target=sglang_srv.serve_forever, daemon=True).start()

            try:
                mydrive = root / "drive/MyDrive"
                data_dir = mydrive / "colab-llm/minimax-h3-image"
                (data_dir / "inputs").mkdir(parents=True)
                (data_dir / "outputs").mkdir(parents=True)
                (data_dir / "inputs/ref1.png").write_bytes(SAMPLE_PNG_4_3)

                store = mw.MiniMaxH3JobStore(
                    jobs_dir=root / "jobs",
                    uploads_dir=root / "uploads",
                    data_dir=data_dir,
                    drive_outputs_dir=data_dir / "outputs",
                    mydrive_root=mydrive,
                    drive_root=root / "drive",
                    sglang_caller=lambda task, **kw: mw.default_sglang_caller(
                        task,
                        sglang_base_url=sglang_url,
                        sglang_output_dir=sglang_out,
                        **kw,
                    ),
                )
                gw = mw.MiniMaxH3ImageGateway(
                    job_store=store,
                    upstream_health_fn=lambda: mw.check_sglang_upstream_live(sglang_url),
                )

                # Fail-closed when API_TOKEN is empty
                with mock.patch.object(mw, "API_TOKEN", ""):
                    code_empty, _, _ = asyncio.run(_invoke_asgi(gw, "GET", "/v1/models", token="anything"))
                    self.assertEqual(code_empty, 401)

                with mock.patch.object(mw, "API_TOKEN", "secret-123"):
                    # Missing / wrong token -> 401 on EVERY route including /health
                    for route in ("/health", "/v1/models", "/api/ct/jobs", "/api/ct/info"):
                        c_none, _, _ = asyncio.run(_invoke_asgi(gw, "GET", route, token=None))
                        c_bad, _, _ = asyncio.run(_invoke_asgi(gw, "GET", route, token="wrong"))
                        self.assertEqual(c_none, 401, route)
                        self.assertEqual(c_bad, 401, route)

                    # OPTIONS preflight -> 204 with CORS headers
                    c_opt, _, h_opt = asyncio.run(_invoke_asgi(gw, "OPTIONS", "/api/ct/jobs", token=None))
                    self.assertEqual(c_opt, 204)
                    self.assertEqual(h_opt.get("access-control-allow-origin"), "*")

                    # Authenticated /health -> 200
                    c_ok, _, _ = asyncio.run(_invoke_asgi(gw, "GET", "/health", token="secret-123"))
                    self.assertEqual(c_ok, 200)

                    # End-to-end CLI test against live HTTP gateway:
                    # 1. Local edit uploads raw binary once, 2nd edit hits 0-byte SHA-256 dedup cache!
                    svc = _load_service_module()
                    gw_httpd, gw_url, _ = _start_asgi_http_server(gw)
                    try:
                        local_ref = root / "local_person.png"
                        local_ref.write_bytes(SAMPLE_PNG_1_1)
                        uri1, hit1 = svc._upload_local_reference_dedup(gw_url, "secret-123", local_ref)
                        uri2, hit2 = svc._upload_local_reference_dedup(gw_url, "secret-123", local_ref)
                        self.assertEqual(uri1, uri2)
                        self.assertFalse(hit1)
                        self.assertTrue(hit2)

                        # 2. CLI `edit` via internal `/api/ct/jobs` + binary download + Drive save
                        parser = build_parser(svc.SERVICE, svc.configure_cli)
                        local_out = root / "edited_out.png"
                        edit_args = parser.parse_args([
                            "edit",
                            "Put <image1> beside <image2>",
                            "-r", str(local_ref),
                            "--drive-image", "inputs/ref1.png",
                            "--tier", "2k",
                            "--save-drive", "outputs/cloud_edit.png",
                            "-o", str(local_out),
                            "--url", gw_url,
                            "--api-token", "secret-123",
                        ])
                        svc.cmd_edit(edit_args)
                        self.assertTrue(local_out.is_file())
                        self.assertTrue((data_dir / "outputs/cloud_edit.png").is_file())
                        # SGLang scratch file must have been unlinked immediately after handoff!
                        self.assertEqual(list(sglang_out.glob("*.png")), [])
                        # Prompt <image1> was normalized to <Picture 1>
                        self.assertIn("<Picture 1>", sglang_requests[-1]["form"]["prompt"])

                        # 3. Verify HTTP Range 206 partial content on /api/ct/jobs/<id>/content
                        jobs_list = store.list_jobs()
                        jid = jobs_list[0]["job_id"]
                        full_bytes = local_out.read_bytes()
                        c_rng, part_bytes, h_rng = asyncio.run(
                            _invoke_asgi(
                                gw,
                                "GET",
                                f"/api/ct/jobs/{jid}/content?index=0",
                                token="secret-123",
                                extra_headers=[("Range", "bytes=10-29")],
                            )
                        )
                        self.assertEqual(c_rng, 206)
                        self.assertEqual(part_bytes, full_bytes[10:30])
                        self.assertIn("bytes 10-29/", h_rng.get("content-range", ""))

                        # 4. OpenAI compatibility route (/v1/images/generations) and /api/ct/info
                        c_inf, b_inf, _ = asyncio.run(
                            _invoke_asgi(gw, "GET", "/api/ct/info", token="secret-123")
                        )
                        self.assertEqual(c_inf, 200)
                        inf_obj = json.loads(b_inf.decode("utf-8"))
                        self.assertEqual(inf_obj["model_repo"], "MiniMaxAI/MiniMax-H3-Image")
                        self.assertEqual(inf_obj["pdd_steps"], 6)
                        self.assertTrue(inf_obj["drive_inputs_dir"].endswith("inputs"))
                        self.assertTrue(inf_obj["drive_outputs_dir"].endswith("outputs"))

                        c_oai, b_oai, _ = asyncio.run(
                            _invoke_asgi(
                                gw,
                                "POST",
                                "/v1/images/generations",
                                body=json.dumps({
                                    "prompt": "A mountain cabin",
                                    "size": "2048x2048",
                                    "quality": "hd",
                                    "seed": 7,
                                }).encode("utf-8"),
                                token="secret-123",
                            )
                        )
                        self.assertEqual(c_oai, 200)
                        oai_obj = json.loads(b_oai.decode("utf-8"))
                        self.assertTrue(base64.b64decode(oai_obj["data"][0]["b64_json"]).startswith(b"\x89PNG"))
                    finally:
                        gw_httpd.shutdown()
                        gw_httpd.server_close()
            finally:
                sglang_srv.shutdown()
                sglang_srv.server_close()


if __name__ == "__main__":
    unittest.main()
