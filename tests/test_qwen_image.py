"""Unit tests for Qwen-Image-2.1 service, CLI, Google Drive I/O, RGBA/edit routing, and ASGI middleware."""
from __future__ import annotations

import asyncio
import ast
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

import colab_asgi_common as common  # noqa: E402
from cli_support import UsageError  # noqa: E402
from colabsvc import build_parser  # noqa: E402
import qwen_image_middleware as mw  # noqa: E402

TINY_RGBA_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def _load_service_module():
    spec = importlib.util.spec_from_file_location("qwen_image_svc", ROOT / "src/colabllm/services/qwen-image.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class QwenImageNotebookAndMiddlewareTest(unittest.TestCase):
    def test_notebook_code_cells_parse_cleanly(self):
        nb = json.loads((ROOT / "src/colabllm/services/qwen-image.ipynb").read_text(encoding="utf-8"))
        code_cells = [c for c in nb["cells"] if c.get("cell_type") == "code"]
        self.assertEqual(len(code_cells), 6)
        full_src = []
        for idx, cell in enumerate(code_cells):
            src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            ast.parse(src, filename=f"qwen-image.ipynb#cell-{idx}")
            full_src.append(src)
        joined = "\n".join(full_src)
        self.assertIn("QI_SETUP_OK = True", joined)
        self.assertIn('"--vae-use-tiling"', joined)
        self.assertIn('"--step-execution"', joined)
        self.assertIn('"--diffusion-quantization-config"', joined)

    def test_sampling_defaults_aliases_and_32px_alignment(self):
        # Empty env vars forwarded by SERVICE.env (e.g. CT_QWEN_IMAGE_MAX_NUM_SEQS='') must never fail module import
        with mock.patch.dict("os.environ", {"CT_QWEN_IMAGE_MAX_NUM_SEQS": "", "CT_SERVICE_PORT": ""}):
            importlib.reload(mw)
            self.assertEqual(mw.MAX_NUM_SEQS, 4)
            self.assertEqual(mw.SERVICE_PORT, 8091)
        with mock.patch.dict("os.environ", {"CT_QWEN_IMAGE_MAX_NUM_SEQS": "", "CT_QWEN_IMAGE_STEP_EXECUTION": "1"}):
            importlib.reload(mw)
            self.assertEqual(mw.MAX_NUM_SEQS, 8)
        importlib.reload(mw)

        # 1. Default T2I injects 40 steps, true_cfg_scale=1.0, and normalizes aliases (including gpt-image-2)
        for alias in ("", "qwen-image", "qwen-image-2.1", "gpt-image-2"):
            raw, _, _, color_fmt = mw._prepare_generation_request(
                json.dumps({"model": alias, "prompt": "teapot", "size": "1280x720"}).encode("utf-8")
            )
            parsed = json.loads(raw)
            self.assertEqual(parsed["model"], "Qwen/Qwen-Image-2.1")
            self.assertEqual(parsed["num_inference_steps"], 40)
            self.assertEqual(parsed["true_cfg_scale"], 1.0)
            self.assertEqual(parsed["size"], "1280x704")
            self.assertIsNone(color_fmt)

        # 2. Non-empty negative_prompt automatically engages true_cfg_scale=4.0
        raw_neg, _, _, _ = mw._prepare_generation_request(
            json.dumps({"prompt": "portrait", "negative_prompt": "blurry, deformed", "size": "auto"}).encode("utf-8")
        )
        parsed_neg = json.loads(raw_neg)
        self.assertEqual(parsed_neg["true_cfg_scale"], 4.0)
        self.assertEqual(parsed_neg["size"], "1024x1024")

        # 3. CORS headers inject cleanly without duplicating
        hdrs = mw._inject_cors_headers([(b"content-type", b"application/json")])
        hdr_names = {k for k, _ in hdrs}
        self.assertIn(b"access-control-allow-origin", hdr_names)
        self.assertIn(b"access-control-allow-methods", hdr_names)

    def test_four_image_cap_drive_io_and_inprocess_asgi_edit(self):
        # 5 reference images in /v1/chat/completions must fail with ValueError
        five_imgs = [
            {"type": "image_url", "image_url": {"url": f"https://example.com/{i}.png"}}
            for i in range(5)
        ]
        bad_chat = {"messages": [{"role": "user", "content": [{"type": "text", "text": "combine"}, *five_imgs]}]}
        with self.assertRaises(ValueError):
            mw._rewrite_drive_urls_in_chat_body(json.dumps(bad_chat).encode("utf-8"))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mydrive = root / "drive/MyDrive"
            data_dir = mydrive / "colab-llm/qwen-image"
            inputs_dir = data_dir / "inputs"
            outputs_dir = data_dir / "outputs"
            loras_dir = data_dir / "loras"
            local_loras = root / "local-loras"
            for d in (inputs_dir, outputs_dir, loras_dir, local_loras):
                d.mkdir(parents=True)

            (inputs_dir / "person.png").write_bytes(TINY_RGBA_PNG)
            (inputs_dir / "pose.png").write_bytes(TINY_RGBA_PNG)
            (loras_dir / "anime.safetensors").write_bytes(b"SAFETENSORS_DEMO")

            # Fake upstream ASGI app verifying direct in-process dispatch (zero loopback HTTP recursion!)
            upstream_calls: list[dict] = []

            async def fake_upstream_app(scope, receive, send):
                chunks = []
                while True:
                    msg = await receive()
                    if msg["type"] == "http.request":
                        chunks.append(msg.get("body", b""))
                        if not msg.get("more_body", False):
                            break
                raw = b"".join(chunks)
                hdrs = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
                ct = hdrs.get("content-type", "")
                if scope["path"] == "/v1/images/edits":
                    parsed = mw._parse_multipart_edit_body(ct, raw)
                    upstream_calls.append({"path": scope["path"], "fields": parsed})
                    resp_body = json.dumps({
                        "created": 123,
                        "size": "1024x1024",
                        "output_format": "png",
                        "data": [{"b64_json": base64.b64encode(TINY_RGBA_PNG).decode("ascii")}],
                    }).encode("utf-8")
                else:
                    parsed = json.loads(raw.decode("utf-8"))
                    upstream_calls.append({"path": scope["path"], "fields": parsed})
                    b64_u = f"data:image/png;base64,{base64.b64encode(TINY_RGBA_PNG).decode('ascii')}"
                    resp_body = json.dumps({
                        "choices": [{"message": {"content": [{"type": "image_url", "image_url": {"url": b64_u}}]}}]
                    }).encode("utf-8")
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(resp_body)).encode())],
                })
                await send({"type": "http.response.body", "body": resp_body, "more_body": False})

            async def invoke_asgi(app, path: str, body: bytes, content_type: str = "application/json"):
                sent = False

                async def receive():
                    nonlocal sent
                    if not sent:
                        sent = True
                        return {"type": "http.request", "body": body, "more_body": False}
                    return {"type": "http.disconnect"}

                out_status = 500
                out_chunks: list[bytes] = []

                async def send(msg):
                    nonlocal out_status
                    if msg["type"] == "http.response.start":
                        out_status = msg["status"]
                    elif msg["type"] == "http.response.body":
                        out_chunks.append(msg.get("body", b""))

                scope = {
                    "type": "http",
                    "method": "POST",
                    "path": path,
                    "raw_path": path.encode("utf-8"),
                    "root_path": "",
                    "headers": [
                        (b"authorization", b"Bearer test-tok"),
                        (b"content-type", content_type.encode("utf-8")),
                        (b"content-length", str(len(body)).encode("ascii")),
                    ],
                }
                await app(scope, receive, send)
                return out_status, json.loads(b"".join(out_chunks).decode("utf-8"))

            with (
                mock.patch.object(mw, "API_TOKEN", "test-tok"),
                mock.patch.object(mw, "DATA_DIR", data_dir),
                mock.patch.object(mw, "DRIVE_OUTPUTS_DIR", outputs_dir),
                mock.patch.object(mw, "LOCAL_LORA_DIR", local_loras),
                mock.patch.object(mw, "MYDRIVE_ROOT", mydrive),
                mock.patch.object(mw, "DRIVE_ROOT", root / "drive"),
            ):
                app = mw.QwenImageMiddleware(fake_upstream_app)
                status, result = asyncio.run(
                    invoke_asgi(
                        app,
                        "/v1/images/edits",
                        json.dumps({
                            "model": "gpt-image-2",
                            "prompt": "Make the person in image 1 <image1> strike the pose from image 2 <image2>",
                            "images": ["person.png", "drive://inputs/pose.png"],
                            "size": "auto",
                            "lora": "drive://loras/anime.safetensors",
                            "save_to_drive": "outputs/pose_out.png",
                        }).encode("utf-8"),
                    )
                )
                self.assertEqual(status, 200)
                self.assertEqual(len(upstream_calls), 1)
                up_fields = upstream_calls[0]["fields"]
                self.assertEqual(upstream_calls[0]["path"], "/v1/images/edits")
                self.assertEqual(up_fields["model"], "Qwen/Qwen-Image-2.1")
                self.assertEqual(up_fields["size"], "auto")
                self.assertEqual(up_fields["num_inference_steps"], 40)
                self.assertEqual(up_fields["true_cfg_scale"], 1.0)
                self.assertEqual(len(up_fields["_ordered_inputs"]), 2)
                self.assertTrue((outputs_dir / "pose_out.png").is_file())
                self.assertEqual(result["rel_drive_paths"], ["colab-llm/qwen-image/outputs/pose_out.png"])

                # Mixed HTTP URL + Drive file routes cleanly to /v1/chat/completions and converts back to ImagesResponse
                status_mix, result_mix = asyncio.run(
                    invoke_asgi(
                        app,
                        "/v1/images/edits",
                        json.dumps({
                            "prompt": "combine",
                            "images": ["https://example.com/a.png", "person.png"],
                        }).encode("utf-8"),
                    )
                )
                self.assertEqual(status_mix, 200)
                self.assertEqual(upstream_calls[1]["path"], "/v1/chat/completions")
                self.assertEqual(len(result_mix["data"]), 1)
                self.assertTrue(result_mix["data"][0]["b64_json"])

                # 5 images must be rejected with HTTP 400 before calling upstream
                status_bad, _ = asyncio.run(
                    invoke_asgi(
                        app,
                        "/v1/images/edits",
                        json.dumps({"prompt": "too many", "images": ["person.png"] * 5}).encode("utf-8"),
                    )
                )
                self.assertEqual(status_bad, 400)
                self.assertEqual(len(upstream_calls), 2)


class QwenImageCliTest(unittest.TestCase):
    def setUp(self):
        self.svc = _load_service_module()
        self.requests_log: list[tuple[str, str, dict, dict]] = []
        outer = self

        class FakeHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):  # noqa: A002
                return

            def do_GET(self):
                hdrs = {k.lower(): v for k, v in self.headers.items()}
                outer.requests_log.append(("GET", self.path, hdrs, {}))
                if hdrs.get("authorization") != "Bearer test-secret":
                    self._send(401, {"error": "unauthorized"})
                    return
                if self.path.startswith("/api/ct/drive-list"):
                    self._send(200, {
                        "drive_mounted": True,
                        "kind": "images",
                        "count": 1,
                        "files": [{"rel_path": "colab-llm/qwen-image/inputs/ref.png", "type": "image", "size_mb": 0.25}],
                    })
                elif self.path == "/api/ct/info":
                    self._send(200, {
                        "service": "qwen-image",
                        "model_repo": "Qwen/Qwen-Image-2.1",
                        "served_models": ["Qwen/Qwen-Image-2.1", "qwen-image-2.1"],
                        "quantization": "none",
                        "prefix_kv_cache_dtype": "auto",
                        "step_execution": False,
                        "max_num_seqs": 4,
                        "vae_use_tiling": True,
                        "default_steps": 40,
                        "default_true_cfg_scale": 1.0,
                        "default_true_cfg_scale_with_negative_prompt": 4.0,
                        "rgba_vae": True,
                        "max_reference_images": 4,
                        "drive_inputs_dir": "/content/drive/MyDrive/colab-llm/qwen-image/inputs",
                        "drive_outputs_dir": "/content/drive/MyDrive/colab-llm/qwen-image/outputs",
                        "drive_loras_dir": "/content/drive/MyDrive/colab-llm/qwen-image/loras",
                    })
                elif self.path == "/v1/models":
                    self._send(200, {"data": [{"id": "Qwen/Qwen-Image-2.1"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                hdrs = {k.lower(): v for k, v in self.headers.items()}
                length = int(hdrs.get("content-length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                outer.requests_log.append(("POST", self.path, hdrs, body))
                if hdrs.get("authorization") != "Bearer test-secret":
                    self._send(401, {"error": "unauthorized"})
                    return
                b64 = base64.b64encode(TINY_RGBA_PNG).decode("ascii")
                if self.path in ("/v1/images/generations", "/v1/images/edits", "/api/ct/edit"):
                    drive_only = bool(body.get("drive_only"))
                    save_spec = body.get("save_to_drive")
                    item = {} if drive_only else {"b64_json": b64}
                    resp = {"created": 1, "data": [item]}
                    extra = {}
                    if save_spec:
                        saved = f"/content/drive/MyDrive/colab-llm/qwen-image/outputs/{Path(str(save_spec)).name or 'out.png'}"
                        resp["saved_drive_paths"] = [saved]
                        resp["rel_drive_paths"] = [f"colab-llm/qwen-image/outputs/{Path(saved).name}"]
                        extra["X-Saved-Drive-Path"] = saved
                    self._send(200, resp, extra)
                elif self.path == "/v1/chat/completions":
                    data_uri = f"data:image/png;base64,{b64}"
                    resp = {"choices": [{"message": {"content": [{"type": "image_url", "image_url": {"url": data_uri}}]}}]}
                    if body.get("save_to_drive"):
                        resp["rel_drive_paths"] = [str(body["save_to_drive"])]
                    self._send(200, resp)
                else:
                    self._send(404, {"error": "not found"})

            def _send(self, code: int, payload: dict, extra_headers: dict | None = None):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in (extra_headers or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def _parse(self, argv: list[str]):
        parser = build_parser(self.svc.SERVICE, self.svc.configure_cli)
        return parser.parse_args(argv)

    def test_prepare_up_resets_between_calls(self):
        self.assertEqual(self.svc.SERVICE.gpu, "G4")
        for gpu_tier in ("G4", "H100", "A100"):
            parsed_gpu = self._parse(["up", "--gpu", gpu_tier])
            self.assertEqual(parsed_gpu.gpu, gpu_tier)

        args1 = self._parse([
            "up", "--quantization", "fp8-all", "--prefix-kv-dtype", "fp8_v",
            "--step-execution", "--max-num-seqs", "8", "--no-vae-tiling",
        ])
        self.svc.prepare_up(args1)
        self.assertEqual(self.svc.SERVICE.env["CT_QWEN_IMAGE_QUANTIZATION"], "fp8-all")
        self.assertEqual(self.svc.SERVICE.env["CT_QWEN_IMAGE_PREFIX_KV_DTYPE"], "fp8_v")
        self.assertEqual(self.svc.SERVICE.env["CT_QWEN_IMAGE_STEP_EXECUTION"], "1")
        self.assertEqual(self.svc.SERVICE.env["CT_QWEN_IMAGE_MAX_NUM_SEQS"], "8")
        self.assertEqual(self.svc.SERVICE.env["CT_QWEN_IMAGE_VAE_TILING"], "0")

        args2 = self._parse(["up"])
        self.svc.prepare_up(args2)
        self.assertEqual(self.svc.SERVICE.env["CT_QWEN_IMAGE_QUANTIZATION"], "none")
        self.assertEqual(self.svc.SERVICE.env["CT_QWEN_IMAGE_PREFIX_KV_DTYPE"], "auto")
        self.assertEqual(self.svc.SERVICE.env["CT_QWEN_IMAGE_STEP_EXECUTION"], "0")
        self.assertEqual(self.svc.SERVICE.env["CT_QWEN_IMAGE_VAE_TILING"], "1")

    def test_generate_and_edit_rgba_and_drive_io(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_ref = tmp_path / "local_ref.png"
            local_ref.write_bytes(TINY_RGBA_PNG)
            local_out = tmp_path / "rgba_out.png"

            # 1. T2I with negative_prompt automatically switches true_cfg_scale to 4.0
            args_gen = self._parse([
                "generate", "çini çaydanlık",
                "--negative-prompt", "bulanık",
                "--save-drive", "outputs/teapot.png",
                "--url", self.base_url, "--api-token", "test-secret",
            ])
            self.svc.cmd_generate(args_gen)
            method, path, _, body = self.requests_log[-1]
            self.assertEqual((method, path), ("POST", "/v1/images/generations"))
            self.assertEqual(body["num_inference_steps"], 40)
            self.assertEqual(body["true_cfg_scale"], 4.0)
            self.assertTrue(body["drive_only"])

            # 2. Multi-ref RGBA edit preserving exact <image1>, <image2> order + Drive & local output
            args_edit = self._parse([
                "edit", "Image 1 <image1> wears the clothes from image 2 <image2>, transparent background",
                "--drive-image", "inputs/person.jpg",
                "--image", str(local_ref),
                "--rgba",
                "--save-drive", "outputs/garment.png",
                "-o", str(local_out),
                "--url", self.base_url, "--api-token", "test-secret",
            ])
            self.svc.cmd_edit(args_edit)
            method, path, _, body = self.requests_log[-1]
            self.assertEqual((method, path), ("POST", "/v1/images/edits"))
            self.assertEqual(body["size"], "auto")
            self.assertEqual(body["true_cfg_scale"], 1.0)
            self.assertEqual(body["color_format"], "RGBA")
            self.assertEqual(body["background"], "transparent")
            self.assertEqual(body["images"][0], "drive://inputs/person.jpg")
            self.assertTrue(body["images"][1].startswith("data:image/png;base64,"))
            self.assertTrue(local_out.is_file())

            # 3. Reject >4 reference images
            args_bad = self._parse([
                "edit", "too many",
                "--drive-image", "a.png", "b.png", "c.png", "d.png", "e.png",
                "--url", self.base_url, "--api-token", "test-secret",
            ])
            with self.assertRaises(UsageError):
                self.svc.cmd_edit(args_bad)

            # 4. ls-drive and models JSON output
            for argv, fn in (
                (["ls-drive", "--json", "--url", self.base_url, "--api-token", "test-secret"], self.svc.cmd_ls_drive),
                (["models", "--json", "--url", self.base_url, "--api-token", "test-secret"], self.svc.cmd_models),
            ):
                buf = io.StringIO()
                with mock.patch("sys.stdout", buf):
                    fn(self._parse(argv))
                json.loads(buf.getvalue())


if __name__ == "__main__":
    unittest.main()
