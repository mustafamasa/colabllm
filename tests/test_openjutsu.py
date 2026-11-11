"""Unit tests for the Openjutsu Colab runtime helper, notebook, and service CLI."""
from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

from cli_support import CliError, UsageError  # noqa: E402
import colabsvc  # noqa: E402
import openjutsu_runtime as runtime  # noqa: E402


FAKE_MP4 = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isomfakevideobytes"
FAKE_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRfakeimagebytes"


def load_openjutsu_service():
    path = ROOT / "src/colabllm/services/openjutsu.py"
    spec = importlib.util.spec_from_file_location("openjutsu_service_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OpenjutsuRuntimeTest(unittest.TestCase):
    def test_validate_model_destination_confines_to_models_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            comfy_dir = Path(tmp)
            dest = runtime.validate_model_destination("ultralytics/bbox", "face_yolov8m.pt", comfy_dir)
            self.assertEqual(dest, (comfy_dir / "models/ultralytics/bbox/face_yolov8m.pt").resolve())

            for directory, name in (
                ("", "x.safetensors"),
                ("loras", ""),
                ("../custom_nodes", "evil.py"),
                ("loras/../../custom_nodes", "evil.py"),
                ("loras", "../evil.safetensors"),
                ("loras", "sub/evil.safetensors"),
                ("loras", "sub\\evil.safetensors"),
                ("loras", "bad\x00.safetensors"),
            ):
                with self.assertRaises(ValueError, msg=(directory, name)):
                    runtime.validate_model_destination(directory, name, comfy_dir)

    def test_model_profiles_and_setup_payloads(self):
        core = runtime.models_for_profile("core")
        full = runtime.models_for_profile("full")
        self.assertEqual(len(core), 7)
        self.assertEqual(len(full), 16)
        with self.assertRaises(ValueError):
            runtime.models_for_profile("unknown")

        core_payloads = runtime.setup_check_payloads("core")
        full_payloads = runtime.setup_check_payloads("full")
        self.assertNotIn("face", core_payloads)
        self.assertIn("face", full_payloads)
        self.assertIn("simple", core_payloads)

    def test_normalize_linux_workflows_fixes_backslash_detectors_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp)
            wf_dir = app_dir / "workflows"
            wf_dir.mkdir(parents=True)
            face_wf = wf_dir / "face.api.json"
            face_wf.write_text(
                json.dumps(
                    {
                        "6": {
                            "class_type": "H3FaceTrackCrop",
                            "inputs": {
                                "detector": "bbox\\face_yolov8m.pt",
                                "fallback_detector": "segm\\person_yolov8m-seg.pt",
                            },
                        },
                        "9": {
                            "class_type": "OtherNode",
                            "inputs": {"text": "keep\\literal"},
                        },
                    }
                ),
                encoding="utf-8",
            )
            changed = runtime.normalize_linux_workflows(app_dir)
            self.assertEqual(changed, ["workflows/face.api.json"])
            parsed = json.loads(face_wf.read_text(encoding="utf-8"))
            self.assertEqual(parsed["6"]["inputs"]["detector"], "bbox/face_yolov8m.pt")
            self.assertEqual(parsed["6"]["inputs"]["fallback_detector"], "segm/person_yolov8m-seg.pt")
            self.assertEqual(parsed["9"]["inputs"]["text"], "keep\\literal")
            self.assertEqual(runtime.normalize_linux_workflows(app_dir), [])

    def test_render_extra_model_paths_registers_ultralytics_and_sams(self):
        yaml_text = runtime.render_extra_model_paths(Path("/content/ComfyUI"))
        self.assertIn("ultralytics: models/ultralytics", yaml_text)
        self.assertIn("ultralytics_bbox: models/ultralytics/bbox", yaml_text)
        self.assertIn("ultralytics_segm: models/ultralytics/segm", yaml_text)
        self.assertIn("sams: models/sams", yaml_text)

    def test_select_comfy_vram_flags_handles_auto_and_explicit_modes(self):
        self.assertEqual(runtime.select_comfy_vram_flags(97887, "auto"), ["--highvram"])
        self.assertEqual(runtime.select_comfy_vram_flags(24000, "auto"), [])
        self.assertEqual(runtime.select_comfy_vram_flags(24000, "highvram"), ["--highvram"])
        self.assertEqual(runtime.select_comfy_vram_flags(97887, "dynamic"), [])
        self.assertEqual(runtime.select_comfy_vram_flags(97887, "gpu-only"), ["--gpu-only"])
        with self.assertRaises(ValueError):
            runtime.select_comfy_vram_flags(97887, "invalid")

    def test_torch_supports_sm120_parses_version_file_without_importing_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            vf = Path(tmp) / "version.py"
            vf.write_text("__version__ = '2.6.0+cu126'\ncuda = '12.6'\n", encoding="utf-8")
            self.assertFalse(runtime.torch_supports_sm120(vf))
            vf.write_text("from typing import Optional\ncuda: Optional[str] = '12.8'\n", encoding="utf-8")
            self.assertTrue(runtime.torch_supports_sm120(vf))
            vf.write_text("cuda = '13.0'\n", encoding="utf-8")
            self.assertTrue(runtime.torch_supports_sm120(vf))
            vf.write_text("cuda = None\n", encoding="utf-8")
            self.assertFalse(runtime.torch_supports_sm120(vf))

    def test_configure_chains_storage_handles_local_and_drive_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            data_dir = Path(tmp) / "data"
            self.assertEqual(runtime.configure_chains_storage(work_dir, data_dir, "ssd"), "ssd-native")
            self.assertTrue((data_dir / "output" / "h3_chains").is_dir())
            self.assertEqual(runtime.configure_chains_storage(work_dir, data_dir, "drive"), "drive")
            with self.assertRaises(ValueError):
                runtime.configure_chains_storage(work_dir, data_dir, "bad-mode")

    def test_parse_hf_resolve_url_and_parallel_range_download(self):
        self.assertEqual(
            runtime.parse_hf_resolve_url(
                "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors?download=true"
            ),
            ("Comfy-Org/MiniMax-H3", "main", "vae/minimax_h3_video_vae_int8_convrot.safetensors"),
        )
        self.assertIsNone(runtime.parse_hf_resolve_url("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"))

        payload = bytes((i * 31) % 251 for i in range(65536))
        range_hits: list[str] = []

        class RangeHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def do_GET(self):
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    range_hits.append(rng)
                    start_s, end_s = rng[6:].split("-", 1)
                    start = int(start_s)
                    end = int(end_s) if end_s else len(payload) - 1
                    chunk = payload[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/weights.safetensors"
            with tempfile.TemporaryDirectory() as tmp:
                dest1 = Path(tmp) / "parallel.safetensors"
                sz1 = runtime._download_via_http(
                    url, dest1, {"User-Agent": "Test"}, "parallel.safetensors",
                    parallel_threshold=4096, range_workers=4,
                )
                self.assertEqual(sz1, len(payload))
                self.assertEqual(dest1.read_bytes(), payload)
                self.assertGreaterEqual(len(range_hits), 5)  # 1 probe + 4 slice requests

                # Verify single-stream byte-level Range resume when a partial file already exists
                dest2 = Path(tmp) / "resumed.safetensors"
                partial2 = dest2.with_name(f".{dest2.name}.{runtime.os.getpid()}.partial")
                partial2.write_bytes(payload[:20000])
                range_hits.clear()
                sz2 = runtime._download_via_http(
                    url, dest2, {"User-Agent": "Test"}, "resumed.safetensors",
                    parallel_threshold=10**9, range_workers=1,
                )
                self.assertEqual(sz2, len(payload))
                self.assertEqual(dest2.read_bytes(), payload)
                self.assertEqual(range_hits, ["bytes=20000-"])
        finally:
            server.shutdown()
            server.server_close()


class OpenjutsuServiceCliTest(unittest.TestCase):
    def test_service_metadata_and_prepare_up_idempotency(self):
        svc_mod = load_openjutsu_service()
        self.assertEqual(svc_mod.SERVICE.name, "openjutsu")
        self.assertEqual(svc_mod.SERVICE.gpu, "G4")
        self.assertEqual(svc_mod.SERVICE.auth, "caddy")
        self.assertEqual(svc_mod.SERVICE.service_port, 3000)
        self.assertEqual(svc_mod.SERVICE.env["CT_OPENJUTSU_CHAINS_STORAGE"], "ssd")
        self.assertEqual(svc_mod.SERVICE.env["CT_COMFY_VRAM_MODE"], "auto")

        svc_mod.prepare_up(types.SimpleNamespace(profile="core", chains_storage="drive", vram_mode="dynamic"))
        self.assertEqual(svc_mod.SERVICE.env["CT_OPENJUTSU_PROFILE"], "core")
        self.assertEqual(svc_mod.SERVICE.env["CT_OPENJUTSU_CHAINS_STORAGE"], "drive")
        self.assertEqual(svc_mod.SERVICE.env["CT_COMFY_VRAM_MODE"], "dynamic")
        svc_mod.prepare_up(types.SimpleNamespace(profile="full", chains_storage=None, vram_mode=None))
        self.assertEqual(svc_mod.SERVICE.env["CT_OPENJUTSU_PROFILE"], "full")
        self.assertEqual(svc_mod.SERVICE.env["CT_OPENJUTSU_CHAINS_STORAGE"], "ssd")
        self.assertEqual(svc_mod.SERVICE.env["CT_COMFY_VRAM_MODE"], "auto")

    def test_cli_parser_supports_profile_storage_vram_and_business_commands(self):
        svc_mod = load_openjutsu_service()
        parser = colabsvc.build_parser(svc_mod.SERVICE, svc_mod.configure_cli)
        up_args = parser.parse_args(
            ["up", "--profile", "core", "--chains-storage", "drive", "--vram-mode", "highvram", "--dry-run"]
        )
        self.assertEqual(up_args.profile, "core")
        self.assertEqual(up_args.chains_storage, "drive")
        self.assertEqual(up_args.vram_mode, "highvram")
        self.assertTrue(up_args.dry_run)

        chk_args = parser.parse_args(["check", "--mode", "simple", "--timeout", "45"])
        self.assertEqual(chk_args.cmd, "check")
        self.assertEqual(chk_args.mode, "simple")
        self.assertEqual(chk_args.timeout, 45)

        swap_args = parser.parse_args(
            ["swap", "--video", "a.mp4", "-p", "p1.png", "-p", "p2.png", "--clip-seconds", "9.5", "-o", "out.mp4"]
        )
        self.assertEqual(swap_args.cmd, "swap")
        self.assertEqual(swap_args.pictures, ["p1.png", "p2.png"])
        self.assertEqual(swap_args.clip_seconds, 9.5)
        self.assertEqual(swap_args.output, "out.mp4")

    def test_build_submission_validates_files_modes_and_parameters(self):
        svc_mod = load_openjutsu_service()
        parser = svc_mod.custom_parser()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vid = root / "in.mp4"
            vid.write_bytes(FAKE_MP4)
            p1 = root / "p1.png"
            p1.write_bytes(FAKE_PNG)
            p2 = root / "p2.png"
            p2.write_bytes(FAKE_PNG)
            bad = root / "bad.txt"
            bad.write_text("not-media", encoding="utf-8")

            # Single character -> simple
            kind, fields, files = svc_mod.build_submission(
                parser.parse_args(["swap", "--video", str(vid), "-p", str(p1), "--clip-seconds", "9.5",
                                   "--extra-lora", "drive/style.safetensors:0.75"])
            )
            self.assertEqual(kind, "simple")
            field_map = dict(fields)
            self.assertEqual(field_map["mode"], "simple")
            self.assertEqual(field_map["clip_seconds"], "9.5")
            lora_names = [val for key, val in fields if key == "lora"]
            self.assertEqual(len(lora_names), 2)
            self.assertEqual(lora_names[1], "drive/style.safetensors")
            self.assertEqual(len(files), 2)

            # Dual character + --max-quality -> advanced_maxq
            kind2, fields2, files2 = svc_mod.build_submission(
                parser.parse_args(["swap", "--video", str(vid), "-p", str(p1), "-p", str(p2), "--max-quality"])
            )
            self.assertEqual(kind2, "advanced_maxq")
            self.assertIn(("mode", "advanced"), fields2)
            self.assertIn(("max_quality", "true"), fields2)
            self.assertEqual(len(files2), 3)

            # Invalid media bytes rejected locally before upload
            with self.assertRaises(UsageError):
                svc_mod.build_submission(parser.parse_args(["swap", "--video", str(bad), "-p", str(p1)]))

            # Object add without --placement rejected
            with self.assertRaises(UsageError):
                svc_mod.build_submission(
                    parser.parse_args(["object", "--video", str(vid), "--edit-mode", "add",
                                       "--target", "on table", "--change-to", "mug"])
                )

            # Color with both --no-detail and --no-tone rejected
            with self.assertRaises(UsageError):
                svc_mod.build_submission(
                    parser.parse_args(["color", "--video", str(vid), "--no-detail", "--no-tone"])
                )

    def test_end_to_end_submit_sse_wait_download_and_cancel(self):
        svc_mod = load_openjutsu_service()
        state_store = {"post_attempts": 0, "cancelled": False, "auth_headers": []}
        rendered_bytes = FAKE_MP4 + b"-rendered-output"

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                return

            def do_POST(self):
                state_store["auth_headers"].append(self.headers.get("Authorization"))
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length)
                if self.path == "/api/jobs":
                    state_store["post_attempts"] += 1
                    if state_store["post_attempts"] == 1:
                        self.send_response(429)
                        self.send_header("Retry-After", "1")
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Another upload is in progress."}).encode())
                        return
                    assert b'name="mode"\r\n\r\nsimple' in body
                    assert FAKE_MP4 in body
                    assert FAKE_PNG in body
                    self.send_response(202)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"id": "job_abc123", "status": "pending", "queuePosition": 1,
                                    "error": None, "video": None, "data": None}).encode()
                    )

            def do_GET(self):
                state_store["auth_headers"].append(self.headers.get("Authorization"))
                if self.path.startswith("/api/jobs/job_abc123/events"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    events = [
                        {"type": "progress", "node": {"title": "Sample H3", "clip": 1}, "step": 3, "steps": 6},
                        {
                            "type": "job",
                            "job": {
                                "id": "job_abc123",
                                "status": "succeeded",
                                "queuePosition": None,
                                "error": None,
                                "video": {"id": "out_1", "name": "swap.mp4", "url": "/api/jobs/job_abc123/outputs/out_1"},
                                "data": None,
                            },
                        },
                    ]
                    for ev in events:
                        self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                    self.wfile.flush()
                elif self.path == "/api/jobs/job_abc123":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({
                            "id": "job_abc123",
                            "status": "succeeded",
                            "queuePosition": None,
                            "error": None,
                            "video": {"id": "out_1", "name": "swap.mp4", "url": "/api/jobs/job_abc123/outputs/out_1"},
                            "data": None,
                        }).encode()
                    )
                elif self.path == "/api/jobs/job_abc123/outputs/out_1?download=1":
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.end_headers()
                    self.wfile.write(rendered_bytes)
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_DELETE(self):
                state_store["auth_headers"].append(self.headers.get("Authorization"))
                if self.path == "/api/jobs/job_abc123":
                    state_store["cancelled"] = True
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"id": "job_abc123", "status": "canceled", "queuePosition": None,
                                    "error": None, "video": None, "data": None}).encode()
                    )

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                vid = root / "src.mp4"
                vid.write_bytes(FAKE_MP4)
                pic = root / "ref.png"
                pic.write_bytes(FAKE_PNG)
                out_mp4 = root / "result.mp4"

                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    svc_mod._run_business_command(
                        [
                            "--video", str(vid),
                            "-p", str(pic),
                            "--clip-seconds", "9.5",
                            "-o", str(out_mp4),
                            "--url", base_url,
                            "--api-token", "secret-token",
                        ],
                        "swap",
                    )
                result_json = json.loads(stdout.getvalue())
                self.assertEqual(result_json["status"], "succeeded")
                self.assertEqual(result_json["local_output"], str(out_mp4.resolve()))
                self.assertEqual(out_mp4.read_bytes(), rendered_bytes)
                self.assertEqual(state_store["post_attempts"], 2)
                self.assertIn("clip 1 | Sample H3 (3/6)", stderr.getvalue())
                self.assertTrue(all(h == "Bearer secret-token" for h in state_store["auth_headers"]))

                cancel_out = io.StringIO()
                with contextlib.redirect_stdout(cancel_out), contextlib.redirect_stderr(io.StringIO()):
                    svc_mod._run_business_command(
                        ["job_abc123", "--url", base_url, "--api-token", "secret-token"],
                        "cancel",
                    )
                self.assertEqual(json.loads(cancel_out.getvalue())["status"], "canceled")
                self.assertTrue(state_store["cancelled"])

                # Failed status raises CliError cleanly
                client = svc_mod.OpenjutsuClient(base_url, "secret-token")
                with self.assertRaises(UsageError):
                    client.download({"id": "job_abc123", "status": "in_progress"}, str(root / "x.mp4"))
                with self.assertRaises(CliError):
                    client.download({"id": "job_abc123", "status": "succeeded", "video": None, "data": None}, str(root / "x.mp4"))
        finally:
            server.shutdown()
            server.server_close()

    def test_notebook_includes_vram_flags_and_caddy_headers(self):
        self.assertEqual(runtime.DEFAULT_OPENJUTSU_REPO, "https://github.com/D3voz/openjutsu.git")
        nb = json.loads((ROOT / "src/colabllm/services/openjutsu.ipynb").read_text(encoding="utf-8"))
        setup_src = "".join(nb["cells"][2]["source"])
        self.assertIn("https://github.com/D3voz/openjutsu.git", setup_src)
        self.assertNotIn("https://github.com/deluxebear/colab_openjutsu", setup_src)
        svc_src = "".join(nb["cells"][3]["source"])
        self.assertIn("select_comfy_vram_flags", svc_src)
        self.assertIn("CT_COMFY_VRAM_MODE", svc_src)
        self.assertIn("--cache-classic", svc_src)
        caddy_src = nb["cells"][4]["source"]
        text = "".join(caddy_src) if isinstance(caddy_src, list) else caddy_src
        self.assertIn("REWRITE_HOST = True", text)
        self.assertIn("header_up -X-Forwarded-Host", text)
        self.assertIn("header_up -X-Forwarded-Proto", text)

    def test_sync_app_checkout_switches_origin_cleanly(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_a = root / "repo_a"
            repo_b = root / "repo_b"
            for repo, marker in ((repo_a, "v1"), (repo_b, "v2")):
                repo.mkdir()
                subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
                subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
                subprocess.run(["git", "-C", str(repo), "config", "user.name", "Tester"], check=True)
                (repo / "marker.txt").write_text(marker, encoding="utf-8")
                subprocess.run(["git", "-C", str(repo), "add", "marker.txt"], check=True)
                subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", marker], check=True)

            app_dir = root / "app"
            log_file = root / "env.log"
            runtime.sync_app_checkout(app_dir, str(repo_a), log_file)
            self.assertEqual((app_dir / "marker.txt").read_text(encoding="utf-8"), "v1")

            # Same origin -> pull in place
            runtime.sync_app_checkout(app_dir, str(repo_a), log_file)
            self.assertEqual((app_dir / "marker.txt").read_text(encoding="utf-8"), "v1")

            # Different origin -> replaces checkout cleanly
            runtime.sync_app_checkout(app_dir, str(repo_b), log_file)
            self.assertEqual((app_dir / "marker.txt").read_text(encoding="utf-8"), "v2")


if __name__ == "__main__":
    unittest.main()
