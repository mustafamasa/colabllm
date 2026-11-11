"""Generated notebooks stay in sync, and notebook-embedded modules enforce their security contracts."""
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

import colab_asgi_common as common  # noqa: E402
import comfyui_workflow_helper as comfy  # noqa: E402


class NotebookSyncTest(unittest.TestCase):
    def test_generated_notebooks_match_committed_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copytree(ROOT / "templates", root / "templates")
            shutil.copytree(ROOT / "src/colabllm/lib", root / "src/colabllm/lib")
            subprocess.run([sys.executable, str(root / "templates/build_notebooks.py")],
                           check=True, capture_output=True, text=True)
            generated = sorted((root / "src/colabllm/services").glob("*.ipynb"))
            self.assertEqual({p.name for p in generated},
                             {p.name for p in (ROOT / "src/colabllm/services").glob("*.ipynb")})
            for path in [*generated, root / "templates/service.ipynb"]:
                committed = ROOT / path.relative_to(root)
                self.assertEqual(path.read_text(encoding="utf-8"), committed.read_text(encoding="utf-8"),
                                 f"{committed.relative_to(ROOT)} is stale; run python3 templates/build_notebooks.py")

    def test_notebooks_have_no_unresolved_placeholders_or_legacy_patterns(self):
        for path in (ROOT / "src/colabllm/services").glob("*.ipynb"):
            source = "".join("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                             for c in json.loads(path.read_text())["cells"])
            for marker in ("__BEFORE_READY__", "_SOURCE__", "__EXTRA_", "__TITLE__", "__WORK__"):
                self.assertNotIn(marker, source, f"{path.name}: unresolved {marker}")
            self.assertNotIn("urlretrieve(", source, path.name)
            self.assertNotIn('"--allowed-local-media-path", "/content"', source, path.name)
            self.assertNotIn('client_host in ("127.0.0.1"', source, path.name)


class AsgiCommonTest(unittest.TestCase):
    def test_bearer_auth_is_fail_closed_and_exact(self):
        self.assertFalse(common.verify_bearer_auth({"authorization": "Bearer "}, ""))
        self.assertFalse(common.verify_bearer_auth({}, "secret"))
        self.assertFalse(common.verify_bearer_auth({"authorization": "Bearer wrong"}, "secret"))
        self.assertFalse(common.verify_bearer_auth({"authorization": "Bearer secretX"}, "secret"))
        self.assertTrue(common.verify_bearer_auth({"Authorization": "Bearer secret"}, "secret"))
        self.assertTrue(common.verify_bearer_auth({"x-api-key": "secret"}, "secret"))

    def test_voice_names_reject_traversal(self):
        self.assertEqual(common.validate_voice_name(" Alice_1 "), "alice_1")
        for bad in ("", "../x", "a/b", "..", ".hidden", "x" * 65, "a b", "a\x00"):
            with self.assertRaises(ValueError, msg=bad):
                common.validate_voice_name(bad)

    def test_multipart_header_parameters_are_sanitized(self):
        content_type, body = common.build_multipart_form(
            [("name", "v")], [("audio_sample", 'evil"\r\nX-Injected: 1.wav', "audio/wav\r\nX: y", b"RIFF")])
        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertNotIn(b"\r\nX-Injected", body)
        self.assertNotIn(b"\r\nX: y", body)
        self.assertIn(b'filename="evil\\"X-Injected: 1.wav"', body)

    def test_drive_paths_stay_inside_allowed_roots(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mydrive = root / "drive/MyDrive"
            data = mydrive / "colab-llm/svc"
            (data / "voices").mkdir(parents=True)
            (data / "voices/a.wav").write_bytes(b"RIFF")
            (root / "secret.txt").write_text("x")
            kwargs = dict(data_dir=data, default_subdir="voices", mydrive_root=mydrive, drive_root=root / "drive")
            self.assertEqual(common.resolve_drive_path("a.wav", **kwargs), (data / "voices/a.wav").resolve())
            self.assertEqual(common.resolve_drive_path("drive://colab-llm/svc/voices/a.wav", **kwargs),
                             (data / "voices/a.wav").resolve())
            self.assertEqual(common.resolve_drive_path("MyDrive/colab-llm/svc/voices/a.wav", **kwargs),
                             (data / "voices/a.wav").resolve())
            for bad in ("../../../../secret.txt", "/etc/passwd", "file:///etc/passwd",
                        "file:///content/ct/service.log", "drive://../../secret.txt", "/content/ct/x"):
                with self.assertRaises(ValueError, msg=bad):
                    common.resolve_drive_path(bad, **kwargs)
            with self.assertRaises(ValueError):
                common.list_drive_audio_files("../../*", **kwargs)
            listed = common.list_drive_audio_files("", **kwargs)
            self.assertEqual([item["name"] for item in listed], ["a.wav"])

    def test_no_drive_mode_uses_data_dir_as_root(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data/svc"
            data.mkdir(parents=True)
            (data / "in.wav").write_bytes(b"RIFF")
            missing = Path(temp) / "not-mounted"
            kwargs = dict(data_dir=data, mydrive_root=missing, drive_root=missing.parent)
            self.assertEqual(common.resolve_drive_path(str(data / "in.wav"), **kwargs), (data / "in.wav").resolve())
            with self.assertRaises(ValueError):
                common.resolve_drive_path(str(Path(temp) / "other.wav"), **kwargs)

    def test_save_output_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "data"
            out = data / "outputs"
            missing = Path(temp) / "nodrive"
            saved = common.save_output_audio_to_drive(b"abc", "auto", data_dir=data, output_dir=out,
                                                      prefix="t", mydrive_root=missing, drive_root=missing)
            self.assertTrue(Path(saved).is_relative_to(out))
            with self.assertRaises(ValueError):
                common.save_output_audio_to_drive(b"abc", "../../escape.wav", data_dir=data, output_dir=out,
                                                  prefix="t", mydrive_root=missing, drive_root=missing)


class ComfyWorkflowHelperTest(unittest.TestCase):
    def test_model_destination_is_confined_to_models_root(self):
        with tempfile.TemporaryDirectory() as temp:
            comfy_dir = Path(temp)
            dest = comfy.validate_model_destination("loras", "x.safetensors", comfy_dir=comfy_dir)
            self.assertEqual(dest, (comfy_dir / "models/loras/x.safetensors").resolve())
            for directory, name in (("../custom_nodes", "evil.py"), ("loras", "../../custom_nodes/evil.py"),
                                    ("", "x.safetensors"), ("loras", ""), ("/etc", "passwd")):
                if directory == "/etc":
                    # Leading slashes are stripped and stay relative to models/.
                    self.assertTrue(comfy.validate_model_destination(directory, name, comfy_dir=comfy_dir)
                                    .is_relative_to((comfy_dir / "models").resolve()))
                    continue
                with self.assertRaises(ValueError, msg=(directory, name)):
                    comfy.validate_model_destination(directory, name, comfy_dir=comfy_dir)

    def test_workflow_models_with_traversal_are_rejected(self):
        workflow = {"properties": {"models": [
            {"url": "https://example.com/m.safetensors", "directory": "../../custom_nodes", "name": "x.py"}]}}
        with self.assertRaises(ValueError):
            comfy.extract_workflow_models(workflow)


class MiddlewareImportTest(unittest.TestCase):
    def test_nemotron_chat_rewrite_rejects_non_drive_file_urls(self):
        middleware = importlib.import_module("nemotron_middleware")
        body = json.dumps({"messages": [{"role": "user", "content": [
            {"type": "audio_url", "audio_url": {"url": "file:///etc/passwd"}}]}]}).encode()
        with self.assertRaises(ValueError):
            middleware._rewrite_drive_urls_in_chat_body(body)

    def test_tts_middlewares_require_valid_voice_names(self):
        indextts = importlib.import_module("indextts_middleware")
        code, out = indextts._upload_drive_voice_sync({"name": "../x", "drive_path": "a.wav"})
        self.assertEqual(code, 400)
        qwen = importlib.import_module("qwen3_tts_middleware")
        code, out = qwen._upload_drive_voice_sync({"name": "../x", "audio_base64": "AAAA"})
        self.assertEqual(code, 400)
        with self.assertRaises(ValueError):
            qwen._prepare_speech_request(json.dumps({"input": "hi", "ref_audio": "file:///etc/passwd"}).encode())
        drama3 = importlib.import_module("drama3_middleware")
        code, out = drama3._upload_drive_voice_sync({"name": "../x", "drive_path": "a.wav"})
        self.assertEqual(code, 400)
        with self.assertRaises(ValueError):
            drama3._prepare_speech_request(json.dumps({"input": "hi", "ref_audio": "file:///etc/passwd"}).encode())
        with self.assertRaises(ValueError):
            drama3._prepare_speech_request(
                json.dumps({"input": "hi", "references": [{"audio_path": "file:///etc/passwd"}]}).encode()
            )
        with self.assertRaises(ValueError):
            drama3._prepare_speech_request(json.dumps({"input": "hi", "save_to_drive": "../../escape.wav"}).encode())
        tangoflux = importlib.import_module("tangoflux_server")
        with self.assertRaises(ValueError):
            tangoflux._prepare_generate_request(json.dumps({"prompt": "rain", "save_to_drive": "../../escape.wav"}).encode())
        with self.assertRaises(ValueError):
            tangoflux._prepare_batch_request(
                json.dumps({"items": [{"prompt": "rain", "save_to_drive": "/etc/passwd"}]}).encode()
            )

    def test_flux_klein_middleware_rejects_traversal_and_non_drive_urls(self):
        flux = importlib.import_module("flux_klein_middleware")
        bad_chat = json.dumps({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
        ]}]}).encode()
        with self.assertRaises(ValueError):
            flux._rewrite_drive_urls_in_chat_body(bad_chat, is_distilled=True)
        bad_lora = json.dumps({"prompt": "hi", "lora": {"path": "/etc/passwd"}}).encode()
        with self.assertRaises(ValueError):
            flux._prepare_generation_request(bad_lora, is_distilled=True)
        bad_save = json.dumps({"prompt": "hi", "save_to_drive": "../../escape.png"}).encode()
        with self.assertRaises(ValueError):
            flux._prepare_generation_request(bad_save, is_distilled=True)

    def test_qwen_image_middleware_rejects_traversal_and_non_drive_urls(self):
        qwen_img = importlib.import_module("qwen_image_middleware")
        bad_chat = json.dumps({"messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "file:///etc/passwd"}},
        ]}]}).encode()
        with self.assertRaises(ValueError):
            qwen_img._rewrite_drive_urls_in_chat_body(bad_chat)
        bad_lora = json.dumps({"prompt": "hi", "lora": {"path": "/etc/passwd"}}).encode()
        with self.assertRaises(ValueError):
            qwen_img._prepare_generation_request(bad_lora)
        bad_save = json.dumps({"prompt": "hi", "save_to_drive": "../../escape.png"}).encode()
        with self.assertRaises(ValueError):
            qwen_img._prepare_generation_request(bad_save)

    def test_deepseek_coder_notebook_embeds_runtime_and_loopback_auth(self):
        ds_rt = importlib.import_module("deepseek_coder_runtime")
        plan = ds_rt.plan_hardware_config(vram_mib=97_887, ram_mib=180_224)
        args = ds_rt.build_server_args(
            server_bin="/tmp/llama-server",
            model_path="/tmp/m.gguf",
            alias="deepseek-v4.1-flash-coder",
            port=8001,
            key_file="/tmp/keys.txt",
            plan=plan,
        )
        self.assertIn("127.0.0.1", args)
        self.assertIn("--api-key-file", args)
        nb = (ROOT / "src/colabllm/services/deepseek-coder.ipynb").read_text(encoding="utf-8")
        self.assertIn("deepseek_coder_runtime.py", nb)
        self.assertIn("per_layer_token_embd.weight=CPU", nb)


if __name__ == "__main__":
    unittest.main()
