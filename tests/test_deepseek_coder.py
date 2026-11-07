"""Unit tests for deepseek-coder hardware-adaptive context planning, Engram shard symlink, and CLI."""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

import deepseek_coder_runtime as rt  # noqa: E402
from cli_support import CliError  # noqa: E402


def load_deepseek_service():
    path = ROOT / "src/colabllm/services/deepseek-coder.py"
    spec = importlib.util.spec_from_file_location("deepseek_coder_service_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DeepSeekRuntimePlanningTest(unittest.TestCase):
    def test_dsv41_kv_bytes_per_token_matches_compressed_source_layers(self):
        # 3 CSA layers (0.5 * 640) + 1 HCA layer (1.0 * 640) = 1,600 elements/token -> 3,200 B/tok in f16
        self.assertEqual(rt.dsv41_kv_bytes_per_token("f16", parallel=1), 3200.0)
        self.assertEqual(rt.dsv41_kv_bytes_per_token("f16", parallel=2), 6400.0)
        self.assertEqual(rt.dsv41_kv_bytes_per_token("q8_0", parallel=1), 1700.0)

    def test_g4_96gb_selects_full_gpu_engram_ram_and_1m_context(self):
        plan = rt.plan_hardware_config(vram_mib=97_887, ram_mib=180_224, requested_context="auto")
        self.assertEqual(plan["mode"], "full-gpu")
        self.assertEqual(plan["gpu_layers"], "all")
        self.assertEqual(plan["n_cpu_moe"], 0)
        self.assertTrue(plan["engram_in_ram"])
        self.assertEqual(plan["context"], 1_048_576)
        self.assertTrue(plan["context_auto"])
        self.assertEqual(plan["batch"], 4096)
        self.assertEqual(plan["ubatch"], 512)
        self.assertEqual(plan["kv_cache_mib"], 3200.0)
        self.assertEqual(plan["context_ladder"][0], 1_048_576)
        self.assertEqual(plan["context_ladder"][-1], 16_384)

    def test_h100_80gb_auto_context_and_1m_moe_offload(self):
        auto_plan = rt.plan_hardware_config(vram_mib=81_559, ram_mib=171_008, requested_context="auto")
        self.assertEqual(auto_plan["mode"], "full-gpu")
        self.assertEqual(auto_plan["n_cpu_moe"], 0)
        self.assertTrue(auto_plan["engram_in_ram"])
        self.assertEqual(auto_plan["ubatch"], 128)
        self.assertGreaterEqual(auto_plan["context"], 524_288)
        self.assertLess(auto_plan["context"], 1_048_576)

        one_m_plan = rt.plan_hardware_config(vram_mib=81_559, ram_mib=171_008, requested_context="1m")
        self.assertEqual(one_m_plan["context"], 1_048_576)
        self.assertFalse(one_m_plan["context_auto"])
        self.assertTrue(one_m_plan["engram_in_ram"])
        self.assertEqual(one_m_plan["context_ladder"], [1_048_576])
        self.assertGreater(one_m_plan["n_cpu_moe"], 0)
        self.assertLessEqual(one_m_plan["n_cpu_moe"], 6)

    def test_a100_40gb_uses_hybrid_cpu_moe_offload_and_ssd_lazy_engram(self):
        plan = rt.plan_hardware_config(vram_mib=40_960, ram_mib=85_504, requested_context="auto")
        self.assertTrue(plan["mode"].startswith("hybrid-cpu-moe-"))
        self.assertFalse(plan["engram_in_ram"])
        self.assertGreater(plan["n_cpu_moe"], 20)
        self.assertGreaterEqual(plan["context"], 262_144)

    def test_cpu_only_and_insufficient_hardware_errors(self):
        cpu_plan = rt.plan_hardware_config(vram_mib=0, ram_mib=131_072, requested_context="auto")
        self.assertEqual(cpu_plan["mode"], "cpu-only")
        self.assertEqual(cpu_plan["gpu_layers"], "0")
        self.assertGreaterEqual(cpu_plan["context"], 131_072)

        with self.assertRaises(ValueError):
            rt.plan_hardware_config(vram_mib=0, ram_mib=32_768)
        with self.assertRaises(ValueError):
            # 16 GiB VRAM cannot even hold dense attention + runtime buffers
            rt.plan_hardware_config(vram_mib=16_384, ram_mib=131_072)
        with self.assertRaises(ValueError):
            # 24 GiB VRAM + 48 GiB RAM lacks enough host RAM for the offloaded MoE layers
            rt.plan_hardware_config(vram_mib=24_576, ram_mib=49_152)
        with self.assertRaises(ValueError):
            # Explicit 1M context while forcing 0 CPU MoE layers on 80 GB VRAM must fail closed
            rt.plan_hardware_config(
                vram_mib=81_559,
                ram_mib=171_008,
                requested_context="1m",
                requested_cpu_moe="0",
            )

    def test_parse_specs_and_oom_detection(self):
        self.assertIsNone(rt.parse_context_spec("auto"))
        self.assertEqual(rt.parse_context_spec("512k"), 524_288)
        self.assertEqual(rt.parse_context_spec("1m"), 1_048_576)
        with self.assertRaises(ValueError):
            rt.parse_context_spec("8192")
        with self.assertRaises(ValueError):
            rt.parse_context_spec("2m")

        self.assertTrue(rt.looks_like_oom("ggml_backend_cuda_buffer_type_alloc_buffer: cuda_malloc failed: out of memory"))
        self.assertFalse(rt.looks_like_oom("main: model loaded cleanly"))


class DeepSeekShardSymlinkAndArgsTest(unittest.TestCase):
    def test_derive_and_create_engram_shard_symlink_flat_and_nested(self):
        flat_first = "DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf"
        self.assertEqual(
            rt.derive_shard6_link_name(flat_first),
            "DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-IQ2_XXS-00006-of-00006.gguf",
        )
        nested_first = "IQ2_XXS/DeepSeek-V4.1-Flash-IQ2_XXS-00001-of-00006.gguf"
        nested_sixth = rt.derive_shard6_link_name(nested_first)
        self.assertEqual(nested_sixth, "IQ2_XXS/DeepSeek-V4.1-Flash-IQ2_XXS-00006-of-00006.gguf")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            engram = root / rt.ENGRAM_FILENAME
            engram.write_bytes(b"GGUF-ENGRAM")

            flat_link = rt.ensure_engram_shard_symlink(root, rt.SHARD6_LINK_NAME, rt.ENGRAM_FILENAME)
            self.assertTrue(flat_link.is_symlink())
            self.assertEqual(flat_link.read_bytes(), b"GGUF-ENGRAM")
            # Idempotent re-run
            self.assertEqual(rt.ensure_engram_shard_symlink(root, rt.SHARD6_LINK_NAME, rt.ENGRAM_FILENAME), flat_link)

            nested_link = rt.ensure_engram_shard_symlink(root, nested_sixth, rt.ENGRAM_FILENAME)
            self.assertTrue(nested_link.is_symlink())
            self.assertEqual(nested_link.read_bytes(), b"GGUF-ENGRAM")

    def test_build_server_args_enforces_loopback_auth_engram_residency_and_default_reasoning_on(self):
        plan = rt.plan_hardware_config(vram_mib=97_887, ram_mib=180_224)
        args = rt.build_server_args(
            server_bin="/tmp/llama-server",
            model_path="/tmp/shard1.gguf",
            alias="deepseek-v4.1-flash-coder",
            port=8001,
            key_file="/tmp/api_keys.txt",
            plan=plan,
            projector_path="/tmp/mmproj.gguf",
            chat_template_file="/tmp/chat_template.jinja",
        )
        self.assertIn("127.0.0.1", args)
        self.assertIn("--api-key-file", args)
        self.assertIn("/tmp/api_keys.txt", args)
        lm_idx = args.index("--load-mode")
        self.assertEqual(args[lm_idx + 1], "none")
        lzm_idx = args.index("--lazy-mode")
        self.assertEqual(args[lzm_idx + 1], "off")
        rea_idx = args.index("--reasoning")
        self.assertEqual(args[rea_idx + 1], "on")
        rf_idx = args.index("--reasoning-format")
        self.assertEqual(args[rf_idx + 1], "deepseek")
        self.assertLess(args.index("--jinja"), args.index("--chat-template-file"))
        self.assertIn("--no-host", args)
        self.assertEqual(args[args.index("--ctx-checkpoints") + 1], "32")
        self.assertEqual(args[args.index("--cache-ram") + 1], "16384")
        self.assertEqual(plan["ubatch_ladder"], [512, 256])
        self.assertTrue(rt.should_evict_page_cache(plan))
        self.assertNotIn("--reasoning-effort", args)
        self.assertIn("per_layer_token_embd.weight=CPU", args)
        self.assertIn("--mmproj", args)
        self.assertNotIn("--n-cpu-moe", args)

        hybrid_plan = rt.plan_hardware_config(vram_mib=40_960, ram_mib=85_504, enable_vision=False)
        hybrid_args = rt.build_server_args(
            server_bin="/tmp/llama-server",
            model_path="/tmp/shard1.gguf",
            alias="deepseek-v4.1-flash-coder",
            port=8001,
            key_file="/tmp/api_keys.txt",
            plan=hybrid_plan,
            projector_path="/tmp/mmproj.gguf",
            reasoning="off",
            reasoning_effort="medium",
        )
        self.assertEqual(hybrid_args[hybrid_args.index("--load-mode") + 1], "mmap")
        self.assertEqual(hybrid_args[hybrid_args.index("--lazy-mode") + 1], "on")
        self.assertEqual(hybrid_args[hybrid_args.index("--reasoning") + 1], "off")
        self.assertEqual(hybrid_args[hybrid_args.index("--reasoning-effort") + 1], "medium")
        self.assertNotIn("--no-host", hybrid_args)
        self.assertFalse(rt.should_evict_page_cache(hybrid_plan))
        self.assertIn("--n-cpu-moe", hybrid_args)
        self.assertNotIn("--mmproj", hybrid_args)

    def test_gguf_chat_template_extraction_patching_and_page_cache_eviction(self):
        import struct

        stock_tmpl = (
            '{%- set DSML = "\uff5cDSML\uff5c" -%}\n'
            '{%- if reasoning_effort is not defined or reasoning_effort is none -%}{%- set reasoning_effort = "high" -%}{%- endif -%}\n'
            '{%- set effort_map = {"low": 50, "high": 75, "max": 100} -%}\n'
            '{{- raise_exception("Invalid reasoning effort: should be int within [1,100] or one of \\"low\\", \\"high\\", \\"max\\"") -}}\n'
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gguf = root / "shard1.gguf"
            k1 = b"general.architecture"
            v1 = b"deepseek41"
            k2 = b"tokenizer.chat_template"
            v2 = stock_tmpl.encode("utf-8")
            payload = bytearray()
            payload += b"GGUF" + struct.pack("<IQQ", 3, 0, 2)
            payload += struct.pack("<Q", len(k1)) + k1 + struct.pack("<IQ", 8, len(v1)) + v1
            payload += struct.pack("<Q", len(k2)) + k2 + struct.pack("<IQ", 8, len(v2)) + v2
            gguf.write_bytes(bytes(payload))

            out_jinja = root / "chat_template.jinja"
            written = rt.prepare_patched_chat_template(gguf, out_jinja)
            self.assertEqual(written, out_jinja)
            patched_text = out_jinja.read_text(encoding="utf-8")
            self.assertIn('"medium": 60', patched_text)
            self.assertIn('"minimal": 25', patched_text)
            self.assertIn("reasoning_effort = 50", patched_text)
            self.assertIn("\uff5cDSML\uff5c", patched_text)

            engram = root / rt.ENGRAM_FILENAME
            engram.write_bytes(b"GGUF-ENGRAM")
            (root / rt.SHARD6_LINK_NAME).symlink_to(rt.ENGRAM_FILENAME)
            self.assertGreaterEqual(rt.evict_gguf_page_cache(root, keep_engram=True, log=lambda *_: None), 0)
            self.assertGreaterEqual(rt.evict_gguf_page_cache(root, keep_engram=False, log=lambda *_: None), 0)

    def test_concurrent_staging_does_not_delete_active_cache_files(self):
        import hashlib
        import threading
        import time

        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            service_dir = Path(tmp) / "svc"
            service_dir.mkdir(parents=True)
            b1 = b"shard-one"
            b2 = b"shard-two-longer"
            fake_lock = {
                "repo": "example/deepseek-coder",
                "revision": "deadbeef",
                "shard6_link": "m-00002-of-00002.gguf",
                "engram_file": "engram.gguf",
                "files": [
                    {"name": "m-00001-of-00002.gguf", "size": len(b1), "sha256": hashlib.sha256(b1).hexdigest()},
                    {"name": "engram.gguf", "size": len(b2), "sha256": hashlib.sha256(b2).hexdigest()},
                ],
                "mmproj": None,
            }
            (service_dir / "model-lock.json").write_text(json.dumps(fake_lock), encoding="utf-8")
            barrier = threading.Barrier(2)

            stage_dirs: list[Path] = []

            def fake_download(*, repo_id, filename, revision, local_dir):
                ld = Path(local_dir)
                stage_dirs.append(ld)
                inc = ld / ".cache/huggingface/download" / (filename + ".incomplete")
                inc.parent.mkdir(parents=True, exist_ok=True)
                payload = b1 if filename.startswith("m-") else b2
                inc.write_bytes(payload)
                barrier.wait(timeout=5)
                if filename.endswith("engram.gguf"):
                    time.sleep(0.05)
                # Isolated per-file staging dir must never collide with sibling worker cleanup!
                self.assertTrue(inc.is_file(), f"{inc} was deleted by concurrent worker")
                dest = ld / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                inc.replace(dest)
                return str(dest)

            with (
                patch.object(rt, "_resolve_model_lock", return_value=fake_lock),
                patch.object(rt, "_hf_client", return_value=(None, fake_download)),
                patch.object(rt, "require_space", return_value=None),
            ):
                first_shard, mmproj = rt.stage_model_and_projector(
                    repo="example/deepseek-coder",
                    quant="IQ2_XXS",
                    enable_vision=False,
                    service_dir=service_dir,
                    work_dir=work_dir,
                )
            self.assertEqual(len(set(stage_dirs)), 2)
            self.assertTrue(first_shard.is_file())
            self.assertIsNone(mmproj)
            self.assertTrue((work_dir / "model/m-00002-of-00002.gguf").is_symlink())
            self.assertFalse(any(p.exists() for p in stage_dirs))


class DeepSeekServiceCliTest(unittest.TestCase):
    def test_prepare_up_applies_and_resets_overrides(self):
        svc = load_deepseek_service()
        baseline = dict(svc.SERVICE.env)
        svc.prepare_up(
            types.SimpleNamespace(
                up_CT_DS_CONTEXT="524288",
                up_CT_DS_REASONING_EFFORT="75",
                up_no_vision=True,
            )
        )
        self.assertEqual(svc.SERVICE.env["CT_DS_CONTEXT"], "524288")
        self.assertEqual(svc.SERVICE.env["CT_DS_REASONING_EFFORT"], "75")
        self.assertEqual(svc.SERVICE.env["CT_DS_VISION"], "0")

        svc.prepare_up(types.SimpleNamespace())
        self.assertEqual(svc.SERVICE.env, baseline)

    def test_cmd_plan_and_cmd_models(self):
        svc = load_deepseek_service()
        buf = io.StringIO()
        plan_args = argparse.Namespace(
            preset="G4",
            vram_gb=None,
            ram_gb=None,
            context="auto",
            parallel=1,
            cache_type="f16",
            batch=2048,
            ubatch="auto",
            cpu_moe_layers="auto",
            no_vision=False,
        )
        with patch("sys.stdout", buf):
            rc = svc.cmd_plan(plan_args)
        self.assertEqual(rc, 0)
        parsed = json.loads(buf.getvalue())
        self.assertEqual(parsed["context"], 1_048_576)

        class _FakeResp:
            def __init__(self, payload):
                self._raw = json.dumps(payload).encode("utf-8")

            def read(self):
                return self._raw

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req, timeout=None):
            self.assertEqual(req.get_header("Authorization"), "Bearer test-secret")
            if req.full_url.endswith("/v1/models"):
                return _FakeResp({"data": [{"id": "deepseek-v4.1-flash-coder"}]})
            if req.full_url.endswith("/props"):
                return _FakeResp(
                    {
                        "default_generation_settings": {"n_ctx": 1_048_576},
                        "total_slots": 1,
                        "modalities": {"vision": True, "audio": False},
                    }
                )
            raise AssertionError(f"unexpected URL {req.full_url}")

        models_buf = io.StringIO()
        with patch.object(svc, "safe_urlopen", side_effect=fake_urlopen), patch("sys.stdout", models_buf):
            rc = svc.cmd_models(argparse.Namespace(url="https://example.invalid", token="test-secret"))
        self.assertEqual(rc, 0)
        models_out = json.loads(models_buf.getvalue())
        self.assertEqual(models_out["models"], ["deepseek-v4.1-flash-coder"])
        self.assertEqual(models_out["context_allocated"], 1_048_576)
        self.assertEqual(models_out["modalities"]["vision"], True)

        with self.assertRaises(CliError):
            svc.cmd_models(argparse.Namespace(url="https://example.invalid", token=None))


if __name__ == "__main__":
    unittest.main()
