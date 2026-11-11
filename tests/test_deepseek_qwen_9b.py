"""Unit tests for deepseek-qwen-9b L4/T4 256K-context planning, GGUF staging, and CLI."""
from __future__ import annotations

import argparse
import hashlib
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

import deepseek_qwen_9b_runtime as rt  # noqa: E402
from colabsvc import build_parser  # noqa: E402


def load_service_module():
    path = ROOT / "src/colabllm/services/deepseek-qwen-9b.py"
    spec = importlib.util.spec_from_file_location("deepseek_qwen_9b_service_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DeepSeekQwen9BPlanningTest(unittest.TestCase):
    def test_qwen35_kv_bytes_per_token_and_256k_cache_budget(self):
        # 8 trunk full-attention layers * 4 KV heads * (256 K + 256 V) = 16,384 elems/token (+ 2,048 with MTP)
        self.assertEqual(rt.qwen35_kv_bytes_per_token("f16", parallel=1, enable_mtp=False), 32_768.0)
        self.assertEqual(rt.qwen35_kv_bytes_per_token("q8_0", parallel=1, enable_mtp=False), 17_408.0)
        self.assertEqual(rt.qwen35_kv_bytes_per_token("q8_0", parallel=1, enable_mtp=True), 19_584.0)
        self.assertEqual(rt.qwen35_kv_bytes_per_token("q4_0", parallel=1, enable_mtp=False), 9_216.0)

        # At native 262,144 tokens: 4,352 MiB KV + 52 MiB SSM = 4,404.0 MiB (no MTP); +544 MiB = 4,948.0 MiB (with MTP)
        self.assertEqual(rt.estimate_kv_cache_mib(262_144, "q8_0", parallel=1, enable_mtp=False), 4_404.0)
        self.assertEqual(rt.estimate_kv_cache_mib(262_144, "q8_0", parallel=1, enable_mtp=True), 4_948.0)
        self.assertEqual(rt.estimate_kv_cache_mib(262_144, "f16", parallel=1, enable_mtp=False), 8_244.0)

    def test_l4_default_selects_imatrix_q4_k_m_native_mtp_256k_and_vision(self):
        plan = rt.plan_hardware_config(
            vram_mib=23_034,
            ram_mib=54_272,
            quant="auto",
            requested_context="auto",
            cache_type="q8_0",
            enable_vision=True,
        )
        self.assertEqual(plan["mode"], "full-gpu")
        self.assertEqual(plan["quant"], "i1-Q4_K_M")
        self.assertTrue(plan["quant_auto"])
        self.assertTrue(plan["enable_mtp"])
        self.assertEqual(plan["resolved_repo"], rt.IMATRIX_REPO)
        self.assertEqual(plan["gpu_layers"], "all")
        self.assertEqual(plan["context"], 262_144)
        self.assertEqual(plan["rope_scaling"], "")
        self.assertEqual(plan["rope_scale"], 1.0)
        self.assertEqual(plan["kv_cache_mib"], 4_948.0)
        self.assertGreaterEqual(plan["estimated_vram_headroom_mib"], 9_000.0)

    def test_t4_auto_enables_mtp_with_208k_context_or_full_256k_without_mtp(self):
        plan_mtp = rt.plan_hardware_config(
            vram_mib=15_360,
            ram_mib=12_980,
            quant="auto",
            requested_context="auto",
            cache_type="q8_0",
            enable_vision=True,
        )
        self.assertEqual(plan_mtp["mode"], "full-gpu")
        self.assertEqual(plan_mtp["quant"], "i1-Q4_K_M")
        self.assertTrue(plan_mtp["enable_mtp"])
        self.assertEqual(plan_mtp["context"], 212_992)

        plan_no_mtp = rt.plan_hardware_config(
            vram_mib=15_360,
            ram_mib=12_980,
            quant="auto",
            requested_context="auto",
            cache_type="q8_0",
            enable_vision=True,
            mtp="off",
        )
        self.assertFalse(plan_no_mtp["enable_mtp"])
        self.assertEqual(plan_no_mtp["context"], 262_144)
        self.assertGreaterEqual(plan_no_mtp["estimated_vram_headroom_mib"], 4_500.0)

    def test_forced_q8_0_on_t4_auto_disables_mtp_and_steps_down_context_safely(self):
        plan = rt.plan_hardware_config(
            vram_mib=15_360,
            ram_mib=12_980,
            quant="Q8_0",
            requested_context="auto",
            cache_type="q8_0",
            enable_vision=True,
        )
        self.assertEqual(plan["quant"], "Q8_0")
        self.assertFalse(plan["enable_mtp"])
        self.assertLess(plan["context"], 262_144)
        self.assertGreaterEqual(plan["context"], 131_072)
        self.assertEqual(plan["context"] % rt.CONTEXT_ALIGN, 0)

    def test_cpu_only_and_yarn_512k_scaling(self):
        cpu_plan = rt.plan_hardware_config(vram_mib=0, ram_mib=13_000, quant="auto", requested_context="auto")
        self.assertEqual(cpu_plan["mode"], "cpu-only")
        self.assertEqual(cpu_plan["gpu_layers"], "0")
        self.assertEqual(cpu_plan["context"], 16_384)

        yarn_plan = rt.plan_hardware_config(
            vram_mib=23_034,
            ram_mib=54_272,
            quant="i1-Q4_K_M",
            requested_context="512k",
            cache_type="q8_0",
        )
        self.assertEqual(yarn_plan["context"], 524_288)
        self.assertEqual(yarn_plan["rope_scaling"], "yarn")
        self.assertEqual(yarn_plan["rope_scale"], 2.0)
        self.assertEqual(yarn_plan["yarn_orig_ctx"], 262_144)


class DeepSeekQwen9BServerArgsAndStagingTest(unittest.TestCase):
    def test_build_server_args_for_l4_mtp_and_yarn(self):
        plan = rt.plan_hardware_config(vram_mib=23_034, ram_mib=54_272)
        args = rt.build_server_args(
            server_bin="/tmp/llama-server",
            model_path="/tmp/model.i1-Q4_K_M.gguf",
            alias="deepseek-v4-pro-qwen3.5-9b",
            port=8000,
            key_file="/tmp/api_key.txt",
            plan=plan,
            projector_path="/tmp/mmproj-f16.gguf",
            mtp_path="/tmp/mtp-Q8_0.gguf",
            reasoning="on",
        )
        self.assertIn("127.0.0.1", args)
        self.assertEqual(args[args.index("-ngl") + 1], "all")
        self.assertEqual(args[args.index("--fit") + 1], "off")
        self.assertEqual(args[args.index("-c") + 1], "262144")
        self.assertEqual(args[args.index("-ctk") + 1], "q8_0")
        self.assertEqual(args[args.index("-ctv") + 1], "q8_0")
        self.assertEqual(args[args.index("-b") + 1], "2048")
        self.assertEqual(args[args.index("-ub") + 1], "512")
        self.assertEqual(args[args.index("--ctx-checkpoints") + 1], "32")
        self.assertEqual(args[args.index("--mmproj") + 1], "/tmp/mmproj-f16.gguf")
        self.assertEqual(args[args.index("-md") + 1], "/tmp/mtp-Q8_0.gguf")
        self.assertEqual(args[args.index("--spec-type") + 1], "draft-mtp")
        self.assertEqual(args[args.index("-ngld") + 1], "all")
        self.assertEqual(args[args.index("-ctkd") + 1], "q8_0")
        self.assertEqual(args[args.index("-ctvd") + 1], "q8_0")
        self.assertEqual(args[args.index("--spec-draft-n-max") + 1], "3")
        self.assertEqual(args[args.index("--api-key-file") + 1], "/tmp/api_key.txt")
        self.assertEqual(args[args.index("--reasoning") + 1], "on")
        self.assertNotIn("--rope-scaling", args)
        self.assertNotIn("--override-tensor", args)

    def test_stage_mtp_sidecar_off_and_drive_cache_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            drive_root = Path(tmp) / "drive_models"
            self.assertIsNone(
                rt.stage_mtp_sidecar(
                    work_dir=work_dir,
                    drive_backup_root=drive_root,
                    mtp_mode="off",
                    log=lambda *_: None,
                )
            )

            # Seed Drive cache with a sparse GGUF file (>=500 MB logical size, b"GGUF" magic)
            drive_dir = drive_root / rt.UPSTREAM_HF_REPO
            drive_dir.mkdir(parents=True)
            drive_file = drive_dir / rt.MTP_SIDECAR_FILENAME
            with drive_file.open("wb") as handle:
                handle.write(b"GGUF\x03\x00\x00\x00")
                handle.seek(500_000_000)
                handle.write(b"\x00")
            expected_sha = rt.digest_file(drive_file)
            (drive_dir / f"{rt.MTP_SIDECAR_FILENAME}.sha256").write_text(expected_sha + "\n", encoding="utf-8")

            staged = rt.stage_mtp_sidecar(
                work_dir=work_dir,
                drive_backup_root=drive_root,
                mtp_mode="auto",
                log=lambda *_: None,
            )
            self.assertIsNotNone(staged)
            assert staged is not None
            self.assertTrue(staged.is_file())
            self.assertEqual(staged.stat().st_size, 500_000_001)

    def test_cross_repo_imatrix_quant_stages_model_from_i1_repo_and_mmproj_from_default_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp) / "work"
            svc_dir = Path(tmp) / "svc"
            svc_dir.mkdir(parents=True)
            model_bytes = b"imatrix-q4-k-m-payload"
            proj_bytes = b"f16-vision-mmproj-payload"
            recorded_calls: list[tuple[str, str]] = []

            patched_specs = dict(rt.PINNED_QUANT_SPECS)
            patched_specs["i1-Q4_K_M"] = {
                "repo": rt.IMATRIX_REPO,
                "revision": rt.IMATRIX_REVISION,
                "name": "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf",
                "size": len(model_bytes),
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
                "weights_mib": 1,
            }
            patched_proj = {
                "repo": rt.DEFAULT_REPO,
                "revision": rt.DEFAULT_REVISION,
                "name": "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.mmproj-f16.gguf",
                "size": len(proj_bytes),
                "sha256": hashlib.sha256(proj_bytes).hexdigest(),
            }

            def fake_dl(*, repo_id, filename, revision, local_dir):
                recorded_calls.append((repo_id, filename))
                dest = Path(local_dir) / filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(proj_bytes if "mmproj" in filename else model_bytes)
                return str(dest)

            with (
                patch.dict(rt.PINNED_QUANT_SPECS, patched_specs, clear=True),
                patch.object(rt, "VISION_PROJECTOR_SPEC", patched_proj),
                patch.object(rt, "_hf_client", return_value=(types.SimpleNamespace, fake_dl)),
                patch.object(rt, "_xet_available", return_value=False),
            ):
                model_path, proj_path = rt.stage_model_and_projector(
                    repo=rt.DEFAULT_REPO,
                    quant="i1-Q4_K_M",
                    enable_vision=True,
                    service_dir=svc_dir,
                    work_dir=work_dir,
                    drive_backup_root=None,
                    log=lambda *_: None,
                )

            self.assertTrue(model_path.is_file())
            self.assertIsNotNone(proj_path)
            assert proj_path is not None
            self.assertTrue(proj_path.is_file())
            self.assertEqual(
                recorded_calls,
                [
                    (rt.IMATRIX_REPO, "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.i1-Q4_K_M.gguf"),
                    (rt.DEFAULT_REPO, "DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic.mmproj-f16.gguf"),
                ],
            )


class DeepSeekQwen9BCliTest(unittest.TestCase):
    def test_service_defaults_to_l4_and_plan_command_outputs_presets(self):
        svc = load_service_module()
        self.assertEqual(svc.SERVICE.name, "deepseek-qwen-9b")
        self.assertEqual(svc.SERVICE.gpu, "L4")
        self.assertEqual(svc.SERVICE.default_hostname, "deepseek-9b")
        self.assertEqual(svc.SERVICE.auth, "builtin")

        parser = build_parser(svc.SERVICE, svc.configure_cli)
        parsed = parser.parse_args(
            ["up", "--quant", "i1-Q6_K", "--context", "128k", "--no-vision", "--no-mtp", "--mtp-draft", "4"]
        )
        svc.prepare_up(parsed)
        self.assertEqual(svc.SERVICE.env["CT_DS9B_QUANT"], "i1-Q6_K")
        self.assertEqual(svc.SERVICE.env["CT_DS9B_CONTEXT"], "131072")
        self.assertEqual(svc.SERVICE.env["CT_DS9B_VISION"], "0")
        self.assertEqual(svc.SERVICE.env["CT_DS9B_MTP"], "off")
        self.assertEqual(svc.SERVICE.env["CT_DS9B_MTP_DRAFT"], "4")

        # Reset env when no flags passed
        parsed_default = parser.parse_args(["up"])
        svc.prepare_up(parsed_default)
        self.assertEqual(svc.SERVICE.env["CT_DS9B_QUANT"], "auto")
        self.assertEqual(svc.SERVICE.env["CT_DS9B_VISION"], "1")
        self.assertEqual(svc.SERVICE.env["CT_DS9B_MTP"], "auto")

        plan_args = parser.parse_args(["plan", "--preset", "L4"])
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            rc = svc.cmd_plan(plan_args)
        self.assertEqual(rc, 0)
        out = json.loads(buf.getvalue())
        self.assertEqual(out["quant"], "i1-Q4_K_M")
        self.assertTrue(out["enable_mtp"])
        self.assertEqual(out["context"], 262_144)


if __name__ == "__main__":
    unittest.main()
