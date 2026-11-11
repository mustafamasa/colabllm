"""Unit tests for Ollama runtime helpers, Caddy routing, startup tuning, and CLI subcommands."""
from __future__ import annotations

import contextlib
import http.server
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

from cli_support import UsageError  # noqa: E402
import colabsvc  # noqa: E402
import ollama_runtime as rt  # noqa: E402


def load_ollama_service():
    path = ROOT / "src/colabllm/services/ollama.py"
    spec = importlib.util.spec_from_file_location("ollama_service_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OllamaRuntimeConfigTest(unittest.TestCase):
    def test_model_specs_and_aliases(self):
        self.assertEqual(rt.parse_model_specs(None), ["qwen2.5:0.5b"])
        self.assertEqual(rt.parse_model_specs(None, no_models=True), [])
        self.assertEqual(
            rt.parse_model_specs(["qwen3:8b, nomic-embed-text", "hf.co/bartowski/Model-GGUF:Q4_K_M", "qwen3:8b"]),
            ["qwen3:8b", "nomic-embed-text", "hf.co/bartowski/Model-GGUF:Q4_K_M"],
        )
        self.assertEqual(rt.parse_model_specs(["none"]), [])
        for bad in ("-rf", "--flag", "../escape", "bad model", "a\nb"):
            with self.assertRaises(ValueError, msg=bad):
                rt.validate_model_spec(bad)
        with self.assertRaises(ValueError):
            rt.parse_model_specs(["qwen3:8b"], no_models=True)

        aliases = rt.parse_model_aliases(["claude-3-5-sonnet=qwen2.5-coder:7b, gpt-4o=qwen3:8b"])
        self.assertEqual(aliases, [("claude-3-5-sonnet", "qwen2.5-coder:7b"), ("gpt-4o", "qwen3:8b")])
        self.assertEqual(
            rt.models_to_pull(["qwen3:8b"], aliases),
            ["qwen3:8b", "qwen2.5-coder:7b"],
        )
        for bad_alias in ("missing_equals", "same=same", "gpt-4o=a,gpt-4o=b"):
            with self.assertRaises(ValueError, msg=bad_alias):
                rt.parse_model_aliases([bad_alias])

    def test_compat_api_parsing(self):
        self.assertEqual(rt.parse_compat_apis(None), ("ollama", "openai", "anthropic"))
        self.assertEqual(rt.parse_compat_apis(["all"]), ("ollama", "openai", "anthropic"))
        self.assertEqual(rt.parse_compat_apis(["anthropic,openai"]), ("openai", "anthropic"))
        self.assertEqual(rt.parse_compat_apis(["claude", "oai"]), ("openai", "anthropic"))
        self.assertEqual(rt.parse_compat_apis(["native"]), ("ollama",))
        with self.assertRaises(ValueError):
            rt.parse_compat_apis(["unknown"])

    def test_server_env_profiles_and_validation(self):
        low = rt.build_ollama_server_env(profile="low-vram", context_length=16384)
        self.assertEqual(low["OLLAMA_FLASH_ATTENTION"], "1")
        self.assertEqual(low["OLLAMA_KV_CACHE_TYPE"], "q8_0")
        self.assertEqual(low["OLLAMA_NUM_PARALLEL"], "1")
        self.assertEqual(low["OLLAMA_MAX_LOADED_MODELS"], "1")
        self.assertEqual(low["OLLAMA_CONTEXT_LENGTH"], "16384")

        # q8_0 auto-enables Flash Attention when unspecified, and conflicts with explicit False
        auto_fa = rt.build_ollama_server_env(kv_cache_type="q8_0")
        self.assertEqual(auto_fa["OLLAMA_FLASH_ATTENTION"], "1")
        self.assertEqual(auto_fa["OLLAMA_KV_CACHE_TYPE"], "q8_0")
        with self.assertRaises(ValueError):
            rt.build_ollama_server_env(kv_cache_type="q8_0", flash_attention=False)

        # Explicit --no-flash-attention resets preset q8_0 back to f16 safely
        no_fa = rt.build_ollama_server_env(profile="low-vram", flash_attention=False)
        self.assertEqual(no_fa["OLLAMA_FLASH_ATTENTION"], "0")
        self.assertEqual(no_fa["OLLAMA_KV_CACHE_TYPE"], "f16")

        tuned = rt.build_ollama_server_env(
            keep_alive="-1",
            load_timeout="15m",
            gpu_overhead="512M",
            num_parallel=2,
            max_loaded_models=2,
            max_queue=256,
            sched_spread=True,
            max_transfer_streams=8,
            origins="chrome-extension://*",
            no_cloud=True,
            ollama_debug="1",
            extra_env=["OLLAMA_NOPRUNE=1", "CUDA_VISIBLE_DEVICES=0"],
        )
        self.assertEqual(tuned["OLLAMA_GPU_OVERHEAD"], str(512 * 1024 * 1024))
        self.assertEqual(tuned["OLLAMA_KEEP_ALIVE"], "-1")
        self.assertEqual(tuned["OLLAMA_LOAD_TIMEOUT"], "15m")
        self.assertEqual(tuned["OLLAMA_NO_CLOUD"], "1")
        self.assertEqual(tuned["OLLAMA_NOPRUNE"], "1")
        self.assertEqual(tuned["CUDA_VISIBLE_DEVICES"], "0")

        # Security / safety rejections
        for bad_env in ("OLLAMA_HOST=0.0.0.0:11434", "HTTP_PROXY=http://1.2.3.4:8080", "LD_PRELOAD=/tmp/x.so"):
            with self.assertRaises(ValueError, msg=bad_env):
                rt.parse_extra_ollama_env([bad_env])

    def test_caddyfile_supports_bearer_x_api_key_and_compat_guards(self):
        full_cf = rt.render_ollama_caddyfile(
            public_port=18080,
            service_port=11434,
            auth_user="admin",
            hashed_pass="$2a$14$hash",
            api_token="sk-test-token",
            cookie_tok="cookietok",
            compat_apis=("ollama", "openai", "anthropic"),
            read_only_models=False,
        )
        self.assertIn('@bearer header Authorization "Bearer sk-test-token"', full_cf)
        self.assertIn('@apikey header X-Api-Key "sk-test-token"', full_cf)
        self.assertIn("header_up -Authorization", full_cf)
        self.assertIn("header_up -X-Api-Key", full_cf)
        self.assertNotIn("@disallowed_api", full_cf)
        self.assertNotIn("@blocked_mgmt", full_cf)

        restricted_cf = rt.render_ollama_caddyfile(
            public_port=18080,
            service_port=11434,
            auth_user="admin",
            hashed_pass="$2a$14$hash",
            api_token="sk-test-token",
            cookie_tok="cookietok",
            compat_apis=("openai", "anthropic"),
            read_only_models=True,
        )
        self.assertIn("@disallowed_api not path / /v1/chat/completions", restricted_cf)
        self.assertIn("/v1/responses", restricted_cf)
        self.assertIn("/v1/messages", restricted_cf)
        self.assertNotIn(" /api/* ", restricted_cf)
        self.assertIn("@blocked_mgmt path /api/pull", restricted_cf)

    def test_normalize_api_keep_alive(self):
        self.assertEqual(rt.normalize_api_keep_alive("-1"), -1)
        self.assertEqual(rt.normalize_api_keep_alive("0"), 0)
        self.assertEqual(rt.normalize_api_keep_alive("3600"), 3600)
        self.assertEqual(rt.normalize_api_keep_alive("30m"), "30m")
        self.assertIsNone(rt.normalize_api_keep_alive(None))


class _FakeOllamaHandler(http.server.BaseHTTPRequestHandler):
    token = "sk-unit-test-secret"
    state: dict

    def log_message(self, fmt, *args):
        return

    def _authorized(self) -> bool:
        auth = self.headers.get("Authorization", "")
        xkey = self.headers.get("X-Api-Key", "") or self.headers.get("x-api-key", "")
        return auth == f"Bearer {self.token}" or xkey == self.token

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path == "/api/tags":
            if self.state.get("block_native_api"):
                self._send_json(403, {"error": "disabled"})
                return
            self._send_json(200, {
                "models": [
                    {"name": m, "size": 1_500_000_000, "details": {"family": "qwen3", "quantization_level": "Q4_K_M"}}
                    for m in sorted(self.state["installed"])
                ]
            })
        elif self.path == "/api/ps":
            self._send_json(200, {
                "models": [
                    {"name": m, "size": 1_500_000_000, "size_vram": 1_500_000_000, "context_length": 32768, "expires_at": "permanent"}
                    for m in sorted(self.state["loaded"])
                ]
            })
        elif self.path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{"id": m, "owned_by": "library"} for m in sorted(self.state["installed"])],
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_DELETE(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path == "/api/delete":
            body = self._read_json()
            self.state["installed"].discard(body["model"])
            self.state["loaded"].discard(body["model"])
            self._send_json(200, {"status": "deleted"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        body = self._read_json()
        if self.path == "/api/pull":
            model = body["model"]
            self.state["pulls"].append(model)
            self.state["installed"].add(model)
            events = [
                {"status": "pulling manifest"},
                {"status": "downloading layer", "completed": 500_000_000, "total": 1_000_000_000},
                {"status": "success"},
            ]
            payload = ("\n".join(json.dumps(e) for e in events) + "\n").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/api/copy":
            self.state["installed"].add(body["destination"])
            self.state["copies"].append((body["source"], body["destination"]))
            self._send_json(200, {"status": "copied"})
        elif self.path == "/api/generate":
            model = body["model"]
            self.state["last_generate"] = body
            if "embed" in model:
                self._send_json(400, {"error": f'"{model}" does not support generate'})
                return
            if body.get("keep_alive") == 0:
                self.state["loaded"].discard(model)
            else:
                self.state["loaded"].add(model)
            self._send_json(200, {"model": model, "response": "", "done": True, "done_reason": "load"})
        elif self.path == "/api/embed":
            model = body["model"]
            self.state["last_embed"] = body
            if body.get("keep_alive") == 0:
                self.state["loaded"].discard(model)
            else:
                self.state["loaded"].add(model)
            self._send_json(200, {"model": model, "embeddings": [[0.1, 0.2]]})
        elif self.path == "/api/show":
            self._send_json(200, {
                "details": {"family": "qwen3", "parameter_size": "8B", "quantization_level": "Q4_K_M"},
                "capabilities": ["completion", "tools", "thinking"],
                "parameters": "num_ctx 32768",
            })
        elif self.path == "/api/chat":
            self.state["last_chat"] = body
            prompt = body["messages"][-1]["content"]
            self._send_json(200, {"model": body["model"], "message": {"role": "assistant", "content": f"ollama:{prompt}"}})
        elif self.path == "/v1/chat/completions":
            prompt = body["messages"][-1]["content"]
            self._send_json(200, {"choices": [{"message": {"role": "assistant", "content": f"openai:{prompt}"}}]})
        elif self.path == "/v1/messages":
            self.state["last_anthropic_headers"] = dict(self.headers)
            prompt = body["messages"][-1]["content"]
            self._send_json(200, {"content": [{"type": "text", "text": f"anthropic:{prompt}"}]})
        else:
            self._send_json(404, {"error": "not found"})


class OllamaServiceCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "colabllm-home"
        self.env_patch = mock.patch.dict(os.environ, {"COLABLLM_HOME": str(self.home)}, clear=False)
        self.env_patch.start()
        # Reload with temporary COLABLLM_HOME
        self.orig_state_root = colabsvc.STATE_ROOT
        colabsvc.STATE_ROOT = self.home
        self.mod = load_ollama_service()

        self.server_state = {
            "installed": {"qwen2.5:0.5b"},
            "loaded": {"qwen2.5:0.5b"},
            "pulls": [],
            "copies": [],
            "block_native_api": False,
        }
        handler_cls = type("BoundFakeOllamaHandler", (_FakeOllamaHandler,), {"state": self.server_state})
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.port}"

        colabsvc.State(self.mod.SERVICE).save(
            url=self.base_url,
            api_token=_FakeOllamaHandler.token,
            ready=True,
            env={"CT_OLLAMA_COMPAT": "openai,anthropic", "CT_OLLAMA_READ_ONLY_MODELS": "1"},
        )

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        colabsvc.STATE_ROOT = self.orig_state_root
        self.env_patch.stop()
        self.tmp.cleanup()

    def _run_subcommand(self, argv: list[str]) -> str:
        parser = colabsvc.build_parser(self.mod.SERVICE, self.mod.configure_cli)
        args = parser.parse_args(argv)
        buf = io.StringIO()
        err_buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err_buf):
            self.mod.HANDLERS[args.cmd](args)
        return buf.getvalue()

    def test_prepare_up_populates_and_resets_cleanly(self):
        parser = colabsvc.build_parser(self.mod.SERVICE, self.mod.configure_cli)
        args = parser.parse_args([
            "up",
            "--models", "qwen3:8b,nomic-embed-text",
            "--alias", "claude-3-5-sonnet=qwen2.5-coder:7b",
            "--preload-all",
            "--models-storage", "drive",
            "--ollama-version", "0.17.7",
            "--compat", "openai,anthropic",
            "--read-only-models",
            "--profile", "coding",
            "--context-length", "32768",
            "--gpu-overhead", "256M",
            "--no-cloud",
        ])
        self.mod.prepare_up(args)
        env = dict(self.mod.SERVICE.env)
        self.assertEqual(env["CT_MODELS"], "qwen3:8b,nomic-embed-text")
        self.assertEqual(env["CT_OLLAMA_ALIASES"], "claude-3-5-sonnet=qwen2.5-coder:7b")
        self.assertEqual(env["CT_OLLAMA_PRELOAD"], "all")
        self.assertEqual(env["CT_OLLAMA_MODELS_STORAGE"], "drive")
        self.assertEqual(env["CT_OLLAMA_VERSION"], "0.17.7")
        self.assertEqual(env["CT_OLLAMA_COMPAT"], "openai,anthropic")
        self.assertEqual(env["CT_OLLAMA_READ_ONLY_MODELS"], "1")
        server_env = json.loads(env["CT_OLLAMA_SERVER_ENV"])
        self.assertEqual(server_env["OLLAMA_CONTEXT_LENGTH"], "32768")
        self.assertEqual(server_env["OLLAMA_KV_CACHE_TYPE"], "q8_0")
        self.assertEqual(server_env["OLLAMA_KEEP_ALIVE"], "-1")
        self.assertEqual(server_env["OLLAMA_GPU_OVERHEAD"], str(256 * 1024 * 1024))
        self.assertEqual(server_env["OLLAMA_NO_CLOUD"], "1")

        # Subsequent call with defaults must not retain previous overrides
        self.mod.prepare_up(types.SimpleNamespace(models=None))
        self.assertEqual(self.mod.SERVICE.env, {"CT_MODELS": "qwen2.5:0.5b"})

    def test_model_preset_applies_gpu_model_alias_tuning_and_allows_overrides(self):
        parser = colabsvc.build_parser(self.mod.SERVICE, self.mod.configure_cli)
        base_gpu = self.mod.SERVICE.gpu
        self.mod.prepare_up(parser.parse_args(["up", "--preset", "huihui-27b"]))
        env = dict(self.mod.SERVICE.env)
        self.assertEqual(self.mod.SERVICE.gpu, "G4")
        self.assertEqual(env["CT_MODELS"], "huihui_ai/Qwen3.8-abliterated:27b-bf16")
        self.assertEqual(env["CT_OLLAMA_ALIASES"], "huihui-qwen3.8-27b=huihui_ai/Qwen3.8-abliterated:27b-bf16")
        server_env = json.loads(env["CT_OLLAMA_SERVER_ENV"])
        self.assertEqual(server_env["OLLAMA_CONTEXT_LENGTH"], "262144")
        self.assertEqual(server_env["OLLAMA_FLASH_ATTENTION"], "1")
        self.assertEqual(server_env["OLLAMA_KV_CACHE_TYPE"], "f16")
        self.assertEqual(server_env["OLLAMA_KEEP_ALIVE"], "-1")

        # Explicit flags override preset values; `-m` drops the preset alias
        self.mod.prepare_up(parser.parse_args([
            "up", "--preset", "huihui-27b", "--gpu", "L4",
            "-m", "huihui_ai/Qwen3.8-abliterated:27b-q4_K_M", "--context-length", "65536", "--kv-cache-type", "q8_0",
        ]))
        env = dict(self.mod.SERVICE.env)
        self.assertEqual(self.mod.SERVICE.gpu, base_gpu)  # --gpu is applied later by colabsvc from args
        self.assertEqual(env["CT_MODELS"], "huihui_ai/Qwen3.8-abliterated:27b-q4_K_M")
        self.assertNotIn("CT_OLLAMA_ALIASES", env)
        server_env = json.loads(env["CT_OLLAMA_SERVER_ENV"])
        self.assertEqual(server_env["OLLAMA_CONTEXT_LENGTH"], "65536")
        self.assertEqual(server_env["OLLAMA_KV_CACHE_TYPE"], "q8_0")

        self.mod.prepare_up(types.SimpleNamespace(models=None))
        self.assertEqual(self.mod.SERVICE.gpu, base_gpu)
        self.assertEqual(self.mod.SERVICE.env, {"CT_MODELS": "qwen2.5:0.5b"})

    def test_endpoints_redacts_token_by_default(self):
        redacted = json.loads(self._run_subcommand(["endpoints", "--json"]))
        self.assertEqual(redacted["compat"], ["openai", "anthropic"])
        self.assertTrue(redacted["read_only_models"])
        self.assertEqual(redacted["interfaces"]["openai"]["api_key"], "<API_TOKEN>")
        self.assertNotIn(_FakeOllamaHandler.token, json.dumps(redacted))

        revealed = json.loads(self._run_subcommand(["endpoints", "--json", "--show-secrets"]))
        self.assertEqual(revealed["interfaces"]["anthropic"]["api_key"], _FakeOllamaHandler.token)

    def test_models_supports_native_and_v1_fallback(self):
        out = json.loads(self._run_subcommand(["models", "--json"]))
        self.assertEqual(out["source"], "ollama")
        self.assertEqual(out["models"][0]["name"], "qwen2.5:0.5b")
        self.assertEqual(out["running"][0]["name"], "qwen2.5:0.5b")

        self.server_state["block_native_api"] = True
        fallback = json.loads(self._run_subcommand(["models", "--json"]))
        self.assertEqual(fallback["source"], "v1")
        self.assertEqual(fallback["models"][0]["name"], "qwen2.5:0.5b")

    def test_pull_alias_preload_embedding_fallback_and_unload(self):
        pull_out = json.loads(
            self._run_subcommand(["pull", "qwen3:8b", "--alias", "gpt-4o", "--preload", "--keep-alive", "-1"])
        )
        self.assertEqual(pull_out["model"], "qwen3:8b")
        self.assertEqual(pull_out["alias"], "gpt-4o")
        self.assertEqual(pull_out["preloaded"], "/api/generate")
        self.assertEqual(self.server_state["last_generate"]["keep_alive"], -1)
        self.assertIn("gpt-4o", self.server_state["installed"])
        self.assertIn("qwen3:8b", self.server_state["loaded"])

        # Embedding model auto-pulls and falls back to /api/embed on preload
        emb_out = json.loads(self._run_subcommand(["preload", "nomic-embed-text", "--keep-alive", "30m"]))
        self.assertIn("nomic-embed-text", self.server_state["pulls"])
        self.assertEqual(emb_out["endpoint"], "/api/embed")
        self.assertEqual(self.server_state["last_embed"]["keep_alive"], "30m")

        # Unload sends integer 0 keep_alive
        unload_out = json.loads(self._run_subcommand(["unload", "qwen3:8b"]))
        self.assertEqual(unload_out["status"], "unloaded")
        self.assertEqual(self.server_state["last_generate"]["keep_alive"], 0)
        self.assertNotIn("qwen3:8b", self.server_state["loaded"])

    def test_run_auto_pulls_and_supports_all_three_compatible_apis(self):
        # 1. Native ollama API auto-pulls missing model first
        out_ollama = self._run_subcommand(["run", "gemma4:e4b", "hello", "--num-ctx", "8192", "--keep-alive", "-1"])
        self.assertEqual(out_ollama.strip(), "ollama:hello")
        self.assertIn("gemma4:e4b", self.server_state["pulls"])
        self.assertEqual(self.server_state["last_chat"]["options"]["num_ctx"], 8192)
        self.assertEqual(self.server_state["last_chat"]["keep_alive"], -1)

        # 2. OpenAI compatible API (already installed -> no duplicate pull)
        pulls_before = len(self.server_state["pulls"])
        out_openai = self._run_subcommand(["run", "gemma4:e4b", "world", "--api", "openai"])
        self.assertEqual(out_openai.strip(), "openai:world")
        self.assertEqual(len(self.server_state["pulls"]), pulls_before)

        # 3. Anthropic Messages API sends x-api-key + anthropic-version
        out_anthropic = self._run_subcommand(["run", "gemma4:e4b", "claude", "--api", "anthropic"])
        self.assertEqual(out_anthropic.strip(), "anthropic:claude")
        hdrs = {k.lower(): v for k, v in self.server_state["last_anthropic_headers"].items()}
        self.assertEqual(hdrs.get("x-api-key"), _FakeOllamaHandler.token)
        self.assertEqual(hdrs.get("anthropic-version"), "2023-06-01")

    def test_cp_show_and_rm(self):
        cp_out = json.loads(self._run_subcommand(["cp", "qwen2.5:0.5b", "my-alias"]))
        self.assertEqual(cp_out["alias"], "my-alias")
        show_out = json.loads(self._run_subcommand(["show", "qwen2.5:0.5b"]))
        self.assertEqual(show_out["family"], "qwen3")
        rm_out = json.loads(self._run_subcommand(["rm", "my-alias"]))
        self.assertEqual(rm_out["deleted"], ["my-alias"])
        self.assertNotIn("my-alias", self.server_state["installed"])

        with self.assertRaises(UsageError):
            self._run_subcommand(["pull", "a", "b", "--alias", "only-one-allowed"])


if __name__ == "__main__":
    unittest.main()
