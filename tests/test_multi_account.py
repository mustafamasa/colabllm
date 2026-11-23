"""Multi-account and shared-Drive integration tests for colabllm."""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

import colab_accounts as ca  # noqa: E402
import colabsvc as svc  # noqa: E402
from cli_support import CliError, UsageError  # noqa: E402
from processes import stop_owned_process  # noqa: E402


def _write_registry(cfg_dir: Path, *, active: str = "main", default_drive: str | None = None, accounts: dict | None = None) -> None:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if accounts is None:
        accounts = {
            "main": {"email": "main@example.com"},
            "gpu1": {"email": "gpu1@example.com", "drive_account": "main"},
            "gpu2": {"email": "gpu2@example.com"},
        }
    for name in accounts:
        (cfg_dir / "accounts" / name).mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "active": active,
        "default_drive_account": default_drive,
        "accounts": accounts,
    }
    (cfg_dir / "accounts.json").write_text(json.dumps(payload), encoding="utf-8")


class MultiAccountResolverTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = self.root / "colab-cli"

    def test_validate_account_rejects_path_traversal_and_bad_names(self):
        for good in ("default", "gpu2", "work.acct", "user@example.com", "a_b-1"):
            self.assertEqual(ca.validate_account(good), good)
        for bad in ("", "   ", "..", "a..b", "-bad", ".hidden", "a/b", "中文", "x" * 65):
            with self.subTest(bad=bad), self.assertRaises(UsageError):
                ca.validate_account(bad)

    def test_legacy_mode_without_accounts_json_returns_none_without_creating_files(self):
        home = self.root / "home"
        with patch.dict(os.environ, {"COLAB_CLI_CONFIG_DIR": ""}, clear=False), \
             patch.object(ca.Path, "home", return_value=home):
            env = {k: v for k, v in os.environ.items() if k not in ("COLABLLM_ACCOUNT", "COLAB_ACCOUNT", "COLABLLM_DRIVE_ACCOUNT", "COLAB_DRIVE_ACCOUNT", "COLAB_DFS_ACCOUNT", "COLAB_CLI_CONFIG_DIR")}
            with patch.dict(os.environ, env, clear=True):
                self.assertIsNone(ca.load_accounts_registry())
                self.assertEqual(ca.resolve_effective_account(), (None, "none"))
                self.assertEqual(ca.resolve_effective_drive_account(None), (None, "none"))
                self.assertEqual(ca.account_sessions_path(None), home / ".config/colab-cli/sessions.json")
                self.assertEqual(ca.account_dfs_credentials_path(), home / ".config/colab-cli/dfs_credentials.json")
                self.assertIsNone(ca.colab_subprocess_env(None, None))
                self.assertFalse(home.exists())

    def test_registry_email_lookup_precedence_and_errors(self):
        _write_registry(
            self.cfg,
            active="main",
            default_drive="main",
            accounts={
                "main": {"email": "Main@Example.com"},
                "gpu1": {"email": "GPU1@Example.com", "drive_account": "main"},
                "dup1": {"email": "shared@example.com"},
                "dup2": {"email": "shared@example.com"},
            },
        )
        base_env = {
            k: v for k, v in os.environ.items()
            if k not in ("COLABLLM_ACCOUNT", "COLAB_ACCOUNT", "COLABLLM_DRIVE_ACCOUNT", "COLAB_DRIVE_ACCOUNT", "COLAB_DFS_ACCOUNT")
        }
        base_env["COLAB_CLI_CONFIG_DIR"] = str(self.cfg)
        with patch.dict(os.environ, base_env, clear=True):
            # Case-insensitive email lookup via explicit flag
            self.assertEqual(ca.resolve_effective_account(explicit="gpu1@example.com"), ("gpu1", "flag"))
            # Ambiguous email fails clearly
            with self.assertRaises(UsageError):
                ca.resolve_effective_account(explicit="shared@example.com")
            # Unknown account fails clearly
            with self.assertRaises(UsageError):
                ca.resolve_effective_account(explicit="missing")
            # Bound drive_account beats default_drive_account; explicit beats all
            self.assertEqual(ca.resolve_effective_drive_account("gpu1"), ("main", "bound"))
            self.assertEqual(ca.resolve_effective_drive_account("gpu1", explicit_drive="gpu1@example.com"), ("gpu1", "flag"))

        # Env precedence: explicit > COLABLLM_ACCOUNT > (saved when prefer_saved) > COLAB_ACCOUNT > active
        with patch.dict(os.environ, {**base_env, "COLABLLM_ACCOUNT": "gpu1", "COLAB_ACCOUNT": "main"}, clear=True):
            self.assertEqual(ca.resolve_effective_account(explicit="main"), ("main", "flag"))
            self.assertEqual(ca.resolve_effective_account(saved="main", prefer_saved=True), ("gpu1", "env"))
        with patch.dict(os.environ, {**base_env, "COLAB_ACCOUNT": "gpu1"}, clear=True):
            self.assertEqual(ca.resolve_effective_account(saved="main", prefer_saved=True), ("main", "state"))
            self.assertEqual(ca.resolve_effective_account(prefer_saved=False), ("gpu1", "colab_env"))

    def test_corrupt_accounts_json_raises_cli_error(self):
        self.cfg.mkdir(parents=True)
        (self.cfg / "accounts.json").write_text("{corrupt", encoding="utf-8")
        with patch.dict(os.environ, {"COLAB_CLI_CONFIG_DIR": str(self.cfg)}):
            with self.assertRaises(CliError):
                ca.load_accounts_registry()

    def test_ensure_session_reads_isolated_per_account_sessions_json(self):
        _write_registry(self.cfg, active="main")
        (self.cfg / "accounts/gpu2/sessions.json").write_text(json.dumps({"test": {"id": "sess-2"}}), encoding="utf-8")
        service = svc.Service(name="test", notebook="test.ipynb", service_port=8000, account="gpu2")
        with patch.dict(os.environ, {"COLAB_CLI_CONFIG_DIR": str(self.cfg)}), \
             patch.object(svc, "check_remote_session", return_value="alive") as mock_check, \
             patch.object(svc, "run") as mock_run:
            self.assertTrue(svc.ensure_session(service))
        mock_check.assert_called_once()
        mock_run.assert_not_called()

        # Different account has no session recorded -> calls `colab new` with COLAB_ACCOUNT=gpu1
        service_other = svc.Service(name="test", notebook="test.ipynb", service_port=8000, account="gpu1")
        with patch.dict(os.environ, {"COLAB_CLI_CONFIG_DIR": str(self.cfg)}), \
             patch.object(svc, "run") as mock_run:
            self.assertFalse(svc.ensure_session(service_other))
        mock_run.assert_called_once()
        self.assertEqual(mock_run.call_args.kwargs["env"]["COLAB_ACCOUNT"], "gpu1")

    def test_drive_account_conflicts_with_no_drive(self):
        service = svc.Service(name="ollama", notebook="ollama.ipynb", service_port=11434)
        args = type("Args", (), {
            "dry_run": True,
            "auto_restart": False,
            "restart_delay": None,
            "idle_check": None,
            "max_restarts": 0,
            "drive_account": "main",
            "no_drive": True,
            "domain": None,
            "base_domain": None,
            "quick_tunnel": True,
            "transport": None,
            "gpu": None,
        })()
        with self.assertRaisesRegex(ValueError, "--drive-account"):
            svc._start_up(service, ROOT / "src/colabllm/services/ollama.py", args)


class MultiAccountSubprocessLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = self.root / "colab-cli"
        _write_registry(
            self.cfg,
            active="gpu1",
            default_drive="main",
            accounts={
                "main": {"email": "main@example.com"},
                "gpu1": {"email": "gpu1@example.com"},
                "gpu2": {"email": "gpu2@example.com"},
            },
        )
        root_dir = self.root

        class Health(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                code = 502 if (root_dir / "http-502").exists() else 200
                self.send_response(code)
                self.end_headers()
                self.wfile.write(b"ok" if code == 200 else b"bad gateway")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

        fake_colab = self.root / "colab"
        fake_colab.write_text('''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time

root = Path(os.environ['FAKE_COLAB_ROOT'])
record = {
    'argv': sys.argv[1:],
    'account': os.environ.get('COLAB_ACCOUNT'),
    'drive_account': os.environ.get('COLAB_DRIVE_ACCOUNT'),
}
with (root / 'colab-calls.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(record) + '\\n')

cmd = sys.argv[1] if len(sys.argv) > 1 else ''
if cmd == 'status':
    sname = sys.argv[sys.argv.index('-s') + 1] if '-s' in sys.argv else 'ollama'
    if (root / f'missing-{sname}').exists():
        print(f"[colab] Session '{sname}' not found.", flush=True)
    else:
        print(f"[{sname}] gpu-t4-s-fake | Hardware: T4 | Status: BUSY", flush=True)
    sys.exit(0)
if cmd == 'new':
    sname = sys.argv[sys.argv.index('-s') + 1] if '-s' in sys.argv else 'ollama'
    (root / f'missing-{sname}').unlink(missing_ok=True)
    sys.exit(0)
if cmd == 'dfs':
    sys.exit(0)
if cmd == 'exec':
    sname = sys.argv[sys.argv.index('-s') + 1] if '-s' in sys.argv else 'ollama'
    print('CT_PUBLIC_URL=' + os.environ['FAKE_COLAB_URL'], flush=True)
    print('CT_READY', flush=True)
    if (root / f'crash-once-{sname}').exists():
        (root / f'crash-once-{sname}').unlink()
        (root / f'missing-{sname}').touch()
        time.sleep(0.05)
        sys.exit(1)
    while True:
        time.sleep(1)
''', encoding="utf-8")
        fake_colab.chmod(0o755)
        self.profile = self.root / "profile"
        self.env = {
            **{
                k: v for k, v in os.environ.items()
                if k not in ("CT_INSTANCE", "COLABLLM_ACCOUNT", "COLAB_ACCOUNT", "COLABLLM_DRIVE_ACCOUNT", "COLAB_DRIVE_ACCOUNT", "COLAB_DFS_ACCOUNT")
            },
            "COLABLLM_HOME": str(self.profile),
            "COLAB_CLI_CONFIG_DIR": str(self.cfg),
            "FAKE_COLAB_ROOT": str(self.root),
            "FAKE_COLAB_URL": f"http://127.0.0.1:{self.server.server_port}",
            "PATH": str(self.root) + os.pathsep + os.environ.get("PATH", ""),
        }
        self.addCleanup(self._cleanup_runners)

    def _cleanup_runners(self):
        services_dir = self.profile / "services"
        if not services_dir.is_dir():
            return
        for d in services_dir.iterdir():
            sp = d / "state.json"
            if sp.is_file():
                try:
                    st = json.loads(sp.read_text(encoding="utf-8"))
                except Exception:
                    continue
                for prefix in ("exec", "runner"):
                    stop_owned_process(st.get(prefix + "_pid"), st.get(prefix + "_identity"), group=True)

    def cli(self, *args):
        return subprocess.run([str(ROOT / "colabllm"), *args], env=self.env, cwd=self.root, text=True, capture_output=True, timeout=15)

    def calls(self):
        path = self.root / "colab-calls.jsonl"
        if not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def wait_state(self, inst_name: str, predicate, timeout: float = 10.0):
        sp = self.profile / "services" / inst_name / "state.json"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if sp.is_file():
                st = json.loads(sp.read_text(encoding="utf-8"))
                if predicate(st):
                    return st
            time.sleep(0.03)
        self.fail(f"timed out waiting for {inst_name} state condition")

    def test_pinned_accounts_auto_restart_active_switch_and_down_all(self):
        # 1. Start -i 1 explicitly on gpu1 + shared drive main, with --auto-restart
        (self.root / "crash-once-ollama-one").touch()
        up1 = self.cli(
            "ollama", "up", "-i", "one", "-a", "gpu1", "--drive-account", "main",
            "--auto-restart", "--restart-delay", ".05", "--quick-tunnel", "--detach",
        )
        self.assertEqual(up1.returncode, 0, up1.stderr)
        st1 = self.wait_state(
            "ollama-one",
            lambda s: s.get("ready") and int(s.get("restart_count") or 0) >= 1 and s.get("exec_identity"),
        )
        self.assertEqual(st1["account"], "gpu1")
        self.assertEqual(st1["drive_account"], "main")

        # 2. Switch active account in accounts.json to gpu2 AFTER -i one started
        _write_registry(
            self.cfg,
            active="gpu2",
            default_drive="main",
            accounts={
                "main": {"email": "main@example.com"},
                "gpu1": {"email": "gpu1@example.com"},
                "gpu2": {"email": "gpu2@example.com"},
            },
        )
        # Crash -i one again after active account switched -> auto-restart MUST still recreate on gpu1!
        stop_owned_process(st1.get("exec_pid"), st1.get("exec_identity"), group=True)
        (self.root / "missing-ollama-one").touch()
        st1_after = self.wait_state(
            "ollama-one",
            lambda s: s.get("ready") and int(s.get("restart_count") or 0) >= 2 and s.get("exec_identity"),
        )
        self.assertEqual(st1_after["account"], "gpu1")
        self.assertEqual(st1_after["drive_account"], "main")

        # Every colab call for ollama-one so far must have used gpu1 + main
        one_calls = [c for c in self.calls() if "ollama-one" in c["argv"]]
        self.assertTrue(one_calls)
        self.assertTrue(all(c["account"] == "gpu1" and c["drive_account"] == "main" for c in one_calls), one_calls)

        # Status on ollama-one continues to report gpu1 + main despite active=gpu2
        st_res = self.cli("ollama", "status", "-i", "one", "--json")
        self.assertEqual(st_res.returncode, 0, st_res.stderr)
        st_json = json.loads(st_res.stdout)
        self.assertEqual(st_json["account"], "gpu1")
        self.assertEqual(st_json["drive_account"], "main")

        # 3. Start -i two WITHOUT -a -> uses new active account gpu2 and default_drive_account main
        up2 = self.cli("ollama", "up", "-i", "two", "--quick-tunnel", "--detach")
        self.assertEqual(up2.returncode, 0, up2.stderr)
        self.wait_state("ollama-two", lambda s: s.get("ready") and s.get("exec_identity"))
        two_calls = [c for c in self.calls() if "ollama-two" in c["argv"]]
        self.assertTrue(two_calls)
        self.assertTrue(all(c["account"] == "gpu2" and c["drive_account"] == "main" for c in two_calls), two_calls)

        # ps --json exposes accounts cleanly without secrets
        ps_res = self.cli("ollama", "ps", "--json")
        self.assertEqual(ps_res.returncode, 0, ps_res.stderr)
        ps_items = {item["instance"]: item for item in json.loads(ps_res.stdout)}
        self.assertEqual(ps_items["one"]["account"], "gpu1")
        self.assertEqual(ps_items["one"]["drive_account"], "main")
        self.assertEqual(ps_items["two"]["account"], "gpu2")
        self.assertEqual(ps_items["two"]["drive_account"], "main")

        # 4. down --all stops each instance using its own pinned account
        down_all = self.cli("ollama", "down", "--all")
        self.assertEqual(down_all.returncode, 0, down_all.stderr)
        stop_one = [c for c in self.calls() if c["argv"] == ["stop", "-s", "ollama-one"]][-1]
        stop_two = [c for c in self.calls() if c["argv"] == ["stop", "-s", "ollama-two"]][-1]
        self.assertEqual(stop_one["account"], "gpu1")
        self.assertEqual(stop_two["account"], "gpu2")

        # 5. After down, starting -i one without -a now adopts active gpu2
        up1_again = self.cli("ollama", "up", "-i", "one", "--quick-tunnel", "--detach")
        self.assertEqual(up1_again.returncode, 0, up1_again.stderr)
        st1_new = self.wait_state("ollama-one", lambda s: s.get("ready") and s.get("account") == "gpu2")
        self.assertEqual(st1_new["account"], "gpu2")


class AccountFlagNonCollisionAndForwardingTest(unittest.TestCase):
    def test_minimax_short_a_remains_aspect_ratio_and_long_account_works(self):
        path = ROOT / "src/colabllm/services/minimax-h3-image.py"
        spec = importlib.util.spec_from_file_location("minimax_h3_image_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Exact regression check: `generate "cat" -a 16:9` must bind -a to aspect='16:9'
        inst1, argv1 = svc.extract_instance_arg(["generate", "cat", "-a", "16:9"])
        acct1, argv1 = svc.extract_account_arg(argv1)
        p1 = svc.build_parser(svc.Service(**vars(mod.SERVICE)), mod.configure_cli)
        parsed1 = p1.parse_args(argv1)
        self.assertIsNone(inst1)
        self.assertIsNone(acct1)
        self.assertEqual(parsed1.aspect, "16:9")

        # `--account gpu1` before or after subcommand does NOT collide with `generate -a 16:9`
        inst2, argv2 = svc.extract_instance_arg(["--account", "gpu1", "generate", "cat", "-a", "16:9"])
        acct2, argv2 = svc.extract_account_arg(argv2)
        parsed2 = p1.parse_args(argv2)
        self.assertEqual(acct2, "gpu1")
        self.assertEqual(parsed2.aspect, "16:9")

        # `up -a gpu1 --drive-account main` parses both on `up`
        inst3, argv3 = svc.extract_instance_arg(["up", "-a", "gpu1", "--drive-account", "main", "--dry-run"])
        acct3, argv3 = svc.extract_account_arg(argv3)
        parsed3 = p1.parse_args(argv3)
        self.assertIsNone(acct3)
        self.assertEqual(parsed3.account, "gpu1")
        self.assertEqual(parsed3.drive_account, "main")

    def test_global_cli_account_flag_forwarded_to_service_info(self):
        with tempfile.TemporaryDirectory() as home:
            result = subprocess.run(
                [str(ROOT / "colabllm"), "--account", "gpu1", "ollama", "info", "--json"],
                env={**os.environ, "COLABLLM_HOME": home},
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["account"], "gpu1")

    def test_nemotron_transcribe_passes_pinned_account_env_to_colab(self):
        path = ROOT / "src/colabllm/services/nemotron-transcribe.py"
        spec = importlib.util.spec_from_file_location("nemotron_transcribe_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with tempfile.TemporaryDirectory() as home, \
             patch.object(svc, "STATE_ROOT", Path(home)), \
             patch.object(mod, "colab_env_for_session", wraps=svc.colab_env_for_session):
            state_dir = Path(home) / "services/nemotron-transcribe-2"
            state_dir.mkdir(parents=True)
            (state_dir / "state.json").write_text(json.dumps({"account": "gpu2", "drive_account": "main"}), encoding="utf-8")
            with patch.object(mod.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "dir/\naudio.wav\n", "")) as mock_run:
                dirs, files = mod._colab_ls_single("nemotron-transcribe-2", "/content/drive/MyDrive")
            self.assertEqual(dirs, ["dir"])
            self.assertEqual(files, ["audio.wav"])
            env_passed = mock_run.call_args.kwargs["env"]
            self.assertEqual(env_passed["COLAB_ACCOUNT"], "gpu2")
            self.assertEqual(env_passed["COLAB_DRIVE_ACCOUNT"], "main")


if __name__ == "__main__":
    unittest.main()
