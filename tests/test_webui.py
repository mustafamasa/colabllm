"""Local web control panel: catalog parsing, state reading and the HTTP security boundary."""
import http.client
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/colabllm/lib"))

import colabsvc  # noqa: E402
import webui  # noqa: E402


def op_command(popen):
    """The service command the panel ran, unwrapped from `op_runner.py EXIT_FILE -- ...`."""
    argv = popen.call_args.args[0]
    return argv[argv.index("--") + 1:]


def _killpg(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except OSError:
        pass


class CatalogTest(unittest.TestCase):
    def test_every_packaged_service_is_listed_with_static_metadata(self):
        names = {item["name"] for item in webui.catalog()}
        expected = {p.stem for p in (ROOT / "src/colabllm/services").glob("*.py") if p.with_suffix(".ipynb").is_file()}
        self.assertEqual(names, expected)
        for item in webui.catalog():
            self.assertIn(item["default_gpu"], (*webui.GPUS, "none"))
            self.assertNotIn("notebook:", item["summary"])
        minimax = next(item for item in webui.catalog() if item["name"] == "minimax-h3-image")
        self.assertEqual(minimax["category"], "Image")
        self.assertIn("/mustafamasa/colabllm/blob/english/docs/", minimax["docs"])

    def test_assets_exist_and_have_no_inline_script(self):
        html = (webui.ASSETS_DIR / "index.html").read_text(encoding="utf-8")
        for asset in ("app.css", "app.js", "favicon.svg"):
            self.assertTrue((webui.ASSETS_DIR / asset).is_file())
        self.assertNotIn("<script>", html)
        self.assertNotIn(" style=", html)

    def test_assets_are_snapshotted_at_startup(self):
        self.assertEqual(set(webui.ASSETS), {"index.html", "app.css", "app.js", "favicon.svg"})
        self.assertEqual(webui.ASSETS["app.js"][1], "text/javascript; charset=utf-8")

    def test_async_handlers_do_not_touch_current_target_after_await(self):
        # event.currentTarget is null after an await, which left buttons disabled and modals stuck on errors.
        js = (webui.ASSETS_DIR / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("e.currentTarget.disabled = false", js)


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        home = Path(cls.tmp.name)
        cls.patch = patch.object(colabsvc, "STATE_ROOT", home)
        cls.patch.start()
        directory = home / "services" / "ollama-2"
        directory.mkdir(parents=True)
        (directory / "state.json").write_text(json.dumps({
            "base_service": "ollama", "auth_user": "admin", "auth_pass": "pw-secret", "api_token": "tok-secret",
            "url": "https://x.trycloudflare.com", "runner_pid": 999999, "runner_identity": "nope",
        }))
        (directory / "run.log").write_text("\x1b[32mline one\x1b[0m\nline two\n")
        cls.token = "test-token"
        cls.server = webui.Server(("127.0.0.1", 0), webui.Handler)
        cls.port = cls.server.server_address[1]
        webui.Handler.token = cls.token
        webui.Handler.allowed_hosts = {f"127.0.0.1:{cls.port}", f"localhost:{cls.port}"}
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.patch.stop()
        cls.tmp.cleanup()

    def request(self, method, path, *, token=True, host=None, origin=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        headers = {"Host": host or f"127.0.0.1:{self.port}"}
        if token:
            headers["X-Colabllm-Token"] = self.token if token is True else token
        if origin:
            headers["Origin"] = origin
        payload = None
        if body is not None:
            payload = json.dumps(body)
            headers["Content-Type"] = "application/json"
        conn.request(method, path, body=payload, headers=headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response, data

    def test_static_page_is_public_but_api_requires_token(self):
        response, data = self.request("GET", "/", token=False)
        self.assertEqual(response.status, 200)
        self.assertIn(b"colabllm", data)
        self.assertIn("default-src 'self'", response.getheader("Content-Security-Policy"))
        self.assertEqual(self.request("GET", "/api/overview", token=False)[0].status, 401)
        self.assertEqual(self.request("GET", "/api/overview", token="wrong")[0].status, 401)

    def test_rejects_foreign_host_and_cross_origin_posts(self):
        self.assertEqual(self.request("GET", "/api/overview", host="evil.example:80")[0].status, 421)
        response, _ = self.request("POST", "/api/instances/ollama-2/stop", origin="https://evil.example", body={})
        self.assertEqual(response.status, 403)

    def test_tunnel_config_requires_auth_and_validates_domains(self):
        self.assertEqual(self.request("POST", "/api/tunnel-config", token=False, body={"base_domain": "example.com"})[0].status, 401)
        self.assertEqual(self.request("POST", "/api/tunnel-config", origin="https://evil.example", body={"base_domain": "example.com"})[0].status, 403)
        self.assertEqual(self.request("POST", "/api/tunnel-config", body={"base_domain": "https://example.com"})[0].status, 400)
        response, data = self.request("POST", "/api/tunnel-config", body={"base_domain": "Example.COM."})
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(data)["saved_base_domain"], "example.com")
        self.assertEqual(colabsvc.load_tunnel_config()["base_domain"], "example.com")
        response, data = self.request("POST", "/api/tunnel-config", body={"base_domain": ""})
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(data)["transport"], "quick")

    def test_tunnel_config_respects_environment_override(self):
        with patch.dict(os.environ, {"COLABSVC_BASE_DOMAIN": "override.example"}):
            response, data = self.request("POST", "/api/tunnel-config", body={"base_domain": "saved.example"})
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(data)["base_domain"], "override.example")
        self.assertEqual(json.loads(data)["saved_base_domain"], "saved.example")
        colabsvc.update_tunnel_config(clear=True)

    def test_cloudflare_status_does_not_expose_command_output(self):
        with patch.object(webui.shutil, 'which', return_value='/usr/bin/cf'), \
             patch.object(webui.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=json.dumps({
                 'authenticated': True, 'email': 'private@example.com', 'accounts': [{'id': 'private-id'}]
             }))) as run:
            response, data = self.request('GET', '/api/cloudflare')
        self.assertEqual(response.status, 200)
        self.assertTrue(json.loads(data)['authorized'])
        self.assertNotIn(b'private', data)
        self.assertEqual(run.call_args.args[0], ['cf', 'auth', 'whoami', '--json'])
        self.assertIs(run.call_args.kwargs['stdout'], subprocess.PIPE)

    def test_cloudflare_status_requires_authenticated_json_even_on_success(self):
        for output in ('{"authenticated": false}', '{"authenticated": true, "tokenValid": false}', 'not json'):
            with self.subTest(output=output), patch.object(webui.shutil, 'which', return_value='/usr/bin/cf'), \
                 patch.object(webui.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=output)):
                response, data = self.request('GET', '/api/cloudflare')
            self.assertEqual(response.status, 200)
            self.assertFalse(json.loads(data)['authorized'])

    def test_cloudflare_zones_only_return_valid_names_and_status(self):
        self.assertEqual(self.request('GET', '/api/cloudflare/zones', token=False)[0].status, 401)
        data = json.dumps([
            {'name': 'Example.COM', 'status': 'active', 'id': 'private-id', 'account': {'id': 'private-account'}},
            {'name': 'https://bad.example', 'status': 'active'},
            {'name': 'pending.example', 'status': 'pending'},
        ])
        with patch.object(webui.shutil, 'which', return_value='/usr/bin/cf'), \
             patch.object(webui.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, stdout=data)) as run:
            response, raw = self.request('GET', '/api/cloudflare/zones')
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(raw), {'zones': [{'name': 'example.com', 'status': 'active'},
                                                    {'name': 'pending.example', 'status': 'pending'}]})
        self.assertNotIn(b'private-id', raw)
        self.assertEqual(run.call_args.args[0], ['cf', 'zones', 'list'])

    def test_cloudflare_zones_hides_cli_errors(self):
        with patch.object(webui.shutil, 'which', return_value='/usr/bin/cf'), \
             patch.object(webui.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, stdout='secret')):
            response, data = self.request('GET', '/api/cloudflare/zones')
        self.assertEqual(response.status, 500)
        self.assertNotIn(b'secret', data)

    def test_cloudflare_login_requires_auth_and_local_cli(self):
        self.assertEqual(self.request('POST', '/api/cloudflare/login', token=False, body={})[0].status, 401)
        with patch.object(webui.shutil, 'which', return_value=None):
            self.assertEqual(self.request('POST', '/api/cloudflare/login', body={})[0].status, 400)

    def test_cloudflare_logout_is_authorized_and_checks_remaining_credentials(self):
        self.assertEqual(self.request('POST', '/api/cloudflare/logout', token=False, body={})[0].status, 401)
        self.assertEqual(self.request('POST', '/api/cloudflare/logout', origin='https://evil.example', body={})[0].status, 403)
        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, stdout='{"authenticated": false}')
        with patch.object(webui.shutil, 'which', return_value='/usr/bin/cf'), \
             patch.object(webui.subprocess, 'run', side_effect=fake_run) as run:
            response, data = self.request('POST', '/api/cloudflare/logout', body={})
        self.assertEqual(response.status, 200)
        self.assertFalse(json.loads(data)['authorized'])
        self.assertEqual(run.call_args_list[0].args[0], ['cf', 'auth', 'logout'])
        self.assertEqual(run.call_args_list[1].args[0], ['cf', 'auth', 'whoami', '--json'])
        self.assertIs(run.call_args_list[0].kwargs['stdout'], subprocess.DEVNULL)

    def test_cloudflare_logout_failure_and_concurrent_login_are_safe(self):
        with patch.object(webui.shutil, 'which', return_value='/usr/bin/cf'), \
             patch.object(webui.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1)):
            self.assertEqual(self.request('POST', '/api/cloudflare/logout', body={})[0].status, 409)
        with webui._cf_login_lock:
            webui._cf_login['state'] = 'running'
        try:
            with patch.object(webui.shutil, 'which', return_value='/usr/bin/cf'), \
                 patch.object(webui.subprocess, 'run') as run:
                self.assertEqual(self.request('POST', '/api/cloudflare/logout', body={})[0].status, 409)
                run.assert_not_called()
        finally:
            with webui._cf_login_lock:
                webui._cf_login['state'] = 'idle'

    def test_cloudflare_login_runs_browser_callback_and_checks_identity(self):
        completed = threading.Event()
        calls = []
        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            if command[2] == 'whoami':
                completed.set()
            return subprocess.CompletedProcess(command, 0, stdout='{"authenticated": true}')
        with patch.object(webui.shutil, 'which', return_value='/usr/bin/cf'), \
             patch.object(webui.subprocess, 'run', side_effect=fake_run):
            response, _ = self.request('POST', '/api/cloudflare/login', body={})
            self.assertEqual(response.status, 202)
            self.assertTrue(completed.wait(2))
            for _ in range(50):
                with webui._cf_login_lock:
                    state = webui._cf_login['state']
                if state != 'running':
                    break
                time.sleep(0.01)
        self.assertEqual(state, 'succeeded')
        self.assertEqual(calls[0][0], ['cf', 'auth', 'login', '--no-device'])
        self.assertEqual(calls[1][0], ['cf', 'auth', 'whoami', '--json'])
        self.assertIs(calls[0][1]['stdout'], subprocess.DEVNULL)
        with webui._cf_login_lock:
            webui._cf_login['state'] = 'idle'

    def test_tunnel_clear_keeps_enabled_frp_configuration(self):
        config = {'base_domain': 'example.com', 'frp': {'enabled': True}}
        with patch.object(colabsvc, 'load_tunnel_config', return_value=config), \
             patch.object(colabsvc, 'save_tunnel_config') as save:
            response, _ = self.request('POST', '/api/tunnel-config', body={'base_domain': ''})
        self.assertEqual(response.status, 400)
        save.assert_not_called()

    def test_static_paths_cannot_escape_assets(self):
        for path in ("/../lib/webui.py", "/%2e%2e/cli.py", "/app.py", "/nested/app.js"):
            self.assertEqual(self.request("GET", path, token=False)[0].status, 404, path)

    def test_overview_strips_credentials(self):
        response, data = self.request("GET", "/api/overview")
        self.assertEqual(response.status, 200)
        self.assertNotIn(b"pw-secret", data)
        self.assertNotIn(b"tok-secret", data)
        overview = json.loads(data)
        row = next(r for r in overview["instances"] if r["id"] == "ollama-2")
        self.assertEqual((row["service"], row["instance"], row["status"]), ("ollama", "2", "stopped"))
        self.assertIsNone(row["url"])
        self.assertTrue(row["has_secrets"])

    def test_secrets_and_logs_endpoints(self):
        secrets = json.loads(self.request("GET", "/api/instances/ollama-2/secrets")[1])
        self.assertEqual(secrets["auth_pass"], "pw-secret")
        log = json.loads(self.request("GET", "/api/instances/ollama-2/logs")[1])
        self.assertEqual(log["text"], "line one\nline two\n")
        self.assertEqual(self.request("GET", "/api/instances/..%2Fx/secrets")[0].status, 404)
        self.assertEqual(self.request("GET", "/api/instances/missing/logs")[0].status, 404)

    def test_deploy_validates_input_before_running(self):
        with patch.object(webui, "_popen") as popen:
            self.assertEqual(self.request("POST", "/api/services/nope/deploy", body={})[0].status, 404)
            self.assertEqual(self.request("POST", "/api/services/ollama/deploy", body={"gpu": "V100"})[0].status, 400)
            self.assertEqual(self.request("POST", "/api/services/ollama/deploy", body={"instance": "-x"})[0].status, 400)
            popen.assert_not_called()

    def test_deploy_runs_the_packaged_script_detached(self):
        class FakeProc:
            stdout = iter(["starting\n"])
            def wait(self):
                return 0
        with patch.object(webui, "_popen", return_value=FakeProc()) as popen:
            response, data = self.request("POST", "/api/services/ollama/deploy",
                                          body={"instance": "7", "gpu": "L4", "quick_tunnel": True, "no_drive": True})
            self.assertEqual(response.status, 202)
            cmd = op_command(popen)
            self.assertEqual(cmd[:3], [sys.executable, str(webui.SERVICES_DIR / "ollama.py"), "--no-color"])
            self.assertEqual(cmd[3:], ["up", "--detach", "-i", "7", "--gpu", "L4", "--quick-tunnel", "--no-drive"])
            op = json.loads(data)
            self.assertEqual(op["target"], "ollama-7")
            for _ in range(50):
                if webui.OPS.get(op["id"])["state"] != "running":
                    break
                time.sleep(0.02)
            self.assertEqual(webui.OPS.get(op["id"])["state"], "succeeded")

    def test_stop_targets_the_instance(self):
        class FakeProc:
            stdout = iter([])
            def wait(self):
                return 0
        with patch.object(webui, "_popen", return_value=FakeProc()) as popen:
            response, _ = self.request("POST", "/api/instances/ollama-2/stop", body={"delete_tunnel": True})
            self.assertEqual(response.status, 202)
            self.assertEqual(op_command(popen)[3:], ["down", "-i", "2", "--delete-tunnel"])


    def test_options_schema_endpoint_reads_the_cli_parser(self):
        data = json.loads(self.request("GET", "/api/services/comfyui/options")[1])
        self.assertEqual([(o["key"], o["kind"]) for o in data["options"]], [("--workflow", "list")])
        self.assertEqual(self.request("GET", "/api/services/nope/options")[0].status, 404)

    def test_deploy_passes_validated_service_options(self):
        class FakeProc:
            stdout = iter([])
            def wait(self):
                return 0
        with patch.object(webui, "_popen", return_value=FakeProc()) as popen:
            response, _ = self.request("POST", "/api/services/comfyui/deploy", body={
                "instance": "wf", "options": {"--workflow": ["https://example.com/a.json", "flux-dev"]}})
            self.assertEqual(response.status, 202)
            self.assertEqual(op_command(popen)[-2:], ["--workflow=https://example.com/a.json", "--workflow=flux-dev"])
            response, data = self.request("POST", "/api/services/ollama/deploy", body={"options": {"--context-length": "lots"}})
            self.assertEqual(response.status, 400)
            self.assertIn("invalid int value", json.loads(data)["error"])
            self.assertEqual(self.request("POST", "/api/services/ollama/deploy", body={"options": {"--profile": "turbo"}})[0].status, 400)
            self.assertEqual(self.request("POST", "/api/services/ollama/deploy", body={"options": {"--gpu": "A100"}})[0].status, 400)
            self.assertEqual(popen.call_count, 1)

    def test_bulk_stop_reports_each_instance(self):
        class FakeProc:
            stdout = iter([])
            def wait(self):
                return 0
        with patch.object(webui, "_popen", return_value=FakeProc()) as popen:
            response, data = self.request("POST", "/api/instances/stop", body={"ids": ["ollama-2", "missing", "ollama-2"]})
            self.assertEqual(response.status, 202)
            results = json.loads(data)["results"]
            self.assertEqual([(r["id"], r["ok"]) for r in results], [("ollama-2", True), ("missing", False)])
            self.assertEqual(popen.call_count, 1)
            self.assertEqual(self.request("POST", "/api/instances/stop", body={"ids": []})[0].status, 400)


    def test_delete_runs_cli_purge_for_stopped_instances(self):
        class FakeProc:
            stdout = iter([])
            def wait(self):
                return 0
        with patch.object(webui, "_popen", return_value=FakeProc()) as popen:
            response, data = self.request("POST", "/api/instances/ollama-2/delete", body={})
            self.assertEqual(response.status, 202, data)
            self.assertEqual(op_command(popen)[3:], ["purge", "-i", "2"])
            response, data = self.request("POST", "/api/instances/delete", body={"ids": ["ollama-2", "missing"]})
            self.assertEqual([(r["id"], r["ok"]) for r in json.loads(data)["results"]], [("ollama-2", True), ("missing", False)])

    def test_delete_refuses_running_instances(self):
        running = {"id": "ollama-2", "service": "ollama", "instance_arg": "2", "status": "ready", "owns_tunnel": False}
        with patch.object(webui, "instances", return_value=[running]), patch.object(webui, "_popen") as popen:
            response, data = self.request("POST", "/api/instances/ollama-2/delete", body={})
            self.assertEqual(response.status, 409)
            self.assertIn("still running", json.loads(data)["error"])
            popen.assert_not_called()

    def test_delete_passes_tunnel_choice(self):
        stopped = {"id": "ollama-2", "service": "ollama", "instance_arg": "2", "status": "stopped", "owns_tunnel": True}
        class FakeProc:
            stdout = iter([])
            def wait(self):
                return 0
        with patch.object(webui, "instances", return_value=[stopped]), patch.object(webui, "_popen", return_value=FakeProc()) as popen:
            self.request("POST", "/api/instances/ollama-2/delete", body={"delete_tunnel": False})
            self.assertEqual(op_command(popen)[-1], "--keep-tunnel")
            time.sleep(0.05)
            self.request("POST", "/api/instances/ollama-2/delete", body={})
            self.assertEqual(op_command(popen)[-1], "--delete-tunnel")


class PersistentOperationsTest(unittest.TestCase):
    """Real op_runner processes: operations outlive the panel and can be cancelled after a restart."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        services = root / "services-src"
        services.mkdir()
        (services / "fake.py").write_text(
            "import sys, time\n"
            "print('waiting for G4: ' + ' '.join(sys.argv[1:]), flush=True)\n"
            "time.sleep(0 if 'down' in sys.argv else 30)\n")
        for patcher in (patch.object(colabsvc, "STATE_ROOT", root / "home"), patch.object(webui, "SERVICES_DIR", services)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def wait_for(self, ops, op_id, predicate, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            op = ops.get(op_id)
            if predicate(op):
                return op
            time.sleep(0.05)
        self.fail(f"operation never reached the expected state: {ops.get(op_id)}")

    def test_deploy_survives_panel_restart_and_cancels(self):
        panel = webui.Operations()
        op = panel.start("fake", "deploy", ["up", "--detach"], "fake", "Deploy fake")
        pid = json.loads((colabsvc.STATE_ROOT / "ui/operations" / f"{op['id']}.json").read_text())["pid"]
        self.addCleanup(_killpg, pid)
        self.assertEqual(os.getsid(pid), pid, "the operation must run in its own session")
        self.assertNotIn("pid", op)

        reopened = webui.Operations()  # a new panel process reads the persisted operation
        running = self.wait_for(reopened, op["id"], lambda o: "waiting for G4" in o["output"])
        self.assertEqual((running["state"], running["cancellable"]), ("running", True))
        self.assertTrue(reopened.busy("fake"))
        with self.assertRaisesRegex(RuntimeError, "already running"):
            reopened.start("fake", "deploy", ["up"], "fake", "again")

        reopened.cancel(op["id"])
        done = self.wait_for(reopened, op["id"], lambda o: o["state"] != "running")
        self.assertEqual((done["state"], done["exit_code"], done["cancellable"]), ("cancelled", 143, False))
        with self.assertRaisesRegex(RuntimeError, "already finished"):
            reopened.cancel(op["id"])

    def test_only_running_deployments_are_cancellable(self):
        panel = webui.Operations()
        op = panel.start("fake", "stop", ["down"], "fake", "Stop fake")
        done = self.wait_for(panel, op["id"], lambda o: o["state"] != "running")
        self.assertEqual((done["state"], done["exit_code"]), ("succeeded", 0))
        with self.assertRaises(LookupError):
            panel.cancel("../../etc")


class ServiceOptionsTest(unittest.TestCase):
    def test_every_service_exposes_a_valid_schema(self):
        for item in webui.catalog():
            with self.subTest(service=item["name"]):
                schema = webui.option_schema(item["name"])
                keys = [o["key"] for o in schema]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertFalse({"--gpu", "--detach", "--domain", "--hf-token"} & set(keys))

    def test_option_kinds_map_to_argv(self):
        args = webui.service_option_args("ollama", {
            "--models": ["qwen3:8b", "llama3"], "--preload": "off", "--profile": "coding",
            "--context-length": "32768", "--read-only-models": True, "--no-cloud": False})
        self.assertEqual(args, ["--models=qwen3:8b", "--models=llama3", "--no-preload", "--profile=coding",
                                "--context-length=32768", "--read-only-models"])
        self.assertEqual(webui.service_option_args("deepseek-coder", {"--context": "auto"}), ["--context=auto"])

    def test_rejects_unsafe_or_invalid_values(self):
        for options in ({"--models": ["a\nb"]}, {"--unknown": "x"}, {"--profile": "x"}, {"--read-only-models": "yes"},
                        {"--models": ["m"] * 60}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                webui.service_option_args("ollama", options)
        with self.assertRaises(ValueError):
            webui.service_option_args("deepseek-coder", {"--context": "huge"})
        # A value that looks like a flag stays a value because it is passed as --flag=value.
        self.assertEqual(webui.service_option_args("ollama", {"--ollama-version": "--debug"}), ["--ollama-version=--debug"])


class UsageParsingTest(unittest.TestCase):
    def test_parses_colab_usage_output(self):
        class Result:
            returncode = 0
            stdout = "Current balance: 2402.09 compute units\nUsage rate: 1.50/hr\nActive assignments: 1\n"
        with patch.object(webui.subprocess, "run", return_value=Result()):
            data = webui.account_usage(force=True)
        self.assertEqual((data["balance"], data["rate"], data["assignments"]), (2402.09, 1.5, 1))


if __name__ == "__main__":
    unittest.main()
