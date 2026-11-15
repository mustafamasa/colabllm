"""Exercise real CLI/runner subprocesses using a fake Colab CLI and local HTTP server."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/colabllm/lib'))
from processes import owned_process, stop_owned_process


class RunnerLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        root_dir = self.root
        class Health(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                code = 502 if (root_dir / 'http-502').exists() else 200
                self.send_response(code); self.end_headers(); self.wfile.write(b'ok' if code == 200 else b'bad gateway')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Health)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        executable = self.root / 'colab'
        executable.write_text('''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys
import time
root = Path(os.environ['FAKE_COLAB_ROOT'])
with (root / 'colab-calls.jsonl').open('a', encoding='utf-8') as logf:
    logf.write(json.dumps(sys.argv[1:]) + '\\n')
cmd = sys.argv[1] if len(sys.argv) > 1 else ''
if cmd == 'status':
    if (root / 'session-missing').exists():
        print("[colab] Session 'ollama-test' not found.", flush=True)
    elif (root / 'session-alive').exists():
        print("[ollama-test] gpu-t4-s-fake | Hardware: T4 | Status: BUSY", flush=True)
    sys.exit(0)
if cmd == 'new':
    if (root / 'fail-new-once').exists():
        (root / 'fail-new-once').unlink()
        print('GPU capacity temporarily unavailable', file=sys.stderr)
        sys.exit(1)
    if (root / 'new-clears-502').exists():
        (root / 'http-502').unlink(missing_ok=True)
    (root / 'session-missing').unlink(missing_ok=True)
    sys.exit(0)
if cmd == 'exec':
    count_file = root / 'exec-count'
    count = int(count_file.read_text()) + 1 if count_file.exists() else 1
    count_file.write_text(str(count))
    if (root / 'exit-before-ready').exists():
        sys.exit(0)
    if count >= 2 and (root / 'queued-behind-busy-kernel').exists():
        # Simulate re-attaching `colab exec` to a live VM whose Jupyter kernel is still blocked in Cell 5
        while True:
            time.sleep(1)
    print('CT_PUBLIC_URL=' + os.environ['FAKE_COLAB_URL'], flush=True)
    if (root / 'false-ready').exists():
        print('ERROR: expected CT_READY but startup failed', flush=True)
        sys.exit(0)
    if not (root / 'never-ready').exists():
        print('CT_READY', flush=True)
    if (root / 'crash-once-after-ready').exists():
        (root / 'crash-once-after-ready').unlink()
        time.sleep(0.05)
        sys.exit(1)
    if (root / 'crash-always-after-ready').exists():
        time.sleep(0.05)
        sys.exit(1)
    while True:
        time.sleep(1)
if cmd == 'stop' and (root / 'fail-stop').exists():
    print('simulated remote failure', file=sys.stderr)
    sys.exit(42)
''')
        executable.chmod(0o755)
        self.env = {**os.environ, 'COLABLLM_HOME': str(self.root/'profile'),
            'FAKE_COLAB_ROOT': str(self.root), 'FAKE_COLAB_URL': f'http://127.0.0.1:{self.server.server_port}',
            'PATH': str(self.root) + os.pathsep + os.environ.get('PATH', '')}
        self.env.pop('CT_INSTANCE', None)
        self.state_path = self.root/'profile/services/ollama-test/state.json'
        self.addCleanup(self.stop_children)

    def stop_children(self):
        if self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            for prefix in ('exec', 'runner'):
                stop_owned_process(st.get(prefix+'_pid'), st.get(prefix+'_identity'), group=True)

    def cli(self, *args):
        return subprocess.run([str(ROOT/'colabllm'), 'ollama', *args, '-i', 'test'], env=self.env,
            cwd=self.root, text=True, capture_output=True, timeout=10)

    def wait_ready(self):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if self.state_path.exists():
                st = json.loads(self.state_path.read_text())
                if st.get('ready') and st.get('exec_identity'):
                    return st
            time.sleep(0.02)
        self.fail('fake Colab runner never became ready')

    def test_background_up_status_failed_down_retry_and_cleanup(self):
        start = self.cli('up', '--quick-tunnel', '--no-drive', '--detach')
        self.assertEqual(start.returncode, 0, start.stderr)
        st = self.wait_ready()
        status = self.cli('status', '--json')
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertTrue(json.loads(status.stdout)['ready'])
        self.assertNotIn('api_token', status.stdout)
        (self.root/'fail-stop').touch()
        failed = self.cli('down')
        self.assertEqual(failed.returncode, 42, failed.stderr)
        self.assertTrue(owned_process(st['runner_pid'], st['runner_identity']))
        self.assertEqual(json.loads(self.state_path.read_text())['runner_pid'], st['runner_pid'])
        (self.root/'fail-stop').unlink()
        stop = self.cli('down')
        self.assertEqual(stop.returncode, 0, stop.stderr)
        self.assertFalse(owned_process(st['runner_pid'], st['runner_identity']))
        self.assertFalse(owned_process(st['exec_pid'], st['exec_identity']))
        self.assertNotIn('runner_pid', json.loads(self.state_path.read_text()))

    def test_foreground_early_zero_exit_is_still_a_startup_failure(self):
        (self.root/'exit-before-ready').touch()
        result = self.cli('up', '--quick-tunnel', '--no-drive', '--foreground')
        self.assertEqual(result.returncode, 1, result.stderr)
        st = json.loads(self.state_path.read_text())
        self.assertFalse(st['ready'])
        self.assertEqual(st['runner_exit'], 1)


    def test_log_mention_is_not_a_ready_signal(self):
        (self.root/'false-ready').touch()
        result = self.cli('up', '--quick-tunnel', '--no-drive', '--foreground')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(self.state_path.read_text())['ready'])

    def test_background_readiness_timeout_keeps_runner(self):
        (self.root/'never-ready').touch()
        result = self.cli('up', '--quick-tunnel', '--no-drive', '--ready-timeout', '.1')
        self.assertEqual(result.returncode, 124, result.stderr)
        st = json.loads(self.state_path.read_text())
        self.assertTrue(owned_process(st['runner_pid'], st['runner_identity']))
        self.assertFalse(st['ready'])
        self.assertEqual(self.cli('down').returncode, 0)

    def test_foreground_readiness_timeout_stops_connection(self):
        (self.root/'never-ready').touch()
        result = self.cli('up', '--quick-tunnel', '--no-drive', '--foreground', '--ready-timeout', '.1')
        self.assertEqual(result.returncode, 124, result.stderr)
        self.assertIsNone(json.loads(self.state_path.read_text())['runner_pid'])


    def test_foreground_ps_duplicate_rejection_and_down(self):
        front = subprocess.Popen([str(ROOT/'colabllm'), 'ollama', 'up', '--quick-tunnel', '--no-drive',
            '--foreground', '-i', 'test'], env=self.env, cwd=self.root,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: front.poll() is None and front.kill())
        st = self.wait_ready()
        self.assertEqual(st['runner_pid'], front.pid)
        listing = self.cli('ps', '--json')
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertTrue(json.loads(listing.stdout)[0]['running'])
        duplicate = self.cli('up', '--quick-tunnel', '--no-drive', '--detach')
        self.assertNotEqual(duplicate.returncode, 0)
        self.assertIn('already running', duplicate.stderr)
        stop = self.cli('down')
        self.assertEqual(stop.returncode, 0, stop.stderr)
        front.wait(timeout=5)
        self.assertFalse(owned_process(st['exec_pid'], st['exec_identity']))

    def colab_calls(self):
        path = self.root / 'colab-calls.jsonl'
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def wait_restart_ready(self, min_restarts=1):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if self.state_path.exists():
                st = json.loads(self.state_path.read_text())
                if st.get('ready') and int(st.get('restart_count') or 0) >= min_restarts and st.get('exec_identity'):
                    return st
            time.sleep(0.02)
        self.fail(f'auto-restart never reached restart_count >= {min_restarts}')

    def test_auto_restart_recreates_reclaimed_vm_and_stops_cleanly_on_down(self):
        (self.root / 'crash-once-after-ready').touch()
        (self.root / 'session-missing').touch()
        (self.root / 'fail-new-once').touch()
        start = self.cli(
            'up', '--quick-tunnel', '--gpu', 'L4', '--auto-restart', '--restart-delay', '.05', '--detach'
        )
        # Note: initial `up` consumes `fail-new-once`? Keep `fail-new-once` only for the restart cycle:
        self.assertNotEqual(start.returncode, 0)
        # Now run clean initial start and fail the first restart `colab new` once to verify backoff recovery:
        (self.root / 'colab-calls.jsonl').unlink(missing_ok=True)
        (self.root / 'crash-once-after-ready').touch()
        start = self.cli(
            'up', '--quick-tunnel', '--gpu', 'L4', '--auto-restart', '--restart-delay', '.05', '--detach'
        )
        self.assertEqual(start.returncode, 0, start.stderr)
        (self.root / 'session-missing').touch()
        (self.root / 'fail-new-once').touch()
        st = self.wait_restart_ready(min_restarts=1)
        self.assertTrue(st['auto_restart'])
        self.assertEqual(st['session_gpu'], 'L4')
        status = self.cli('status', '--json')
        self.assertEqual(status.returncode, 0, status.stderr)
        payload = json.loads(status.stdout)
        self.assertTrue(payload['ready'])
        self.assertTrue(payload['auto_restart'])
        self.assertEqual(payload['restart_count'], 1)

        calls = self.colab_calls()
        new_calls = [c for c in calls if c[:1] == ['new']]
        mount_calls = [c for c in calls if c[:2] == ['dfs', 'mount']]
        # 1 initial `new` + 1 failed restart `new` + 1 succeeded restart `new` = 3, all preserving `--gpu L4`
        self.assertEqual(len(new_calls), 3, calls)
        for call in new_calls:
            self.assertIn('L4', call)
        self.assertGreaterEqual(len(mount_calls), 2, calls)

        stop = self.cli('down')
        self.assertEqual(stop.returncode, 0, stop.stderr)
        self.assertFalse(owned_process(st['runner_pid'], st['runner_identity']))
        self.assertFalse(owned_process(st['exec_pid'], st['exec_identity']))

    def test_auto_restart_reuses_live_vm_on_transient_exec_disconnect(self):
        (self.root / 'crash-once-after-ready').touch()
        (self.root / 'session-alive').touch()
        (self.root / 'queued-behind-busy-kernel').touch()
        start = self.cli(
            'up', '--quick-tunnel', '--no-drive', '--auto-restart',
            '--restart-delay', '.05', '--ready-timeout', '.25', '--detach',
        )
        self.assertEqual(start.returncode, 0, start.stderr)
        st = self.wait_restart_ready(min_restarts=1)
        status = self.cli('status', '--json')
        self.assertEqual(status.returncode, 0, status.stderr)
        payload = json.loads(status.stdout)
        self.assertTrue(payload['ready'])
        self.assertEqual(payload['http_status'], '200')
        # Ensure ready_timeout (.25s) does NOT tear down an already-healthy re-attached VM:
        time.sleep(0.35)
        self.assertEqual(json.loads(self.state_path.read_text())['restart_count'], 1)
        calls = self.colab_calls()
        new_calls = [c for c in calls if c[:1] == ['new']]
        stop_calls = [c for c in calls if c[:1] == ['stop']]
        # Initial `new` only; transient disconnect with `session-alive` must NOT destroy or recreate the VM!
        self.assertEqual(len(new_calls), 1, calls)
        self.assertEqual(len(stop_calls), 0, calls)
        self.assertEqual(self.cli('down').returncode, 0)
        self.assertFalse(owned_process(st['runner_pid'], st['runner_identity']))

    def test_auto_restart_detects_half_open_dead_session_and_recovers(self):
        self.env['COLABLLM_IDLE_CHECK_SECONDS'] = '0.12'
        start = self.cli(
            'up', '--quick-tunnel', '--no-drive', '--auto-restart', '--restart-delay', '.05', '--detach'
        )
        self.assertEqual(start.returncode, 0, start.stderr)
        first = self.wait_ready()
        # Simulate remote Colab reclaiming the VM while local `colab exec` hangs silently (half-open TCP):
        (self.root / 'session-missing').touch()
        restarted = self.wait_restart_ready(min_restarts=1)
        self.assertNotEqual(restarted['exec_pid'], first['exec_pid'])
        self.assertEqual(restarted['api_token'], first['api_token'])
        self.assertEqual(self.cli('down').returncode, 0)

    def test_auto_restart_fast_edge_probe_detects_502_immediately_without_waiting_idle_check(self):
        # Keep idle_check huge (300s), so ONLY the fast edge probe (0.08s) can trigger immediate recovery:
        self.env['COLABLLM_IDLE_CHECK_SECONDS'] = '300'
        self.env['COLABLLM_FAST_PROBE_SECONDS'] = '0.08'
        start = self.cli('up', '--quick-tunnel', '--no-drive', '--auto-restart', '--detach')
        self.assertEqual(start.returncode, 0, start.stderr)
        first = self.wait_ready()

        # Subcase A: transient 502 while remote Colab VM is still ALIVE must NOT kill or recreate the VM:
        (self.root / 'session-alive').touch()
        (self.root / 'http-502').touch()
        time.sleep(0.25)
        self.assertEqual(json.loads(self.state_path.read_text()).get('restart_count', 0), 0)

        # Subcase B: 502 + remote Colab VM MISSING triggers immediate restart without waiting 300s:
        (self.root / 'http-502').unlink()
        time.sleep(0.12)  # 200 resets status check cooldown
        (self.root / 'session-alive').unlink()
        (self.root / 'new-clears-502').touch()
        (self.root / 'http-502').touch()
        (self.root / 'session-missing').touch()
        restarted = self.wait_restart_ready(min_restarts=1)
        self.assertNotEqual(restarted['exec_pid'], first['exec_pid'])
        self.assertEqual(self.cli('down').returncode, 0)

    def test_auto_restart_stops_after_consecutive_pre_ready_startup_failures(self):
        self.env['COLABLLM_MAX_STARTUP_RETRIES'] = '2'
        (self.root / 'exit-before-ready').touch()
        result = self.cli(
            'up', '--quick-tunnel', '--no-drive', '--foreground', '--auto-restart', '--restart-delay', '.02'
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        exec_calls = [c for c in self.colab_calls() if c[:1] == ['exec']]
        self.assertEqual(len(exec_calls), 2)

    def test_auto_restart_respects_max_restarts_cap(self):
        (self.root / 'crash-always-after-ready').touch()
        (self.root / 'session-missing').touch()
        result = self.cli(
            'up', '--quick-tunnel', '--no-drive', '--foreground',
            '--auto-restart', '--restart-delay', '.02', '--max-restarts', '1',
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        exec_calls = [c for c in self.colab_calls() if c[:1] == ['exec']]
        # 1 initial run + 1 allowed auto-restart = 2 total `colab exec` runs
        self.assertEqual(len(exec_calls), 2)


if __name__ == '__main__':
    unittest.main()
