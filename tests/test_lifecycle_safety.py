"""State and process lifecycle regressions, using isolated profiles and child processes."""
import argparse
import concurrent.futures
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/colabllm/lib'))
import colabsvc as svc
import processes
from cli_support import CliError


def save_key(profile, number):
    svc.STATE_ROOT = Path(profile)
    svc.State(svc.Service('test', 'test.ipynb', 9000)).save(**{f'key_{number}': number})


class LifecycleSafetyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(svc, 'STATE_ROOT', self.root / 'profile'))
        self.service = svc.Service('test', 'test.ipynb', 9000)
        self.state = svc.State(self.service)
        self.script = self.root / 'test.py'
        self.script.write_text('')
        self.script.with_suffix('.ipynb').write_text(json.dumps({'cells': []}))

    def args(self, **kw):
        return argparse.Namespace(**dict(domain=None, base_domain=None, quick_tunnel=True, transport=None,
            gpu=None, skip='', dry_run=False, new_credentials=False, hf_token=None, no_drive=True,
            foreground=False, **kw))

    def test_concurrent_process_updates_keep_every_field(self):
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(save_key, str(svc.STATE_ROOT), n) for n in range(40)]
            for future in futures:
                future.result(timeout=15)
        self.assertEqual(self.state.load(), {f'key_{n}': n for n in range(40)})

    def test_concurrent_lifecycle_operation_is_rejected(self):
        with self.state.lock('operation'), self.assertRaisesRegex(RuntimeError, 'lifecycle'):
            svc.cmd_down(self.service, argparse.Namespace(all=False, delete_tunnel=False))

    def purge_args(self, **kw):
        return argparse.Namespace(**{'delete_tunnel': False, 'keep_tunnel': False, **kw})

    def test_purge_deletes_stopped_state_after_confirming_session_stop(self):
        self.state.save(api_token='test-only', url=None)
        self.state.log_path.write_text('log')
        with patch.object(svc, 'run', return_value=argparse.Namespace(returncode=0)) as run, patch('sys.stderr'):
            svc.cmd_purge(self.service, self.purge_args())
        run.assert_called_once_with(['colab', 'stop', '-s', 'test'], check=False)
        self.assertFalse(self.state.dir.exists())

    def test_purge_refuses_running_instance_and_keeps_state(self):
        before = self.state.save(runner_pid=os.getpid(), runner_identity=processes.process_identity(os.getpid()))
        with patch.object(svc, 'run') as run, self.assertRaisesRegex(CliError, 'still running'):
            svc.cmd_purge(self.service, self.purge_args())
        run.assert_not_called()
        self.assertEqual(self.state.load(), before)

    def test_purge_keeps_state_when_colab_stop_is_not_confirmed(self):
        before = self.state.save(api_token='test-only')
        with patch.object(svc, 'run', return_value=argparse.Namespace(returncode=3)), patch('sys.stderr'), \
             self.assertRaises(CliError) as error:
            svc.cmd_purge(self.service, self.purge_args())
        self.assertEqual(error.exception.code, 3)
        self.assertEqual(self.state.load(), before)

    def test_purge_requires_an_explicit_choice_for_owned_tunnels(self):
        self.state.save(tunnel_id='t-1', hostname='test.example.com')
        with patch.object(svc, 'run') as run, self.assertRaisesRegex(ValueError, '--delete-tunnel'):
            svc.cmd_purge(self.service, self.purge_args())
        run.assert_not_called()
        with patch.object(svc, 'run', return_value=argparse.Namespace(returncode=0)), \
             patch.object(svc, 'delete_named_tunnel') as delete, patch('sys.stderr'):
            svc.cmd_purge(self.service, self.purge_args(delete_tunnel=True))
        delete.assert_called_once()
        self.assertFalse(self.state.dir.exists())

    def test_purge_is_rejected_during_another_lifecycle_operation(self):
        self.state.save(api_token='test-only')
        with self.state.lock('operation'), self.assertRaisesRegex(RuntimeError, 'lifecycle'):
            svc.cmd_purge(self.service, self.purge_args())
        self.assertTrue(self.state.path.exists())

    def test_failed_down_preserves_all_tracking_and_does_not_kill_runner(self):
        before = self.state.save(runner_pid=2345, runner_identity='example', ready=True,
                                 url='https://example.invalid', api_token='test-only')
        with patch.object(svc, 'run', return_value=argparse.Namespace(returncode=42)), \
             patch.object(svc, 'stop_owned_process') as stop, self.assertRaises(CliError) as error:
            svc.cmd_down(self.service, argparse.Namespace(all=False, delete_tunnel=False))
        self.assertEqual(error.exception.code, 42)
        self.assertEqual(self.state.load(), before)
        stop.assert_not_called()

    def test_foreground_registers_running_identity_before_entering_runner(self):
        def run(service, notebook):
            st = self.state.load()
            self.assertEqual(st['runner_pid'], os.getpid())
            self.assertEqual(st['runner_mode'], 'foreground')
            self.assertTrue(svc.runner_alive(st))
            self.assertTrue(svc.list_service_instances('test')[0]['running'])
            # The operation lock must be available to a separate down command.
            with self.state.lock('operation', blocking=False):
                pass
            return 0
        args = self.args(); args.foreground = True
        with patch.object(svc.shutil, 'which', return_value='/bin/colab'), \
             patch.object(svc, 'ensure_session'), patch.object(svc, 'exec_runner', side_effect=run), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            svc.cmd_up(self.service, self.script, args)
        self.assertEqual(result.exception.code, 0)
        self.assertIsNone(self.state.load()['runner_pid'])

    def test_readiness_timeout_is_a_failure_and_preserves_background_state(self):
        self.service.ready_timeout = 0
        self.state.save(runner_pid=2345, runner_identity='example', ready=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(TimeoutError):
            svc._wait_ready(self.service, Mock(pid=2345))
        self.assertEqual(self.state.load()['runner_pid'], 2345)

    def test_reused_pid_is_never_signalled(self):
        with patch.object(processes, 'process_identity', return_value='new-start'), \
             patch.object(processes.os, 'kill') as kill, patch.object(processes.os, 'killpg') as killpg:
            processes.stop_owned_process(123, 'old-start', group=True)
            kill.assert_not_called(); killpg.assert_not_called()
            self.assertFalse(processes.runner_alive({'runner_pid': 123, 'runner_identity': 'old-start'}))

    def test_macos_identity_survives_exec_and_process_state_changes(self):
        stamp = 'Mon Nov  3 10:00:00 2026'
        results = [subprocess.CompletedProcess([], 0, f'{state} {stamp}\n', '') for state in ('R', 'S', 'Z')]
        with patch.object(processes.sys, 'platform', 'darwin'),              patch.object(processes.subprocess, 'run', side_effect=results) as run:
            self.assertEqual(processes.process_identity(123), stamp)
            self.assertEqual(processes.process_identity(123), stamp)
            self.assertIsNone(processes.process_identity(123))
            self.assertNotIn('command=', run.call_args.args[0])
            self.assertEqual(run.call_args.kwargs['env']['TZ'], 'UTC')


    def test_owned_process_is_stopped_and_waited_for(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        self.addCleanup(lambda: child.poll() is None and child.kill())
        identity = processes.process_identity(child.pid)
        self.assertIsNotNone(identity)
        processes.stop_owned_process(child.pid, identity, group=True)
        child.wait(timeout=5)
        self.assertFalse(processes.owned_process(child.pid, identity))


if __name__ == '__main__':
    unittest.main()
