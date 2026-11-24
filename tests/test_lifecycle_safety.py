"""State and process lifecycle regressions, using isolated profiles and child processes."""
import argparse
import concurrent.futures
import contextlib
import io
import json
import os
import signal
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
        run.assert_called_once_with(['colab', 'stop', '-s', 'test'], check=False, env=svc.colab_subprocess_env())
        self.assertFalse(self.state.dir.exists())

    def test_purge_uses_saved_account_even_when_active_environment_changes(self):
        self.state.save(account='original', drive_account='shared')
        with patch.dict(os.environ, {'COLABLLM_ACCOUNT': 'other', 'COLAB_ACCOUNT': 'other'}), \
             patch.object(svc, 'run', return_value=argparse.Namespace(returncode=0)) as run:
            svc.cmd_purge(self.service, self.purge_args())
        self.assertEqual(run.call_args.kwargs['env']['COLAB_ACCOUNT'], 'original')
        self.assertEqual(run.call_args.kwargs['env']['COLAB_DRIVE_ACCOUNT'], 'shared')
        self.assertFalse(self.state.dir.exists())

    def test_purge_refuses_an_explicit_different_account(self):
        self.state.save(account='original')
        self.service.account = 'other'
        with patch.object(svc, 'resolve_effective_account', return_value=('other', 'flag')), \
             patch.object(svc, 'run') as run, self.assertRaisesRegex(CliError, 'belongs to account'):
            svc.cmd_purge(self.service, self.purge_args())
        run.assert_not_called()
        self.assertTrue(self.state.path.exists())

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

    def test_parse_colab_session_status_detects_missing_despite_zero_exit(self):
        self.assertEqual(
            svc.parse_colab_session_status(
                'qwen-flash',
                0,
                '[qwen-flash] gpu-g4-s-asso1c0 | Hardware: G4 | Status: BUSY',
            ),
            'alive',
        )
        self.assertEqual(
            svc.parse_colab_session_status(
                'minimax-h3-image',
                0,
                "[colab] Session 'minimax-h3-image' not found.",
            ),
            'missing',
        )
        self.assertEqual(svc.parse_colab_session_status('ollama', 0, ''), 'unknown')

    def test_restart_tuning_flags_require_auto_restart(self):
        args = self.args(auto_restart=False, restart_delay=10.0, max_restarts=0)
        with self.assertRaisesRegex(ValueError, '--auto-restart'):
            svc.cmd_up(self.service, self.script, args)
        args2 = self.args(auto_restart=False, restart_delay=None, max_restarts=2)
        with self.assertRaisesRegex(ValueError, '--auto-restart'):
            svc.cmd_up(self.service, self.script, args2)

    def test_ensure_session_recreates_when_cached_session_was_reclaimed(self):
        fake_home = self.root / 'home'
        cfg_dir = fake_home / '.config/colab-cli'
        cfg_dir.mkdir(parents=True)
        (cfg_dir / 'sessions.json').write_text(json.dumps({'test': {'id': 'stale'}}))
        with patch.object(svc.Path, 'home', return_value=fake_home), \
             patch.object(svc, 'check_remote_session', return_value='missing'), \
             patch.object(svc, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as mock_run:
            svc.ensure_session(self.service)
        cmds = [call.args[0] for call in mock_run.call_args_list]
        self.assertEqual(cmds, [['colab', 'stop', '-s', 'test'], ['colab', 'new', '-s', 'test', '--gpu', 'T4']])

    def colab_new_results(self, *results):
        return [subprocess.CompletedProcess([], code, '', err) for code, err in results]

    def test_ensure_session_waits_for_unavailable_accelerator(self):
        refused = (1, "[colab] Allocation refused (precondition failed). This can mean too many active sessions\n")
        # Real rich-rendered `colab new` output for a G4 503: the error line is hard-wrapped at 80 columns.
        busy = (1, "[colab] Creating session 'test'...\n"
                   "│ ❱ 238 │   │   │   raise ColabRequestError(                                   │\n"
                   "╰──────────────────────────────────────────────────────────────────────────────╯\n"
                   "ColabRequestError: Failed to issue request POST \n"
                   "https://colab.research.google.com/tun/m/assign?nbh=9120f6af_6487_4e9b_9a3f_8a2d1\n"
                   "54410c9&variant=GPU&accelerator=G4: Service Unavailable\n")
        with patch.object(svc, 'run', side_effect=self.colab_new_results(refused, busy, (0, ''))) as mock_run, \
             patch.object(svc.time, 'sleep') as sleep, contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertFalse(svc.ensure_session(self.service, force_new=True, retry_interval=15, single_assign=True))
        self.assertEqual(mock_run.call_count, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [15, 15])
        self.assertTrue(all(call.kwargs['env']['COLAB_CLI_ASSIGN_RETRIES'] == '0' for call in mock_run.call_args_list))
        self.assertIn('Service Unavailable', err.getvalue())
        self.assertNotIn('ColabRequestError', err.getvalue())

    def test_ensure_session_does_not_retry_other_failures_or_disabled_retry(self):
        with patch.object(svc, 'run', side_effect=self.colab_new_results((1, 'Traceback: auth token expired\n'))), \
             patch.object(svc.time, 'sleep') as sleep, contextlib.redirect_stderr(io.StringIO()), \
             self.assertRaises(CliError) as caught:
            svc.ensure_session(self.service, force_new=True, retry_interval=15)
        self.assertNotIsInstance(caught.exception, svc.AcceleratorUnavailableError)
        sleep.assert_not_called()
        refused = (1, '[colab] Allocation refused (precondition failed).\n')
        with patch.object(svc, 'run', side_effect=self.colab_new_results(refused)) as mock_run, \
             self.assertRaises(svc.AcceleratorUnavailableError):
            svc.ensure_session(self.service, force_new=True, retry_interval=0)
        self.assertNotIn('COLAB_CLI_ASSIGN_RETRIES', mock_run.call_args.kwargs['env'] or {})

    def test_ensure_session_gives_up_after_wait_timeout(self):
        refused = (1, '[colab] Allocation refused (precondition failed).\n')
        clock = iter([0, 0, 10, 20, 30])
        with patch.object(svc, 'run', side_effect=self.colab_new_results(*[refused] * 4)) as mock_run, \
             patch.object(svc.time, 'sleep'), patch.object(svc.time, 'monotonic', side_effect=lambda: next(clock)), \
             self.assertRaisesRegex(svc.AcceleratorUnavailableError, 'gave up after 3 attempts'):
            svc.ensure_session(self.service, force_new=True, retry_interval=10, wait_timeout=25)
        self.assertEqual(mock_run.call_count, 3)

    def cancelled_wait(self, run_side_effect, sleep_side_effect=None):
        with patch.object(svc, 'run', side_effect=run_side_effect) as mock_run, \
             patch.object(svc.time, 'sleep', side_effect=sleep_side_effect), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(svc.DeployCancelled) as caught:
            with svc._cancellable_up(self.service, self.state):
                svc.ensure_session(self.service, force_new=True, retry_interval=15, single_assign=True)
        self.assertEqual(caught.exception.code, 143)
        self.assertIsNone(svc._UP_CANCEL)
        return [call.args[0][:2] for call in mock_run.call_args_list]

    def test_cancel_while_waiting_for_accelerator_stops_without_a_session(self):
        refused = subprocess.CompletedProcess([], 1, '', '[colab] Allocation refused (precondition failed).\n')
        cmds = self.cancelled_wait(lambda *a, **k: refused, lambda seconds: os.kill(os.getpid(), signal.SIGTERM))
        self.assertEqual(cmds, [['colab', 'new']])

    def test_cancel_during_colab_new_waits_for_it_and_releases_the_new_session(self):
        def colab(cmd, **kwargs):
            if cmd[1] == 'new':
                os.kill(os.getpid(), signal.SIGTERM)  # arrives while the assign request is in flight
            return subprocess.CompletedProcess(cmd, 0, '', '')
        self.assertEqual(self.cancelled_wait(colab), [['colab', 'new'], ['colab', 'stop']])


if __name__ == '__main__':
    unittest.main()
