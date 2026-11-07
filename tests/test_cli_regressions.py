"""Public CLI safety and automation regressions; no Colab or model dependencies."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/colabllm/lib'))
import colabsvc as svc


class CliRegressionTest(unittest.TestCase):
    def test_invalid_instance_never_falls_back_to_default(self):
        for value in ('çalışma', '', 'a/b', '?', '-bad'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                svc.extract_instance_arg(['down', '--instance=' + value])
        with self.assertRaises(ValueError):
            svc.extract_instance_arg(['down', '-i'])

    def test_instance_extraction_respects_end_of_options(self):
        self.assertEqual(svc.extract_instance_arg(['tts', '--', '-i', 'literal']),
                         (None, ['tts', '--', '-i', 'literal']))
        self.assertEqual(svc.extract_instance_arg(['down', '-i', 'test_2']), ('test_2', ['down']))

    def test_comfyui_help_cannot_submit_a_task(self):
        module = runpy.run_path(str(ROOT / 'src/colabllm/services/comfyui.py'))
        with patch.dict(module['_cmd_pull_workflow'].__globals__,
                        {'_resolve_target_services': lambda **kw: [module['SERVICE']],
                         '_run_remote_pull_job': lambda *a, **kw: self.fail('help submitted a task')}), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as result:
            module['_cmd_pull_workflow'](['--help'])
        self.assertEqual(result.exception.code, 0)

    def test_public_help_lists_business_commands(self):
        expected = {'qwen3-tts': 'tts', 'indextts': 'tts', 'unlimited-ocr': 'ocr',
                    'longcat-avatar': 'generate', 'openjutsu': 'swap', 'comfyui': 'pull-workflow',
                    'nemotron-transcribe': 'transcribe'}
        with tempfile.TemporaryDirectory() as home:
            for service, command in expected.items():
                with self.subTest(service=service):
                    result = subprocess.run([str(ROOT/'colabllm'), service, '--help'],
                        env={**os.environ, 'COLABLLM_HOME': home}, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    choices = [line for line in result.stdout.splitlines() if line.strip().startswith('{')]
                    self.assertTrue(any(command in line for line in choices), result.stdout)
                    self.assertNotIn('_exec', result.stdout)

    def test_status_and_down_propagate_remote_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / 'colab'
            executable.write_text('#!/bin/sh\necho simulated-failure >&2\nexit 42\n')
            executable.chmod(0o755)
            env = {**os.environ, 'COLABLLM_HOME': str(root/'profile'),
                   'PATH': str(root) + os.pathsep + os.environ.get('PATH', '')}
            for command in ('status', 'down'):
                with self.subTest(command=command):
                    result = subprocess.run([str(ROOT/'colabllm'), 'ollama', command],
                        env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 42, result.stderr)

    def test_invalid_instance_down_does_not_invoke_colab(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            marker = root / 'called'
            executable = root / 'colab'
            executable.write_text('#!/bin/sh\ntouch "' + str(marker) + '"\n')
            executable.chmod(0o755)
            result = subprocess.run([str(ROOT/'colabllm'), 'ollama', 'down', '-i', 'çalışma'],
                env={**os.environ, 'COLABLLM_HOME': str(root/'profile'),
                     'PATH': str(root) + os.pathsep + os.environ.get('PATH', '')},
                text=True, capture_output=True)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertFalse(marker.exists())


    def test_expected_network_errors_have_no_traceback(self):
        with tempfile.TemporaryDirectory() as home:
            result = subprocess.run([str(ROOT/'colabllm'), 'qwen3-tts', 'voices',
                '--url', 'http://127.0.0.1:1', '--api-token', 'test-only'],
                env={**os.environ, 'COLABLLM_HOME': home}, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn('Traceback', result.stderr)
            self.assertTrue(result.stderr.strip())

    def test_transcription_json_stdout_contains_only_data(self):
        module = runpy.run_path(str(ROOT / 'src/colabllm/services/nemotron-transcribe.py'))
        output, diagnostics = io.StringIO(), io.StringIO()
        with patch.dict(module['_cmd_transcribe'].__globals__, {
            '_get_endpoint_pool': lambda args: [('test', 'http://example.invalid', 'test-only')],
            '_expand_tasks': lambda *args: [{'id': 'test'}],
            '_run_one_task': lambda *args: {'label': 'test', 'text': 'hello', 'elapsed_sec': 0.1},
        }), contextlib.redirect_stdout(output), contextlib.redirect_stderr(diagnostics):
            module['_cmd_transcribe'](['--json'])
        self.assertEqual(json.loads(output.getvalue())['text'], 'hello')
        self.assertIn('Preparing', diagnostics.getvalue())
        self.assertNotIn('\x1b', diagnostics.getvalue())

    def test_debug_enables_traceback_and_quiet_does_not_hide_errors(self):
        with tempfile.TemporaryDirectory() as home:
            for flag in ('--debug', '--quiet'):
                with self.subTest(flag=flag):
                    result = subprocess.run([str(ROOT/'colabllm'), flag, 'qwen3-tts', 'voices',
                        '--url', 'http://127.0.0.1:1', '--api-token', 'test-only'],
                        env={**os.environ, 'COLABLLM_HOME': home}, text=True, capture_output=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertTrue(result.stderr.strip())
                    self.assertEqual('Traceback' in result.stderr, flag == '--debug')

    def test_info_json_masks_credentials_unless_requested(self):
        with tempfile.TemporaryDirectory() as home:
            state = Path(home) / 'services/ollama'
            state.mkdir(parents=True)
            (state/'state.json').write_text(json.dumps({'api_token': 'test-secret', 'auth_pass': 'password'}))
            for flags, reveal in (([], False), (['--show-secrets'], True)):
                result = subprocess.run([str(ROOT/'colabllm'), 'ollama', 'info', '--json', *flags],
                    env={**os.environ, 'COLABLLM_HOME': home}, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                self.assertEqual('api_token' in payload, reveal)
                self.assertEqual('test-secret' in result.stdout, reveal)

    def test_corrupt_state_reports_the_file_without_traceback(self):
        with tempfile.TemporaryDirectory() as home:
            state = Path(home) / 'services/ollama'
            state.mkdir(parents=True)
            (state/'state.json').write_text('{bad json')
            result = subprocess.run([str(ROOT/'colabllm'), 'ollama', 'info'],
                env={**os.environ, 'COLABLLM_HOME': home}, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(str(state/'state.json'), result.stderr)
            self.assertNotIn('Traceback', result.stderr)


if __name__ == '__main__':
    unittest.main()
