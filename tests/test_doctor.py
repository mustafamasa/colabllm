"""Diagnostics stay read-only, bounded and free of raw command output or credentials."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/colabllm/lib'))
import doctor


class DoctorTest(unittest.TestCase):
    def test_offline_checks_never_probe_authorization_or_create_files(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(doctor, 'STATE_ROOT', Path(temp)/'profile'), \
             patch.object(doctor, 'load_tunnel_config', return_value={}), \
             patch.object(doctor.Path, 'home', return_value=Path(temp)/'home'), \
             patch.dict(os.environ, {k: v for k, v in os.environ.items() if k not in ('COLAB_CLI_CONFIG_DIR', 'COLABLLM_ACCOUNT', 'COLAB_ACCOUNT', 'COLABLLM_DRIVE_ACCOUNT', 'COLAB_DRIVE_ACCOUNT', 'COLAB_DFS_ACCOUNT')}, clear=True), \
             patch.object(doctor.shutil, 'which', return_value='/fake/colab'), \
             patch.object(doctor.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'fake-secret', '')) as run:
            report = doctor.diagnose()
            self.assertTrue(report['ok'])
            self.assertEqual([call.args[0] for call in run.call_args_list], [['colab', 'version'], ['colab', 'dfs', '--help']])
            self.assertNotIn('fake-secret', json.dumps(report))
            self.assertFalse((Path(temp)/'profile').exists())
            self.assertTrue(all(call.kwargs['timeout'] == 10 for call in run.call_args_list))
            self.assertTrue(all(call.kwargs['env']['BROWSER'] == 'true' for call in run.call_args_list))

    def test_missing_dependencies_fail_but_no_drive_skips_drive_extension(self):
        with patch.object(doctor, 'load_tunnel_config', return_value={}), \
             patch.object(doctor.shutil, 'which', return_value=None):
            report = doctor.diagnose(no_drive=True)
        self.assertFalse(report['ok'])
        self.assertNotIn('drive-extension', [item['name'] for item in report['checks']])

    def test_public_json_and_optional_online_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fake = root/'colab'
            fake.write_text('#!/bin/sh\necho fake-secret\nexit 0\n')
            fake.chmod(0o755)
            env = {**os.environ, 'COLABLLM_HOME': str(root/'profile'),
                   'PATH': str(root)+os.pathsep+os.environ.get('PATH','')}
            result = subprocess.run([str(ROOT/'colabllm'), 'doctor', '--json', '--no-drive', '--transport', 'quick', '--online'], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report['ok'])
            self.assertIn('colab-auth', [item['name'] for item in report['checks']])
            self.assertNotIn('fake-secret', result.stdout)
            self.assertFalse((root/'profile').exists())

    def test_multi_account_shared_drive_and_unknown_account(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = root / 'colab-cli'
            (cfg / 'accounts/main').mkdir(parents=True)
            (cfg / 'accounts/gpu2').mkdir(parents=True)
            (cfg / 'accounts/main/dfs_credentials.json').write_text('{"refresh_token":"fake"}')
            (cfg / 'accounts.json').write_text(json.dumps({
                'version': 1,
                'active': 'main',
                'accounts': {
                    'main': {'email': 'main@example.com'},
                    'gpu2': {'email': 'gpu2@example.com', 'drive_account': 'main'},
                },
            }))
            env = {
                **os.environ,
                'COLABLLM_HOME': str(root/'profile'),
                'COLAB_CLI_CONFIG_DIR': str(cfg),
            }
            with patch.dict(os.environ, env, clear=True), \
                 patch.object(doctor, 'STATE_ROOT', root/'profile'), \
                 patch.object(doctor, 'load_tunnel_config', return_value={}), \
                 patch.object(doctor.shutil, 'which', return_value='/fake/colab'), \
                 patch.object(doctor.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')) as run:
                report = doctor.diagnose(account='gpu2', online=True)
            self.assertTrue(report['ok'])
            self.assertEqual(report['account'], 'gpu2')
            self.assertEqual(report['drive_account'], 'main')
            by_name = {c['name']: c for c in report['checks']}
            self.assertEqual(by_name['colab-account']['status'], 'pass')
            self.assertEqual(by_name['drive-consent']['status'], 'pass')
            self.assertTrue(all(call.kwargs['env']['COLAB_ACCOUNT'] == 'gpu2' for call in run.call_args_list))
            self.assertTrue(all(call.kwargs['env']['COLAB_DRIVE_ACCOUNT'] == 'main' for call in run.call_args_list))

            with patch.dict(os.environ, env, clear=True), \
                 patch.object(doctor, 'STATE_ROOT', root/'profile'), \
                 patch.object(doctor, 'load_tunnel_config', return_value={}), \
                 patch.object(doctor.shutil, 'which', return_value='/fake/colab'), \
                 patch.object(doctor.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')):
                bad = doctor.diagnose(account='nonexistent', online=True)
            self.assertFalse(bad['ok'])
            bad_by_name = {c['name']: c for c in bad['checks']}
            self.assertEqual(bad_by_name['colab-account']['status'], 'fail')
            self.assertNotIn('colab-auth', bad_by_name)


if __name__ == '__main__':
    unittest.main()
