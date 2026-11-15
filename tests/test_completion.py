"""Completion uses real parser registrations and works offline without secret reads."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src/colabllm/lib'))
from completion import candidates, shell_script


class CompletionTest(unittest.TestCase):
    def test_real_commands_options_and_choices(self):
        self.assertIn('doctor', candidates(['d']))
        self.assertIn('tts', candidates(['qwen3-tts', 't']))
        self.assertNotIn('_exec', candidates(['ollama', '']))
        self.assertEqual(candidates(['qwen3-tts', 'tts', '--format', 'm']), ['mp3'])
        self.assertEqual(candidates(['qwen3-tts', 'tts', '--format=m']), ['--format=mp3'])
        self.assertIn('--online', candidates(['doctor', '--']))
        self.assertIn('--base-domain', candidates(['tunnel-config', '--']))
        self.assertEqual(candidates(['ollama', 'up', '--transport', 'f']), ['frp'])
        self.assertEqual(candidates(['ollama', 'up', '--', '-i', '']), [])

    def test_instances_are_completed_without_reading_state(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'COLABLLM_HOME': temp}):
            directory = Path(temp)/'services/ollama-demo'
            directory.mkdir(parents=True)
            (directory/'state.json').write_text('not-json-must-not-be-read')
            self.assertEqual(candidates(['ollama', 'status', '-i', 'd']), ['demo'])
            self.assertEqual(candidates(['ollama', 'status', '--instance=d']), ['--instance=demo'])

    def test_accounts_are_completed_without_reading_secret_files(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'COLAB_CLI_CONFIG_DIR': temp}):
            for name in ('main', 'gpu1', 'gpu2'):
                acct_dir = Path(temp)/'accounts'/name
                acct_dir.mkdir(parents=True)
                (acct_dir/'oauth_token.json').write_text('not-json-must-not-be-read')
                (acct_dir/'dfs_credentials.json').write_text('not-json-must-not-be-read')
            self.assertEqual(candidates(['--account', 'g']), ['gpu1', 'gpu2'])
            self.assertEqual(candidates(['--account=g']), ['--account=gpu1', '--account=gpu2'])
            self.assertEqual(candidates(['ollama', 'up', '-a', 'g']), ['gpu1', 'gpu2'])
            self.assertEqual(candidates(['ollama', 'up', '--drive-account=m']), ['--drive-account=main'])
            self.assertEqual(candidates(['doctor', '-a', 'g']), ['gpu1', 'gpu2'])
            # Ensure minimax-h3-image's `-a` still completes aspect ratios, NOT accounts
            self.assertEqual(candidates(['minimax-h3-image', 'generate', '-a', '16:']), ['16:9'])

    def run_shell(self, shell, body):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'colabllm').symlink_to(ROOT/'colabllm')
            script = root/'completion'
            script.write_text(shell_script(shell))
            env = {**os.environ, 'COLABLLM_HOME': str(root/'profile'),
                   'PATH': str(root)+os.pathsep+os.environ.get('PATH','')}
            result = subprocess.run([shell, '-c', body, 'completion-test', str(script)], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('tts', result.stdout.splitlines())
            self.assertFalse((root/'profile').exists())

    def test_bash_integration(self):
        self.run_shell('bash', 'source "$1"; COMP_WORDS=(colabllm qwen3-tts t); COMP_CWORD=2; _colabllm_complete; printf "%s\\n" "${COMPREPLY[@]}"')

    @unittest.skipUnless(shutil.which('zsh'), 'zsh shell is not installed; exercised in CI')
    def test_zsh_integration(self):
        self.run_shell('zsh', 'compdef() { :; }; _describe() { print -l -- "${candidates[@]}"; }; source "$1"; words=(colabllm qwen3-tts t); CURRENT=3; _colabllm_complete')

    @unittest.skipUnless(shutil.which('fish'), 'fish shell is not installed; exercised in CI')
    def test_fish_integration(self):
        self.run_shell('fish', '''function commandline
    if test "$argv[1]" = -opc
        printf '%s\\n' colabllm qwen3-tts
    else
        echo t
    end
end
source $argv[2]
__colabllm_candidates''')


if __name__ == '__main__':
    unittest.main()
