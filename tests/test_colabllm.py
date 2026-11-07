import contextlib
import importlib.machinery
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.machinery.SourceFileLoader('colabllm_cli', str(ROOT / 'src/colabllm/cli.py'))
spec = importlib.util.spec_from_loader(loader.name, loader)
cli = importlib.util.module_from_spec(spec)
loader.exec_module(cli)


class UnifiedCliTest(unittest.TestCase):
    def test_dispatch_preserves_arguments_and_working_directory_for_every_service(self):
        before = Path.cwd()
        for name, path in cli.services().items():
            with self.subTest(name=name), patch.object(cli.os, 'execv') as execute:
                cli.main([name, 'up', '-i', '2', '--domain', 'my.example.com'])
                execute.assert_called_once_with(sys.executable, [sys.executable, str(path), 'up', '-i', '2', '--domain', 'my.example.com'])
                self.assertEqual(Path.cwd(), before)
        with patch.object(cli.os, 'execv') as execute:
            cli.main(['unlimited-ocr', 'ocr', './yerel klasör/sözleşme 1.pdf', '-O', './sonuçlar'])
            self.assertEqual(execute.call_args.args[1][-4:], ['ocr', './yerel klasör/sözleşme 1.pdf', '-O', './sonuçlar'])

    def test_global_configuration_dispatch(self):
        for name in cli.GLOBAL_COMMANDS:
            with patch.object(cli.os, 'execv') as execute:
                cli.main([name, '--help'])
                execute.assert_called_once_with(sys.executable, [sys.executable, str(ROOT / 'src/colabllm/control.py'), name, '--help'])

    def test_listing_and_unknown_command_never_execute_a_service(self):
        with patch.object(cli.os, 'execv') as execute, contextlib.redirect_stdout(io.StringIO()) as output, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['services']), 0)
            self.assertIn('unlimited-ocr', output.getvalue())
            self.assertEqual(cli.main(['../../etc/passwd']), 2)
            self.assertEqual(cli.main(['services', 'up']), 2)
            execute.assert_not_called()

    def test_repository_symlink_installer_is_removed(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(['install']), 2)


if __name__ == '__main__':
    unittest.main()
