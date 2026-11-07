"""Only complete protocol lines may alter readiness, including across byte boundaries."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/colabllm/lib'))
from protocol import NotebookProtocol


class NotebookProtocolTest(unittest.TestCase):
    def test_log_mentions_are_not_ready(self):
        parser = NotebookProtocol()
        result = parser.feed(b'ERROR: expected CT_READY but startup failed\nprint("CT_READY")\nNOT_CT_READY\n')
        self.assertNotIn('ready', result)
        self.assertNotIn('ready', parser.feed(b'CT_READY'))
        self.assertTrue(parser.feed(b'\n')['ready'])

    def test_split_utf8_ansi_and_markers(self):
        parser = NotebookProtocol()
        data = '\x1b[32mCT_READY\x1b[0m\r\nCT_PUBLIC_URL=https://test.invalid\nCT_CREDENTIALS user=kullanıcı_ş pass=fake\n[colab] Executing cell 6/6\n'.encode()
        result = {}
        for value in data:
            result.update(parser.feed(bytes([value])))
        self.assertEqual(result, dict(ready=True, url='https://test.invalid', login_user='kullanıcı_ş', login_pass='fake', last_cell_started=True))

    def test_eof_and_long_log_line(self):
        parser = NotebookProtocol()
        self.assertNotIn('ready', parser.feed(b'X' * 70000))
        self.assertNotIn('ready', parser.feed(b'CT_READY\n'))
        self.assertTrue(parser.feed(b'CT_READY', final=True)['ready'])


if __name__ == '__main__':
    unittest.main()
