"""Authenticated redirects must never send credentials to another origin."""
import argparse
import contextlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import runpy
import sys
import threading
import unittest
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/colabllm/lib'))
from cli_support import CliError, safe_urlopen


class HttpSafetyTest(unittest.TestCase):
    def setUp(self):
        self.received = []
        received = self.received
        class Sink(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                received.append(self.headers.get('Authorization'))
                body = b'{"ok":true}'
                self.send_response(200); self.send_header('Content-Length', str(len(body)))
                self.end_headers(); self.wfile.write(body)
        self.sink = ThreadingHTTPServer(('127.0.0.1', 0), Sink)
        target = f'http://127.0.0.1:{self.sink.server_port}/target'
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def do_GET(self):
                if self.path == '/same':
                    body = b'{"ok":true}'
                    self.send_response(200); self.send_header('Content-Length', str(len(body)))
                    self.end_headers(); self.wfile.write(body)
                    return
                self.send_response(302)
                self.send_header('Location', '/same' if self.path == '/same-start' else target)
                self.end_headers()
        self.source = ThreadingHTTPServer(('127.0.0.1', 0), Redirect)
        self.url = f'http://127.0.0.1:{self.source.server_port}'
        for server in (self.source, self.sink):
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)

    def test_same_origin_authenticated_redirect_still_works(self):
        request = urllib.request.Request(self.url + '/same-start', headers={'Authorization': 'Bearer test-only'})
        with safe_urlopen(request, timeout=3) as response:
            self.assertTrue(json.load(response)['ok'])
        self.assertEqual(self.received, [])

    def test_service_clients_block_cross_origin_authenticated_redirect(self):
        for name in ('qwen3-tts', 'indextts', 'drama3', 'tangoflux', 'nemotron-transcribe', 'unlimited-ocr', 'comfyui'):
            with self.subTest(service=name):
                module = runpy.run_path(str(ROOT / 'src/colabllm/services' / (name + '.py')))
                with self.assertRaises(CliError):
                    if name in ('qwen3-tts', 'indextts', 'drama3', 'tangoflux'):
                        module['_http_json'](self.url + '/redirect', 'test-only', timeout=3)
                    elif name == 'nemotron-transcribe':
                        module['_http_request'](self.url + '/redirect', 'test-only', timeout=3)
                    else:
                        with patch.object(module['State'], 'load', return_value={'url': self.url, 'api_token': 'test-only'}):
                            if name == 'unlimited-ocr':
                                module['request']('/redirect', argparse.Namespace(url=None))
                            else:
                                # The pull endpoint is POST; a 302 switches it to GET.
                                original = self.source.RequestHandlerClass.do_GET
                                self.source.RequestHandlerClass.do_POST = original
                                module['_run_remote_pull_job']({'workflow': 'test'})
                self.assertEqual(self.received, [])


if __name__ == '__main__':
    unittest.main()
