"""Local web control panel for colabllm (`colabllm ui`). Standard library only.

The panel never reimplements lifecycle logic: it reads the same state files as the CLI and runs
`up`/`down` through the packaged service scripts, so locks, readiness and shutdown checks stay in one place.
Security model: loopback-only listener, a random per-run access token, strict Host/Origin checks.
"""
import argparse
import ast
import contextlib
import http.server
import io
import json
import os
from pathlib import Path
import re
import runpy
import secrets
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid

import colabsvc
from processes import runner_alive

PACKAGE = Path(__file__).resolve().parents[1]
SERVICES_DIR = PACKAGE / 'services'
ASSETS_DIR = PACKAGE / 'webui'
DOCS_URL = 'https://github.com/deluxebear/colabllm/blob/master/docs/{}.md'
GPUS = ('T4', 'L4', 'G4', 'A100', 'H100', 'none')
SECRET_KEYS = ('auth_user', 'auth_pass', 'api_token', 'login_user', 'login_pass')
CATEGORIES = {
    'ollama': 'LLM', 'deepseek-coder': 'LLM', 'qwen-flash': 'LLM',
    'comfyui': 'Image', 'flux-klein': 'Image', 'qwen-image': 'Image',
    'longcat-avatar': 'Video', 'longcat-video': 'Video', 'openjutsu': 'Video',
    'indextts': 'Audio', 'qwen3-tts': 'Audio', 'nemotron-transcribe': 'Audio',
    'unlimited-ocr': 'Document', 'unsloth-studio': 'Training',
}
CONTENT_TYPES = {'.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
                 '.js': 'text/javascript; charset=utf-8', '.svg': 'image/svg+xml'}
NAME_RE = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_-]{0,127}')
MAX_BODY = 64 * 1024
MAX_OPTION_VALUE = 4096
MAX_LIST_ITEMS = 50
OUTPUT_LIMIT = 200_000


# --------------------------------------------------------------------------- catalog
def _service_meta(path: Path) -> dict:
    """Read SERVICE defaults and the summary line statically, without importing the service script."""
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    doc = (ast.get_docstring(tree) or '').strip().splitlines()
    summary = re.sub(r'\s*\(notebook: [^)]*\)', '', doc[0] if doc else '')
    summary = re.sub(r',\s*notebook: [^)]*', '', summary).strip().rstrip('.')
    fields = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'Service':
            for kw in node.keywords:
                try:
                    fields[kw.arg] = ast.literal_eval(kw.value)
                except (ValueError, SyntaxError):
                    pass
            break
    gpu = fields.get('tpu') or fields.get('gpu') or 'none'
    return dict(name=path.stem, summary=summary, category=CATEGORIES.get(path.stem, 'Other'),
                default_gpu=gpu, auth=fields.get('auth', 'caddy'), high_mem=bool(fields.get('high_mem')),
                ready_timeout=fields.get('ready_timeout', 3600), docs=DOCS_URL.format(path.stem))


_catalog_cache = None


def catalog() -> list[dict]:
    global _catalog_cache
    if _catalog_cache is None:
        items = []
        for path in sorted(SERVICES_DIR.glob('*.py')):
            if path.with_suffix('.ipynb').is_file():
                try:
                    items.append(_service_meta(path))
                except (OSError, SyntaxError, UnicodeError):
                    items.append(dict(name=path.stem, summary='', category='Other', default_gpu='?',
                                      auth='?', high_mem=False, ready_timeout=3600, docs=DOCS_URL.format(path.stem)))
        _catalog_cache = items
    return _catalog_cache


def _service_names() -> set[str]:
    return {item['name'] for item in catalog()}


# --------------------------------------------------------------------------- service-specific options
_options_cache: dict[str, tuple] = {}
_options_lock = threading.RLock()


def _subparser(parser, name):
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return action.choices[name]


def _service_options(service: str):
    """(full parser, service-specific `up` actions), read from the same argparse tree the CLI builds."""
    with _options_lock:
        if service not in _options_cache:
            if service not in _service_names():
                raise LookupError('unknown service')
            module = runpy.run_path(str(SERVICES_DIR / f'{service}.py'))
            parser = colabsvc.build_parser(module['SERVICE'], module.get('configure_cli'))
            base = colabsvc.build_parser(colabsvc.Service(name='_', notebook='_', service_port=1))
            common = {a.dest for a in _subparser(base, 'up')._actions}
            actions = [a for a in _subparser(parser, 'up')._actions
                       if a.option_strings and a.dest not in common and not isinstance(a, argparse._HelpAction)]
            _options_cache[service] = (parser, actions)
        return _options_cache[service]


def _long_flag(action):
    """Stable option key: the first long flag (the positive one for --x/--no-x pairs); dests can be shared."""
    longs = [s for s in action.option_strings if s.startswith('--')]
    if isinstance(action, argparse.BooleanOptionalAction):
        longs = [s for s in longs if not s.startswith('--no-')]
    return longs[0] if longs else action.option_strings[0]


def _option_kind(action):
    if isinstance(action, argparse.BooleanOptionalAction):
        return 'tristate'
    if isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction, argparse._StoreConstAction)):
        return 'flag'
    if isinstance(action, argparse._AppendAction):
        return 'list'
    if action.choices:
        return 'choice'
    if action.type is int:
        return 'number'
    return 'text'


def option_schema(service: str) -> list[dict]:
    _, actions = _service_options(service)
    items = []
    for action in actions:
        kind = _option_kind(action)
        flag = _long_flag(action)
        default = action.default if isinstance(action.default, (str, int, float, bool)) else None
        help_text = (action.help or '').replace('%%', '%')
        if '%(default)' in help_text:
            help_text = help_text.replace('%(default)s', str(default))
        items.append(dict(key=flag, flag=flag, kind=kind, help=help_text, default=default,
                          choices=[str(c) for c in action.choices] if action.choices else None,
                          metavar=action.metavar if isinstance(action.metavar, str) else None))
    return items


def _clean_value(value, flag):
    if isinstance(value, bool) or value is None:
        raise ValueError(f'{flag} expects a value')
    text = str(value).strip()
    if len(text) > MAX_OPTION_VALUE or any(c in text for c in '\r\n\x00'):
        raise ValueError(f'{flag}: value is too long or contains line breaks')
    return text


def service_option_args(service: str, options) -> list[str]:
    """Turn panel form values into argv and validate them with the service's own argparse parser."""
    if not options:
        return []
    if not isinstance(options, dict):
        raise ValueError('options must be an object')
    parser, actions = _service_options(service)
    by_key = {_long_flag(a): a for a in actions}
    args = []
    for key, value in options.items():
        if key not in by_key:
            raise ValueError(f'unknown option for {service}: {key}')
        action = by_key[key]
        kind, flag = _option_kind(action), _long_flag(action)
        if kind == 'flag':
            if value is True:
                args.append(flag)
            elif value not in (False, None):
                raise ValueError(f'{flag} expects true or false')
        elif kind == 'tristate':
            positive = flag
            negative = next(s for s in action.option_strings if s.startswith('--no-'))
            if value in ('on', True):
                args.append(positive)
            elif value in ('off', False):
                args.append(negative)
            elif value not in ('', None, 'default'):
                raise ValueError(f'{positive} expects on, off or default')
        elif kind == 'list':
            values = value.split('\n') if isinstance(value, str) else value
            if not isinstance(values, list) or len(values) > MAX_LIST_ITEMS:
                raise ValueError(f'{flag} expects a list of at most {MAX_LIST_ITEMS} values')
            args += [f'{flag}={_clean_value(v, flag)}' for v in values if str(v).strip()]
        elif value not in ('', None):
            text = _clean_value(value, flag)
            if kind == 'choice' and text not in [str(c) for c in action.choices]:
                raise ValueError(f'{flag} must be one of: ' + ', '.join(map(str, action.choices)))
            args.append(f'{flag}={text}')
    errors = io.StringIO()
    with _options_lock, contextlib.redirect_stderr(errors), contextlib.redirect_stdout(io.StringIO()):
        try:
            parser.parse_args(['up', *args])
        except SystemExit:
            message = errors.getvalue().strip().splitlines()
            raise ValueError(message[-1].split('error: ', 1)[-1] if message else 'invalid service options') from None
    return args


# --------------------------------------------------------------------------- instances
def _parse_started(value):
    try:
        return time.mktime(time.strptime(value, '%Y-%m-%d %H:%M:%S'))
    except (TypeError, ValueError):
        return None


def _read_state(directory: Path) -> dict | None:
    try:
        data = json.loads((directory / 'state.json').read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError, UnicodeError):
        return None


def instances() -> list[dict]:
    """All known instances, credentials stripped. Mirrors `colabllm <svc> ps --json` for every service."""
    meta = {item['name']: item for item in catalog()}
    rows = []
    for base in meta:
        for item in colabsvc.list_service_instances(base):
            st = _read_state(colabsvc.STATE_ROOT / 'services' / item['name']) or {}
            started = _parse_started(item.get('started')) if item['alive'] else None
            if item['ready']:
                status = 'ready'
            elif item['alive']:
                status = 'starting'
            elif st.get('runner_exit') not in (None, 0):
                status = 'failed'
            else:
                status = 'stopped'
            log_path = colabsvc.STATE_ROOT / 'services' / item['name'] / 'run.log'
            rows.append(dict(
                id=item['name'], service=base, instance=item['instance'], instance_arg=item['instance_arg'],
                status=status, url=item.get('url') if item['alive'] else None,
                started=item.get('started'), started_ts=started,
                uptime=(time.time() - started) if started else None,
                gpu=st.get('gpu') or meta[base]['default_gpu'], gpu_recorded=bool(st.get('gpu')),
                transport=st.get('transport') or 'quick', hostname=st.get('hostname'),
                owns_tunnel=bool(st.get('tunnel_id') or st.get('cloudflare_route') or st.get('frp_route')),
                runner_pid=item.get('runner_pid') if item['alive'] else None, runner_exit=st.get('runner_exit'),
                has_secrets=any(st.get(k) for k in SECRET_KEYS), auth=meta[base]['auth'],
                category=meta[base]['category'], has_log=log_path.is_file(),
                log_size=log_path.stat().st_size if log_path.is_file() else 0,
            ))
    order = {'ready': 0, 'starting': 1, 'failed': 2, 'stopped': 3}
    rows.sort(key=lambda r: (order[r['status']], r['service'], r['instance']))
    return rows


def _instance_dir(instance_id: str) -> Path:
    if not NAME_RE.fullmatch(instance_id or ''):
        raise LookupError('invalid instance id')
    directory = colabsvc.STATE_ROOT / 'services' / instance_id
    if not (directory / 'state.json').is_file():
        raise LookupError('unknown instance')
    return directory


def instance_secrets(instance_id: str) -> dict:
    st = _read_state(_instance_dir(instance_id)) or {}
    return {k: st[k] for k in ('url', *SECRET_KEYS) if st.get(k)}


def tail_log(instance_id: str, max_bytes=256 * 1024) -> dict:
    path = _instance_dir(instance_id) / 'run.log'
    if not path.is_file():
        return dict(text='', size=0, truncated=False)
    size = path.stat().st_size
    with path.open('rb') as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        data = fh.read()
    text = data.decode('utf-8', 'replace')
    if size > max_bytes and '\n' in text:
        text = text.split('\n', 1)[1]
    return dict(text=re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', text), size=size, truncated=size > max_bytes)


def probe_instance(instance_id: str) -> dict:
    st = _read_state(_instance_dir(instance_id)) or {}
    if not (runner_alive(st) and st.get('url')):
        return dict(ok=False, status='offline', latency_ms=None)
    started = time.monotonic()
    url = st['url']
    if st.get('auth_user') and st.get('auth_pass'):
        result = colabsvc.probe(url, (st['auth_user'], st['auth_pass']))
    elif st.get('api_token') and not st.get('login_user'):
        result = colabsvc.probe(url.rstrip('/') + '/v1/models', bearer=st['api_token'])
    else:
        result = colabsvc.probe(url)
    ok = result.isdigit() and 200 <= int(result) < 400
    return dict(ok=ok, status=result, latency_ms=round((time.monotonic() - started) * 1000))


# --------------------------------------------------------------------------- account
_usage = {'at': 0.0, 'data': None}
_usage_lock = threading.Lock()


def account_usage(force=False) -> dict | None:
    """Parse `colab usage` (balance, burn rate, active assignments). Cached; failures return None."""
    with _usage_lock:
        if not force and time.time() - _usage['at'] < 60:
            return _usage['data']
        data = None
        try:
            result = subprocess.run(['colab', 'usage'], capture_output=True, text=True, timeout=25,
                                    stdin=subprocess.DEVNULL, env={**os.environ, 'BROWSER': 'true', 'NO_COLOR': '1'})
            if result.returncode == 0:
                text = result.stdout
                def num(pattern):
                    match = re.search(pattern, text, re.I)
                    return float(match.group(1)) if match else None
                data = dict(balance=num(r'balance:\s*([\d.]+)'), rate=num(r'rate:\s*([\d.]+)'),
                            assignments=num(r'assignments:\s*(\d+)'), checked=time.time())
                if all(v is None for k, v in data.items() if k != 'checked'):
                    data = None
        except (OSError, subprocess.TimeoutExpired):
            data = None
        _usage.update(at=time.time(), data=data)
        return data


def environment() -> dict:
    try:
        config = colabsvc.load_tunnel_config()
    except Exception:
        config = {}
    frp = config.get('frp') or {}
    domain = os.environ.get('COLABSVC_BASE_DOMAIN') or config.get('base_domain')
    transport = 'frp' if frp.get('enabled') else ('cloudflare' if domain else 'quick')
    try:
        from importlib.metadata import version
        pkg_version = version('colabllm')
    except Exception:
        pkg_version = 'dev'
    return dict(home=str(colabsvc.STATE_ROOT), base_domain=domain, transport=transport,
                frp_server=frp.get('server_ip') if frp.get('enabled') else None, version=pkg_version,
                python=sys.version.split()[0])


# --------------------------------------------------------------------------- operations
def _popen(*args, **kwargs):
    return subprocess.Popen(*args, **kwargs)


class Operations:
    """Lifecycle commands started from the panel, run as the same service scripts the CLI execs."""

    def __init__(self):
        self.items: dict[str, dict] = {}
        self.lock = threading.Lock()

    def busy(self, instance_id):
        with self.lock:
            return any(op['target'] == instance_id and op['state'] == 'running' for op in self.items.values())

    def start(self, service: str, action: str, args: list[str], instance_id: str, summary: str) -> dict:
        if self.busy(instance_id):
            raise RuntimeError(f'An operation is already running for {instance_id}')
        script = SERVICES_DIR / f'{service}.py'
        cmd = [sys.executable, str(script), '--no-color', *args]
        op = dict(id=uuid.uuid4().hex[:12], service=service, target=instance_id, action=action, summary=summary,
                  command='colabllm ' + ' '.join([service, *args]), state='running', exit_code=None,
                  started=time.time(), finished=None, output='')
        with self.lock:
            self.items[op['id']] = op
        env = {**os.environ, 'NO_COLOR': '1', 'PYTHONUNBUFFERED': '1'}
        try:
            proc = _popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    text=True, bufsize=1, env=env, cwd=str(Path.home()))
        except OSError as error:
            self._finish(op, 127, f'Failed to start: {error}\n')
            return self.public(op)
        threading.Thread(target=self._pump, args=(op, proc), daemon=True).start()
        return self.public(op)

    def _pump(self, op, proc):
        for line in proc.stdout:
            with self.lock:
                op['output'] = (op['output'] + line)[-OUTPUT_LIMIT:]
        self._finish(op, proc.wait(), '')

    def _finish(self, op, code, extra):
        with self.lock:
            op['output'] = (op['output'] + extra)[-OUTPUT_LIMIT:]
            op.update(state='succeeded' if code == 0 else 'failed', exit_code=code, finished=time.time())

    def public(self, op, with_output=False):
        data = {k: v for k, v in op.items() if k != 'output'}
        lines = op['output'].strip().splitlines()
        data['last_line'] = lines[-1][:240] if lines else ''
        if with_output:
            data['output'] = op['output']
        return data

    def list(self):
        with self.lock:
            return sorted((self.public(op) for op in self.items.values()), key=lambda o: -o['started'])

    def get(self, op_id):
        with self.lock:
            if op_id not in self.items:
                raise LookupError('unknown operation')
            return self.public(self.items[op_id], with_output=True)


OPS = Operations()


def start_up(service: str, body: dict) -> dict:
    if service not in _service_names():
        raise LookupError('unknown service')
    instance = (body.get('instance') or '').strip()
    args = ['up', '--detach']
    if instance:
        colabsvc.validate_instance(instance)
        args += ['-i', instance]
    gpu = body.get('gpu')
    if gpu:
        if gpu not in GPUS:
            raise ValueError('unsupported GPU')
        args += ['--gpu', gpu]
    if body.get('quick_tunnel'):
        args.append('--quick-tunnel')
    if body.get('no_drive'):
        args.append('--no-drive')
    if body.get('new_credentials'):
        args.append('--new-credentials')
    args += service_option_args(service, body.get('options'))
    target = f'{service}-{instance}' if instance and not service.endswith(f'-{instance}') else service
    return OPS.start(service, 'deploy', args, target, f'Deploy {target}' + (f' on {gpu}' if gpu else ''))


def start_down(instance_id: str, body: dict, rows=None) -> dict:
    _instance_dir(instance_id)
    match = next((row for row in (rows if rows is not None else instances()) if row['id'] == instance_id), None)
    if not match:
        raise LookupError('unknown instance')
    args = ['down']
    if match['instance_arg']:
        args += ['-i', match['instance_arg']]
    if body.get('delete_tunnel'):
        args.append('--delete-tunnel')
    return OPS.start(match['service'], 'stop', args, instance_id, f'Stop {instance_id}')


def start_delete(instance_id: str, body: dict, rows=None) -> dict:
    """Run `colabllm <svc> purge` for a stopped instance; the CLI re-checks liveness under its own lock."""
    _instance_dir(instance_id)
    match = next((row for row in (rows if rows is not None else instances()) if row['id'] == instance_id), None)
    if not match:
        raise LookupError('unknown instance')
    if match['status'] in ('ready', 'starting'):
        raise RuntimeError(f'{instance_id} is still running; stop it before deleting')
    args = ['purge']
    if match['instance_arg']:
        args += ['-i', match['instance_arg']]
    if match['owns_tunnel']:
        args.append('--delete-tunnel' if body.get('delete_tunnel', True) else '--keep-tunnel')
    return OPS.start(match['service'], 'delete', args, instance_id, f'Delete {instance_id}')


def _many(body: dict, action) -> dict:
    """Apply a lifecycle action to several instances; each is its own tracked operation, so one failure never hides the others."""
    ids = body.get('ids')
    if not isinstance(ids, list) or not ids or len(ids) > 100 or not all(isinstance(i, str) for i in ids):
        raise ValueError('ids must be a non-empty list of instance ids')
    rows = instances()
    results = []
    for instance_id in dict.fromkeys(ids):
        try:
            results.append(dict(id=instance_id, ok=True, operation=action(instance_id, body, rows)))
        except (LookupError, ValueError, RuntimeError) as error:
            results.append(dict(id=instance_id, ok=False, error=str(error)))
    return dict(results=results)


def start_down_many(body: dict) -> dict:
    return _many(body, start_down)


def start_delete_many(body: dict) -> dict:
    return _many(body, start_delete)


def doctor(online=False) -> dict:
    from doctor import diagnose
    return diagnose(online=online, timeout=10)


# --------------------------------------------------------------------------- HTTP
class Handler(http.server.BaseHTTPRequestHandler):
    server_version = 'colabllm-ui'
    sys_version = ''
    token = ''
    allowed_hosts: set[str] = set()

    def log_message(self, fmt, *args):  # keep the terminal clean; errors are reported to the client
        if os.environ.get('COLABLLM_UI_DEBUG'):
            super().log_message(fmt, *args)

    # -- helpers
    def _security_headers(self):
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Security-Policy',
                         "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
                         "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")

    def _send(self, status, payload=None, *, body=None, content_type='application/json; charset=utf-8'):
        data = body if body is not None else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self._security_headers()
        self.end_headers()
        if self.command != 'HEAD':
            self.wfile.write(data)

    def _error(self, status, message):
        self._send(status, {'error': message})

    def _host_ok(self):
        return self.headers.get('Host', '') in self.allowed_hosts

    def _authorized(self):
        supplied = self.headers.get('X-Colabllm-Token', '')
        return bool(supplied) and secrets.compare_digest(supplied, self.token)

    def _body(self):
        length = int(self.headers.get('Content-Length') or 0)
        if length > MAX_BODY:
            raise ValueError('request body too large')
        if not length:
            return {}
        if 'application/json' not in (self.headers.get('Content-Type') or ''):
            raise ValueError('expected application/json')
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError('expected a JSON object')
        return data

    # -- routing
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._error(421, 'unexpected Host header')
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path
        if not path.startswith('/api/'):
            return self._static(path)
        if not self._authorized():
            return self._error(401, 'missing or invalid access token')
        query = urllib.parse.parse_qs(parsed.query)
        parts = path.strip('/').split('/')[1:]
        try:
            if parts == ['overview']:
                return self._send(200, dict(environment=environment(), services=catalog(), instances=instances(),
                                            operations=OPS.list()[:50], gpus=GPUS, now=time.time()))
            if parts == ['usage']:
                return self._send(200, dict(usage=account_usage(force='refresh' in query)))
            if parts == ['doctor']:
                return self._send(200, doctor(online='online' in query))
            if parts == ['operations']:
                return self._send(200, OPS.list())
            if len(parts) == 2 and parts[0] == 'operations':
                return self._send(200, OPS.get(parts[1]))
            if len(parts) == 3 and parts[0] == 'services' and parts[2] == 'options':
                return self._send(200, dict(service=parts[1], options=option_schema(parts[1])))
            if len(parts) == 3 and parts[0] == 'instances':
                if parts[2] == 'secrets':
                    return self._send(200, instance_secrets(parts[1]))
                if parts[2] == 'logs':
                    return self._send(200, tail_log(parts[1]))
                if parts[2] == 'health':
                    return self._send(200, probe_instance(parts[1]))
            return self._error(404, 'not found')
        except LookupError as error:
            return self._error(404, str(error))
        except Exception as error:  # surface, never crash the server thread silently
            return self._error(500, f'{type(error).__name__}: {error}')

    def do_POST(self):
        if not self._host_ok():
            return self._error(421, 'unexpected Host header')
        origin = self.headers.get('Origin')
        if origin and urllib.parse.urlsplit(origin).netloc not in self.allowed_hosts:
            return self._error(403, 'cross-origin request rejected')
        if not self._authorized():
            return self._error(401, 'missing or invalid access token')
        parts = urllib.parse.urlsplit(self.path).path.strip('/').split('/')[1:]
        try:
            body = self._body()
            if len(parts) == 3 and parts[0] == 'services' and parts[2] == 'deploy':
                return self._send(202, start_up(parts[1], body))
            if len(parts) == 3 and parts[0] == 'instances' and parts[2] == 'stop':
                return self._send(202, start_down(parts[1], body))
            if len(parts) == 3 and parts[0] == 'instances' and parts[2] == 'delete':
                return self._send(202, start_delete(parts[1], body))
            if parts == ['instances', 'stop']:
                return self._send(202, start_down_many(body))
            if parts == ['instances', 'delete']:
                return self._send(202, start_delete_many(body))
            return self._error(404, 'not found')
        except LookupError as error:
            return self._error(404, str(error))
        except (ValueError, RuntimeError) as error:
            return self._error(409 if isinstance(error, RuntimeError) else 400, str(error))
        except Exception as error:
            return self._error(500, f'{type(error).__name__}: {error}')

    def _static(self, path):
        name = 'index.html' if path in ('/', '/index.html') else path.lstrip('/')
        asset = ASSETS.get(name) if re.fullmatch(r'[A-Za-z0-9_-]+\.(html|css|js|svg)', name) else None
        if asset is None:
            return self._error(404, 'not found')
        self._send(200, body=asset[0], content_type=asset[1])


def _load_assets() -> dict[str, tuple[bytes, str]]:
    """Snapshot the UI once, so a long-running server never serves a newer page than its own API understands."""
    return {f.name: (f.read_bytes(), CONTENT_TYPES[f.suffix])
            for f in sorted(ASSETS_DIR.iterdir()) if f.is_file() and f.suffix in CONTENT_TYPES}


ASSETS = _load_assets()


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(port=8765, open_browser=True, token=None):
    token = token or secrets.token_urlsafe(24)
    server = Server(('127.0.0.1', port), Handler)
    actual = server.server_address[1]
    Handler.token = token
    Handler.allowed_hosts = {f'127.0.0.1:{actual}', f'localhost:{actual}'}
    url = f'http://127.0.0.1:{actual}/#token={token}'
    print(f'colabllm control panel: {url}', flush=True)
    print('Listening on 127.0.0.1 only. Press Ctrl+C to stop; running services are not affected.', file=sys.stderr, flush=True)
    if open_browser:
        import webbrowser
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
