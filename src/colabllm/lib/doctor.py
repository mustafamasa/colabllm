"""Bounded, read-only environment diagnostics. Offline by default."""
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

from cli_support import command_timeout, positive_seconds
from colabsvc import STATE_ROOT, load_tunnel_config


def diagnose(*, transport=None, no_drive=False, online=False, timeout=10):
    timeout = min(positive_seconds(timeout), command_timeout())
    checks = []
    def add(name, status, message, required=True):
        checks.append(dict(name=name, status=status, message=message, required=required))
    def command(name, args, message, required=True):
        try:
            result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                # A successful no-op browser prevents webbrowser from falling back to a real browser.
                timeout=timeout, env={**os.environ, 'BROWSER': 'true', 'NO_COLOR': '1'})
            add(name, 'pass' if result.returncode == 0 else 'fail', message if result.returncode else 'available', required)
        except subprocess.TimeoutExpired:
            add(name, 'fail', f'check timed out ({timeout:g} s); {message}', required)
        except OSError:
            add(name, 'fail', message, required)
    add('python', 'pass' if sys.version_info >= (3, 12) else 'fail', f'Python {sys.version.split()[0]}; 3.12+ required')
    config = {}
    try:
        config = load_tunnel_config()
        add('config', 'pass', str(STATE_ROOT / 'tunnel-config.json'))
    except (OSError, ValueError, RuntimeError):
        add('config', 'fail', 'cannot read the configuration; check the format and permissions of tunnel-config.json')
    effective_domain = os.environ.get('COLABSVC_BASE_DOMAIN') or config.get('base_domain')
    selected = transport or ('frp' if (config.get('frp') or {}).get('enabled') else ('cloudflare' if effective_domain else 'quick'))
    needs_cf = selected == 'frp' or (selected == 'cloudflare' and bool(transport == 'cloudflare' or effective_domain))
    required = ['colab'] + (['cf'] if needs_cf else []) + (['ssh'] if selected == 'frp' else [])
    for executable in required:
        add(executable, 'pass' if shutil.which(executable) else 'fail',
            'found' if shutil.which(executable) else f'{executable} not found; install it as described in the README and add it to PATH')
    if selected == 'frp':
        from frp_backend import credentials_path
        frp = config.get('frp') or {}
        has_cfg = bool(frp.get('server_ip') and frp.get('ssh') and effective_domain)
        has_creds = bool(has_cfg and credentials_path(STATE_ROOT, frp).is_file())
        if has_cfg and has_creds:
            add('frp-config', 'pass', 'FRP configuration and local credentials are ready')
        elif has_cfg:
            add('frp-config', 'fail', 'local FRP credential file is missing; run colabllm frp-config again')
        else:
            add('frp-config', 'fail', 'FRP needs server_ip, ssh and base_domain; configure them with colabllm frp-config')
    if shutil.which('colab'):
        command('colab-version', ['colab', 'version'], 'cannot run colab version; check the Colab CLI installation')
        if not no_drive:
            command('drive-extension', ['colab', 'dfs', '--help'], 'the Drive extension is missing or cannot run; install colab dfs as described in the README')
    if not no_drive:
        credentials = Path.home() / '.config/colab-cli/dfs_credentials.json'
        add('drive-consent', 'pass' if credentials.is_file() else 'warn',
            'Drive credential file found (offline check)' if credentials.is_file() else 'no Drive credential file found; run colab dfs consent', False)
    # Inspect paths without creating or changing any files.
    parent = STATE_ROOT
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    add('state-writable', 'pass' if os.access(parent, os.W_OK | os.X_OK) else 'fail', str(STATE_ROOT))
    if STATE_ROOT.exists():
        private = stat.S_IMODE(STATE_ROOT.stat().st_mode) & 0o077 == 0
        add('state-permissions', 'pass' if private else 'warn',
            'user directory permissions are private' if private else 'setting the user directory permissions to 0700 is recommended', False)
    if online:
        if shutil.which('colab'):
            command('colab-auth', ['colab', 'sessions'], 'cannot query sessions read-only; run colab sessions interactively first to authorize')
        if 'cf' in required and shutil.which('cf'):
            command('cloudflare-auth', ['cf', 'zones', 'list'], 'cannot list DNS zones; run cf auth login and check permissions')
        target = (config.get('frp') or {}).get('ssh') if selected == 'frp' else None
        if target and shutil.which('ssh'):
            command('ssh-auth', ['ssh', '-o', 'BatchMode=yes', '-o', f'ConnectTimeout={max(1, int(timeout))}', '--', target, 'true'],
                    'non-interactive SSH authentication failed; configure a key and a verified host fingerprint first')
    else:
        add('authentication', 'warn', 'authorization has not been verified online; use doctor --online for a time-limited read-only check', False)
    ok = not any(item['required'] and item['status'] == 'fail' for item in checks)
    return dict(ok=ok, transport=selected, checks=checks)
