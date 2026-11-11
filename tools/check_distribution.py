#!/usr/bin/env python3
"""Verify a built wheel or sdist with uv tool install, without touching the user's installation."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def _inspect_wheel(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        services = sorted(Path(n).stem for n in names if n.startswith('colabllm/services/') and n.endswith('.py'))
        notebooks = sorted(Path(n).stem for n in names if n.startswith('colabllm/services/') and n.endswith('.ipynb'))
        assert services, 'Wheel contains no services'
        assert services == notebooks, f'Service script/notebook mismatch: {services} != {notebooks}'
        assert 'colabllm/lib/frp_server.py' in names, 'Missing SSH provisioning source'
        for asset in ('index.html', 'app.css', 'app.js', 'favicon.svg'):
            assert f'colabllm/webui/{asset}' in names, f'Missing web UI asset {asset}'
        assert any(n.endswith('/licenses/LICENSE') or n.endswith('/LICENSE') for n in names), 'Missing LICENSE in wheel'
        meta_files = [n for n in names if n.endswith('.dist-info/METADATA')]
        assert meta_files, 'Missing METADATA in wheel'
        metadata = archive.read(meta_files[0]).decode('utf-8')
        assert 'License-Expression: MIT' in metadata, 'Missing MIT License-Expression in wheel METADATA'
        for service in services:
            notebook = json.loads(archive.read(f'colabllm/services/{service}.ipynb'))
            assert notebook['cells'], f'Empty notebook: {service}'
            assert all(not cell.get('outputs') for cell in notebook['cells']), f'Notebook outputs: {service}'
        assert not any('/.state/' in n or n.endswith('state.json') for n in names)
        return services


def _inspect_sdist(sdist: Path) -> list[str]:
    with tarfile.open(sdist, 'r:gz') as archive:
        members = {m.name: m for m in archive.getmembers() if m.isfile()}
        names = sorted(members)
        services = sorted(Path(n).stem for n in names if '/src/colabllm/services/' in n and n.endswith('.py'))
        notebooks = sorted(Path(n).stem for n in names if '/src/colabllm/services/' in n and n.endswith('.ipynb'))
        assert services, 'Source distribution contains no services'
        assert services == notebooks, f'Service script/notebook mismatch in sdist: {services} != {notebooks}'
        assert any(n.endswith('/src/colabllm/lib/frp_server.py') for n in names), 'Missing SSH provisioning source in sdist'
        assert any(n.endswith('/src/colabllm/webui/app.js') for n in names), 'Missing web UI assets in sdist'
        assert any(n.endswith('/LICENSE') for n in names), 'Missing LICENSE in sdist'
        pkg_infos = [n for n in names if n.endswith('/PKG-INFO')]
        assert pkg_infos, 'Missing PKG-INFO in sdist'
        pkg_info = archive.extractfile(members[pkg_infos[0]]).read().decode('utf-8')
        assert 'License-Expression: MIT' in pkg_info, 'Missing MIT License-Expression in sdist PKG-INFO'
        for name in names:
            if '/src/colabllm/services/' in name and name.endswith('.ipynb'):
                notebook = json.loads(archive.extractfile(members[name]).read())
                assert notebook['cells'], f'Empty notebook in sdist: {name}'
                assert all(not cell.get('outputs') for cell in notebook['cells']), f'Notebook outputs in sdist: {name}'
        assert not any('/.state/' in n or n.endswith('state.json') for n in names)
        return services


def check(artifact):
    artifact = Path(artifact).resolve(strict=True)
    if artifact.name.endswith('.whl'):
        services = _inspect_wheel(artifact)
    elif artifact.name.endswith('.tar.gz'):
        services = _inspect_sdist(artifact)
    else:
        raise ValueError(f'Unsupported distribution artifact: {artifact}')

    repo_services_dir = ROOT / 'src/colabllm/services'
    if repo_services_dir.is_dir():
        expected = sorted(p.stem for p in repo_services_dir.glob('*.py') if p.with_suffix('.ipynb').is_file())
        assert services == expected, f'Packaged services {services} do not match checkout {expected}'

    with tempfile.TemporaryDirectory(prefix='colabllm-dist-') as temp:
        root = Path(temp).resolve()
        home = root / 'home'
        home.mkdir()
        env = {**os.environ, 'HOME': str(home), 'UV_TOOL_DIR': str(root / 'tools'),
               'UV_TOOL_BIN_DIR': str(root / 'bin'), 'UV_CACHE_DIR': str(root / 'cache')}
        env.pop('COLABLLM_HOME', None)
        env.pop('PYTHONPATH', None)
        env.pop('COLABSVC_BASE_DOMAIN', None)
        env.pop('CT_INSTANCE', None)

        # All subprocesses run outside the source checkout.
        def run(args):
            result = subprocess.run(args, env=env, cwd=root, text=True, capture_output=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f'{args}:\n{result.stdout}\n{result.stderr}')
            return result.stdout

        run(['uv', 'tool', 'install', '--python', sys.executable, str(artifact)])
        command = str(root / 'bin/colabllm')
        assert run([command, '--version']).startswith('colabllm ')
        assert run([command, 'services']).splitlines() == services
        business = {
            'ollama': ('endpoints', 'models', 'pull', 'run', 'preload', 'unload', 'cp', 'rm', 'show'),
            'qwen3-tts': ('tts', 'voices', 'design-voice'),
            'indextts': ('tts', 'voices'),
            'drama3': ('tts', 'narrate', 'voices', 'upload-voice', 'delete-voice', 'ls-drive'),
            'tangoflux': ('generate', 'sfx', 'batch', 'models', 'ls-drive'),
            'comfyui': ('pull-workflow', 'pull-model'),
            'openjutsu': ('check',),
            'unlimited-ocr': ('ocr', 'job', 'resume'),
            'longcat-avatar': ('generate', 'job', 'download'),
            'nemotron-transcribe': ('transcribe', 'download', 'ls-drive'),
            'flux-klein': ('generate', 'edit', 'upsample', 'ls-drive', 'models'),
            'qwen-image': ('generate', 'edit', 'ls-drive', 'models'),
            'qwen-flash': ('models', 'plan'),
            'deepseek-coder': ('models', 'plan'),
            'deepseek-qwen-9b': ('models', 'plan'),
        }
        for service in services:
            help_text = run([command, service, '--help'])
            assert '_exec' not in help_text
            for subcommand in business.get(service, ()):
                assert subcommand in help_text
                run([command, service, subcommand, '--help'])
            json.loads(run([command, service, 'info', '--json']))
        run([command, 'doctor', '--help'])
        for shell in ('bash', 'zsh', 'fish'):
            script = run([command, 'completion', shell])
            assert 'colabllm _complete' in script
        assert 'tts' in run([command, '_complete', '--', 'qwen3-tts', 't']).splitlines()
        run([command, 'frp-config', '--help'])
        skill_text = run([command, 'skill'])
        assert 'name: colabllm' in skill_text
        skill_parent = root / 'agent-skills'
        exported = Path(run([command, 'skill', '--output', str(skill_parent)]).strip())
        assert exported == skill_parent / 'colabllm'
        assert (exported / 'SKILL.md').read_text() == skill_text
        for reference in (*services, 'runtime-common', 'ocr', 'speech', 'inference', 'media', 'operations'):
            assert (exported / 'references' / (reference + '.md')).is_file()
        config = json.loads(run([command, 'tunnel-config', '--base-domain', 'example.com']))
        state_home = home / '.colabllm'
        assert Path(config['config_file']) == state_home / 'tunnel-config.json'
        assert json.loads((state_home / 'tunnel-config.json').read_text())['base_domain'] == 'example.com'
        assert stat.S_IMODE(state_home.stat().st_mode) == 0o700
        assert stat.S_IMODE((state_home / 'tunnel-config.json').stat().st_mode) == 0o600
        override = root / 'other-profile'
        env['COLABLLM_HOME'] = str(override)
        config = json.loads(run([command, 'tunnel-config']))
        assert Path(config['config_file']) == override / 'tunnel-config.json'
        # A stub ensures dry-run cannot accidentally contact real Colab.
        stub = root / 'bin/colab'
        stub.write_text('#!/bin/sh\nexit 99\n')
        stub.chmod(0o755)
        env['PATH'] = str(root / 'bin') + os.pathsep + env.get('PATH', '')
        for service in services:
            run([command, service, 'up', '--dry-run', '--quick-tunnel', '-i', 'check'])
        assert not override.exists(), 'Dry-run unexpectedly wrote state'
    kind = 'Wheel' if artifact.name.endswith('.whl') else 'Source distribution'
    print(f'{kind} verified: {len(services)} services; isolated uv installation, LICENSE, notebook resources, user config and dry-runs passed.')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python tools/check_distribution.py path/to/colabllm-VERSION.(whl|tar.gz)')
    check(sys.argv[1])
