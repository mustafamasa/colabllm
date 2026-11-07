"""Offline completion from the real argparse registrations; never read credential contents."""
import argparse
from pathlib import Path
import re
import runpy

from cli_support import UsageError
from colabsvc import build_parser
from paths import user_home

ROOT = Path(__file__).resolve().parents[1]
GLOBAL = ('services', 'help', 'doctor', 'completion', 'skill', 'ui', 'tunnel-config', 'frp-config', 'frp-status')
FLAGS = ('--help', '--version', '--debug', '--quiet', '--no-color', '--command-timeout')


def service_scripts():
    return {p.stem: p for p in (ROOT/'services').glob('*.py') if p.with_suffix('.ipynb').is_file()}


def instance_ids(service):
    directory = user_home() / 'services'
    if not directory.is_dir():
        return []
    scripts = service_scripts()
    longer = {name for name in scripts if len(name) > len(service) and name.startswith(service + '-')}
    prefix = service + '-'
    results = []
    for p in directory.iterdir():
        if not p.is_dir() or not p.name.startswith(prefix):
            continue
        if any(p.name == other or p.name.startswith(other + '-') for other in longer):
            continue
        inst = p.name[len(prefix):]
        if re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_-]{0,63}', inst):
            results.append(inst)
    return sorted(results)


def _normalize_wordbreaks(words):
    """Rejoin ['--flag', '=', 'val'] or ['--flag', '='] produced by Bash COMP_WORDBREAKS."""
    merged = []
    idx = 0
    while idx < len(words):
        tok = words[idx]
        if tok == '=' and merged and merged[-1].startswith('-'):
            if idx + 1 < len(words):
                merged[-1] = merged[-1] + '=' + words[idx + 1]
                idx += 2
                continue
            merged[-1] = merged[-1] + '='
            idx += 1
            continue
        merged.append(tok)
        idx += 1
    return merged


def candidates(words):
    words = _normalize_wordbreaks(list(words) or [''])
    if words[0] == 'help' and len(words) > 1:
        words = words[1:]
    prefix, done = words[-1], words[:-1]
    scripts = service_scripts()
    # Leading global flags can precede the service name.
    while done and done[0].startswith('-'):
        flag = done.pop(0)
        if flag == '--command-timeout':
            if not done:
                return []
            done.pop(0)
    if not done:
        return sorted(value for value in (*scripts, *GLOBAL, *FLAGS) if value.startswith(prefix))
    service, *done = done
    if service == 'completion':
        return [shell for shell in ('bash', 'zsh', 'fish') if shell.startswith(prefix)]
    if service not in scripts:
        parser = runpy.run_path(str(ROOT/'utility.py'))['build_parser']() if service in ('doctor', 'skill', 'ui') else None
        if service in ('tunnel-config', 'frp-config', 'frp-status'):
            # Global controller parsers are also shared with completion.
            parser = runpy.run_path(str(ROOT/'control.py'))['build_parser']()
        if parser is None:
            return []
        done.insert(0, service)
    else:
        module = runpy.run_path(str(scripts[service]))
        parser = build_parser(module['SERVICE'], module.get('configure_cli'))
    root_options = {flag: action for action in parser._actions for flag in action.option_strings}
    expected = None
    for token in done:
        if token == '--':
            return []
        if expected is not None:
            if expected.nargs == '?' and token.startswith('-'):
                expected = None
            else:
                expected = None
                continue
        options = {**root_options, **{flag: action for action in parser._actions for flag in action.option_strings}}
        if token.split('=', 1)[0] in options:
            action = options[token.split('=', 1)[0]]
            if '=' not in token and action.nargs != 0:
                expected = action
            continue
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction) and token in action.choices:
                parser = action.choices[token]
                break
    options = {**root_options, **{flag: action for action in parser._actions for flag in action.option_strings}}
    flag_prefix = ''
    if '=' in prefix and prefix.split('=', 1)[0] in options:
        flag, prefix = prefix.split('=', 1)
        expected = options[flag]
        flag_prefix = flag + '='
    if expected is not None and not (expected.nargs == '?' and prefix.startswith('-') and not flag_prefix):
        if expected.dest == 'instance':
            values = instance_ids(service)
        else:
            values = list(expected.choices or ())
        return [flag_prefix + str(value) for value in values if str(value).startswith(prefix)]
    values = list(options)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            values.extend(name for name in action.choices if not name.startswith('_'))
    return sorted(set(value for value in values if value.startswith(prefix)))


SCRIPTS = {
    'bash': r'''# colabllm Bash completion
_colabllm_complete() {
    local candidate
    COMPREPLY=()
    while IFS= read -r candidate; do
        [[ -n "$candidate" ]] && COMPREPLY+=("$candidate")
    done < <(command colabllm _complete -- "${COMP_WORDS[@]:1:COMP_CWORD}" 2>/dev/null)
}
complete -o default -F _colabllm_complete colabllm
''',
    'zsh': r'''#compdef colabllm
_colabllm_complete() {
    local out
    local -a candidates
    out="$(command colabllm _complete -- "${words[@]:1:$((CURRENT-1))}" 2>/dev/null)"
    candidates=(${(f)out})
    (( ${#candidates[@]} )) && _describe 'colabllm' candidates || _files
}
compdef _colabllm_complete colabllm
''',
    'fish': r'''function __colabllm_candidates
    set -l words (commandline -opc)
    set -l current (commandline -ct)
    command colabllm _complete -- $words[2..-1] "$current" 2>/dev/null
end
complete -c colabllm -f -a '(__colabllm_candidates)'
''',
}


def shell_script(shell):
    try:
        return SCRIPTS[shell]
    except KeyError:
        raise UsageError('Supported shells: bash, zsh, fish') from None
