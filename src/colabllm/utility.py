"""Environment diagnostics and offline shell completion commands."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/'lib'))
from cli_support import CliError, positive_seconds, run_cli
from doctor import diagnose


def _port(value):
    port = int(value)
    if not 0 <= port <= 65535:
        raise ValueError('port must be between 0 and 65535')
    return port


def build_parser():
    parser = argparse.ArgumentParser(prog='colabllm', allow_abbrev=False)
    parser.add_argument('--command-timeout', type=positive_seconds, help='timeout for external commands, in seconds')
    parser.add_argument('--debug', action='store_true', help='print exception tracebacks')
    parser.add_argument('--quiet', action='store_true', help='suppress diagnostic output')
    parser.add_argument('--no-color', action='store_true', help='disable colored diagnostic output')
    commands = parser.add_subparsers(dest='command', required=True)
    doctor = commands.add_parser('doctor', help='check local dependencies, configuration and authorization')
    doctor.add_argument('--json', action='store_true', help='print the check results as JSON')
    doctor.add_argument('--transport', choices=('quick', 'cloudflare', 'frp'), help='check the given tunnel backend; read from the configuration by default')
    doctor.add_argument('--no-drive', action='store_true', help='skip the Drive extension checks')
    doctor.add_argument('--online', action='store_true', help='also run time-limited read-only authorization checks; never opens a browser or creates a session')
    doctor.add_argument('--timeout', type=positive_seconds, default=10, help='timeout per command check, in seconds (default 10)')
    completion = commands.add_parser('completion', help='print a shell completion script')
    completion.add_argument('shell', choices=('bash', 'zsh', 'fish'))
    skill = commands.add_parser('skill', help='show or export the bundled agent skill')
    skill.add_argument('--output', metavar='DIR', help='export the full skill to DIR/colabllm; never overwrites an existing directory')
    ui = commands.add_parser('ui', help='open the local web control panel (listens on 127.0.0.1 only)')
    ui.add_argument('--port', type=_port, default=8765, help='local port (default 8765; 0 picks a free port)')
    ui.add_argument('--no-browser', action='store_true', help='print the access link without opening a browser')
    commands.metavar = '{doctor,completion,skill,ui}'
    hidden = commands.add_parser('_complete')
    hidden.add_argument('words', nargs=argparse.REMAINDER)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == 'doctor':
        result = diagnose(transport=args.transport, no_drive=args.no_drive, online=args.online, timeout=args.timeout)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            for check in result['checks']:
                print(f"[{check['status'].upper()}] {check['name']}: {check['message']}")
        return 0 if result['ok'] else 1
    if args.command == 'ui':
        from webui import serve
        try:
            return serve(port=args.port, open_browser=not args.no_browser)
        except OSError as error:
            raise CliError(f'Cannot listen on 127.0.0.1:{args.port}: {error.strerror or error}; use --port to pick another port') from None
    if args.command == 'skill':
        from skill_bundle import bundle_root, export_skill
        if args.output:
            print(export_skill(args.output))
        else:
            print((bundle_root() / 'SKILL.md').read_text(encoding='utf-8'), end='')
        return 0
    from completion import candidates, shell_script
    if args.command == 'completion':
        print(shell_script(args.shell), end='')
    else:
        words = args.words[1:] if args.words[:1] == ['--'] else args.words
        for value in candidates(words):
            print(value)
    return 0


if __name__ == '__main__':
    run_cli(main)
