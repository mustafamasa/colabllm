#!/usr/bin/env python3
"""Project-wide configuration shared by every Colab runtime service."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'lib'))
from cli_support import UsageError, positive_seconds, run_cli
import colabsvc
from colabsvc import cmd_tunnel_config, validate_domain


def build_parser():
    parser = argparse.ArgumentParser(description='Global configuration shared by all runtimes of this project', allow_abbrev=False)
    parser.add_argument('--command-timeout', type=positive_seconds, help='timeout for external commands, in seconds')
    parser.add_argument('--debug', action='store_true', help='print exception tracebacks')
    parser.add_argument('--quiet', action='store_true', help='suppress diagnostic output')
    parser.add_argument('--no-color', action='store_true', help='disable colored diagnostic output')
    commands = parser.add_subparsers(dest='command', required=True)
    tunnel = commands.add_parser('tunnel-config', help='show, set or clear the project-wide base domain')
    settings = tunnel.add_mutually_exclusive_group()
    settings.add_argument('--base-domain', metavar='DOMAIN', type=validate_domain,
                          help='base domain shared by all services, e.g. example.com')
    settings.add_argument('--clear', action='store_true', help='clear the global base domain; dynamic URLs are used by default')
    frp = commands.add_parser('frp-config', help='configure the FRP server and install it over SSH; shows the configuration when given no arguments')
    frp.add_argument('--server-ip', help='public IPv4/IPv6 address of the FRP server')
    frp.add_argument('--ssh', help='SSH target, e.g. root@frpip or an alias from ~/.ssh/config')
    frp.add_argument('--base-domain', type=validate_domain, help='project-wide base domain; DNS is managed by the cf CLI')
    frp.add_argument('--server-port', type=int, help='FRP server port, default 7000')
    frp.add_argument('--disable', action='store_true', help='disable global FRP; later starts go back to Cloudflare')
    commands.add_parser('frp-status', help='check the FRP/Caddy services and assigned routes over SSH')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'tunnel-config':
            cmd_tunnel_config(args)
        elif args.command == 'frp-config':
            from frp_backend import configure
            if args.disable and any(getattr(args, k) is not None for k in ('server_ip', 'ssh', 'base_domain', 'server_port')):
                parser.error('--disable cannot be combined with server configuration arguments')
            result = configure(args, colabsvc.STATE_ROOT)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            from frp_backend import remote
            config = colabsvc.load_tunnel_config().get('frp')
            if not config:
                raise UsageError('No FRP server is configured yet; run colabllm frp-config first')
            print(json.dumps(remote(config, {'action': 'status'}), ensure_ascii=False, indent=2))
    except ValueError as error:
        if isinstance(error, UsageError):
            raise
        raise UsageError(str(error)) from None


if __name__ == '__main__':
    run_cli(main)
