"""Shared CLI errors, diagnostic output and authenticated HTTP transport."""
import http.client
import math
import time
import os
import subprocess
import sys
import traceback
import urllib.error
import urllib.parse
import urllib.request

QUIET = False
NO_COLOR = False


class CliError(RuntimeError):
    def __init__(self, message, code=1):
        super().__init__(message)
        self.code = code


class UsageError(ValueError):
    code = 2


def diagnostic(message, color=None):
    if QUIET:
        return
    if color and not NO_COLOR and not os.environ.get('NO_COLOR') and sys.stderr.isatty():
        message = f'\033[{color}m{message}\033[0m'
    print(message, file=sys.stderr, flush=True)


def nonnegative_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be a finite, non-negative number')
    return seconds


def positive_seconds(value):
    seconds = nonnegative_seconds(value)
    if seconds == 0:
        raise ValueError('seconds must be greater than 0')
    return seconds


def command_timeout(default=300):
    value = os.environ.get('COLABLLM_COMMAND_TIMEOUT')
    try:
        return positive_seconds(value) if value else default
    except ValueError:
        raise UsageError('COLABLLM_COMMAND_TIMEOUT must be a finite number of seconds greater than 0') from None


def http_timeout(timeout=None, default=120):
    if timeout is None:
        return command_timeout(default)
    seconds = nonnegative_seconds(timeout)
    return None if seconds == 0 else seconds


class Deadline:
    def __init__(self, timeout=0, message='Timed out waiting; the background task keeps running and can be queried later'):
        timeout = nonnegative_seconds(timeout)
        self.end = time.monotonic() + timeout if timeout else None
        self.message = message

    def remaining(self, cap=None):
        if self.end is None:
            return cap
        remaining = self.end - time.monotonic()
        if remaining <= 0:
            raise CliError(self.message, 124)
        return min(remaining, cap) if cap is not None else remaining

    def sleep(self, seconds):
        time.sleep(self.remaining(seconds))


def origin(url):
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        raise CliError('Invalid port in service URL') from None
    if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password:
        raise CliError('The service URL must be an HTTP(S) address without embedded credentials')
    return (parsed.scheme.lower(), parsed.hostname.lower(), port or (443 if parsed.scheme == 'https' else 80))


def resolve_endpoint(state, url=None, token=None, *, url_env=None, token_env=None):
    selected = (url or (os.environ.get(url_env) if url_env else None) or state.get('url') or '').rstrip('/')
    explicit = (token or (os.environ.get(token_env) if token_env else None) or '').strip()
    if not selected:
        raise CliError('No service address: start the service first, or pass --url and an API token explicitly')
    selected_origin = origin(selected)
    if explicit:
        return selected, explicit
    saved_url, saved_token = state.get('url'), state.get('api_token')
    if not saved_url or not saved_token or origin(saved_url) != selected_origin:
        raise CliError('The service origin changed or has no matching credentials; pass the API token argument or its environment variable explicitly')
    return selected, saved_token


class SameOriginRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if req.has_header('Authorization') and origin(req.full_url) != origin(newurl):
            raise CliError('Refusing to forward credentials to another site; check the service URL or redirect configuration')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def safe_urlopen(request, *args, **kwargs):
    return urllib.request.build_opener(SameOriginRedirect()).open(request, *args, **kwargs)


def configure_output():
    global QUIET, NO_COLOR
    args = []
    literal = False
    debug = False
    argv = iter(sys.argv[1:])
    for arg in argv:
        if arg == '--':
            literal = True
        if not literal and arg in ('--debug', '--quiet', '--no-color'):
            debug |= arg == '--debug'
            QUIET |= arg == '--quiet'
            NO_COLOR |= arg == '--no-color'
        elif not literal and (arg == '--command-timeout' or arg.startswith('--command-timeout=')):
            value = next(argv, None) if arg == '--command-timeout' else arg.split('=', 1)[1]
            try:
                os.environ['COLABLLM_COMMAND_TIMEOUT'] = str(positive_seconds(value))
            except (ValueError, TypeError):
                raise UsageError('--command-timeout requires a finite number of seconds greater than 0') from None
        else:
            args.append(arg)
    sys.argv[1:] = args
    return debug


def run_cli(action):
    """Handle expected user-facing failures consistently, including early parsing."""
    debug = False
    try:
        debug = configure_output()
        code = action()
        if isinstance(code, int) and code:
            raise SystemExit(code)
    except BrokenPipeError:
        # Avoid a second flush failure at interpreter shutdown.
        with open(os.devnull, 'w') as sink:
            os.dup2(sink.fileno(), sys.stdout.fileno())
        raise SystemExit(141) from None
    except KeyboardInterrupt:
        diagnostic('Command interrupted; use status/job to check the service or task state.')
        raise SystemExit(130) from None
    except (OSError, ValueError, RuntimeError, http.client.HTTPException, subprocess.SubprocessError) as error:
        if debug:
            traceback.print_exc()
        else:
            message = str(error)
            if isinstance(error, urllib.error.HTTPError):
                message = f'HTTP {error.code}: {error.reason}'
            # Errors remain visible in quiet mode.
            print(f'colabllm: {message}', file=sys.stderr)
        is_timeout = isinstance(error, (TimeoutError, subprocess.TimeoutExpired)) or (
            isinstance(error, urllib.error.URLError)
            and isinstance(getattr(error, 'reason', None), TimeoutError)
        )
        code = error.code if isinstance(error, (CliError, UsageError)) else (124 if is_timeout else 1)
        raise SystemExit(code) from None
