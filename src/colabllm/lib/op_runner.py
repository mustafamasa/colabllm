"""Run one control-panel operation independently of the panel process. Standard library only.

Usage: op_runner.py EXIT_FILE -- COMMAND [ARGS...]

The panel starts this wrapper in its own session with stdout/stderr already pointing at the
operation's log file, so closing the panel neither signals nor breaks the operation. The wrapper
forwards SIGTERM/SIGINT/SIGHUP to the command (letting `up` cancel cleanly) and records the exit
status in EXIT_FILE, which a later panel run reads to show the result.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def write_exit(path: Path, code: int) -> None:
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
        json.dump({'exit_code': code, 'finished': time.time()}, handle)
    os.replace(tmp, path)


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != '--':
        print('usage: op_runner.py EXIT_FILE -- COMMAND [ARGS...]', file=sys.stderr)
        return 2
    exit_path, cmd = Path(argv[1]), argv[3:]
    try:
        child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL)
    except OSError as error:
        print(f'Failed to start: {error}', flush=True)
        write_exit(exit_path, 127)
        return 127

    def forward(signum, frame):
        if child.poll() is None:
            child.send_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, forward)
    code = child.wait()
    code = code if code >= 0 else 128 - code
    write_exit(exit_path, code)
    return code


if __name__ == '__main__':
    sys.exit(main(sys.argv))
