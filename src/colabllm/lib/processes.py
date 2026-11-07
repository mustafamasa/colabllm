"""Process ownership identities and bounded shutdown using only the standard library."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_identity(pid):
    if not pid:
        return None
    try:
        pid = int(pid)
        if pid <= 0:
            return None
        if sys.platform.startswith('linux'):
            fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z':
                return None
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            return boot + ':' + fields[19]
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'lstart='],
                                text=True, capture_output=True, timeout=3,
                                env={**os.environ, 'LC_ALL': 'C', 'TZ': 'UTC'})
        if result.returncode != 0 or not result.stdout.strip():
            return None
        state, identity = result.stdout.strip().split(maxsplit=1)
        return None if state.startswith('Z') else identity
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def owned_process(pid, identity):
    return bool(identity and process_identity(pid) == identity)


def runner_alive(state):
    return owned_process(state.get('runner_pid'), state.get('runner_identity'))


def stop_owned_process(pid, identity, *, group=False, timeout=5):
    """Never signal an unverifiable or reused PID; wait before escalation."""
    if not owned_process(pid, identity):
        return
    pid = int(pid)
    def send(sig):
        if owned_process(pid, identity):
            try:
                if group and os.getpgid(pid) == pid:
                    os.killpg(pid, sig)
                else:
                    os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
    send(signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while owned_process(pid, identity) and time.monotonic() < deadline:
        time.sleep(0.05)
    if owned_process(pid, identity):
        send(signal.SIGKILL)
        deadline = time.monotonic() + 2
        while owned_process(pid, identity) and time.monotonic() < deadline:
            time.sleep(0.05)
    if owned_process(pid, identity):
        raise RuntimeError(f'Process {pid} has not exited yet; keeping the runtime state so the operation can be retried')
