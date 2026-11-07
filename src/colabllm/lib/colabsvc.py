"""colabsvc -- shared runner for "one LLM service = one notebook + one script" on Google Colab.

A service script (``services/<name>.py``) declares a :class:`Service` and calls :func:`main`.
Responsibilities are split like this:

  service script (local)                       notebook (Colab VM)
  ---------------------------------------      --------------------------------------------
  colab new / reuse session                    idempotent setup (models cached on Drive)
  colab dfs mount (browser-free Drive)         start the model service on 127.0.0.1
  --domain: cf CLI -> named tunnel, ingress,   AUTH=caddy: Caddy basic-auth / bearer proxy
            DNS CNAME, tunnel token            cloudflared (token -> fixed domain, else quick)
  generate credentials, pass via --env         print CT_* markers, then block forever
  keep `colab exec` attached (keep-alive)

Notebook contract (env injected with ``colab exec --env``):
  CT_SERVICE, CT_SERVICE_PORT, CT_PUBLIC_PORT, CT_AUTH (caddy|builtin|none),
  CT_AUTH_USER, CT_AUTH_PASS, CT_API_TOKEN, CT_TUNNEL_TOKEN, CT_PUBLIC_HOSTNAME, CT_DATA_DIR
Markers printed by the notebook (parsed here):
  CT_PUBLIC_URL=<url>   CT_CREDENTIALS user=<u> pass=<p>   CT_READY

Lessons baked in (verified on real runs):
  * `colab exec -f` sends cells one by one from the client: disconnecting early means later cells never run.
  * The session survives only while a `colab exec` connection stays open, so `up` keeps it attached
    in a detached local runner process; `down` stops both.
  * `drive.mount()` waits for browser consent forever -> patched to a no-op once `colab dfs` mounted Drive.
  * A failing cell does not change `colab exec`'s exit code -> readiness comes from the CT_READY marker.
"""
from __future__ import annotations

import argparse
import dataclasses
import contextlib
import fcntl
import signal
import selectors
import tempfile
import ipaddress
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from paths import private_dir, user_home
from cli_support import CliError, UsageError, command_timeout, positive_seconds, diagnostic, safe_urlopen
from protocol import NotebookProtocol
from processes import owned_process, process_identity, runner_alive, stop_owned_process

STATE_ROOT = user_home()
DEFAULT_ZONE = "example.com"

SKIP_RE = re.compile(r"^[ \t]*#[ \t]*@ct:skip\b", re.M)


@dataclasses.dataclass
class Service:
    name: str                       # also the Colab session name and the tunnel name suffix
    notebook: str                   # path relative to the service script
    service_port: int               # port the model service listens on (127.0.0.1)
    auth: str = "caddy"             # caddy (no built-in auth) | builtin | none
    public_port: int = 18080        # Caddy port when auth == "caddy" (not 8080/8012: used by Colab)
    gpu: str | None = "T4"          # T4|L4|G4|H100|A100 or None for CPU
    tpu: str | None = None
    high_mem: bool = False
    zone: str = DEFAULT_ZONE        # legacy suffix used to extract service aliases
    default_hostname: str | None = None  # legacy host supplying an alias (e.g. qwen.<zone>)
    auth_user: str = "admin"
    env: dict = dataclasses.field(default_factory=dict)   # extra env for the notebook
    ready_timeout: int = 3600       # seconds `up` waits for CT_READY
    base_name: str = ""             # base service name before --instance suffix
    base_default_hostname: str | None = None
    instance: str | None = None     # optional instance id (-i / --instance)

    def __post_init__(self) -> None:
        if not self.base_name:
            self.base_name = self.name
        if self.base_default_hostname is None:
            self.base_default_hostname = self.default_hostname

    @property
    def tunnel_port(self) -> int:
        return self.public_port if self.auth == "caddy" else self.service_port


def extract_instance_arg(argv: list[str] | None = None) -> tuple[str | None, list[str]]:
    """Extract `-i <id>` / `--instance <id>` from anywhere in argv so it works before or after the subcommand."""
    args = list(sys.argv[1:] if argv is None else argv)
    inst = os.environ.get("CT_INSTANCE") or None
    cleaned = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            cleaned.extend(args[i:])
            break
        if a in ("-i", "--instance"):
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise UsageError(f"{a} requires a valid instance ID")
            inst = validate_instance(args[i + 1])
            i += 2
        elif a.startswith("--instance="):
            inst = validate_instance(a.split("=", 1)[1])
            i += 1
        elif a.startswith("-i") and not a.startswith("--") and len(a) > 2:
            inst = validate_instance(a[2:])
            i += 1
        else:
            cleaned.append(a)
            i += 1
    return validate_instance(inst) if inst is not None else None, cleaned


def validate_instance(value):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_-]{0,63}", value):
        raise UsageError("The instance ID must be 1-64 letters, digits, underscores or hyphens and must not start with a hyphen")
    return value


def apply_instance(svc: Service, inst: str | None) -> Service:
    """Apply an instance suffix (`-i 2` -> `<base>-2`, domain `<host>-2.<zone>`) to `svc` in place."""
    if inst is not None:
        validate_instance(inst)
    base = svc.base_name or svc.name
    svc.base_name = base
    base_host = svc.base_default_hostname
    if base_host is None:
        base_host = svc.default_hostname
        svc.base_default_hostname = base_host
    svc.instance = inst
    if not inst:
        svc.name = base
        svc.default_hostname = base_host
        return svc
    suffix = f"-{inst}"
    svc.name = base if base.endswith(suffix) else f"{base}{suffix}"
    if base_host:
        if "." in base_host:
            sub, rest = base_host.split(".", 1)
            svc.default_hostname = f"{sub}{suffix}.{rest}" if not sub.endswith(suffix) else base_host
        else:
            svc.default_hostname = f"{base_host}{suffix}" if not base_host.endswith(suffix) else base_host
    return svc


def validate_domain(value: str) -> str:
    """Accept DNS hostnames, never URLs, wildcard patterns, ports or IPs."""
    try:
        host = value.strip().rstrip('.').encode('idna').decode('ascii').lower()
    except (UnicodeError, AttributeError):
        raise ValueError('Invalid domain; use a form like example.com, without scheme, port or path') from None
    labels = host.split('.')
    if len(host) > 253 or len(labels) < 2 or any(
        not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label) for label in labels
    ):
        raise ValueError('Invalid domain; use a form like example.com, without scheme, port or path')
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return host
    raise ValueError('Enter a DNS domain name; IP addresses are not allowed')


def load_tunnel_config() -> dict:
    path = STATE_ROOT / 'tunnel-config.json'
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise RuntimeError(f"Corrupted configuration file: {path}; fix the JSON file or restore it from a backup") from error
    if not isinstance(data, dict):
        raise ValueError('tunnel-config.json must be a JSON object')
    if data.get('base_domain'):
        data['base_domain']=validate_domain(data['base_domain'])
    if data.get('frp') is not None:
        from frp_backend import validate_config
        data['frp']=validate_config(data['frp'])
    return data


def save_tunnel_config(data: dict) -> None:
    from frp_backend import write_private
    path=STATE_ROOT/'tunnel-config.json'
    if data:
        write_private(path,data)
    else:
        path.unlink(missing_ok=True)


def tunnel_transport(args) -> str:
    explicit=getattr(args,'transport',None)
    if getattr(args,'quick_tunnel',False):
        if explicit=='frp':
            raise ValueError('--quick-tunnel cannot be used with --transport frp')
        return 'cloudflare'
    if explicit:
        return explicit
    return 'frp' if load_tunnel_config().get('frp',{}).get('enabled') else 'cloudflare'


def tunnel_hostname(svc: Service, args) -> str | None:
    if getattr(args, 'quick_tunnel', False):
        return None
    domain = getattr(args, 'domain', None)
    if domain and domain not in (True, '@default'):
        return validate_domain(domain)
    base = (getattr(args, 'base_domain', None) or os.environ.get('COLABSVC_BASE_DOMAIN')
            or load_tunnel_config().get('base_domain'))
    if not base:
        return None
    base = validate_domain(base)
    # Keep service aliases (qwen, transcribe, etc.) and instance suffixes,
    # replacing only their historical base domain.
    alias = svc.default_hostname or svc.name
    if '.' in alias:
        alias = alias.split('.', 1)[0]
    return validate_domain(f'{alias}.{base}')


def cmd_tunnel_config(args) -> None:
    path = STATE_ROOT / 'tunnel-config.json'
    saved=load_tunnel_config()
    if args.clear:
        if saved.get('frp',{}).get('enabled'):
            raise ValueError('FRP requires a base domain; disable FRP before clearing it')
        saved.pop('base_domain',None)
        save_tunnel_config(saved)
    elif args.base_domain is not None:
        saved['base_domain']=validate_domain(args.base_domain)
        save_tunnel_config(saved)
    saved = load_tunnel_config()
    print(json.dumps({'base_domain': saved.get('base_domain'),
                      'transport': 'frp' if saved.get('frp',{}).get('enabled') else 'cloudflare',
                      'frp': saved.get('frp'),
                      'environment_override': os.environ.get('COLABSVC_BASE_DOMAIN') or None,
                      'config_file': str(path)}, ensure_ascii=False, indent=2))


def _known_service_names() -> set[str]:
    services_dir = Path(__file__).resolve().parents[1] / "services"
    if not services_dir.is_dir():
        return set()
    return {p.stem for p in services_dir.glob("*.py") if p.with_suffix(".ipynb").is_file()}


def list_service_instances(base_name: str) -> list[dict]:
    """Return status dicts for the default instance and all `<base_name>-<id>` instances found in `~/.colabllm/services/`."""
    directory = STATE_ROOT / "services"
    if not directory.is_dir():
        return []
    longer_services = {name for name in _known_service_names() if len(name) > len(base_name) and name.startswith(base_name + "-")}
    items: list[dict] = []
    for d in sorted(directory.iterdir()):
        if not d.is_dir():
            continue
        if d.name == base_name:
            inst_id = ""
        elif d.name.startswith(base_name + "-"):
            inst_id = d.name[len(base_name) + 1:]
        else:
            continue
        sp = d / "state.json"
        if not sp.is_file():
            continue
        try:
            st = json.loads(sp.read_text())
        except Exception:
            continue
        recorded_base = st.get("base_service")
        if recorded_base is not None:
            if recorded_base != base_name:
                continue
        elif any(d.name == other or d.name.startswith(other + "-") for other in longer_services):
            continue
        alive = runner_alive(st)
        items.append({
            "instance": inst_id or "default",
            "instance_arg": inst_id,
            "name": d.name,
            "alive": alive,
            "running": alive,
            "ready": bool(st.get("ready") and alive),
            "runner_pid": st.get("runner_pid"),
            "url": st.get("url"),
            "api_token": st.get("api_token"),
            "auth_user": st.get("auth_user"),
            "auth_pass": st.get("auth_pass"),
            "started": st.get("started"),
        })
    return items


# --------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    diagnostic(f"[colabsvc] {msg}", "36")


def warn(msg: str) -> None:
    diagnostic(f"[colabsvc] {msg}", "33")


def die(msg: str, code: int = 1) -> None:
    raise CliError(msg, code)


def banner(lines: list[str]) -> None:
    w = max(len(l) for l in lines) + 8
    print("\n" + "=" * w + "\n" + "\n".join(f"    {line}" for line in lines) + "\n" + "=" * w + "\n", flush=True)


def run(cmd: list[str], *, capture: bool = False, check: bool = True, input: str | None = None, timeout=None):
    res = subprocess.run(cmd, text=True, input=input,
                         stdout=subprocess.PIPE if capture else None,
                         stderr=subprocess.PIPE if capture else None,
                         timeout=command_timeout() if timeout is None else timeout)
    if check and res.returncode != 0:
        if capture:
            sys.stderr.write((res.stdout or "") + (res.stderr or ""))
        raise CliError(f"Command failed ({res.returncode}): {' '.join(cmd[:4])}", res.returncode if res.returncode > 0 else 1)
    return res


def cf_json(*args: str):
    res = run(["cf", *args], capture=True)
    try:
        return json.loads(res.stdout)
    except json.JSONDecodeError:
        die(f"unexpected output from cf {' '.join(args[:3])}: {res.stdout[:300]}")


# --------------------------------------------------------------------------- state
class State:
    def __init__(self, svc: Service):
        self.dir = STATE_ROOT / "services" / svc.name
        self.path = self.dir / "state.json"

    def load(self) -> dict:
        try:
            data = json.loads(self.path.read_text()) if self.path.exists() else {}
            if not isinstance(data, dict):
                raise ValueError("expected an object")
            return data
        except (ValueError, UnicodeError) as error:
            raise RuntimeError(f"Corrupted state file: {self.path}; fix the JSON file or restore it from a backup") from error

    @contextlib.contextmanager
    def lock(self, name="state", blocking=True):
        private_dir(STATE_ROOT)
        private_dir(self.dir.parent)
        private_dir(self.dir)
        with (self.dir / f".{name}.lock").open("a") as lockfile:
            os.chmod(lockfile.name, 0o600)
            try:
                fcntl.flock(lockfile, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            except BlockingIOError:
                raise RuntimeError(f"{self.dir.name} is running another lifecycle operation; try again later") from None
            try:
                yield
            finally:
                fcntl.flock(lockfile, fcntl.LOCK_UN)

    def update(self, change):
        with self.lock():
            st = change(self.load())
            fd, filename = tempfile.mkstemp(prefix=".state-", suffix=".tmp", dir=self.dir)
            temporary = Path(filename)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as output:
                    json.dump(st, output, indent=2, ensure_ascii=False)
                    output.flush()
                    os.fsync(output.fileno())
                temporary.replace(self.path)
            finally:
                temporary.unlink(missing_ok=True)
            return st

    def save(self, **kw) -> dict:
        return self.update(lambda st: {**st, **kw})

    @property
    def log_path(self) -> Path:
        return self.dir / "run.log"


# --------------------------------------------------------------------------- notebook
PRELUDE = r'''# [colabsvc] prelude: drive.mount() is a no-op once `colab dfs` has mounted Drive;
# google.colab.userdata.get() returns None immediately in headless CLI runs instead of timing out (~10s).
import os
try:
    from google.colab import drive as _ct_drive
    if not getattr(_ct_drive.mount, "_ct_patched", False):
        _ct_orig_mount = _ct_drive.mount
        def _ct_mount(mountpoint="/content/drive", *a, **kw):
            if os.path.isdir(os.path.join(mountpoint, "MyDrive")):
                print(f"[colabsvc] Drive already mounted at {mountpoint}; skipping drive.mount()")
                return
            return _ct_orig_mount(mountpoint, *a, **kw)
        _ct_mount._ct_patched = True
        _ct_drive.mount = _ct_mount
    from google.colab import userdata as _ct_userdata
    if not getattr(_ct_userdata.get, "_ct_patched", False):
        def _ct_userdata_get(key, *a, **kw):
            return os.environ.get(key)
        _ct_userdata_get._ct_patched = True
        _ct_userdata.get = _ct_userdata_get
except Exception as _e:
    print("[colabsvc] colab patch skipped:", _e)
for _p in ("/content/drive/MyDrive/colab-llm/hf_token.txt",
           "/content/drive/MyDrive/colab-services/hf_token.txt",
           "/content/drive/MyDrive/colab-services/qwen-flash/hf_token.txt"):
    if not os.environ.get("HF_TOKEN") and os.path.isfile(_p):
        _tok = open(_p).read().strip()
        if _tok:
            os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
            print(f"[colabsvc] loaded HF_TOKEN from {_p}")
            break
'''


def build_notebook(src: Path, skip: set[int]) -> tuple[dict, list[str]]:
    nb = json.loads(src.read_text(encoding="utf-8"))
    cells = [{"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": PRELUDE}]
    plan, n = [], 0
    for c in nb.get("cells", []):
        source = "".join(c.get("source", ""))
        if c.get("cell_type") != "code" or not source.strip():
            continue
        n += 1
        title = re.search(r"#\s*@title\s+(.+)", source)
        first = (title.group(1) if title else
                 next((l for l in source.strip().splitlines() if l.strip() and not l.startswith("#")),
                      source.strip().splitlines()[0]))[:72]
        if n in skip or SKIP_RE.search(source):
            plan.append(f"  cell {n}: SKIP  {first}")
            continue
        plan.append(f"  cell {n}: run   {first}")
        cells.append(dict(c, outputs=[], execution_count=None))
    for k, c in enumerate(cells):
        c["id"] = c.get("id") or f"ct-{k}"
    return dict(nb, cells=cells, nbformat=4, nbformat_minor=5), plan


# --------------------------------------------------------------------------- Cloudflare (cf CLI)
def find_zone(hostname: str) -> dict:
    zones = cf_json("zones", "list")
    match = [z for z in zones if hostname == z["name"] or hostname.endswith("." + z["name"])]
    if not match:
        die(f"no Cloudflare zone in this account matches {hostname}")
    return max(match, key=lambda z: len(z["name"]))


def ensure_named_tunnel(svc: Service, hostname: str) -> dict:
    """Idempotently create tunnel `colab-<svc>`, point <hostname> at it, return id + token."""
    name = f"colab-{svc.name}"
    found = [t for t in cf_json("tunnels", "list", "--name", name, "--is-deleted", "false") if t["name"] == name]
    if found:
        tunnel = found[0]
        log(f"cf: reusing tunnel {name} ({tunnel['id']})")
    else:
        tunnel = cf_json("tunnels", "create", "--name", name, "--config-source", "cloudflare")
        log(f"cf: created tunnel {name} ({tunnel['id']})")
    tid = tunnel["id"]

    ingress = [{"hostname": hostname, "service": f"http://localhost:{svc.tunnel_port}"},
               {"service": "http_status:404"}]
    # NB: the --tunnel-config-* flags of `cf` (beta) demand unrelated Access flags -> use --body
    cf_json("tunnels", "config", "update", tid, "--body", json.dumps({"config": {"ingress": ingress}}))
    log(f"cf: ingress {hostname} -> http://localhost:{svc.tunnel_port}")

    zone = find_zone(hostname)
    target = f"{tid}.cfargotunnel.com"
    body = json.dumps({"type": "CNAME", "name": hostname, "content": target, "proxied": True,
                       "comment": f"colabsvc {svc.name}"})
    records = cf_json("dns", "records", "list", "-z", zone["id"], "--name", hostname)
    if len(records)>1:
        raise ValueError(f'DNS conflict for {hostname}: multiple existing records')
    if not records:
        cf_json("dns", "records", "create", "-z", zone["id"], "--body", body)
        log(f"cf: DNS CNAME {hostname} -> {target} (created)")
    elif records[0]["type"] == "CNAME" and records[0]["content"] == target:
        log(f"cf: DNS CNAME {hostname} already points at the tunnel")
    elif records[0].get('comment')==f'colabsvc {svc.name}' or (
        records[0].get('comment')==State(svc).load().get('frp_route',{}).get('dns_owner')
        and records[0].get('comment','').startswith('colabsvc-frp ')
    ):
        cf_json("dns", "records", "edit", records[0]["id"], "-z", zone["id"], "--body", body)
        log(f"cf: DNS CNAME {hostname} -> {target} (updated from {records[0]['content']})")
    else:
        die(f"{hostname} already has a {records[0]['type']} record; refusing to overwrite it")

    prev_cf = State(svc).load().get("cloudflare_route") or {}
    old_host, old_zone = prev_cf.get("hostname"), prev_cf.get("zone_id")
    if old_host and old_zone and old_host != hostname:
        for old_rec in cf_json("dns", "records", "list", "-z", old_zone, "--name", old_host):
            if (
                old_rec.get("type") == "CNAME"
                and old_rec.get("content") == target
                and old_rec.get("comment") == f"colabsvc {svc.name}"
            ):
                run(["cf", "dns", "records", "delete", old_rec["id"], "-z", old_zone, "--force"], capture=True)
                log(f"cf: removed previous DNS CNAME {old_host}")

    token = cf_json("tunnels", "token", "get", tid)
    return {"tunnel_id": tid, "tunnel_name": name, "zone_id": zone["id"], "hostname": hostname, "token": token}


def delete_named_tunnel(st: dict) -> None:
    host, tid = st.get("hostname"), st.get("tunnel_id")
    if host and tid and st.get("zone_id"):
        for r in cf_json("dns", "records", "list", "-z", st["zone_id"], "--name", host):
            if r.get("type") == "CNAME" and r.get("content", "").startswith(tid):
                run(["cf", "dns", "records", "delete", r["id"], "-z", st["zone_id"], "--force"], capture=True)
                log(f"cf: deleted DNS record {host}")
    if tid:
        run(["cf", "tunnels", "connections", "cleanup", tid, "--force"], capture=True, check=False)
        run(["cf", "tunnels", "delete", tid, "--force"], capture=True)
        log(f"cf: deleted tunnel {st.get('tunnel_name')}")


def _write_private_notebook(nb_path: Path, built: dict, env: dict | None = None) -> None:
    notebook_copy = dict(built)
    cells = [dict(c) for c in built.get("cells", [])]
    if env and cells:
        env_inject = (
            "# [colabsvc] private runtime environment (injected locally via 0600 notebook; never exposed on ps argv)\n"
            f"import json as _ct_json, os as _ct_os\n"
            f"_ct_os.environ.update(_ct_json.loads({json.dumps(env, ensure_ascii=False)!r}))\n"
        )
        src0 = cells[0].get("source", "")
        if isinstance(src0, list):
            src0 = "".join(src0)
        if "_ct_os.environ.update(" not in src0:
            cells[0]["source"] = env_inject + src0
    notebook_copy["cells"] = cells
    fd, tmp_name = tempfile.mkstemp(prefix=".nb-", suffix=".tmp", dir=nb_path.parent)
    tmp_path = Path(tmp_name)
    try:
        os.chmod(tmp_path, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(notebook_copy, handle, indent=1, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(nb_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# --------------------------------------------------------------------------- runner (attached `colab exec`)
def finish_runner(state, pid, identity, exit_code=None):
    def finish(st):
        if st.get("runner_pid") == pid and st.get("runner_identity") == identity:
            st.update(runner_pid=None, runner_identity=None, exec_pid=None, exec_identity=None, ready=False,
                      runner_exit=exit_code)
        return st
    state.update(finish)


def exec_runner(svc: Service, nb_path: Path) -> int:
    """Own the Colab connection and always reap its separate process group."""
    state = State(svc)
    st = state.load()
    pid, identity = os.getpid(), process_identity(os.getpid())
    state.save(runner_pid=pid, runner_identity=identity)
    if nb_path.is_file() and st.get("env"):
        try:
            current_nb = json.loads(nb_path.read_text(encoding="utf-8"))
            _write_private_notebook(nb_path, current_nb, st["env"])
        except (OSError, ValueError):
            pass
    # Keep credentials out of argv (`ps aux`); secrets live in the 0600 runtime notebook prelude.
    cmd = [
        "colab", "exec", "-s", svc.name, "-f", str(nb_path),
        "--timeout", str(7 * 24 * 3600),
        "--env", f"CT_SERVICE={svc.name}",
    ]
    proc = None
    child_identity = None
    old_handler = signal.getsignal(signal.SIGTERM)
    def terminated(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminated)
    rc = None
    selector = selectors.DefaultSelector()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        child_identity = process_identity(proc.pid)
        state.save(exec_pid=proc.pid, exec_identity=child_identity)
        protocol = NotebookProtocol()
        selector.register(proc.stdout, selectors.EVENT_READ)
        deadline = time.monotonic() + st.get('ready_timeout', svc.ready_timeout) if st.get('runner_mode') == 'foreground' else None
        while True:
            remaining = max(0, deadline - time.monotonic()) if deadline is not None and not st.get('ready') else None
            if remaining == 0:
                raise TimeoutError('Timed out waiting for CT_READY; the foreground connection was stopped. Use status to check the remote session')
            if not selector.select(remaining):
                continue
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            update = {key: value for key, value in protocol.feed(chunk).items() if st.get(key) != value}
            if update:
                st = state.save(**update)
        update = protocol.feed(b"", final=True)
        if update:
            st = state.save(**update)
        rc = proc.wait()
        if rc == 0 and not st.get("ready"):
            warn("Colab exited before CT_READY; check the log for startup errors")
            rc = 1
        return rc if rc >= 0 else 128 - rc
    finally:
        signal.signal(signal.SIGTERM, old_handler)
        selector.close()
        if proc is not None:
            stop_owned_process(proc.pid, child_identity, group=True)
            proc.wait(timeout=5)
            if proc.stdout:
                proc.stdout.close()
        finish_runner(state, pid, identity, rc)


# --------------------------------------------------------------------------- commands
def ensure_session(svc: Service) -> None:
    try:
        sessions = json.loads((Path.home() / ".config/colab-cli/sessions.json").read_text())
    except Exception:
        sessions = {}
    if svc.name in sessions:
        log(f"reusing Colab session '{svc.name}'")
        return
    args = ["colab", "new", "-s", svc.name]
    if svc.tpu:
        args += ["--tpu", svc.tpu]
    elif svc.gpu:
        args += ["--gpu", svc.gpu]
    if svc.high_mem:
        args += ["--high-mem"]
    log(" ".join(args))
    run(args)


def cmd_up(svc: Service, script: Path, a) -> None:
    state = State(svc)
    if a.dry_run:
        return _start_up(svc, script, a)
    with state.lock("operation", blocking=False):
        launched = _start_up(svc, script, a)
    if a.foreground:
        try:
            raise SystemExit(exec_runner(svc, launched))
        finally:
            if state.load().get("exec_pid") is None:
                finish_runner(state, os.getpid(), process_identity(os.getpid()))
    if getattr(a, "detach", False):
        show_info(svc, reveal=False)
        return
    _wait_ready(svc, launched, timeout=getattr(a, "ready_timeout", None))


def _start_up(svc: Service, script: Path, a):
    hostname = tunnel_hostname(svc, a)
    transport=tunnel_transport(a)
    frp_config=None
    if transport=='frp':
        frp_config=load_tunnel_config().get('frp')
        if not frp_config or not frp_config.get('server_ip') or not hostname:
            raise ValueError('FRP requires server configuration and a public domain; run colabllm frp-config')
        from frp_backend import validate_config
        frp_config=validate_config({**frp_config,'enabled':True})
    if a.gpu:
        svc.gpu = None if a.gpu.lower() in ("none", "cpu") else a.gpu
    for tool in ("colab",) + (("cf",) if hostname else ()) + (("ssh",) if transport=='frp' else ()):
        if not shutil.which(tool):
            die(f"`{tool}` CLI not found")
    state = State(svc)
    st = state.load()
    busy = {8080: "Colab runtime server", 8012: "colab dfs creds shim"}
    for port in {svc.service_port, svc.tunnel_port} & busy.keys():
        die(f"port {port} is used on Colab VMs by the {busy[port]}; pick another port in services/{svc.name}.py")
    if runner_alive(st):
        die(f"'{svc.name}' is already running (runner pid {st['runner_pid']}); use `down` first")

    nb_src = (script.parent / svc.notebook).resolve()
    skip = {int(x) for x in a.skip.split(",")} if a.skip else set()
    built, plan = build_notebook(nb_src, skip)
    log(f"service {svc.name}: notebook {nb_src.name}, gpu {svc.tpu or svc.gpu or 'cpu'}, auth {svc.auth}, "
        f"transport {transport}, {'domain ' + hostname if hostname else 'quick tunnel'}")
    print("\n".join(plan))
    if a.dry_run:
        return

    private_dir(STATE_ROOT)
    private_dir(state.dir.parent)
    private_dir(state.dir)
    nb_path = state.dir / nb_src.name

    # credentials: stable across restarts, regenerated with --new-credentials
    if a.new_credentials or not st.get("auth_pass"):
        st = state.save(auth_user=svc.auth_user, auth_pass=secrets.token_urlsafe(18),
                        api_token="sk-" + secrets.token_urlsafe(32))
    env = {"CT_SERVICE": svc.name, "CT_SERVICE_PORT": str(svc.service_port),
           "CT_PUBLIC_PORT": str(svc.tunnel_port), "CT_AUTH": svc.auth,
           "CT_API_TOKEN": st["api_token"],
           "CT_MOUNT_DRIVE": "0" if a.no_drive else "1",
           "CT_DATA_DIR": (
               f"/content/local-data/{svc.base_name or svc.name}"
               if a.no_drive else f"/content/drive/MyDrive/colab-llm/{svc.base_name or svc.name}"
           ),
           **svc.env}
    # Explicitly clear previous notebook-kernel values when switching to quick
    # tunnels; reused Colab kernels retain env vars from earlier executions.
    env.update(CT_TRANSPORT=transport,CT_TUNNEL_TOKEN="", CT_PUBLIC_HOSTNAME="")
    for key in ('SERVER_IP','SERVER_PORT','REMOTE_PORT','PROXY_NAME','TOKEN','CA_CERT','VERSION','SERVER_NAME','AUTH_SCOPES'):
        env['CT_FRP_'+key]=''
    host_hf_tok = None
    for p in (Path.home() / ".cache/huggingface/token", Path.home() / ".huggingface/token"):
        if p.is_file() and p.read_text().strip():
            host_hf_tok = p.read_text().strip()
            break
    hf_tok = (
        a.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or host_hf_tok
        or (None if a.new_credentials else st.get("hf_token"))
    )
    if a.hf_token:
        st = state.save(hf_token=a.hf_token)
    elif a.new_credentials and "hf_token" in st:
        st = state.update(lambda current: {k: v for k, v in current.items() if k != "hf_token"})
    if hf_tok:
        env.update(HF_TOKEN=hf_tok, HUGGING_FACE_HUB_TOKEN=hf_tok)
    if svc.auth == "caddy":
        env.update(CT_AUTH_USER=st["auth_user"], CT_AUTH_PASS=st["auth_pass"])

    # 1. named tunnel + DNS (local, cf CLI)
    if transport=='frp':
        from frp_backend import ensure_route
        if st.get('frp_route'):
            previous=st['frp_route']
            if previous['hostname']!=hostname or any(previous['frp'].get(k)!=frp_config.get(k)
                for k in ('server_ip','server_port','ssh','project_id')):
                raise ValueError('Existing FRP route targets another endpoint; use down --delete-tunnel first')
        route,frp_env=ensure_route(svc,frp_config,hostname,STATE_ROOT)
        env.update(frp_env)
        st=state.save(transport=transport,hostname=hostname,frp_route=route)
    elif hostname:
        info = ensure_named_tunnel(svc, hostname)
        env.update(CT_TUNNEL_TOKEN=info["token"], CT_PUBLIC_HOSTNAME=hostname)
        route={k: v for k, v in info.items() if k != 'token'}
        st = state.save(transport=transport,cloudflare_route=route,**route)
    else:
        st=state.save(transport=transport)
    _write_private_notebook(nb_path, built, env)
    # 2. session + 3. Drive
    ensure_session(svc)
    if not a.no_drive:
        log("colab dfs mount")
        run(["colab", "dfs", "mount", "-s", svc.name])
    st = state.save(
        base_service=svc.base_name or svc.name,
        env=env, url=None, ready=False, last_cell_started=False, login_user=None, login_pass=None,
        runner_exit=None, ready_timeout=getattr(a, "ready_timeout", None) or svc.ready_timeout,
        started=time.strftime("%Y-%m-%d %H:%M:%S"), gpu=svc.tpu or svc.gpu or "none",
    )

    # 4. notebook through an attached `colab exec` (keep-alive)
    if a.foreground:
        state.save(runner_pid=os.getpid(), runner_identity=process_identity(os.getpid()), runner_mode="foreground")
        return nb_path
    state.log_path.write_bytes(b"")
    state.log_path.chmod(0o600)
    exec_cmd = [sys.executable, str(script), *(["-i", svc.instance] if svc.instance else []), "_exec", str(nb_path)]
    with state.log_path.open("ab") as logfile:
        proc = subprocess.Popen(exec_cmd, stdin=subprocess.DEVNULL, stdout=logfile,
                                stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(proc.pid)
    if not identity:
        raise RuntimeError(f"The background process exited before it was registered; see {state.log_path}")
    state.save(runner_pid=proc.pid, runner_identity=identity, runner_mode="background")
    return proc


def _wait_ready(svc, proc, timeout=None):
    state = State(svc)
    log(f"runner started (pid {proc.pid}) -- it keeps `colab exec` attached; log: {state.log_path}")
    log("streaming progress until the service reports CT_READY (Ctrl-C only stops watching)")
    offset, deadline = 0, time.monotonic() + (svc.ready_timeout if timeout is None else timeout)

    def _flush_log():
        nonlocal offset
        if not state.log_path.is_file():
            return
        with open(state.log_path, "rb") as f:
            f.seek(offset)
            data = f.read()
        offset += len(data)
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.flush()

    try:
        while time.monotonic() < deadline:
            _flush_log()
            st = state.load()
            if st.get("ready") and st.get("url"):
                break
            if proc.poll() is not None:
                _flush_log()
                st = state.load()
                if st.get("ready") and st.get("url"):
                    break
                die(f"runner exited early (code {proc.returncode}); see {state.log_path}")
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        else:
            raise TimeoutError("Timed out waiting for CT_READY; the background process keeps running. Check with status or logs -f")
    except KeyboardInterrupt:
        raise
    show_info(svc, do_probe=True, reveal=False)


def probe(url: str, auth: tuple[str, str] | None = None, bearer: str | None = None) -> str:
    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    if auth:
        import base64
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode())
    elif bearer:
        req.add_header("Authorization", f"Bearer {bearer}")
    try:
        with safe_urlopen(req, timeout=15) as r:
            return str(r.status)
    except urllib.error.HTTPError as e:
        return str(e.code)
    except Exception as e:
        return f"error ({type(e).__name__})"


def show_info(svc: Service, do_probe=False, reveal=False, output_json=False):
    st = State(svc).load()
    alive = runner_alive(st)
    ready = bool(alive and st.get("ready"))
    data = dict(service=svc.name, running=alive, ready=ready, url=st.get("url"))
    lines = [f"service : {svc.name}   (runner {'alive' if alive else 'stopped'})",
             f"ready   : {ready}"]
    if st.get("url"):
        lines.append(f"URL     : {st['url']}")
    if reveal:
        for key in ("auth_user", "auth_pass", "api_token", "login_user", "login_pass"):
            if st.get(key):
                data[key] = st[key]
                lines.append(f"{key}: {st[key]}")
    healthy = ready
    if do_probe and st.get("url"):
        url = st["url"]
        if svc.auth == "caddy":
            result = probe(url, (st.get("auth_user", ""), st.get("auth_pass", "")))
        elif svc.auth == "builtin" and st.get("api_token") and not st.get("login_user"):
            result = probe(url.rstrip('/') + '/v1/models', bearer=st["api_token"])
        else:
            result = probe(url)
        data["http_status"] = result
        lines.append(f"probe   : {result}")
        healthy = bool(ready and result.isdigit() and 200 <= int(result) < 400)
    elif do_probe:
        healthy = False
    if output_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        banner(lines)
    return healthy


def show_instances(svc: Service, reveal=False, output_json=False):
    items = list_service_instances(svc.base_name or svc.name)
    if not reveal:
        items = [{key: value for key, value in item.items() if key not in ("api_token", "auth_pass", "auth_user")}
                 for item in items]
    if output_json:
        print(json.dumps(items, ensure_ascii=False, indent=2))
        return
    if not items:
        log(f"no instances found for '{svc.base_name or svc.name}'")
        return
    print(f"{'INSTANCE':<12} {'SESSION':<28} {'STATE':<12} URL" + (" / API TOKEN" if reveal else ""))
    for item in items:
        label = "READY" if item["ready"] else ("STARTING" if item["alive"] else "STOPPED")
        suffix = f" / {item.get('api_token', '-')}" if reveal else ""
        print(f"{item['instance']:<12} {item['name']:<28} {label:<12} {item.get('url') or '-'}{suffix}")


def cmd_down(svc: Service, a) -> None:
    if getattr(a, "all", False):
        items = list_service_instances(svc.base_name or svc.name)
        errors: list[tuple[str, Exception]] = []
        for it in items:
            sub_svc = dataclasses.replace(svc)
            apply_instance(sub_svc, it["instance_arg"] or None)
            sub_a = argparse.Namespace(delete_tunnel=a.delete_tunnel, all=False)
            try:
                cmd_down(sub_svc, sub_a)
            except (CliError, RuntimeError, OSError) as err:
                errors.append((sub_svc.name, err))
        if errors:
            summary = "; ".join(f"{name}: {err}" for name, err in errors)
            code = errors[0][1].code if isinstance(errors[0][1], CliError) else 1
            raise CliError(f"Some instances failed to stop: {summary}", code)
        return
    state = State(svc)
    with state.lock("operation", blocking=False):
        st = state.load()
        log(f"colab stop -s {svc.name}")
        result = run(["colab", "stop", "-s", svc.name], check=False)
        if result.returncode != 0:
            raise CliError(f"Failed to stop {svc.name}; the runtime state was kept, so down can be retried", result.returncode if result.returncode > 0 else 1)
        stop_owned_process(st.get("exec_pid"), st.get("exec_identity"), group=True)
        stop_owned_process(st.get("runner_pid"), st.get("runner_identity"), group=st.get("runner_mode") == "background")
        if a.delete_tunnel and st.get('frp_route'):
            from frp_backend import remove_route
            remove_route(st['frp_route'],svc.name)
            st.pop('frp_route',None)
            if not (st.get("tunnel_id") or st.get("cloudflare_route")):
                st.pop("hostname", None)
                st.pop("transport", None)
        if a.delete_tunnel and (st.get("tunnel_id") or st.get('cloudflare_route')):
            delete_named_tunnel(st.get('cloudflare_route') or st)
            st.pop('cloudflare_route',None)
            drop = {"tunnel_id", "tunnel_name", "zone_id"}
            if not st.get("frp_route"):
                drop.update({"hostname", "transport"})
            st = {k: v for k, v in st.items() if k not in drop}
        keys = ("base_service", "auth_user", "auth_pass", "api_token", "hf_token", "tunnel_id", "tunnel_name",
                "zone_id", "hostname", "transport", "frp_route", "cloudflare_route")
        state.update(lambda current: {k: st[k] for k in keys if k in st})


def _owns_tunnel(st: dict) -> bool:
    return bool(st.get("tunnel_id") or st.get("cloudflare_route") or st.get("frp_route"))


def cmd_purge(svc: Service, a) -> None:
    """Delete a stopped instance's local state; never leaves an untracked Colab session or tunnel behind silently."""
    state = State(svc)
    if not state.dir.is_dir():
        raise CliError(f"No local state for '{svc.name}'")
    with state.lock("operation", blocking=False):
        st = state.load()
        if runner_alive(st) or owned_process(st.get("exec_pid"), st.get("exec_identity")):
            raise CliError(f"'{svc.name}' is still running; stop it with `down` first")
        if _owns_tunnel(st) and not (a.delete_tunnel or a.keep_tunnel):
            raise UsageError(f"'{svc.name}' owns the named tunnel {st.get('hostname') or ''}".rstrip()
                             + "; pass --delete-tunnel to remove it or --keep-tunnel to leave it in place")
        # `colab stop` succeeds for unknown sessions, so a non-zero exit means we could not confirm the VM is released.
        log(f"colab stop -s {svc.name}")
        result = run(["colab", "stop", "-s", svc.name], check=False)
        if result.returncode != 0:
            raise CliError(f"Could not confirm that the Colab session '{svc.name}' is stopped; local state was kept so purge can be retried",
                           result.returncode if result.returncode > 0 else 1)
        if a.delete_tunnel and st.get("frp_route"):
            from frp_backend import remove_route
            remove_route(st["frp_route"], svc.name)
        if a.delete_tunnel and (st.get("tunnel_id") or st.get("cloudflare_route")):
            delete_named_tunnel(st.get("cloudflare_route") or st)
        # Removed while the operation lock is held so no concurrent `up` can recreate state in between.
        shutil.rmtree(state.dir)
    log(f"deleted local state for {svc.name}")


def build_parser(svc: Service, configure=None):
    p = argparse.ArgumentParser(prog=f"colabllm {svc.base_name}", description=(sys.modules["__main__"].__doc__ or "").strip(),
                                formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("-i", "--instance", metavar="ID", type=validate_instance,
                   help="instance ID to create/manage multiple parallel instances (e.g. -i 1, -i 2)")
    sub = p.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("up", help="session -> Drive -> (cf tunnel) -> run notebook, keep it alive")
    tunnel = u.add_mutually_exclusive_group()
    tunnel.add_argument("--domain", nargs="?", const=True, metavar="HOST", type=validate_domain,
                        help="full fixed hostname; without HOST the configured base domain is used, or a dynamic URL if none is configured")
    tunnel.add_argument("--base-domain", metavar="DOMAIN", type=validate_domain,
                        help="base domain for this start, e.g. example.com; <service>.example.com is generated automatically")
    tunnel.add_argument("--quick-tunnel", action="store_true", help="ignore the base domain configuration and force a dynamic trycloudflare.com URL")
    u.add_argument('--transport',choices=('cloudflare','frp'),help='override the global tunnel backend for this start')
    u.add_argument("--gpu", help="override GPU (T4|L4|G4|H100|A100|none)")
    u.add_argument("--skip", default="", help="1-based code cells to skip, e.g. 1,2")
    u.add_argument("--no-drive", action="store_true", help="do not mount Google Drive")
    u.add_argument("--hf-token", metavar="TOKEN",
                   help="Hugging Face token (saved in ~/.colabllm and forwarded as HF_TOKEN; or set env HF_TOKEN)")
    u.add_argument("--new-credentials", action="store_true", help="rotate the Caddy password / API token")
    mode = u.add_mutually_exclusive_group()
    mode.add_argument("--foreground", action="store_true", help="stay attached in this terminal instead")
    mode.add_argument("--detach", action="store_true", help="return after starting the runner without waiting for readiness")
    u.add_argument("--ready-timeout", type=positive_seconds, help="seconds to wait for readiness (default: service-specific)")
    u.add_argument("--dry-run", action="store_true", help="only show the cell plan")
    d = sub.add_parser("down", help="stop runner + Colab session")
    d.add_argument("--delete-tunnel", action="store_true", help="also delete the named tunnel and its DNS record")
    d.add_argument("--all", action="store_true", help="stop all instances of this service")
    purge = sub.add_parser("purge", help="delete a stopped instance's local state (credentials, logs, notebook copy); Drive data is kept")
    purge_tunnel = purge.add_mutually_exclusive_group()
    purge_tunnel.add_argument("--delete-tunnel", action="store_true", help="also delete the named tunnel / FRP route owned by the instance")
    purge_tunnel.add_argument("--keep-tunnel", action="store_true", help="leave the named tunnel / FRP route in place")
    for name, help_text in (("info", "URL and state"), ("ps", "list instances"), ("list", "alias for ps"), ("status", "Colab status and authenticated health probe")):
        output = sub.add_parser(name, help=help_text)
        output.add_argument("--json", action="store_true", help="output machine-readable JSON")
        output.add_argument("--show-secrets", action="store_true", help="include credentials in output")
    lg = sub.add_parser("logs", help="notebook output")
    lg.add_argument("-f", "--follow", action="store_true")
    lg.add_argument("-n", "--lines", type=int, default=80)
    tc = sub.add_parser("tunnel-config", help="show or set the Cloudflare base domain shared by all services; dynamic URLs are used when unset")
    settings = tc.add_mutually_exclusive_group()
    settings.add_argument("--base-domain", metavar="DOMAIN", type=validate_domain)
    settings.add_argument("--clear", action="store_true", help="remove the persistent base domain and go back to the dynamic URL default")
    p.add_argument("--command-timeout", type=positive_seconds, help="external command timeout in seconds (default: 300, or COLABLLM_COMMAND_TIMEOUT)")
    p.add_argument("--debug", action="store_true", help="show exception tracebacks")
    p.add_argument("--quiet", action="store_true", help="suppress diagnostic progress")
    p.add_argument("--no-color", action="store_true", help="disable diagnostic colors")
    if configure:
        configure(p, sub, u)
    public = list(sub.choices)
    sub.metavar = "{" + ",".join(public) + "}"
    ex = sub.add_parser("_exec")
    ex.add_argument("nb")
    return p


def main(svc: Service, *, configure=None, handlers=None, prepare_up=None) -> None:
    script = Path(sys.argv[0]).resolve()
    inst, cleaned_argv = extract_instance_arg(sys.argv[1:])
    apply_instance(svc, inst)
    p = build_parser(svc, configure)
    a = p.parse_args(cleaned_argv)
    if getattr(a, "instance", None):
        apply_instance(svc, a.instance)
    if handlers and a.cmd in handlers:
        code = handlers[a.cmd](a)
        if isinstance(code, int) and code:
            raise SystemExit(code)
        return

    if a.cmd == "up":
        try:
            if prepare_up:
                prepare_up(a)
            cmd_up(svc, script, a)
        except ValueError as error:
            p.error(str(error))
    elif a.cmd == "tunnel-config":
        try:
            cmd_tunnel_config(a)
        except (ValueError, OSError) as error:
            p.error(str(error))
    elif a.cmd == "_exec":
        sys.exit(exec_runner(svc, Path(a.nb)))
    elif a.cmd == "down":
        cmd_down(svc, a)
    elif a.cmd == "purge":
        cmd_purge(svc, a)
    elif a.cmd == "info":
        show_info(svc, reveal=a.show_secrets, output_json=a.json)
    elif a.cmd in ("ps", "list"):
        show_instances(svc, reveal=a.show_secrets, output_json=a.json)
    elif a.cmd == "status":
        result = run(["colab", "status", "-s", svc.name], capture=True, check=False)
        if result.returncode != 0:
            raise CliError((result.stderr or "Colab status query failed").strip(), result.returncode if result.returncode > 0 else 1)
        if not show_info(svc, do_probe=True, reveal=a.show_secrets, output_json=a.json):
            raise SystemExit(1)
    elif a.cmd == "logs":
        lp = State(svc).log_path
        if not lp.exists():
            die(f"no log at {lp}")
        os.execvp("tail", ["tail", "-n", str(a.lines), *(["-f"] if a.follow else []), str(lp)])
