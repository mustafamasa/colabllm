#!/usr/bin/env python3
"""Generate service notebooks from the standard cells.

Every service notebook is self-contained (runs from the service script *or* pasted into the
Colab UI) and follows the same cell order:

  0. config        -- read CT_* env injected by `colab exec --env`, helpers
  1. setup         -- idempotent install                         (service specific)
  2. service       -- def start_service() -> Popen on 127.0.0.1  (service specific)
  3. auth          -- Caddy basic auth + Bearer token when CT_AUTH == "caddy"
  4. tunnel        -- cloudflared: token -> fixed domain, else quick tunnel; prints CT_PUBLIC_URL
  5. keep-alive    -- prints CT_READY, watchdog restarts dead processes, never returns

Usage:  python3 templates/build_notebooks.py      (rewrites templates/ and src/colabllm/services/*.ipynb)
Edit the generated notebooks directly if a service needs something special; re-running this
script overwrites them.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONFIG = r'''# @title 0. Config -- values come from `colab exec --env` (defaults allow running in the Colab UI)
import json, os, re, secrets, shutil, stat, subprocess, sys, time, urllib.request, urllib.error

SERVICE      = os.environ.get("CT_SERVICE", "%(name)s")
SERVICE_PORT = int(os.environ.get("CT_SERVICE_PORT", "%(service_port)d"))
AUTH         = os.environ.get("CT_AUTH", "%(auth)s")            # caddy | builtin | none
PUBLIC_PORT  = int(os.environ.get("CT_PUBLIC_PORT", "%(public_port)d"))  # what cloudflared targets
AUTH_USER    = os.environ.get("CT_AUTH_USER", "admin")
AUTH_PASS    = os.environ.get("CT_AUTH_PASS") or secrets.token_urlsafe(18)
API_TOKEN    = os.environ.get("CT_API_TOKEN") or "sk-" + secrets.token_urlsafe(32)
TUNNEL_TOKEN = os.environ.get("CT_TUNNEL_TOKEN", "")            # set -> fixed domain (named tunnel)
PUBLIC_HOST  = os.environ.get("CT_PUBLIC_HOSTNAME", "")
CREDS_FROM_ENV = bool(os.environ.get("CT_AUTH_PASS"))

WORK = "/content/ct"                                            # logs + binaries of this notebook
os.makedirs(WORK, exist_ok=True)
for _p in ("/content/drive/MyDrive/colab-llm/hf_token.txt",
           "/content/drive/MyDrive/colab-services/hf_token.txt",
           "/content/drive/MyDrive/colab-services/qwen-flash/hf_token.txt"):
    if not os.environ.get("HF_TOKEN") and os.path.isfile(_p):
        _tok = open(_p).read().strip()
        if _tok:
            os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
            break
MOUNT_DRIVE = os.environ.get("CT_MOUNT_DRIVE", "1") != "0"
DATA_DIR = os.environ.get("CT_DATA_DIR", f"/content/drive/MyDrive/colab-llm/{SERVICE}")
if not MOUNT_DRIVE or not os.path.isdir("/content/drive/MyDrive"):   # Drive not mounted -> local fallback
    if DATA_DIR.startswith("/content/drive/"):
        DATA_DIR = f"/content/data/{SERVICE}"
os.makedirs(DATA_DIR, exist_ok=True)
PROCS = {}                                                      # name -> (Popen, restart function)

def ct_spawn(name, args, env=None, cwd=None):
    """Start a background process; output goes to /content/ct/<name>.log."""
    with open(f"{WORK}/{name}.log", "a") as logf:   # child keeps its own dup of the fd
        return subprocess.Popen(args, stdout=logf, stderr=subprocess.STDOUT, cwd=cwd,
                                env={**os.environ, **(env or {})}, start_new_session=True)

def ct_stop_proc(proc, timeout=15):
    """Terminate a ct_spawn process group, escalating to SIGKILL after a bounded wait."""
    import signal
    if proc is None or proc.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.send_signal(sig)
            except ProcessLookupError:
                return
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue

def ct_gpu_stat(query="memory.used,memory.total,utilization.gpu"):
    """nvidia-smi summary that never raises (CPU runtimes / transient driver errors)."""
    try:
        return subprocess.check_output(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
                                       text=True, stderr=subprocess.DEVNULL, timeout=20).strip()
    except (OSError, subprocess.SubprocessError):
        return "cpu/no-gpu"

def ct_log_size(name):
    p = f"{WORK}/{name}.log"
    return os.path.getsize(p) if os.path.exists(p) else 0

def ct_log_since(name, offset):
    p = f"{WORK}/{name}.log"
    if not os.path.exists(p):
        return ""
    with open(p, "rb") as handle:
        handle.seek(min(offset, os.path.getsize(p)))
        return handle.read().decode("utf-8", errors="replace")

def ct_public_gate(public_url, probe, verify_auth, log, timeout=120, interval=8):
    """Wait for the public endpoint; auth failures are fatal, network errors are retried."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            code = probe()
        except Exception as err:   # DNS / edge propagation / TLS warm-up
            log(f"Public probe pending ({type(err).__name__}: {err})")
            time.sleep(interval)
            continue
        if code == 200:
            verify_auth()            # RuntimeError propagates: never report CT_READY with broken auth
            return
        log(f"Public probe pending (HTTP {code})")
        time.sleep(interval)
    raise RuntimeError(f"Public endpoint {public_url} failed readiness verification within {timeout}s")

def ct_watchdog(health, start_service, log, *, service_key="service", log_name="service",
                interval=60, grace=900, dead_markers=(), heartbeat=None):
    """Supervise PROCS forever: restart dead processes, debounce health failures after reloads."""
    started = time.monotonic()
    failures = 0
    grace_until = time.monotonic()
    log_offset = ct_log_size(log_name)

    def restart_service(reason):
        nonlocal failures, grace_until, log_offset
        log(f"[watchdog] {reason}; restarting {service_key}")
        ct_stop_proc(PROCS[service_key][0])
        PROCS[service_key] = (start_service(), start_service)
        failures = 0
        grace_until = time.monotonic() + grace
        log_offset = ct_log_size(log_name)

    while True:
        time.sleep(interval)
        for name, (proc, restart) in list(PROCS.items()):
            if proc.poll() is not None:
                log(f"[watchdog] {name} exited with {proc.returncode}; restarting\n{ct_tail(name, 20)}")
                PROCS[name] = (restart(), restart)
                if name == service_key:
                    failures = 0
                    grace_until = time.monotonic() + grace
                    log_offset = ct_log_size(log_name)
        if dead_markers:
            fresh = ct_log_since(log_name, log_offset)
            log_offset += len(fresh.encode("utf-8", errors="replace"))
            if any(marker in fresh for marker in dead_markers):
                restart_service("Detected dead worker marker in service log")
                continue
        try:
            healthy = bool(health())
        except Exception:
            healthy = False
        if healthy:
            failures = 0
        elif time.monotonic() >= grace_until:
            failures += 1
        if failures >= 6:
            restart_service("6 consecutive failed health checks")
        if heartbeat is not None:
            try:
                heartbeat(int(time.monotonic() - started))
            except Exception as err:
                log(f"[watchdog] heartbeat warning: {err}")

def ct_wait_http(url, timeout=600, ok=lambda code: code < 500):
    """Wait until url answers with a non-5xx status (401/404 count as 'listening')."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if ok(r.status): return r.status
        except urllib.error.HTTPError as e:
            if ok(e.code): return e.code
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"{url} not ready after {timeout}s")

def ct_tail(name, n=30):
    p = f"{WORK}/{name}.log"
    return "".join(open(p, errors="replace").readlines()[-n:]) if os.path.exists(p) else ""

def ct_fetch_binary(path, url):
    if not os.path.exists(path):
        partial = f"{path}.{os.getpid()}.partial"
        with urllib.request.urlopen(url, timeout=60) as src, open(partial, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.chmod(partial, os.stat(partial).st_mode | stat.S_IEXEC)
        os.replace(partial, path)
    return path

print(f"service={SERVICE} port={SERVICE_PORT} auth={AUTH} public_port={PUBLIC_PORT} "
      f"tunnel={'named ' + PUBLIC_HOST if TUNNEL_TOKEN else 'quick'} data={DATA_DIR}")
'''

CADDY_TEMPLATE = r'''# @title 3. Auth layer -- Caddy: basic auth for browsers, `Authorization: Bearer <token>` for API clients
# PUBLIC_PORT must be free on the VM: Colab itself uses 8080 (runtime server) and 8012 (colab dfs shim).
# Only used when the service has no auth of its own (CT_AUTH=caddy). The service itself listens on
# 127.0.0.1 only, cloudflared targets Caddy, so every public request passes through auth.
REWRITE_HOST = __REWRITE_HOST__   # True: send Host: 127.0.0.1:<port> upstream (Ollama & co. reject foreign Host headers)
import platform
CADDY_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), "amd64")

def start_caddy():
    if not os.path.exists("/usr/local/bin/caddy"):
        tag = json.loads(urllib.request.urlopen(
            "https://api.github.com/repos/caddyserver/caddy/releases/latest", timeout=30).read())["tag_name"]
        tgz = "/tmp/caddy.tgz"
        with urllib.request.urlopen(f"https://github.com/caddyserver/caddy/releases/download/{tag}/"
                                    f"caddy_{tag.lstrip('v')}_linux_{CADDY_ARCH}.tar.gz", timeout=60) as src, \
                open(tgz, "wb") as dst:
            shutil.copyfileobj(src, dst)
        subprocess.run(["tar", "-xzf", tgz, "-C", "/usr/local/bin", "caddy"], check=True)
    import hashlib
    cookie_tok = hashlib.sha256(f"ct-cookie-v2:{AUTH_USER}:{AUTH_PASS}:{API_TOKEN}".encode()).hexdigest()
    hashed = subprocess.check_output(["caddy", "hash-password", "--plaintext", AUTH_PASS], text=True).strip()
    upstream = f"""reverse_proxy 127.0.0.1:{SERVICE_PORT} {{
            {"header_up Host {upstream_hostport}" if REWRITE_HOST else ""}
            {"header_up Origin http://{upstream_hostport}" if REWRITE_HOST else ""}
            header_up -Authorization
__EXTRA_PROXY_LINES__            flush_interval -1
        }}"""
    caddyfile = f"""{{
    admin off
    auto_https off
}}
:{PUBLIC_PORT} {{
    encode zstd gzip
    @bearer header Authorization "Bearer {API_TOKEN}"
    handle @bearer {{
        {upstream}
    }}
    @cookie header Cookie *ct_session={cookie_tok}*
    handle @cookie {{
        {upstream}
    }}
    handle {{
        route {{
            basic_auth {{
                "{AUTH_USER}" {hashed}
            }}
            header +Set-Cookie "ct_session={cookie_tok}; Path=/; HttpOnly; SameSite=Lax"
            {upstream}
        }}
    }}
}}
"""
    open(f"{WORK}/Caddyfile", "w").write(caddyfile)
    subprocess.run(["pkill", "-f", "caddy run"], check=False)
    return ct_spawn("caddy", ["caddy", "run", "--config", f"{WORK}/Caddyfile", "--adapter", "caddyfile"])

AUTH_OK = False   # cell 4 refuses to open the tunnel unless this becomes True (fail closed)
if AUTH == "caddy":
    PROCS["caddy"] = (start_caddy(), start_caddy)
    time.sleep(2)
    if PROCS["caddy"][0].poll() is not None:   # e.g. port taken (Colab uses 8080 itself)
        raise RuntimeError(f"caddy exited -- is :{PUBLIC_PORT} in use?\n" + ct_tail("caddy"))
    # Must be 401 from *our* Caddy and must NOT leak Set-Cookie on unauthenticated 401 responses.
    ct_wait_http(f"http://127.0.0.1:{PUBLIC_PORT}/", timeout=60, ok=lambda c: c == 401)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PUBLIC_PORT}/", timeout=5)
    except urllib.error.HTTPError as _e:
        if _e.headers.get("Set-Cookie"):
            raise RuntimeError("Caddy leaked Set-Cookie on unauthenticated 401 response!")
    AUTH_OK = True
    print(f"caddy on :{PUBLIC_PORT} -> 127.0.0.1:{SERVICE_PORT} (unauthenticated request -> 401)")
else:
    AUTH_OK = True
    print(f"auth={AUTH}: no proxy, cloudflared targets the service on :{PUBLIC_PORT} directly")
'''



def render_caddy_cell(rewrite_host=True, extra_proxy_lines=""):
    """Render the Caddy auth cell; extra_proxy_lines are inserted inside reverse_proxy {...}."""
    extra = "".join(f"            {line}\n" for line in extra_proxy_lines.splitlines() if line.strip())
    return CADDY_TEMPLATE.replace("__REWRITE_HOST__", "True" if rewrite_host else "False").replace(
        "__EXTRA_PROXY_LINES__", extra)


CADDY = render_caddy_cell()

TUNNEL = ("# @title 4. Tunnel -- Cloudflare by default, or project-configured FRP + HTTPS\n"
          + (ROOT / "templates/tunnel_client.py").read_text()
          + "\nrun_tunnel(globals())\n")

KEEPALIVE = r'''# @title 5. Ready + keep-alive -- KEEP THIS CELL RUNNING
# The Colab session only lives while the `colab exec` connection is open and this cell blocks.
# The watchdog restarts any process from PROCS that died; a heartbeat line is printed every 10 min.
ct_wait_http(f"http://127.0.0.1:{SERVICE_PORT}%(health_path)s", timeout=%(ready_timeout)d)
if AUTH == "caddy" and not CREDS_FROM_ENV:   # started from the Colab UI: show generated credentials
    print(f"CT_CREDENTIALS user={AUTH_USER} pass={AUTH_PASS}")
    print(f"API token: {API_TOKEN}")
__BEFORE_READY__
print(f"{SERVICE} is up: {PUBLIC_URL}")
print("CT_READY", flush=True)

last_beat = time.time()
while True:
    time.sleep(30)
    for name, (proc, restart) in list(PROCS.items()):
        if proc.poll() is not None:
            print(f"[watchdog] {name} exited with {proc.returncode}; restarting\n{ct_tail(name, 15)}", flush=True)
            ct_stop_proc(proc)
            PROCS[name] = (restart(), restart)
    if time.time() - last_beat > 600:
        last_beat = time.time()
        print(f"[heartbeat] {time.strftime('%%H:%%M:%%S')} " +
              " ".join(f"{n}={'up' if p.poll() is None else 'down'}" for n, (p, _) in PROCS.items()), flush=True)
'''


def render_keepalive(cfg, before_ready=""):
    """Standard keep-alive cell; before_ready runs after health succeeds and before CT_READY."""
    return (KEEPALIVE % cfg).replace("__BEFORE_READY__\n", before_ready.rstrip("\n") + "\n" if before_ready else "")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": text}


def notebook(title, intro, cfg, setup, service, keepalive=None, extra_after_tunnel=None, auth_cell=None):
    cells = [md(f"# {title}\n\n{intro}\n\n"
                "Run it with the service script (`colabllm <name> up`), which creates the Colab session, "
                "mounts Drive with `colab dfs`, optionally creates a named Cloudflare tunnel, and injects "
                "`CT_*` settings. Cell order is the standard contract -- see "
                "`.agents/skills/colab-llm-service/SKILL.md`."),
             code(CONFIG % cfg), code(setup), code(service), code(auth_cell if auth_cell is not None else CADDY), code(TUNNEL)]
    if extra_after_tunnel:
        cells.append(code(extra_after_tunnel))
    cells.append(code(keepalive if keepalive is not None else render_keepalive(cfg)))
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i}"
    return {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def write(path, nb):
    import ast
    for idx, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = c["source"] if isinstance(c["source"], str) else "".join(c["source"])
            py_src = "\n".join(
                ("# " + line if line.lstrip().startswith(("%", "!")) else line)
                for line in src.splitlines()
            )
            ast.parse(py_src, filename=f"{path.name}:cell-{idx}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


# --------------------------------------------------------------------------- template (no-auth demo service)
TEMPLATE_CFG = dict(name="my-service", service_port=8000, auth="caddy", public_port=18080,
                    health_path="/", ready_timeout=600)
TEMPLATE_SETUP = r'''# @title 1. Setup -- idempotent: safe to re-run on a reused session
# Install the service here. Skip work that is already done, cache big downloads in DATA_DIR (Drive).
# Example:
#   if not shutil.which("llama-server"): subprocess.run("pip install -q ...", shell=True, check=True)
print("nothing to install for the demo service")
'''
TEMPLATE_SERVICE = r'''# @title 2. Service -- must listen on 127.0.0.1:SERVICE_PORT; register it in PROCS for the watchdog
def start_service():
    # replace with the real server, e.g. ["vllm", "serve", MODEL, "--host", "127.0.0.1", "--port", str(SERVICE_PORT)]
    return ct_spawn("service", [sys.executable, "-m", "http.server", str(SERVICE_PORT), "--bind", "127.0.0.1"],
                    cwd=DATA_DIR)

PROCS["service"] = (start_service(), start_service)
print(f"service started on 127.0.0.1:{SERVICE_PORT}")
'''

# --------------------------------------------------------------------------- ollama (no built-in auth -> caddy)
OLLAMA_CFG = dict(name="ollama", service_port=11434, auth="caddy", public_port=18080,
                  health_path="/api/version", ready_timeout=1800)
OLLAMA_SETUP = (
    r'''# @title 1. Setup -- install Ollama and load runtime configuration (idempotent)
from pathlib import Path
(Path(WORK) / "ollama_runtime.py").write_text('''
    + repr((ROOT / "src/colabllm/lib/ollama_runtime.py").read_text())
    + r''', encoding="utf-8")
if WORK not in sys.path:
    sys.path.insert(0, WORK)

from ollama_runtime import (
    COMPAT_APIS,
    build_ollama_server_env,
    compat_endpoints_info,
    copy_model_alias,
    models_to_pull,
    parse_compat_apis,
    parse_model_aliases,
    parse_model_specs,
    preload_or_unload_model,
    pull_model_stream,
    render_ollama_caddyfile,
    validate_ollama_version,
)

def olog(msg):
    print(time.strftime("[%F %T] [ollama] ") + msg, flush=True)

OLLAMA_NO_MODELS = os.environ.get("CT_OLLAMA_NO_MODELS", "0") == "1"
OLLAMA_MODELS_LIST = parse_model_specs(os.environ.get("CT_MODELS"), no_models=OLLAMA_NO_MODELS)
OLLAMA_ALIASES = parse_model_aliases(os.environ.get("CT_OLLAMA_ALIASES"))
OLLAMA_PULL_LIST = models_to_pull(OLLAMA_MODELS_LIST, OLLAMA_ALIASES)
OLLAMA_PRELOAD_MODE = os.environ.get("CT_OLLAMA_PRELOAD", "primary").strip().lower()
OLLAMA_COMPAT = parse_compat_apis(os.environ.get("CT_OLLAMA_COMPAT"))
OLLAMA_READ_ONLY_MODELS = os.environ.get("CT_OLLAMA_READ_ONLY_MODELS", "0") == "1"
OLLAMA_STORAGE_MODE = os.environ.get("CT_OLLAMA_MODELS_STORAGE", "ssd").strip().lower()
OLLAMA_MODELS_DIR = os.environ.get("CT_OLLAMA_MODELS") or (
    f"{DATA_DIR}/models" if OLLAMA_STORAGE_MODE == "drive" else "/content/ollama/models"
)
os.makedirs(OLLAMA_MODELS_DIR, exist_ok=True)

OLLAMA_REQUESTED_VERSION = (
    validate_ollama_version(os.environ["CT_OLLAMA_VERSION"])
    if os.environ.get("CT_OLLAMA_VERSION")
    else ""
)
_installed_ver = (
    subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    if shutil.which("ollama")
    else ""
)
if not shutil.which("ollama") or (OLLAMA_REQUESTED_VERSION and OLLAMA_REQUESTED_VERSION not in _installed_ver):
    olog("Installing Ollama" + (f" v{OLLAMA_REQUESTED_VERSION}" if OLLAMA_REQUESTED_VERSION else "") + " ...")
    _inst_env = dict(os.environ)
    if OLLAMA_REQUESTED_VERSION:
        _inst_env["OLLAMA_VERSION"] = OLLAMA_REQUESTED_VERSION
    subprocess.run(
        "apt-get install -y -qq zstd pciutils lshw >/dev/null 2>&1; "
        "curl -fsSL https://ollama.com/install.sh | sh",
        shell=True,
        check=True,
        env=_inst_env,
    )
# Ensure systemd does not hold :11434 with default settings instead of our tuned OLLAMA_ENV
subprocess.run("systemctl stop ollama >/dev/null 2>&1 || true; systemctl disable ollama >/dev/null 2>&1 || true",
               shell=True, check=False)
olog(subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip())
'''
)

OLLAMA_SERVICE = r'''# @title 2. Service -- ollama serve on 127.0.0.1 + automatic model pull, alias & GPU VRAM preload
if "OLLAMA_COMPAT" not in globals():
    raise RuntimeError("Ollama setup cell failed before completing initialization; see traceback above.")

_server_env_raw = os.environ.get("CT_OLLAMA_SERVER_ENV", "").strip()
OLLAMA_ENV = json.loads(_server_env_raw) if _server_env_raw else build_ollama_server_env()
for _k, _v in os.environ.items():
    if _k.startswith("OLLAMA_") and _k not in ("OLLAMA_HOST", "OLLAMA_MODELS") and _k not in OLLAMA_ENV:
        OLLAMA_ENV[_k] = _v
OLLAMA_ENV["OLLAMA_HOST"] = f"127.0.0.1:{SERVICE_PORT}"
OLLAMA_ENV["OLLAMA_MODELS"] = OLLAMA_MODELS_DIR

olog(f"GPU: {ct_gpu_stat('name,memory.total,compute_cap')} | storage={OLLAMA_STORAGE_MODE} ({OLLAMA_MODELS_DIR})")
olog(f"Compatible APIs: {','.join(OLLAMA_COMPAT)} | read_only_models={OLLAMA_READ_ONLY_MODELS}")
olog("Server env: " + ", ".join(f"{k}={v}" for k, v in sorted(OLLAMA_ENV.items())))

def start_service():
    subprocess.run(["pkill", "-f", "ollama serve"], check=False)
    return ct_spawn("service", ["ollama", "serve"], env=OLLAMA_ENV)

def _wait_ollama_local(timeout=120):
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{SERVICE_PORT}/api/version"
    while time.time() < deadline:
        proc = PROCS["service"][0]
        if proc.poll() is not None:
            raise RuntimeError(f"ollama serve exited early (code {proc.returncode}):\n{ct_tail('service', 50)}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"ollama serve not ready after {timeout}s:\n{ct_tail('service', 50)}")

PROCS["service"] = (start_service(), start_service)
_wait_ollama_local(timeout=120)

LOCAL_OLLAMA_URL = f"http://127.0.0.1:{SERVICE_PORT}"
for _m in OLLAMA_PULL_LIST:
    pull_model_stream(LOCAL_OLLAMA_URL, _m, log_fn=olog, timeout=1800, retries=3)

for _alias_name, _src_model in OLLAMA_ALIASES:
    copy_model_alias(LOCAL_OLLAMA_URL, _src_model, _alias_name, timeout=60)
    olog(f"Created model alias: {_alias_name} -> {_src_model}")

_preload_targets = []
if OLLAMA_PRELOAD_MODE == "all":
    _preload_targets = list(OLLAMA_MODELS_LIST or [a for a, _ in OLLAMA_ALIASES])
elif OLLAMA_PRELOAD_MODE != "none":
    _candidates = list(OLLAMA_MODELS_LIST or [a for a, _ in OLLAMA_ALIASES])
    if _candidates:
        _preload_targets = [_candidates[0]]

for _pm in _preload_targets:
    olog(f"Preloading {_pm} into GPU memory (keep_alive={OLLAMA_ENV.get('OLLAMA_KEEP_ALIVE')}) ...")
    _preload_res = preload_or_unload_model(
        LOCAL_OLLAMA_URL,
        _pm,
        keep_alive=OLLAMA_ENV.get("OLLAMA_KEEP_ALIVE"),
        timeout=600,
    )
    olog(f"Preloaded {_pm} via {_preload_res['endpoint']}")

print(subprocess.run(["ollama", "list"], env={**os.environ, **OLLAMA_ENV}, capture_output=True, text=True).stdout)
print(subprocess.run(["ollama", "ps"], env={**os.environ, **OLLAMA_ENV}, capture_output=True, text=True).stdout)
'''

OLLAMA_CADDY = r'''# @title 3. Auth & compatible API routing -- Caddy (Basic Auth + Bearer + Anthropic X-Api-Key + --compat filter)
import hashlib, platform
CADDY_ARCH = {"x86_64": "amd64", "aarch64": "arm64"}.get(platform.machine(), "amd64")

def start_caddy():
    if not os.path.exists("/usr/local/bin/caddy"):
        tag = json.loads(urllib.request.urlopen(
            "https://api.github.com/repos/caddyserver/caddy/releases/latest", timeout=30).read())["tag_name"]
        tgz = "/tmp/caddy.tgz"
        with urllib.request.urlopen(f"https://github.com/caddyserver/caddy/releases/download/{tag}/"
                                    f"caddy_{tag.lstrip('v')}_linux_{CADDY_ARCH}.tar.gz", timeout=60) as src, \
                open(tgz, "wb") as dst:
            shutil.copyfileobj(src, dst)
        subprocess.run(["tar", "-xzf", tgz, "-C", "/usr/local/bin", "caddy"], check=True)
    cookie_tok = hashlib.sha256(f"ct-cookie-v2:{AUTH_USER}:{AUTH_PASS}:{API_TOKEN}".encode()).hexdigest()
    hashed = subprocess.check_output(["caddy", "hash-password", "--plaintext", AUTH_PASS], text=True).strip()
    caddyfile = render_ollama_caddyfile(
        public_port=PUBLIC_PORT,
        service_port=SERVICE_PORT,
        auth_user=AUTH_USER,
        hashed_pass=hashed,
        api_token=API_TOKEN,
        cookie_tok=cookie_tok,
        compat_apis=OLLAMA_COMPAT,
        read_only_models=OLLAMA_READ_ONLY_MODELS,
    )
    open(f"{WORK}/Caddyfile", "w", encoding="utf-8").write(caddyfile)
    subprocess.run(["pkill", "-f", "caddy run"], check=False)
    return ct_spawn("caddy", ["caddy", "run", "--config", f"{WORK}/Caddyfile", "--adapter", "caddyfile"])

AUTH_OK = False
if AUTH == "caddy":
    PROCS["caddy"] = (start_caddy(), start_caddy)
    time.sleep(2)
    if PROCS["caddy"][0].poll() is not None:
        raise RuntimeError(f"caddy exited -- is :{PUBLIC_PORT} in use?\n" + ct_tail("caddy"))
    ct_wait_http(f"http://127.0.0.1:{PUBLIC_PORT}/", timeout=60, ok=lambda c: c == 401)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PUBLIC_PORT}/", timeout=5)
    except urllib.error.HTTPError as _e:
        if _e.headers.get("Set-Cookie"):
            raise RuntimeError("Caddy leaked Set-Cookie on unauthenticated 401 response!")
    AUTH_OK = True
    olog(f"Caddy on :{PUBLIC_PORT} -> 127.0.0.1:{SERVICE_PORT} (Bearer + X-Api-Key + Basic Auth, compat={','.join(OLLAMA_COMPAT)})")
else:
    AUTH_OK = True
    olog(f"auth={AUTH}: targeting :{PUBLIC_PORT} directly")
'''

OLLAMA_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- verify Bearer, X-Api-Key, Basic Auth & compatible API policy before CT_READY
import base64

if "OLLAMA_COMPAT" not in globals() or "service" not in PROCS:
    raise RuntimeError("Ollama service did not start because an earlier notebook cell failed; see traceback above.")

ct_wait_http(f"http://127.0.0.1:{SERVICE_PORT}/api/version", timeout=120, ok=lambda c: c == 200)

def _expect_http(url, *, headers=None, method="GET", body=None, expected=401, label="probe"):
    req = urllib.request.Request(url, data=body, headers={"User-Agent": "Mozilla/5.0", **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status != expected:
                raise RuntimeError(f"{label}: expected HTTP {expected}, got HTTP {resp.status}")
            return resp
    except urllib.error.HTTPError as err:
        if err.code != expected:
            raise RuntimeError(f"{label}: expected HTTP {expected}, got HTTP {err.code}")
        return err

def _public_bearer_probe():
    req = urllib.request.Request(
        PUBLIC_URL + "/",
        headers={"Authorization": f"Bearer {API_TOKEN}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.status

def _verify_public_auth():
    if AUTH != "caddy":
        return
    # 1. Unauthenticated request must return 401 without Set-Cookie
    err_unauth = _expect_http(PUBLIC_URL + "/", expected=401, label="unauthenticated /")
    if getattr(err_unauth, "headers", None) and err_unauth.headers.get("Set-Cookie"):
        raise RuntimeError("Public 401 response leaked Set-Cookie")
    # 2. Invalid Bearer & invalid X-Api-Key must return 401
    _expect_http(PUBLIC_URL + "/", headers={"Authorization": "Bearer invalid-probe-token"}, expected=401, label="invalid Bearer")
    _expect_http(PUBLIC_URL + "/", headers={"X-Api-Key": "invalid-probe-token"}, expected=401, label="invalid X-Api-Key")
    # 3. Valid Anthropic-style X-Api-Key must succeed
    _expect_http(PUBLIC_URL + "/", headers={"X-Api-Key": API_TOKEN}, expected=200, label="valid X-Api-Key")
    # 4. Valid Basic Auth must succeed
    basic = base64.b64encode(f"{AUTH_USER}:{AUTH_PASS}".encode("utf-8")).decode("ascii")
    _expect_http(PUBLIC_URL + "/", headers={"Authorization": f"Basic {basic}"}, expected=200, label="valid Basic Auth")
    # 5. Disabled API families & read-only model guard must return 403
    auth_hdr = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"}
    if "ollama" not in OLLAMA_COMPAT:
        _expect_http(PUBLIC_URL + "/api/tags", headers=auth_hdr, expected=403, label="disabled ollama /api/tags")
    elif OLLAMA_READ_ONLY_MODELS:
        _expect_http(PUBLIC_URL + "/api/pull", headers=auth_hdr, method="POST", body=b"{}", expected=403, label="read-only /api/pull")
    if "openai" not in OLLAMA_COMPAT:
        _expect_http(PUBLIC_URL + "/v1/chat/completions", headers=auth_hdr, method="POST", body=b"{}", expected=403, label="disabled openai /v1/chat/completions")
    if "anthropic" not in OLLAMA_COMPAT:
        _expect_http(PUBLIC_URL + "/v1/messages", headers={"X-Api-Key": API_TOKEN, "Content-Type": "application/json"},
                     method="POST", body=b"{}", expected=403, label="disabled anthropic /v1/messages")

ct_public_gate(PUBLIC_URL, _public_bearer_probe, _verify_public_auth, olog, timeout=180)
if AUTH == "caddy" and not CREDS_FROM_ENV:
    print(f"CT_CREDENTIALS user={AUTH_USER} pass={AUTH_PASS}")
    print(f"API token: {API_TOKEN}")
for _api_name, _detail in compat_endpoints_info(PUBLIC_URL, OLLAMA_COMPAT, read_only_models=OLLAMA_READ_ONLY_MODELS)["interfaces"].items():
    olog(f"[{_api_name}] base_url={_detail['base_url']} routes={','.join(_detail['endpoints'])}")
print(f"{SERVICE} is up: {PUBLIC_URL}")
print("CT_READY", flush=True)

def _ollama_health():
    if PROCS["service"][0].poll() is not None:
        return False
    with urllib.request.urlopen(f"http://127.0.0.1:{SERVICE_PORT}/api/version", timeout=10) as resp:
        return resp.status == 200

def _ollama_heartbeat(uptime_s):
    if uptime_s % 600 < 60:
        states = " ".join(f"{n}={'up' if p.poll() is None else 'down'}" for n, (p, _) in PROCS.items())
        olog(f"[heartbeat] uptime={uptime_s}s {states} gpu={ct_gpu_stat()}")

ct_watchdog(_ollama_health, start_service, olog, interval=60, grace=300, heartbeat=_ollama_heartbeat)
'''

# --------------------------------------------------------------------------- unsloth studio (built-in auth)
UNSLOTH_CFG = dict(name="unsloth-studio", service_port=8888, auth="builtin", public_port=8888,
                   health_path="/api/health", ready_timeout=900)
UNSLOTH_SETUP = r'''# @title 1. Setup -- clone or update the repo, then Unsloth's own installer (re-runs are fast)
%cd /content
!test -d unsloth/.git && git -C unsloth pull --ff-only || git clone --depth 1 --branch main https://github.com/unslothai/unsloth.git
%cd /content/unsloth
!chmod +x studio/setup.sh && ./studio/setup.sh --local
'''
UNSLOTH_SERVICE = r'''# @title 2. Service -- Unsloth Studio is started by its own blocking start() in cell 5;
# this cell only prepares the import path. It has built-in auth (CT_AUTH=builtin), so no Caddy.
sys.path.insert(0, "/content/unsloth/studio/backend")
print("Unsloth Studio will listen on port", SERVICE_PORT)
'''
UNSLOTH_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- KEEP THIS CELL RUNNING
# start() blocks forever (its own keep-alive loop). Our tunnel from cell 4 is used, so cloudflare=False.
# A watcher thread reports the admin password + CT_READY and restarts cloudflared if it dies.
import threading

def _watch():
    try:
        _watch_inner()
    except BaseException as err:   # a dead watcher would otherwise leave the CLI waiting forever
        import traceback
        traceback.print_exc()
        print(f"[watchdog] fatal: {err}", flush=True)
        os._exit(1)

def _watch_inner():
    ct_wait_http(f"http://127.0.0.1:{SERVICE_PORT}/api/health", timeout=900)
    auth_dir = "/root/.unsloth/studio/auth"
    for _ in range(60):
        for fn in (".bootstrap_password", ".colab_notebook_login"):
            p = os.path.join(auth_dir, fn)
            if os.path.exists(p):
                lines = open(p).read().split()
                pw = lines[-1] if lines else ""
                print(f"CT_CREDENTIALS user=unsloth pass={pw}", flush=True)
                break
        else:
            time.sleep(2); continue
        break
    print(f"{SERVICE} is up: {PUBLIC_URL}")
    print("CT_READY", flush=True)
    while True:
        time.sleep(30)
        for name, (proc, restart) in list(PROCS.items()):
            if proc.poll() is not None:
                print(f"[watchdog] {name} exited; restarting", flush=True)
                ct_stop_proc(proc)
                PROCS[name] = (restart(), restart)

threading.Thread(target=_watch, daemon=True).start()
from colab import start
start(port=SERVICE_PORT, cloudflare=False)
'''

# --------------------------------------------------------------------------- qwen-flash (G4, llama.cpp sm_120, UD-Q4_K_XL-V2, 1M YaRN 4x + q8_0 KV + vision)
QWEN_FLASH_CFG = dict(name="qwen-flash", service_port=8000, auth="builtin", public_port=8000,
                      health_path="/health", ready_timeout=5400)

QWEN_FLASH_SETUP = (
    r'''# @title 1. Setup -- parallel llama.cpp b11540 sm_120 build/cache + 103.7 GiB UD-Q4_K_XL-V2 GGUF + BF16 mmproj + 1M G4 hardware plan
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

QWEN_WORK = Path(os.environ.get("CT_QWEN_WORK_DIR", "/content/qwen-flash"))
QWEN_WORK.mkdir(parents=True, exist_ok=True)
(QWEN_WORK / "qwen_flash_runtime.py").write_text('''
    + repr((ROOT / "src/colabllm/lib/qwen_flash_runtime.py").read_text())
    + r''', encoding="utf-8")
if str(QWEN_WORK) not in sys.path:
    sys.path.insert(0, str(QWEN_WORK))

from qwen_flash_runtime import (
    DEFAULT_ALIAS,
    DEFAULT_CONTEXT,
    DEFAULT_LLAMA_COMMIT,
    DEFAULT_LLAMA_REPO,
    DEFAULT_QUANT,
    DEFAULT_REPO,
    MODEL_NUM_LAYERS,
    build_server_args,
    compute_yarn_params,
    detect_gpu_info,
    detect_host_ram_mib,
    evict_gguf_page_cache,
    looks_like_oom,
    plan_hardware_config,
    prepare_llama_runtime,
    stage_model_and_projector,
    write_json_atomic,
)

def qlog(msg):
    print(time.strftime("[%F %T] [qwen-flash] ") + msg, flush=True)

if MOUNT_DRIVE and not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Drive is not mounted. Run via `colabllm qwen-flash up`.")

SERVICE_DIR = Path(DATA_DIR)
SERVICE_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_BACKUP_ROOT = (
    Path(os.environ.get("CT_QWEN_MODEL_ROOT", "/content/drive/MyDrive/models"))
    if (MOUNT_DRIVE and Path("/content/drive/MyDrive").is_dir())
    else None
)
QWEN_RUNTIME_CACHE = (
    (DRIVE_BACKUP_ROOT / ".runtimes" / "llama.cpp")
    if DRIVE_BACKUP_ROOT is not None
    else (SERVICE_DIR / "runtime-cache")
)

CFG = {
    "repo": os.environ.get("CT_QWEN_REPO", DEFAULT_REPO),
    "quant": os.environ.get("CT_QWEN_QUANT", DEFAULT_QUANT),
    "alias": os.environ.get("CT_QWEN_ALIAS", DEFAULT_ALIAS),
    "context": os.environ.get("CT_QWEN_CONTEXT", str(DEFAULT_CONTEXT)),
    "cache_type": os.environ.get("CT_QWEN_CACHE_TYPE", "q8_0"),
    "cpu_moe": os.environ.get("CT_QWEN_CPU_MOE", "auto"),
    "parallel": int(os.environ.get("CT_QWEN_PARALLEL", "1")),
    "batch": int(os.environ.get("CT_QWEN_BATCH", "512")),
    "ubatch": int(os.environ.get("CT_QWEN_UBATCH", "128")),
    "reasoning": os.environ.get("CT_QWEN_REASONING", "on"),
    "enable_vision": os.environ.get("CT_QWEN_VISION", "1") != "0",
    "llama_repo": os.environ.get("CT_LLAMA_REPO", DEFAULT_LLAMA_REPO),
    "llama_commit": os.environ.get("CT_LLAMA_COMMIT", DEFAULT_LLAMA_COMMIT),
}

QWEN_GPU = detect_gpu_info()
QWEN_RAM_MIB = detect_host_ram_mib()
HW_PLAN = plan_hardware_config(
    vram_mib=QWEN_GPU["vram_mib"],
    ram_mib=QWEN_RAM_MIB,
    quant=CFG["quant"],
    requested_context=CFG["context"],
    cache_type=CFG["cache_type"],
    requested_cpu_moe=CFG["cpu_moe"],
    enable_vision=CFG["enable_vision"],
    parallel=CFG["parallel"],
    batch=CFG["batch"],
    ubatch=CFG["ubatch"],
)
qlog(
    f"Hardware plan: gpu={QWEN_GPU['name']} ({QWEN_GPU['vram_mib']} MiB VRAM, sm_{QWEN_GPU['cuda_arch'] or 'cpu'}) "
    f"ram={QWEN_RAM_MIB} MiB -> mode={HW_PLAN['mode']} context={HW_PLAN['context']} "
    f"(yarn={HW_PLAN['rope_scale']}x, kv={HW_PLAN['cache_type']} {HW_PLAN['kv_cache_mib']} MiB, "
    f"n_cpu_moe={HW_PLAN['n_cpu_moe']}, ple_in_ram={HW_PLAN['ple_in_ram']}, "
    f"vram_headroom={HW_PLAN['estimated_vram_headroom_mib']} MiB)"
)

with ThreadPoolExecutor(max_workers=2) as _pool:
    _f_runtime = _pool.submit(
        prepare_llama_runtime,
        work_dir=QWEN_WORK,
        cache_root=QWEN_RUNTIME_CACHE,
        llama_repo=CFG["llama_repo"],
        llama_commit=CFG["llama_commit"],
        gpu_info=QWEN_GPU,
        log=qlog,
    )
    _f_model = _pool.submit(
        stage_model_and_projector,
        repo=CFG["repo"],
        quant=CFG["quant"],
        enable_vision=CFG["enable_vision"],
        service_dir=SERVICE_DIR,
        work_dir=QWEN_WORK,
        drive_backup_root=DRIVE_BACKUP_ROOT,
        log=qlog,
    )
    while not (_f_runtime.done() and _f_model.done()):
        for _fut in (_f_runtime, _f_model):
            if _fut.done() and _fut.exception():
                raise _fut.exception()
        _free_gb = shutil.disk_usage(QWEN_WORK).free / 1e9
        qlog(
            f"Preparation heartbeat | build={'done' if _f_runtime.done() else 'running'} "
            f"model={'done' if _f_model.done() else 'running'} | free_ssd={_free_gb:.1f} GB"
        )
        time.sleep(30)
    BIN_DIR, LLAMA_ENV, PINNED_COMMIT = _f_runtime.result()
    MODEL_PATH, PROJECTOR_PATH = _f_model.result()

QWEN_SETUP_OK = True
qlog(f"Setup complete: commit={PINNED_COMMIT[:10]} model={MODEL_PATH} projector={PROJECTOR_PATH}")
'''
)

QWEN_FLASH_SERVICE = r'''# @title 2. Service -- llama-server on 127.0.0.1 (UD-Q4_K_XL-V2, 1M YaRN 4x, q8_0 KV, PLE in DDR5 RAM, --api-key-file)
if not globals().get("QWEN_SETUP_OK"):
    raise RuntimeError("qwen-flash setup cell failed before completing model/runtime preparation; check traceback above.")

KEY_FILE = QWEN_WORK / "api_key.txt"
KEY_FILE.write_text(API_TOKEN + "\n", encoding="utf-8")
KEY_FILE.chmod(0o600)
SPAWN_LOG_OFFSET = 0

def start_service():
    global SPAWN_LOG_OFFSET
    subprocess.run(["pkill", "-f", str(BIN_DIR / "llama-server")], check=False, timeout=15)
    if HW_PLAN.get("ple_in_ram"):
        evict_gguf_page_cache(QWEN_WORK / "model", log=qlog)
    args = build_server_args(
        server_bin=BIN_DIR / "llama-server",
        model_path=MODEL_PATH,
        alias=CFG["alias"],
        port=SERVICE_PORT,
        key_file=KEY_FILE,
        plan=HW_PLAN,
        projector_path=PROJECTOR_PATH,
        reasoning=CFG["reasoning"],
    )
    SPAWN_LOG_OFFSET = ct_log_size("llama-server")
    ple_loc = "host-ram" if HW_PLAN.get("ple_in_ram") else "ssd-mmap-lazy"
    qlog(
        f"Starting llama-server on 127.0.0.1:{SERVICE_PORT} | ctx={HW_PLAN['context']} "
        f"mode={HW_PLAN['mode']} ple={ple_loc} kv={HW_PLAN['cache_type']} "
        f"batch={HW_PLAN['batch']}/{HW_PLAN['ubatch']} reasoning={CFG['reasoning']}"
    )
    return ct_spawn("llama-server", args, env=LLAMA_ENV)

PROCS["service"] = (start_service(), start_service)
'''

QWEN_FLASH_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- auto-stepup CPU MoE on OOM, verify auth/1M context/vision, smoke completion & supervise
if not globals().get("QWEN_SETUP_OK") or "service" not in PROCS:
    raise RuntimeError("qwen-flash service did not start because an earlier notebook cell failed; see traceback above.")

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def http_json(url, key=None, payload=None, timeout=15):
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url,
        headers=headers,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, {}

def auth_check(base, key):
    for candidate, expected in ((None, 401), ("invalid-deployment-test-key", 401), (key, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Authentication test failed on {base}: expected {expected}, got {code}")

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
moe_stepups = 0
ctx_fallback = [819_200, 524_288]
ctx_idx = 0
deadline = time.monotonic() + 2400
last_notice = 0.0

while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        recent_log = ct_log_since("llama-server", SPAWN_LOG_OFFSET)
        if (
            CFG["cpu_moe"].strip().lower() == "auto"
            and moe_stepups < 3
            and HW_PLAN["n_cpu_moe"] + 2 <= MODEL_NUM_LAYERS
            and looks_like_oom(recent_log)
        ):
            moe_stepups += 1
            prev_moe = HW_PLAN["n_cpu_moe"]
            HW_PLAN["n_cpu_moe"] = min(MODEL_NUM_LAYERS, prev_moe + 2)
            HW_PLAN["mode"] = f"hybrid-cpu-moe-{HW_PLAN['n_cpu_moe']}"
            qlog(
                f"Startup CUDA OOM at n_cpu_moe={prev_moe}; offloading +2 expert layers "
                f"(-> n_cpu_moe={HW_PLAN['n_cpu_moe']}) while keeping ctx={HW_PLAN['context']}"
            )
            PROCS["service"] = (start_service(), start_service)
            continue
        if HW_PLAN.get("context_auto") and ctx_idx < len(ctx_fallback) and looks_like_oom(recent_log):
            prev_ctx = HW_PLAN["context"]
            next_ctx = ctx_fallback[ctx_idx]
            ctx_idx += 1
            if next_ctx < prev_ctx:
                rs, rscale, yorig = compute_yarn_params(next_ctx)
                HW_PLAN["context"] = next_ctx
                HW_PLAN["rope_scaling"] = rs
                HW_PLAN["rope_scale"] = rscale
                HW_PLAN["yarn_orig_ctx"] = yorig
                qlog(f"Startup OOM at ctx={prev_ctx}; stepping down to ctx={next_ctx} and retrying")
                PROCS["service"] = (start_service(), start_service)
                continue
        raise RuntimeError("llama-server exited during load:\n" + ct_tail("llama-server", 60))
    try:
        if http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200:
            break
    except (OSError, ValueError):
        pass
    if time.monotonic() - last_notice > 45:
        qlog(
            f"Waiting for model load / warmup (ctx={HW_PLAN['context']}, mode={HW_PLAN['mode']}) | "
            f"GPU: {ct_gpu_stat('memory.used,memory.total,utilization.gpu')}"
        )
        last_notice = time.monotonic()
    time.sleep(5)
else:
    raise RuntimeError("Model load timeout:\n" + ct_tail("llama-server", 60))

if HW_PLAN.get("ple_in_ram"):
    evict_gguf_page_cache(QWEN_WORK / "model", log=qlog)

auth_check(LOCAL_BASE, API_TOKEN)
code, props = http_json(LOCAL_BASE + "/props", API_TOKEN, timeout=20)
allocated = (props.get("default_generation_settings") or {}).get("n_ctx")
if code != 200 or allocated != HW_PLAN["context"]:
    raise RuntimeError(f"Actual context ({allocated}) differs from requested context ({HW_PLAN['context']})")
vision_enabled = bool((props.get("modalities") or {}).get("vision", False))
if PROJECTOR_PATH is not None and HW_PLAN.get("enable_vision", True) and not vision_enabled:
    raise RuntimeError("Vision projector configured but server does not advertise vision")
qlog(
    f"Local verified: auth=401/401/200, n_ctx={allocated}, vision={vision_enabled}, "
    f"mode={HW_PLAN['mode']}, kv={HW_PLAN['cache_type']}, ple_in_ram={HW_PLAN['ple_in_ram']}"
)

# Public endpoint verification (DNS/edge propagation can take a few seconds)
ct_public_gate(
    PUBLIC_URL,
    lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
    lambda: auth_check(PUBLIC_URL, API_TOKEN),
    qlog,
    timeout=180,
    interval=8,
)
smoke_payload = {
    "model": CFG["alias"],
    "messages": [{"role": "user", "content": "Reply with exactly two words: deployment successful"}],
    "max_tokens": 64,
    "temperature": 0,
}
if CFG["reasoning"] != "off":
    smoke_payload["chat_template_kwargs"] = {"enable_thinking": False}
code, resp = http_json(PUBLIC_URL + "/v1/chat/completions", API_TOKEN, smoke_payload, timeout=180)
if code != 200:
    raise RuntimeError(f"Public smoke completion failed: HTTP {code}")
msg_obj = ((resp.get("choices") or [{}])[0].get("message") or {})
reply = msg_obj.get("content") or msg_obj.get("reasoning_content") or ""
qlog(f"PUBLIC VERIFIED: {PUBLIC_URL}/v1 | n_ctx={allocated} | reply={reply.strip()[:60]!r}")

ver_rec = {
    "service": SERVICE,
    "verified_at": time.strftime("%FT%T%z"),
    "base_url": PUBLIC_URL + "/v1",
    "repo": CFG["repo"],
    "quant": CFG["quant"],
    "alias": CFG["alias"],
    "mode": HW_PLAN["mode"],
    "n_cpu_moe": HW_PLAN["n_cpu_moe"],
    "cache_type": HW_PLAN["cache_type"],
    "ple_in_ram": HW_PLAN["ple_in_ram"],
    "context_allocated": allocated,
    "vision_enabled": vision_enabled,
    "reply": reply[:120],
}
write_json_atomic(SERVICE_DIR / "last-verification.json", ver_rec)
if SERVICE != "qwen-flash":
    write_json_atomic(SERVICE_DIR / f"last-verification-{SERVICE}.json", ver_rec)

print(f"{SERVICE} is up: {PUBLIC_URL} (n_ctx={allocated}, mode={HW_PLAN['mode']})")
print("CT_READY", flush=True)

def qwen_heartbeat(uptime):
    gpu_mem = ct_gpu_stat("memory.used,memory.total,utilization.gpu")
    status_rec = {
        "service": SERVICE,
        "kernel_pid": os.getpid(),
        "server_pid": PROCS["service"][0].pid,
        "tunnel_pid": PROCS[TUNNEL_PROCESS_KEY][0].pid,
        "context": HW_PLAN["context"],
        "mode": HW_PLAN["mode"],
        "n_cpu_moe": HW_PLAN["n_cpu_moe"],
        "cache_type": HW_PLAN["cache_type"],
        "alias": CFG["alias"],
        "url": PUBLIC_URL + "/v1",
        "updated_at": time.strftime("%FT%T%z"),
        "uptime_sec": uptime,
        "gpu_memory": gpu_mem,
    }
    try:
        write_json_atomic(SERVICE_DIR / "last-status.json", status_rec)
        if SERVICE != "qwen-flash":
            write_json_atomic(SERVICE_DIR / f"last-status-{SERVICE}.json", status_rec)
    except OSError:
        pass
    qlog(f"heartbeat | up={uptime}s | n_ctx={HW_PLAN['context']} | GPU: {gpu_mem}")

ct_watchdog(
    lambda: http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200,
    start_service,
    qlog,
    log_name="llama-server",
    heartbeat=qwen_heartbeat,
)
'''

# --------------------------------------------------------------------------- deepseek-qwen-9b (L4 default / T4, llama.cpp b11540, i1-Q4_K_M + native MTP, 256K + q8_0 KV + F16 vision)
DEEPSEEK_QWEN_9B_CFG = dict(name="deepseek-qwen-9b", service_port=8000, auth="builtin", public_port=8000,
                            health_path="/health", ready_timeout=2400)

DEEPSEEK_QWEN_9B_SETUP = (
    r'''# @title 1. Setup -- parallel llama.cpp b11540 CUDA cache + i1-Q4_K_M GGUF + native MTP Q8_0 sidecar + F16 mmproj + 256K L4/T4 plan
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DS9B_WORK = Path(os.environ.get("CT_DS9B_WORK_DIR", "/content/deepseek-qwen-9b"))
DS9B_WORK.mkdir(parents=True, exist_ok=True)
(DS9B_WORK / "deepseek_qwen_9b_runtime.py").write_text('''
    + repr((ROOT / "src/colabllm/lib/deepseek_qwen_9b_runtime.py").read_text())
    + r''', encoding="utf-8")
if str(DS9B_WORK) not in sys.path:
    sys.path.insert(0, str(DS9B_WORK))

from deepseek_qwen_9b_runtime import (
    DEFAULT_ALIAS,
    DEFAULT_LLAMA_COMMIT,
    DEFAULT_LLAMA_REPO,
    DEFAULT_MTP_DRAFT_MAX,
    DEFAULT_MTP_MODE,
    DEFAULT_QUANT,
    DEFAULT_REPO,
    UPSTREAM_HF_REPO,
    build_server_args,
    compute_yarn_params,
    detect_gpu_info,
    detect_host_ram_mib,
    looks_like_oom,
    plan_hardware_config,
    prepare_llama_runtime,
    stage_model_and_projector,
    stage_mtp_sidecar,
    write_json_atomic,
)

def ds9log(msg):
    print(time.strftime("[%F %T] [deepseek-qwen-9b] ") + msg, flush=True)

if MOUNT_DRIVE and not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Drive is not mounted. Run via `colabllm deepseek-qwen-9b up`.")

SERVICE_DIR = Path(DATA_DIR)
SERVICE_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_BACKUP_ROOT = (
    Path(os.environ.get("CT_DS9B_MODEL_ROOT", "/content/drive/MyDrive/models"))
    if (MOUNT_DRIVE and Path("/content/drive/MyDrive").is_dir())
    else None
)
DS9B_RUNTIME_CACHE = (
    (DRIVE_BACKUP_ROOT / ".runtimes" / "llama.cpp")
    if DRIVE_BACKUP_ROOT is not None
    else (SERVICE_DIR / "runtime-cache")
)

CFG = {
    "repo": os.environ.get("CT_DS9B_REPO", DEFAULT_REPO),
    "upstream_repo": os.environ.get("CT_DS9B_UPSTREAM_REPO", UPSTREAM_HF_REPO),
    "quant": os.environ.get("CT_DS9B_QUANT", DEFAULT_QUANT),
    "alias": os.environ.get("CT_DS9B_ALIAS", DEFAULT_ALIAS),
    "context": os.environ.get("CT_DS9B_CONTEXT", "auto"),
    "cache_type": os.environ.get("CT_DS9B_CACHE_TYPE", "q8_0"),
    "parallel": int(os.environ.get("CT_DS9B_PARALLEL", "1")),
    "batch": int(os.environ.get("CT_DS9B_BATCH", "2048")),
    "ubatch": int(os.environ.get("CT_DS9B_UBATCH", "512")),
    "reasoning": os.environ.get("CT_DS9B_REASONING", "on"),
    "enable_vision": os.environ.get("CT_DS9B_VISION", "1") != "0",
    "mtp": os.environ.get("CT_DS9B_MTP", DEFAULT_MTP_MODE),
    "mtp_draft_max": int(os.environ.get("CT_DS9B_MTP_DRAFT", str(DEFAULT_MTP_DRAFT_MAX))),
    "llama_repo": os.environ.get("CT_LLAMA_REPO", DEFAULT_LLAMA_REPO),
    "llama_commit": os.environ.get("CT_LLAMA_COMMIT", DEFAULT_LLAMA_COMMIT),
}

DS9B_GPU = detect_gpu_info()
DS9B_RAM_MIB = detect_host_ram_mib()
HW_PLAN = plan_hardware_config(
    vram_mib=DS9B_GPU["vram_mib"],
    ram_mib=DS9B_RAM_MIB,
    quant=CFG["quant"],
    requested_context=CFG["context"],
    cache_type=CFG["cache_type"],
    enable_vision=CFG["enable_vision"],
    mtp=CFG["mtp"],
    mtp_draft_max=CFG["mtp_draft_max"],
    parallel=CFG["parallel"],
    batch=CFG["batch"],
    ubatch=CFG["ubatch"],
)
ds9log(
    f"Hardware plan: gpu={DS9B_GPU['name']} ({DS9B_GPU['vram_mib']} MiB VRAM, sm_{DS9B_GPU['cuda_arch'] or 'cpu'}) "
    f"ram={DS9B_RAM_MIB} MiB -> mode={HW_PLAN['mode']} quant={HW_PLAN['quant']} mtp={HW_PLAN['enable_mtp']} "
    f"context={HW_PLAN['context']} (yarn={HW_PLAN['rope_scale']}x, kv={HW_PLAN['cache_type']} {HW_PLAN['kv_cache_mib']} MiB, "
    f"vram_headroom={HW_PLAN['estimated_vram_headroom_mib']} MiB)"
)

with ThreadPoolExecutor(max_workers=3) as _pool:
    _f_runtime = _pool.submit(
        prepare_llama_runtime,
        work_dir=DS9B_WORK,
        cache_root=DS9B_RUNTIME_CACHE,
        llama_repo=CFG["llama_repo"],
        llama_commit=CFG["llama_commit"],
        gpu_info=DS9B_GPU,
        log=ds9log,
    )
    _f_model = _pool.submit(
        stage_model_and_projector,
        repo=CFG["repo"],
        quant=HW_PLAN["quant"],
        enable_vision=CFG["enable_vision"],
        service_dir=SERVICE_DIR,
        work_dir=DS9B_WORK,
        drive_backup_root=DRIVE_BACKUP_ROOT,
        log=ds9log,
    )
    _f_mtp = _pool.submit(
        stage_mtp_sidecar,
        upstream_repo=CFG["upstream_repo"],
        llama_repo=CFG["llama_repo"],
        llama_commit=CFG["llama_commit"],
        work_dir=DS9B_WORK,
        drive_backup_root=DRIVE_BACKUP_ROOT,
        mtp_mode=("off" if not HW_PLAN.get("enable_mtp") else CFG["mtp"]),
        log=ds9log,
    )
    while not (_f_runtime.done() and _f_model.done() and _f_mtp.done()):
        for _fut in (_f_runtime, _f_model, _f_mtp):
            if _fut.done() and _fut.exception():
                raise _fut.exception()
        _free_gb = shutil.disk_usage(DS9B_WORK).free / 1e9
        ds9log(
            f"Preparation heartbeat | build={'done' if _f_runtime.done() else 'running'} "
            f"model={'done' if _f_model.done() else 'running'} "
            f"mtp={'done' if _f_mtp.done() else 'running'} | free_ssd={_free_gb:.1f} GB"
        )
        time.sleep(25)
    BIN_DIR, LLAMA_ENV, PINNED_COMMIT = _f_runtime.result()
    MODEL_PATH, PROJECTOR_PATH = _f_model.result()
    MTP_PATH = _f_mtp.result()

if MTP_PATH is None:
    HW_PLAN["enable_mtp"] = False

DS9B_SETUP_OK = True
ds9log(f"Setup complete: commit={PINNED_COMMIT[:10]} model={MODEL_PATH} projector={PROJECTOR_PATH} mtp={MTP_PATH}")
'''
)

DEEPSEEK_QWEN_9B_SERVICE = r'''# @title 2. Service -- llama-server on 127.0.0.1 (DeepSeek-V4-Pro-Qwen3.5-9B, i1-Q4_K_M + draft-mtp, 256K q8_0 KV, F16 vision)
if not globals().get("DS9B_SETUP_OK"):
    raise RuntimeError("deepseek-qwen-9b setup cell failed before completing model/runtime preparation; check traceback above.")

KEY_FILE = DS9B_WORK / "api_key.txt"
KEY_FILE.write_text(API_TOKEN + "\n", encoding="utf-8")
KEY_FILE.chmod(0o600)
SPAWN_LOG_OFFSET = 0

def start_service():
    global SPAWN_LOG_OFFSET
    subprocess.run(["pkill", "-f", str(BIN_DIR / "llama-server")], check=False, timeout=15)
    args = build_server_args(
        server_bin=BIN_DIR / "llama-server",
        model_path=MODEL_PATH,
        alias=CFG["alias"],
        port=SERVICE_PORT,
        key_file=KEY_FILE,
        plan=HW_PLAN,
        projector_path=PROJECTOR_PATH,
        mtp_path=MTP_PATH,
        reasoning=CFG["reasoning"],
    )
    SPAWN_LOG_OFFSET = ct_log_size("llama-server")
    ds9log(
        f"Starting llama-server on 127.0.0.1:{SERVICE_PORT} | quant={HW_PLAN['quant']} ctx={HW_PLAN['context']} "
        f"mtp={bool(MTP_PATH and HW_PLAN.get('enable_mtp'))}(n={HW_PLAN.get('mtp_draft_max')}) "
        f"mode={HW_PLAN['mode']} kv={HW_PLAN['cache_type']} "
        f"batch={HW_PLAN['batch']}/{HW_PLAN['ubatch']} reasoning={CFG['reasoning']}"
    )
    return ct_spawn("llama-server", args, env=LLAMA_ENV)

PROCS["service"] = (start_service(), start_service)
'''

DEEPSEEK_QWEN_9B_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- auto-stepdown context on OOM, verify auth/256K context/vision, smoke completion & supervise
if not globals().get("DS9B_SETUP_OK") or "service" not in PROCS:
    raise RuntimeError("deepseek-qwen-9b service did not start because an earlier notebook cell failed; see traceback above.")

BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def http_json(url, key=None, payload=None, timeout=15):
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url,
        headers=headers,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, {}

def auth_check(base, key):
    for candidate, expected in ((None, 401), ("invalid-deployment-test-key", 401), (key, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Authentication test failed on {base}: expected {expected}, got {code}")

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
ctx_fallback = [196_608, 131_072, 65_536, 32_768]
ctx_idx = 0
deadline = time.monotonic() + 900
last_notice = 0.0

while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        recent_log = ct_log_since("llama-server", SPAWN_LOG_OFFSET)
        if HW_PLAN.get("context_auto") and ctx_idx < len(ctx_fallback) and looks_like_oom(recent_log):
            prev_ctx = HW_PLAN["context"]
            while ctx_idx < len(ctx_fallback) and ctx_fallback[ctx_idx] >= prev_ctx:
                ctx_idx += 1
            if ctx_idx < len(ctx_fallback):
                next_ctx = ctx_fallback[ctx_idx]
                ctx_idx += 1
                rs, rscale, yorig = compute_yarn_params(next_ctx)
                HW_PLAN["context"] = next_ctx
                HW_PLAN["rope_scaling"] = rs
                HW_PLAN["rope_scale"] = rscale
                HW_PLAN["yarn_orig_ctx"] = yorig
                ds9log(f"Startup OOM at ctx={prev_ctx}; stepping down to ctx={next_ctx} and retrying")
                PROCS["service"] = (start_service(), start_service)
                continue
        if HW_PLAN.get("enable_mtp") and str(CFG.get("mtp", "auto")).lower() == "auto":
            HW_PLAN["enable_mtp"] = False
            ds9log("llama-server exited during MTP draft startup in auto mode; retrying without MTP draft:\n" + recent_log[-500:])
            PROCS["service"] = (start_service(), start_service)
            continue
        raise RuntimeError("llama-server exited during load:\n" + ct_tail("llama-server", 60))
    try:
        if http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200:
            break
    except (OSError, ValueError):
        pass
    if time.monotonic() - last_notice > 30:
        ds9log(
            f"Waiting for model load / warmup (quant={HW_PLAN['quant']}, mtp={HW_PLAN.get('enable_mtp')}, ctx={HW_PLAN['context']}) | "
            f"GPU: {ct_gpu_stat('memory.used,memory.total,utilization.gpu')}"
        )
        last_notice = time.monotonic()
    time.sleep(4)
else:
    raise RuntimeError("Model load timeout:\n" + ct_tail("llama-server", 60))

auth_check(LOCAL_BASE, API_TOKEN)
code, props = http_json(LOCAL_BASE + "/props", API_TOKEN, timeout=20)
allocated = (props.get("default_generation_settings") or {}).get("n_ctx")
if code != 200 or allocated != HW_PLAN["context"]:
    raise RuntimeError(f"Actual context ({allocated}) differs from requested context ({HW_PLAN['context']})")
vision_enabled = bool((props.get("modalities") or {}).get("vision", False))
if PROJECTOR_PATH is not None and HW_PLAN.get("enable_vision", True) and not vision_enabled:
    raise RuntimeError("Vision projector configured but server does not advertise vision")
mtp_active = bool(MTP_PATH is not None and HW_PLAN.get("enable_mtp", False))
ds9log(
    f"Local verified: auth=401/401/200, quant={HW_PLAN['quant']}, mtp={mtp_active}, n_ctx={allocated}, "
    f"vision={vision_enabled}, mode={HW_PLAN['mode']}, kv={HW_PLAN['cache_type']}"
)

# Public endpoint verification (DNS/edge propagation can take a few seconds)
ct_public_gate(
    PUBLIC_URL,
    lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
    lambda: auth_check(PUBLIC_URL, API_TOKEN),
    ds9log,
    timeout=180,
    interval=8,
)
smoke_payload = {
    "model": CFG["alias"],
    "messages": [{"role": "user", "content": "Reply with exactly two words: deployment successful"}],
    "max_tokens": 64,
    "temperature": 0,
}
if CFG["reasoning"] != "off":
    smoke_payload["chat_template_kwargs"] = {"enable_thinking": False}
code, resp = http_json(PUBLIC_URL + "/v1/chat/completions", API_TOKEN, smoke_payload, timeout=120)
if code != 200:
    raise RuntimeError(f"Public smoke completion failed: HTTP {code}")
msg_obj = ((resp.get("choices") or [{}])[0].get("message") or {})
reply = msg_obj.get("content") or msg_obj.get("reasoning_content") or ""
ds9log(f"PUBLIC VERIFIED: {PUBLIC_URL}/v1 | quant={HW_PLAN['quant']} | mtp={mtp_active} | n_ctx={allocated} | reply={reply.strip()[:60]!r}")

ver_rec = {
    "service": SERVICE,
    "verified_at": time.strftime("%FT%T%z"),
    "base_url": PUBLIC_URL + "/v1",
    "repo": HW_PLAN.get("resolved_repo") or CFG["repo"],
    "quant": HW_PLAN["quant"],
    "alias": CFG["alias"],
    "mode": HW_PLAN["mode"],
    "cache_type": HW_PLAN["cache_type"],
    "context_allocated": allocated,
    "vision_enabled": vision_enabled,
    "mtp_enabled": mtp_active,
    "mtp_draft_max": HW_PLAN.get("mtp_draft_max"),
    "reply": reply[:120],
}
write_json_atomic(SERVICE_DIR / "last-verification.json", ver_rec)
if SERVICE != "deepseek-qwen-9b":
    write_json_atomic(SERVICE_DIR / f"last-verification-{SERVICE}.json", ver_rec)

print(f"{SERVICE} is up: {PUBLIC_URL} (quant={HW_PLAN['quant']}, mtp={mtp_active}, n_ctx={allocated})")
print("CT_READY", flush=True)

def ds9b_heartbeat(uptime):
    gpu_mem = ct_gpu_stat("memory.used,memory.total,utilization.gpu")
    status_rec = {
        "service": SERVICE,
        "kernel_pid": os.getpid(),
        "server_pid": PROCS["service"][0].pid,
        "tunnel_pid": PROCS[TUNNEL_PROCESS_KEY][0].pid,
        "quant": HW_PLAN["quant"],
        "mtp_enabled": mtp_active,
        "context": HW_PLAN["context"],
        "mode": HW_PLAN["mode"],
        "cache_type": HW_PLAN["cache_type"],
        "alias": CFG["alias"],
        "url": PUBLIC_URL + "/v1",
        "updated_at": time.strftime("%FT%T%z"),
        "uptime_sec": uptime,
        "gpu_memory": gpu_mem,
    }
    try:
        write_json_atomic(SERVICE_DIR / "last-status.json", status_rec)
        if SERVICE != "deepseek-qwen-9b":
            write_json_atomic(SERVICE_DIR / f"last-status-{SERVICE}.json", status_rec)
    except OSError:
        pass
    ds9log(f"heartbeat | up={uptime}s | quant={HW_PLAN['quant']} | mtp={mtp_active} | n_ctx={HW_PLAN['context']} | GPU: {gpu_mem}")

ct_watchdog(
    lambda: http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200,
    start_service,
    ds9log,
    log_name="llama-server",
    heartbeat=ds9b_heartbeat,
)
'''

# --------------------------------------------------------------------------- comfyui (Caddy auth + ComfyUI-Manager)
COMFYUI_CFG = dict(name="comfyui", service_port=8188, auth="caddy", public_port=18080,
                   health_path="/system_stats", ready_timeout=2400)
COMFYUI_SETUP = r'''# @title 1. Setup -- install ComfyUI + ComfyUI-Manager (models on fast local SSD, outputs/workflows on Drive)
from pathlib import Path
from urllib.parse import unquote

COMFY_DIR = "/content/ComfyUI"
COMFY_REPO = os.environ.get("CT_COMFY_REPO", "https://github.com/comfyanonymous/ComfyUI.git")
MANAGER_REPO = os.environ.get("CT_COMFY_MANAGER_REPO", "https://github.com/Comfy-Org/ComfyUI-Manager.git")
UI_MODE = os.environ.get("CT_COMFY_MANAGER_UI", "both")   # v4 | legacy | both (custom_nodes)

if not os.path.isdir(f"{COMFY_DIR}/.git"):
    subprocess.run(["git", "clone", "--depth", "1", COMFY_REPO, COMFY_DIR], check=True)
else:
    subprocess.run(["git", "-C", COMFY_DIR, "pull", "--ff-only"], check=False)

# Remove any legacy models stored on Google Drive so Drive is never bloated by multi-GB weights
if os.path.isdir(f"{DATA_DIR}/models") and not os.path.islink(f"{DATA_DIR}/models"):
    shutil.rmtree(f"{DATA_DIR}/models", ignore_errors=True)
if os.path.islink(f"{COMFY_DIR}/models"):
    os.unlink(f"{COMFY_DIR}/models")
os.makedirs(f"{COMFY_DIR}/models", exist_ok=True)

# Persist only lightweight user data (custom_nodes, output, input, user) on Google Drive
for sub in ("custom_nodes", "output", "input", "user"):
    drive_sub = f"{DATA_DIR}/{sub}"
    local_sub = f"{COMFY_DIR}/{sub}"
    os.makedirs(drive_sub, exist_ok=True)
    if os.path.islink(local_sub):
        if os.readlink(local_sub) != drive_sub:
            os.unlink(local_sub)
            os.symlink(drive_sub, local_sub)
    elif os.path.isdir(local_sub):
        subprocess.run(["cp", "-rn", f"{local_sub}/.", f"{drive_sub}/"], check=True)
        shutil.rmtree(local_sub)
        os.symlink(drive_sub, local_sub)
    else:
        os.symlink(drive_sub, local_sub)

# Install ComfyUI core requirements + ComfyUI-Manager + hf_xet for fast local SSD model downloads
req_marker = f"{WORK}/comfyui_deps.ok"
if not os.path.exists(req_marker):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_xet"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{COMFY_DIR}/requirements.txt"], check=True)
    if os.path.exists(f"{COMFY_DIR}/manager_requirements.txt"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{COMFY_DIR}/manager_requirements.txt"],
                       check=True)
    open(req_marker, "w").write("ok\n")

# Patch ComfyUI frontend so <script type="module">, <link rel="modulepreload">, and Vite's preload helper
# use crossorigin="use-credentials", and strip embedded URL credentials (https://user:pass@host) before
# Vue Router calls history.replaceState() (which throws SecurityError when user:pass@ is in location.href)
import glob, site
_URL_CRED_GUARD = ("<script>;(function(){if(location.username||location.password){"
                   "location.replace(location.origin+location.pathname+location.search+location.hash);"
                   "}})()</script>")
_SSD_UI_SCRIPT = '<script defer src="/api/ct/ssd-downloader.js"></script>'
for sp in site.getsitepackages():
    fe_dir = f"{sp}/comfyui_frontend_package/static"
    if not os.path.isdir(fe_dir):
        continue
    idx = f"{fe_dir}/index.html"
    if os.path.isfile(idx):
        html = open(idx, encoding="utf-8").read()
        patched = re.sub(r'\bcrossorigin(?:="")?(?!="use-credentials")(?=[\s>/])',
                         'crossorigin="use-credentials"', html)
        if _URL_CRED_GUARD not in patched:
            patched = patched.replace("<head>", "<head>" + _URL_CRED_GUARD, 1)
        if _SSD_UI_SCRIPT not in patched:
            patched = patched.replace("</body>", _SSD_UI_SCRIPT + "</body>", 1)
        if patched != html:
            open(idx, "w", encoding="utf-8").write(patched)
            print("Patched crossorigin, URL credential guard, and SSD UI button in", idx)
    for js_path in glob.glob(f"{fe_dir}/assets/*.js"):
        js = open(js_path, encoding="utf-8", errors="ignore").read()
        patched_js = js.replace('crossOrigin=``', 'crossOrigin=`use-credentials`').replace(
            'crossOrigin=""', 'crossOrigin="use-credentials"')
        if patched_js != js:
            open(js_path, "w", encoding="utf-8").write(patched_js)
            print("Patched crossOrigin in", js_path)

if UI_MODE in ("legacy", "both"):
    mgr_dir = f"{COMFY_DIR}/custom_nodes/ComfyUI-Manager"
    if not os.path.isdir(mgr_dir) or not os.listdir(mgr_dir):
        subprocess.run(["git", "clone", "--depth", "1", MANAGER_REPO, mgr_dir], check=True)
    if os.path.exists(f"{mgr_dir}/requirements.txt"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{mgr_dir}/requirements.txt"], check=True)

# Built-in custom nodes always provisioned before ComfyUI starts
BUILTIN_CUSTOM_NODES = [
    ("ComfyUI-MiniMaxH3-PrefixStream", "https://github.com/knoic/ComfyUI-MiniMaxH3-PrefixStream.git"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
]
for node_name, node_repo in BUILTIN_CUSTOM_NODES:
    # Manager may install a registry package using a lowercase directory name.
    # Loading both copies registers the same frontend twice and breaks VHSHelp.
    node_root = Path(COMFY_DIR) / "custom_nodes"
    for duplicate in sorted(node_root.iterdir()):
        if duplicate.name != node_name and duplicate.name.casefold() == node_name.casefold():
            backup_root = Path(DATA_DIR) / "custom_nodes-disabled"
            backup_root.mkdir(parents=True, exist_ok=True)
            import uuid
            backup = backup_root / f"{duplicate.name}-{uuid.uuid4().hex}"
            shutil.move(str(duplicate), str(backup))
            print(f"Archived duplicate custom node: {duplicate} -> {backup}")
    node_dir = f"{COMFY_DIR}/custom_nodes/{node_name}"
    if not os.path.isdir(f"{node_dir}/.git"):
        if os.path.isdir(node_dir):
            shutil.rmtree(node_dir)
        subprocess.run(["git", "clone", "--depth", "1", node_repo, node_dir], check=True)
    else:
        subprocess.run(["git", "-C", node_dir, "pull", "--ff-only"], check=False)
    if os.path.isfile(f"{node_dir}/requirements.txt"):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{node_dir}/requirements.txt"], check=True)

# Install requirements of any other persisted custom_nodes on Drive
for entry in sorted(os.listdir(f"{COMFY_DIR}/custom_nodes")):
    req = f"{COMFY_DIR}/custom_nodes/{entry}/requirements.txt"
    if entry not in ("ComfyUI-Manager", *(n for n, _ in BUILTIN_CUSTOM_NODES)) and os.path.isfile(req):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", req], check=False)

# --- On-demand workflow model downloader (downloads directly to local NVMe SSD /content/ComfyUI/models) ---
open("/content/ComfyUI/ct_workflow_helper.py", "w", encoding="utf-8").write(__COMFY_HELPER_SOURCE__)

sys.path.insert(0, "/content/ComfyUI")
from ct_workflow_helper import sync_workflow_models

for wf_spec in [x.strip() for x in os.environ.get("CT_WORKFLOW", "").split(",") if x.strip()]:
    sync_workflow_models(wf_spec, data_dir=DATA_DIR)

print("ComfyUI + ComfyUI-Manager setup complete; local SSD models =", f"{COMFY_DIR}/models", "drive data =", DATA_DIR)
'''.replace(
    "__COMFY_HELPER_SOURCE__", repr((ROOT / "templates/comfyui_workflow_helper.py").read_text()))
COMFYUI_SERVICE = r'''# @title 2. Service -- ComfyUI + ComfyUI-Manager on 127.0.0.1:8188 (with async /api/ct/pull-workflow + UI toolbar button)
open("/content/ComfyUI/ct_ssd_downloader.js", "w", encoding="utf-8").write(__COMFY_JS_SOURCE__)

os.makedirs("/content/ComfyUI/custom_nodes", exist_ok=True)
open("/content/ComfyUI/custom_nodes/ct_workflow_downloader.py", "w", encoding="utf-8").write(__COMFY_NODE_SOURCE__)

EXTRA_ARGS = os.environ.get("CT_COMFY_ARGS", "").split()

def start_service():
    subprocess.run(["pkill", "-f", "ComfyUI/main.py"], check=False)
    args = [sys.executable, f"{COMFY_DIR}/main.py",
            "--listen", "127.0.0.1", "--port", str(SERVICE_PORT),
            "--enable-manager", "--preview-method", "auto", *EXTRA_ARGS]
    if os.environ.get("CT_COMFY_LEGACY_UI", "1") == "1":
        args.append("--enable-manager-legacy-ui")
    return ct_spawn("service", args, cwd=COMFY_DIR)

PROCS["service"] = (start_service(), start_service)
ct_wait_http(f"http://127.0.0.1:{SERVICE_PORT}/system_stats", timeout=300)
print(f"ComfyUI listening on 127.0.0.1:{SERVICE_PORT}")
'''.replace(
    "__COMFY_JS_SOURCE__", repr((ROOT / "templates/comfyui_ssd_downloader.js").read_text())).replace(
    "__COMFY_NODE_SOURCE__", repr((ROOT / "templates/comfyui_workflow_downloader.py").read_text()))

# --------------------------------------------------------------------------- nemotron-transcribe (G4, vLLM + NeMo Speech SALM)
NEMOTRON_TRANSCRIBE_CFG = dict(
    name="nemotron-transcribe",
    service_port=8000,
    auth="builtin",
    public_port=8000,
    health_path="/health",
    ready_timeout=3600,
)

NEMOTRON_TRANSCRIBE_SETUP = r'''# @title 1. Setup -- parallel NeMo Speech + vLLM 0.32 env & local SSD model download (67.5 GB BF16 / FP8 / NVFP4)
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib, platform

PRECISION_REPOS = {
    "bf16": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B",
    "fp8": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8",
    "nvfp4": "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4",
}
PRECISION = os.environ.get("CT_NEMOTRON_PRECISION", "bf16").strip().lower()
MODEL_REPO = os.environ.get("CT_NEMOTRON_REPO", "").strip() or PRECISION_REPOS.get(PRECISION, PRECISION_REPOS["bf16"])
MODEL_ALIAS = os.environ.get("CT_NEMOTRON_ALIAS", "nemotron-transcribe").strip()
NEMO_REPO = os.environ.get("CT_NEMO_SPEECH_REPO", "https://github.com/NVIDIA-NeMo/Speech.git").strip()

NT_WORK = Path("/content/nemotron-transcribe")
NEMO_DIR = Path("/content/nemo-speech")
VENV_DIR = NEMO_DIR / ".venv"
MODEL_DIR = NT_WORK / "model" / MODEL_REPO.split("/")[-1]
DRIVE_AUDIO_DIR = Path(DATA_DIR) / "audio"
DRIVE_OUTPUT_DIR = Path(DATA_DIR) / "outputs"
FLASHINFER_DRIVE_DIR = Path(DATA_DIR) / "flashinfer-cache"
FLASHINFER_CACHE_DIR = NT_WORK / "flashinfer-cache"
TRITON_DRIVE_DIR = Path(DATA_DIR) / "triton-cache"
TRITON_CACHE_DIR = NT_WORK / "triton-cache"
VLLM_DRIVE_DIR = Path(DATA_DIR) / "vllm-cache"
VLLM_CACHE_DIR = NT_WORK / "vllm-cache"

for d in (
    NT_WORK, MODEL_DIR.parent, DRIVE_AUDIO_DIR, DRIVE_OUTPUT_DIR,
    FLASHINFER_DRIVE_DIR, FLASHINFER_CACHE_DIR,
    TRITON_DRIVE_DIR, TRITON_CACHE_DIR,
    VLLM_DRIVE_DIR, VLLM_CACHE_DIR,
):
    d.mkdir(parents=True, exist_ok=True)
def nlog(msg):
    print(time.strftime("[%F %T] [nemotron] ") + msg, flush=True)

# One archive per cache/environment/GPU avoids Drive's small-file latency.
(NT_WORK / "colab_cache.py").write_text(__NEMOTRON_CACHE_SOURCE__, encoding="utf-8")
sys.path.insert(0, str(NT_WORK))
from colab_cache import restore_cache, save_cache
CACHE_KEY = "vllm-0.32-py313-" + PRECISION + "-sm" + ct_gpu_stat("compute_cap").splitlines()[0].replace(".", "")
CACHE_ARCHIVES = Path(DATA_DIR) / "cache-archives"
CACHE_PAIRS = (
    (FLASHINFER_CACHE_DIR, FLASHINFER_DRIVE_DIR),
    (TRITON_CACHE_DIR, TRITON_DRIVE_DIR),
    (VLLM_CACHE_DIR, VLLM_DRIVE_DIR),
)
for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        restore_cache(_local_c, _legacy_c, CACHE_ARCHIVES, CACHE_KEY, nlog)
    except (OSError, ValueError) as err:
        nlog(f"Cache restore warning ({_local_c.name}): {err}; continuing with local cache")

def run_logged(cmd, log_path, env=None, cwd=None, check=True):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(str(x) for x in cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True)
    if check and res.returncode != 0:
        tail = "".join(open(log_path, errors="replace").readlines()[-40:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(str(x) for x in cmd[:4])}\n{tail}")
    return res

gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], text=True
).strip()
nlog(f"GPU detected: {gpu_info} | repo={MODEL_REPO} precision={PRECISION}")

if not shutil.which("ffmpeg"):
    subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1", shell=True, check=False)

if not shutil.which("yt-dlp"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "yt-dlp"], check=False)

if not shutil.which("uv"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)

def prepare_env():
    env_log = NT_WORK / "env-setup.log"
    marker = VENV_DIR / ".ct_env_ready"
    vllm_bin = VENV_DIR / "bin" / "vllm"
    py_bin = VENV_DIR / "bin" / "python"
    if marker.is_file() and vllm_bin.is_file() and py_bin.is_file():
        nlog("Reusing existing NeMo Speech + vLLM 0.32 virtualenv")
        return VENV_DIR

    nlog(f"Setting up NeMo Speech + vLLM 0.32 environment (log: {env_log})...")
    if not (NEMO_DIR / ".git").is_dir():
        run_logged(["git", "clone", "--depth", "1", NEMO_REPO, str(NEMO_DIR)], env_log)
    else:
        run_logged(["git", "-C", str(NEMO_DIR), "pull", "--ff-only"], env_log, check=False)

    uv_bin = shutil.which("uv") or "uv"
    run_logged([uv_bin, "venv", "--python", "3.13", str(VENV_DIR), "--clear"], env_log, cwd=str(NEMO_DIR))
    sync_env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(VENV_DIR)}
    res = run_logged(
        [uv_bin, "sync", "--extra", "asr", "--extra", "vllm", "--locked", "--no-install-project"],
        env_log, env=sync_env, cwd=str(NEMO_DIR), check=False
    )
    if res.returncode != 0:
        nlog("`uv sync --locked` exited non-zero; retrying without `--locked`...")
        run_logged(
            [uv_bin, "sync", "--extra", "asr", "--extra", "vllm", "--no-install-project"],
            env_log, env=sync_env, cwd=str(NEMO_DIR)
        )
    run_logged([uv_bin, "pip", "install", "--python", str(py_bin), "--no-deps", "-e", str(NEMO_DIR)], env_log)
    run_logged(
        [uv_bin, "pip", "install", "--python", str(py_bin), "--no-deps", "av==19.0.2", "opencv-python-headless==5.0.0.93"],
        env_log,
    )
    run_logged([uv_bin, "pip", "install", "--python", str(py_bin), "-U", "yt-dlp"], env_log, check=False)

    # Apply official NVIDIA DFlash2 safety patch to vLLM 0.32 if applicable
    patch_code = r"""
from pathlib import Path
import vllm
path = Path(vllm.__file__).parent / "model_executor/models/qwen3_dflash2.py"
if path.is_file():
    source = path.read_text()
    old = "    successors = successor_table[candidate_ids]\n"
    anchor = "            anchor_token_ids[:, None, None].expand(-1, 1, top_k),"
    if "safe_anchor_token_ids" not in source and source.count(old) == 1 and source.count(anchor) == 1:
        source = source.replace(old, (
            "    safe_anchor_token_ids = anchor_token_ids.clamp(\n"
            "        min=0, max=predecessor_table.shape[0] - 1\n"
            "    )\n" + old
        )).replace(anchor, "            safe_anchor_token_ids[:, None, None].expand(-1, 1, top_k),")
        path.write_text(source)
"""
    run_logged([str(py_bin), "-c", patch_code], env_log, check=False)
    run_logged([str(vllm_bin), "download-kernels"], env_log, env=sync_env, check=False)
    marker.write_text("ok\n", encoding="utf-8")
    nlog("NeMo Speech + vLLM 0.32 environment ready")
    return VENV_DIR

def _cleanup_hf_temp():
    for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub", str(MODEL_DIR / ".cache")):
        shutil.rmtree(p, ignore_errors=True)

def prepare_model():
    dl_log = NT_WORK / "model-download.log"
    marker = MODEL_DIR / ".ct_model_ready"
    has_weights = any(MODEL_DIR.glob("*.safetensors")) if MODEL_DIR.is_dir() else False
    if marker.is_file() and (MODEL_DIR / "config.json").is_file() and has_weights:
        nlog(f"Reusing local SSD model at {MODEL_DIR}")
        return MODEL_DIR

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_xet"], check=True)
    _cleanup_hf_temp()
    free_gb = shutil.disk_usage("/content").free / 1e9
    nlog(f"Downloading {MODEL_REPO} -> local SSD {MODEL_DIR} (free SSD: {free_gb:.1f} GB; log: {dl_log})...")
    dl_env = {**os.environ, "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    dl_env.pop("HF_XET_HIGH_PERFORMANCE", None)
    dl_code = (
        "import sys; from huggingface_hub import snapshot_download; "
        "snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])"
    )
    try:
        run_logged([sys.executable, "-c", dl_code, MODEL_REPO, str(MODEL_DIR)], dl_log, env=dl_env)
    finally:
        _cleanup_hf_temp()

    if not (MODEL_DIR / "config.json").is_file() or not any(MODEL_DIR.glob("*.safetensors")):
        raise RuntimeError(f"Model download incomplete in {MODEL_DIR}")
    marker.write_text(MODEL_REPO + "\n", encoding="utf-8")
    total_gb = sum(p.stat().st_size for p in MODEL_DIR.rglob("*") if p.is_file()) / 1e9
    nlog(f"Model ready on local SSD: {MODEL_DIR} ({total_gb:.2f} GB)")
    return MODEL_DIR

with ThreadPoolExecutor(max_workers=2) as pool:
    f_env = pool.submit(prepare_env)
    f_mod = pool.submit(prepare_model)
    while not (f_env.done() and f_mod.done()):
        for fut in (f_env, f_mod):
            if fut.done() and fut.exception():
                raise fut.exception()
        free_gb = shutil.disk_usage("/content").free / 1e9
        nlog(f"Setup heartbeat | env={'done' if f_env.done() else 'running'} "
             f"model={'done' if f_mod.done() else 'running'} | free_ssd={free_gb:.1f} GB")
        time.sleep(30)
    VENV_DIR = f_env.result()
    MODEL_DIR = f_mod.result()

nlog(f"Setup complete: venv={VENV_DIR} model={MODEL_DIR} drive_data={DATA_DIR}")
'''.replace("__NEMOTRON_CACHE_SOURCE__", repr((ROOT / "templates/colab_cache.py").read_text()))

NEMOTRON_TRANSCRIBE_SERVICE = r'''# @title 2. Service -- vLLM + NeMo Speech SALM on 127.0.0.1:8000 with Bearer Auth + Google Drive Audio API
(NT_WORK / "colab_asgi_common.py").write_text(__ASGI_COMMON_SOURCE__, encoding="utf-8")
(NT_WORK / "nemotron_audio.py").write_text(__NEMOTRON_AUDIO_SOURCE__, encoding="utf-8")
(NT_WORK / "ct_nemotron_middleware.py").write_text(__NEMOTRON_MW_SOURCE__, encoding="utf-8")

GPU_MEM_UTIL = os.environ.get("CT_NEMOTRON_GPU_MEM", "0.92" if PRECISION == "bf16" else "0.85")
MAX_MODEL_LEN = int(os.environ.get("CT_NEMOTRON_MAX_MODEL_LEN", "65536"))
MAX_NUM_SEQS = int(os.environ.get("CT_NEMOTRON_MAX_SEQS", "64"))
MTP_TOKENS = int(os.environ.get("CT_NEMOTRON_MTP", "8"))
LIMIT_MM = os.environ.get(
    "CT_NEMOTRON_LIMIT_MM",
    '{"audio":{"count":6,"length":120000000}}' if PRECISION == "bf16" else '{"audio":{"count":1,"length":57600000}}',
)

VLLM_ENV = {
    "PATH": f"{VENV_DIR}/bin:/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
    "PYTHONPATH": f"{NT_WORK}:{os.environ.get('PYTHONPATH', '')}",
    "VIRTUAL_ENV": str(VENV_DIR),
    "VLLM_PLUGINS": "nemo_speechlm",
    "VLLM_MAX_AUDIO_DECODE_DURATION_S": "3600",
    "VLLM_MAX_AUDIO_CLIP_FILESIZE_MB": "180",
    "VLLM_USE_V2_MODEL_RUNNER": "0",
    "FLASHINFER_WORKSPACE_BASE": str(FLASHINFER_CACHE_DIR),
    "TRITON_CACHE_DIR": str(TRITON_CACHE_DIR),
    "VLLM_CACHE_ROOT": str(VLLM_CACHE_DIR),
    "TORCHINDUCTOR_CACHE_DIR": str(VLLM_CACHE_DIR / "inductor"),
    "CT_API_TOKEN": API_TOKEN,
    "CT_SERVICE_PORT": str(SERVICE_PORT),
    "CT_DATA_DIR": str(DATA_DIR),
    "CT_NEMOTRON_REPO": MODEL_REPO,
}
if PRECISION in ("fp8", "nvfp4"):
    VLLM_ENV["VLLM_DISABLED_KERNELS"] = "FlashInferFP8ScaledMMLinearKernel"

SERVER_ARGS = [
    str(VENV_DIR / "bin" / "vllm"), "serve", str(MODEL_DIR),
    "--served-model-name", MODEL_REPO, MODEL_ALIAS,
    "--host", "127.0.0.1",
    "--port", str(SERVICE_PORT),
    "--api-key", API_TOKEN,
    "--trust-remote-code",
    "--dtype", "bfloat16",
    "--gpu-memory-utilization", str(GPU_MEM_UTIL),
    "--max-model-len", str(MAX_MODEL_LEN),
    "--max-num-batched-tokens", "4096",
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--enable-chunked-prefill",
    "--no-enable-prefix-caching",
    "--use-replayssm",
    "--mamba-backend", "flashinfer",
    "--mamba-cache-mode", "none",
    "--mm-processor-cache-gb", "0",
    "--no-async-scheduling",
    "--limit-mm-per-prompt", LIMIT_MM,
    "--reasoning-parser", "nemotron_v3",
    "--trust-request-mm-kwargs",
    "--middleware", "ct_nemotron_middleware.nemotron_middleware",
]
if PRECISION in ("fp8", "nvfp4"):
    SERVER_ARGS += ["--kv-cache-dtype", "auto"]
if MTP_TOKENS > 0:
    SERVER_ARGS += [
        "--speculative-config",
        json.dumps({"method": "mtp", "num_speculative_tokens": MTP_TOKENS}),
    ]

def start_service():
    subprocess.run(["pkill", "-f", "vllm serve"], check=False)
    nlog(f"Starting vLLM server on 127.0.0.1:{SERVICE_PORT} (model={MODEL_REPO}, mtp={MTP_TOKENS}, ctx={MAX_MODEL_LEN})")
    return ct_spawn("service", SERVER_ARGS, env=VLLM_ENV, cwd=str(NT_WORK))

PROCS["service"] = (start_service(), start_service)
'''.replace(
    "__ASGI_COMMON_SOURCE__", repr((ROOT / "templates/colab_asgi_common.py").read_text())).replace(
    "__NEMOTRON_AUDIO_SOURCE__", repr((ROOT / "src/colabllm/lib/nemotron_audio.py").read_text())).replace(
    "__NEMOTRON_MW_SOURCE__", repr((ROOT / "templates/nemotron_middleware.py").read_text()))

NEMOTRON_TRANSCRIBE_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- wait for model load/FlashInfer warmup, verify Bearer auth, supervise
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def http_json(url, key=None, payload=None, timeout=20):
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, headers=headers, data=None if payload is None else json.dumps(payload).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else {})
    except urllib.error.HTTPError as err:
        return err.code, {}

def auth_check(base, key):
    for candidate, expected in ((None, 401), ("invalid-token-probe", 401), (key, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Auth test failed on {base}/v1/models: expected {expected}, got {code}")

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
deadline = time.monotonic() + 2400
last_notice = 0
while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        raise RuntimeError("vLLM exited during startup:\n" + ct_tail("service", 60))
    try:
        code, _ = http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)
        if code == 200:
            break
    except Exception:
        pass
    if time.monotonic() - last_notice >= 30:
        gpu_stat = ct_gpu_stat()
        last_line = (ct_tail("service", 3).strip().splitlines() or ["starting..."])[-1]
        nlog(f"Waiting for vLLM load / FlashInfer kernel warmup | GPU: {gpu_stat} | {last_line[:100]}")
        last_notice = time.monotonic()
    time.sleep(5)
else:
    raise RuntimeError("Timed out waiting for vLLM server to become ready:\n" + ct_tail("service", 60))

auth_check(LOCAL_BASE, API_TOKEN)
# Exercise the audio path once so lazily compiled kernels exist before archiving caches.
import base64 as _b64, io as _io, wave as _wave
_buf = _io.BytesIO()
with _wave.open(_buf, "wb") as _wf:
    _wf.setnchannels(1); _wf.setsampwidth(2); _wf.setframerate(16000); _wf.writeframes(b"\x00\x00" * 16000)
_wcode, _ = http_json(LOCAL_BASE + "/v1/chat/completions", API_TOKEN, {
    "model": MODEL_ALIAS,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Transcribe this audio."},
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64," + _b64.b64encode(_buf.getvalue()).decode()}},
    ]}],
    "max_tokens": 16, "temperature": 0,
}, timeout=600)
nlog(f"Warmup transcription: HTTP {_wcode}")
for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        save_cache(_local_c, CACHE_ARCHIVES, CACHE_KEY, nlog)
    except (OSError, ValueError) as err:
        nlog(f"Cache archive save warning ({_local_c.name}): {err}")
code, presets_info = http_json(LOCAL_BASE + "/api/ct/prompts", API_TOKEN)
if code != 200:
    raise RuntimeError(f"Local /api/ct/prompts check failed: HTTP {code}")
nlog(f"Local endpoint verified: /v1/models (401/401/200) & /api/ct/prompts ({ len(presets_info.get('presets', {})) } presets)")

ct_public_gate(PUBLIC_URL, lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
               lambda: auth_check(PUBLIC_URL, API_TOKEN), nlog, timeout=120, interval=8)
nlog(f"Public endpoint verified: {PUBLIC_URL}/v1/models")

print(f"{SERVICE} is up: {PUBLIC_URL}")
print("CT_READY", flush=True)

ct_watchdog(lambda: http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200, start_service, nlog, dead_markers=(),
            heartbeat=lambda up: nlog(f"heartbeat | up={up}s | GPU: {ct_gpu_stat()}"))
'''

# ---------------------------------------------------------------------------
# IndexTTS-2.5 (L4 GPU / vLLM-Omni 2-stage zero-shot voice cloning & emotion TTS)
# ---------------------------------------------------------------------------

INDEXTTS_CFG = dict(
    name="indextts",
    service_port=8000,
    auth="builtin",
    public_port=8000,
    health_path="/health",
    ready_timeout=3600,
)

INDEXTTS_SETUP = r'''# @title 1. Setup -- install vLLM-Omni + pre-stage IndexTTS-2.5 & sub-models on local SSD
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_REPO = os.environ.get("CT_INDEXTTS_MODEL", "IndexTeam/IndexTTS-2.5").strip() or "IndexTeam/IndexTTS-2.5"
MODEL_ALIAS = os.environ.get("CT_INDEXTTS_ALIAS", "indextts-2.5").strip() or "indextts-2.5"
STAGE0_GPU_MEM = float(os.environ.get("CT_INDEXTTS_GPU_MEM_STAGE0", "0.4"))
STAGE1_GPU_MEM = float(os.environ.get("CT_INDEXTTS_GPU_MEM_STAGE1", "0.4"))
MAX_MODEL_LEN = int(os.environ.get("CT_INDEXTTS_MAX_MODEL_LEN", "2560"))
MAX_NUM_SEQS = int(os.environ.get("CT_INDEXTTS_MAX_NUM_SEQS", "4"))

IT_WORK = Path("/content/indextts")
VENV_DIR = IT_WORK / "venv"
MODEL_DIR = IT_WORK / "model" / "IndexTTS-2.5"
SPEAKERS_DIR = IT_WORK / "speakers"
DEFAULT_REF_WAV = IT_WORK / "default_ref.wav"

DRIVE_VOICES_DIR = Path(DATA_DIR) / "voices"
DRIVE_OUTPUT_DIR = Path(DATA_DIR) / "outputs"
FLASHINFER_DRIVE_DIR = Path(DATA_DIR) / "flashinfer-cache"
FLASHINFER_CACHE_DIR = IT_WORK / "flashinfer-cache"
TRITON_DRIVE_DIR = Path(DATA_DIR) / "triton-cache"
TRITON_CACHE_DIR = IT_WORK / "triton-cache"
VLLM_DRIVE_DIR = Path(DATA_DIR) / "vllm-cache"
VLLM_CACHE_DIR = IT_WORK / "vllm-cache"

for d in (
    IT_WORK, MODEL_DIR, SPEAKERS_DIR,
    DRIVE_VOICES_DIR, DRIVE_OUTPUT_DIR,
    FLASHINFER_DRIVE_DIR, FLASHINFER_CACHE_DIR,
    TRITON_DRIVE_DIR, TRITON_CACHE_DIR,
    VLLM_DRIVE_DIR, VLLM_CACHE_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

def ilog(msg):
    print(time.strftime("[%F %T] [indextts] ") + msg, flush=True)

# One archive per cache/environment/GPU avoids Drive's small-file latency.
(IT_WORK / "colab_cache.py").write_text(__INDEXTTS_CACHE_SOURCE__, encoding="utf-8")
sys.path.insert(0, str(IT_WORK))
from colab_cache import restore_cache, save_cache
_gpu_arch = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True).strip().splitlines()[0]
CACHE_KEY = "vllm-0.32.0-py312-sm" + _gpu_arch.replace(".", "")
CACHE_ARCHIVES = Path(DATA_DIR) / "cache-archives"
CACHE_PAIRS = (
    (FLASHINFER_CACHE_DIR, FLASHINFER_DRIVE_DIR),
    (TRITON_CACHE_DIR, TRITON_DRIVE_DIR),
    (VLLM_CACHE_DIR, VLLM_DRIVE_DIR),
)
for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        restore_cache(_local_c, _legacy_c, CACHE_ARCHIVES, CACHE_KEY, ilog)
    except (OSError, ValueError) as err:
        ilog(f"Cache restore warning ({_local_c.name}): {err}; continuing with local cache")
ilog("Restoring registered voice samples from Drive")
subprocess.run(["cp", "-rn", f"{DRIVE_VOICES_DIR}/.", f"{SPEAKERS_DIR}/"], check=False)
ilog("Cache/voice restore finished; preparing environment and model")

def run_logged(cmd, log_path, env=None, cwd=None, check=True):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(str(x) for x in cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True)
    if check and res.returncode != 0:
        tail = "".join(open(log_path, errors="replace").readlines()[-45:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(str(x) for x in cmd[:4])}\n{tail}")
    return res

gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], text=True
).strip()
ilog(f"GPU detected: {gpu_info} | repo={MODEL_REPO} stage0_mem={STAGE0_GPU_MEM} stage1_mem={STAGE1_GPU_MEM}")

if not shutil.which("ffmpeg"):
    subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1", shell=True, check=False)

if not shutil.which("uv"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)

def prepare_env():
    env_log = IT_WORK / "env-setup.log"
    marker = VENV_DIR / ".ct_env_ready"
    vllm_bin = VENV_DIR / "bin" / "vllm"
    py_bin = VENV_DIR / "bin" / "python"
    if marker.is_file() and vllm_bin.is_file() and py_bin.is_file():
        chk = subprocess.run(
            [str(py_bin), "-c", "import vllm, vllm_omni, tiktoken, fugashi"],
            capture_output=True,
        )
        if chk.returncode == 0:
            ilog("Reusing existing vLLM + vLLM-Omni[indextts2] virtualenv")
            return VENV_DIR

    ilog(f"Setting up vLLM 0.32 + vLLM-Omni[indextts2] environment (log: {env_log})...")
    uv_bin = shutil.which("uv") or "uv"
    run_logged([uv_bin, "venv", "--python", "3.12", str(VENV_DIR), "--clear"], env_log)
    run_logged(
        [
            uv_bin, "pip", "install",
            "--python", str(py_bin),
            "vllm==0.32.0",
            "vllm-omni[indextts2]==0.32.0",
            "tiktoken",
            "soundfile",
        ],
        env_log,
    )
    subprocess.run([uv_bin, "cache", "clean"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_logged([str(vllm_bin), "download-kernels"], env_log, check=False)
    marker.write_text("ok\n", encoding="utf-8")
    ilog("vLLM 0.32 + vLLM-Omni[indextts2] environment ready")
    return VENV_DIR

def _cleanup_hf_temp():
    for p in (
        "/root/.cache/huggingface/xet",
        "/root/.cache/huggingface/hub",
        str(MODEL_DIR / ".cache"),
        str(MODEL_DIR / "w2v-bert-2.0" / ".cache"),
        str(MODEL_DIR / "bigvgan" / ".cache"),
    ):
        shutil.rmtree(p, ignore_errors=True)

def _ensure_default_ref_wav():
    if DEFAULT_REF_WAV.is_file() and DEFAULT_REF_WAV.stat().st_size > 1024:
        return
    drive_copy = DRIVE_VOICES_DIR / "default_ref.wav"
    if drive_copy.is_file() and drive_copy.stat().st_size > 1024:
        shutil.copyfile(drive_copy, DEFAULT_REF_WAV)
        return
    ref_url = "https://raw.githubusercontent.com/vllm-project/vllm-omni/main/tests/assets/indextts2/ref_audio.wav"
    try:
        with urllib.request.urlopen(ref_url, timeout=60) as _src, open(DEFAULT_REF_WAV, "wb") as _dst:
            shutil.copyfileobj(_src, _dst)
    except Exception:
        import math, struct, wave
        sr = 24000
        dur = 3.0
        n = int(sr * dur)
        with wave.open(str(DEFAULT_REF_WAV), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            samples = [
                int(12000 * (0.6 * math.sin(2 * math.pi * 180 * i / sr) + 0.4 * math.sin(2 * math.pi * 360 * i / sr)))
                for i in range(n)
            ]
            wf.writeframes(struct.pack(f"<{n}h", *samples))
    if not drive_copy.is_file():
        shutil.copyfile(DEFAULT_REF_WAV, drive_copy)

def prepare_model():
    dl_log = IT_WORK / "model-download.log"
    marker = MODEL_DIR / ".ct_model_ready"
    required_files = (
        MODEL_DIR / "gpt.pth",
        MODEL_DIR / "s2mel.pth",
        MODEL_DIR / "codec.pth",
        MODEL_DIR / "multilingual_zh_ja_yue_char_del.tiktoken",
        MODEL_DIR / "qwen0.6bemo4-merge" / "model.safetensors",
        MODEL_DIR / "w2v-bert-2.0" / "model.safetensors",
        MODEL_DIR / "campplus_cn_common.bin",
        MODEL_DIR / "bigvgan" / "bigvgan_generator.pt",
    )
    _ensure_default_ref_wav()
    if marker.is_file() and all(p.is_file() for p in required_files):
        ilog(f"Reusing local SSD IndexTTS-2.5 bundle at {MODEL_DIR}")
        return MODEL_DIR

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_xet"], check=True)
    _cleanup_hf_temp()
    free_gb = shutil.disk_usage("/content").free / 1e9
    ilog(f"Downloading {MODEL_REPO} + sub-models -> local SSD {MODEL_DIR} (free SSD: {free_gb:.1f} GB; log: {dl_log})...")
    dl_env = {**os.environ, "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    dl_env.pop("HF_XET_HIGH_PERFORMANCE", None)
    dl_script = r"""
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

model_repo = sys.argv[1]
model_dir = Path(sys.argv[2])
model_dir.mkdir(parents=True, exist_ok=True)

def clean():
    for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub", str(model_dir / ".cache")):
        shutil.rmtree(p, ignore_errors=True)

print(f"[dl] 1/4 Downloading {model_repo} -> {model_dir}", flush=True)
snapshot_download(repo_id=model_repo, local_dir=str(model_dir), max_workers=4)
clean()

w2v_dir = model_dir / "w2v-bert-2.0"
print(f"[dl] 2/4 Downloading facebook/w2v-bert-2.0 -> {w2v_dir}", flush=True)
snapshot_download(
    repo_id="facebook/w2v-bert-2.0",
    local_dir=str(w2v_dir),
    allow_patterns=["config.json", "preprocessor_config.json", "model.safetensors"],
    max_workers=4,
)
clean()

print(f"[dl] 3/4 Downloading funasr/campplus -> {model_dir / 'campplus_cn_common.bin'}", flush=True)
hf_hub_download(repo_id="funasr/campplus", filename="campplus_cn_common.bin", local_dir=str(model_dir))
clean()

bigvgan_dir = model_dir / "bigvgan"
print(f"[dl] 4/4 Downloading nvidia/bigvgan_v2_22khz_80band_256x -> {bigvgan_dir}", flush=True)
snapshot_download(
    repo_id="nvidia/bigvgan_v2_22khz_80band_256x",
    local_dir=str(bigvgan_dir),
    allow_patterns=["config.json", "bigvgan_generator.pt"],
    max_workers=4,
)
clean()
"""
    try:
        run_logged([sys.executable, "-c", dl_script, MODEL_REPO, str(MODEL_DIR)], dl_log, env=dl_env)
    finally:
        _cleanup_hf_temp()

    missing = [str(p) for p in required_files if not p.is_file()]
    if missing:
        raise RuntimeError(f"Model download incomplete; missing files: {missing}")
    marker.write_text(MODEL_REPO + "\n", encoding="utf-8")
    total_gb = sum(p.stat().st_size for p in MODEL_DIR.rglob("*") if p.is_file()) / 1e9
    ilog(f"IndexTTS-2.5 + sub-models ready on local SSD: {MODEL_DIR} ({total_gb:.2f} GB)")
    return MODEL_DIR

with ThreadPoolExecutor(max_workers=2) as pool:
    f_env = pool.submit(prepare_env)
    f_mod = pool.submit(prepare_model)
    while not (f_env.done() and f_mod.done()):
        for fut in (f_env, f_mod):
            if fut.done() and fut.exception():
                raise fut.exception()
        free_gb = shutil.disk_usage("/content").free / 1e9
        ilog(f"Setup heartbeat | env={'done' if f_env.done() else 'running'} "
             f"model={'done' if f_mod.done() else 'running'} | free_ssd={free_gb:.1f} GB")
        time.sleep(30)
    VENV_DIR = f_env.result()
    MODEL_DIR = f_mod.result()

ilog(f"Setup complete: venv={VENV_DIR} model={MODEL_DIR} drive_data={DATA_DIR}")
'''.replace("__INDEXTTS_CACHE_SOURCE__", repr((ROOT / 'templates/colab_cache.py').read_text()))

INDEXTTS_SERVICE = r'''# @title 2. Service -- vLLM-Omni IndexTTS-2.5 on 127.0.0.1:8000 (L4 2-stage pipeline + Bearer Auth + Google Drive Voices)
DEPLOY_YAML = IT_WORK / "indextts2_5.yaml"
DEPLOY_YAML.write_text(f"""pipeline: indextts2_5
async_chunk: false

stages:
  - stage_id: 0
    devices: "0"
    attention_backend: TRITON_ATTN
    max_num_seqs: {MAX_NUM_SEQS}
    max_model_len: {MAX_MODEL_LEN}
    max_num_batched_tokens: {MAX_MODEL_LEN}
    gpu_memory_utilization: {STAGE0_GPU_MEM}
    enable_prefix_caching: false
    enable_chunked_prefill: false
    hf_overrides:
      use_gpt_latent: false
    default_sampling_params:
      temperature: 0.8
      top_p: 0.8
      top_k: 30
      max_tokens: 1815
      repetition_penalty: 10.0
      seed: 42

  - stage_id: 1
    devices: "0"
    max_num_seqs: {MAX_NUM_SEQS}
    gpu_memory_utilization: {STAGE1_GPU_MEM}
    enforce_eager: true
    hf_overrides:
      use_gpt_latent: false
      s2mel_dit_bf16: true
      s2mel_cfm_batch_size: 4
      s2mel_dit_torch_compile: false
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 1815
      seed: 42
""", encoding="utf-8")

(IT_WORK / "colab_asgi_common.py").write_text(__ASGI_COMMON_SOURCE__, encoding="utf-8")
(IT_WORK / "ct_indextts_middleware.py").write_text(__INDEXTTS_MW_SOURCE__, encoding="utf-8")

VLLM_ENV = {
    "PATH": f"{VENV_DIR}/bin:/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
    "PYTHONPATH": f"{IT_WORK}:{os.environ.get('PYTHONPATH', '')}",
    "VIRTUAL_ENV": str(VENV_DIR),
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "FLASHINFER_WORKSPACE_BASE": str(FLASHINFER_CACHE_DIR),
    "TRITON_CACHE_DIR": str(TRITON_CACHE_DIR),
    "VLLM_CACHE_ROOT": str(VLLM_CACHE_DIR),
    "TORCHINDUCTOR_CACHE_DIR": str(VLLM_CACHE_DIR / "inductor"),
    "SPEAKER_SAMPLES_DIR": str(SPEAKERS_DIR),
    "CT_API_TOKEN": API_TOKEN,
    "CT_SERVICE_PORT": str(SERVICE_PORT),
    "CT_DATA_DIR": str(DATA_DIR),
    "CT_DEFAULT_REF_WAV": str(DEFAULT_REF_WAV),
}

SERVER_ARGS = [
    str(VENV_DIR / "bin" / "vllm"), "serve", str(MODEL_DIR),
    "--omni",
    "--served-model-name", MODEL_REPO, MODEL_ALIAS,
    "--host", "127.0.0.1",
    "--port", str(SERVICE_PORT),
    "--api-key", API_TOKEN,
    "--trust-remote-code",
    "--allowed-local-media-path", str(IT_WORK),
    "--stage-init-timeout", "900",
    "--deploy-config", str(DEPLOY_YAML),
    "--middleware", "ct_indextts_middleware.IndexTTSMiddleware",
]

def start_service():
    subprocess.run(["pkill", "-9", "-f", "vllm|EngineCore"], check=False)
    time.sleep(1)
    ilog(f"Starting vLLM-Omni IndexTTS-2.5 server on 127.0.0.1:{SERVICE_PORT} (model={MODEL_REPO})")
    return ct_spawn("service", SERVER_ARGS, env=VLLM_ENV, cwd=str(IT_WORK))

PROCS["service"] = (start_service(), start_service)
'''.replace(
    "__ASGI_COMMON_SOURCE__", repr((ROOT / "templates/colab_asgi_common.py").read_text())).replace(
    "__INDEXTTS_MW_SOURCE__", repr((ROOT / "templates/indextts_middleware.py").read_text()))

INDEXTTS_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- wait for 2-stage load, register default voice, warmup synthesis & sync caches
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def http_json(url, key=None, payload=None, timeout=30):
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, headers=headers, data=None if payload is None else json.dumps(payload).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else {})
    except urllib.error.HTTPError as err:
        return err.code, {}

def auth_check(base, key):
    for candidate, expected in ((None, 401), ("invalid-token-probe", 401), (key, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Auth test failed on {base}/v1/models: expected {expected}, got {code}")

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
deadline = time.monotonic() + 2400
last_notice = 0
while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        raise RuntimeError("vLLM-Omni exited during startup:\n" + ct_tail("service", 80))
    try:
        code, _ = http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)
        if code == 200:
            break
    except Exception:
        pass
    if time.monotonic() - last_notice >= 30:
        gpu_stat = ct_gpu_stat()
        last_line = (ct_tail("service", 3).strip().splitlines() or ["starting..."])[-1]
        ilog(f"Waiting for 2-stage IndexTTS-2.5 init | GPU: {gpu_stat} | {last_line[:100]}")
        last_notice = time.monotonic()
    time.sleep(5)
else:
    raise RuntimeError("Timed out waiting for vLLM-Omni server to become ready:\n" + ct_tail("service", 80))

auth_check(LOCAL_BASE, API_TOKEN)

# Register 'alloy' voice preset from DEFAULT_REF_WAV if not already registered
code, voices_data = http_json(LOCAL_BASE + "/v1/audio/voices", API_TOKEN, timeout=20)
existing_voices = set(voices_data.get("voices") or [])
for preset_voice in ("alloy",):
    if preset_voice not in existing_voices and DEFAULT_REF_WAV.is_file():
        v_code, v_res = http_json(
            LOCAL_BASE + "/api/ct/drive-upload-voice",
            API_TOKEN,
            payload={
                "name": preset_voice,
                "use_default_ref": True,
                "consent": "builtin-default-reference",
                "speaker_description": "Default neutral reference speaker for IndexTTS-2.5",
            },
            timeout=60,
        )
        ilog(f"Registered preset voice '{preset_voice}': HTTP {v_code}")

# Warmup synthesis to verify Stage 0 Talker + Stage 1 S2Mel DiT / BigVGAN and sync JIT caches to Drive
warmup_req = urllib.request.Request(
    LOCAL_BASE + "/v1/audio/speech",
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_TOKEN}"},
    data=json.dumps({
        "model": MODEL_REPO,
        "input": "Hello, IndexTTS is ready.",
        "voice": "default",
        "response_format": "wav",
        "extra_params": {"lang": "en"},
    }).encode("utf-8"),
)
try:
    with urllib.request.urlopen(warmup_req, timeout=180) as wresp:
        wav_bytes = wresp.read()
        ilog(f"Warmup synthesis succeeded ({len(wav_bytes)} bytes WAV)")
except Exception as werr:
    ilog(f"Warmup synthesis warning: {werr}")

for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        save_cache(_local_c, CACHE_ARCHIVES, CACHE_KEY, ilog)
    except (OSError, ValueError) as err:
        ilog(f"Cache archive save warning ({_local_c.name}): {err}")
# User voice samples remain ordinary Drive files, not compiler-cache archives.
subprocess.run(["cp", "-ru", f"{SPEAKERS_DIR}/.", f"{DRIVE_VOICES_DIR}/"], check=False)

ct_public_gate(PUBLIC_URL, lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
               lambda: auth_check(PUBLIC_URL, API_TOKEN), ilog, timeout=120, interval=8)
ilog(f"Public endpoint verified: {PUBLIC_URL}/v1/models")

print(f"{SERVICE} is up: {PUBLIC_URL}")
print("CT_READY", flush=True)

ct_watchdog(lambda: http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200, start_service, ilog, dead_markers=('replica-0 is dead',),
            heartbeat=lambda up: ilog(f"heartbeat | up={up}s | GPU: {ct_gpu_stat()}"))
'''

# --------------------------------------------------------------------------- qwen3-tts (L4, vLLM-Omni 2-stage Qwen3-TTS CustomVoice / VoiceDesign / Base Voice Clone)
QWEN3_TTS_CFG = dict(
    name="qwen3-tts",
    service_port=8000,
    auth="builtin",
    public_port=8000,
    health_path="/health",
    ready_timeout=3600,
)

QWEN3_TTS_SETUP = r'''# @title 1. Setup -- install vLLM-Omni + Qwen3-TTS & pre-stage 12Hz models on local SSD
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VARIANT_REPOS = {
    "customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "1.7b-customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voicedesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "1.7b-voicedesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "1.7b-base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "0.6b-customvoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "0.6b-base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
VARIANT = os.environ.get("CT_QWEN3_TTS_VARIANT", "customvoice").strip().lower()
MODEL_REPO = os.environ.get("CT_QWEN3_TTS_MODEL", "").strip() or VARIANT_REPOS.get(VARIANT, VARIANT_REPOS["customvoice"])
MODEL_ALIAS = os.environ.get("CT_QWEN3_TTS_ALIAS", "qwen3-tts").strip()
PRELOAD_ALL = os.environ.get("CT_QWEN3_TTS_PRELOAD_ALL", "1").strip()
STAGE0_GPU_MEM = float(os.environ.get("CT_QWEN3_TTS_GPU_MEM_STAGE0", "0.30"))
STAGE1_GPU_MEM = float(os.environ.get("CT_QWEN3_TTS_GPU_MEM_STAGE1", "0.20"))
MAX_MODEL_LEN = int(os.environ.get("CT_QWEN3_TTS_MAX_MODEL_LEN", "4096"))
MAX_NUM_SEQS = int(os.environ.get("CT_QWEN3_TTS_MAX_NUM_SEQS", "16"))
AUX_PORT = int(os.environ.get("CT_QWEN3_TTS_AUX_PORT", "8001"))

QT_WORK = Path("/content/qwen3-tts")
VENV_DIR = QT_WORK / "venv"
AUX_VENV_DIR = QT_WORK / "aux-venv"
MODELS_ROOT = QT_WORK / "model"
MODEL_DIR = MODELS_ROOT / MODEL_REPO.split("/")[-1]
SPEAKERS_DIR = QT_WORK / "speakers"
DRIVE_VOICES_DIR = Path(DATA_DIR) / "voices"
DRIVE_OUTPUT_DIR = Path(DATA_DIR) / "outputs"
DEFAULT_REF_WAV = QT_WORK / "default_ref.wav"
DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)

# Two-tier JIT kernel precompilation cache (Local NVMe SSD <-> Google Drive)
FLASHINFER_DRIVE_DIR = Path(DATA_DIR) / "flashinfer-cache"
FLASHINFER_CACHE_DIR = QT_WORK / "flashinfer-cache"
TRITON_DRIVE_DIR = Path(DATA_DIR) / "triton-cache"
TRITON_CACHE_DIR = QT_WORK / "triton-cache"
VLLM_DRIVE_DIR = Path(DATA_DIR) / "vllm-cache"
VLLM_CACHE_DIR = QT_WORK / "vllm-cache"

for d in (
    QT_WORK, MODELS_ROOT, SPEAKERS_DIR, DRIVE_VOICES_DIR, DRIVE_OUTPUT_DIR,
    FLASHINFER_DRIVE_DIR, FLASHINFER_CACHE_DIR,
    TRITON_DRIVE_DIR, TRITON_CACHE_DIR,
    VLLM_DRIVE_DIR, VLLM_CACHE_DIR,
):
    d.mkdir(parents=True, exist_ok=True)

def qlog(msg):
    print(time.strftime("[%F %T] [qwen3-tts] ") + msg, flush=True)

# One archive per cache/environment/GPU avoids Drive's small-file latency.
(QT_WORK / "colab_cache.py").write_text(__QWEN3_CACHE_SOURCE__, encoding="utf-8")
sys.path.insert(0, str(QT_WORK))
from colab_cache import restore_cache, save_cache
_gpu_arch = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True).strip().splitlines()[0]
CACHE_KEY = "vllm-0.32.0-py312-sm" + _gpu_arch.replace(".", "")
CACHE_ARCHIVES = Path(DATA_DIR) / "cache-archives"
CACHE_PAIRS = (
    (FLASHINFER_CACHE_DIR, FLASHINFER_DRIVE_DIR),
    (TRITON_CACHE_DIR, TRITON_DRIVE_DIR),
    (VLLM_CACHE_DIR, VLLM_DRIVE_DIR),
)
for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        restore_cache(_local_c, _legacy_c, CACHE_ARCHIVES, CACHE_KEY, qlog)
    except (OSError, ValueError) as err:
        qlog(f"Cache restore warning ({_local_c.name}): {err}; continuing with local cache")
qlog("Restoring registered voice samples from Drive")
subprocess.run(["cp", "-rn", f"{DRIVE_VOICES_DIR}/.", f"{SPEAKERS_DIR}/"], check=False)
qlog("Cache/voice restore finished; preparing environment and model")

def run_logged(cmd, log_path, env=None, cwd=None, check=True):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(str(x) for x in cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True)
    if check and res.returncode != 0:
        tail = "".join(open(log_path, errors="replace").readlines()[-40:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(str(x) for x in cmd[:4])}\n{tail}")
    return res

gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader"], text=True
).strip()
qlog(f"GPU detected: {gpu_info} | repo={MODEL_REPO} variant={VARIANT} stage0_mem={STAGE0_GPU_MEM} stage1_mem={STAGE1_GPU_MEM}")

if not shutil.which("ffmpeg") or not shutil.which("sox"):
    subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg sox libsox-fmt-all >/dev/null 2>&1", shell=True, check=False)

if not shutil.which("uv"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)

def prepare_env():
    env_log = QT_WORK / "env-setup.log"
    marker = VENV_DIR / ".ct_qwen3_tts_env_ready"
    vllm_bin = VENV_DIR / "bin" / "vllm"
    py_bin = VENV_DIR / "bin" / "python"
    aux_py_bin = AUX_VENV_DIR / "bin" / "python"
    if marker.is_file() and vllm_bin.is_file() and py_bin.is_file() and aux_py_bin.is_file():
        qlog("Reusing existing vLLM-Omni + Qwen3-TTS virtualenvs")
        return VENV_DIR

    qlog(f"Installing vLLM + vLLM-Omni + Qwen3-TTS environments (log: {env_log})...")
    uv_bin = shutil.which("uv") or "uv"
    run_logged([uv_bin, "venv", "--python", "3.12", str(VENV_DIR), "--clear"], env_log)
    run_logged(
        [
            uv_bin, "pip", "install", "--python", str(py_bin),
            "vllm==0.32.0", "vllm-omni==0.32.0",
            "tiktoken", "soundfile", "huggingface_hub", "hf_xet",
        ],
        env_log,
    )

    # Lightweight auxiliary venv (reusing system PyTorch) for cross-task Qwen3TTSModel fallback
    run_logged([uv_bin, "venv", "--python", "3.12", "--system-site-packages", str(AUX_VENV_DIR), "--clear"], env_log)
    run_logged(
        [
            uv_bin, "pip", "install", "--python", str(aux_py_bin),
            "transformers==4.57.3", "accelerate==1.12.0", "qwen-tts",
            "soundfile", "librosa", "einops", "onnxruntime", "sox",
        ],
        env_log,
    )
    marker.write_text("ok\n", encoding="utf-8")
    qlog("vLLM-Omni + Qwen3-TTS virtualenvs ready")
    return VENV_DIR

def _cleanup_hf_temp():
    for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
        shutil.rmtree(p, ignore_errors=True)
    for mp in MODELS_ROOT.glob("*/.cache"):
        shutil.rmtree(mp, ignore_errors=True)

def _ensure_default_ref_wav():
    if DEFAULT_REF_WAV.is_file() and DEFAULT_REF_WAV.stat().st_size > 1024:
        return
    drive_copy = DRIVE_VOICES_DIR / "default_ref.wav"
    if drive_copy.is_file() and drive_copy.stat().st_size > 1024:
        shutil.copyfile(drive_copy, DEFAULT_REF_WAV)
        return
    for ref_url in (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
        "https://raw.githubusercontent.com/vllm-project/vllm-omni/main/tests/assets/indextts2/ref_audio.wav",
    ):
        try:
            with urllib.request.urlopen(ref_url, timeout=60) as _src, open(DEFAULT_REF_WAV, "wb") as _dst:
                shutil.copyfileobj(_src, _dst)
            if DEFAULT_REF_WAV.is_file() and DEFAULT_REF_WAV.stat().st_size > 1024:
                break
        except Exception:
            pass
    if not DEFAULT_REF_WAV.is_file() or DEFAULT_REF_WAV.stat().st_size <= 1024:
        import math, struct, wave
        sr = 24000
        n = sr * 3
        with wave.open(str(DEFAULT_REF_WAV), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            samples = [int(12000 * math.sin(2 * math.pi * 220 * i / sr)) for i in range(n)]
            wf.writeframes(struct.pack(f"<{n}h", *samples))
    if not drive_copy.is_file():
        shutil.copyfile(DEFAULT_REF_WAV, drive_copy)

def prepare_model():
    dl_log = QT_WORK / "model-download.log"
    _ensure_default_ref_wav()
    repos_to_fetch = [MODEL_REPO]
    if PRELOAD_ALL == "1":
        for extra_repo in (
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ):
            if extra_repo not in repos_to_fetch:
                repos_to_fetch.append(extra_repo)

    def _is_repo_ready(r_id: str) -> bool:
        d = MODELS_ROOT / r_id.split("/")[-1]
        return (
            (d / ".ct_model_ready").is_file()
            and (d / "config.json").is_file()
            and (d / "model.safetensors").is_file()
            and (d / "speech_tokenizer" / "model.safetensors").is_file()
        )

    missing_repos = [r for r in repos_to_fetch if not _is_repo_ready(r)]
    if not missing_repos:
        qlog(f"Reusing local SSD Qwen3-TTS models in {MODELS_ROOT}: {[r.split('/')[-1] for r in repos_to_fetch]}")
        return MODEL_DIR

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_xet"], check=True)
    _cleanup_hf_temp()
    free_gb = shutil.disk_usage("/content").free / 1e9
    qlog(f"Downloading {missing_repos} -> local SSD {MODELS_ROOT} (free SSD: {free_gb:.1f} GB; log: {dl_log})...")
    dl_env = {**os.environ, "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    dl_env.pop("HF_XET_HIGH_PERFORMANCE", None)
    dl_script = r"""
import shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download

models_root = Path(sys.argv[1])
repos = sys.argv[2:]

def clean():
    for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
        shutil.rmtree(p, ignore_errors=True)
    for mp in models_root.glob("*/.cache"):
        shutil.rmtree(mp, ignore_errors=True)

for idx, repo_id in enumerate(repos, 1):
    target_dir = models_root / repo_id.split("/")[-1]
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dl] {idx}/{len(repos)} Downloading {repo_id} -> {target_dir}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target_dir), max_workers=4)
    clean()
    if (target_dir / "model.safetensors").is_file() and (target_dir / "speech_tokenizer" / "model.safetensors").is_file():
        (target_dir / ".ct_model_ready").write_text(repo_id + "\n", encoding="utf-8")
"""
    try:
        run_logged([sys.executable, "-c", dl_script, str(MODELS_ROOT), *missing_repos], dl_log, env=dl_env)
    finally:
        _cleanup_hf_temp()

    if not _is_repo_ready(MODEL_REPO):
        raise RuntimeError(f"Primary model download incomplete in {MODEL_DIR}")
    total_gb = sum(p.stat().st_size for p in MODELS_ROOT.rglob("*") if p.is_file()) / 1e9
    qlog(f"Qwen3-TTS models ready on local SSD: {MODELS_ROOT} ({total_gb:.2f} GB total)")
    return MODEL_DIR

with ThreadPoolExecutor(max_workers=2) as pool:
    f_env = pool.submit(prepare_env)
    f_mod = pool.submit(prepare_model)
    while not (f_env.done() and f_mod.done()):
        for fut in (f_env, f_mod):
            if fut.done() and fut.exception():
                raise fut.exception()
        free_gb = shutil.disk_usage("/content").free / 1e9
        qlog(f"Setup heartbeat | env={'done' if f_env.done() else 'running'} "
             f"model={'done' if f_mod.done() else 'running'} | free_ssd={free_gb:.1f} GB")
        time.sleep(30)
    VENV_DIR = f_env.result()
    MODEL_DIR = f_mod.result()

qlog(f"Setup complete: venv={VENV_DIR} model={MODEL_DIR} drive_data={DATA_DIR}")
'''.replace("__QWEN3_CACHE_SOURCE__", repr((ROOT / 'templates/colab_cache.py').read_text()))

QWEN3_TTS_SERVICE = r'''# @title 2. Service -- vLLM-Omni Qwen3-TTS on 127.0.0.1:8000 + Aux Multi-Task Worker on :8001 + Bearer Auth
DEPLOY_YAML = QT_WORK / "qwen3_tts.yaml"
DEPLOY_YAML.write_text(f"""pipeline: qwen3_tts
model_runner: v2
async_chunk: true

connectors:
  connector_of_shared_memory:
    name: SharedMemoryConnector
    extra:
      talker_first_audio: true
      codec_streaming: true
      connector_get_sleep_s: 0.01
      connector_get_max_wait_first_chunk: 3000
      connector_get_max_wait: 300
      codec_chunk_frames: 25
      codec_left_context_frames: 72
      initial_codec_chunk_frames: 1
      decode_batch_max_size: 4

stages:
  - stage_id: 0
    devices: "0"
    max_num_seqs: {MAX_NUM_SEQS}
    gpu_memory_utilization: {STAGE0_GPU_MEM}
    trust_remote_code: true
    enable_prefix_caching: false
    async_scheduling: true
    max_num_batched_tokens: 512
    max_model_len: {MAX_MODEL_LEN}
    output_connectors:
      to_stage_1: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.9
      top_k: 50
      max_tokens: {MAX_MODEL_LEN}
      min_tokens: 2
      repetition_penalty: 1.05
    silence_ban_frames: 0
    subtalker_sampling_params:
      do_sample: true
      temperature: 0.9
      top_k: 50
      top_p: 1.0

  - stage_id: 1
    devices: "0"
    dtype: bfloat16
    max_num_seqs: {MAX_NUM_SEQS}
    gpu_memory_utilization: {STAGE1_GPU_MEM}
    enforce_eager: false
    trust_remote_code: true
    enable_prefix_caching: false
    async_scheduling: true
    max_num_batched_tokens: 65536
    max_model_len: 65536
    input_connectors:
      from_stage_0: connector_of_shared_memory
    default_sampling_params:
      temperature: 0.0
      top_p: 1.0
      top_k: -1
      max_tokens: 65536
      repetition_penalty: 1.0
""", encoding="utf-8")

# Auxiliary multi-task worker on 127.0.0.1:8001 so CustomVoice, VoiceDesign, and Base (Voice Clone)
# all work seamlessly regardless of which checkpoint is loaded as the primary vLLM-Omni engine.
(QT_WORK / "ct_qwen3_aux_worker.py").write_text(__QWEN3_AUX_SOURCE__, encoding="utf-8")

(QT_WORK / "colab_asgi_common.py").write_text(__ASGI_COMMON_SOURCE__, encoding="utf-8")
(QT_WORK / "ct_qwen3_tts_middleware.py").write_text(__QWEN3_MW_SOURCE__, encoding="utf-8")

VLLM_ENV = {
    "PATH": f"{VENV_DIR}/bin:/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
    "PYTHONPATH": f"{QT_WORK}:{os.environ.get('PYTHONPATH', '')}",
    "VIRTUAL_ENV": str(VENV_DIR),
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "FLASHINFER_WORKSPACE_BASE": str(FLASHINFER_CACHE_DIR),
    "TRITON_CACHE_DIR": str(TRITON_CACHE_DIR),
    "VLLM_CACHE_ROOT": str(VLLM_CACHE_DIR),
    "TORCHINDUCTOR_CACHE_DIR": str(VLLM_CACHE_DIR / "inductor"),
    "SPEAKER_SAMPLES_DIR": str(SPEAKERS_DIR),
    "CT_API_TOKEN": API_TOKEN,
    "CT_SERVICE_PORT": str(SERVICE_PORT),
    "CT_QWEN3_TTS_AUX_PORT": str(AUX_PORT),
    "CT_DATA_DIR": str(DATA_DIR),
    "CT_MODELS_ROOT": str(MODELS_ROOT),
    "CT_DEFAULT_REF_WAV": str(DEFAULT_REF_WAV),
    "CT_DEFAULT_REF_TEXT": DEFAULT_REF_TEXT,
    "CT_QWEN3_TTS_MODEL": MODEL_REPO,
}

SERVER_ARGS = [
    str(VENV_DIR / "bin" / "vllm"), "serve", str(MODEL_DIR),
    "--omni",
    "--served-model-name", MODEL_REPO, MODEL_ALIAS,
    "--host", "127.0.0.1",
    "--port", str(SERVICE_PORT),
    "--api-key", API_TOKEN,
    "--trust-remote-code",
    "--allowed-local-media-path", str(QT_WORK),
    "--stage-init-timeout", "900",
    "--deploy-config", str(DEPLOY_YAML),
    "--middleware", "ct_qwen3_tts_middleware.Qwen3TTSMiddleware",
]

AUX_ARGS = [
    str(AUX_VENV_DIR / "bin" / "python"),
    str(QT_WORK / "ct_qwen3_aux_worker.py"),
]

def start_aux():
    subprocess.run(["pkill", "-9", "-f", "ct_qwen3_aux_worker.py"], check=False)
    return ct_spawn("aux", AUX_ARGS, env=VLLM_ENV, cwd=str(QT_WORK))

def start_service():
    subprocess.run(["pkill", "-9", "-f", "vllm|EngineCore"], check=False)
    time.sleep(1)
    qlog(f"Starting vLLM-Omni Qwen3-TTS server on 127.0.0.1:{SERVICE_PORT} (model={MODEL_REPO})")
    return ct_spawn("service", SERVER_ARGS, env=VLLM_ENV, cwd=str(QT_WORK))

PROCS["aux"] = (start_aux(), start_aux)
PROCS["service"] = (start_service(), start_service)
'''.replace(
    "__ASGI_COMMON_SOURCE__", repr((ROOT / "templates/colab_asgi_common.py").read_text())).replace(
    "__QWEN3_AUX_SOURCE__", repr((ROOT / "templates/qwen3_tts_aux_worker.py").read_text())).replace(
    "__QWEN3_MW_SOURCE__", repr((ROOT / "templates/qwen3_tts_middleware.py").read_text()))

QWEN3_TTS_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- wait for 2-stage load, warmup synthesis & sync JIT caches
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def http_json(url, key=None, payload=None, timeout=30):
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, headers=headers, data=None if payload is None else json.dumps(payload).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else {})
    except urllib.error.HTTPError as err:
        return err.code, {}

def auth_check(base, key):
    for candidate, expected in ((None, 401), ("invalid-token-probe", 401), (key, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Auth test failed on {base}/v1/models: expected {expected}, got {code}")

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
deadline = time.monotonic() + 2400
last_notice = 0
while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        raise RuntimeError("vLLM-Omni exited during startup:\n" + ct_tail("service", 80))
    try:
        code, _ = http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)
        if code == 200:
            break
    except Exception:
        pass
    if time.monotonic() - last_notice >= 30:
        gpu_stat = ct_gpu_stat()
        last_line = (ct_tail("service", 3).strip().splitlines() or ["starting..."])[-1]
        qlog(f"Waiting for 2-stage Qwen3-TTS init | GPU: {gpu_stat} | {last_line[:100]}")
        last_notice = time.monotonic()
    time.sleep(5)
else:
    raise RuntimeError("Timed out waiting for vLLM-Omni server to become ready:\n" + ct_tail("service", 80))

auth_check(LOCAL_BASE, API_TOKEN)

def qwen3_healthy():
    if http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] != 200:
        return False
    return http_json(f"http://127.0.0.1:{AUX_PORT}/health", timeout=10)[0] == 200

if http_json(f"http://127.0.0.1:{AUX_PORT}/health", timeout=10)[0] != 200:
    qlog("Auxiliary worker not healthy yet:\n" + ct_tail("aux", 20))

# Warmup synthesis (payload matches the primary checkpoint task) to sync JIT caches to Drive
_warm = {"model": MODEL_REPO, "input": "Hello, the Qwen3 TTS speech synthesis service is ready.",
         "language": "Chinese", "response_format": "wav"}
_low_repo = MODEL_REPO.lower()
if "voicedesign" in _low_repo:
    _warm.update(task_type="VoiceDesign", instructions="Natural, warm female narrator.")
elif "base" in _low_repo:
    _warm.update(task_type="Base")
else:
    _warm.update(voice="vivian")
warmup_req = urllib.request.Request(
    LOCAL_BASE + "/v1/audio/speech",
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_TOKEN}"},
    data=json.dumps(_warm).encode("utf-8"),
)
try:
    with urllib.request.urlopen(warmup_req, timeout=180) as wresp:
        wav_bytes = wresp.read()
        qlog(f"Warmup synthesis succeeded ({len(wav_bytes)} bytes WAV)")
except Exception as werr:
    qlog(f"Warmup synthesis warning: {werr}")

for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        save_cache(_local_c, CACHE_ARCHIVES, CACHE_KEY, qlog)
    except (OSError, ValueError) as err:
        qlog(f"Cache archive save warning ({_local_c.name}): {err}")
# User voice samples remain ordinary Drive files, not compiler-cache archives.
subprocess.run(["cp", "-ru", f"{SPEAKERS_DIR}/.", f"{DRIVE_VOICES_DIR}/"], check=False)

ct_public_gate(PUBLIC_URL, lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
               lambda: auth_check(PUBLIC_URL, API_TOKEN), qlog, timeout=120, interval=8)
qlog(f"Public endpoint verified: {PUBLIC_URL}/v1/models")

print(f"{SERVICE} is up: {PUBLIC_URL}")
print("CT_READY", flush=True)

ct_watchdog(qwen3_healthy, start_service, qlog, dead_markers=('replica-0 is dead',),
            heartbeat=lambda up: qlog(f"heartbeat | up={up}s | GPU: {ct_gpu_stat()}"))
'''

# --------------------------------------------------------------------------- longcat-video (single G4 / Caddy + Gradio)
LONGCAT_VIDEO_CFG = dict(name="longcat-video", service_port=7860, auth="caddy", public_port=18080,
                         health_path="/health", ready_timeout=3600)
# Gradio builds absolute media/API URLs from forwarded host and scheme. The
# Cloudflare HTTPS edge connects to Caddy over HTTP inside this authenticated VM.
LONGCAT_VIDEO_CADDY = render_caddy_cell(rewrite_host=False, extra_proxy_lines="header_up X-Forwarded-Proto https")

LONGCAT_SETUP_TEMPLATE = r"""# @title 1. Setup -- __TITLE__ on local SSD, Blackwell-compatible CUDA environment
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

LC_WORK = Path("__WORK__")
LC_REPO = LC_WORK / "repo"
LC_MODEL = LC_WORK / "model"
LC_VENV = LC_WORK / "venv"
LC_REVISION = "6b3f4b8582a8bc3f20f795735f5383716c4ba794"
LC_WORK.mkdir(parents=True, exist_ok=True)
LC_OUTPUT = Path(DATA_DIR) / "output"
LC_OUTPUT.mkdir(parents=True, exist_ok=True)
LC_ENV = {**os.environ, "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
          "HF_HUB_DISABLE_PROGRESS_BARS": "1", "PYTHONUNBUFFERED": "1"}
# Compiler caches: one tar archive per environment/GPU on Drive, unpacked to local SSD.
(LC_WORK / "colab_cache.py").write_text(__CACHE_SOURCE__, encoding="utf-8")
sys.path.insert(0, str(LC_WORK))
from colab_cache import restore_cache, save_cache
LC_CACHE_KEY = "torch211-cu130-" + re.sub(r"[^A-Za-z0-9_.-]+", "_", ct_gpu_stat("name").splitlines()[0])
LC_CACHE_ARCHIVES = Path(DATA_DIR) / "kernel-archives"
LC_CACHE_PAIRS = []
for name in ("triton-cache", "inductor-cache"):
    local_cache = LC_WORK / name
    try:
        restore_cache(local_cache, Path(DATA_DIR) / name, LC_CACHE_ARCHIVES, LC_CACHE_KEY)
    except (OSError, ValueError) as err:
        print(f"Cache restore warning ({name}): {err}", flush=True)
    LC_CACHE_PAIRS.append(local_cache)

if not (LC_REPO / ".git").exists():
    subprocess.run(["git", "clone", "https://github.com/meituan-longcat/LongCat-Video.git", str(LC_REPO)], check=True)
subprocess.run(["git", "-C", str(LC_REPO), "checkout", "--detach", LC_REVISION], check=True)

if not shutil.which("uv"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
UV = shutil.which("uv")
if not (LC_VENV / "bin/python").exists():
    subprocess.run([UV, "venv", "--python", "3.11", "--seed", str(LC_VENV)], check=True)
LC_PYTHON = str(LC_VENV / "bin/python")

def lc_install():
    marker = LC_WORK / "environment-v1.ok"
    if marker.exists():
        return
    log_path = LC_WORK / "install.log"
    with log_path.open("a") as log_file:
        def run(args):
            subprocess.run(args, stdout=log_file, stderr=subprocess.STDOUT, env=LC_ENV, check=True)
        # Upstream's torch 2.6/cu124 predates sm_120; use CUDA 13 wheels for G4.
        run([UV, "pip", "install", "--python", LC_PYTHON, "torch==2.11.0", "torchvision==0.26.0",__EXTRA_TORCH__
             "--index-url", "https://download.pytorch.org/whl/cu130"])
        run([UV, "pip", "install", "--python", LC_PYTHON,
             "numpy==1.26.4", "transformers==4.41.0", "diffusers==0.35.1",
             "einops==0.8.0", "ftfy==6.2.0", "loguru==0.7.2", "psutil",
             "av==12.0.0", "opencv-python-headless==4.9.0.80", "imageio==2.37.0",
             "imageio-ffmpeg==0.6.0", "huggingface-hub==0.36.0", "hf_xet",
             "accelerate>=1,<2", "sentencepiece",__EXTRA_PKGS__ "gradio==5.49.1", "uvicorn", "fastapi"])
    marker.write_text("ok\n")

__DOWNLOAD__
with ThreadPoolExecutor(max_workers=2) as pool:
    jobs = [pool.submit(lc_install), pool.submit(lc_download)]
    while not all(job.done() for job in jobs):
        for job in jobs:
            if job.done():
                job.result()
        print("LongCat setup: environment=" + ("ready" if jobs[0].done() else "installing") +
              " model=" + ("ready" if jobs[1].done() else "downloading") +
              f" SSD free={shutil.disk_usage('/content').free / 1e9:.1f} GB; logs={LC_WORK}", flush=True)
        time.sleep(20)
    for job in jobs:
        job.result()
print("LongCat environment and model ready", flush=True)
"""

LONGCAT_SERVICE_TEMPLATE = r"""# @title 2. Service -- __TITLE__, verify inference, then serve Gradio on loopback
__WRITE_SOURCES__
def start_service():
    return ct_spawn("service", [LC_PYTHON, str(LC_APP)], cwd=str(LC_REPO), env={
        "PYTHONPATH": str(LC_REPO), "PYTHONUNBUFFERED": "1",
        "CT_LONGCAT_MODEL_DIR": str(LC_MODEL), "CT_LONGCAT_WORK_DIR": str(LC_WORK),
        "CT_LONGCAT_OUTPUT_DIR": str(LC_OUTPUT),
        "CT_SERVICE_PORT": str(SERVICE_PORT),__EXTRA_ENV__
        "TRITON_CACHE_DIR": str(LC_WORK / "triton-cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(LC_WORK / "inductor-cache"),
        "GRADIO_ANALYTICS_ENABLED": "False",
    })

PROCS["service"] = (start_service(), start_service)
# Wait for model + real inference before the shared authentication/tunnel cells.
deadline = time.time() + 1800
while time.time() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        raise RuntimeError("LongCat service exited:\n" + ct_tail("service", 60))
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{SERVICE_PORT}/health", timeout=3) as response:
            if json.load(response).get("inference_verified"):
                break
    except (OSError, ValueError):
        pass
    print("LongCat loading / warming up:\n" + ct_tail("service", 3), flush=True)
    time.sleep(20)
else:
    raise TimeoutError("LongCat inference readiness timed out:\n" + ct_tail("service", 60))
for local_cache in LC_CACHE_PAIRS:
    try:
        save_cache(local_cache, LC_CACHE_ARCHIVES, LC_CACHE_KEY)
    except (OSError, ValueError) as err:
        print(f"Cache archive save warning ({local_cache.name}): {err}", flush=True)
print("LongCat inference verified; compiled kernels archived to Drive", flush=True)
"""


def embed_source(directory_var, filename, source):
    """Return a notebook line that writes a repository file next to the service."""
    return f"({directory_var} / {filename!r}).write_text({(ROOT / source).read_text()!r}, encoding=\"utf-8\")\n"


def build_longcat_setup(title, work, download, extra_torch="", extra_pkgs=""):
    return (LONGCAT_SETUP_TEMPLATE.replace("__TITLE__", title).replace("__WORK__", work)
            .replace("__CACHE_SOURCE__", repr((ROOT / "templates/colab_cache.py").read_text()))
            .replace("__EXTRA_TORCH__", extra_torch).replace("__EXTRA_PKGS__", extra_pkgs)
            .replace("__DOWNLOAD__", download))


def build_longcat_service(title, app_file, sources, extra_env=""):
    writes = "".join(embed_source("LC_WORK", name, src) for name, src in sources)
    writes += f'LC_APP = LC_WORK / {app_file!r}\n'
    return (LONGCAT_SERVICE_TEMPLATE.replace("__TITLE__", title).replace("__WRITE_SOURCES__", writes)
            .replace("__EXTRA_ENV__", extra_env))


LONGCAT_VIDEO_DOWNLOAD = r'''def lc_download():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub", "hf_xet"], check=True)
    # snapshot_download validates the local snapshot on each run; interrupted downloads resume.
    download = (
        "from huggingface_hub import snapshot_download; "
        "snapshot_download('meituan-longcat/LongCat-Video', local_dir=" + repr(str(LC_MODEL)) +
        ", allow_patterns=['*.json', 'dit/*', 'text_encoder/*', 'tokenizer/*', 'vae/*', "
        "'scheduler/*', 'lora/cfg_step_lora.safetensors'])"
    )
    with (LC_WORK / "download.log").open("a") as log_file:
        subprocess.run([sys.executable, "-c", download], stdout=log_file, stderr=subprocess.STDOUT,
                       env=LC_ENV, check=True)
    shutil.rmtree("/root/.cache/huggingface/xet", ignore_errors=True)
'''
LONGCAT_VIDEO_SETUP = build_longcat_setup("LongCat-Video", "/content/longcat-video", LONGCAT_VIDEO_DOWNLOAD)
LONGCAT_VIDEO_SERVICE = build_longcat_service(
    "load BF16 model", "longcat_video_app.py",
    [("longcat_video_app.py", "templates/longcat_video_app.py")])


# --------------------------------------------------------------------------- longcat-avatar (Avatar 1.5 / single G4)
LONGCAT_AVATAR_CFG = dict(name="longcat-avatar", service_port=7860, auth="caddy", public_port=18080,
                          health_path="/health", ready_timeout=3600)
# Reuse the validated G4 environment and cache layout. Download only Avatar's
# BF16 DiT/Whisper/DMD and the shared tokenizer/T5/VAE, not the base DiT.
LONGCAT_AVATAR_DOWNLOAD = r'''def lc_download():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface-hub", "hf_xet"], check=True)
    download = r"""
from huggingface_hub import snapshot_download
import sys
from pathlib import Path
root = Path(sys.argv[1])
snapshot_download('meituan-longcat/LongCat-Video', local_dir=root / 'LongCat-Video',
                  allow_patterns=['tokenizer/*', 'text_encoder/*', 'vae/*'])
snapshot_download('meituan-longcat/LongCat-Video-Avatar-1.5', local_dir=root / 'LongCat-Video-Avatar-1.5',
                  allow_patterns=['base_model/*', 'scheduler/*', 'lora/dmd_lora.safetensors',
                                  'whisper-large-v3/*.json', 'whisper-large-v3/*.txt',
                                  'whisper-large-v3/model.safetensors'])
"""
    with (LC_WORK / "download.log").open("a") as log_file:
        subprocess.run([sys.executable, "-c", download, str(LC_MODEL)], stdout=log_file, stderr=subprocess.STDOUT,
                       env=LC_ENV, check=True)
    shutil.rmtree("/root/.cache/huggingface/xet", ignore_errors=True)
'''
LONGCAT_AVATAR_SETUP = build_longcat_setup(
    "LongCat-Video-Avatar 1.5", "/content/longcat-avatar", LONGCAT_AVATAR_DOWNLOAD,
    extra_torch=' "torchaudio==2.11.0",',
    extra_pkgs=' "librosa==0.11.0", "pyloudnorm==0.1.1", "soundfile==0.13.1", "scipy==1.15.3",')
LONGCAT_AVATAR_SERVICE = build_longcat_service(
    "load Avatar 1.5 BF16 + Whisper + DMD", "longcat_avatar_app.py",
    [("longcat_video_app.py", "templates/longcat_video_app.py"),
     ("longcat_avatar_app.py", "templates/longcat_avatar_app.py"),
     ("longcat_avatar_jobs.py", "templates/longcat_avatar_jobs.py"),
     ("longcat_avatar_drive.py", "templates/longcat_avatar_drive.py"),
     ("avatar_sampling.py", "src/colabllm/lib/avatar_sampling.py")],
    extra_env=' "CT_LONGCAT_REPO_DIR": str(LC_REPO),')

# Unlimited-OCR: source files are embedded to keep the notebook self-contained.
UNLIMITED_OCR_CFG = dict(name="unlimited-ocr", service_port=8000, auth="caddy", public_port=18080,
                         health_path="/health", ready_timeout=5400)
UNLIMITED_OCR_SETUP = ("OCR_CACHE_SOURCE = " + repr((ROOT / "templates/colab_cache.py").read_text()) + "\n"
                       + (ROOT / "templates/unlimited_ocr_setup.py").read_text())
UNLIMITED_OCR_SERVICE = ("# @title 2. Unlimited-OCR Drive job service\n"
    + "(OCR_WORK / 'ocr_parameters.py').write_text(" + repr((ROOT / "src/colabllm/lib/ocr_parameters.py").read_text()) + ", encoding='utf-8')\n"
    + "(OCR_WORK / 'unlimited_ocr_app.py').write_text(" + repr((ROOT / "templates/unlimited_ocr_app.py").read_text()) + ", encoding='utf-8')\n"
    + """def start_service():
    return ct_spawn("service", [OCR_PY, str(OCR_WORK / 'unlimited_ocr_app.py')], cwd=str(OCR_WORK),
                    env={"CT_SERVICE_PORT": str(SERVICE_PORT), "CT_DATA_DIR": DATA_DIR,
                         "CT_OCR_MODEL_DIR": str(OCR_MODEL),
                         "TRITON_CACHE_DIR": str(OCR_WORK / 'triton-cache'),
                         "TORCHINDUCTOR_CACHE_DIR": str(OCR_WORK / 'inductor-cache'),
                         "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
PROCS["service"] = (start_service(), start_service)
""")
UNLIMITED_OCR_KEEPALIVE = render_keepalive(UNLIMITED_OCR_CFG, before_ready="""for local in OCR_CACHE_PAIRS:
    try:
        save_cache(local, Path(DATA_DIR) / 'kernel-archives', OCR_CACHE_KEY)
    except (OSError, ValueError) as err:
        print(f'Cache archive save warning ({local.name}): {err}', flush=True)""")

# --------------------------------------------------------------------------- openjutsu (G4, ComfyUI + comfy-api-proxy + Next.js behind Caddy)
OPENJUTSU_CFG = dict(name="openjutsu", service_port=3000, auth="caddy", public_port=18080,
                     health_path="/api/models", ready_timeout=3600)
OPENJUTSU_CADDY = render_caddy_cell(
    rewrite_host=True,
    extra_proxy_lines="header_up -X-Forwarded-Host\nheader_up -X-Forwarded-Proto",
)
OPENJUTSU_SETUP = (
    "# @title 1. Setup -- ComfyUI + 13 custom node packs + comfy-api-proxy + Openjutsu Next.js + local SSD models\n"
    "from concurrent.futures import ThreadPoolExecutor\n"
    "from pathlib import Path\n"
    "import hashlib, platform\n\n"
    "OJ_WORK = Path(os.environ.get('CT_OPENJUTSU_WORK_DIR', '/content/openjutsu'))\n"
    "OJ_COMFY = Path(os.environ.get('CT_COMFY_DIR', '/content/ComfyUI'))\n"
    "OJ_PROFILE = (os.environ.get('CT_OPENJUTSU_PROFILE', 'full').strip().lower() or 'full')\n"
    "OJ_WORK.mkdir(parents=True, exist_ok=True)\n"
    "(OJ_WORK / 'colab_cache.py').write_text("
    + repr((ROOT / "templates/colab_cache.py").read_text())
    + ", encoding='utf-8')\n"
    "(OJ_WORK / 'openjutsu_runtime.py').write_text("
    + repr((ROOT / "templates/openjutsu_runtime.py").read_text())
    + ", encoding='utf-8')\n"
    "if str(OJ_WORK) not in sys.path:\n"
    "    sys.path.insert(0, str(OJ_WORK))\n"
    "from colab_cache import restore_cache, save_cache\n"
    "from openjutsu_runtime import (\n"
    "    detect_gpu_vram_mib,\n"
    "    olog,\n"
    "    prepare_comfy_tree,\n"
    "    select_comfy_vram_flags,\n"
    "    setup_check_payloads,\n"
    "    setup_environment,\n"
    "    stage_models,\n"
    ")\n\n"
    "OJ_GPU_SIG = ct_gpu_stat('name,compute_cap')\n"
    "OJ_CACHE_KEY = hashlib.sha256(f'{sys.version_info[:2]}:{platform.machine()}:{OJ_GPU_SIG}'.encode()).hexdigest()[:16]\n"
    "OJ_CACHE_DIRS = [OJ_WORK / 'triton-cache', OJ_WORK / 'inductor-cache']\n"
    "for _cache_dir in OJ_CACHE_DIRS:\n"
    "    restore_cache(_cache_dir, Path(DATA_DIR) / 'legacy-caches' / _cache_dir.name,\n"
    "                  Path(DATA_DIR) / 'kernel-archives', OJ_CACHE_KEY, log=olog)\n\n"
    "prepare_comfy_tree(OJ_WORK, OJ_COMFY, Path(DATA_DIR))\n"
    "olog(f'Starting parallel environment build and {OJ_PROFILE} model staging on {OJ_GPU_SIG}...')\n"
    "with ThreadPoolExecutor(max_workers=2) as _pool:\n"
    "    _f_env = _pool.submit(setup_environment, OJ_WORK, OJ_COMFY, Path(DATA_DIR))\n"
    "    _f_models = _pool.submit(stage_models, OJ_COMFY, OJ_PROFILE, 3, OJ_WORK)\n"
    "    while not (_f_env.done() and _f_models.done()):\n"
    "        for _fut in (_f_env, _f_models):\n"
    "            if _fut.done() and _fut.exception():\n"
    "                raise _fut.exception()\n"
    "        olog(f\"Setup heartbeat | env={'done' if _f_env.done() else 'running'} \"\n"
    "             f\"models={'done' if _f_models.done() else 'running'} | gpu={ct_gpu_stat()}\")\n"
    "        time.sleep(30)\n"
    "    OJ_META = _f_env.result()\n"
    "    OJ_MODELS = _f_models.result()\n"
    "olog(f\"Setup complete: {len(OJ_MODELS)} models staged, app={OJ_META['app_dir']}\")\n"
)

OPENJUTSU_SERVICE = r'''# @title 2. Service -- ComfyUI (:8188) + patched comfy-api-proxy (:8189) + Openjutsu Next.js (:3000)
if "OJ_META" not in globals():
    raise RuntimeError("Openjutsu setup cell failed before completing environment build; see setup traceback above.")
OJ_PROXY_STATE = OJ_WORK / "proxy-state"
OJ_PROXY_STATE.mkdir(parents=True, exist_ok=True)

def _wait_proc_http(proc_key, url, timeout=600, ok=lambda c: c < 500):
    deadline = time.time() + timeout
    while time.time() < deadline:
        proc = PROCS[proc_key][0]
        if proc.poll() is not None:
            raise RuntimeError(f"{proc_key} exited early (code {proc.returncode}):\n{ct_tail(proc_key, 60)}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if ok(resp.status):
                    return resp.status
        except urllib.error.HTTPError as err:
            if ok(err.code):
                return err.code
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"{proc_key} ({url}) not ready after {timeout}s:\n{ct_tail(proc_key, 50)}")

def start_comfyui():
    subprocess.run(["pkill", "-f", f"{OJ_COMFY}/main.py"], check=False)
    vram_flags = select_comfy_vram_flags(
        detect_gpu_vram_mib(),
        os.environ.get("CT_COMFY_VRAM_MODE", "auto"),
    )
    olog(f"Starting ComfyUI (VRAM flags={vram_flags or ['dynamic-default']})")
    return ct_spawn(
        "comfyui",
        [
            sys.executable,
            str(OJ_COMFY / "main.py"),
            "--listen", "127.0.0.1",
            "--port", "8188",
            "--extra-model-paths-config", OJ_META["extra_model_paths"],
            "--cache-classic",
            *vram_flags,
        ],
        cwd=str(OJ_COMFY),
        env={
            "TRITON_CACHE_DIR": str(OJ_WORK / "triton-cache"),
            "TORCHINDUCTOR_CACHE_DIR": str(OJ_WORK / "inductor-cache"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )

def start_comfy_proxy():
    subprocess.run(["pkill", "-f", "comfy-api-proxy"], check=False)
    return ct_spawn(
        "comfy_proxy",
        [
            OJ_META["proxy_bin"],
            "--comfyui", "http://127.0.0.1:8188",
            "--host", "127.0.0.1",
            "--port", "8189",
            "--max-upload-mb", "500",
            "--state-dir", str(OJ_PROXY_STATE),
        ],
        cwd=str(OJ_WORK),
    )

def start_service():
    subprocess.run(["pkill", "-f", "next/dist/bin/next"], check=False)
    return ct_spawn(
        "service",
        [
            str(Path(OJ_META["app_dir"]) / "node_modules" / ".bin" / "next"),
            "start",
            "--hostname", "127.0.0.1",
            "--port", str(SERVICE_PORT),
        ],
        cwd=OJ_META["app_dir"],
        env={
            "PATH": f"{OJ_META['node_bin_dir']}:{os.environ.get('PATH', '')}",
            "NODE_ENV": "production",
            "NEXT_TELEMETRY_DISABLED": "1",
            "COMFY_BASE_URL": "http://127.0.0.1:8189",
            "COMFY_API_KEY": "",
            "COMFYUI_URL": "http://127.0.0.1:8188",
            "OPENJUTSU_PASSWORD": "",
        },
    )

PROCS["comfyui"] = (start_comfyui(), start_comfyui)
_wait_proc_http("comfyui", "http://127.0.0.1:8188/system_stats", timeout=600, ok=lambda c: c == 200)
_wait_proc_http("comfyui", "http://127.0.0.1:8188/websocket_reconnect_fix", timeout=60, ok=lambda c: c == 200)
olog("ComfyUI + custom nodes listening on 127.0.0.1:8188")

PROCS["comfy_proxy"] = (start_comfy_proxy(), start_comfy_proxy)
_wait_proc_http("comfy_proxy", "http://127.0.0.1:8189/", timeout=120, ok=lambda c: c < 500)
olog("Patched comfy-api-proxy listening on 127.0.0.1:8189")

PROCS["service"] = (start_service(), start_service)
_wait_proc_http("service", f"http://127.0.0.1:{SERVICE_PORT}/api/models", timeout=180, ok=lambda c: c == 200)
olog(f"Openjutsu Next.js server listening on 127.0.0.1:{SERVICE_PORT}")
'''

OPENJUTSU_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- verify /api/models, /api/setup & public Caddy auth before CT_READY
import base64

if "OJ_META" not in globals() or "service" not in PROCS:
    raise RuntimeError("Openjutsu service did not start because an earlier notebook cell failed; see traceback above.")

LOCAL_ORIGIN = f"http://127.0.0.1:{SERVICE_PORT}"

def _oj_local_json(path, payload=None, timeout=30):
    headers = {"Origin": LOCAL_ORIGIN, "Accept": "application/json"}
    data = None
    method = "GET"
    if payload is not None:
        method = "POST"
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(LOCAL_ORIGIN + path, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

_, _models_info = _oj_local_json("/api/models", timeout=30)
if _models_info.get("error") or not _models_info.get("models"):
    raise RuntimeError(f"Openjutsu /api/models failed: {_models_info}\n{ct_tail('comfyui', 50)}")

for _mode, _payload in setup_check_payloads(OJ_PROFILE).items():
    _, _setup_res = _oj_local_json("/api/setup", payload=_payload, timeout=45)
    if not _setup_res.get("ok"):
        raise RuntimeError(
            f"Openjutsu /api/setup check failed for mode={_mode!r}: {_setup_res}\n{ct_tail('comfyui', 60)}"
        )
olog(f"Verified Openjutsu /api/models and /api/setup for profile={OJ_PROFILE}")

for _cache_dir in OJ_CACHE_DIRS:
    try:
        save_cache(_cache_dir, Path(DATA_DIR) / "kernel-archives", OJ_CACHE_KEY, log=olog)
    except (OSError, ValueError) as _err:
        olog(f"Cache archive save warning ({_cache_dir.name}): {_err}")

def _public_bearer_probe():
    req = urllib.request.Request(
        PUBLIC_URL + "/api/models",
        headers={"Authorization": f"Bearer {API_TOKEN}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        body = json.loads(resp.read().decode("utf-8"))
        if body.get("error"):
            raise RuntimeError(f"Public /api/models returned error: {body}")
        return resp.status

def _verify_public_auth():
    # 1. Unauthenticated request must return 401 and must NOT leak Set-Cookie
    req_unauth = urllib.request.Request(PUBLIC_URL + "/", headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req_unauth, timeout=15) as resp:
            raise RuntimeError(f"Unauthenticated public request unexpectedly returned HTTP {resp.status}")
    except urllib.error.HTTPError as err:
        if err.code != 401:
            raise RuntimeError(f"Expected HTTP 401 without credentials, got HTTP {err.code}")
        if err.headers.get("Set-Cookie"):
            raise RuntimeError("Public 401 response leaked Set-Cookie")
    # 2. Invalid Bearer token must return 401
    req_bad = urllib.request.Request(
        PUBLIC_URL + "/api/models",
        headers={"Authorization": "Bearer invalid-probe-token", "User-Agent": "Mozilla/5.0"},
    )
    try:
        with urllib.request.urlopen(req_bad, timeout=15) as resp:
            raise RuntimeError(f"Invalid Bearer token unexpectedly returned HTTP {resp.status}")
    except urllib.error.HTTPError as err:
        if err.code != 401:
            raise RuntimeError(f"Expected HTTP 401 for invalid Bearer token, got HTTP {err.code}")
    # 3. Browser Basic Auth must return 200 and issue ct_session cookie
    basic = base64.b64encode(f"{AUTH_USER}:{AUTH_PASS}".encode("utf-8")).decode("ascii")
    req_basic = urllib.request.Request(
        PUBLIC_URL + "/",
        headers={"Authorization": f"Basic {basic}", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(req_basic, timeout=20) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Basic Auth request returned HTTP {resp.status}")

ct_public_gate(PUBLIC_URL, _public_bearer_probe, _verify_public_auth, olog, timeout=180)
if AUTH == "caddy" and not CREDS_FROM_ENV:
    print(f"CT_CREDENTIALS user={AUTH_USER} pass={AUTH_PASS}")
    print(f"API token: {API_TOKEN}")
print(f"{SERVICE} is up: {PUBLIC_URL}")
print("CT_READY", flush=True)

def _oj_health():
    if PROCS["comfyui"][0].poll() is not None or PROCS["comfy_proxy"][0].poll() is not None:
        return False
    req = urllib.request.Request(LOCAL_ORIGIN + "/", headers={"Host": f"127.0.0.1:{SERVICE_PORT}"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status == 200

def _oj_heartbeat(uptime_s):
    if uptime_s % 600 < 60:
        states = " ".join(f"{n}={'up' if p.poll() is None else 'down'}" for n, (p, _) in PROCS.items())
        olog(f"[heartbeat] uptime={uptime_s}s {states} gpu={ct_gpu_stat()}")

ct_watchdog(_oj_health, start_service, olog, interval=60, grace=900, heartbeat=_oj_heartbeat)
'''

# --------------------------------------------------------------------------- flux-klein (G4 default / H100 / L4, vLLM-Omni FLUX 3 Image Klein T2I + I2I + optional PUP)
FLUX_KLEIN_CFG = dict(
    name="flux-klein",
    service_port=8099,
    auth="builtin",
    public_port=8099,
    health_path="/health",
    ready_timeout=3600,
)

FLUX_KLEIN_SETUP = (
    "# @title 1. Setup -- install vLLM + vLLM-Omni (flux3) & stage FLUX 3 Image Klein on local NVMe SSD\n"
    "from concurrent.futures import ThreadPoolExecutor\n"
    "from pathlib import Path\n\n"
    "FK_SETUP_OK = False\n"
    "MODEL_REPO = os.environ.get('CT_FLUX_MODEL', 'black-forest-labs/flux-3-image-klein').strip() or 'black-forest-labs/flux-3-image-klein'\n"
    "VARIANT = os.environ.get('CT_FLUX_VARIANT', 'bf16').strip().lower() or 'bf16'\n"
    "MODEL_ALIAS = os.environ.get('CT_FLUX_ALIAS', 'flux-3-image-klein').strip() or 'flux-3-image-klein'\n"
    "LORA_PATH_SPEC = os.environ.get('CT_FLUX_LORA_PATH', '').strip()\n"
    "LORA_SCALE = os.environ.get('CT_FLUX_LORA_SCALE', '1.0').strip() or '1.0'\n"
    "FP8_ATTENTION = os.environ.get('CT_FLUX_FP8_ATTENTION', '0').strip() in ('1', 'true', 'True', 'yes')\n"
    "ENABLE_PUP = os.environ.get('CT_FLUX_ENABLE_PUP', '0').strip() in ('1', 'true', 'True', 'yes')\n"
    "PUP_REPO = os.environ.get('CT_FLUX_PUP_REPO', 'black-forest-labs/flux-3-image-klein-pup').strip() or 'black-forest-labs/flux-3-image-klein-pup'\n"
    "PUP_PORT = int(os.environ.get('CT_FLUX_PUP_PORT', '8098'))\n"
    "PUP_GPU_MEM = float(os.environ.get('CT_FLUX_PUP_GPU_MEM', '0.35'))\n"
    "VLLM_VERSION = os.environ.get('CT_FLUX_VLLM_VERSION', '0.31.0').strip() or '0.31.0'\n"
    "VLLM_OMNI_REF = os.environ.get('CT_VLLM_OMNI_REF', '389a02a149ced03a9550368565571fff9d1d99f3').strip() or '389a02a149ced03a9550368565571fff9d1d99f3'\n\n"
    "FK_WORK = Path('/content/flux-klein')\n"
    "VENV_DIR = FK_WORK / 'venv'\n"
    "MODELS_ROOT = FK_WORK / 'model'\n"
    "MODEL_DIR = MODELS_ROOT / MODEL_REPO.split('/')[-1]\n"
    "PUP_MODEL_DIR = FK_WORK / 'pup-model'\n"
    "LOCAL_LORA_DIR = FK_WORK / 'loras'\n"
    "DRIVE_INPUTS_DIR = Path(DATA_DIR) / 'inputs'\n"
    "DRIVE_OUTPUTS_DIR = Path(DATA_DIR) / 'outputs'\n"
    "DRIVE_LORAS_DIR = Path(DATA_DIR) / 'loras'\n\n"
    "FLASHINFER_DRIVE_DIR = Path(DATA_DIR) / 'flashinfer-cache'\n"
    "FLASHINFER_CACHE_DIR = FK_WORK / 'flashinfer-cache'\n"
    "TRITON_DRIVE_DIR = Path(DATA_DIR) / 'triton-cache'\n"
    "TRITON_CACHE_DIR = FK_WORK / 'triton-cache'\n"
    "VLLM_DRIVE_DIR = Path(DATA_DIR) / 'vllm-cache'\n"
    "VLLM_CACHE_DIR = FK_WORK / 'vllm-cache'\n\n"
    "for _d in (\n"
    "    FK_WORK, MODELS_ROOT, LOCAL_LORA_DIR,\n"
    "    DRIVE_INPUTS_DIR, DRIVE_OUTPUTS_DIR, DRIVE_LORAS_DIR,\n"
    "    FLASHINFER_CACHE_DIR, TRITON_CACHE_DIR, VLLM_CACHE_DIR,\n"
    "):\n"
    "    _d.mkdir(parents=True, exist_ok=True)\n\n"
    "def fklog(msg):\n"
    "    print(time.strftime('[%F %T] [flux-klein] ') + msg, flush=True)\n\n"
    + embed_source("FK_WORK", "colab_cache.py", "templates/colab_cache.py")
    + r"""if str(FK_WORK) not in sys.path:
    sys.path.insert(0, str(FK_WORK))
from colab_cache import restore_cache, save_cache

try:
    _gpu_arch = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
except Exception:
    _gpu_arch = "90"
CACHE_KEY = f"vllm-{VLLM_VERSION}-flux3-{VLLM_OMNI_REF[:8]}-sm{_gpu_arch.replace('.', '')}"
CACHE_ARCHIVES = Path(DATA_DIR) / "cache-archives"
CACHE_PAIRS = (
    (FLASHINFER_CACHE_DIR, FLASHINFER_DRIVE_DIR),
    (TRITON_CACHE_DIR, TRITON_DRIVE_DIR),
    (VLLM_CACHE_DIR, VLLM_DRIVE_DIR),
)
for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        restore_cache(_local_c, _legacy_c, CACHE_ARCHIVES, CACHE_KEY, fklog)
    except (OSError, ValueError) as _err:
        fklog(f"Cache restore warning ({_local_c.name}): {_err}; continuing with local SSD cache")

def run_logged(cmd, log_path, env=None, cwd=None, check=True):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(str(x) for x in cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True)
    if check and res.returncode != 0:
        tail = "".join(open(log_path, errors="replace").readlines()[-45:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(str(x) for x in cmd[:4])}\n{tail}")
    return res

fklog(
    f"GPU detected: {ct_gpu_stat('name,memory.total,compute_cap')} | "
    f"repo={MODEL_REPO} variant={VARIANT} pup={'on' if ENABLE_PUP else 'off'}"
)

if not shutil.which("uv"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)

def prepare_env():
    env_log = FK_WORK / "env-setup.log"
    use_prims_ts = _gpu_arch.startswith("10.") and VARIANT in ("fp8r", "nvfp4")
    marker = VENV_DIR / f".ct_flux_env_{VLLM_VERSION}_{VLLM_OMNI_REF[:10]}_{int(use_prims_ts)}.ok"
    vllm_bin = VENV_DIR / "bin" / "vllm"
    py_bin = VENV_DIR / "bin" / "python"
    if marker.is_file() and vllm_bin.is_file() and py_bin.is_file():
        fklog("Reusing existing vLLM-Omni (flux3) virtualenv")
        return VENV_DIR

    fklog(f"Installing vllm=={VLLM_VERSION} + vllm-omni@{VLLM_OMNI_REF[:10]} (log: {env_log})...")
    uv_bin = shutil.which("uv") or "uv"
    run_logged([uv_bin, "venv", "--python", "3.12", str(VENV_DIR), "--clear"], env_log)
    omni_pkg = (
        f"vllm-omni[prims-ts] @ git+https://github.com/vllm-project/vllm-omni.git@{VLLM_OMNI_REF}"
        if use_prims_ts
        else f"vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@{VLLM_OMNI_REF}"
    )
    run_logged(
        [
            uv_bin, "pip", "install", "--python", str(py_bin),
            f"vllm=={VLLM_VERSION}",
            omni_pkg,
            "pillow", "huggingface_hub", "hf_xet",
        ],
        env_log,
    )
    marker.write_text("ok\n", encoding="utf-8")
    fklog("vLLM-Omni (flux3) virtualenv ready")
    return VENV_DIR

def _cleanup_hf_temp():
    for _p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
        shutil.rmtree(_p, ignore_errors=True)
    for _mp in MODELS_ROOT.glob("*/.cache"):
        shutil.rmtree(_mp, ignore_errors=True)
    shutil.rmtree(PUP_MODEL_DIR / ".cache", ignore_errors=True)

def prepare_model():
    dl_log = FK_WORK / "model-download.log"
    norm_variant = "" if VARIANT in ("", "bf16", "default", "none") else VARIANT
    ready_stamp = MODEL_DIR / f".ct_model_ready_{norm_variant or 'bf16'}"
    pup_stamp = PUP_MODEL_DIR / ".ct_pup_ready"

    need_flux = not (
        ready_stamp.is_file()
        and (MODEL_DIR / "config.json").is_file()
        and (MODEL_DIR / "ae.safetensors").is_file()
    )
    need_pup = ENABLE_PUP and not (pup_stamp.is_file() and (PUP_MODEL_DIR / "config.json").is_file())
    if not need_flux and not need_pup:
        fklog(f"Reusing staged local SSD checkpoint at {MODEL_DIR} (variant={norm_variant or 'bf16'})")
        return MODEL_DIR

    if need_flux and MODEL_REPO.startswith("black-forest-labs/") and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            f"{MODEL_REPO} is gated on Hugging Face (FLUX Kommunity License). "
            f"Accept the license at https://huggingface.co/{MODEL_REPO} and pass `--hf-token hf_...`, "
            "export `HF_TOKEN`, or save your token to `MyDrive/colab-llm/hf_token.txt`."
        )

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_xet"], check=True)
    _cleanup_hf_temp()
    free_gb = shutil.disk_usage("/content").free / 1e9
    fklog(f"Staging checkpoints onto local SSD (free SSD: {free_gb:.1f} GB; log: {dl_log})...")
    dl_env = {**os.environ, "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    dl_env.pop("HF_XET_HIGH_PERFORMANCE", None)

    dl_script = r'''
import os, shutil, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

model_repo, model_dir_s, variant, need_flux_s, enable_pup_s, pup_repo, pup_dir_s = sys.argv[1:8]
model_dir = Path(model_dir_s)
pup_dir = Path(pup_dir_s)
token = os.environ.get("HF_TOKEN") or None
api = HfApi(token=token)

def clean():
    for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
        shutil.rmtree(p, ignore_errors=True)
    shutil.rmtree(model_dir / ".cache", ignore_errors=True)
    shutil.rmtree(pup_dir / ".cache", ignore_errors=True)

def download_repo(repo_id, local_dir, allow_patterns=None):
    try:
        kwargs = {"repo_id": repo_id, "local_dir": str(local_dir), "token": token, "max_workers": 4}
        if allow_patterns is not None:
            kwargs["allow_patterns"] = allow_patterns
        snapshot_download(**kwargs)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if any(k in msg for k in ("401", "403", "GatedRepoError", "restricted", "authorized list")):
            raise SystemExit(
                f"Hugging Face gated access denied for {repo_id} ({type(exc).__name__}).\n"
                f"1) Visit https://huggingface.co/{repo_id} and agree to the FLUX Kommunity License;\n"
                "2) Ensure HF_TOKEN (or MyDrive/colab-llm/hf_token.txt) belongs to that authorized account, "
                "and if using a Fine-grained Token, enable 'Read access to contents of all public gated repos you can access'."
            )
        raise

if need_flux_s == "1":
    try:
        info = api.model_info(model_repo, token=token)
    except Exception as exc:
        raise SystemExit(
            f"Failed to query {model_repo} on Hugging Face ({exc}). "
            f"Verify HF_TOKEN and gated license access at https://huggingface.co/{model_repo}"
        )
    files = {s.rfilename for s in (info.siblings or [])}
    model_dir.mkdir(parents=True, exist_ok=True)
    if "ae.safetensors" in files and "config.json" in files:
        allow = [
            "config.json", "manifest.json", "modular_model_index.json",
            "LICENSE*", "NOTICE*", "ae.safetensors", "text_encoder/*",
        ]
        if not variant:
            allow.append("model.safetensors")
        else:
            req_cfg = f"variants/{variant}/config.json"
            if req_cfg not in files:
                avail = sorted({f.split("/")[1] for f in files if f.startswith("variants/") and f.endswith("/config.json")})
                raise SystemExit(f"Variant {variant!r} not found in {model_repo}; available variants: {avail}")
            allow.append(f"variants/{variant}/*")
        print(f"[dl] Downloading {model_repo} (variant={variant or 'bf16'}) -> {model_dir}", flush=True)
        download_repo(model_repo, model_dir, allow_patterns=allow)
    else:
        print(f"[dl] Downloading full Diffusers checkpoint {model_repo} -> {model_dir}", flush=True)
        download_repo(model_repo, model_dir)
    clean()
    (model_dir / f".ct_model_ready_{variant or 'bf16'}").write_text(f"{model_repo}:{variant or 'bf16'}\n", encoding="utf-8")

if enable_pup_s == "1":
    pup_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dl] Downloading prompt upsampler {pup_repo} -> {pup_dir}", flush=True)
    download_repo(pup_repo, pup_dir)
    clean()
    (pup_dir / ".ct_pup_ready").write_text(f"{pup_repo}\n", encoding="utf-8")
'''
    try:
        run_logged(
            [
                sys.executable, "-c", dl_script,
                MODEL_REPO, str(MODEL_DIR), norm_variant,
                "1" if need_flux else "0",
                "1" if need_pup else "0",
                PUP_REPO, str(PUP_MODEL_DIR),
            ],
            dl_log,
            env=dl_env,
        )
    finally:
        _cleanup_hf_temp()
    fklog(f"Checkpoints staged on local SSD (free SSD: {shutil.disk_usage('/content').free / 1e9:.1f} GB)")
    return MODEL_DIR

with ThreadPoolExecutor(max_workers=2) as _pool:
    _f_env = _pool.submit(prepare_env)
    _f_model = _pool.submit(prepare_model)
    while not (_f_env.done() and _f_model.done()):
        for _fut in (_f_env, _f_model):
            if _fut.done() and _fut.exception():
                raise _fut.exception()
        fklog(
            f"Setup heartbeat | env={'done' if _f_env.done() else 'running'} "
            f"model={'done' if _f_model.done() else 'running'} | "
            f"SSD free={shutil.disk_usage('/content').free / 1e9:.1f} GB"
        )
        time.sleep(20)
    _f_env.result()
    _f_model.result()
FK_SETUP_OK = True
fklog("Environment and local SSD model staging complete")
"""
)

FLUX_KLEIN_SERVICE = (
    "# @title 2. Service -- start vLLM-Omni FLUX 3 Image Klein (and optional PUP upsampler) on loopback\n"
    "if not globals().get('FK_SETUP_OK'):\n"
    "    raise RuntimeError('FLUX Klein setup cell failed before completing model staging; check HF_TOKEN gated repo access in setup traceback above.')\n"
    + embed_source("FK_WORK", "colab_asgi_common.py", "templates/colab_asgi_common.py")
    + embed_source("FK_WORK", "ct_flux_klein_middleware.py", "templates/flux_klein_middleware.py")
    + r"""import ct_flux_klein_middleware as _fk_mw

PRELOADED_LORA_LOCAL = ""
if LORA_PATH_SPEC:
    _staged = _fk_mw.resolve_and_stage_lora(
        {"path": LORA_PATH_SPEC, "scale": float(LORA_SCALE)},
        data_dir=Path(DATA_DIR),
        local_lora_dir=LOCAL_LORA_DIR,
    )
    PRELOADED_LORA_LOCAL = _staged["path"]
    fklog(f"Staged preload LoRA adapter: {PRELOADED_LORA_LOCAL} (scale={LORA_SCALE})")

SERVED_NAMES = list(dict.fromkeys([MODEL_REPO, MODEL_ALIAS, "flux-klein"]))

VLLM_ENV = {
    "PATH": f"{VENV_DIR}/bin:/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
    "PYTHONPATH": f"{FK_WORK}:{os.environ.get('PYTHONPATH', '')}",
    "VIRTUAL_ENV": str(VENV_DIR),
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "FLASHINFER_WORKSPACE_BASE": str(FLASHINFER_CACHE_DIR),
    "TRITON_CACHE_DIR": str(TRITON_CACHE_DIR),
    "VLLM_CACHE_ROOT": str(VLLM_CACHE_DIR),
    "TORCHINDUCTOR_CACHE_DIR": str(VLLM_CACHE_DIR / "inductor"),
    "CT_API_TOKEN": API_TOKEN,
    "CT_SERVICE_PORT": str(SERVICE_PORT),
    "CT_DATA_DIR": str(DATA_DIR),
    "CT_FLUX_WORK_DIR": str(FK_WORK),
    "CT_FLUX_MODEL_DIR": str(MODEL_DIR),
    "CT_FLUX_PUP_MODEL_DIR": str(PUP_MODEL_DIR),
    "CT_FLUX_LOCAL_LORA_DIR": str(LOCAL_LORA_DIR),
    "CT_FLUX_MODEL": MODEL_REPO,
    "CT_FLUX_ALIAS": MODEL_ALIAS,
    "CT_FLUX_VARIANT": VARIANT,
    "CT_FLUX_ENABLE_PUP": "1" if ENABLE_PUP else "0",
    "CT_FLUX_PUP_PORT": str(PUP_PORT),
    "CT_FLUX_PRELOADED_LORA": PRELOADED_LORA_LOCAL,
}
if FP8_ATTENTION:
    VLLM_ENV.pop("DIFFUSION_ATTENTION_BACKEND", None)
if _gpu_arch.startswith("12.") and VARIANT in ("fp8r", "nvfp4"):
    VLLM_ENV["VLLM_DISABLED_KERNELS"] = "FlashInferFP8ScaledMMLinearKernel"

SERVER_ARGS = [
    str(VENV_DIR / "bin" / "vllm"), "serve", str(MODEL_DIR),
    "--omni",
    "--served-model-name", *SERVED_NAMES,
    "--host", "127.0.0.1",
    "--port", str(SERVICE_PORT),
    "--api-key", API_TOKEN,
    "--allowed-local-media-path", str(FK_WORK),
    "--middleware", "ct_flux_klein_middleware.FluxKleinMiddleware",
]
if VARIANT not in ("", "bf16", "default", "none"):
    SERVER_ARGS.extend([
        "--stage-overrides",
        json.dumps({"0": {"additional_config": {"variant": VARIANT}}}),
    ])
if PRELOADED_LORA_LOCAL:
    SERVER_ARGS.extend(["--lora-path", PRELOADED_LORA_LOCAL, "--lora-scale", str(LORA_SCALE)])
if FP8_ATTENTION:
    SERVER_ARGS.extend([
        "--diffusion-attention-config",
        '{"default":{"backend":"FLASHINFER_ATTN","quant":{"dtype_qk":"fp8_e4m3"}}}',
    ])

PUP_ARGS = [
    str(VENV_DIR / "bin" / "vllm"), "serve", str(PUP_MODEL_DIR),
    "--served-model-name", "flux-3-image-klein-pup", PUP_REPO,
    "--host", "127.0.0.1",
    "--port", str(PUP_PORT),
    "--api-key", API_TOKEN,
    "--max-model-len", "16384",
    "--max-num-seqs", "16",
    "--gpu-memory-utilization", str(PUP_GPU_MEM),
    "--reasoning-parser", "qwen3",
]

# Reclaim GPU VRAM from any leftover workers of an earlier notebook execution on the same VM
subprocess.run(["pkill", "-f", f"{VENV_DIR}/bin/"], check=False)
time.sleep(2)
subprocess.run(["pkill", "-9", "-f", f"{VENV_DIR}/bin/|DiffusionWorker|EngineCore"], check=False)
time.sleep(1)
for _stale_log in (f"{WORK}/service.log", f"{WORK}/pup.log"):
    try:
        Path(_stale_log).unlink(missing_ok=True)
    except OSError:
        pass

def start_service():
    if "service" in PROCS:
        ct_stop_proc(PROCS["service"][0])
    subprocess.run(["pkill", "-9", "-f", f"port {SERVICE_PORT}"], check=False)
    time.sleep(1)
    fklog(f"Starting vLLM-Omni FLUX 3 server on 127.0.0.1:{SERVICE_PORT} (model={MODEL_REPO}, variant={VARIANT})")
    return ct_spawn("service", SERVER_ARGS, env=VLLM_ENV, cwd=str(FK_WORK))

def start_pup():
    if "pup" in PROCS:
        ct_stop_proc(PROCS["pup"][0])
    subprocess.run(["pkill", "-9", "-f", f"port {PUP_PORT}"], check=False)
    time.sleep(1)
    fklog(f"Starting Prompt Upsampler (PUP) vLLM server on 127.0.0.1:{PUP_PORT}")
    return ct_spawn("pup", PUP_ARGS, env=VLLM_ENV, cwd=str(FK_WORK))

PROCS["service"] = (start_service(), start_service)
"""
)

FLUX_KLEIN_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- verify auth, run warmup generation, archive JIT kernels & supervise
if not globals().get("FK_SETUP_OK") or "service" not in PROCS:
    raise RuntimeError("FLUX Klein service did not start because an earlier notebook cell failed; see traceback above.")
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def http_json(url, key=None, payload=None, timeout=30):
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, headers=headers, data=None if payload is None else json.dumps(payload).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else {})
    except urllib.error.HTTPError as err:
        return err.code, {}

def auth_check(base, key):
    for candidate, expected in ((None, 401), ("invalid-token-probe", 401), (key, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Auth test failed on {base}/v1/models: expected {expected}, got {code}")

def _abort_startup(msg):
    for _p, _ in list(PROCS.values()):
        ct_stop_proc(_p)
    raise RuntimeError(msg)

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
deadline = time.monotonic() + 2400
last_notice = 0
while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        _abort_startup("vLLM-Omni FLUX 3 server exited during startup:\n" + ct_tail("service", 80))
    try:
        code_main, _ = http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)
        if code_main == 200:
            break
    except Exception:
        pass
    if time.monotonic() - last_notice >= 30:
        last_line = (ct_tail("service", 3).strip().splitlines() or ["starting..."])[-1]
        fklog(f"Waiting for FLUX 3 init | GPU: {ct_gpu_stat()} | {last_line[:100]}")
        last_notice = time.monotonic()
    time.sleep(5)
else:
    _abort_startup("Timed out waiting for vLLM-Omni FLUX 3 server:\n" + ct_tail("service", 80))

fklog(f"FLUX 3 server ready on {LOCAL_BASE} | GPU: {ct_gpu_stat()}")

if ENABLE_PUP:
    PROCS["pup"] = (start_pup(), start_pup)
    last_notice = 0
    while time.monotonic() < deadline:
        if PROCS["service"][0].poll() is not None:
            _abort_startup("vLLM-Omni FLUX 3 server exited while starting PUP:\n" + ct_tail("service", 80))
        if PROCS["pup"][0].poll() is not None:
            _abort_startup("PUP vLLM server exited during startup:\n" + ct_tail("pup", 80))
        try:
            code_pup, _ = http_json(f"http://127.0.0.1:{PUP_PORT}/health", API_TOKEN, timeout=10)
            if code_pup == 200:
                break
        except Exception:
            pass
        if time.monotonic() - last_notice >= 30:
            last_line = (ct_tail("pup", 3).strip().splitlines() or ["starting..."])[-1]
            fklog(f"Waiting for PUP init | GPU: {ct_gpu_stat()} | {last_line[:100]}")
            last_notice = time.monotonic()
        time.sleep(5)
    else:
        _abort_startup("Timed out waiting for PUP vLLM server:\n" + ct_tail("pup", 80))
    fklog(f"PUP upsampler ready on 127.0.0.1:{PUP_PORT} | GPU: {ct_gpu_stat()}")

auth_check(LOCAL_BASE, API_TOKEN)

# Fast 2-step 512x512 warmup generation to verify DiT + Qwen3-VL + VAE and populate Triton/FlashInfer caches
try:
    w_code, w_body = http_json(
        LOCAL_BASE + "/v1/images/generations",
        API_TOKEN,
        payload={
            "model": MODEL_REPO,
            "prompt": "a clean minimalist geometric sphere on a soft neutral backdrop",
            "size": "512x512",
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": 2,
            "seed": 42,
        },
        timeout=300,
    )
    if w_code == 200 and (w_body.get("data") or [{}])[0].get("b64_json"):
        fklog("Warmup T2I generation succeeded (512x512, 2 steps)")
    else:
        fklog(f"Warmup T2I returned HTTP {w_code}: {str(w_body)[:160]}")
except Exception as werr:
    fklog(f"Warmup T2I warning: {werr}")

for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        save_cache(_local_c, CACHE_ARCHIVES, CACHE_KEY, fklog)
    except (OSError, ValueError) as _err:
        fklog(f"Cache archive save warning ({_local_c.name}): {_err}")

ct_public_gate(
    PUBLIC_URL,
    lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
    lambda: auth_check(PUBLIC_URL, API_TOKEN),
    fklog,
    timeout=120,
    interval=8,
)
fklog(f"Public endpoint verified: {PUBLIC_URL}/v1/models")

print(f"{SERVICE} is up: {PUBLIC_URL}")
print("CT_READY", flush=True)

def _fk_healthy():
    if ENABLE_PUP:
        if PROCS["pup"][0].poll() is not None:
            return False
        if http_json(f"http://127.0.0.1:{PUP_PORT}/health", API_TOKEN, timeout=10)[0] != 200:
            return False
    return http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200

ct_watchdog(
    _fk_healthy,
    start_service,
    fklog,
    dead_markers=("replica-0 is dead", "EngineCore encountered an issue"),
    heartbeat=lambda up: fklog(f"heartbeat | up={up}s | GPU: {ct_gpu_stat()}"),
)
'''

# --------------------------------------------------------------------------- qwen-image (G4 default / H100 / A100, vLLM-Omni Qwen-Image-2.1 T2I + 1-4 Ref RGBA I2I)
QWEN_IMAGE_CFG = dict(
    name="qwen-image",
    service_port=8091,
    auth="builtin",
    public_port=8091,
    health_path="/health",
    ready_timeout=3600,
)

QWEN_IMAGE_SETUP = (
    "# @title 1. Setup -- install vLLM + vLLM-Omni (Qwen-Image-2.1) & stage checkpoint on local NVMe SSD\n"
    "from concurrent.futures import ThreadPoolExecutor\n"
    "from pathlib import Path\n\n"
    "QI_SETUP_OK = False\n"
    "MODEL_REPO = os.environ.get('CT_QWEN_IMAGE_MODEL', 'Qwen/Qwen-Image-2.1').strip() or 'Qwen/Qwen-Image-2.1'\n"
    "MODEL_ALIAS = os.environ.get('CT_QWEN_IMAGE_ALIAS', 'qwen-image-2.1').strip() or 'qwen-image-2.1'\n"
    "QUANTIZATION = os.environ.get('CT_QWEN_IMAGE_QUANTIZATION', 'none').strip().lower() or 'none'\n"
    "PREFIX_KV_DTYPE = os.environ.get('CT_QWEN_IMAGE_PREFIX_KV_DTYPE', 'auto').strip().lower() or 'auto'\n"
    "STEP_EXECUTION = os.environ.get('CT_QWEN_IMAGE_STEP_EXECUTION', '0').strip() in ('1', 'true', 'True', 'yes')\n"
    "VAE_TILING = os.environ.get('CT_QWEN_IMAGE_VAE_TILING', '1').strip() not in ('0', 'false', 'False', 'no')\n"
    "_raw_mns = os.environ.get('CT_QWEN_IMAGE_MAX_NUM_SEQS', '').strip()\n"
    "MAX_NUM_SEQS = int(_raw_mns) if _raw_mns else (8 if STEP_EXECUTION else 4)\n"
    "LORA_PATH_SPEC = os.environ.get('CT_QWEN_IMAGE_LORA_PATH', '').strip()\n"
    "LORA_SCALE = os.environ.get('CT_QWEN_IMAGE_LORA_SCALE', '1.0').strip() or '1.0'\n"
    "VLLM_VERSION = os.environ.get('CT_QWEN_IMAGE_VLLM_VERSION', '0.31.0').strip() or '0.31.0'\n"
    "VLLM_OMNI_REF = os.environ.get('CT_VLLM_OMNI_REF', '389a02a149ced03a9550368565571fff9d1d99f3').strip() or '389a02a149ced03a9550368565571fff9d1d99f3'\n\n"
    "QI_WORK = Path('/content/qwen-image')\n"
    "VENV_DIR = QI_WORK / 'venv'\n"
    "MODELS_ROOT = QI_WORK / 'model'\n"
    "MODEL_DIR = MODELS_ROOT / MODEL_REPO.split('/')[-1]\n"
    "LOCAL_LORA_DIR = QI_WORK / 'loras'\n"
    "DRIVE_INPUTS_DIR = Path(DATA_DIR) / 'inputs'\n"
    "DRIVE_OUTPUTS_DIR = Path(DATA_DIR) / 'outputs'\n"
    "DRIVE_LORAS_DIR = Path(DATA_DIR) / 'loras'\n\n"
    "FLASHINFER_DRIVE_DIR = Path(DATA_DIR) / 'flashinfer-cache'\n"
    "FLASHINFER_CACHE_DIR = QI_WORK / 'flashinfer-cache'\n"
    "TRITON_DRIVE_DIR = Path(DATA_DIR) / 'triton-cache'\n"
    "TRITON_CACHE_DIR = QI_WORK / 'triton-cache'\n"
    "VLLM_DRIVE_DIR = Path(DATA_DIR) / 'vllm-cache'\n"
    "VLLM_CACHE_DIR = QI_WORK / 'vllm-cache'\n\n"
    "for _d in (\n"
    "    QI_WORK, MODELS_ROOT, LOCAL_LORA_DIR,\n"
    "    DRIVE_INPUTS_DIR, DRIVE_OUTPUTS_DIR, DRIVE_LORAS_DIR,\n"
    "    FLASHINFER_CACHE_DIR, TRITON_CACHE_DIR, VLLM_CACHE_DIR,\n"
    "):\n"
    "    _d.mkdir(parents=True, exist_ok=True)\n\n"
    "def qilog(msg):\n"
    "    print(time.strftime('[%F %T] [qwen-image] ') + msg, flush=True)\n\n"
    + embed_source("QI_WORK", "colab_cache.py", "templates/colab_cache.py")
    + r"""if str(QI_WORK) not in sys.path:
    sys.path.insert(0, str(QI_WORK))
from colab_cache import restore_cache, save_cache

try:
    _gpu_arch = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True
    ).strip().splitlines()[0]
except Exception:
    _gpu_arch = "90"
CACHE_KEY = f"vllm-{VLLM_VERSION}-qwenimg21-{VLLM_OMNI_REF[:8]}-sm{_gpu_arch.replace('.', '')}"
CACHE_ARCHIVES = Path(DATA_DIR) / "cache-archives"
CACHE_PAIRS = (
    (FLASHINFER_CACHE_DIR, FLASHINFER_DRIVE_DIR),
    (TRITON_CACHE_DIR, TRITON_DRIVE_DIR),
    (VLLM_CACHE_DIR, VLLM_DRIVE_DIR),
)
for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        restore_cache(_local_c, _legacy_c, CACHE_ARCHIVES, CACHE_KEY, qilog)
    except (OSError, ValueError) as _err:
        qilog(f"Cache restore warning ({_local_c.name}): {_err}; continuing with local SSD cache")

def run_logged(cmd, log_path, env=None, cwd=None, check=True):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(str(x) for x in cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True)
    if check and res.returncode != 0:
        tail = "".join(open(log_path, errors="replace").readlines()[-45:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(str(x) for x in cmd[:4])}\n{tail}")
    return res

qilog(
    f"GPU detected: {ct_gpu_stat('name,memory.total,compute_cap')} | "
    f"repo={MODEL_REPO} quant={QUANTIZATION} step_exec={STEP_EXECUTION} max_seqs={MAX_NUM_SEQS}"
)

if not shutil.which("uv"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)

def prepare_env():
    env_log = QI_WORK / "env-setup.log"
    marker = VENV_DIR / f".ct_qwen_img_env_{VLLM_VERSION}_{VLLM_OMNI_REF[:10]}.ok"
    vllm_bin = VENV_DIR / "bin" / "vllm"
    py_bin = VENV_DIR / "bin" / "python"
    if marker.is_file() and vllm_bin.is_file() and py_bin.is_file():
        qilog("Reusing existing vLLM-Omni (Qwen-Image-2.1) virtualenv")
        return VENV_DIR

    qilog(f"Installing vllm=={VLLM_VERSION} + vllm-omni@{VLLM_OMNI_REF[:10]} (log: {env_log})...")
    uv_bin = shutil.which("uv") or "uv"
    run_logged([uv_bin, "venv", "--python", "3.12", str(VENV_DIR), "--clear"], env_log)
    run_logged(
        [
            uv_bin, "pip", "install", "--python", str(py_bin),
            f"vllm=={VLLM_VERSION}",
            f"vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@{VLLM_OMNI_REF}",
            "pillow", "huggingface_hub", "hf_xet",
        ],
        env_log,
    )
    marker.write_text("ok\n", encoding="utf-8")
    qilog("vLLM-Omni (Qwen-Image-2.1) virtualenv ready")
    return VENV_DIR

def _cleanup_hf_temp():
    for _p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
        shutil.rmtree(_p, ignore_errors=True)
    for _mp in MODELS_ROOT.glob("*/.cache"):
        shutil.rmtree(_mp, ignore_errors=True)

def prepare_model():
    dl_log = QI_WORK / "model-download.log"
    ready_stamp = MODEL_DIR / ".ct_model_ready"
    if (
        ready_stamp.is_file()
        and (MODEL_DIR / "model_index.json").is_file()
        and (MODEL_DIR / "vae" / "diffusion_pytorch_model.safetensors").is_file()
    ):
        qilog(f"Reusing staged local SSD checkpoint at {MODEL_DIR}")
        return MODEL_DIR

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_xet"], check=True)
    _cleanup_hf_temp()
    free_gb = shutil.disk_usage("/content").free / 1e9
    qilog(f"Staging {MODEL_REPO} (~33.1 GB) onto local SSD (free SSD: {free_gb:.1f} GB; log: {dl_log})...")
    dl_env = {**os.environ, "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    dl_env.pop("HF_XET_HIGH_PERFORMANCE", None)

    dl_script = r'''
import os, shutil, sys
from pathlib import Path
from huggingface_hub import snapshot_download

model_repo, model_dir_s = sys.argv[1:3]
model_dir = Path(model_dir_s)
token = os.environ.get("HF_TOKEN") or None
model_dir.mkdir(parents=True, exist_ok=True)
print(f"[dl] Downloading {model_repo} -> {model_dir}", flush=True)
snapshot_download(
    repo_id=model_repo,
    local_dir=str(model_dir),
    token=token,
    max_workers=4,
    ignore_patterns=["*.md", ".gitattributes"],
)
for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
    shutil.rmtree(p, ignore_errors=True)
shutil.rmtree(model_dir / ".cache", ignore_errors=True)
(model_dir / ".ct_model_ready").write_text(f"{model_repo}\n", encoding="utf-8")
'''
    try:
        run_logged([sys.executable, "-c", dl_script, MODEL_REPO, str(MODEL_DIR)], dl_log, env=dl_env)
    finally:
        _cleanup_hf_temp()
    qilog(f"Checkpoint staged on local SSD (free SSD: {shutil.disk_usage('/content').free / 1e9:.1f} GB)")
    return MODEL_DIR

with ThreadPoolExecutor(max_workers=2) as _pool:
    _f_env = _pool.submit(prepare_env)
    _f_model = _pool.submit(prepare_model)
    while not (_f_env.done() and _f_model.done()):
        for _fut in (_f_env, _f_model):
            if _fut.done() and _fut.exception():
                raise _fut.exception()
        qilog(
            f"Setup heartbeat | env={'done' if _f_env.done() else 'running'} "
            f"model={'done' if _f_model.done() else 'running'} | "
            f"SSD free={shutil.disk_usage('/content').free / 1e9:.1f} GB"
        )
        time.sleep(20)
    _f_env.result()
    _f_model.result()
QI_SETUP_OK = True
qilog("Environment and local SSD model staging complete")
"""
)

QWEN_IMAGE_SERVICE = (
    "# @title 2. Service -- start vLLM-Omni Qwen-Image-2.1 server on loopback\n"
    "if not globals().get('QI_SETUP_OK'):\n"
    "    raise RuntimeError('Qwen-Image-2.1 setup cell failed before completing model staging; check setup traceback above.')\n"
    + embed_source("QI_WORK", "colab_asgi_common.py", "templates/colab_asgi_common.py")
    + embed_source("QI_WORK", "ct_qwen_image_middleware.py", "templates/qwen_image_middleware.py")
    + r"""import ct_qwen_image_middleware as _qi_mw

PRELOADED_LORA_LOCAL = ""
if LORA_PATH_SPEC:
    _staged = _qi_mw.resolve_and_stage_lora(
        {"path": LORA_PATH_SPEC, "scale": float(LORA_SCALE)},
        data_dir=Path(DATA_DIR),
        local_lora_dir=LOCAL_LORA_DIR,
    )
    PRELOADED_LORA_LOCAL = _staged["path"]
    qilog(f"Staged preload LoRA adapter: {PRELOADED_LORA_LOCAL} (scale={LORA_SCALE})")

SERVED_NAMES = list(dict.fromkeys([MODEL_REPO, MODEL_ALIAS, "qwen-image", "qwen-image-2.1"]))

VLLM_ENV = {
    "PATH": f"{VENV_DIR}/bin:/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
    "PYTHONPATH": f"{QI_WORK}:{os.environ.get('PYTHONPATH', '')}",
    "VIRTUAL_ENV": str(VENV_DIR),
    "VLLM_NO_USAGE_STATS": "1",
    "DO_NOT_TRACK": "1",
    "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "FLASHINFER_WORKSPACE_BASE": str(FLASHINFER_CACHE_DIR),
    "TRITON_CACHE_DIR": str(TRITON_CACHE_DIR),
    "VLLM_CACHE_ROOT": str(VLLM_CACHE_DIR),
    "TORCHINDUCTOR_CACHE_DIR": str(VLLM_CACHE_DIR / "inductor"),
    "CT_API_TOKEN": API_TOKEN,
    "CT_SERVICE_PORT": str(SERVICE_PORT),
    "CT_DATA_DIR": str(DATA_DIR),
    "CT_QWEN_IMAGE_WORK_DIR": str(QI_WORK),
    "CT_QWEN_IMAGE_MODEL_DIR": str(MODEL_DIR),
    "CT_QWEN_IMAGE_LOCAL_LORA_DIR": str(LOCAL_LORA_DIR),
    "CT_QWEN_IMAGE_MODEL": MODEL_REPO,
    "CT_QWEN_IMAGE_ALIAS": MODEL_ALIAS,
    "CT_QWEN_IMAGE_QUANTIZATION": QUANTIZATION,
    "CT_QWEN_IMAGE_PREFIX_KV_DTYPE": PREFIX_KV_DTYPE,
    "CT_QWEN_IMAGE_STEP_EXECUTION": "1" if STEP_EXECUTION else "0",
    "CT_QWEN_IMAGE_VAE_TILING": "1" if VAE_TILING else "0",
    "CT_QWEN_IMAGE_MAX_NUM_SEQS": str(MAX_NUM_SEQS),
    "CT_QWEN_IMAGE_PRELOADED_LORA": PRELOADED_LORA_LOCAL,
}
if _gpu_arch.startswith("12.") and QUANTIZATION != "none":
    VLLM_ENV["VLLM_DISABLED_KERNELS"] = "FlashInferFP8ScaledMMLinearKernel"

SERVER_ARGS = [
    str(VENV_DIR / "bin" / "vllm"), "serve", str(MODEL_DIR),
    "--omni",
    "--served-model-name", *SERVED_NAMES,
    "--host", "127.0.0.1",
    "--port", str(SERVICE_PORT),
    "--api-key", API_TOKEN,
    "--max-num-seqs", str(MAX_NUM_SEQS),
    "--allowed-local-media-path", str(QI_WORK),
    "--middleware", "ct_qwen_image_middleware.QwenImageMiddleware",
]
if VAE_TILING:
    SERVER_ARGS.append("--vae-use-tiling")
if STEP_EXECUTION:
    SERVER_ARGS.append("--step-execution")

if QUANTIZATION == "fp8":
    SERVER_ARGS.extend([
        "--diffusion-quantization-config",
        json.dumps({"transformer": {"method": "fp8", "ignored_layers": ["img_mlp"]}}),
    ])
elif QUANTIZATION == "fp8-text":
    SERVER_ARGS.extend([
        "--diffusion-quantization-config",
        json.dumps({"text_encoder": {"method": "fp8"}}),
    ])
elif QUANTIZATION == "fp8-all":
    SERVER_ARGS.extend([
        "--diffusion-quantization-config",
        json.dumps({
            "transformer": {"method": "fp8", "ignored_layers": ["img_mlp"]},
            "text_encoder": {"method": "fp8"},
        }),
    ])

if PREFIX_KV_DTYPE in ("fp8_v", "fp8"):
    SERVER_ARGS.extend([
        "--stage-overrides",
        json.dumps({"0": {"extras": {"prefix_kv_cache_dtype": PREFIX_KV_DTYPE}}}),
    ])

if PRELOADED_LORA_LOCAL:
    SERVER_ARGS.extend(["--lora-path", PRELOADED_LORA_LOCAL, "--lora-scale", str(LORA_SCALE)])

# Reclaim GPU VRAM from any leftover workers of an earlier notebook execution on the same VM
subprocess.run(["pkill", "-f", f"{VENV_DIR}/bin/"], check=False)
time.sleep(2)
subprocess.run(["pkill", "-9", "-f", f"{VENV_DIR}/bin/|DiffusionWorker|EngineCore"], check=False)
time.sleep(1)
try:
    Path(f"{WORK}/service.log").unlink(missing_ok=True)
except OSError:
    pass

def start_service():
    if "service" in PROCS:
        ct_stop_proc(PROCS["service"][0])
    subprocess.run(["pkill", "-9", "-f", f"port {SERVICE_PORT}"], check=False)
    time.sleep(1)
    qilog(
        f"Starting vLLM-Omni Qwen-Image-2.1 server on 127.0.0.1:{SERVICE_PORT} "
        f"(model={MODEL_REPO}, quant={QUANTIZATION}, step_exec={STEP_EXECUTION})"
    )
    return ct_spawn("service", SERVER_ARGS, env=VLLM_ENV, cwd=str(QI_WORK))

PROCS["service"] = (start_service(), start_service)
"""
)

QWEN_IMAGE_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- verify auth, run warmup generation, archive JIT kernels & supervise
if not globals().get("QI_SETUP_OK") or "service" not in PROCS:
    raise RuntimeError("Qwen-Image-2.1 service did not start because an earlier notebook cell failed; see traceback above.")
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def http_json(url, key=None, payload=None, timeout=30):
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, headers=headers, data=None if payload is None else json.dumps(payload).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else {})
    except urllib.error.HTTPError as err:
        return err.code, {}

def auth_check(base, key):
    for candidate, expected in ((None, 401), ("invalid-token-probe", 401), (key, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Auth test failed on {base}/v1/models: expected {expected}, got {code}")

def _abort_startup(msg):
    for _p, _ in list(PROCS.values()):
        ct_stop_proc(_p)
    raise RuntimeError(msg)

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
deadline = time.monotonic() + 2400
last_notice = 0
while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        _abort_startup("vLLM-Omni Qwen-Image-2.1 server exited during startup:\n" + ct_tail("service", 80))
    try:
        code_main, _ = http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)
        if code_main == 200:
            break
    except Exception:
        pass
    if time.monotonic() - last_notice >= 30:
        last_line = (ct_tail("service", 3).strip().splitlines() or ["starting..."])[-1]
        qilog(f"Waiting for Qwen-Image-2.1 init | GPU: {ct_gpu_stat()} | {last_line[:100]}")
        last_notice = time.monotonic()
    time.sleep(5)
else:
    _abort_startup("Timed out waiting for vLLM-Omni Qwen-Image-2.1 server:\n" + ct_tail("service", 80))

qilog(f"Qwen-Image-2.1 server ready on {LOCAL_BASE} | GPU: {ct_gpu_stat()}")
auth_check(LOCAL_BASE, API_TOKEN)

# Fast 2-step 512x512 warmup generation to verify DiT + Qwen3-VL-8B + 4ch RGBA VAE and populate JIT caches
try:
    w_code, w_body = http_json(
        LOCAL_BASE + "/v1/images/generations",
        API_TOKEN,
        payload={
            "model": MODEL_REPO,
            "prompt": "a ceramic teapot on a wooden table, soft window light",
            "size": "512x512",
            "n": 1,
            "response_format": "b64_json",
            "num_inference_steps": 2,
            "true_cfg_scale": 1.0,
            "seed": 42,
        },
        timeout=300,
    )
    if w_code == 200 and (w_body.get("data") or [{}])[0].get("b64_json"):
        qilog("Warmup T2I generation succeeded (512x512, 2 steps, true_cfg_scale=1.0)")
    else:
        qilog(f"Warmup T2I returned HTTP {w_code}: {str(w_body)[:160]}")
except Exception as werr:
    qilog(f"Warmup T2I warning: {werr}")

for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        save_cache(_local_c, CACHE_ARCHIVES, CACHE_KEY, qilog)
    except (OSError, ValueError) as _err:
        qilog(f"Cache archive save warning ({_local_c.name}): {_err}")

ct_public_gate(
    PUBLIC_URL,
    lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
    lambda: auth_check(PUBLIC_URL, API_TOKEN),
    qilog,
    timeout=120,
    interval=8,
)
qilog(f"Public endpoint verified: {PUBLIC_URL}/v1/models")

print(f"{SERVICE} is up: {PUBLIC_URL}")
print("CT_READY", flush=True)

ct_watchdog(
    lambda: http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200,
    start_service,
    qilog,
    dead_markers=("replica-0 is dead", "EngineCore encountered an issue"),
    heartbeat=lambda up: qilog(f"heartbeat | up={up}s | GPU: {ct_gpu_stat()}"),
)
'''

# --------------------------------------------------------------------------- deepseek-coder (G4/H100/A100, llama.cpp DSV4.1 + Engram SSD mmap + Vision)
DEEPSEEK_CODER_CFG = dict(name="deepseek-coder", service_port=8001, auth="builtin", public_port=8001,
                          health_path="/health", ready_timeout=5400)

DEEPSEEK_CODER_SETUP = (
    r'''# @title 1. Setup -- parallel llama.cpp CUDA build/cache + 183.7 GB Coder GGUF + Engram shard-6 symlink + hardware-max context plan
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DS_WORK = Path(os.environ.get("CT_DS_WORK_DIR", "/content/deepseek-coder"))
DS_WORK.mkdir(parents=True, exist_ok=True)
(DS_WORK / "deepseek_coder_runtime.py").write_text('''
    + repr((ROOT / "src/colabllm/lib/deepseek_coder_runtime.py").read_text())
    + r''', encoding="utf-8")
if str(DS_WORK) not in sys.path:
    sys.path.insert(0, str(DS_WORK))

from deepseek_coder_runtime import (
    DEFAULT_ALIAS,
    DEFAULT_LLAMA_COMMIT,
    DEFAULT_LLAMA_REPO,
    DEFAULT_QUANT,
    DEFAULT_REPO,
    build_server_args,
    detect_gpu_info,
    detect_host_ram_mib,
    evict_gguf_page_cache,
    looks_like_oom,
    plan_hardware_config,
    prepare_llama_runtime,
    prepare_patched_chat_template,
    should_evict_page_cache,
    stage_model_and_projector,
    write_json_atomic,
)

def dslog(msg):
    print(time.strftime("[%F %T] [deepseek-coder] ") + msg, flush=True)

if MOUNT_DRIVE and not Path("/content/drive/MyDrive").is_dir():
    raise RuntimeError("Drive is not mounted. Run via `colabllm deepseek-coder up`.")

DS_SERVICE_DIR = Path(DATA_DIR)
DS_SERVICE_DIR.mkdir(parents=True, exist_ok=True)
DS_DRIVE_BACKUP_ROOT = (
    Path("/content/drive/MyDrive/models") if (MOUNT_DRIVE and Path("/content/drive/MyDrive").is_dir()) else None
)
DS_RUNTIME_CACHE = DS_SERVICE_DIR / "runtime-cache"

DS_CFG = {
    "repo": os.environ.get("CT_DS_REPO", DEFAULT_REPO),
    "quant": os.environ.get("CT_DS_QUANT", DEFAULT_QUANT),
    "alias": os.environ.get("CT_DS_ALIAS", DEFAULT_ALIAS),
    "context": os.environ.get("CT_DS_CONTEXT", "auto"),
    "parallel": int(os.environ.get("CT_DS_PARALLEL", "1")),
    "cache_type": os.environ.get("CT_DS_CACHE_TYPE", "f16"),
    "batch": int(os.environ.get("CT_DS_BATCH", "4096")),
    "ubatch": os.environ.get("CT_DS_UBATCH", "auto"),
    "cpu_moe": os.environ.get("CT_DS_CPU_MOE", "auto"),
    "reasoning": os.environ.get("CT_DS_REASONING", "on"),
    "reasoning_effort": os.environ.get("CT_DS_REASONING_EFFORT", ""),
    "enable_vision": os.environ.get("CT_DS_VISION", "1") != "0",
    "llama_repo": os.environ.get("CT_LLAMA_REPO", DEFAULT_LLAMA_REPO),
    "llama_commit": os.environ.get("CT_LLAMA_COMMIT", DEFAULT_LLAMA_COMMIT),
}

DS_GPU = detect_gpu_info()
DS_RAM_MIB = detect_host_ram_mib()
HW_PLAN = plan_hardware_config(
    vram_mib=DS_GPU["vram_mib"],
    ram_mib=DS_RAM_MIB,
    requested_context=DS_CFG["context"],
    parallel=DS_CFG["parallel"],
    cache_type=DS_CFG["cache_type"],
    enable_vision=DS_CFG["enable_vision"],
    batch=DS_CFG["batch"],
    requested_ubatch=DS_CFG["ubatch"],
    requested_cpu_moe=DS_CFG["cpu_moe"],
)
dslog(
    f"Hardware plan: gpu={DS_GPU['name']} ({DS_GPU['vram_mib']} MiB VRAM, sm_{DS_GPU['cuda_arch'] or 'cpu'}) "
    f"ram={DS_RAM_MIB} MiB -> mode={HW_PLAN['mode']} context={HW_PLAN['context']} "
    f"(kv={HW_PLAN['kv_cache_mib']} MiB, ubatch={HW_PLAN['ubatch']}, n_cpu_moe={HW_PLAN['n_cpu_moe']}, "
    f"engram_in_ram={HW_PLAN['engram_in_ram']})"
)

with ThreadPoolExecutor(max_workers=2) as _pool:
    _f_runtime = _pool.submit(
        prepare_llama_runtime,
        work_dir=DS_WORK,
        cache_root=DS_RUNTIME_CACHE,
        llama_repo=DS_CFG["llama_repo"],
        llama_commit=DS_CFG["llama_commit"],
        gpu_info=DS_GPU,
        log=dslog,
    )
    _f_model = _pool.submit(
        stage_model_and_projector,
        repo=DS_CFG["repo"],
        quant=DS_CFG["quant"],
        enable_vision=DS_CFG["enable_vision"],
        service_dir=DS_SERVICE_DIR,
        work_dir=DS_WORK,
        drive_backup_root=DS_DRIVE_BACKUP_ROOT,
        log=dslog,
    )
    while not (_f_runtime.done() and _f_model.done()):
        for _fut in (_f_runtime, _f_model):
            if _fut.done() and _fut.exception():
                raise _fut.exception()
        _free_gb = shutil.disk_usage(DS_WORK).free / 1e9
        dslog(
            f"Setup heartbeat | runtime={'done' if _f_runtime.done() else 'running'} "
            f"model={'done' if _f_model.done() else 'running'} | free_ssd={_free_gb:.1f} GB"
        )
        time.sleep(30)
    BIN_DIR, LLAMA_ENV, _ = _f_runtime.result()
    MODEL_PATH, PROJECTOR_PATH = _f_model.result()

CHAT_TEMPLATE_PATH = prepare_patched_chat_template(MODEL_PATH, DS_WORK / "chat_template.jinja")
dslog(f"Setup complete: model={MODEL_PATH} projector={PROJECTOR_PATH} chat_template={CHAT_TEMPLATE_PATH}")
'''
)

DEEPSEEK_CODER_SERVICE = r'''# @title 2. Service -- llama-server on 127.0.0.1 (Engram host-RAM or SSD lazy read, hardware-max context, --api-key-file)
KEY_FILE = DS_WORK / "api_key.txt"
KEY_FILE.write_text(API_TOKEN + "\n", encoding="utf-8")
KEY_FILE.chmod(0o600)
SPAWN_LOG_OFFSET = 0

def start_service():
    global SPAWN_LOG_OFFSET
    subprocess.run(["pkill", "-f", str(BIN_DIR / "llama-server")], check=False, timeout=15)
    args = build_server_args(
        server_bin=BIN_DIR / "llama-server",
        model_path=MODEL_PATH,
        alias=DS_CFG["alias"],
        port=SERVICE_PORT,
        key_file=KEY_FILE,
        plan=HW_PLAN,
        projector_path=PROJECTOR_PATH,
        chat_template_file=CHAT_TEMPLATE_PATH,
        reasoning=DS_CFG["reasoning"],
        reasoning_effort=DS_CFG["reasoning_effort"],
    )
    SPAWN_LOG_OFFSET = ct_log_size("llama-server")
    engram_loc = "host-ram" if HW_PLAN.get("engram_in_ram") else "ssd-lazy"
    dslog(
        f"Starting llama-server on 127.0.0.1:{SERVICE_PORT} | ctx={HW_PLAN['context']} "
        f"mode={HW_PLAN['mode']} engram={engram_loc} kv={HW_PLAN['cache_type']} "
        f"batch={HW_PLAN['batch']}/{HW_PLAN['ubatch']} reasoning={DS_CFG['reasoning']}"
    )
    return ct_spawn("llama-server", args, env=LLAMA_ENV)

PROCS["service"] = (start_service(), start_service)
'''

DEEPSEEK_CODER_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- auto-stepdown on OOM, verify auth/context/vision, smoke completion & supervise
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def http_json(url, key=None, payload=None, timeout=15):
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url,
        headers=headers,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        return err.code, {}

def auth_check(base, key):
    for candidate, expected in ((None, 401), ("invalid-deployment-test-key", 401), (key, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Authentication test failed on {base}: expected {expected}, got {code}")

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
ladder = list(HW_PLAN.get("context_ladder") or [HW_PLAN["context"]])
ub_ladder = list(HW_PLAN.get("ubatch_ladder") or [HW_PLAN["ubatch"]])
rung_idx = 0
ub_idx = 0
deadline = time.monotonic() + 2400
last_notice = 0.0

while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        recent_log = ct_log_since("llama-server", SPAWN_LOG_OFFSET)
        if looks_like_oom(recent_log) and ub_idx + 1 < len(ub_ladder):
            # Prefer a smaller micro-batch (slower prefill) over a smaller context window
            ub_idx += 1
            prev_ub = HW_PLAN["ubatch"]
            HW_PLAN["ubatch"] = ub_ladder[ub_idx]
            dslog(f"Startup OOM at ubatch={prev_ub}; stepping down to ubatch={HW_PLAN['ubatch']} and retrying")
            PROCS["service"] = (start_service(), start_service)
            continue
        if HW_PLAN.get("context_auto") and rung_idx + 1 < len(ladder) and looks_like_oom(recent_log):
            rung_idx += 1
            prev_ctx = HW_PLAN["context"]
            HW_PLAN["context"] = ladder[rung_idx]
            dslog(f"Startup OOM at ctx={prev_ctx}; stepping down to ctx={HW_PLAN['context']} and retrying")
            PROCS["service"] = (start_service(), start_service)
            continue
        raise RuntimeError("llama-server exited during load:\n" + ct_tail("llama-server", 60))
    try:
        if http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200:
            break
    except (OSError, ValueError):
        pass
    if time.monotonic() - last_notice > 45:
        dslog(f"Waiting for model load / warmup (ctx={HW_PLAN['context']}) | GPU: {ct_gpu_stat()}")
        last_notice = time.monotonic()
    time.sleep(5)
else:
    raise RuntimeError("Model load timeout:\n" + ct_tail("llama-server", 60))

if should_evict_page_cache(HW_PLAN):
    evict_gguf_page_cache(MODEL_PATH.parent, keep_engram=not HW_PLAN.get("engram_in_ram", False), log=dslog)

auth_check(LOCAL_BASE, API_TOKEN)
code, props = http_json(LOCAL_BASE + "/props", API_TOKEN, timeout=20)
allocated = (props.get("default_generation_settings") or {}).get("n_ctx")
if code != 200 or allocated != HW_PLAN["context"]:
    raise RuntimeError(f"Allocated context ({allocated}) differs from target ({HW_PLAN['context']})")
vision_enabled = bool((props.get("modalities") or {}).get("vision", False))
if PROJECTOR_PATH is not None and not vision_enabled:
    raise RuntimeError("Vision projector configured but server does not advertise vision")
dslog(
    f"Local verified: auth=401/401/200, n_ctx={allocated}, vision={vision_enabled}, "
    f"mode={HW_PLAN['mode']}, engram_in_ram={HW_PLAN['engram_in_ram']}"
)

ct_public_gate(
    PUBLIC_URL,
    lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
    lambda: auth_check(PUBLIC_URL, API_TOKEN),
    dslog,
    timeout=180,
    interval=8,
)
smoke_payload = {
    "model": DS_CFG["alias"],
    "messages": [{"role": "user", "content": "Reply with the exact token OK"}],
    "max_tokens": 32,
    "temperature": 0,
}
if DS_CFG["reasoning"] != "off":
    smoke_payload["chat_template_kwargs"] = {"enable_thinking": False}
code, resp = http_json(
    PUBLIC_URL + "/v1/chat/completions",
    API_TOKEN,
    smoke_payload,
    timeout=240,
)
if code != 200:
    raise RuntimeError(f"Public smoke completion failed: HTTP {code}")
msg_obj = ((resp.get("choices") or [{}])[0].get("message") or {})
reply = msg_obj.get("content") or msg_obj.get("reasoning_content") or ""
dslog(f"PUBLIC VERIFIED: {PUBLIC_URL}/v1 | n_ctx={allocated} | reply={reply.strip()[:60]!r}")

ver_rec = {
    "service": SERVICE,
    "verified_at": time.strftime("%FT%T%z"),
    "base_url": PUBLIC_URL + "/v1",
    "repo": DS_CFG["repo"],
    "alias": DS_CFG["alias"],
    "mode": HW_PLAN["mode"],
    "n_cpu_moe": HW_PLAN["n_cpu_moe"],
    "engram_in_ram": HW_PLAN["engram_in_ram"],
    "default_reasoning": DS_CFG["reasoning"],
    "context_allocated": allocated,
    "vision_enabled": vision_enabled,
    "reply": reply[:120],
}
write_json_atomic(DS_SERVICE_DIR / "last-verification.json", ver_rec)

print(f"{SERVICE} is up: {PUBLIC_URL} (n_ctx={allocated}, mode={HW_PLAN['mode']})")
print("CT_READY", flush=True)

def ds_heartbeat(uptime):
    gpu_mem = ct_gpu_stat("memory.used,memory.total,utilization.gpu")
    try:
        write_json_atomic(
            DS_SERVICE_DIR / "last-status.json",
            {
                "service": SERVICE,
                "kernel_pid": os.getpid(),
                "server_pid": PROCS["service"][0].pid,
                "tunnel_pid": PROCS[TUNNEL_PROCESS_KEY][0].pid,
                "context": HW_PLAN["context"],
                "mode": HW_PLAN["mode"],
                "engram_in_ram": HW_PLAN["engram_in_ram"],
                "alias": DS_CFG["alias"],
                "url": PUBLIC_URL + "/v1",
                "updated_at": time.strftime("%FT%T%z"),
                "uptime_sec": uptime,
                "gpu_memory": gpu_mem,
            },
        )
    except OSError:
        pass
    dslog(f"heartbeat | up={uptime}s | n_ctx={HW_PLAN['context']} | GPU: {gpu_mem}")

ct_watchdog(
    lambda: http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200,
    start_service,
    dslog,
    log_name="llama-server",
    heartbeat=ds_heartbeat,
)
'''

# --------------------------------------------------------------------------- drama3 (G4 / H100 / A100 · SGLang-Omni Fish Audio Drama 3 TTS & Novel Narration)
DRAMA3_CFG = dict(
    name="drama3",
    service_port=8000,
    auth="builtin",
    public_port=8000,
    health_path="/health",
    ready_timeout=3600,
)

DRAMA3_SETUP = r'''# @title 1. Setup -- install SGLang-Omni 0.2.0 + stage Fish Audio Drama 3 & S2-Pro codec on local SSD
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VARIANT_REPOS = {
    "bf16": "fishaudio/drama-3",
    "drama-3": "fishaudio/drama-3",
    "int4": "fishaudio/drama-3-int4",
    "drama-3-int4": "fishaudio/drama-3-int4",
}
CODEC_REPO = "fishaudio/s2-pro"
CODEC_REVISION = "1de9996b6be38b745688de084d87a5633f714e4e"

def dlog(msg):
    print(time.strftime("[%F %T] [drama3] ") + msg, flush=True)

_gpu_raw = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
    text=True,
).strip().splitlines()[0]
_gpu_parts = [p.strip() for p in _gpu_raw.split(",")]
GPU_NAME = _gpu_parts[0]
GPU_VRAM_MIB = int(float(_gpu_parts[1])) if len(_gpu_parts) > 1 else 0
GPU_ARCH = (_gpu_parts[2] if len(_gpu_parts) > 2 else "8.0").replace(".", "")

RAW_VARIANT = os.environ.get("CT_DRAMA3_VARIANT", "auto").strip().lower()
EXPLICIT_MODEL = os.environ.get("CT_DRAMA3_MODEL", "").strip()
if EXPLICIT_MODEL:
    MODEL_REPO = EXPLICIT_MODEL
    VARIANT = "int4" if "int4" in EXPLICIT_MODEL.lower() else "bf16"
elif RAW_VARIANT in VARIANT_REPOS:
    MODEL_REPO = VARIANT_REPOS[RAW_VARIANT]
    VARIANT = "int4" if "int4" in RAW_VARIANT else "bf16"
else:
    VARIANT = "bf16" if GPU_VRAM_MIB >= 72000 else "int4"
    MODEL_REPO = VARIANT_REPOS[VARIANT]

MODEL_ALIAS = os.environ.get("CT_DRAMA3_ALIAS", "drama3").strip() or "drama3"
ENGINE_GPU_MEM = float(os.environ.get("CT_DRAMA3_ENGINE_GPU_MEM", "0.70"))
VOCODER_GPU_MEM = float(os.environ.get("CT_DRAMA3_VOCODER_GPU_MEM", "0.10"))
MAX_RUNNING_REQS = int(os.environ.get("CT_DRAMA3_MAX_RUNNING_REQS", "16"))
CONTEXT_LENGTH = int(os.environ.get("CT_DRAMA3_CONTEXT_LENGTH", "16384"))
BATCH_MAX_ITEMS = int(os.environ.get("CT_DRAMA3_BATCH_MAX_ITEMS", "32"))
MAX_REF_SEC = float(os.environ.get("CT_DRAMA3_MAX_REF_SEC", "60.0"))
SGL_OMNI_VER = os.environ.get("CT_DRAMA3_SGL_OMNI_VERSION", "0.2.0").strip() or "0.2.0"

D3_WORK = Path("/content/drama3")
VENV_DIR = D3_WORK / "venv"
MODELS_ROOT = D3_WORK / "models"
MODEL_DIR = MODELS_ROOT / MODEL_REPO.split("/")[-1]
CODEC_DIR = D3_WORK / "codec"
SPEAKERS_DIR = D3_WORK / "speakers"
DRIVE_VOICES_DIR = Path(DATA_DIR) / "voices"
DRIVE_OUTPUT_DIR = Path(DATA_DIR) / "outputs"
DEFAULT_REF_WAV = D3_WORK / "default_ref.wav"
DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)

# Local NVMe SSD compiler caches archived as single tarballs on Google Drive
FLASHINFER_CACHE_DIR = D3_WORK / "flashinfer-cache"
TRITON_CACHE_DIR = D3_WORK / "triton-cache"
INDUCTOR_CACHE_DIR = D3_WORK / "inductor-cache"
FLASHINFER_LEGACY_DIR = Path(DATA_DIR) / "flashinfer-cache"
TRITON_LEGACY_DIR = Path(DATA_DIR) / "triton-cache"
INDUCTOR_LEGACY_DIR = Path(DATA_DIR) / "inductor-cache"

for _d in (
    D3_WORK, MODELS_ROOT, CODEC_DIR, SPEAKERS_DIR, DRIVE_VOICES_DIR, DRIVE_OUTPUT_DIR,
    FLASHINFER_CACHE_DIR, TRITON_CACHE_DIR, INDUCTOR_CACHE_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)

(D3_WORK / "colab_cache.py").write_text(__DRAMA3_CACHE_SOURCE__, encoding="utf-8")
if str(D3_WORK) not in sys.path:
    sys.path.insert(0, str(D3_WORK))
from colab_cache import restore_cache, save_cache

CACHE_KEY = f"sglomni-{SGL_OMNI_VER}-py312-{VARIANT}-sm{GPU_ARCH}"
CACHE_ARCHIVES = Path(DATA_DIR) / "cache-archives"
CACHE_PAIRS = (
    (FLASHINFER_CACHE_DIR, FLASHINFER_LEGACY_DIR),
    (TRITON_CACHE_DIR, TRITON_LEGACY_DIR),
    (INDUCTOR_CACHE_DIR, INDUCTOR_LEGACY_DIR),
)
for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        restore_cache(_local_c, _legacy_c, CACHE_ARCHIVES, CACHE_KEY, dlog)
    except (OSError, ValueError) as err:
        dlog(f"Cache restore warning ({_local_c.name}): {err}; continuing with local cache")

dlog("Restoring registered voice samples from Drive")
subprocess.run(["cp", "-rn", f"{DRIVE_VOICES_DIR}/.", f"{SPEAKERS_DIR}/"], check=False)
dlog(
    f"GPU: {GPU_NAME} ({GPU_VRAM_MIB} MiB, sm_{GPU_ARCH}) | model={MODEL_REPO} ({VARIANT}) | "
    f"engine_mem={ENGINE_GPU_MEM} vocoder_mem={VOCODER_GPU_MEM} max_reqs={MAX_RUNNING_REQS} batch_max={BATCH_MAX_ITEMS}"
)

if not shutil.which("ffmpeg") or not shutil.which("sox"):
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq ffmpeg sox libsox-fmt-all >/dev/null 2>&1",
        shell=True,
        check=False,
    )
if not shutil.which("uv"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)

def run_logged(cmd, log_path, env=None, cwd=None, check=True):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(str(x) for x in cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True)
    if check and res.returncode != 0:
        tail = "".join(open(log_path, errors="replace").readlines()[-45:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(str(x) for x in cmd[:4])}\n{tail}")
    return res

def prepare_env():
    env_log = D3_WORK / "env-setup.log"
    marker = VENV_DIR / f".ct_drama3_env_{SGL_OMNI_VER}_ready"
    py_bin = VENV_DIR / "bin" / "python"
    sgl_bin = VENV_DIR / "bin" / "sgl-omni"
    if marker.is_file() and py_bin.is_file() and sgl_bin.is_file():
        dlog(f"Reusing existing sglang-omni=={SGL_OMNI_VER} virtualenv at {VENV_DIR}")
        return VENV_DIR

    dlog(f"Installing sglang-omni[fishaudio]=={SGL_OMNI_VER} in Python 3.12 venv (log: {env_log})...")
    uv_bin = shutil.which("uv") or "uv"
    overrides_file = D3_WORK / "uv-overrides.txt"
    # descript-audiotools pins protobuf<3.20 while sglang/nvidia-cutlass-dsl requires protobuf>=4.21.12;
    # sglang-omni upstream pyproject.toml resolves this via override-dependencies protobuf>=6.31.1,<7.0.0.
    overrides_file.write_text("protobuf>=6.31.1,<7.0.0\n", encoding="utf-8")
    run_logged([uv_bin, "venv", "--python", "3.12", str(VENV_DIR), "--clear"], env_log)
    run_logged(
        [
            uv_bin, "pip", "install",
            "--python", str(py_bin),
            "--prerelease=allow",
            "--overrides", str(overrides_file),
            f"sglang-omni[fishaudio]=={SGL_OMNI_VER}",
            "huggingface_hub",
            "hf_xet",
        ],
        env_log,
    )
    run_logged(
        [str(py_bin), "-c", "import torch, sglang, sglang_omni, dac, audiotools; print('imports-ok', torch.__version__)"],
        env_log,
    )
    marker.write_text(f"{SGL_OMNI_VER}\n", encoding="utf-8")
    dlog(f"sglang-omni[fishaudio]=={SGL_OMNI_VER} virtualenv ready")
    return VENV_DIR

def _cleanup_hf_temp():
    for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
        shutil.rmtree(p, ignore_errors=True)
    for mp in (MODEL_DIR / ".cache", CODEC_DIR / ".cache"):
        shutil.rmtree(mp, ignore_errors=True)

def _ensure_default_ref_wav():
    if DEFAULT_REF_WAV.is_file() and DEFAULT_REF_WAV.stat().st_size > 1024:
        return
    drive_copy = DRIVE_VOICES_DIR / "default_ref.wav"
    if drive_copy.is_file() and drive_copy.stat().st_size > 1024:
        shutil.copyfile(drive_copy, DEFAULT_REF_WAV)
        return
    for ref_url in (
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
        "https://raw.githubusercontent.com/vllm-project/vllm-omni/main/tests/assets/indextts2/ref_audio.wav",
    ):
        try:
            with urllib.request.urlopen(ref_url, timeout=60) as _src, open(DEFAULT_REF_WAV, "wb") as _dst:
                shutil.copyfileobj(_src, _dst)
            if DEFAULT_REF_WAV.is_file() and DEFAULT_REF_WAV.stat().st_size > 1024:
                break
        except Exception:
            pass
    if not DEFAULT_REF_WAV.is_file() or DEFAULT_REF_WAV.stat().st_size <= 1024:
        import math, struct, wave
        sr = 44100
        n = sr * 3
        with wave.open(str(DEFAULT_REF_WAV), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            samples = [int(10000 * math.sin(2 * math.pi * 220 * i / sr)) for i in range(n)]
            wf.writeframes(struct.pack(f"<{n}h", *samples))
    if not drive_copy.is_file():
        shutil.copyfile(DEFAULT_REF_WAV, drive_copy)

def prepare_model():
    dl_log = D3_WORK / "model-download.log"
    _ensure_default_ref_wav()
    marker = MODEL_DIR / ".ct_model_ready"
    codec_file = CODEC_DIR / "codec.pth"
    has_weights = any(MODEL_DIR.glob("*.safetensors"))
    if marker.is_file() and (MODEL_DIR / "config.json").is_file() and has_weights and codec_file.is_file():
        dlog(f"Reusing local SSD model {MODEL_DIR} and codec {codec_file}")
        return MODEL_DIR

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_xet"], check=True)
    _cleanup_hf_temp()
    free_gb = shutil.disk_usage("/content").free / 1e9
    dlog(f"Downloading {MODEL_REPO} + {CODEC_REPO} codec -> local SSD {D3_WORK} (free SSD: {free_gb:.1f} GB; log: {dl_log})...")
    dl_env = {**os.environ, "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0", "HF_HUB_ENABLE_HF_TRANSFER": "0"}
    dl_env.pop("HF_XET_HIGH_PERFORMANCE", None)
    dl_script = r"""
import shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

repo_id, model_dir, codec_repo, codec_rev, codec_dir = sys.argv[1:6]
model_path = Path(model_dir)
codec_path = Path(codec_dir)
model_path.mkdir(parents=True, exist_ok=True)
codec_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    local_dir=str(model_path),
    max_workers=8,
)
if not (codec_path / "codec.pth").is_file():
    hf_hub_download(
        repo_id=codec_repo,
        filename="codec.pth",
        revision=codec_rev,
        local_dir=str(codec_path),
    )
for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
    shutil.rmtree(p, ignore_errors=True)
for mp in (model_path / ".cache", codec_path / ".cache"):
    shutil.rmtree(mp, ignore_errors=True)
"""
    try:
        run_logged(
            [sys.executable, "-c", dl_script, MODEL_REPO, str(MODEL_DIR), CODEC_REPO, CODEC_REVISION, str(CODEC_DIR)],
            dl_log,
            env=dl_env,
        )
    finally:
        _cleanup_hf_temp()

    if not (MODEL_DIR / "config.json").is_file() or not any(MODEL_DIR.glob("*.safetensors")) or not codec_file.is_file():
        raise RuntimeError(f"Model/codec download incomplete in {MODEL_DIR} / {CODEC_DIR}")
    marker.write_text(MODEL_REPO + "\n", encoding="utf-8")
    total_gb = (
        sum(p.stat().st_size for p in MODEL_DIR.rglob("*") if p.is_file())
        + codec_file.stat().st_size
    ) / 1e9
    dlog(f"Drama 3 weights + S2-Pro codec ready on local SSD ({total_gb:.2f} GB)")
    return MODEL_DIR

with ThreadPoolExecutor(max_workers=2) as pool:
    f_env = pool.submit(prepare_env)
    f_mod = pool.submit(prepare_model)
    while not (f_env.done() and f_mod.done()):
        for fut in (f_env, f_mod):
            if fut.done() and fut.exception():
                raise fut.exception()
        free_gb = shutil.disk_usage("/content").free / 1e9
        dlog(
            f"Setup heartbeat | env={'done' if f_env.done() else 'running'} "
            f"model={'done' if f_mod.done() else 'running'} | free_ssd={free_gb:.1f} GB"
        )
        time.sleep(25)
    VENV_DIR = f_env.result()
    MODEL_DIR = f_mod.result()

dlog(f"Setup complete: venv={VENV_DIR} model={MODEL_DIR} codec={CODEC_DIR} drive_data={DATA_DIR}")
'''.replace("__DRAMA3_CACHE_SOURCE__", repr((ROOT / "templates/colab_cache.py").read_text()))

DRAMA3_SERVICE = r'''# @title 2. Service -- SGLang-Omni Fish Audio Drama 3 on 127.0.0.1:8000 (Bearer Auth + Drive Voices & Batch TTS)
(D3_WORK / "colab_asgi_common.py").write_text(__DRAMA3_COMMON_SOURCE__, encoding="utf-8")
(D3_WORK / "ct_drama3_middleware.py").write_text(__DRAMA3_MW_SOURCE__, encoding="utf-8")

SGL_ENV = {
    "PATH": f"{VENV_DIR}/bin:/usr/local/cuda/bin:{os.environ.get('PATH', '')}",
    "PYTHONPATH": f"{D3_WORK}:{os.environ.get('PYTHONPATH', '')}",
    "VIRTUAL_ENV": str(VENV_DIR),
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    "FLASHINFER_WORKSPACE_BASE": str(FLASHINFER_CACHE_DIR),
    "TRITON_CACHE_DIR": str(TRITON_CACHE_DIR),
    "TORCHINDUCTOR_CACHE_DIR": str(INDUCTOR_CACHE_DIR),
    "SPEAKER_SAMPLES_DIR": str(SPEAKERS_DIR),
    "CT_API_TOKEN": API_TOKEN,
    "CT_SERVICE_PORT": str(SERVICE_PORT),
    "CT_DATA_DIR": str(DATA_DIR),
    "CT_DRAMA3_MODEL": MODEL_REPO,
    "CT_DRAMA3_ALIAS": MODEL_ALIAS,
    "CT_DRAMA3_MAX_REF_SEC": str(MAX_REF_SEC),
    "CT_DEFAULT_REF_WAV": str(DEFAULT_REF_WAV),
    "CT_DEFAULT_REF_TEXT": DEFAULT_REF_TEXT,
}

SERVER_ARGS = [
    str(VENV_DIR / "bin" / "python"),
    str(D3_WORK / "ct_drama3_middleware.py"),
    "serve",
    "--model-path", str(MODEL_DIR),
    "--host", "127.0.0.1",
    "--port", str(SERVICE_PORT),
    "--model-name", MODEL_REPO,
    "--allowed-local-media-path", str(D3_WORK),
    "--tts-batch-max-items", str(BATCH_MAX_ITEMS),
    "--tts_engine.gpu_memory_fraction", str(ENGINE_GPU_MEM),
    "--vocoder.gpu_memory_fraction", str(VOCODER_GPU_MEM),
    "--vocoder.factory.codec_path", str(CODEC_DIR),
    "--tts_engine.factory.max_running_requests", str(MAX_RUNNING_REQS),
    "--tts_engine.factory.context_length", str(CONTEXT_LENGTH),
    "--preprocessing.factory.max_reference_audio_seconds", str(MAX_REF_SEC),
]

def start_service():
    subprocess.run(["pkill", "-9", "-f", "ct_drama3_middleware|sglang_omni"], check=False)
    time.sleep(1)
    dlog(f"Starting SGLang-Omni Drama 3 server on 127.0.0.1:{SERVICE_PORT} (model={MODEL_REPO})")
    return ct_spawn("service", SERVER_ARGS, env=SGL_ENV, cwd=str(D3_WORK))

PROCS["service"] = (start_service(), start_service)
'''.replace(
    "__DRAMA3_COMMON_SOURCE__", repr((ROOT / "templates/colab_asgi_common.py").read_text())).replace(
    "__DRAMA3_MW_SOURCE__", repr((ROOT / "templates/drama3_middleware.py").read_text()))

DRAMA3_KEEPALIVE = r'''# @title 5. Ready + keep-alive -- wait for CUDA graph capture, verify auth, register default voice, warmup & sync caches
BROWSER_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"

def http_json(url, key=None, payload=None, timeout=30):
    headers = {"Content-Type": "application/json", "User-Agent": BROWSER_UA}
    if key is not None:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(
        url, headers=headers, data=None if payload is None else json.dumps(payload).encode("utf-8")
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(raw) if raw.strip().startswith(("{", "[")) else {})
    except urllib.error.HTTPError as err:
        return err.code, {}

def auth_check(base, key):
    for candidate, expected in ((None, 401), ("invalid-token-probe", 401), (key, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Auth test failed on {base}/v1/models: expected {expected}, got {code}")

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
deadline = time.monotonic() + 2400
last_notice = 0
while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        raise RuntimeError("SGLang-Omni Drama 3 exited during startup:\n" + ct_tail("service", 90))
    try:
        code, _ = http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)
        if code == 200:
            break
    except Exception:
        pass
    if time.monotonic() - last_notice >= 25:
        gpu_stat = ct_gpu_stat()
        last_line = (ct_tail("service", 3).strip().splitlines() or ["starting..."])[-1]
        dlog(f"Waiting for Drama 3 pipeline & CUDA graphs | GPU: {gpu_stat} | {last_line[:100]}")
        last_notice = time.monotonic()
    time.sleep(5)
else:
    raise RuntimeError("Timed out waiting for SGLang-Omni Drama 3 server to become ready:\n" + ct_tail("service", 90))

auth_check(LOCAL_BASE, API_TOKEN)

# Register default 'alloy' voice preset if not yet saved
code, voices_data = http_json(LOCAL_BASE + "/v1/audio/voices", API_TOKEN, timeout=20)
existing_voices = set(voices_data.get("voices") or [])
for preset_voice in ("alloy",):
    if preset_voice not in existing_voices and DEFAULT_REF_WAV.is_file():
        v_code, _ = http_json(
            LOCAL_BASE + "/api/ct/drive-upload-voice",
            API_TOKEN,
            payload={
                "name": preset_voice,
                "use_default_ref": True,
                "consent": "builtin-default-reference",
                "speaker_description": "Default English reference voice for Fish Audio Drama 3",
            },
            timeout=90,
        )
        dlog(f"Registered preset voice '{preset_voice}': HTTP {v_code}")

# Warmup synthesis to populate Triton/FlashInfer JIT kernels before saving cache archives
warmup_req = urllib.request.Request(
    LOCAL_BASE + "/v1/audio/speech",
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_TOKEN}"},
    data=json.dumps({
        "model": MODEL_REPO,
        "input": "Hello, the Fish Audio Drama 3 speech service is ready.",
        "response_format": "wav",
        "max_new_tokens": 384,
    }, ensure_ascii=False).encode("utf-8"),
)
try:
    with urllib.request.urlopen(warmup_req, timeout=180) as wresp:
        wav_bytes = wresp.read()
        dlog(f"Warmup synthesis succeeded ({len(wav_bytes):,} bytes 44.1 kHz WAV)")
except Exception as werr:
    dlog(f"Warmup synthesis warning: {werr}")

for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        save_cache(_local_c, CACHE_ARCHIVES, CACHE_KEY, dlog)
    except (OSError, ValueError) as err:
        dlog(f"Cache archive save warning ({_local_c.name}): {err}")
subprocess.run(["cp", "-ru", f"{SPEAKERS_DIR}/.", f"{DRIVE_VOICES_DIR}/"], check=False)

ct_public_gate(
    PUBLIC_URL,
    lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
    lambda: auth_check(PUBLIC_URL, API_TOKEN),
    dlog,
    timeout=180,
    interval=8,
)
dlog(f"Public endpoint verified: {PUBLIC_URL}/v1/models")

print(f"{SERVICE} is up: {PUBLIC_URL} (model={MODEL_REPO})")
print("CT_READY", flush=True)

ct_watchdog(
    lambda: http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200,
    start_service,
    dlog,
    heartbeat=lambda up: dlog(f"heartbeat | up={up}s | GPU: {ct_gpu_stat()}"),
)
'''

# --------------------------------------------------------------------------- tangoflux (T4 / L4 / G4 · 44.1 kHz Stereo Text-to-Audio / Sound Effect Generation)
TANGOFLUX_CFG = dict(
    name="tangoflux",
    service_port=8000,
    auth="builtin",
    public_port=8000,
    health_path="/health",
    ready_timeout=2400,
)

TANGOFLUX_SETUP = r'''# @title 1. Setup -- install TangoFlux + stage weights & Flan-T5-Large on local NVMe SSD
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VARIANT_REPOS = {
    "crpo": "declare-lab/TangoFlux",
    "default": "declare-lab/TangoFlux",
    "tangoflux": "declare-lab/TangoFlux",
    "base": "declare-lab/TangoFlux-base",
    "tangoflux-base": "declare-lab/TangoFlux-base",
}
TANGOFLUX_GIT_URL = "https://github.com/declare-lab/TangoFlux.git"

def tflog(msg):
    print(time.strftime("[%F %T] [tangoflux] ") + msg, flush=True)

_gpu_raw = ct_gpu_stat("name,memory.total,compute_cap").splitlines()[0]
_gpu_parts = [p.strip() for p in _gpu_raw.split(",")]
GPU_NAME = _gpu_parts[0]
_vram_tok = _gpu_parts[1].split()[0] if len(_gpu_parts) > 1 and _gpu_parts[1].strip() else ""
GPU_VRAM_MIB = int(float(_vram_tok)) if _vram_tok.replace(".", "", 1).isdigit() else 0
GPU_ARCH = (_gpu_parts[2] if len(_gpu_parts) > 2 else "7.5").replace(".", "")

RAW_VARIANT = os.environ.get("CT_TANGOFLUX_VARIANT", "crpo").strip().lower()
EXPLICIT_MODEL = os.environ.get("CT_TANGOFLUX_MODEL", "").strip()
if EXPLICIT_MODEL:
    MODEL_REPO = EXPLICIT_MODEL
    VARIANT = "base" if "base" in EXPLICIT_MODEL.lower() else "crpo"
elif RAW_VARIANT in VARIANT_REPOS:
    MODEL_REPO = VARIANT_REPOS[RAW_VARIANT]
    VARIANT = "base" if "base" in RAW_VARIANT else "crpo"
else:
    MODEL_REPO = "declare-lab/TangoFlux"
    VARIANT = "crpo"

MODEL_ALIAS = os.environ.get("CT_TANGOFLUX_ALIAS", "tangoflux").strip() or "tangoflux"

TF_WORK = Path("/content/tangoflux")
VENV_DIR = TF_WORK / "venv"
SRC_DIR = TF_WORK / "src"
MODELS_ROOT = TF_WORK / "models"
MODEL_DIR = MODELS_ROOT / MODEL_REPO.split("/")[-1]
TEXT_ENCODER_DIR = MODELS_ROOT / "flan-t5-large"
DRIVE_OUTPUT_DIR = Path(DATA_DIR) / "outputs"

TRITON_CACHE_DIR = TF_WORK / "triton-cache"
INDUCTOR_CACHE_DIR = TF_WORK / "inductor-cache"
TRITON_LEGACY_DIR = Path(DATA_DIR) / "triton-cache"
INDUCTOR_LEGACY_DIR = Path(DATA_DIR) / "inductor-cache"

for _d in (TF_WORK, MODELS_ROOT, DRIVE_OUTPUT_DIR, TRITON_CACHE_DIR, INDUCTOR_CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

(TF_WORK / "colab_cache.py").write_text(__TANGOFLUX_CACHE_SOURCE__, encoding="utf-8")
if str(TF_WORK) not in sys.path:
    sys.path.insert(0, str(TF_WORK))
from colab_cache import restore_cache, save_cache

CACHE_KEY = f"tangoflux-diffusers0303-py{sys.version_info.major}{sys.version_info.minor}-sm{GPU_ARCH}"
CACHE_ARCHIVES = Path(DATA_DIR) / "cache-archives"
CACHE_PAIRS = (
    (TRITON_CACHE_DIR, TRITON_LEGACY_DIR),
    (INDUCTOR_CACHE_DIR, INDUCTOR_LEGACY_DIR),
)
for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        restore_cache(_local_c, _legacy_c, CACHE_ARCHIVES, CACHE_KEY, tflog)
    except (OSError, ValueError) as err:
        tflog(f"Cache restore warning ({_local_c.name}): {err}")

tflog(f"GPU: {GPU_NAME} ({GPU_VRAM_MIB} MiB, sm_{GPU_ARCH}) | model={MODEL_REPO} ({VARIANT})")

if not shutil.which("ffmpeg"):
    subprocess.run(
        "apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1",
        shell=True,
        check=False,
    )
if not shutil.which("uv"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], check=True)

def run_logged(cmd, log_path, env=None, cwd=None, check=True):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n$ {' '.join(str(x) for x in cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True, timeout=900)
    if check and res.returncode != 0:
        tail = "".join(open(log_path, errors="replace").readlines()[-45:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(str(x) for x in cmd[:4])}\n{tail}")
    return res

def prepare_env():
    env_log = TF_WORK / "env-setup.log"
    marker = VENV_DIR / ".ct_tangoflux_env_v2_ready"
    py_bin = VENV_DIR / "bin" / "python"
    if marker.is_file() and py_bin.is_file():
        tflog(f"Reusing existing TangoFlux virtualenv at {VENV_DIR}")
        return py_bin

    if (SRC_DIR / ".git").is_dir():
        run_logged(["git", "-C", str(SRC_DIR), "pull", "--ff-only"], env_log, check=False)
    else:
        shutil.rmtree(SRC_DIR, ignore_errors=True)
        run_logged(["git", "clone", "--depth", "1", TANGOFLUX_GIT_URL, str(SRC_DIR)], env_log)

    tflog(f"Creating TangoFlux venv inheriting Colab PyTorch (log: {env_log})...")
    uv_bin = shutil.which("uv") or "uv"
    run_logged([uv_bin, "venv", "--system-site-packages", str(VENV_DIR), "--clear"], env_log)
    if subprocess.run([str(py_bin), "-c", "import torch"], check=False).returncode != 0:
        run_logged([uv_bin, "pip", "install", "--python", str(py_bin), "torch"], env_log)
    run_logged(
        [
            uv_bin, "pip", "install",
            "--python", str(py_bin),
            "diffusers==0.30.3",
            "transformers>=4.44.2,<4.50.0",
            "safetensors>=0.4.3",
            "huggingface_hub>=0.25.0",
            "hf_xet",
            "sentencepiece>=0.2.0",
            "protobuf>=4.25.0,<7.0.0",
            "psutil",
            "packaging",
            "pyyaml",
            "tqdm",
            "uvicorn>=0.30.0",
            "starlette>=0.38.0",
        ],
        env_log,
    )
    run_logged(
        [uv_bin, "pip", "install", "--python", str(py_bin), "--no-deps", "accelerate>=0.34.2", str(SRC_DIR)],
        env_log,
    )
    check_env = dict(os.environ)
    check_env.update({"USE_TF": "0", "USE_FLAX": "0", "TRANSFORMERS_NO_TF": "1", "TRANSFORMERS_NO_FLAX": "1"})
    run_logged(
        [
            str(py_bin), "-c",
            "import importlib.machinery, sys, types\n"
            "try:\n"
            "    import datasets\n"
            "except ImportError:\n"
            "    _s = types.ModuleType('datasets')\n"
            "    _s.__spec__ = importlib.machinery.ModuleSpec('datasets', loader=None)\n"
            "    _s.load_dataset = None; _s.Audio = None\n"
            "    sys.modules['datasets'] = _s\n"
            "import torch, diffusers, transformers\n"
            "from tangoflux.model import TangoFlux\n"
            "print('imports-ok', torch.__version__, 'cuda=', torch.cuda.is_available(), diffusers.__version__)",
        ],
        env_log,
        env=check_env,
    )
    marker.write_text("v2\n", encoding="utf-8")
    tflog("TangoFlux virtualenv ready")
    return py_bin

def _cleanup_hf_temp():
    for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
        shutil.rmtree(p, ignore_errors=True)
    for mp in (MODEL_DIR / ".cache", TEXT_ENCODER_DIR / ".cache"):
        shutil.rmtree(mp, ignore_errors=True)

def stage_models(py_bin):
    dl_env = dict(os.environ)
    dl_env["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    dl_env["HF_HUB_DISABLE_TELEMETRY"] = "1"

    def _stage_main():
        if (MODEL_DIR / "tangoflux.safetensors").is_file() and (MODEL_DIR / "vae.safetensors").is_file():
            tflog(f"Reusing local SSD TangoFlux weights at {MODEL_DIR}")
            return
        tflog(f"Downloading {MODEL_REPO} -> {MODEL_DIR} ...")
        subprocess.run(
            [
                str(py_bin), "-c",
                "import os; from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id={MODEL_REPO!r}, local_dir={str(MODEL_DIR)!r}, "
                "allow_patterns=['config.json', 'tangoflux.safetensors', 'vae.safetensors'])",
            ],
            env=dl_env,
            check=True,
            timeout=1800,
        )

    def _stage_t5():
        if (TEXT_ENCODER_DIR / "model.safetensors").is_file() and (TEXT_ENCODER_DIR / "spiece.model").is_file():
            tflog(f"Reusing local SSD Flan-T5-Large weights at {TEXT_ENCODER_DIR}")
            return
        tflog(f"Downloading google/flan-t5-large (safetensors only) -> {TEXT_ENCODER_DIR} ...")
        subprocess.run(
            [
                str(py_bin), "-c",
                "from huggingface_hub import snapshot_download; "
                f"snapshot_download(repo_id='google/flan-t5-large', local_dir={str(TEXT_ENCODER_DIR)!r}, "
                "allow_patterns=['*.json', 'spiece.model', 'model.safetensors'])",
            ],
            env=dl_env,
            check=True,
            timeout=1800,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_main = pool.submit(_stage_main)
        f_t5 = pool.submit(_stage_t5)
        f_main.result()
        f_t5.result()
    _cleanup_hf_temp()

PY_BIN = prepare_env()
stage_models(PY_BIN)
TF_SETUP_OK = True
tflog("Setup complete")
'''.replace("__TANGOFLUX_CACHE_SOURCE__", repr((ROOT / "templates/colab_cache.py").read_text(encoding="utf-8")))

TANGOFLUX_SERVICE = r'''# @title 2. Service -- launch authenticated TangoFlux ASGI server on 127.0.0.1
from pathlib import Path
if not globals().get("TF_SETUP_OK"):
    raise RuntimeError("TangoFlux setup cell failed before completing model staging; check setup traceback above.")

(TF_WORK / "colab_asgi_common.py").write_text(__TANGOFLUX_ASGI_COMMON_SOURCE__, encoding="utf-8")
(TF_WORK / "tangoflux_server.py").write_text(__TANGOFLUX_SERVER_SOURCE__, encoding="utf-8")

def start_service():
    if not API_TOKEN:
        raise RuntimeError("CT_API_TOKEN is empty; refusing to start unauthenticated TangoFlux server")
    if "service" in PROCS:
        ct_stop_proc(PROCS["service"][0])
    subprocess.run(["pkill", "-9", "-f", "tangoflux_server.py"], check=False)
    time.sleep(1)
    svc_env = dict(os.environ)
    svc_env.update({
        "CT_API_TOKEN": API_TOKEN,
        "CT_SERVICE_PORT": str(SERVICE_PORT),
        "CT_DATA_DIR": str(DATA_DIR),
        "CT_TANGOFLUX_SERVED_MODEL": MODEL_REPO,
        "CT_TANGOFLUX_ALIAS": MODEL_ALIAS,
        "CT_TANGOFLUX_VARIANT": VARIANT,
        "CT_TANGOFLUX_MODEL_DIR": str(MODEL_DIR),
        "CT_TANGOFLUX_TEXT_ENCODER_DIR": str(TEXT_ENCODER_DIR),
        "TRITON_CACHE_DIR": str(TRITON_CACHE_DIR),
        "TORCHINDUCTOR_CACHE_DIR": str(INDUCTOR_CACHE_DIR),
        "USE_TF": "0",
        "USE_FLAX": "0",
        "TRANSFORMERS_NO_TF": "1",
        "TRANSFORMERS_NO_FLAX": "1",
        "PYTHONPATH": f"{TF_WORK}:{svc_env.get('PYTHONPATH', '')}".rstrip(":"),
    })
    proc = ct_spawn(
        "service",
        [str(VENV_DIR / "bin" / "python"), str(TF_WORK / "tangoflux_server.py")],
        env=svc_env,
        cwd=str(TF_WORK),
    )
    PROCS["service"] = (proc, start_service)
    return proc

start_service()
tflog(f"Spawned TangoFlux server (pid={PROCS['service'][0].pid}) on 127.0.0.1:{SERVICE_PORT}")
'''.replace(
    "__TANGOFLUX_ASGI_COMMON_SOURCE__", repr((ROOT / "templates/colab_asgi_common.py").read_text(encoding="utf-8"))
).replace(
    "__TANGOFLUX_SERVER_SOURCE__", repr((ROOT / "templates/tangoflux_server.py").read_text(encoding="utf-8"))
)

TANGOFLUX_KEEPALIVE = r'''# @title 5. Readiness gate, warmup SFX generation, cache archiving & watchdog
import urllib.error
import urllib.request
if not globals().get("TF_SETUP_OK") or "service" not in PROCS:
    raise RuntimeError("TangoFlux service did not start because an earlier notebook cell failed; see traceback above.")

def http_json(url, token="", payload=None, timeout=20):
    headers = {"User-Agent": "Mozilla/5.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            return r.status, (json.loads(body) if body.strip().startswith("{") else {})
    except urllib.error.HTTPError as e:
        return e.code, {}

def auth_check(base, token):
    for candidate, expected in (("", 401), ("invalid-probe-token", 401), (token, 200)):
        code, _ = http_json(base + "/v1/models", candidate)
        if code != expected:
            raise RuntimeError(f"Auth test failed on {base}/v1/models: expected {expected}, got {code}")

LOCAL_BASE = f"http://127.0.0.1:{SERVICE_PORT}"
deadline = time.monotonic() + 1800
last_notice = 0
while time.monotonic() < deadline:
    proc = PROCS["service"][0]
    if proc.poll() is not None:
        raise RuntimeError("TangoFlux server exited during startup:\n" + ct_tail("service", 90))
    try:
        code, _ = http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)
        if code == 200:
            break
    except Exception:
        pass
    if time.monotonic() - last_notice >= 20:
        last_line = (ct_tail("service", 3).strip().splitlines() or ["loading weights..."])[-1]
        tflog(f"Waiting for TangoFlux server | GPU: {ct_gpu_stat()} | {last_line[:100]}")
        last_notice = time.monotonic()
    time.sleep(4)
else:
    raise RuntimeError("Timed out waiting for TangoFlux server to become ready:\n" + ct_tail("service", 90))

auth_check(LOCAL_BASE, API_TOKEN)

# Warmup generation to compile/cache CUDA & Triton kernels before opening public gate
warmup_req = urllib.request.Request(
    LOCAL_BASE + "/api/ct/generate",
    headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_TOKEN}"},
    data=json.dumps({
        "prompt": "Gentle rain falling on wooden roof",
        "duration": 2.0,
        "steps": 10,
        "guidance_scale": 4.5,
        "seed": 0,
        "response_format": "wav",
    }).encode("utf-8"),
)
try:
    with urllib.request.urlopen(warmup_req, timeout=180) as wresp:
        wav_bytes = wresp.read()
        tflog(f"Warmup SFX generation succeeded ({len(wav_bytes):,} bytes 44.1 kHz stereo WAV)")
except Exception as werr:
    tflog(f"Warmup SFX generation warning: {werr}")

for _local_c, _legacy_c in CACHE_PAIRS:
    try:
        save_cache(_local_c, CACHE_ARCHIVES, CACHE_KEY, tflog)
    except (OSError, ValueError) as err:
        tflog(f"Cache archive save warning ({_local_c.name}): {err}")

ct_public_gate(
    PUBLIC_URL,
    lambda: http_json(PUBLIC_URL + "/v1/models", API_TOKEN, timeout=15)[0],
    lambda: auth_check(PUBLIC_URL, API_TOKEN),
    tflog,
    timeout=180,
    interval=8,
)
tflog(f"Public endpoint verified: {PUBLIC_URL}/v1/models")

print(f"{SERVICE} is up: {PUBLIC_URL} (model={MODEL_REPO})")
print("CT_READY", flush=True)

ct_watchdog(
    lambda: http_json(LOCAL_BASE + "/health", API_TOKEN, timeout=10)[0] == 200,
    start_service,
    tflog,
    heartbeat=lambda up: tflog(f"heartbeat | up={up}s | GPU: {ct_gpu_stat()}"),
)
'''

if __name__ == "__main__":
    write(ROOT / "src/colabllm/services/tangoflux.ipynb",
          notebook("TangoFlux (T4 / L4 / G4 · 44.1 kHz Stereo Text-to-Audio & Sound Effect Generation)",
                   "Serves `declare-lab/TangoFlux` (CRPO preference-optimized 515M FluxTransformer + Flan-T5-Large + "
                   "AutoencoderOobleck VAE, or `declare-lab/TangoFlux-base` via `--variant base`) on a Colab T4 GPU by "
                   "default (also fast on L4/G4/A100), generating 1–30s 44.1 kHz stereo sound effects, ambient audio, "
                   "and musical cues with fail-closed Bearer token auth, single & batch SFX endpoints, and Google Drive "
                   "output persistence.",
                   TANGOFLUX_CFG, TANGOFLUX_SETUP, TANGOFLUX_SERVICE,
                   keepalive=TANGOFLUX_KEEPALIVE))
    write(ROOT / "src/colabllm/services/drama3.ipynb",
          notebook("Fish Audio Drama 3 (G4 / H100 / A100 · SGLang-Omni Zero-Shot Multi-Ref TTS & Novel Narration)",
                   "Serves `fishaudio/drama-3` (BF16 on G4 / H100 / A100-80GB) or `fishaudio/drama-3-int4` "
                   "(auto-selected on <72 GiB GPUs or via `--variant int4`) through `sglang-omni[fishaudio]==0.2.0` "
                   "(3-stage `preprocessing` -> `tts_engine` -> `vocoder`, 44.1 kHz mono output, ~22x aggregate "
                   "realtime throughput on G4) with fail-closed Bearer token auth, multi-reference voice cloning, "
                   "expressive inline event tags (`[sighs]`, `[laughs]`), `/v1/audio/speech/batch` novel narration, "
                   "and persistent Google Drive voices/outputs.",
                   DRAMA3_CFG, DRAMA3_SETUP, DRAMA3_SERVICE,
                   keepalive=DRAMA3_KEEPALIVE))
    write(ROOT / "src/colabllm/services/deepseek-coder.ipynb",
          notebook("DeepSeek-V4.1-Flash-1017 Uncensored GSQ-RCO Coder (G4 / H100 / A100 · Auto Max Context up to 1M + Vision)",
                   "Serves `Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-GGUF` (50% expert-pruned 192-expert "
                   "`IQ2_XXS` MoE trunk, 72.1 GB + 110.6 GB `IQ4_NL` Engram lookup table + BF16 `deepseek41v` vision "
                   "projector) via `llama-server --api-key-file` on a Colab G4 GPU by default (100% GPU offload + full "
                   "1,048,576 native 1M context + 100% DDR5 RAM-resident Engram + default `--reasoning on` with "
                   "per-request client thinking control), and automatic SSD lazy fallback on lower-RAM GPUs.",
                   DEEPSEEK_CODER_CFG, DEEPSEEK_CODER_SETUP, DEEPSEEK_CODER_SERVICE,
                   keepalive=DEEPSEEK_CODER_KEEPALIVE))
    write(ROOT / "src/colabllm/services/qwen-image.ipynb",
          notebook("Qwen-Image-2.1 (G4 / H100 / A100 · vLLM-Omni T2I, 1-4 Ref RGBA I2I & Drive I/O)",
                   "Serves `Qwen/Qwen-Image-2.1` (7.1B single-stream DiT with cross-step prefix KV cache + "
                   "Qwen3-VL-8B vision-language encoder + 16x 4-channel RGBA VAE) on a Colab G4 GPU by default "
                   "(also compatible with `--gpu H100` and `--gpu A100 --quantization fp8-all`) via `vllm serve --omni` "
                   "with fail-closed Bearer token auth, CORS preflight support, OpenAI-compatible `/v1/images/generations`, "
                   "`/v1/images/edits`, `/api/ct/edit`, and `/v1/chat/completions`, 1–4 `<image1>..<image4>` reference "
                   "images, RGBA transparency, direct Google Drive reference image reading & output persistence, and LoRA.",
                   QWEN_IMAGE_CFG, QWEN_IMAGE_SETUP, QWEN_IMAGE_SERVICE,
                   keepalive=QWEN_IMAGE_KEEPALIVE))
    write(ROOT / "src/colabllm/services/flux-klein.ipynb",
          notebook("FLUX 3 Image Klein (G4 / H100 / L4 · vLLM-Omni T2I, Multi-Ref I2I, Drive I/O & PUP)",
                   "Serves `black-forest-labs/flux-3-image-klein` (BF16 / FP8R / NVFP4, 5.6B DiT + Qwen3-VL-4B text "
                   "encoder + VAE, 16-step flow-map distillation) on a Colab G4 GPU by default (also compatible with "
                   "`--gpu H100` and `--gpu L4`) via `vllm serve --omni` with fail-closed Bearer token auth, "
                   "OpenAI-compatible `/v1/images/generations`, `/v1/chat/completions`, and `/v1/images/edits` / "
                   "`/api/ct/edit` endpoints, direct Google Drive reference image reading & result persistence, LoRA "
                   "support, and optional `flux-3-image-klein-pup` prompt upsampling.",
                   FLUX_KLEIN_CFG, FLUX_KLEIN_SETUP, FLUX_KLEIN_SERVICE,
                   keepalive=FLUX_KLEIN_KEEPALIVE))
    write(ROOT / "src/colabllm/services/unlimited-ocr.ipynb",
          notebook("Unlimited-OCR 2 (A100)", "Drive-native PDF/image batch OCR; official Transformers backend, "
                   "local SSD model and kernels, persistent Drive results, Caddy auth and asynchronous CLI jobs.",
                   UNLIMITED_OCR_CFG, UNLIMITED_OCR_SETUP, UNLIMITED_OCR_SERVICE,
                   keepalive=UNLIMITED_OCR_KEEPALIVE))
    write(ROOT / "src/colabllm/services/longcat-avatar.ipynb",
          notebook("LongCat-Video-Avatar 1.5 (G4)",
                   "Audio-driven single-person video, BF16 + Whisper + 8-step DMD; outputs on Drive.",
                   LONGCAT_AVATAR_CFG, LONGCAT_AVATAR_SETUP, LONGCAT_AVATAR_SERVICE,
                   auth_cell=LONGCAT_VIDEO_CADDY))
    write(ROOT / "src/colabllm/services/longcat-video.ipynb",
          notebook("LongCat-Video (G4)", "Single-GPU BF16 text-to-video, image-to-video and video continuation; "
                   "models on local SSD, generated videos on Drive, Caddy authentication.",
                   LONGCAT_VIDEO_CFG, LONGCAT_VIDEO_SETUP, LONGCAT_VIDEO_SERVICE,
                   auth_cell=LONGCAT_VIDEO_CADDY))
    write(ROOT / "templates/service.ipynb",
          notebook("my-service (template)", "Template: replace cells 1 and 2 with the real model service.",
                   TEMPLATE_CFG, TEMPLATE_SETUP, TEMPLATE_SERVICE))
    write(ROOT / "src/colabllm/services/comfyui.ipynb",
          notebook("ComfyUI + ComfyUI-Manager",
                   "ComfyUI with ComfyUI-Manager enabled behind Caddy auth (basic auth for browser UI, "
                   "Bearer token for API clients). Models live on fast local NVMe SSD (`/content/ComfyUI/models`) "
                   "and are downloaded on demand per workflow (`--workflow <url-or-template>` or `pull-workflow`); "
                   "custom_nodes, user settings, input, and output persist on Google Drive.",
                   COMFYUI_CFG, COMFYUI_SETUP, COMFYUI_SERVICE))
    write(ROOT / "src/colabllm/services/ollama.ipynb",
          notebook("Ollama (Native / OpenAI / Anthropic Compatible)",
                   "Ollama server behind Caddy auth (Basic Auth + Bearer + Anthropic `x-api-key`), "
                   "with automatic model download & GPU VRAM warmup (`--models`, `--alias`, `--preload`), "
                   "selectable or full-open compatible APIs (`--compat all|ollama|openai|anthropic`), "
                   "and full `OLLAMA_*` server optimization tuning.",
                   OLLAMA_CFG, OLLAMA_SETUP, OLLAMA_SERVICE,
                   keepalive=OLLAMA_KEEPALIVE, auth_cell=OLLAMA_CADDY))
    write(ROOT / "src/colabllm/services/unsloth-studio.ipynb",
          notebook("Unsloth Studio", "Based on https://github.com/unslothai/unsloth/blob/main/studio/"
                   "Unsloth_Studio_Colab.ipynb -- built-in login, our own cloudflared (quick or fixed domain).",
                   UNSLOTH_CFG, UNSLOTH_SETUP, UNSLOTH_SERVICE, keepalive=UNSLOTH_KEEPALIVE))
    write(ROOT / "src/colabllm/services/qwen-flash.ipynb",
          notebook("Qwen3.8-Flash-Next (G4 · UD-Q4_K_XL-V2 · 1M Context · q8_0 KV + BF16 Vision)",
                   "Serves `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` (`unsloth-UD-Q4_K_XL-V2` "
                   "calibrated 4-bit MoE, 1,048,576 1M context via YaRN 4.0x + `q8_0` KV/QSA k-pool cache, "
                   "26.82 GiB 3-gram PLE table in DDR5 host RAM via `--load-mode none --lazy-mode off --no-host`, "
                   "hardware-adaptive `--n-cpu-moe` (`3/48` layers on G4), and `mmproj-Qwen3.8-Flash-Next-BF16.gguf` "
                   "vision projector) on a Colab G4 (RTX PRO 6000 Blackwell sm_120) with `llama-server --api-key-file`.",
                   QWEN_FLASH_CFG, QWEN_FLASH_SETUP, QWEN_FLASH_SERVICE, keepalive=QWEN_FLASH_KEEPALIVE))
    write(ROOT / "src/colabllm/services/deepseek-qwen-9b.ipynb",
          notebook("DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic (L4 Default / T4 · i1-Q4_K_M + Native MTP · 256K Context + F16 Vision)",
                   "Serves `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` GGUF (`i1-Q4_K_M` 5.24 GiB imatrix "
                   "4-bit default, native Multi-Token Prediction `mtp-Q8_0.gguf` draft sidecar via `--spec-type draft-mtp`, "
                   "`mmproj-f16.gguf` multimodal vision projector, and full native 262,144 256K context with `q8_0` KV cache) "
                   "on a Colab L4 GPU by default via `llama-server --api-key-file`.",
                   DEEPSEEK_QWEN_9B_CFG, DEEPSEEK_QWEN_9B_SETUP, DEEPSEEK_QWEN_9B_SERVICE,
                   keepalive=DEEPSEEK_QWEN_9B_KEEPALIVE))
    write(ROOT / "src/colabllm/services/nemotron-transcribe.ipynb",
          notebook("NVIDIA-Nemotron-3.5-Transcribe-30B-A4B (G4 / vLLM + NeMo Speech)",
                   "Serves `nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` (BF16 / FP8 / NVFP4, 65536 context, "
                   "up to 1-hour audio, 40 languages, MTP8 speculative decoding, CTC word timestamps & speaker "
                   "diarization) on a Colab G4 GPU via `vllm serve` + `nemo_speechlm`, with Bearer token auth "
                   "and direct Google Drive audio batch/single transcription endpoints.",
                   NEMOTRON_TRANSCRIBE_CFG, NEMOTRON_TRANSCRIBE_SETUP, NEMOTRON_TRANSCRIBE_SERVICE,
                   keepalive=NEMOTRON_TRANSCRIBE_KEEPALIVE))
    write(ROOT / "src/colabllm/services/indextts.ipynb",
          notebook("IndexTTS-2.5 (L4 / vLLM-Omni 2-Stage Zero-Shot Voice Cloning & Emotion TTS)",
                   "Serves `IndexTeam/IndexTTS-2.5` on a Colab L4 GPU via `vllm serve --omni` (Stage 0 Talker + "
                   "Stage 1 EnhancedCodec/S2Mel DiT/BigVGAN) with Bearer token auth, zero-shot voice cloning, "
                   "reusable uploaded voices (`/v1/audio/voices`), native speed control (0.5-2.0x), multilingual "
                   "synthesis (`zh`, `en`, `zhen`, `ja`, `es`, `ar`, `yue`), emotion control (`emo_vector`, "
                   "`emo_text`, `emo_audio`, `emo_alpha`), and Google Drive voice/output persistence.",
                   INDEXTTS_CFG, INDEXTTS_SETUP, INDEXTTS_SERVICE,
                   keepalive=INDEXTTS_KEEPALIVE))
    write(ROOT / "src/colabllm/services/qwen3-tts.ipynb",
          notebook("Qwen3-TTS (L4 / vLLM-Omni 2-Stage CustomVoice, VoiceDesign & 3s Voice Clone)",
                   "Serves `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`, `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, and "
                   "`Qwen/Qwen3-TTS-12Hz-1.7B-Base` on a Colab L4 GPU via `vllm serve --omni` (Stage 0 Talker + "
                   "Stage 1 Code2Wav 12Hz Tokenizer) plus an auxiliary multi-task GPU worker, supporting 9 built-in "
                   "premium timbres (`Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, "
                   "`Sohee`), natural-language instruction control, VoiceDesign persona creation, 3-second zero-shot "
                   "voice cloning, Voice-Design-then-Clone workflow, 10 languages, and Google Drive voice/output persistence.",
                   QWEN3_TTS_CFG, QWEN3_TTS_SETUP, QWEN3_TTS_SERVICE,
                   keepalive=QWEN3_TTS_KEEPALIVE))
    write(ROOT / "src/colabllm/services/openjutsu.ipynb",
          notebook("Openjutsu (G4 / MiniMax H3 Motion Transfer, Video Edit, Face & Colour Fix, FLUX.3 Picture Prep)",
                   "Co-locates ComfyUI (`127.0.0.1:8188` + 13 custom node packs), patched `comfy-api-proxy` "
                   "(`127.0.0.1:8189`), and Openjutsu's Next.js 16 web UI (`127.0.0.1:3000`) on a Colab G4 "
                   "(RTX PRO 6000 Blackwell, 96 GB VRAM) behind Caddy Basic Auth + session cookie + Bearer token. "
                   "Models live on fast local NVMe SSD (`/content/ComfyUI/models`); generated videos, clip "
                   "checkpoints, inputs, and custom LoRAs persist on Google Drive.",
                   OPENJUTSU_CFG, OPENJUTSU_SETUP, OPENJUTSU_SERVICE,
                   keepalive=OPENJUTSU_KEEPALIVE, auth_cell=OPENJUTSU_CADDY))
