# Project-wide FRP tunnel

The default backend is still Cloudflare Tunnel. Once FRP is enabled globally, the next `up` of every service and instance uses FRP automatically, with HTTPS provided by Caddy on the configured server. Configuration changes never switch or restart services that are already running.

## Configuration and server installation

Locally you need a logged-in `cf` CLI and a working SSH key/SSH config. The server needs a public IP, Linux amd64/arm64, Python 3, systemd, and root or passwordless sudo. Public TCP 7000 receives Colab connections and 80/443 serve the sites and HTTPS certificates; security groups/firewalls must allow these ports.

```bash
cf auth login

# Replace the sample IP and domain; --ssh supports aliases, ports, keys and jump hosts from ~/.ssh/config
colabllm frp-config \
  --server-ip 203.0.113.10 \
  --ssh root@frpip \
  --base-domain example.com

colabllm frp-config       # show the global configuration without the token
colabllm frp-status       # check frps, Caddy and routes over SSH
```

The command first verifies Cloudflare zone permissions, then checks listening processes, systemd units, configuration paths and installed binaries over SSH; port 7000 being in use is not treated as a conflict by itself.

- An frps is already running: custom binary/service names (including `frpcolab-frps`) are recognized; the existing token, TLS certificate, port limits and version are read and reused when compatible, without restarting or rewriting the existing frps.
- An frps is installed but stopped: after checking its configuration, the existing systemd unit is started.
- Only the binary is installed: it is reused to create a project-specific configuration/service, without downloading again.
- Nothing installed: frps 0.61.2 and Caddy 2.10.2 are installed after verifying the official release hashes, then the configuration and systemd services are generated.

Colab frpc downloads a stable version matching the server (>= 0.61.0 required); when the existing certificate uses a DNS SAN, TLS is verified against that DNS name even when connecting by IP. The server private key and the existing token are never replaced.

An existing frps must use token authentication, an explicit TLS certificate and loopback proxy listeners. Automatic reuse does not support OIDC or mTLS configurations that require client certificates; such configurations are reported with the reason and the shared service is not modified. Port 7000 used by another program is still reported as an error. The first configuration must provide the IP, the SSH target and the base domain together.

With an existing Caddy, the binary and Caddyfile are located from the listening process or the installed systemd unit, including custom paths and service names. The original file is backed up and only an import of the project site directory is appended; file permissions and ownership are kept, the configuration is validated before reload, and the original is restored on failure. An existing Caddy binary is never reinstalled. JSON/API-only configurations or ambiguous multi-instance layouts are reported as errors. A newly installed Caddy uses its own `colabsvc-caddy.service` and does not replace other programs.

Changing the IP/listening port of a deployed server requires an explicit migration; the command never restarts a shared frps directly and interrupts all connections.

## Using it with runtimes

```bash
# Global FRP configured: FRP is used automatically, hostname indextts.example.com
colabllm indextts up

# Multiple instances keep the original alias and instance suffix, e.g. qwen-2.example.com
colabllm qwen-flash up -i 2

# Choose the backend for one start, or override this start's service hostname
colabllm longcat-avatar up --transport frp --domain avatar.example.com
colabllm longcat-avatar up --transport cloudflare
colabllm longcat-avatar up --quick-tunnel

# Disable global FRP; the base domain is kept, so later starts default to a fixed Cloudflare domain
colabllm frp-config --disable

# To go back to the default dynamic Cloudflare URL, also clear the base domain and any environment override
colabllm tunnel-config --clear
```

Selection rules: `--quick-tunnel` forces a dynamic Cloudflare URL; `--transport` overrides the global backend; otherwise FRP is used when enabled and Cloudflare when not. `--quick-tunnel` and `--transport frp` cannot be combined. The hostname override rules of `--base-domain`, `--domain` and `COLABSVC_BASE_DOMAIN` still apply.

When FRP is chosen explicitly, the server must already be configured with `frp-config`; an incomplete configuration or a connection failure never falls back to another backend automatically.

## Data path and authentication

```text
client -> HTTPS :443 / Caddy (FRP server)
       -> 127.0.0.1:dedicated mapped port / frps
       -> TLS tunnel / Colab frpc
       -> Colab's existing auth entry point -> model service
```

Colab frpc connects out to the server on 7000; FRP mapped ports are allocated from 20000-29999 and listen only on the server loopback address. Allocation is managed by the server under a lock, keyed by project ID and service/instance name, and restarts reuse the same port.

Locally, `cf` creates or updates A/AAAA records automatically, with `proxied=false` (DNS only) by default. The hostname resolves to the FRP server, and Caddy obtains/renews public HTTPS certificates automatically, supporting WebSocket, SSE and large files. FRP's 7000 connection uses the server IP directly.

DNS updates only touch records owned by the service; records used for anything else cause a conflict error. If a Caddy route update fails, the site and this run's DNS changes are rolled back.

frpc verifies the server with the token and a pinned trusted certificate; the server private key never leaves the server. FRP does not replace service authentication: existing Bearer, Basic Auth and app logins stay in effect. The notebook checks the auth layer first, then registers the tunnel; the tunnel cell verifies the public TLS response, and the service's own health check still decides `CT_READY`. A temporary 502 while a large model loads does not block the rest of the loading.

frpc joins the existing watchdog and is restarted when it exits; frpc reconnects automatically after network drops. All notebooks share the same implementation.

## Configuration, state and cleanup

- Global base domain and FRP metadata: `~/.colabllm/tunnel-config.json`.
- Client token/trusted certificate: `~/.colabllm/frp/<server-id>.json`, permissions `0600`.
- Each runtime's backend and owned resource records: `~/.colabllm/services/<service>/state.json`, permissions `0600`.
- Server frps and dedicated Caddy configuration: `/etc/colabsvc-frp/`.
- Route registry and certificate data: `/var/lib/colabsvc-frp/`.
- Newly installed binaries: `/usr/local/lib/colabsvc-frp/`.
- Project sites for an existing standard Caddy: `/etc/caddy/colabsvc-frp/`.
- Colab client log: `/content/ct/frpc.log`; configuration file `/content/ct/frpc.toml` with permissions `0600`.

```bash
colabllm indextts down                   # keep the route and DNS for a quick restart
colabllm indextts down --delete-tunnel   # delete this instance's project route/DNS
```

Cleanup acts on the backend resources saved for the instance and never guesses from a global configuration changed later. Cloudflare and FRP resources are recorded separately; only DNS records whose ownership still matches are deleted. Before changing an instance's FRP server or domain, clean up the old route with `down --delete-tunnel`. A shared frps/Caddy is never uninstalled when a single instance stops.

`frp-status` shows the units, versions and reuse state actually in use. For example, an existing setup may use `frpcolab-frps.service` and `caddy.service` instead of the service names the project would create.

Server diagnostics (adjust to the actual units reported by `frp-status`):

```bash
ssh root@frpip 'systemctl status colabsvc-frps'
ssh root@frpip 'journalctl -u colabsvc-frps -n 80 --no-pager'
# A newly installed Caddy uses colabsvc-caddy; a reused standard installation uses caddy
ssh root@frpip 'journalctl -u colabsvc-caddy -n 80 --no-pager'
```

## Verification

`tests/test_frp_backend.py` verifies configuration, DNS ownership, rollback, idempotent installation and backend-switch cleanup; `tests/test_frp_transport.py` uses real frps/frpc/Caddy on unprivileged local ports to verify HTTPS, FRP TLS, authentication, SSE, WebSocket, reconnection and rejection of wrong credentials.

```bash
python tests/test_frp_backend.py
COLABSVC_FRP_TEST_BIN=/path/to/verified/binaries python tests/test_frp_transport.py
```

Local tests use dedicated test certificates. Reuse of existing services and a real TLS tunnel request round trip were also verified on a server with an existing frps 0.71.1 / Caddy 2.6.2, with both original process PIDs unchanged; public DNS and ACME issuance still need to be verified when a specific runtime starts.


## ComfyUI page stays blank for a long time

First distinguish an unavailable service from slow frontend asset loading: an authenticated `/api/system_stats` responding does not mean the JavaScript, workflow toolbar and `/ws` have loaded. Check the slowest `/assets/` file in the browser's Network tab, then compare the FRP origin fetch on the server itself with the public download time.

With Caddy 2.10.2 and later, the ComfyUI FRP route sets a long-lived private browser cache for JS/CSS under `/assets/` that carry an eight-character content hash, only for successful GET/HEAD responses. The index page, API, custom extensions and 401 responses do not use this cache rule. After a frontend upgrade the file hashes change and new files are downloaded automatically; the first visit still depends on public bandwidth. Existing runtimes need their server site configuration updated before this takes effect.

Some older Caddy/Go combinations may hit `context: internal error: missing cancel error` on reload and exit, leaving systemd stuck in the reloading state. The shared backend limits the reload wait and restores the verified configuration with stop/start only when MainPID is already 0; it never force-stops while a main process still exists. Identical route configurations are not reloaded again. Keep the configuration, certificates and service units, and upgrade the problematic Caddy binary separately.

Troubleshooting tips: if pages load much slower than expected, use a newer Caddy version with verified release hashes (override the executable with a systemd drop-in, keeping the original binary, sites and certificates), and check the client network first (proxy/gateway routing rules, actual egress, TCP retransmissions); do not conclude that total server bandwidth is insufficient from one slow HTTPS file download. If a local proxy sends the whole project domain direct and the direct path is poor, add a higher-priority proxy rule for the specific service hostname (e.g. `comfyui.example.com`); back up the proxy configuration and run its own validation command before changing it.
