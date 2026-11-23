# Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| Service worked, then the URL died after a while | `colab exec` disconnected (Ctrl-C, terminal closed, laptop slept, network) -> Colab reclaimed the VM | Use the default detached `up`; keep the local machine awake / run on a server. `colab sessions` shows whether the VM still exists |
| Later cells never ran | Disconnected before the last cell started (cells are sent one by one by the client) | Re-run `up` (session is reused; `--skip` finished cells) |
| `colab exec` / `colab drive status` / `colab dfs status` hangs | The kernel is busy with the blocking last cell; new execs queue behind it | Expected. Use `logs`, `status` (HTTP probe), or `colab stop` / `down` |
| `up` reaches its readiness timeout without `CT_READY` | A cell raised (exit code stays 0) | `logs -n 200`; inspect failures before restarting. Default-mode timeout leaves the runner active; use `--ready-timeout` for expected long loading |
| Unauthenticated probe returns 404/200 on a Caddy service | Caddy is not the thing answering on `CT_PUBLIC_PORT` | Never use 8080 (Colab runtime server) or 8012 (dfs shim). The notebook now fails closed: cell 3 requires a 401 from Caddy and cell 4 refuses to start the tunnel otherwise |
| `caddy exited -- is :PORT in use?` | Port collision on the VM | Change `public_port` in `src/colabllm/services/<name>.py` |
| Ollama returns 403 through the proxy | Ollama rejects foreign `Host` headers | Keep `REWRITE_HOST = True` in cell 3 |
| App generates wrong absolute links behind Caddy | `Host` rewritten to `127.0.0.1:<port>` | Set `REWRITE_HOST = False` for that notebook |
| Notebook hangs at `drive.mount()` | Browser consent flow | Use `colab drive mount` / `colab dfs mount` (done by the script, respecting `COLAB_DRIVE_ACCOUNT`); the injected prelude skips `drive.mount()` |
| `Error while fetching HF_TOKEN secret value from your vault` / `unauthenticated requests to the HF Hub` | `huggingface_hub` on Colab calls `google.colab.userdata.get('HF_TOKEN')`, which times out in CLI runs | Fixed automatically: the prelude patches `userdata.get` to read `os.environ`, and `colabsvc` forwards local `HF_TOKEN` / `~/.cache/huggingface/token` / `--hf-token <tok>` (or reads `/content/drive/MyDrive/colab-llm/hf_token.txt`) |
| `fatal: destination path ... already exists` | Re-running a non-idempotent setup cell on a reused session | `test -d x/.git && git -C x pull --ff-only \|\| git clone ...` |
| `cf ... --tunnel-config-origin-request-access-* are required` | `cf` beta flag validation | Use `--body` JSON (already done in `src/colabllm/lib/colabsvc.py`) |
| `cf` errors about auth | Not logged in / wrong profile | `cf auth whoami`, `cf auth login` |
| Fixed domain shows Cloudflare error 1033 | Tunnel has no connector (VM down or cloudflared died) | `status`; watchdog restarts cloudflared; otherwise `up` again |
| Fixed domain shows 502 | Public proxy responds but the model is loading or the backend service is down | Check actual transport and model logs; a valid HTTPS response is transport readiness, not `CT_READY` |
| `up` unexpectedly uses a fixed URL or FRP | Project domain/FRP settings or an environment override are active | Read `colabllm tunnel-config`; use one-launch `--transport cloudflare` or `--quick-tunnel` if requested |
| FRP setup finds 7000 occupied | Could be an existing frps rather than a conflict | Discover the listener, config and unit, then reuse compatible frps/Caddy; see [FRP adoption](frp.md#discover-before-installing) |
| frpc rejects login/TLS | Server version, Token, scopes or certificate name differs from the client | Use discovered server metadata, preserve credentials and verify TLS SAN; see [FRP compatibility](frp.md#client-compatibility-and-routing) |
| `info` shows no API key | Default credential redaction | `info --show-secrets` or `info --json --show-secrets`; see [key lookup](auth.md#api-key-lookup-and-clients) |
| DNS conflict after reinstall or new configuration | A new project ID differs from the existing DNS/route owner | Compare record comments and `frp-status`; see [identity repair](frp.md#dns-conflict-after-creating-a-new-user-configuration) |
| Changing local `api_token` does not change accepted requests | A running model has its old key loaded | Update persistent state and service-side authentication coherently; see [rotation](auth.md#user-supplied-keys-and-rotation) |
| SSH diagnostic returns HTTP 429 | Another SSH channel is active | Finish the owned diagnostic connection; preserve unrelated user sessions |
| `colab new` fails for a GPU | Quota / tier / availability | Try another `--gpu`, check `colab usage` (or `colab usage --all` across accounts), or launch on another account with `up -a <acct>` |
| Multi-account VM or Drive credential mismatch | Wrong active account or unconfigured shared Drive binding | Inspect `colabllm <name> info --json`, `colab account status`, `colab ext doctor`, and `colabllm doctor -a <acct>` |

Start with offline `colabllm doctor` (or `colabllm doctor -a <acct>`); use `doctor --online` only when read-only authorization probes are needed.
Useful commands: `colab sessions --all`, `colab account status`, `colab ext doctor`, `colab status -s <name>`, `colab usage --all`, `colab log -s <name>`,
`colabllm tunnel-config`, `colabllm frp-status`,
`cf tunnels list --name colab-<name>` (Cloudflare), `cf dns records list -z <zone-id> --name <host>`.
Remote logs live in `/content/ct/<process>.log` on the VM (readable only while the kernel is free,
or via `colab download -s <name> /content/ct/caddy.log ./caddy.log`).
