# Connections and recovery

## First-use dependencies

Run `colabllm doctor` (or `colabllm doctor -a ACCOUNT [--drive-account STORAGE]`); inspect structured results with `--json`. Local dependencies include Python
3.12+, installed Colab CLI and its separate `colab-ext` extension. A successful `colab version` alone does
not prove Drive/multi-account extension (`colab drive` / `colab dfs`) is installed; follow the installed distribution's prerequisite documentation for
the extension (`colab ext doctor`). When `accounts.json` exists under `~/.config/colab-cli/`, `doctor` checks the active or specified account's Drive binding and per-account `dfs_credentials.json`; `doctor --online --timeout 10` additionally checks
selected backend authorization without creating a GPU session or opening a browser.

Google/Drive and Cloudflare login may need human interaction. Report the exact missing prerequisite;
never claim an offline doctor warning proves credentials expired. A user with an existing remote API
URL and its token can call supported clients without creating their own Colab session.

## Domain and transport

```bash
colabllm tunnel-config
colabllm <service> up --quick-tunnel
colabllm <service> up --transport cloudflare
colabllm frp-status
```

Global domain/FRP configuration affects future starts for all services. Preserve the user's global
settings for a single task; use one-launch overrides when appropriate. Quick Tunnel URLs are dynamic.
FRP needs configured server credentials and a fixed hostname. Server setup changes remote resources:
use `frp-config --help` and the user's actual server/domain, never example infrastructure.

If DNS conflict appears after recreating configuration, compare the DNS ownership comment and the
`frp-status` project/service route. Matching server IP is insufficient proof of ownership. Preserve
unrelated records and running services; do not bypass ownership checks or generate another project
ID. The installed CLI has no automatic legacy `.state/` migration. Explain the mismatch and establish
resource ownership before a targeted repair; back up private configuration before changing identity.

## Failure triage

| Symptom | Next check |
| --- | --- |
| `up` timed out | Inspect `status --json` and `logs -n 100`; default-mode runner remains alive |
| 502 while starting | Inspect model loading and tunnel logs; do not repeatedly submit startup |
| `running: true`, but no real VM | Compare `colab sessions --all` (or `colab -a <account> sessions`) and runner ownership; a PID alone is insufficient |
| No visible API token | Use `info --show-secrets`; credentials are redacted by default |
| 401 with saved token | Confirm the runtime/instance, URL origin and actual server key |
| Wrong account or missing Drive credentials in multi-account setup | Inspect `colabllm <service> info --json`, `colab account status`, and `colabllm doctor -a <account>` |
| New `colab exec` never executes | Last notebook cell intentionally blocks; use logs/HTTP instead |
| Failed `down` | Tracking is preserved; resolve the error and retry, instead of deleting state |
| Job wait interrupted | Save the job ID and query it later; do not assume it was cancelled |

Use `--debug` only when exception details are needed, and redact logs before sharing. General command
or wait timeout exits 124, invalid argument usage exits 2 and other operational failures are nonzero.
A zero exit from a low-level `colab` command can still accompany "session not found"; read its result.

Arbitrary live API key rotation has no public setter command. `up --new-credentials` generates a
random key on a stopped service; it does not accept a supplied key. Persisting a key locally does not
change the running server. Preserve active work and use service-specific authenticated reload/restart
procedures only within the user's requested scope. Test an authenticated endpoint, not public health.

## Package and skill updates

After publication, upgrade an index-installed CLI with `uv tool upgrade colabllm`. For a supplied new
wheel, use `uv tool install --force /path/to/new.whl`. Stop services at an appropriate idle point first:
replacing installed code can affect the long-running local runner. An editable development install
still depends on the checkout; normal wheel installs are independent of it.

The skill is a separate installed copy. Export the version bundled with the new CLI using
`colabllm skill --output /path/to/agent/skills`; if `colabllm/` exists, export to a fresh directory and
review differences before replacing it. Package upgrades do not overwrite an agent's custom skills.
