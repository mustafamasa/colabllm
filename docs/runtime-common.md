# Common runtime operations

Every runtime is used through the installed `colabllm`; there is no need to switch to the source directory. The current `--help` output is authoritative.
The local machine only handles control, file transfer and keep-alive; Torch, CUDA or NumPy are not needed locally.

## First-time setup and startup

```bash
colabllm --version
colabllm services
colabllm doctor
colabllm tunnel-config
```

The `colab` CLI, the separate `colab dfs` extension and Google/Drive authorization must be set up separately.
`doctor` runs offline checks by default; `doctor --online --timeout 10` adds time-limited read-only authorization checks.
Fixed domains need Cloudflare authorization; FRP additionally needs a user-configured server and SSH authentication.
The release README is authoritative for installation and prerequisites. The bundled agent skill can be exported with `colabllm skill --output <skills-parent-dir>`.

Check the target runtime's `ps --json` / `status --json` first and reuse an existing ready instance. `up` may create a new paid GPU session.
By default a start reads the global configuration: FRP if enabled, otherwise a Cloudflare named tunnel when a domain is configured, otherwise a dynamic Quick Tunnel.
`up --quick-tunnel` forces a dynamic address; `up --base-domain example.com` sets the suffix for one start; `up --domain service.example.com` sets the full hostname.
Domain examples are not deployment addresses; the real URL is whatever `info --json` reports. The three domain options of one start are mutually exclusive.

## Lifecycle, timeouts and instances

Each runtime guide gives the concrete service name. Shared commands are `up/down/purge/info/ps/list/status/logs/tunnel-config`; service-specific commands are in each guide.

- `up` waits for a complete `CT_READY` protocol line; `up --detach` only submits the background start, so keep checking `status --json`.
- `up --ready-timeout SECONDS` limits the readiness wait. In the default background mode a timeout keeps the runner; with `--foreground` a timeout disconnects.
- `--command-timeout SECONDS` or `COLABLLM_COMMAND_TIMEOUT` limits external control commands; the long-lived keep-alive connection is not subject to this limit.
- `status --json` returns running, ready, URL and probe information; `running: true` only means the local runner is alive and is not a substitute for a model readiness check.
- `logs -n 100` shows recent logs and `logs -f` follows them; logs may contain sensitive information printed by third-party programs, so redact them before sharing.
- `-i ID` works for starting and later calls. An instance ID is 1-64 letters, digits, underscores or hyphens and does not start with a hyphen; the default instance takes no `-i`.
- `ps --json` lists the instances of the service. Separate instances use separate VMs, tunnels and credentials, but instances of the same service may share a Drive data directory.
- `down -i ID` releases that instance's VM; `down --all` stops all instances of the service; `--delete-tunnel` also deletes the tunnel/DNS owned by the instance.
- `purge -i ID` deletes a stopped instance's local state (credentials, `run.log`, notebook copy); it first runs `colab stop` so no untracked session is left, and refuses while the runner is alive. An instance that owns a named tunnel needs `--delete-tunnel` or `--keep-tunnel`. Drive data is never touched.

The local machine must stay online and must not sleep; the background runner keeps the `colab exec` connection open, and the VM may be reclaimed after a disconnect.
Do not repeat `up` just because a client wait timed out. Check the logs and sessions first to avoid loading models or submitting jobs twice.
When `down` fails the tracking information is kept; fix the error and retry. Stop services at a suitable idle time before upgrading the package.

## Authentication, paths and output

`info/ps/status` hide credentials by default; only `info --json --show-secrets` returns fields such as `api_token`.
Use the Bearer token for APIs; Caddy-protected web pages use `auth_user/auth_pass`, and Unsloth uses the app's own login fields.
Capture secrets only for the client; never write them into source code, command examples or shared logs. The FRP token is not a model API key.
When switching `--url` / URL environment variables to a different origin, pass a matching token explicitly; scheme, host and effective port together define the origin.
There is currently no public command to set an arbitrary persistent API key; `up --new-credentials` generates new random credentials for a stopped instance and does not accept given values.

Local state lives in `~/.colabllm/services/<service[-instance]>/`; use `COLABLLM_HOME` to select an isolated configuration.
The global configuration is `tunnel-config.json` in the user directory; external authorization for the CLIs is still managed by Colab/Cloudflare/SSH.
Relative local input/output paths are based on the current directory; `drive://...` means a remote MyDrive path. Drive share pages or file IDs are not accepted in place of paths.
Drive data is usually under `MyDrive/colab-llm/<service>/`; the exact storage of each service is given by its guide and the returned results.

JSON results go to stdout and diagnostics to stderr. Success exits 0, usage errors 2, wait/command timeouts 124, Ctrl+C 130; underlying operation failures are non-zero.
For TTS and transcription `--timeout` is a request limit; the wait limits of OCR/Avatar/ComfyUI never cancel background jobs that were already submitted.

## Troubleshooting order

Check `doctor` first for dependency or authorization errors. A 502 may come from loading, a dropped tunnel or a reclaimed VM; do not decide to restart based on the status code alone.
Check `logs`, `status --json` and `colab sessions`; the blocking last notebook cell is the keep-alive design, and a new `colab exec` may queue behind it.
For DNS ownership conflicts, first compare the project ID/service routes in `tunnel-config` and `frp-status`; never overwrite records of unknown ownership.
Before stopping or resetting, make sure no generation, training or transcription job is running, and keep job IDs, outputs and recovery information.
