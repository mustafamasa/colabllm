---
name: colab-llm-service
description: >-
  Deploy and operate large-model services on Google Colab through this repository's
  one-notebook/one-script convention and unified colabllm CLI. Use for service setup, lifecycle, project-global
  domains, Cloudflare Tunnel or SSH-managed FRP/Caddy, API key lookup and rotation,
  uv package installation checks, and session or tunnel troubleshooting.
---

# Colab LLM service

Every model service in this repo is a pair of files in `src/colabllm/services/`, sharing one library:

```
src/colabllm/cli.py     packaged CLI: service dispatch and project configuration
src/colabllm/services/<name>.py       execution script (local): session, Drive, selected tunnel, credentials, keep-alive
src/colabllm/services/<name>.ipynb    notebook (Colab VM): setup, service, Caddy auth, cloudflared/frpc, keep-alive loop
src/colabllm/lib/colabsvc.py          shared runner used by every script (Service dataclass + CLI)
src/colabllm/control.py             project-global domain and FRP configuration
src/colabllm/lib/frp_backend.py      local SSH/DNS orchestration; frp_server.py provisions/adopts the server
templates/build_notebooks.py  generator of every .ipynb (never hand-edit generated notebooks)
templates/*.py, *.js     modules embedded into notebooks: tunnel_client, colab_cache, colab_asgi_common,
                         service middlewares/workers, ComfyUI helpers (locally unit-tested)
templates/               also service.py + service.ipynb to copy for a new service
~/.colabllm/services/<name>/           private runtime state and resource identities (outside the checkout)
~/.colabllm/tunnel-config.json  shared base domain and optional FRP metadata
~/.colabllm/frp/             private per-server client credentials (0600)
```

Existing services: `unsloth-studio` (built-in login, L4), `ollama` (Caddy auth, T4),
`comfyui` (ComfyUI + ComfyUI-Manager, Caddy auth, L4, on-demand workflow model download to local SSD via `--workflow` / `pull-workflow`),
`qwen-flash` (`SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF` IQ3_S, 800K YaRN 3.125x + BF16 vision,
`llama-server --api-key-file` built-in auth, G4 GPU, domain alias `qwen`),
`deepseek-coder` (`Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-GGUF` IQ2_XXS 72.1 GB pruned trunk + 110.6 GB Engram DDR5 host-RAM resident on G4/H100 or SSD lazy `mmap` fallback + BF16 `deepseek41v` vision, default `--reasoning on` with per-request client `enable_thinking` toggle, hardware-adaptive max context up to 1,048,576 via `llama-server --api-key-file`, G4/H100/A100 GPU, domain alias `deepseek`),
`nemotron-transcribe` (`nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B` via `vllm serve` + `nemo_speechlm`,
Bearer API key auth, G4 GPU, domain alias `transcribe`, single & batch Drive audio CLI `transcribe`),
`indextts` (`IndexTeam/IndexTTS-2.5` via `vllm serve --omni` 2-stage pipeline, Bearer API key auth, L4 GPU,
domain alias `indextts`, zero-shot voice cloning, persistent Drive voices, native speed 0.5-2.0x,
multilingual `zh`/`en`/`zhen`/`ja`/`es`/`ar`/`yue` & emotion control CLI `tts` / `voices` / `upload-voice`),
`qwen3-tts` (`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` / `VoiceDesign` / `Base` via `vllm serve --omni` 2-stage pipeline + auxiliary multi-task GPU worker, Bearer API key auth, L4 GPU, domain alias `qwen3-tts`, 9 built-in speakers, natural-language instruction control, VoiceDesign persona synthesis, 3-second zero-shot voice cloning, and `design-voice` Voice-Design-then-Clone workflow).

Also available: `unlimited-ocr` (A100, Caddy auth, persistent Drive OCR jobs),
`longcat-avatar` (G4, Caddy auth, audio-driven avatar generation with job/download commands),
`longcat-video` (G4, Caddy auth, Gradio text/image-to-video and continuation UI),
and `openjutsu` (G4, Caddy auth, co-located ComfyUI + patched `comfy-api-proxy` + Openjutsu Next.js 16 studio for MiniMax H3 motion transfer, object edit, face/colour post-processing & FLUX.3 picture prep; auto `--highvram` on `>=76 GiB` GPUs via `--vram-mode`, local NVMe SSD `h3_chains` bind-mount via `--chains-storage`, and full `swap`/`object`/`mask`/`face`/`color`/`job`/`download`/`cancel`/`check` CLI),
`flux-klein` (G4 default, H100/L4 compatible, Bearer API key auth, `black-forest-labs/flux-3-image-klein` BF16/FP8R/NVFP4 via `vllm serve --omni` + optional `flux-3-image-klein-pup` Qwen3.5-9B prompt upsampler, OpenAI-compatible `/v1/images/generations` + `/v1/chat/completions` + `/v1/images/edits` + `/api/ct/edit`, and Drive-native `generate`/`edit`/`upsample`/`ls-drive`/`models` CLI),
and `qwen-image` (G4 default, H100/A100 compatible, Bearer API key auth + CORS preflight, `Qwen/Qwen-Image-2.1` 7.1B DiT + Qwen3-VL-8B + 4-channel RGBA VAE via `vllm serve --omni`, 40-step / `true_cfg_scale=1.0` defaults, 1–4 `<image1>..<image4>` reference images, multipart+JSON `/v1/images/edits` + `/api/ct/edit` + `/v1/images/generations` + `/v1/chat/completions`, and Drive-native `generate`/`edit`/`ls-drive`/`models` CLI).
For each runtime's current concrete usage, read `docs/<name>.md`; portable copies are bundled in
`skills/colabllm/references/<name>.md`.

## Unified CLI convention

Use `colabllm <name> <command> [args...]` for runtime operations and user-facing examples.
Use `colabllm tunnel-config`, `colabllm frp-config`, and `colabllm frp-status` for project configuration.
Run `colabllm services` to discover runtimes and `colabllm <name> --help` for their commands.

Install a distributed wheel with `uv tool install /path/to/colabllm-VERSION-py3-none-any.whl`.
This fork is installed with `uv tool install git+https://github.com/mustafamasa/colabllm@english`; the `colabllm` PyPI name belongs to the upstream project.
Developers use `uv tool install --editable .`; `uv build --no-sources` produces wheel and sdist.
Run `python tools/check_distribution.py dist/colabllm-VERSION-py3-none-any.whl` to verify distribution.
Release versions have one source, `pyproject.toml`, and use `MAJOR.MINOR.PATCH` with matching `v` tags.
Before releasing, update `CHANGELOG.md` and run `python tools/check_release.py --tag vVERSION`.
Pushing that tag invokes `release.yml`: the full CI matrix, package build/verification and a GitHub Release.
The upstream PyPI job is removed in this fork. Retry failed jobs after configuration
errors; never move a published tag or replace published packages. See README's release procedure.
For actual local installation, inspect `command -v colabllm` and `uv tool list` first. Replace the
old checkout symlink with `uv tool install --force /path/to/package.whl` when switching installation
methods. Test from outside the checkout: version, service help, JSON info, dry-run, offline doctor
and completion. Keep configuration-write tests in a temporary `COLABLLM_HOME`.
Local regression tests are `python -m unittest discover -s tests` and need only the standard library;
`tests_colab/` belongs in the model environment. See `.github/workflows/ci.yml` for the platform matrix.

The package includes each service script and matching notebook. The dispatcher executes the actual
installed script so the detached `_exec` runner retains the script path. All application state is
in `~/.colabllm/` (override: `COLABLLM_HOME`); instances use `services/<name>/`, global tunnel settings
use `tunnel-config.json`, and FRP credentials use `frp/`. External CLI authentication keeps its own
locations. Relative input/output paths remain relative to the caller's working directory.
Stop services before upgrading and restart them afterwards. Existing repository `.state/` is unused;
there is no migration. New service pairs are discovered automatically and must be included in builds.

Runtime command registration belongs in the shared parser (`main(..., configure=..., handlers=...)`),
and script entry points use `cli_support.run_cli` for consistent errors. Local authenticated HTTP
clients use `safe_urlopen` or `SameOriginRedirect`. Keep diagnostics on stderr and JSON on stdout.
Instance IDs are validated strictly; parsing stops at `--`. State updates are locked transactions;
startup/shutdown also use a per-instance operation lock. Both foreground and detached runners record
PID plus start identity, and shutdown verifies ownership before signalling. `down` preserves state
on remote stop failure; readiness timeout is nonzero, while `up --detach` explicitly skips waiting.
`info/ps/status --json` default to redacted output; credentials require `--show-secrets`.

Readiness is parsed incrementally as complete protocol lines; preserve the exact `CT_READY` line
when changing notebook output. Endpoint overrides reuse stored tokens only for the same origin;
other origins require an explicit token. External CLI calls honor `--command-timeout` /
`COLABLLM_COMMAND_TIMEOUT`, while `up --ready-timeout` bounds readiness waiting. Task polling
supports `--timeout` (`0` unlimited); timeout exits use code 124. `doctor` is offline by default;
`--online` performs bounded read-only authorization probes. `completion bash|zsh|fish` generates
scripts using offline `_complete` introspection; completion reads instance directory names only.

## Daily use and transport selection

Read `colabllm tunnel-config` before choosing a public URL. It reports project-wide
settings, not settings for one runtime. `up` follows these settings:

- FRP enabled: use FRP and the configured base domain.
- Otherwise use Cloudflare: configured domain -> named tunnel; no domain -> dynamic Quick Tunnel URL.
- `--transport cloudflare|frp` overrides the backend for one launch. `--quick-tunnel` forces
  Cloudflare's dynamic URL and conflicts with `--transport frp`.
- `--domain HOST` supplies a full hostname; `--base-domain` supplies a one-launch suffix.
  Bare `--domain` uses the configured suffix or a dynamic URL if none exists. Historical
  `Service.zone` / `default_hostname` values retain short aliases only (e.g. `qwen`, `transcribe`); there is no built-in personal deployment domain.

```bash
colabllm tunnel-config
colabllm tunnel-config --base-domain example.com  # shared by all services/instances
colabllm <name> up                                # selected global backend
colabllm <name> up --quick-tunnel
colabllm <name> up --transport cloudflare --domain llm.example.com --gpu A100
colabllm <name> info --show-secrets               # prints credentials; avoid echoing them in tool output
colabllm <name> logs -f
colabllm <name> down               # stop runner + VM; --delete-tunnel cleans this instance's resources
```

For **API key lookup, user-supplied keys or rotation**, read [references/auth.md](references/auth.md).
For **Cloudflare setup, DNS ownership or cleanup**, read [references/cf-tunnel.md](references/cf-tunnel.md).
For **FRP setup, project-ID/DNS conflicts, existing server adoption or connection failures**, read
[references/frp.md](references/frp.md). Use the user's actual server/domain values, not examples.
Configuration changes affect future starts; preserve running generation tasks when changing code or transport.

`up` normally returns once the notebook prints `CT_READY`; `--detach` returns before readiness.
A detached local runner keeps `colab exec`
attached (= keep-alive). Requirements: `colab` CLI logged in, `colab dfs consent` done once.
Fixed domains use the local `cf` login for DNS in either backend. FRP server management also needs SSH.

## Hard-won rules (verified on real runs -- do not "simplify" them away)

1. **The session lives only while `colab exec` stays connected.** Disconnecting (Ctrl-C, closing the
   terminal, laptop sleep) gets the VM reclaimed a while later even if a cell is still running. The
   Colab CLI has no keep-alive. Hence the detached runner; run `up` on an always-on machine for long jobs.
2. **`colab exec -f` sends cells one at a time from the client.** Disconnect before the last cell
   starts and the remaining cells never run.
3. **The last cell must block forever** (keep-alive/watchdog loop, or e.g. Unsloth's `start()`).
4. **Background work must not depend on a thread being "the cell"**: start servers as subprocesses
   (`ct_spawn`) and register them in `PROCS` so the watchdog restarts them.
5. **Setup cells must be idempotent** -- sessions are reused (`git clone` fails on an existing dir:
   use `test -d x/.git && git -C x pull || git clone ...`; skip installs that already exist).
6. **Mount Drive with `colab dfs mount`** (browser-free, ~10 s). Notebook `drive.mount()` would wait
   forever for consent; `colabsvc` injects a prelude that turns it into a no-op once Drive is mounted.
7. **A failing cell does not change `colab exec`'s exit code** -> readiness is signalled with markers
   (`CT_READY`), never inferred from exit codes.
8. **Each `colab exec` call starts in `/content`**; `%cd` only persists within one exec.
9. **Secrets never go into the notebook**: the script generates them, stores them in `~/.colabllm/services/`
   (0600) and injects them through a private 0600 runtime notebook prelude (`os.environ.update`),
   so they never appear in `colab exec` argv / `ps`; only `CT_SERVICE` is passed with `--env`.
10. **`cf` (beta) quirk**: `cf tunnels config update --tunnel-config-ingress` demands unrelated Access
    flags -> send `--body '{"config":{"ingress":[...]}}'` instead.
11. **Compiler caches (FlashInfer/Triton/Inductor/vLLM) live on local NVMe SSD and persist to Drive as
    archives via `templates/colab_cache.py`** (one tar per cache dir, keyed by environment + GPU).
    Never point compiler caches at `/content/drive` (FUSE lock/rename bottlenecks) and never leave them
    only in `/root/.cache`. Model weights stay on local SSD (`/content/<service>/model`), downloaded with
    `HF_XET_CHUNK_CACHE_SIZE_BYTES=0`. Full pattern: [references/notebooks.md](references/notebooks.md#compiler-cache-standard-colab_cachepy).
12. **Notebooks are generated.** Edit `templates/build_notebooks.py` or the embedded modules, run
    `python3 templates/build_notebooks.py`, and commit generator, modules and regenerated `.ipynb`
    together; `tests/test_notebooks_sync.py` fails on drift. Embedded middlewares must use
    `colab_asgi_common` (fail-closed Bearer auth, Drive path containment, voice-name and multipart
    sanitization). Build and security rules: [references/notebooks.md](references/notebooks.md).

## Notebook contract

Env injected by the script (`CT_*`), defaults let the notebook also run pasted into the Colab UI:

| Var | Meaning |
| --- | --- |
| `CT_SERVICE`, `CT_SERVICE_PORT` | service name; port the service listens on **127.0.0.1** |
| `CT_AUTH` | `caddy` (service has no auth) / `builtin` (own login or API key) / `none` |
| `CT_PUBLIC_PORT` | transport target: Colab authentication Caddy port if `caddy`, else the service port |
| `CT_AUTH_USER`, `CT_AUTH_PASS`, `CT_API_TOKEN` | Caddy basic-auth login and Bearer token |
| `CT_TRANSPORT` | `cloudflare` (default) or `frp`, chosen by the local runner |
| `CT_TUNNEL_TOKEN`, `CT_PUBLIC_HOSTNAME` | Cloudflare named-tunnel token, and public hostname for either fixed-domain backend |
| `CT_FRP_*` | server, token, CA, TLS name, scopes, client version and proxy identity; runner supplies these privately |
| `CT_DATA_DIR` | `/content/drive/MyDrive/colab-llm/<name>` -- cache models/outputs here |
| `CT_MOUNT_DRIVE` | `0` with `up --no-drive`: Drive is not mounted and `DATA_DIR` falls back to `/content/data/<name>` |

Markers the notebook prints (parsed by `colabsvc`): `CT_PUBLIC_URL=<url>`,
`CT_CREDENTIALS user=<u> pass=<p>` (app's own login, e.g. Unsloth), `CT_READY`.

Standard cell order (generated by `templates/build_notebooks.py`):
`0 config` → `1 setup (idempotent)` → `2 service: def start_service(); PROCS["service"]=...` →
`3 Caddy auth` → `4 shared transport (cloudflared or frpc)` → `5 CT_READY + watchdog loop`.
Use `TUNNEL_PROCESS_KEY` to access the active transport in `PROCS`; its key is `cloudflared` or `frpc`.
Cell 0 provides `ct_spawn`, `ct_stop_proc`, `ct_gpu_stat`, `ct_public_gate` and `ct_watchdog`; custom
keep-alive cells must use them (auth failure or public timeout must raise, never print `CT_READY`).
Secrets reach the kernel through a 0600 runtime notebook prelude, not `colab exec --env` argv.
The runner clears unused backend variables when reusing a Colab kernel. A registered tunnel or
HTTPS 502 during model loading is not service readiness; keep the service health/warmup gate before `CT_READY`.
Put `# @ct:skip` in a cell (or `up --skip N`) to skip it.

## Add a new service

1. Copy the pair: `cp templates/service.py src/colabllm/services/<name>.py; cp templates/service.ipynb src/colabllm/services/<name>.ipynb`
   (`<name>` = file name = Colab session; hostname uses the project base domain).
2. Edit `src/colabllm/services/<name>.py`: `name`, `notebook`, `service_port`, `auth`, `gpu`, extra `env`.
3. Add the service's `*_CFG`, setup and service cells to `templates/build_notebooks.py` (plus a
   `write(...)` call in `__main__`), keep larger logic in a module under `templates/`, bind to
   `127.0.0.1`, put large downloads on local SSD and durable data in `DATA_DIR`, then run
   `python3 templates/build_notebooks.py`. Do not hand-edit the generated `.ipynb`.
   See [references/notebooks.md](references/notebooks.md).
4. Pick auth -- see [references/auth.md](references/auth.md). Rule: **never expose a service without
   auth**; if it has none, `auth="caddy"`.
5. Update the distributed user skill in `skills/colabllm/` in the same change. Add the runtime to
   `SKILL.md`'s task/runtime/reference table and document its actual user workflow in an appropriate
   per-runtime reference. Canonical usage guides live in `docs/<name>.md`; run
   `python tools/sync_runtime_docs.py` to update the matching portable skill references. Cover prerequisites/GPU,
   startup, inference or UI access, input/output locations, authentication, waiting/recovery and
   service-specific limits. Use installed CLI commands and portable examples, not checkout paths,
   personal domains or credentials. Update the skill description if the new capability changes
   when an agent should invoke it. Runtime removals and command/behavior changes require the same
   reconciliation; the distributed skill is part of runtime delivery, not optional follow-up work.
6. Verify discovery with `colabllm services` and argument forwarding with `colabllm <name> --help`.
   Run `colabllm <name> up --dry-run`, then a real `up` (CPU `--gpu none` for wiring tests), check
   `status` (unauthenticated probe must be 401 for Caddy services), then `down`.
7. Run `python tools/sync_runtime_docs.py --check`, check user-skill links and parse its new examples against the actual CLI without submitting jobs.
   Build with `uv build --no-sources` and run `python tools/check_distribution.py <wheel>`; verify the
   new runtime and complete skill references are included and `colabllm skill --output <temporary-dir>`
   works outside the checkout. Adjust packaging rules and skill-export checks when adding resource
   types or changing the skill layout. Completion requires both the runtime and its exported user
   guidance to be usable from the built package.

## References

- [references/notebooks.md](references/notebooks.md) -- notebook generator, embedded modules, CONFIG helpers, middleware security, cache archives
- [references/auth.md](references/auth.md) -- choosing builtin vs Caddy, client usage, hardening
- [references/cf-tunnel.md](references/cf-tunnel.md) -- Cloudflare quick/named tunnels and project-domain selection
- [references/frp.md](references/frp.md) -- FRP provisioning/adoption, credentials, DNS, diagnostics and cleanup
- [references/troubleshooting.md](references/troubleshooting.md) -- symptoms → causes → fixes
