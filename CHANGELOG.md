# Changelog

The version number comes only from `pyproject.toml` and follows `MAJOR.MINOR.PATCH`; Git tags are `vMAJOR.MINOR.PATCH`.

## [Unreleased]

### Added

- `colabllm ui`: local web control panel (standard library only, listens on `127.0.0.1` with a per-run access token). Shows the Colab compute balance and burn rate, every instance with status, accelerator, endpoint and uptime, the service catalog, `doctor` checks, runtime logs and masked credentials; deploys (`up --detach`) and stops instances through the packaged service scripts.
- `colabllm ui` deploy dialog: every runtime's own `up` options (e.g. comfyui `--workflow`, ollama `--models`/`--profile`) are read from its argparse parser and validated by that parser before starting. Multiple instances can be selected and stopped together (each as its own `down` operation), or all running instances at once.
- `up` records the accelerator it started with in the instance state (`gpu`).
- `purge` lifecycle command (`colabllm <runtime> purge [-i ID] [--delete-tunnel | --keep-tunnel]`): deletes a stopped instance's local state after `colab stop` confirms no session is left; refuses while the runner is alive. The control panel can delete single instances or a selection.

## [0.1.5] - 2026-11-09

### Added

- New `deepseek-qwen-9b` runtime (L4 by default, T4/A100/H100/G4 compatible, `llama-server` Bearer API key): deploys the `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` GGUF with `i1-Q4_K_M` imatrix weights by default, plus a native `Q8_0` MTP draft sidecar exported remotely and cached on Google Drive (`--spec-type draft-mtp` lossless speculative decoding, about `57–68 tok/s` measured on L4), the `mmproj-f16` vision projector and the full native `256K` (`262,144`) context; provides `plan` and `models` subcommands.
- New `tangoflux` runtime (T4 by default, L4 recommended for speed, Bearer API key): deploys `declare-lab/TangoFlux` 44.1 kHz stereo text-to-sound-effect generation with single `generate`, multiple random variants and resumable `batch` SFX pack commands.
- New `drama3` runtime (G4, SGLang-Omni Bearer API key): deploys `fishaudio/drama-3` multilingual zero-shot voice-cloning speech synthesis with a custom voice library, resumable chapter-by-chapter audiobook synthesis of novels, and automatic multi-character dialogue splitting and voicing through an external OpenAI-compatible LLM.
- `ollama` runtime: new `--preset huihui-27b` one-flag preset that selects the `Huihui-Qwen3.8-27B-abliterated` quantization and context configuration.

### Changed

- `qwen-flash` upgraded to `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` (`unsloth-UD-Q4_K_XL-V2` + the Unsloth `bf16` precision fix patch + `mmproj-BF16`) on `llama.cpp` `b11540`, enabling the full `1M` (`1,048,576`) context on G4 with YaRN 4.0x, DDR5 host-RAM PLE residency and adaptive `--n-cpu-moe 3`; adds the offline `plan` and online `models` subcommands.

### Fixed

- ASGI middlewares now handle the `OSError` raised when `ffmpeg` is missing during the WAV fallback.

### Documentation

- Documented migrating the state directory and fixed-domain configuration across machines, and added a comparison of the DeepSeek runtimes (`deepseek-coder` vs `deepseek-qwen-9b`).

## [0.1.4] - 2026-11-06

### Changed

- `deepseek-coder` now enables thinking by default (`--reasoning on`); clients decide per request through `chat_template_kwargs.enable_thinking` in the request body. Pass `--reasoning off` at startup for the old behavior.

## [0.1.3] - 2026-11-05

### Added

- New `deepseek-coder` runtime (G4, `llama-server` Bearer API key): deploys the `DeepSeek-V4.1-Flash-1017` GSQ-RCO Coder GGUF + Engram conditional memory table + `deepseek41v` vision projector. `--context auto` plans the maximum context from the actual VRAM/memory (full 1M on G4, about 624K on H100/A100 80G, about 320K on A100 40G) and steps down when cold-start VRAM is insufficient; Engram stays resident in host memory when possible and falls back to local SSD paging otherwise; thinking is off by default and can be enabled per request; an offline `plan` preview subcommand is included.
- New `qwen-image` runtime (G4, vLLM-Omni Bearer authentication): deploys `Qwen/Qwen-Image-2.1` with bilingual Chinese/English text to image, editing with 1-4 index-bound `<imageN>` reference images, native RGBA transparency export, LoRA adapters and direct Google Drive image reads/writes, plus ASGI middleware for stateless CORS `OPTIONS` preflight.
- New `flux-klein` runtime (G4, vLLM-Omni Bearer authentication): deploys `black-forest-labs/flux-3-image-klein` with OpenAI-compatible text to image, editing with 1-10 reference images, LoRA, optional PUP prompt expansion and direct Google Drive image reads/writes.
- `ollama` runtime: pulls Ollama/HF GGUF models and aliases automatically at startup and warms up VRAM; `--compat all|ollama|openai|anthropic` compatible API routing (Basic Auth, Bearer, `x-api-key`) with `--read-only-models` protection, `OLLAMA_*` tuning parameters, `--profile` presets, `--models-storage ssd|drive`, and the `endpoints`, `models`, `pull`, `run`, `preload`, `unload`, `cp`, `rm`, `show` subcommands.

### Fixed

- `flux-klein`: normalize the server-side model alias and remove a duplicated Drive subdirectory prefix.

### Documentation

- `openjutsu`: unify the H3 clip length, step and resolution tuning guidance.

## [0.1.2] - 2026-11-04

### Added

- New `openjutsu` runtime (G4, Caddy Basic Auth + Bearer token): runs ComfyUI (with 13 custom node packs and a Linux path patch), `comfy-api-proxy` and a Next.js 16 production workbench based on `colab_openjutsu` on the same machine, with full command-line subcommands (`swap`, `object`, `mask`, `face`, `color`, `job`, `download`, `cancel`, `check`) and live SSE progress tracking.
- `openjutsu` G4 runtime and pull optimizations: `--profile full|core` to prepare weights on demand, native Rust `hf_xet` high-concurrency model pulls in a separate environment (largest-first LPT scheduling with multi-connection HTTP `Range` chunked resume as fallback), millisecond `sm_120` architecture/dependency probing without loading CUDA dynamic libraries, `--vram-mode auto` (enables `--highvram` residency automatically on `>=76 GiB` VRAM), immediate system-memory reclaim for long recursive video segments with ComfyUI `--cache-classic`, and `--chains-storage ssd` checkpoint binding to the local NVMe SSD.

## [0.1.1] - 2026-11-04

Yanked 0.1.0 (its build artifacts contained old deployment example configuration) and fixed security and stability issues.

### Security

- API authentication of the transcription and speech synthesis services now uses constant-time comparison and denies access when the token is empty; the local `/health` authentication exemption was removed.
- Drive/data paths are restricted to MyDrive or the service data directory, rejecting out-of-bounds access such as `../`, `/etc` or arbitrary `file://`; voice names are validated and upload form header fields are filtered.
- ComfyUI model download targets are restricted to `models/`; the local directories vLLM may read were narrowed to the service working directory.
- `colab exec` no longer passes secrets as command-line arguments; they are injected through a 0600 runtime notebook instead.

### Fixed

- Watchdog logic: bounded process-group termination, a 900-second grace period after restarts, incremental log reading and a health check for the qwen3-tts auxiliary worker; public authentication failures or timeouts no longer report readiness.
- All downloads have timeouts; Caddy/cloudflared binaries are chosen by CPU architecture; the public address is refreshed after a Quick Tunnel restart.
- nemotron-transcribe: escaping of remote command paths, local transcription model names for fp8/nvfp4, timestamps enabled automatically for `-o *.srt`, and `download` falling back to local yt-dlp.
- indextts / qwen3-tts: `--timeout 0` means no limit and timeouts exit with 124; qwen3-tts validates the values of `--variant` and `--preload-all`.
- ComfyUI gained `--url` / `--api-token` with progress written to stderr; longcat-avatar gained `--api-token`; environment variables overridden by `up` no longer leak across calls.

### Changed

- Inline notebook code was split into separately unit-testable modules under `templates/`, and `build_notebooks.py` now generates notebooks with constructor functions; a generation consistency test was added.
- The build caches of nemotron, longcat-video and longcat-avatar are archived uniformly with `colab_cache.py`; a `--no-drive` local data directory is supported.
- Personal deployment details were removed from docs and examples, which now use `example.com`; README links now use absolute GitHub URLs.
- Open-sourced under the MIT license; both the wheel and the sdist include `LICENSE`.

## [0.1.0] - 2026-11-03

First packaged release.

- A single `colabllm` CLI, installed and upgraded with `uv tool`, with no model compute dependencies on the local machine.
- 10 Colab runtimes covering inference, OCR, transcription, speech synthesis, video generation, ComfyUI and Unsloth Studio.
- User configuration and state live in `~/.colabllm/`, with multiple instances, Cloudflare Tunnel and FRP.
- Credential queries, diagnostics, shell completion, job recovery and timeout control.
- A bundled agent skill with standalone guides for every runtime and a sync check.
- GitHub Actions verify Linux/macOS with Python 3.12-3.14 and build and publish the wheel and sdist automatically.
