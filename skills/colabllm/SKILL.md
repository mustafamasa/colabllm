---
name: colabllm
description: >-
  Use installed colabllm runtimes on Google Colab for LLM inference, PDF/image OCR,
  speech transcription, speech synthesis, voice cloning, sound-effect/audio generation,
  avatar/video generation, ComfyUI and Unsloth Studio. Use when the user asks to start,
  query, call or stop these runtimes, retrieve their API key, or troubleshoot a colabllm
  service. Intended for package users; runtime development and notebook editing are outside scope.
---

# ColabLLM runtimes

Operate through the installed `colabllm` command. This skill works without the source repository.
Treat installed `--help` as authoritative for available services, commands and options.

## Discover and select

```bash
colabllm --version
colabllm services
colabllm doctor
colabllm <service> --help
colabllm <service> <command> --help
```

If the command is missing, use the user's supplied wheel or package source. A distributed wheel
installs with `uv tool install /path/to/colabllm-VERSION-py3-none-any.whl`; use
`uv tool install colabllm` only when that package has actually been published to the intended index.
`colab`, its Drive extension and authorization are separate prerequisites. Local model libraries
such as Torch, CUDA and NumPy are unnecessary; computation occurs in Colab.

| Task | Runtime | Reference to read |
| --- | --- | --- |
| Qwen3.8-Flash-Next LLM / vision (1M context on G4) | `qwen-flash` | [Qwen Flash](references/qwen-flash.md) |
| DeepSeek-V4.1-Flash Coder (auto max context up to 1M + vision) | `deepseek-coder` | [DeepSeek Coder](references/deepseek-coder.md) |
| DeepSeek-V4-Pro Qwen3.5-9B reasoning / vision (256K on L4/T4) | `deepseek-qwen-9b` | [DeepSeek-Qwen-9B](references/deepseek-qwen-9b.md) |
| Ollama models, OpenAI/Anthropic/native APIs & tuning | `ollama` | [Ollama](references/ollama.md) |
| PDF/image files or directories to Markdown | `unlimited-ocr` | [Unlimited-OCR](references/unlimited-ocr.md) |
| Audio/video transcription, timestamps, subtitles | `nemotron-transcribe` | [Nemotron](references/nemotron-transcribe.md) |
| Built-in speech, voice design and cloning | `qwen3-tts` | [Qwen3-TTS](references/qwen3-tts.md) |
| Voice cloning, emotion and speed | `indextts` | [IndexTTS](references/indextts.md) |
| Multi-ref voice cloning, expressive tags & novel narration | `drama3` | [Fish Audio Drama 3](references/drama3.md) |
| 44.1 kHz stereo sound effects, ambience & batch SFX cues | `tangoflux` | [TangoFlux](references/tangoflux.md) |
| Audio-driven avatar generation | `longcat-avatar` | [LongCat Avatar](references/longcat-avatar.md) |
| Text/image-to-video and continuation UI | `longcat-video` | [LongCat Video](references/longcat-video.md) |
| Motion transfer, object/face/color video CLI & UI | `openjutsu` | [Openjutsu](references/openjutsu.md) |
| Text-to-image, multi-ref image editing & Drive I/O | `flux-klein` | [FLUX 3 Image Klein](references/flux-klein.md) |
| Bilingual T2I, 1-4 `<imageN>` RGBA editing & Drive I/O | `qwen-image` | [Qwen-Image-2.1](references/qwen-image.md) |
| Workflow models, queue and image/video UI | `comfyui` | [ComfyUI](references/comfyui.md) |
| Training, inference and export UI | `unsloth-studio` | [Unsloth Studio](references/unsloth-studio.md) |

Read the selected runtime reference before constructing its task commands. The [common operations
guide](references/runtime-common.md) defines shared lifecycle, path and timeout semantics. These guides
are bundled with the skill and need no repository access. Their command examples are checked offline
against the real parser; GPU behavior still depends on the installed runtime and available hardware.

Discover first; service names above are not a promise that every installed version includes them.
Reuse a ready instance for the task instead of starting another GPU session.

## Runtime lifecycle

```bash
colabllm <service> ps --json
colabllm <service> status --json
colabllm <service> up
colabllm <service> logs -n 100
```

Choose an instance explicitly with `-i ID` when needed, consistently for startup, calls and queries.
IDs contain 1–64 ASCII letters, digits, underscores or hyphens, with no leading hyphen.
`up` may allocate a paid GPU session and install/download large remote dependencies. Choose a GPU
override only when requested or compatible with the runtime; do not silently downgrade hardware.

Default `up` waits for readiness. `up --detach` only launches a runner; poll `status --json` until
`running` and `ready` are true. A reachable domain or 502 during loading is not model readiness.
Use `up --ready-timeout SECONDS` for a suitable startup wait and `--command-timeout SECONDS` for
external command limits. Exit 124 means timeout; default startup timeout leaves the runner active.
Inspect state/logs before retrying startup. The local machine must remain online: its detached runner
keeps Colab connected. Keep foreground sessions attached if `--foreground` was chosen.

`down` stops the VM and local runner. Stop only the instances covered by the user's request; leave
existing services running after submitting work unless the requested lifecycle calls for shutdown.
Use `down --all` or `down --delete-tunnel` only when all-instance stop or resource deletion is wanted.

## Connections and secrets

```bash
colabllm <service> info --json
# Reveal only when needed by a client or explicitly requested by the user:
colabllm <service> info --json --show-secrets
```

Default output hides API keys. Capture secret fields privately and send them as
`Authorization: Bearer <key>`; do not paste keys into replies, saved scripts or shared logs.
Caddy services also have browser Basic Auth credentials; application-login services may have their
own login fields. Model API tokens and FRP connection tokens are different credentials.
When overriding a URL to a different origin, explicitly supply its matching token using the
service's documented environment variable/flag; saved credentials are not transferable by default.
There is no public command for setting an arbitrary persistent key. `up --new-credentials` creates
a random replacement on a stopped instance. Do not pretend a local state edit rotates a live server.

Configuration/state lives in `~/.colabllm/` or `COLABLLM_HOME`. Respect that selected profile.
Local relative file paths resolve against the caller's directory. `drive://...` means a path under
MyDrive on the remote VM, not a local path, browser sharing URL or Drive file ID.
Task/client interruption need not stop server-side work: retain returned job IDs and resume querying
instead of submitting duplicates. Prefer JSON stdout for automation; diagnostics use stderr.

## Failures and completion

Read [Operations](references/operations.md) for missing dependencies, credentials, tunnel errors,
job recovery or package upgrades. Use offline `doctor` first; `doctor --online` performs bounded,
read-only authorization checks. Missing authorization may require the user to complete login.

Report the runtime/instance, readiness, requested operation, output path or job ID, and any pending
work. A submitted job is not a completed artifact. Verify downloaded files before reporting success.
