<!-- Generated from docs/ollama.md; edit the source and run tools/sync_runtime_docs.py. -->

# Ollama: automatic model pulls, compatible APIs and inference tuning

Defaults to **T4** (upgrade with `--gpu L4|A100|G4` for larger models), behind a Caddy auth layer (browser Basic Auth, `Authorization: Bearer <token>` and Anthropic `x-api-key: <token>`); `ollama serve` listens only on the remote loopback `127.0.0.1:11434`.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). For deployment and environment variables see the [official Ollama docs](https://docs.ollama.com/). All commands below run on the local machine.

## 1. Starting with specific models (automatic download, aliases and VRAM warm-up)

```bash
# Pull and warm up qwen2.5:0.5b by default, with all compatible APIs enabled (ollama + openai + anthropic)
colabllm ollama up

# One or more models (weights pulled automatically, first model warmed up into GPU VRAM)
colabllm ollama up --models qwen3:8b,nomic-embed-text

# Pull a Hugging Face GGUF model directly, or warm up all given models into VRAM
colabllm ollama up --gpu L4 --model hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M --preload-all

# Create model aliases for OpenAI / Claude Code tools (source model pulled automatically, then ollama cp)
colabllm ollama up --gpu L4 --model qwen2.5-coder:7b --alias claude-3-5-sonnet=qwen2.5-coder:7b,gpt-4o=qwen2.5-coder:7b

# Run the larger Qwen3-Coder 30B MoE on L4 (24GB) or G4/A100 (official Ollama tag qwen3-coder:30b)
colabllm ollama up --gpu L4 --model qwen3-coder:30b --context-length 32768 --kv-cache-type q8_0 --keep-alive -1

# Start the service only, without pre-downloading models (pulled on demand later via pull or run)
colabllm ollama up --no-models
```

- `--models` / `--model` / `-m`: repeatable or comma-separated; by default pulls are streamed to completion before `CT_READY` and an empty request warms the main model into GPU VRAM (`--no-preload` skips the warm-up, `--preload-all` warms up all startup models).
- `--models-storage ssd|drive`: default `ssd` (`/content/ollama/models`, fast local NVMe; models must be pulled again after the VM is released); `drive` keeps the model cache on Google Drive (`<DATA_DIR>/models`) for reuse across sessions.
- `--ollama-version 0.17.7`: pins the installed Ollama version when needed.

## 2. Choosing compatible APIs (all on, or restricted)

Ollama natively serves three API families; the default `--compat all` **enables all of them**, and you can restrict which families are exposed at startup:

```bash
# All on by default (native Ollama /api/* + OpenAI /v1/* + Anthropic /v1/messages)
colabllm ollama up --compat all --models qwen3:8b

# Expose only the OpenAI and Anthropic compatible APIs (blocks management endpoints such as /api/delete, /api/pull)
colabllm ollama up --compat openai,anthropic --models qwen3:8b

# All APIs on but read-only models (public calls to /api/pull, /api/delete, /api/create, /api/copy, /api/push are forbidden)
colabllm ollama up --compat all --read-only-models --models qwen3:8b

# Show the enabled APIs, base URLs and routes of the current instance (secrets hidden by default)
colabllm ollama endpoints
colabllm ollama endpoints --json
```

Client configuration for each API (the same `api_token` works for all):

| API type | `--compat` value | Base URL | Auth header | Main routes |
| --- | --- | --- | --- | --- |
| **Native Ollama** | `ollama` | `<URL>` | `Authorization: Bearer <TOKEN>` | `/api/chat`, `/api/generate`, `/api/embed`, `/api/tags`, `/api/ps` |
| **OpenAI compatible** | `openai` | `<URL>/v1` | `Authorization: Bearer <TOKEN>` | `/v1/chat/completions`, `/v1/responses`, `/v1/completions`, `/v1/embeddings`, `/v1/models`, `/v1/audio/*` |
| **Anthropic compatible** | `anthropic` | `<URL>` | `x-api-key: <TOKEN>` or `Authorization: Bearer <TOKEN>` | `/v1/messages`, `/v1/models` (works with the Anthropic SDK and Claude Code) |

Python SDK examples (OpenAI Chat / Responses and Anthropic Messages):

```python
import os
from openai import OpenAI
import anthropic

url = os.environ["OLLAMA_PUBLIC_URL"]
token = os.environ["OLLAMA_API_TOKEN"]

# 1. OpenAI SDK (Chat Completions, Responses API and Embeddings)
oai = OpenAI(base_url=f"{url}/v1", api_key=token)
chat = oai.chat.completions.create(
    model="qwen3:8b",
    messages=[{"role": "user", "content": "Introduce yourself in one sentence."}],
)
resp = oai.responses.create(model="qwen3:8b", input="Write one haiku.")

# 2. Anthropic SDK / Claude Code (Caddy natively accepts x-api-key and Bearer)
claude = anthropic.Anthropic(base_url=url, api_key=token)
msg = claude.messages.create(
    model="qwen3:8b",
    max_tokens=512,
    messages=[{"role": "user", "content": "Hello!"}],
)
```

## 3. Startup tuning parameters (`OLLAMA_*`)

Supports `--profile` presets (`balanced`, `low-vram`, `coding`, `agent`, `throughput`) plus fine-grained overrides:

```bash
# Low-VRAM, long-context tuning (suits a 16GB T4: Flash Attention + q8_0 KV cache + single concurrency)
colabllm ollama up --models qwen3:8b --profile low-vram --context-length 16384

# Coding / agent long-context preset (64K context + q8_0 KV cache + resident in VRAM, keep_alive=-1)
colabllm ollama up --gpu L4 --models qwen2.5-coder:7b --profile coding

# Fine-grained manual server tuning
colabllm ollama up --gpu L4 \
  --models qwen3:8b,nomic-embed-text \
  --context-length 32768 \
  --flash-attention \
  --kv-cache-type q8_0 \
  --num-parallel 2 \
  --max-loaded-models 2 \
  --max-queue 512 \
  --keep-alive -1 \
  --load-timeout 15m \
  --gpu-overhead 512M \
  --max-transfer-streams 8 \
  --no-cloud \
  --ollama-env OLLAMA_NOPRUNE=1
```

Common tuning parameters:

- `--context-length` (`OLLAMA_CONTEXT_LENGTH`): default context window size (tokens); `>= 65536` is recommended for agent/coding use, with a larger-VRAM GPU.
- `--flash-attention` / `--no-flash-attention` (`OLLAMA_FLASH_ATTENTION`): explicitly toggles Flash Attention, which greatly reduces VRAM use for long contexts.
- `--kv-cache-type f16|q8_0|q4_0` (`OLLAMA_KV_CACHE_TYPE`): KV cache quantization; `q8_0` saves about half the KV VRAM with almost no loss (enables Flash Attention automatically unless set explicitly).
- `--num-parallel` (`OLLAMA_NUM_PARALLEL`): concurrent requests per model (total KV VRAM grows with `num_parallel * context_length`; set `1` to save VRAM for a single user).
- `--max-loaded-models` (`OLLAMA_MAX_LOADED_MODELS`): maximum models resident in VRAM at once (e.g. `2` to keep a chat model and an embedding model loaded together).
- `--keep-alive` (`OLLAMA_KEEP_ALIVE`): how long an idle model stays loaded (default `30m`, `-1` never unloads, `0` frees VRAM right after the response).
- `--load-timeout` (`OLLAMA_LOAD_TIMEOUT`), `--gpu-overhead` (`OLLAMA_GPU_OVERHEAD`), `--sched-spread`, `--origins` (`OLLAMA_ORIGINS`), `--no-cloud` (`OLLAMA_NO_CLOUD=1`), `--ollama-env KEY=VALUE` (safely passes through other `OLLAMA_*` variables; overriding the loopback binding `OLLAMA_HOST` is forbidden).

## 4. Managing models at runtime and quick inference (automatic on-demand downloads)

Once the instance is ready, manage models with the CLI or test the three compatible APIs directly without restarting:

```bash
# Show downloaded models and current GPU VRAM residency (/api/tags + /api/ps)
colabllm ollama models
colabllm ollama models --json

# Pull new models online on a running instance (or all instances with -A), optionally creating an alias and warming up into VRAM
colabllm ollama pull qwen2.5:7b --alias gpt-4o-mini --preload

# Run a prompt against a model directly: a missing model is pulled first (--no-auto-pull disables this)
colabllm ollama run qwen3:8b "Explain KV cache quantization in one sentence"
colabllm ollama run qwen3:8b "Say hello" --api openai
colabllm ollama run qwen3:8b "Say hello" --api anthropic

# Warm up in VRAM / unload immediately to free VRAM / copy as alias / show details / delete a model
colabllm ollama preload qwen2.5:7b --keep-alive -1
colabllm ollama unload qwen2.5:7b
colabllm ollama cp qwen2.5:7b my-assistant
colabllm ollama show qwen2.5:7b --json
colabllm ollama rm my-assistant
```

If a restriction such as `--compat openai` blocks `/api/*`, `models` falls back to listing via `/v1/models`, and remote management subcommands that depend on `/api/*` (`pull`, `cp`, `rm`, `preload`) return 403 per the policy.

## 5. Multiple instances and stopping

```bash
colabllm ollama up -i 2 --models qwen2.5:0.5b
colabllm ollama ps --json
colabllm ollama status --json
colabllm ollama down -i 2
```
