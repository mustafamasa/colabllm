<!-- Generated from docs/deepseek-coder.md; edit the source and run tools/sync_runtime_docs.py. -->

# DeepSeek Coder: DeepSeek-V4.1-Flash coding and vision inference (hardware-adaptive maximum context)

Defaults to **G4** (RTX PRO 6000 Blackwell, 96 GB VRAM) and also supports **H100** (80 GB HBM3) and **A100** (80 GB / 40 GB). Deploys `Abiray/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-Coder-GGUF` with `llama-server`'s native Bearer authentication (50% routed-expert pruning to `192` experts/layer: `IQ2_XXS` backbone 72.1 GB + `IQ4_NL` Engram conditional memory table 110.6 GB + `BF16` `deepseek41v` vision projector 0.97 GB), and at startup sets the maximum safe context window from the actual GPU VRAM and system memory (`G4` defaults to the full **1,048,576 (1M)** context + **103 GiB Engram fully resident in host DDR5 memory** + **thinking on by default, with clients deciding per request whether to think**).

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Hardware tiers and automatic context window

The default `--context auto` probes the Colab VM's VRAM and system memory live and, combining `deepseek41`'s cross-layer shared compressed KV cache (`f16` only about `3.125 KiB/token`, about `3.125 GiB` KV for a 1M context) with the graph workspace budget, aligns to the maximum window, stepping down automatically when the cold-start VRAM peak does not fit:

| GPU | Weight and Engram placement | Default automatic window (`--context auto`) | Notes |
| :--- | :--- | :--- | :--- |
| **G4 (default, 96 GB VRAM / ~176 GiB RAM)** | `full-gpu` + `engram_in_ram=true` | **`1,048,576` (full 1M)** | 73.1 GB backbone + vision fully in VRAM, 103 GiB Engram fully in host DDR5 memory (no cloud-disk page faults), `-b 4096 -ub 512` fast prefill |
| **H100 / A100 (80 GB VRAM / ~167 GiB RAM)** | `full-gpu` + `engram_in_ram=true` | **`~638,976` (~624K)** | Full GPU + Engram resident in memory; passing `--context 1m` explicitly offloads 3 MoE expert layers to memory to fit a 1M window |
| **A100 (40 GB VRAM / ~83.5 GiB RAM)** | `hybrid-cpu-moe` + `mmap/lazy` | **`~327,680` (320K)** | Attention and shared experts resident on GPU, routed experts offloaded to system memory, Engram falls back to on-demand paging from the local SSD |

Preview the plan for each tier or a custom VRAM size locally and offline:

```bash
colabllm deepseek-coder plan
colabllm deepseek-coder plan --preset G4
colabllm deepseek-coder plan --preset H100 --context 1m
colabllm deepseek-coder plan --vram-gb 80 --ram-gb 167
```

## Startup and credentials

```bash
# Default G4 + automatic full 1M context + thinking on by default (--reasoning on; clients can turn it off per request)
colabllm deepseek-coder up --ready-timeout 5400

# G4 + bind the project's fixed subdomain (default alias deepseek.<base-domain>)
colabllm deepseek-coder up --domain

# H100 (80 GB) with the automatic maximum full-GPU window or an explicit 1M window (a few MoE layers offloaded automatically)
colabllm deepseek-coder up --gpu H100 --domain
colabllm deepseek-coder up --gpu H100 --domain --context 1m

# To answer directly by default for requests that do not set enable_thinking, pass --reasoning off; a default thinking effort can also be set (minimal|low|medium|high|xhigh|max or 1..100)
colabllm deepseek-coder up --domain --reasoning off
colabllm deepseek-coder up --domain --reasoning-effort medium

# Query the running state, the actually allocated n_ctx and the auth key
colabllm deepseek-coder status --json
colabllm deepseek-coder models
colabllm deepseek-coder info --json --show-secrets
colabllm deepseek-coder logs -n 100
```

The first start compiles `llama-server` for the GPU architecture in parallel (cache archived automatically to `MyDrive/colab-llm/deepseek-coder/runtime-cache/`) and downloads about 183.7 GB of model shards to the Colab local NVMe SSD (`/content/deepseek-coder/model`, with an automatic `Engram` symlink for shard 6 and `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` to avoid double disk usage); keeping `--ready-timeout 5400` is recommended.

The API base URL is the returned `url` plus `/v1`; `colabllm deepseek-coder models` verifies authentication and returns the model alias actually in effect, `context_allocated` (`n_ctx`) and the `vision` modality state. `/health` may return 200 without authentication, so verify the API key against `/v1/models`.

## Calling code and multimodal APIs (compatible with Claude Code / OpenAI clients)

The example below calls `/v1/chat/completions` with the Python standard library only, installs no third-party SDK, never prints the key and refuses redirects:

```python
import json
import subprocess
import urllib.request

info = json.loads(subprocess.check_output(
    ["colabllm", "deepseek-coder", "info", "--json", "--show-secrets"], text=True))
base = info["url"].rstrip("/")
headers = {"Authorization": "Bearer " + info["api_token"], "Content-Type": "application/json"}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

opener = urllib.request.build_opener(NoRedirect())
with opener.open(urllib.request.Request(base + "/v1/models", headers=headers), timeout=30) as response:
    model = json.load(response)["data"][0]["id"]

payload = {
    "model": model,
    "messages": [{"role": "user", "content": "Implement an async TCP port reachability probe with timeout control in Python."}],
    "temperature": 0.2,
    "max_tokens": 2048,
    "stream": False,
}
with opener.open(urllib.request.Request(
        base + "/v1/chat/completions", headers=headers, data=json.dumps(payload).encode()), timeout=180) as response:
    msg = json.load(response)["choices"][0]["message"]
    print(msg.get("content") or msg.get("reasoning_content"))
```

Thinking is on by default (`--reasoning on --reasoning-format deepseek`) and clients decide whether to think: in **any single request** to `/v1/chat/completions`, an agent or client can set `chat_template_kwargs.enable_thinking` to `false` for low-latency direct answers (good for frequent tool calls) or `true` to think, adjust the thinking effort (`minimal`/`low`/`medium`/`high`/`xhigh`/`max` or an integer `1..100`, default calibrated value `50`), or set a hard cap on thinking tokens (thinking content is separated into `message.reasoning_content` automatically):

```json
{
  "chat_template_kwargs": {
    "enable_thinking": true,
    "reasoning_effort": "medium"
  },
  "reasoning_budget_tokens": 1024
}
```

For multimodal vision input (e.g. screenshot to code, architecture diagram analysis), set the message `content` to a list of content blocks:

```json
[{"type": "text", "text": "Write the matching HTML and Tailwind CSS for this UI screenshot"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,IMAGE_BASE64"}}]
```

`IMAGE_BASE64` is a placeholder for the real image's base64; the remote service cannot read local disk paths. If you do not need vision and want to free about 1 GB of VRAM, pass `--no-vision` at startup.

## Multiple instances, storage limits and stopping

```bash
colabllm deepseek-coder up -i 2 --domain
colabllm deepseek-coder models -i 2
colabllm deepseek-coder ps --json
colabllm deepseek-coder down -i 2
```

- **Engram resident in host memory with automatic SSD fallback**: on machines with enough system memory (e.g. `G4` ~176 GiB RAM, `H100` ~167 GiB RAM), the 110.6 GB `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` is read fully into host DDR5 memory via `--load-mode none --lazy-mode off --override-tensor per_layer_token_embd.weight=CPU` and the duplicate file page cache is released, eliminating random cloud-disk page-fault latency; on smaller-memory machines (e.g. `A100-40G`) it switches to `--load-mode mmap --lazy-mode on` on-demand paging. Never point the working directory at `/content/drive`.
- **Low-VRAM machine limits**: `T4` (16 GB) and regular `L4` (24 GB VRAM + ~53 GB RAM) do not have enough total memory for the 72.1 GB backbone; use `G4`, `H100` or `A100`.
- **Persistence**: the compiled `llama-server` archive and verification state are kept in `MyDrive/colab-llm/deepseek-coder/`; `down` releases the Colab VM and the local SSD model copy, and chat content is never written to Drive automatically.
