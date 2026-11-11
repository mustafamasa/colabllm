<!-- Generated from docs/deepseek-qwen-9b.md; edit the source and run tools/sync_runtime_docs.py. -->

# DeepSeek-Qwen-9B: high-throughput MTP speculative decoding on L4/T4, 256K long context and multimodal vision

Defaults to **L4** (NVIDIA L4 `sm_89`, `23,034 MiB ≈ 22.5 GiB` VRAM) and is compatible with `--gpu T4` (`16 GB` VRAM) as well as `A100` / `H100` / `G4`. Deploys the GGUF quantization of `xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic` (based on a model distilled from 250K DeepSeek-V4-Pro Max Effect math/STEM samples, decensored with low-loss Heretic v2 ARA) through `llama-server` (`b11540`) with native Bearer authentication. By default it loads **imatrix-calibrated 4-bit weights `i1-Q4_K_M` (`5.24 GiB`)** + **the native Multi-Token Prediction (`MTP`) draft sidecar `mtp-Q8_0.gguf` (`2.27 GiB`, lossless speculative decoding via `--spec-type draft-mtp`)** + the multimodal vision projector **`mmproj-f16.gguf` (`0.86 GiB`)**, and enables the full native **`262,144` (256K)** context window on L4.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## L4 / T4 hybrid architecture + MTP speculative decoding VRAM planning

`Qwen3.5-9B` (`qwen35` architecture) has `32` hybrid trunk layers + `1` native `nextn` multi-token prediction (MTP) head: **`24` Gated DeltaNet linear-attention layers (fixed `~52 MiB` state that does not grow with context) + `8` full-attention trunk layers + `1` full-attention MTP layer** (`4` KV heads, `head_dim=256`):

- **Native 256K context with very little VRAM (`4.83 GiB` KV for trunk + MTP)**: `-ctk q8_0 -ctv q8_0 -ctkd q8_0 -ctvd q8_0 -fa on` uses only **`4,948 MiB` (`4.83 GiB`)** of VRAM for the full `262,144` (256K) window (`4.30 GiB` with `--no-mtp`).
- **L4 default fast profile (`--quant auto` -> `i1-Q4_K_M` + native `MTP`)**: `i1-Q4_K_M` (`5,369 MiB`) + `mmproj-f16` (`876 MiB`) + `mtp-Q8_0` (`2,320 MiB`) + `256K` `q8_0` KV (`4,948 MiB`) measure about **`14,968 / 23,034 MiB`** resident VRAM (still leaving about **`7.9 GiB`** headroom). Halving memory traffic with `i1-Q4_K_M` and verifying several tokens in parallel with `--spec-type draft-mtp` gives a measured **`57–68 tok/s`** for math reasoning/code/vision decoding on L4 (MTP acceptance `78%–93%`, **`2.1x–2.5x`** faster than `27.3 tok/s` with plain `Q8_0`) while keeping the per-token output distribution lossless.
- **L4 high-precision option (`--quant Q8_0`)**: `Q8_0` (`9,087 MiB`) + `mmproj-f16` + `mtp-Q8_0` + `256K` `q8_0` KV total `17.23 GiB` and also fit fully in L4 VRAM.
- **T4 automatic profile (`--gpu T4`)**: `i1-Q4_K_M` + `mmproj-f16` + `mtp-Q8_0` automatically aligns the context to **`212,992` (208K)**; with `--no-mtp` the full **256K** context also fits on T4.

Preview the VRAM plan for each GPU profile or a custom quantization/MTP setting offline on the local machine:

```bash
colabllm deepseek-qwen-9b plan
colabllm deepseek-qwen-9b plan --preset L4
colabllm deepseek-qwen-9b plan --preset L4 --quant Q8_0
colabllm deepseek-qwen-9b plan --preset T4
colabllm deepseek-qwen-9b plan --preset T4 --no-mtp
```

## Startup and credentials

```bash
# Default L4 + i1-Q4_K_M + native MTP speculative decoding + 256K (262,144) context + q8_0 KV + F16 vision
colabllm deepseek-qwen-9b up --ready-timeout 2400

# Bind the fixed project subdomain (default alias deepseek-9b.<base-domain>)
colabllm deepseek-qwen-9b up --domain

# Switch to 8-bit Q8_0 quantization (native MTP + 256K context stay enabled)
colabllm deepseek-qwen-9b up --domain --quant Q8_0

# Low-quota T4 mode (i1-Q4_K_M + MTP, automatic 208K context; add --no-mtp for the full 256K)
colabllm deepseek-qwen-9b up --gpu T4 --domain

# Custom MTP draft length, or extend to 512K context with YaRN
colabllm deepseek-qwen-9b up --domain --mtp-draft 4
colabllm deepseek-qwen-9b up --domain --context 512k

# Show status, the actually allocated n_ctx (262144) and the API key
colabllm deepseek-qwen-9b status --json
colabllm deepseek-qwen-9b models
colabllm deepseek-qwen-9b info --json
colabllm deepseek-qwen-9b info --json --show-secrets
colabllm deepseek-qwen-9b logs -n 100
```

The API base URL is the returned `url` plus `/v1`; do not mistake the web address for a model ID.
`colabllm deepseek-qwen-9b models` (or `GET /v1/models`) returns the actual alias (default `deepseek-v4-pro-qwen3.5-9b`), `context_allocated` and `modalities.vision`. `/health` may return 200 without authentication, so verify the API key with `/v1/models`: a missing/wrong key must return `401`.

## Calling the chat, reasoning and vision API

The example below uses only the Python standard library, installs no OpenAI SDK, never prints the key and refuses redirects:

```python
import json
import subprocess
import urllib.request

info = json.loads(subprocess.check_output(
    ["colabllm", "deepseek-qwen-9b", "info", "--json", "--show-secrets"], text=True))
base = info["url"].rstrip("/")
headers = {"Authorization": "Bearer " + info["api_token"], "Content-Type": "application/json"}
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
opener = urllib.request.build_opener(NoRedirect())
with opener.open(urllib.request.Request(base + "/v1/models", headers=headers), timeout=30) as response:
    model = json.load(response)["data"][0]["id"]
payload = {"model": model, "messages": [{"role": "user", "content": "Find all real roots of x^3 - 6x^2 + 11x - 6 = 0."}],
           "max_tokens": 512, "stream": False,
           "chat_template_kwargs": {"enable_thinking": True}}
with opener.open(urllib.request.Request(base + "/v1/chat/completions", headers=headers,
        data=json.dumps(payload).encode()), timeout=120) as response:
    msg = json.load(response)["choices"][0]["message"]
    print(msg.get("content") or msg.get("reasoning_content"))
```

The server defaults to `--reasoning on`; clients can switch a single request to low-latency direct answers with `chat_template_kwargs: {"enable_thinking": false}`, or set it to `true` for a deep chain of thought (the reasoning is in `message.reasoning_content`).

For vision input, change the message `content` to a list of content blocks:

```json
[{"type":"text","text":"Explain the formulas and charts in the image"},{"type":"image_url","image_url":{"url":"data:image/png;base64,IMAGE_BASE64"}}]
```

Use a reachable URL supported by the service or real base64 data for images; local file paths cannot be passed to the remote model directly.
`IMAGE_BASE64` is a placeholder. The service has no `chat` CLI subcommand; call the API.
For streaming, use `stream: true` and handle SSE; a streamed response cannot be parsed as one JSON document.

## Multiple instances and storage

```bash
colabllm deepseek-qwen-9b up -i 2
colabllm deepseek-qwen-9b info -i 2 --json
colabllm deepseek-qwen-9b ps --json
colabllm deepseek-qwen-9b down -i 2
```

When calling instance 2, add `-i 2` to the `info` call in the Python example as well.
Model weights (default `i1-Q4_K_M`, `5.63 GB`), `mmproj-f16.gguf` (`0.92 GB`) and `mtp-Q8_0.gguf` (about `2.4 GB`) are staged on the fast local NVMe SSD at `/content/deepseek-qwen-9b/model` (`HF_XET_CHUNK_CACHE_SIZE_BYTES=0` prevents double disk usage). The `mtp-Q8_0.gguf` exported via remote HTTP Range requests on first start and the `llama.cpp` runtime compiled per GPU architecture (L4 `sm_89` / T4 `sm_75`) are cached on Drive (`/content/drive/MyDrive/models/`), so later starts restore them in seconds; stopping the VM releases the local SSD files.
The API never stores chat history or responses on Drive automatically; clients must save them.

To change to a specific key, update both the private state and the actual server authentication; editing local state alone does not change a running server.
Changing only the key does not require releasing the GPU, but check in-flight requests first, then reload the service as appropriate and verify the new and old keys.
