# Qwen Flash: Qwen3.8-Flash-Next 1M long context and multimodal vision inference

Defaults to **G4** (RTX PRO 6000 Blackwell, `97,887 MiB ≈ 95.6 GiB` VRAM, `~176 GiB` DDR5 RAM) and deploys `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` through `llama-server` (`b11540`) with native Bearer authentication (default `unsloth-UD-Q4_K_XL-V2` calibrated 4-bit MoE quantization, 4 shards totalling `103.69 GiB` on disk + the `mmproj-Qwen3.8-Flash-Next-BF16.gguf` vision projector, `0.85 GiB`). The full **1,048,576 (1M)** context window (YaRN `4.0x` + `q8_0` compressed KV/QSA indexer cache) and multimodal vision input are enabled by default.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## G4 VRAM/RAM planning and profile preview

`Qwen3.8-Flash-Next` (`qwen4exp` architecture) has `48` hybrid layers (`12` QSA full-attention layers + `36` linear-attention SSM layers) and a `26.82 GiB` 3-gram PLE table (`per_layer_token_embd.weight`):

- **PLE lookup resident in DDR5 host RAM (0 MiB VRAM)**: `--override-tensor per_layer_token_embd.weight=CPU --load-mode none --lazy-mode off --no-host` loads the full `26.82 GiB` PLE table into host DDR5 memory and releases the duplicate Linux page cache, so there are no disk page-fault stalls and no GPU VRAM is used.
- **1M context `q8_0` KV + QSA k-pool indexer cache (`16.05 GiB`)**: `-ctk q8_0 -ctv q8_0 -fa on` compresses the `12` QSA KV layers and the MLA indexer cache for the 1M window from `30.1 GiB` at `f16` to `16.05 GiB`.
- **Adaptive expert-layer offload (`--cpu-moe auto`)**: on `G4` (`95.6 GiB` VRAM) the `UD-Q4_K_XL-V2` GPU trunk + BF16 vision total `77.7 GiB`; after automatically offloading only `3/48` routed-expert layers (`4.39 GiB`) to DDR5, **`45/48` layers (93.75%) + all attention/SSM/shared experts + the vision projector stay on the GPU (`73.3 GiB`)**, leaving about **`6.2 GiB`** of VRAM headroom on top of the `16.05 GiB` 1M KV cache.

Preview the VRAM/RAM plan for other GPUs or quantizations offline on the local machine:

```bash
colabllm qwen-flash plan
colabllm qwen-flash plan --preset G4
colabllm qwen-flash plan --preset G4 --context 512k
colabllm qwen-flash plan --preset G4 --quant unsloth-UD-IQ4_XS
```

## Startup and credentials

```bash
# Default G4 + UD-Q4_K_XL-V2 + 1M (1,048,576) context + q8_0 KV + BF16 vision
colabllm qwen-flash up --ready-timeout 5400

# Bind the fixed project subdomain (default alias qwen.<base-domain>)
colabllm qwen-flash up --domain

# For zero CPU offload (48/48 layers, 100% on GPU), choose 512K context or UD-IQ4_XS quantization
colabllm qwen-flash up --domain --context 512k
colabllm qwen-flash up --domain --quant unsloth-UD-IQ4_XS

# Show status, the actually allocated n_ctx (1048576) and the API key
colabllm qwen-flash status --json
colabllm qwen-flash models
colabllm qwen-flash info --json
colabllm qwen-flash info --json --show-secrets
colabllm qwen-flash logs -n 100
```

The API base URL is the returned `url` plus `/v1`; do not mistake the web address for a model ID.
`colabllm qwen-flash models` (or `GET /v1/models`) returns the actual alias, `context_allocated` and `modalities.vision`. `/health` may return 200 without authentication, so verify the API key with `/v1/models`: a missing/wrong key must return `401`.

## Calling the chat and vision API

The example below uses only the Python standard library, installs no OpenAI SDK, never prints the key and refuses redirects:

```python
import json
import subprocess
import urllib.request

info = json.loads(subprocess.check_output(
    ["colabllm", "qwen-flash", "info", "--json", "--show-secrets"], text=True))
base = info["url"].rstrip("/")
headers = {"Authorization": "Bearer " + info["api_token"], "Content-Type": "application/json"}
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None
opener = urllib.request.build_opener(NoRedirect())
with opener.open(urllib.request.Request(base + "/v1/models", headers=headers), timeout=30) as response:
    model = json.load(response)["data"][0]["id"]
payload = {"model": model, "messages": [{"role": "user", "content": "Explain vector databases in one sentence."}],
           "max_tokens": 256, "stream": False,
           "chat_template_kwargs": {"enable_thinking": False}}
with opener.open(urllib.request.Request(base + "/v1/chat/completions", headers=headers,
        data=json.dumps(payload).encode()), timeout=120) as response:
    msg = json.load(response)["choices"][0]["message"]
    print(msg.get("content") or msg.get("reasoning_content"))
```

The server defaults to `--reasoning on`; clients can switch a single request to low-latency direct answers with `chat_template_kwargs: {"enable_thinking": false}`, or set it to `true` for deep thinking (the thinking content is in `message.reasoning_content`).

For vision input, change the message `content` to a list of content blocks:

```json
[{"type":"text","text":"Describe the image"},{"type":"image_url","image_url":{"url":"data:image/png;base64,IMAGE_BASE64"}}]
```

Use a reachable URL supported by the service or real base64 data for images; local file paths cannot be passed to the remote model directly.
`IMAGE_BASE64` is a placeholder. The service has no `chat` CLI subcommand; call the API.
For streaming, use `stream: true` and handle SSE; a streamed response cannot be parsed as one JSON document.

## Multiple instances and storage

```bash
colabllm qwen-flash up -i 2
colabllm qwen-flash info -i 2 --json
colabllm qwen-flash ps --json
colabllm qwen-flash down -i 2
```

When calling instance 2, add `-i 2` to the `info` call in the Python example as well.
On G4 the first download of about `104.5 GiB` of model shards and the `sm_120` `llama.cpp` compilation take a while, so keep `--ready-timeout 5400`; the VRAM and host-RAM needs make `T4` or `L4` unsuitable.
Model shards are staged on the fast local NVMe SSD at `/content/qwen-flash/model` (`imatrix_micro3.gguf` is skipped automatically and `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` prevents double disk usage); the `llama.cpp` build output is archived on Drive (`/content/drive/MyDrive/models/.runtimes/llama.cpp`). Stopping the VM releases the local SSD files.
The API never stores chat history or responses on Drive automatically; clients must save them. The 1M long-context capability is for very long inputs and multi-turn context; it does not mean a single request should generate without limit.

To change to a specific key, update both the private state and the actual server authentication; editing local state alone does not change a running server.
Changing only the key does not require releasing the GPU, but check in-flight requests first, then reload the service as appropriate and verify the new and old keys.
