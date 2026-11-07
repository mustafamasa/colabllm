<!-- Generated from docs/qwen-flash.md; edit the source and run tools/sync_runtime_docs.py. -->

# Qwen Flash: LLM and vision inference

Defaults to **G4** and uses llama.cpp's native Bearer authentication; provides an OpenAI-compatible API with text and vision input.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and credentials

```bash
colabllm qwen-flash up --ready-timeout 3600
colabllm qwen-flash status --json
colabllm qwen-flash info --json
colabllm qwen-flash info --json --show-secrets
colabllm qwen-flash logs -n 100
```

The API base URL is the returned `url` plus `/v1`; do not mistake the web address for a model ID.
`GET /v1/models` returns the actual alias. `/health` may return 200 without authentication, so verify the API key with `/v1/models`: a missing/wrong key must fail.

## Calling the chat API

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
           "max_tokens": 256, "stream": False}
with opener.open(urllib.request.Request(base + "/v1/chat/completions", headers=headers,
        data=json.dumps(payload).encode()), timeout=120) as response:
    print(json.load(response)["choices"][0]["message"]["content"])
```

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
The current model is an IQ3_S GGUF with a BF16 vision projector, configured for about 800K context, YaRN and full GPU offload.
First-time preparation of the large model and compilation on G4 may take a while; its memory/VRAM needs make T4 unsuitable.
The model is staged at `/content/qwen-flash/model` and the llama.cpp build cache is kept in the Drive runtime directory; stopping the VM loses the local SSD files.
The API never stores chat history or responses on Drive automatically; clients must save them. The long-context capability does not mean a single request should generate without limit.

To change to a specific key, update both the private state and the actual server authentication; editing local state alone does not change a running server.
Changing only the key does not require releasing the GPU, but check in-flight requests first, then reload the service as appropriate and verify the new and old keys.
