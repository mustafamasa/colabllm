<!-- Generated from docs/flux-klein.md; edit the source and run tools/sync_runtime_docs.py. -->

# FLUX 3 Image Klein: text to image, multi-reference editing and direct Google Drive access

Defaults to **G4** (RTX PRO 6000 Blackwell, 96 GB VRAM) and also supports **H100** (80 GB HBM3) and **L4** (24 GB). Deploys `black-forest-labs/flux-3-image-klein` (5.6B distilled Flow-Map DiT + Qwen3-VL-4B text encoder + VAE) as a vLLM-Omni online service with native fail-closed Bearer authentication; supports OpenAI-compatible text to image, editing with 1-10 reference images, LoRA adapters, optional `flux-3-image-klein-pup` prompt expansion (Prompt Upsampler), and reading/writing images on the mounted Google Drive directly.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Prerequisites

`black-forest-labs/flux-3-image-klein` (and the non-distilled `flux-3-image-klein-base`) are gated on Hugging Face under the **FLUX Kommunity License**:

1. First accept the license on the Hugging Face model page (`https://huggingface.co/black-forest-labs/flux-3-image-klein`).
2. At startup, provide a valid token with `--hf-token <TOKEN>`, the `HF_TOKEN` environment variable, or `MyDrive/colab-llm/hf_token.txt` on Drive (for a fine-grained token, enable *"Read access to contents of all public gated repos you can access"*).

## Choosing G4, L4 or H100, with startup and usage details

The service allocates **G4** by default and can switch per use case with `--gpu G4|L4|H100`:

| Colab GPU | Architecture and VRAM | Recommended precision (`--variant`) | Same-machine `--enable-pup` | Use case and notes |
| --- | --- | --- | --- | --- |
| **G4 (default)** | Blackwell `SM120` · **96 GB GDDR7** | `bf16` (default, about 19 GiB) | **Supported** (about 43 GiB total, >50 GB headroom) | **First choice for everyday long-running use**: Colab's large-VRAM tier has a lower compute-unit price per hour; full BF16 + 9B PUP + 10 high-resolution reference images + LoRA stay online with no VRAM pressure |
| **H100** | Hopper `SM90` · **80 GB HBM3** | `bf16` (default) or `fp8r` | **Supported** (about 43 GiB total, ~37 GB headroom) | **Lowest latency / official benchmark**: about 3.35 TB/s HBM3 bandwidth + native `SM90` kernels; a single 16-step image is about 20%-35% faster than G4, with the highest compute-unit usage |
| **L4** | Ada `SM89` · **24 GB GDDR6** | `fp8r` (recommended, about 17 GiB) | **Not supported** (24 GB cannot also hold the 9B PUP) | **Very low-cost light tier**: good for text to image without PUP and low-to-medium concurrency image to image, with the lowest compute-unit cost |

> **Precision and kernel notes**:
> - `bf16` (default): the root package's full BF16 weights, the quality baseline; recommended as-is on both **G4** and **H100**.
> - `fp8r`: loads the `variants/fp8r` quantized package (about 17 GB VRAM minimum); strongly recommended on **L4** to leave VRAM headroom for multi-image VAE encoding/decoding. On `G4` (`SM120`) the notebook automatically blocks the incompatible `FlashInferFP8ScaledMMLinearKernel` and falls back to a stable kernel.
> - `nvfp4` and `--fp8-attention`: aimed at data-center Blackwell `SM100` (such as B200); not needed on `G4` (`SM120`), `H100` (`SM90`) or `L4` (`SM89`).

### 1. Using G4 (default recommendation: good value + 96 GB full BF16 + optional PUP)

Without `--gpu`, **G4** is used by default:

```bash
# Default G4 + full BF16 (temporary tunnel)
colabllm flux-klein up --hf-token hf_example

# G4 + fixed domain + same-machine Qwen3.5-9B prompt expansion (PUP)
colabllm flux-klein up --domain --enable-pup

# G4 + preload a LoRA adapter from Google Drive
colabllm flux-klein up --domain --enable-pup --lora-path drive://loras/style.safetensors --lora-scale 1.0
```

### 2. Using L4 (low-cost light tier: 24 GB VRAM + `fp8r` quantization)

On **L4** (24 GB), combine with `--variant fp8r` to reserve enough VRAM for reference image encoding and VAE decoding, and **do not** pass `--enable-pup` (if you need expanded prompts, prepare detailed English descriptions on the client):

```bash
# L4 + FP8R quantization (about 17 GB VRAM used, ~7 GB left for reference images and activations)
colabllm flux-klein up --gpu L4 --variant fp8r --hf-token hf_example

# L4 + fixed domain + FP8R quantization
colabllm flux-klein up --gpu L4 --domain --variant fp8r
```

### 3. Using H100 (official recipe fast tier: 80 GB HBM3)

When you need the fastest single-image latency, high-frequency batch generation or alignment with the official H100 benchmark, pass `--gpu H100` explicitly:

```bash
# H100 + full BF16 + fixed domain + same-machine Qwen3.5-9B PUP
colabllm flux-klein up --gpu H100 --domain --enable-pup

# H100 + FP8R quantized variant (further increases high-resolution concurrent throughput)
colabllm flux-klein up --gpu H100 --domain --variant fp8r --enable-pup
```

### Checking status and model mode

```bash
colabllm flux-klein status --json
colabllm flux-klein models
colabllm flux-klein models --json
```

The distilled `flux-3-image-klein` uses **16-step** flow-map sampling by default (`flow_shift=6.93`) and **does not support** `guidance_scale` or a non-empty `negative_prompt`; if you need 50-step CFG and negative prompts, start with `--model black-forest-labs/flux-3-image-klein-base` (recommended on `G4` or `H100`).

## Text to image and saving to Google Drive

```bash
# Generate and save to a local file
colabllm flux-klein generate 'a serene alpine meadow at golden hour, wildflowers in the foreground' -o ./meadow.png

# Write the result directly to the mounted Google Drive (no large image bytes sent back; fastest)
colabllm flux-klein generate 'minimalist ceramic tea set on warm oak table, soft morning window light' --save-drive outputs/tea-set.png

# Equivalent: point -o at a drive:// path directly
colabllm flux-klein generate 'isometric cozy bookshop cutaway illustration' -o drive://outputs/bookshop.png

# Save to both Google Drive and a local directory, generating several images at once
colabllm flux-klein generate 'futuristic high-speed train crossing a snowy viaduct' --size 1536x1024 -n 2 --save-drive outputs/train.png -o ./train.png
```

Width and height must be multiples of `16` (range `64`-`4096`, default `1024x1024`).
With only `--save-drive` (or `-o drive://...`) and no local `-o`, the server writes the PNG directly to `MyDrive/colab-llm/flux-klein/outputs/` (or the given subpath) on Drive and skips the base64 download; with both `--save-drive` and a local `-o ./out.png`, both sides get a copy.

## Image to image and multi-reference editing (using images on Google Drive directly)

Supports conditioned editing with **1-10** reference images in the order given; references can mix images from the **mounted Google Drive**, local files or `https://` URLs:

```bash
# List input images, past outputs or LoRA weights on Drive
colabllm flux-klein ls-drive
colabllm flux-klein ls-drive 'MyDrive/photos/*.jpg'
colabllm flux-klein ls-drive --outputs
colabllm flux-klein ls-drive --loras

# Edit a reference image read directly from the mounted Google Drive and save the result back to Google Drive (no uploads or downloads at all)
colabllm flux-klein edit 'restyle the scene as a delicate studio Ghibli watercolor painting' \
  --image drive://inputs/street.jpg \
  --save-drive outputs/street-watercolor.png

# Force a relative path to be resolved as a Google Drive path (even if a file with the same name exists in the local current directory)
colabllm flux-klein edit 'replace the daytime sky with dramatic aurora borealis over snowy peaks' \
  --drive-image inputs/mountain.jpg \
  --save-drive outputs/mountain-aurora.png \
  -o ./mountain-aurora.png

# Combined multi-reference editing (up to 10 images, keeping the command-line order image 1, image 2, image 3)
colabllm flux-klein edit 'dress the person from image 1 in the tailored jacket from image 2 inside the cafe from image 3' \
  --drive-image inputs/person.png inputs/jacket.png inputs/cafe.jpg \
  --save-drive outputs/dressed-in-cafe.png
```

Without `--size` / `--width` / `--height`, FLUX 3 computes an output size of about 1MP (`16`-pixel aligned) from the aspect ratio of the first reference image. If a relative `--image` file does not exist locally, the CLI also hands it to the remote VM as a Drive path, resolved safely under `MyDrive/colab-llm/flux-klein/inputs/`, `MyDrive/colab-llm/flux-klein/` or `MyDrive/`.

## Prompt expansion (PUP) and LoRA adapters

With `up --enable-pup` on `G4` or `H100`, you can expand short prompts on their own, or add `-u` / `--upsample` to `generate` / `edit` to expand before generating:

```bash
# Call Qwen3.5-9B PUP on its own to expand a text-to-image or image-to-image prompt (--bbox outputs a 0-1000 layout box table)
colabllm flux-klein upsample 'A man walking with his dog, afternoon sun' --canvas 1536:1024
colabllm flux-klein upsample 'put a red bandana around the dog neck' --drive-image inputs/dog.jpg
colabllm flux-klein upsample 'artisan bakery display counter with price tags' --bbox

# Expand the prompt inline while generating/editing, and load a LoRA stored on Google Drive (raw .safetensors or a PEFT directory)
colabllm flux-klein generate 'retro sci-fi poster of Saturn express' \
  --upsample \
  --lora drive://loras/retro-poster.safetensors \
  --lora-scale 0.9 \
  --save-drive outputs/saturn-poster.png
```

The first time the server receives a LoRA path on Drive, it validates the directory boundary and caches the LoRA on the VM's local NVMe SSD (`/content/flux-klein/loras/`), avoiding Drive FUSE reads on every inference.

## Full HTTP API reference

First read the current instance address and Bearer token (read the secret only when a client call needs it):

```bash
colabllm flux-klein info --json --show-secrets
```

All endpoints (including `/health`, `/v1/*` and `/api/ct/*`) require `Authorization: Bearer <API_TOKEN>` (or `X-API-Key: <API_TOKEN>`):

| Method and path | Purpose | Google Drive extensions |
| --- | --- | --- |
| `POST /v1/images/generations` | OpenAI text to image (returns `b64_json`) | Supports `"save_to_drive"`, `"drive_only"`, `"upsample"`, Drive `lora.path` and the `X-Saved-Drive-Path` response header |
| `POST /v1/chat/completions` | Native vLLM-Omni multi-reference editing (1-10 `image_url`s) | `image_url.url` accepts `drive://...` / `MyDrive/...` directly; supports `"save_to_drive"` and `"drive_only"` |
| `POST /api/ct/edit` | Simple JSON image to image / multi-image editing (returns the OpenAI Images structure) | `"images": ["drive://inputs/a.jpg", ...]` + `"save_to_drive": "outputs/b.png"` |
| `POST /v1/images/edits` | Compatible with the OpenAI SDK's `client.images.edit` (`multipart/form-data` or JSON) | Supports uploaded file fields or `drive://...` string fields, plus `save_to_drive` |
| `POST /api/ct/upsample` | Calls the same-machine `flux-3-image-klein-pup` to expand T2I / I2I prompts | `"images"` can reference `drive://...` directly |
| `GET /api/ct/drive-list` | Lists Drive images, outputs or LoRAs (`?path=...&kind=images\|outputs\|loras`) | Strictly limited to `MyDrive` and the service data directory |
| `GET /api/ct/info` / `GET /v1/models` | Queries the current model, variant, default sampling parameters, PUP state and Drive directories | Returns `drive_inputs_dir`, `drive_outputs_dir`, `drive_loras_dir` |

Python examples (standard `urllib` or the OpenAI SDK both work):

```python
import json
import os
import urllib.request

base_url = os.environ["FLUX_KLEIN_URL"].rstrip("/")
token = os.environ["FLUX_KLEIN_API_TOKEN"]

payload = {
    "model": "black-forest-labs/flux-3-image-klein",
    "prompt": "turn this daylight street photo into rainy neon cyberpunk night, keeping architecture intact",
    "images": ["drive://inputs/street.jpg"],
    "save_to_drive": "outputs/street-cyberpunk.png",
    "drive_only": True,
    "seed": 42,
}
req = urllib.request.Request(
    f"{base_url}/api/ct/edit",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=300) as resp:
    result = json.loads(resp.read().decode("utf-8"))
print("Saved on Drive:", result["rel_drive_paths"])
```

## Local SSD weights and build cache archives

- **Lean on-demand download**: setup first reads the Hugging Face repository metadata and downloads only the root config, `ae.safetensors`, `text_encoder/*` and the matching `model.safetensors` (or `variants/<variant>/*`) needed by the current `--variant` to the local NVMe SSD (`/content/flux-klein/model/`), skipping duplicate Diffusers shards and unselected quantized packages, saving over 20 GB of download and local disk.
- **Build cache archives isolated per GPU architecture**: FlashInfer, Triton and vLLM/TorchInductor caches run on the local SSD; after the warm-up inference they are packed into separate archives per GPU architecture (`sm120` for `G4`, `sm90` for `H100`, `sm89` for `L4`, no conflicts) under `MyDrive/colab-llm/flux-klein/cache-archives/`, so starting on a different GPU hits its own archive directly.
