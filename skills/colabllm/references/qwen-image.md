<!-- Generated from docs/qwen-image.md; edit the source and run tools/sync_runtime_docs.py. -->

# Qwen-Image-2.1: unified text to image, 1-4 image `<imageN>` editing, native RGBA transparency and direct Google Drive access

Defaults to **G4** (RTX PRO 6000 Blackwell, 96 GB VRAM) and also supports **H100** (80 GB HBM3) and **A100** (40 GB / 80 GB). Deploys `Qwen/Qwen-Image-2.1` online following the [official vLLM-Omni Qwen-Image-2.1 recipe](https://recipes.vllm.ai/Qwen/Qwen-Image-2.1) (7.1B single-stream block-causal attention DiT + exact cross-step prefix KV cache + Qwen3-VL-8B multimodal encoder + 16x four-channel RGBA VAE), with native fail-closed Bearer authentication (including stateless CORS `OPTIONS` preflight); supports bilingual (Chinese/English) text to image, editing with 1-4 index-bound `<image1>..<image4>` references, native RGBA transparent cut-outs/flat-lay exports, LoRA adapters, and reading/writing images on the mounted Google Drive directly.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Prerequisites

`Qwen/Qwen-Image-2.1` is released under the **Apache-2.0** license and is **public and ungated** on Hugging Face (it downloads without accepting a license; an optional `HF_TOKEN` only raises Hugging Face Hub download rate limits).

## Choosing G4, H100 or A100, with startup and usage details

The model weights total **33.1 GB** on disk (`Qwen3-VL-8B` text/vision encoder 17.5 GB BF16 + `QwenImage21Transformer2DModel` 7.1B DiT 14.2 GB BF16 + four-channel RGBA VAE 1.4 GB FP32); with the default `--vae-use-tiling`, the BF16 peak VRAM for `1024x1024` at 40 steps is about **34-40 GB**:

| Colab GPU | Architecture and VRAM | Recommended quantization (`--quantization`) | Peak VRAM (1024²) | Use case and notes |
| --- | --- | --- | --- | --- |
| **G4 (default)** | Blackwell `SM120` · **96 GB GDDR7** | `none` (default full BF16) | ~35-40 GB (>55 GB headroom) | **First choice for everyday long-running use**: Colab's large-VRAM tier gives better compute-unit value per hour; full BF16 + 4 high-resolution reference images + `--step-execution` batch concurrency with no VRAM pressure |
| **H100** | Hopper `SM90` · **80 GB HBM3** | `none` (default full BF16) | ~35-40 GB (~40 GB headroom) | **Lowest-latency tier**: 3.35 TB/s HBM3 bandwidth + CUDA Graphs give the lowest latency for a single 40-step `1024x1024` image |
| **A100** | Ampere `SM80` · **40 GB / 80 GB** | `fp8-all` (required for 40 GB) or `none` (80 GB) | ~28 GB (`fp8-all`) | **40 GB compatibility tier**: online FP8 for both Qwen3-VL-8B and the DiT's non-sensitive layers brings VRAM down to about 28 GB |
| **L4 (not supported)** | Ada `SM89` · **24 GB GDDR6** | - | Exceeds the 24 GB physical limit | 33.1 GB of weights (even the lowest online FP8 needs ~28 GB) cannot stay resident in 24 GB of VRAM; for light 24 GB use cases use [FLUX 3 Image Klein](flux-klein.md) with `--gpu L4 --variant fp8r` |

### 1. Using G4 (default recommendation: 96 GB full BF16)

Without `--gpu`, **G4** is allocated by default and `--vae-use-tiling` adaptive tiled decoding is enabled to prevent OOM:

```bash
# Default G4 + full BF16 (temporary tunnel)
colabllm qwen-image up

# G4 + fixed domain
colabllm qwen-image up --domain

# G4 + fixed domain + cross-request step-level continuous batching (--step-execution, default max-num-seqs=8)
colabllm qwen-image up --domain --step-execution

# G4 + preload a LoRA adapter from Google Drive
colabllm qwen-image up --domain --lora-path drive://loras/anime-style.safetensors --lora-scale 1.0
```

### 2. Using H100 (80 GB HBM3 fast tier)

```bash
# H100 + full BF16 + fixed domain
colabllm qwen-image up --gpu H100 --domain

# H100 + step-level continuous batching + custom maximum concurrent sequences
colabllm qwen-image up --gpu H100 --domain --step-execution --max-num-seqs 8
```

### 3. A100 / VRAM-optimized usage (online FP8 and prefix KV quantization)

```bash
# A100 (40 GB / 80 GB) + online FP8 for all components (keeps the composition-sensitive img_mlp layers in BF16, peak VRAM about 28 GB)
colabllm qwen-image up --gpu A100 --domain --quantization fp8-all

# For long prompts / 4 reference images at high concurrency, also compress V of the cross-step prefix KV cache to FP8
colabllm qwen-image up --domain --quantization fp8-text --prefix-kv-dtype fp8_v
```

### Checking status and service configuration

```bash
colabllm qwen-image status --json
colabllm qwen-image models
colabllm qwen-image models --json
```

> **Default sampling parameters (injected automatically)**:
> Upstream `vllm serve --omni` defaults to 50 steps and `true_cfg_scale=4.0`, while `Qwen/Qwen-Image-2.1` officially recommends **40 steps (`num_inference_steps=40`) and turning off redundant CFG when there is no negative prompt (`true_cfg_scale=1.0`)**. This service's middleware and CLI inject these defaults automatically: without `--negative-prompt` a single-path `true_cfg_scale=1.0` is used (twice as fast); with a non-empty `--negative-prompt` it switches to `true_cfg_scale=4.0` for true CFG two-path guidance.

## Text to image and saving to Google Drive

`Qwen-Image-2.1` natively supports high-quality **Chinese and English** prompts and renders Chinese/English typography in the image; resolutions are aligned to a `32`-pixel grid automatically (default `1024x1024`):

```bash
# Generate from a prompt and save locally
colabllm qwen-image generate 'A celadon teapot on a wooden tea table in morning sunlight, wisps of steam, photorealistic' -o ./teapot.png

# Write the result directly to the mounted Google Drive (no large image bytes sent back)
colabllm qwen-image generate 'A retrofuturistic tram crossing a misty harbor bridge at dawn, crisp signage reading "HARBOR LINE 07"' --save-drive outputs/tram.png

# Equivalent: point -o at a drive:// path directly
colabllm qwen-image generate 'Cyberpunk ramen shop sign on a rainy night, neon lights clearly spelling "MIDNIGHT DINER"' -o drive://outputs/ramen-sign.png

# Enable a negative prompt (activates true_cfg_scale=4.0 automatically) and save to both Google Drive and local
colabllm qwen-image generate 'A detailed anime-style girl character illustration, soft lighting' \
  --size 1024x1536 \
  --negative-prompt 'blurry, low quality, extra fingers, distortion' \
  --save-drive outputs/anime-card.png \
  -o ./anime-card.png
```

## 1-4 image `<imageN>` reference editing, native RGBA transparent cut-outs and direct Google Drive access

Supports **1-4** reference images bound automatically, in the order given, to `<image1>`, `<image2>`, `<image3>` and `<image4>` (a 5th image returns HTTP 400), and the built-in 16x **four-channel RGBA VAE** can keep or generate a transparent alpha channel directly:

```bash
# List input images, past outputs or LoRA weights on Drive
colabllm qwen-image ls-drive
colabllm qwen-image ls-drive 'MyDrive/photos/*.png'
colabllm qwen-image ls-drive --outputs
colabllm qwen-image ls-drive --loras

# Multi-reference pose transfer and outfit swap (precise <image1>..<image3> references, reading Google Drive with no uploads)
colabllm qwen-image edit 'Make the person in image 1 <image1> strike the pose from image 2 <image2> and wear the outfit from image 3 <image3>; strictly keep the face and hairstyle from image 1, with a clean, natural background' \
  --drive-image inputs/person.jpg inputs/pose.jpg inputs/outfit.png \
  --size 1024x1536 \
  --save-drive outputs/pose-costume.png \
  -o ./pose-costume.png

# Outfit extraction / transparent-background asset generation (--rgba keeps a 4-channel RGBA PNG with transparent background)
colabllm qwen-image edit 'Extract the complete outfit (top and bottom) the person in image 1 <image1> is wearing, laid flat and centered on a transparent background, with no body or hanger visible' \
  --drive-image inputs/model-photo.jpg \
  --rgba \
  --save-drive outputs/extracted-garment-rgba.png \
  -o ./extracted-garment-rgba.png

# Without --size (default auto), the reference image aspect ratio is inherited (about 1024x1024 area, 32-pixel aligned)
colabllm qwen-image edit 'Write the words "FRESH BASIL" on this ceramic plate in dark green hand-painted lettering' \
  --image drive://inputs/plate.png \
  --save-drive outputs/plate-basil.png
```

## Connecting frontend workbenches (such as `open-pose-costume`) and the OpenAI Images SDK

The middleware has the following compatibility features built in, so it can serve as an OpenAI Images API-compatible backend for web frontends (such as `open-pose-costume`) or the official `openai` SDK:

- **CORS preflight**: `OPTIONS` preflight returns `204` with `Access-Control-Allow-*` headers, so browsers can connect directly with `Authorization: Bearer <API_TOKEN>`.
- **Automatic model alias mapping**: `"Qwen/Qwen-Image-2.1"`, `"qwen-image-2.1"`, `"qwen-image"` and `"gpt-image-2"` / `"gpt-image-1.5"` / `"gpt-image-1"` in requests all map to the loaded model and never trigger `Model mismatch`.
- **Dual-format `/v1/images/edits`**: upstream vLLM-Omni's native `/v1/images/edits` only accepts `multipart/form-data` (posting JSON directly drops the connection, and the `url` form field has a 1024 KB limit); this service's middleware accepts both **`multipart/form-data` and `application/json`**, converting file streams, `data:image/...` and `drive://...` into unlimited-size local loopback binary uploads that fully keep 4-channel RGBA transparency and `size="auto"` adaptive sizing.

## Full HTTP API reference

Read the current instance address and Bearer token:

```bash
colabllm qwen-image info --json --show-secrets
```

Except for stateless CORS `OPTIONS` preflight, all endpoints (including `/health`, `/v1/*` and `/api/ct/*`) require `Authorization: Bearer <API_TOKEN>` (or `X-API-Key: <API_TOKEN>`):

| Method and path | Purpose | Key parameters and Google Drive extensions |
| --- | --- | --- |
| `POST /v1/images/generations` | OpenAI text to image (returns 4-channel RGBA or RGB `b64_json`) | Defaults `num_inference_steps=40`, `true_cfg_scale=1.0`; supports `"save_to_drive"`, `"drive_only"`, `"color_format"`, Drive `lora.path` |
| `POST /v1/images/edits` | OpenAI-compatible 1-4 image editing (both `multipart/form-data` and JSON) | Supports `image` / `image[]` file uploads, `drive://...`, `size="auto"`, `background="transparent"`, `"save_to_drive"` |
| `POST /api/ct/edit` | Simple JSON 1-4 image editing (returns `{created, data: [{b64_json, ...}]}`) | `"images": ["drive://inputs/a.png", ...]` + `"save_to_drive": "outputs/b.png"` |
| `POST /v1/chat/completions` | vLLM-Omni multimodal `image_url` editing (1-4 reference images) | `image_url.url` accepts `drive://...` directly; supports `extra_body` and `"save_to_drive"` |
| `GET /api/ct/drive-list` | Lists Drive images, outputs or LoRAs (`?path=...&kind=images\|outputs\|loras`) | Strictly limited to `MyDrive` and the service data directory |
| `GET /api/ct/info` / `GET /v1/models` | Queries the current model, quantization, step-batching state, default sampling parameters and Drive paths | Returns `rgba_vae`, `max_reference_images`, `drive_inputs_dir`, `drive_outputs_dir` |

Python example:

```python
import base64
import json
import os
import urllib.request

base_url = os.environ["QWEN_IMAGE_URL"].rstrip("/")
token = os.environ["QWEN_IMAGE_API_TOKEN"]

payload = {
    "model": "Qwen/Qwen-Image-2.1",
    "prompt": "Make the person in image 1 <image1> strike the pose from image 2 <image2>, keeping the face and clothing details consistent",
    "images": ["drive://inputs/person.png", "drive://inputs/pose.png"],
    "size": "auto",
    "num_inference_steps": 40,
    "true_cfg_scale": 1.0,
    "save_to_drive": "outputs/pose-transfer.png",
    "seed": 42,
}
req = urllib.request.Request(
    f"{base_url}/v1/images/edits",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=300) as resp:
    result = json.loads(resp.read().decode("utf-8"))

print("Saved on Drive:", result.get("rel_drive_paths"))
with open("pose-transfer.png", "wb") as f:
    f.write(base64.b64decode(result["data"][0]["b64_json"]))
```

## Local SSD weights and build cache archives

- **Fast loading from the local NVMe SSD**: `Qwen/Qwen-Image-2.1` (about 33.1 GB) is downloaded and cached on the VM's local NVMe SSD (`/content/qwen-image/model/Qwen-Image-2.1`); the Xet temporary chunk cache is disabled to avoid bloat and `/root/.cache/huggingface` is cleaned after the download, keeping plenty of local disk free on Colab.
- **Exact cross-step prefix KV cache and per-architecture build cache archives**: the prompt and up to 4 reference images are encoded only once in the first denoising step and reused exactly for the following 39 steps; after a warm-up generation, the FlashInfer, Triton and vLLM/TorchInductor build caches are packed into separate archives per GPU architecture (e.g. `sm120`, `sm90`, `sm80`) under `MyDrive/colab-llm/qwen-image/cache-archives/`.
