# MiniMax-H3-Image: SGLang Diffusion 6-step PDD, native 4K generation and 1-16 image editing

Defaults to **G4** (`NVIDIA RTX PRO 6000 Blackwell Server Edition`, 96 GB GDDR7, `sm_120`), with **H100** (80 GB HBM3) and **A100** (80 GB) support. **SGLang Diffusion** serves `MiniMaxAI/MiniMax-H3-Image` (33.1B DiT, 33.4B Qwen3-VL encoder and MiniMaxH3ImageVAE) with six-step fused PDD distillation and online AdaLN precomputation that releases 24.6 GiB of VRAM. A fail-closed Bearer authentication gateway protects the service.

Native 4K (16 MP) images can reach 30-50 MB, expanding to 40-67 MB as Base64 JSON; long connections are also vulnerable to tunnel and proxy interruptions. The CLI defaults to the network-friendly `/api/ct/*` protocol: asynchronous jobs, small JSON polls, raw binary reference uploads with SHA-256 deduplication, resumable HTTP `Range` downloads, server-side WebP/JPEG conversion, and direct Google Drive reads and writes without public transfer. OpenAI-compatible `/v1/images/generations` and `/v1/images/edits` endpoints remain available.

See [common runtime operations](runtime-common.md) for installation, domains, authentication, timeouts and shutdown. Run the following commands locally.

## Prerequisites

`MiniMaxAI/MiniMax-H3-Image` is public on Hugging Face; set `HF_TOKEN` optionally for higher download limits. Startup skips duplicate `transformer_diffusers/` and `vae_diffusers/` copies, points `text_encoder` to the local subdirectory and downloads only **135.3 GB** of core weights to Colab's local NVMe SSD, leaving about 110 GB free in `/content`. After loading into VRAM, `POSIX_FADV_DONTNEED` releases the host page cache.

## GPU selection and startup

| Colab GPU | Architecture and VRAM | Recommended quantization and residency | Peak VRAM | Speed reference |
| --- | --- | --- | --- | --- |
| **G4 (recommended default)** | Blackwell `sm_120` · **96 GB GDDR7** | `--quantization fp8-all` (all components resident + online AdaLN) | **~50.4 GiB** (~45 GiB headroom) | **2K ~2.6s / native 4K 16:9 ~15.7s**, measured; 4K editing with 1-16 references fits in VRAM |
| **G4 (original precision)** | Blackwell `sm_120` · **96 GB GDDR7** | `--quantization bf16` (native BF16 resident + online AdaLN) | **~92.2 GiB** | **2K ~3.43s / native 4K ~18.54s**, retaining original precision |
| **H100 / A100-80GB** | Hopper / Ampere · **80 GB HBM** | `fp8-all` (all resident) or `bf16` (automatically offloads `text_encoder`) | ~50.4 GiB (FP8) / ~66.1 GiB (BF16 offload) | H100 HBM3 bandwidth enables low-latency resident FP8 |

```bash
# Default G4 with all components resident in FP8 (50.4 GiB VRAM, 4K ~14.5s, 2K ~2.3s)
colabllm minimax-h3-image up

# G4 with native BF16 components resident (92.2 GiB VRAM; online AdaLN saves 24.6 GiB)
colabllm minimax-h3-image up --quantization bf16

# Skip PDD grid warmup to become ready sooner (initialize on the first request)
colabllm minimax-h3-image up --no-warmup

# H100 80 GB with all FP8 components resident
colabllm minimax-h3-image up --gpu H100 --quantization fp8-all

# Show service status, quantization and native 4K presets
colabllm minimax-h3-image status --json
colabllm minimax-h3-image models
colabllm minimax-h3-image models --json
```

> **Sampling contract (validated and normalized by the gateway):**
> `MiniMax-H3-Image` uses six fused PDD output heads and single-positive-branch distillation. Steps are fixed at **6** (`num_inference_steps=6`) with `guidance_scale=1`; **`negative_prompt` and LoRA are not supported**. The default minimum canvas area is `min_generation_area=4194304` (`2048x2048`).

## Native 4K and fast 2K text-to-image

By default, the CLI submits to `/api/ct/jobs` and polls small JSON responses. For `.webp` or `.jpg` output, the gateway encodes remotely and returns raw binary; a 40 MB 4K PNG can shrink to **2-4 MB** for faster transfer:

```bash
# Default native 4K (1:1 4096x4096), remotely encoded as high-quality WebP (~2-4 MB)
colabllm minimax-h3-image generate 'Rain-soaked Tokyo alley at dusk, warm izakaya lanterns reflected on wet asphalt, detailed editorial photograph' -o ./alley-4k.webp

# Official 16:9 native 4K preset (5472x3072), lossless PNG output
colabllm minimax-h3-image generate 'Aerial photograph of winding Alpine pass at golden hour, crisp snow ridges' --aspect 16:9 --tier 4k -o ./alpine-16x9.png

# Fast 2K tier (--tier 2k, 2048x2048, ~2.3 seconds on G4 FP8) for composition checks
colabllm minimax-h3-image generate 'Minimalist ceramic coffee dripper on walnut counter, morning sunlight' --tier 2k -o ./coffee-2k.webp

# 1K sketch preview (--fast-preview lowers min_generation_area to the canvas pixel area)
colabllm minimax-h3-image generate 'Isometric cozy bookshop cutaway illustration' --tier 1k --fast-preview -o ./sketch-1k.webp

# Write native 4K PNG directly to mounted Google Drive without downloading image bytes
colabllm minimax-h3-image generate 'Macro shot of mechanical tourbillon watch movement' --aspect 3:2 --save-drive outputs/watch-4k.png --drive-only

# Submit asynchronously (--no-wait returns job_id immediately; inference continues after disconnection)
colabllm minimax-h3-image generate 'Futuristic botanical conservatory dome at night' --aspect 21:9 --no-wait
```

Official aspect-ratio presets (`--aspect`):

| Aspect | `--tier 4k` (native 16 MP) | `--tier 2k` (fast 4 MP) | `--tier 1k` (1 MP) |
| --- | --- | --- | --- |
| `1:1` | `4096x4096` | `2048x2048` | `1024x1024` |
| `4:3` / `3:4` | `4736x3552` / `3552x4736` | `2368x1760` / `1760x2368` | `1184x896` / `896x1184` |
| `3:2` / `2:3` | `5024x3360` / `3360x5024` | `2496x1664` / `1664x2496` | `1248x832` / `832x1248` |
| `16:9` / `9:16` | `5472x3072` / `3072x5472` | `2752x1536` / `1536x2752` | `1376x768` / `768x1376` |
| `21:9` / `9:21` | `6272x2688` / `2688x6272` | `3136x1344` / `1344x3136` | `1568x672` / `672x1568` |

## Editing with 1-16 reference images and upload deduplication

References bind in command-line order to `<Picture 1>`, `<Picture 2>`, up to **16 images**. The gateway also converts `<image1>` to `<Picture 1>`. Local images are uploaded as raw binary without Base64's 33% expansion and cached by SHA-256, so repeated edits using the same file require **zero additional upload bytes**:

```bash
# List reference inputs or previous outputs on Google Drive
colabllm minimax-h3-image ls-drive
colabllm minimax-h3-image ls-drive --outputs

# Combine local references (cached after first upload; defaults to the aspect ratio of <Picture 1>)
colabllm minimax-h3-image edit 'Place the character from <Picture 1> inside the sunlit studio from <Picture 2>, wearing the jacket from <Picture 3>, natural shadows' \
  -r ./person.png -r ./studio.jpg -r ./jacket.png \
  --tier 2k \
  -o ./composite.webp

# Cloud-only 4K editing on Google Drive, without upload or download traffic
colabllm minimax-h3-image edit 'Keep the product bottle in <Picture 1> unchanged and place it on mossy volcanic rock beside a mountain stream' \
  --drive-image inputs/bottle.png \
  --tier 4k \
  --save-drive outputs/bottle-outdoor-4k.png \
  --drive-only
```

## Asynchronous jobs and resumable downloads

Jobs submitted with `--no-wait`, or interrupted while waiting by `Ctrl-C` or network trouble, continue on the remote GPU and keep their outputs in the local NVMe queue:

```bash
# List recent jobs, status and duration
colabllm minimax-h3-image job

# Query a job or wait for completion
colabllm minimax-h3-image job job_0123456789abcdef --wait

# Download a completed job with HTTP Range resume and optional server-side WebP/JPEG conversion
colabllm minimax-h3-image download job_0123456789abcdef -o ./result.webp --quality 95
colabllm minimax-h3-image download job_0123456789abcdef -o ./result-lossless.png
```

## HTTP API example

In addition to the CLI's internal protocol (`POST /api/ct/uploads`, `POST /api/ct/jobs`, `GET /api/ct/jobs/{id}` and `GET /api/ct/jobs/{id}/content`), external SDKs can call the standard OpenAI Images endpoints. Every route requires `Authorization: Bearer <API_TOKEN>`:

```python
import base64
import json
import urllib.request

BASE_URL = "https://minimax-h3-image.example.com"
API_TOKEN = "sk-your-token"

req = urllib.request.Request(
    f"{BASE_URL}/v1/images/generations",
    headers={
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "MiniMaxAI/MiniMax-H3-Image",
        "prompt": "Architectural concrete pavilion beside still reflection pool at twilight",
        "size": "2048x2048",
        "seed": 42,
        "output_format": "webp",
        "output_quality": 92,
    }).encode("utf-8"),
)
with urllib.request.urlopen(req, timeout=300) as resp:
    result = json.loads(resp.read().decode("utf-8"))
    with open("pavilion.webp", "wb") as f:
        f.write(base64.b64decode(result["data"][0]["b64_json"]))
```

## Stop the service

```bash
colabllm minimax-h3-image down
```
