# Openjutsu: MiniMax H3 character motion transfer, local video editing, face/color fixes and picture-driven preparation

Defaults to **G4** (RTX PRO 6000 Blackwell, about 96 GB VRAM), behind Caddy Basic Auth / session cookie / Bearer token, running three local service layers on the same Colab GPU VM:

1. **ComfyUI (`127.0.0.1:8188`)**: preinstalls all 13 custom node packs Openjutsu needs, the SolarRenderer long-clip silence patch and Linux path normalization.
2. **Patched `comfy-api-proxy` (`127.0.0.1:8189`)**: relays live generation progress and persists job metadata (500 MB per-file upload limit).
3. **Openjutsu Next.js 16 production server (`127.0.0.1:3000`)**: provides the full web workbench (its UI language comes from the upstream project) and the `/api/*` endpoints, based on [`colab_openjutsu`](https://github.com/deluxebear/colab_openjutsu).

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup, model profiles and health checks

```bash
# Default full profile (about 80.7 GB of weights cached on the local NVMe SSD, --highvram enabled automatically on G4, segment checkpoints written to the local SSD)
colabllm openjutsu up --ready-timeout 3600

# Explicit checkpoint storage and VRAM strategy (--chains-storage drive persists segment checkpoints across VMs)
colabllm openjutsu up --chains-storage ssd --vram-mode auto --ready-timeout 3600

# Prepare only the core H3 motion transfer/local replacement/Montage/grading weights (about 50.8 GB, faster start)
colabllm openjutsu up --profile core --ready-timeout 3600

# Show status, connection info and logs
colabllm openjutsu status --json
colabllm openjutsu info --json
colabllm openjutsu info --show-secrets
colabllm openjutsu logs -n 100

# Verify all installed models and each workflow's nodes through the API (/api/models + /api/setup)
colabllm openjutsu check
colabllm openjutsu check --mode simple
colabllm openjutsu check --profile core
```

Open the web UI from the URL returned by `info`; the browser logs in once with Basic Auth using `auth_user` / `auth_pass` (Caddy then issues a `ct_session` cookie so later SSE progress streams and `<video>` range seeking carry the session). Programmatic calls use `Authorization: Bearer <api_token>` (the environment variables `OPENJUTSU_URL` and `OPENJUTSU_API_TOKEN` can override).

## Command-line usage (submit, track progress and download results without opening the web UI)

With `-o/--output`, the command waits for rendering to finish (printing the current segment `clip N` and sampling-step progress live via SSE; `Ctrl+C` only stops waiting locally and does not interrupt the remote job) and then downloads to a local file. Without `-o` or `--wait`, it returns the job JSON (with `id`) right after the upload is submitted, and the job can be managed later with `job` / `download` / `cancel`.

```bash
# 1. Run a few seconds of SAM 3.1 mask preview first (optional; quickly confirms person-tracking accuracy)
colabllm openjutsu mask --video input.mp4 --length 6 -o mask_preview.mp4

# 2. Single-person motion transfer (simple mode; the DMAD 4-step LoRA is loaded by default; for speed use --steps 4 --clip-seconds 7.5 and a resolution matching the source's short side such as 576p/736p)
colabllm openjutsu swap --video input.mp4 -p character.png --steps 4 --clip-seconds 7.5 -o swap_result.mp4

# 3. Multi-person motion transfer (passing 2-4 -p images switches to advanced mode; for two people add --max-quality for two separate refined passes)
colabllm openjutsu swap --video duet.mp4 -p left.png -p right.png --role "left dancer" --role "right dancer" --steps 4 --clip-seconds 7.5 -o duet_swap.mp4

# 4. Asynchronous background submission (returns the job ID immediately without blocking the terminal), then track progress / download / cancel
colabllm openjutsu swap --video input.mp4 -p character.png --steps 4 --clip-seconds 7.5
colabllm openjutsu job JOB_ID --wait -o swap_result.mp4
colabllm openjutsu download JOB_ID -o swap_result.mp4
colabllm openjutsu cancel JOB_ID

# 5. Local video editing (change: swap outfits/props, remove: erase objects, background: swap background, add: add objects)
colabllm openjutsu object --video input.mp4 --edit-mode change --target jacket --change-to "black leather jacket" --no-keep-outline -o jacket_edit.mp4
colabllm openjutsu object --video input.mp4 --edit-mode remove --target microphone -o clean.mp4
colabllm openjutsu object --video input.mp4 --edit-mode background --target person -p studio_bg.png -o new_bg.mp4

# 6. Post-processing a finished video: Fix faces refinement (requires the full profile) and Fix colours detail softening + DCTL grading
colabllm openjutsu face --video swap_result.mp4 -p character.png --style mouth -o face_fixed.mp4
colabllm openjutsu color --video face_fixed.mp4 --detail -0.2 --tone-mode Reinhard -o final_graded.mp4
```

If you have put custom `.safetensors` files into `MyDrive/colab-llm/openjutsu/loras/` on Drive, stack them on top of the default DMAD acceleration LoRA with `--extra-lora drive/my_lora.safetensors:0.8` (or replace the default LoRA entirely with `--lora`).

## Comparing the two model profiles

| `--profile` | Local SSD size | Models and capabilities |
| --- | --- | --- |
| `full` (default) | ~80.7 GB (16 files) | All core capabilities + **Fix faces** (YOLOv8m face/person segmentation + SAM ViT-B) + **From picture** (FLUX.3 Image Klein INT8, Qwen3.5-9B Upsampler, Qwen3-VL-4B FP8, Flux2 VAE, SolarRenderer 4-step/control LoRA) |
| `core` | ~50.8 GB (7 files) | **Simple / Advanced / Montage motion transfer**, **Object swap local edit/removal/background swap**, **Check mask / Find people**, **Fix colours** (MiniMax H3 Ref2VA INT8 ConvRot + DMAD 4-step LoRA + Qwen3-VL 32B INT4 / 8B INT8 + SAM 3.1 Multiplex FP16 + H3 Video/Audio VAE) |

## Web workbench modes (tabs at the top of the home page `/`)

1. **Simple / Advanced**: upload a reference video and 1-4 character reference images; optionally click **Check mask** to preview the SAM 3.1 mask, then generate in chained segments of about 6 seconds that are stitched into the full video automatically.
2. **Montage (multi-person editing mode)**: optionally run **Find people** to track and label the people in each shot, then match each person to a reference photo and generate.
3. **Object edit (local replace / remove / background swap / add objects)**: text targets, placement boxes or painted keyframes drive SAM 3.1 + MiniMax H3 local repainting.
4. **From picture (requires the `full` profile)**: FLUX.3 Klein relights/blends the character photo into the first frame, then SolarRenderer or H3 generates a continuous motion video.
5. **Fix faces / Fix colours**: chunked face repaint blending on a finished video (**Fix faces** requires the `full` profile) or detail softening with Orion4D DCTL grading (**Fix colours**).

## G4 performance tuning, storage and resuming

- **VRAM residency and immediate system-memory reclaim (`--vram-mode auto|highvram|dynamic|gpu-only`, default `auto`; ComfyUI runs with `--cache-classic` by default)**: the active models of one Openjutsu job total about 48 GB (H3 20 GB + Qwen3-VL 32B/8B 23 GB + VAE 3.3 GB + SAM 3.1 1.7 GB). On GPUs with `>=76 GiB` VRAM (such as G4 96 GB), `auto` passes `--highvram` to ComfyUI so the weights stay resident in VRAM, saving about 8 seconds of DynamicVRAM paging per segment and about 50 seconds of cold-start copying; it also passes `--cache-classic` so CPU intermediate image tensors are freed right after each recursive `Context-Loop` segment, avoiding the VM cgroup OOM caused by ComfyUI 0.44's default `--cache-ram` accumulating system memory across segments for long videos or `10s` segments + Spectrum acceleration.
- **Segment checkpoints on SSD (`--chains-storage ssd|drive`, default `ssd`)**: `MiniMaxH3-Context-Loop` writes a latent checkpoint of about 450 MB `.safetensors` per segment (about 14 GB for a 3-minute video). The default `ssd` bind-mounts `output/h3_chains` onto the local NVMe SSD (`/content/openjutsu/h3_chains`) with Linux VFS `mount --bind`, which satisfies the node's path sandbox check and keeps resuming within the same VM, while avoiding blocking Google Drive FUSE writes per segment and Drive quota growth; **the final stitched video is still saved directly on Google Drive** (`output/h3_oneclick_swap/`, `h3_oneclick_edit/`, `openjutsu_color/`, `openjutsu_face/`, etc.). To resume very long jobs across different Colab VMs, pass `--chains-storage drive`.
- **Inference speed and web/CLI parameter sweet spots**: MiniMax H3 is a **100-layer 3D audio-video full-attention DiT**, and the time per step grows **quadratically, `O(S²)`**, with the total token count (`resolution pixels x segment frames`). Three things matter most for speed: (1) **resolution no higher than the source's short side** (e.g. for a `608x1076` portrait source choose `576p`, i.e. `576x1024`; upscaling to the default `736p` `736x1312` adds 48% pixels and 2.2x attention compute); (2) **set sampling steps to `4`** (the `DMAD 4-step` distillation LoRA is loaded by default, and going from the default `6` steps to `4` saves 33% of diffusion sampling time); (3) **keep the clip length in the `7s-8s` sweet spot** (too short and the fixed `39-frame` overlap overhead dominates; at `10s` a step's sequence reaches ~56K tokens and step time jumps from `~7s` to `18.6s`). With `576p + 4 steps + 7.5s`, a 1-minute video finishes in about **8-9 minutes** on G4 with warm VRAM.
- **Fast concurrent model pulls (Rust `hf_xet` + multi-connection chunked resume)**: startup enables `huggingface_hub` + `hf_xet` (`HF_XET_HIGH_PERFORMANCE=1`) in a separate environment by default and pulls Xet chunks concurrently, largest models first (LPT) (about 600 MB/s per worker measured); non-HF sources fall back to multi-connection HTTP `Range` chunks with resume. As with all runtimes, put `MyDrive/colab-llm/hf_token.txt` on Drive (or use the local `~/.cache/huggingface/token` / `HF_TOKEN` / `--hf-token`) and it is read at startup to avoid Hugging Face anonymous rate-limit warnings.
- **Fast local NVMe SSD**: `/content/ComfyUI/models` (model weights), `/content/ComfyUI/custom_nodes`, `/content/openjutsu/proxy-state` (proxy SQLite state) and the Triton/Inductor build caches all live on the VM's local SSD; build cache archives are synced to Drive to speed up later starts.
- **Google Drive persistence (`MyDrive/colab-llm/openjutsu/`)**:
  - `output/`: the final generated MP4 videos (and the `h3_chains/` segment checkpoints in `--chains-storage drive` mode).
  - `input/`, `user/`: uploaded assets and user configuration.
  - `loras/`: mounted at `ComfyUI/models/loras/drive/`; custom `.safetensors` LoRAs put here appear automatically in the model dropdowns of the Openjutsu web UI.

## Multiple instances and troubleshooting

```bash
colabllm openjutsu up -i 2
colabllm openjutsu check -i 2
colabllm openjutsu ps --json
colabllm openjutsu logs -i 2 -f
colabllm openjutsu down -i 2
```

Remote logs are `/content/ct/comfyui.log` (ComfyUI and node inference), `/content/ct/comfy_proxy.log` (API proxy), `/content/ct/service.log` (Next.js) and `/content/openjutsu/env-setup.log` (install/build log). Keep the session online while generating long videos, and run `down` only after the final MP4 has been downloaded and saved.
