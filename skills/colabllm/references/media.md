# Media runtimes

- [LongCat Avatar](longcat-avatar.md): persistent generation jobs, Drive inputs and video download.
- [LongCat Video](longcat-video.md): Gradio text/image generation and continuation.
- [Openjutsu](openjutsu.md): MiniMax H3 motion transfer (`swap`), object edit (`object`), mask check (`mask`), face/color post-processing (`face`, `color`), job tracking/download (`job`, `download`, `cancel`) and web UI.
- [FLUX 3 Image Klein](flux-klein.md): G4 (default) / H100 / L4 vLLM-Omni text-to-image (`generate`), 1-10 reference image editing (`edit`), optional Qwen3.5-9B prompt upsampling (`upsample`), and direct Google Drive input/output (`ls-drive`, `--save-drive`).
- [Qwen-Image-2.1](qwen-image.md): G4 (default) / H100 / A100 vLLM-Omni bilingual text-to-image (`generate`), 1-4 `<image1>..<image4>` reference image editing (`edit`), native 4-channel RGBA transparency (`--rgba`), and direct Google Drive input/output (`ls-drive`, `--save-drive`).
- [MiniMax-H3-Image](minimax-h3-image.md): G4 (default) / H100 / A100 SGLang Diffusion 6-step PDD native 4K text-to-image (`generate`), 1-16 `<Picture 1>..<Picture 16>` reference image editing with SHA-256 deduplicated binary upload (`edit`), async job tracking & resumable `Range` WebP/PNG download (`job`, `download`), and zero-transfer Google Drive input/output (`ls-drive`, `--drive-image`, `--save-drive`, `--drive-only`).
- [TangoFlux](tangoflux.md): T4 (default) / L4 / G4 44.1 kHz stereo text-to-audio & sound effect generation (`generate`/`sfx`, 1–30s duration, multi-variation `-n`), resumable batch SFX cue sheets (`batch`), and Drive output saving (`ls-drive`, `--save-drive`).
- [ComfyUI](comfyui.md): workflow/model download, queue submission and outputs.

Read the selected runtime guide. Pulling models does not generate media; LongCat Avatar,
Openjutsu and MiniMax-H3-Image provide full CLI job submission, progress tracking and output download workflows.
