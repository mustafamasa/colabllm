# LongCat Video: text to video, image to video and continuation

Defaults to **G4** with Caddy Bearer/Basic Auth; provides a Gradio web UI and its public generation API. It is different from the separate Avatar model.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and login

```bash
colabllm longcat-video up --ready-timeout 3600
colabllm longcat-video status --json
colabllm longcat-video info --json
colabllm longcat-video info --show-secrets
colabllm longcat-video logs -n 100
```

Open the web UI from the `info` URL and log in with `auth_user/auth_pass`; the API uses `api_token`.
Before it is ready, model loading and a small inference warm-up must finish; preparing roughly 80 GB of weights for the first time can take a while.

## Web generation workflow

1. **Text to Video**: choose this mode, describe the shots, characters, actions and scene, set the size and seed, then generate.
2. **Image to Video**: upload a local reference image, choose the mode and 480p/720p, and describe the desired motion.
3. **Video Continuation**: upload a reference video and choose the mode and resolution. The reference video needs at least 13 frames after sampling at 15 fps.
4. After generation, review the result video, download control and parameter JSON; save the MP4 and metadata, and never deliver only a temporary web link.

The default fast mode enables the CFG/step LoRA with a fixed 16 steps; standard mode defaults to 50 steps and CFG 4.
Each segment defaults to 93 frames at 15 fps, about 6.2 seconds. Frames are `4n+1` between 17 and 161, and width/height are multiples of 16 between 256 and 1024.
Consecutive segments are 1-10; later segments reuse the last 13 frames and remove the overlap. Image-to-video, continuation and multi-segment runs use resolution size buckets, which may differ from manually entered sizes.
Seed -1 is random; higher resolutions and more frames/segments increase GPU use, so check with a short preview first.

## API and output

The app's Gradio API is named `generate`; read **the current version's parameter order and types** from the web UI's API page before calling it.
Do not mistake Avatar's `colabllm longcat-avatar generate` for a command of this runtime; this runtime only has the shared lifecycle CLI.
Programmatic calls must attach Caddy Bearer or Basic Auth on the client and wait according to the Gradio queue protocol.
The API and UI share a serial generation queue; a successful submission does not mean generation is finished.

The generated MP4 and a JSON recording the prompt, parameters and seed are written to
`MyDrive/colab-llm/longcat-video/output/`; inputs uploaded through the web UI may only be temporary files, so keep the original assets separately.
Model weights live on the VM SSD at `/content/longcat-video/model`; caches can be persisted, but after the VM stops parts of the environment must be reinstalled/downloaded/loaded again.
Leaving the web page does not cancel GPU generation, and is no reason to stop the VM early. After downloading, verify that the file exists and plays.

## Multiple instances and troubleshooting

```bash
colabllm longcat-video up -i 2
colabllm longcat-video info -i 2 --json
colabllm longcat-video ps --json
colabllm longcat-video logs -i 2 -f
colabllm longcat-video down -i 2
```

Remote logs include `/content/ct/service.log` and the install/download logs in the working directory.
A successful `/health` means the base model and a small inference have been verified; it does not guarantee that every long-video/high-resolution parameter set will succeed.
For out-of-memory errors, reduce resolution, frames or segments first, and do not restart repeatedly while a job is running.
