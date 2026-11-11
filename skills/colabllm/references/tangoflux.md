<!-- Generated from docs/tangoflux.md; edit the source and run tools/sync_runtime_docs.py. -->

# TangoFlux: 44.1 kHz stereo sound effects and text-to-audio

Defaults to **T4** (16 GB VRAM is enough; the model uses about 4.3 GB of VRAM in total; pass `--gpu L4` or `--gpu G4` for faster generation). By default it loads the CRPO preference-aligned `declare-lab/TangoFlux` (515M FluxTransformer + `google/flan-t5-large` text encoder + `AutoencoderOobleck` stereo VAE; `--variant base` switches to `declare-lab/TangoFlux-base`) and generates **1–30 second, 44.1 kHz two-channel stereo** sound effects (SFX), ambiences, Foley and background music clips, with native Bearer API key authentication.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and single sound effect generation

```bash
colabllm tangoflux up
colabllm tangoflux up --gpu L4
colabllm tangoflux up --variant base
colabllm tangoflux status --json
colabllm tangoflux models --json
colabllm tangoflux generate 'Hammer slowly hitting the wooden table' -d 10 -s 50 -o ./hammer.wav
colabllm tangoflux sfx 'Heavy rain and distant thunder against a cabin window' -d 15 --guidance-scale 4.5 --seed 42 -o ./rain.wav
colabllm tangoflux generate 'Futuristic spacecraft warp engine powering up' -d 8 -n 3 --format mp3 --save-drive -o ./warp.mp3
colabllm tangoflux ls-drive --json
```

`generate` and `sfx` are fully equivalent:

- **Prompt advice**: the underlying text encoder is `google/flan-t5-large` and the training data is mostly English acoustic descriptions, so use specific, layered English acoustic descriptions (source material, rhythm of the action, room reverb and distance: `'Crisp footsteps walking slowly on dry autumn leaves in a quiet forest'`).
- **Core sampling parameters**: `-d` / `--duration` sets the audio length (`1.0`–`30.0` seconds, default `10`); `-s` / `--steps` sets the Rectified Flow matching steps (default `50`, use `25` for speed); `-g` / `--guidance-scale` sets the classifier-free guidance strength (default `4.5`); `--seed` fixes the random seed for exact reproduction.
- **Several candidates at once (`-n`)**: `-n 2`…`4` samples several variants in parallel within one request and saves them locally as `<stem>_01.<ext>`, `<stem>_02.<ext>`.
- **Output format and Drive saving**: `--format` supports `wav` (default, 44.1 kHz 16-bit stereo PCM), `mp3`, `flac`, `ogg`, `opus` and `aac`. `--save-drive` (or `--save-drive 'sfx/rain.wav'`) also writes the returned audio to `MyDrive/colab-llm/tangoflux/outputs/`.

## Batch sound pack generation with resume (`batch`)

To build a complete sound pack for audiobook post-production, video editing or game development, put the cues in a `.txt` (one prompt per line, or `name | prompt`), `.json` or `.jsonl` manifest and generate them with `batch`:

```bash
colabllm tangoflux batch ./cues.txt -d 8 -s 50 -O ./audiobook-sfx
colabllm tangoflux batch ./cues.json --batch-size 4 --format mp3 --save-drive -O ./game-sfx
colabllm tangoflux batch ./cues.json -O ./game-sfx --force --json
```

`cues.json` example (`duration` / `steps` / `guidance_scale` not set on an entry inherit the command-line defaults):

```json
[
  {
    "name": "teahouse_fire",
    "prompt": "Warm crackling charcoal fire in a quiet wooden room",
    "duration": 15,
    "steps": 50,
    "seed": 101
  },
  {
    "name": "wooden_door_snow",
    "prompt": "Creaking old wooden door opening with howling winter wind outside",
    "duration": 6
  }
]
```

`<output-dir>/manifest.json` is updated after every batch; rerunning the same command after an interruption skips finished entries automatically (`--force` reruns them).

## HTTP API compatibility

Every endpoint requires `Authorization: Bearer <API_TOKEN>` (or `X-API-Key`):

- `POST /api/ct/generate`, `POST /v1/audio/generations`, `POST /v1/audio/speech`: accept JSON `{"prompt": "...", "duration": 10, "steps": 50, "guidance_scale": 4.5, "seed": 42, "num_waveforms": 1, "response_format": "wav", "save_to_drive": "auto"}` (for compatibility with generic OpenAI audio clients, `prompt` may also be sent as `input`). With `num_waveforms=1` the binary audio stream is returned directly, with metadata in the `X-Audio-Seed`, `X-Audio-Duration` and `X-Saved-Drive-Path` response headers; `"return_json": true` or `num_waveforms > 1` returns Base64 JSON.
- `POST /api/ct/batch`: accepts `{"items": [...], "response_format": "wav"}` and returns each sound effect's Base64 data and seed in order.
- `GET /v1/models`, `GET /api/ct/info`, `GET /api/ct/drive-list`, `GET /health`.

## License notes, connection overrides and local SSD cache

- **Model license restrictions**: upstream `declare-lab/TangoFlux` is bound by the **Stability AI Community License** and the **WavCaps** dataset terms and is limited to non-commercial and academic research use (commercial use requires a separate license from Stability AI).
- **Connection and startup overrides**: `--url` / `TANGOFLUX_URL` and `--api-token` / `TANGOFLUX_API_TOKEN` override the target service and credentials; `up` supports `--variant crpo|base`, `--model` and `--alias`, and multiple instances use `-i ID`.

```bash
colabllm tangoflux up -i 2
colabllm tangoflux sfx 'Soft ocean waves on pebbles' -d 5 -i 2 -o ./waves.wav
colabllm tangoflux down -i 2
```

Model weights (`/content/tangoflux/models/`; only `safetensors` weights are pulled, skipping the redundant Flax/TF copies of `flan-t5-large` to save about 9.7 GB of downloads) and the Triton/Inductor compilation caches live on Colab's local NVMe SSD and are archived to Drive after warm-up:

```text
MyDrive/colab-llm/tangoflux/
  outputs/
  cache-archives/tangoflux-diffusers0303-py312-sm75/
    triton-cache.tar
    inductor-cache.tar
```
