# Qwen3-TTS: synthesis, voice design and cloning

Defaults to **L4** with native Bearer authentication; supports CustomVoice, VoiceDesign, Base cloning and persistent voice registration.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and built-in voices

```bash
colabllm qwen3-tts up
colabllm qwen3-tts status --json
colabllm qwen3-tts voices --json
colabllm qwen3-tts tts --help
```

The models needed for all tasks are preloaded by default, so the first install, download and compilation take a while. Built-in speakers are `vivian`, `serena`, `uncle_fu`,
`dylan`, `eric`, `ryan`, `aiden`, `ono_anna` and `sohee`; query the actual list of this instance with `voices`.

## Plain synthesis and style control

```bash
colabllm qwen3-tts tts 'Hello, welcome to the speech service.' --voice serena --lang English -o ./speech.wav
colabllm qwen3-tts tts --text-file ./script.txt --voice vivian \
  --instruct 'Steady pace, like telling a story.' --speed 1.1 --save-drive -o ./narration.wav
```

Text can be a positional argument, `--text` or a UTF-8 `--text-file`; pick just one.
`-o` is a local file; when omitted a timestamped WAV file name is created in the current directory.
`--format` supports `wav/mp3/flac/opus/pcm`; an explicit format should come with the matching output extension.
Speed ranges from 0.25 to 4.0; `--max-new-tokens` limits the amount generated and is not a text character count.

## One-off design and reusable voices

```bash
# Generate from a description for this request only; no name is registered by default
colabllm qwen3-tts tts "Welcome to today's story." --voice-design 'a warm, clear adult female voice' -o ./designed.wav

# Design a reference audio and register it, then reuse it with --voice
colabllm qwen3-tts design-voice --instruct 'a calm, natural male narrator voice' \
  --text 'This is the reference sentence for the narrator voice.' --lang English --name narrator -o ./narrator-ref.wav
colabllm qwen3-tts tts 'This is another piece of narration.' --voice narrator -o ./reuse.wav
```

`tts --voice-design` is VoiceDesign for the current text; `design-voice --name` is the design-then-register-then-clone workflow.
Named operations change persistent voices; check `voices` first to avoid overwriting an existing name.

## Cloning and registering from reference audio

```bash
colabllm qwen3-tts tts 'Say this sentence in the reference voice.' --ref-audio ./reference.wav \
  --ref-text 'The words actually spoken in the reference audio.' --lang English -o ./clone.wav
colabllm qwen3-tts upload-voice --name myvoice --audio ./reference.wav --ref-text 'The words actually spoken in the reference audio.'
colabllm qwen3-tts upload-voice --name drivevoice --drive 'drive://voices/reference.wav'
colabllm qwen3-tts tts 'Keep using the same voice next time.' --voice myvoice -o ./registered.wav
colabllm qwen3-tts ls-drive 'voices' --json
```

Only use audio you have permission to use. `--ref-audio` selects the Base cloning task; without a transcript the speaker-embedding mode is used,
which `--x-vector-only` selects explicitly; a full reference transcript enables ICL. Reference audio can be local, `drive://` or a URL supported by the service.
`--ref-drive` makes the argument be interpreted as a Drive path.
Delete a registered voice with `delete-voice NAME`, only for names the user explicitly wants deleted.

## Parameters, storage and failure recovery

`up --variant`, `--model`, `--preload-all`, `--stage0-gpu-mem`, `--stage1-gpu-mem`, `--max-model-len` and
`--max-seqs` are parameters for the next start, not hot-switch commands for the current `tts` request. CustomVoice/VoiceDesign/Base tasks must match the capabilities of the loaded models.
Model API entry points include `POST /v1/audio/speech`, `GET /v1/audio/voices` and the voice management endpoints.

`--save-drive` also saves to `MyDrive/colab-llm/qwen3-tts/outputs/`, or to a given directory; the service response/diagnostics give the actual path.
Registered voices and metadata live in the service's Drive `voices/`; model weights are on the local SSD, and build caches are archived to Drive per GPU/environment.
Caches only save compilation and transfer time; they do not make a new VM skip installation or loading.
`--timeout` defaults to 300 seconds and is a request limit, not a resumable job ID. After a timeout check for results first, to avoid blindly generating again.

## Multiple instances and management

```bash
colabllm qwen3-tts up -i 2
colabllm qwen3-tts tts 'Testing the second instance.' --voice serena -i 2 -o ./instance2.wav
colabllm qwen3-tts logs -i 2 -n 100
colabllm qwen3-tts down -i 2
```

For calls across machines use `--url` with `QWEN3_TTS_API_TOKEN`, or `QWEN3_TTS_URL` / `--api-token`; when changing origin, pass the token explicitly.

## Startup cache optimization

Qwen3-TTS and IndexTTS share the archive cache implementation, but each keeps its cache data in its own service directory. For L4:

```text
MyDrive/colab-llm/qwen3-tts/
  cache-archives/vllm-0.32.0-py312-sm89/
    flashinfer-cache.tar
    triton-cache.tar
    vllm-cache.tar
```

At startup a single archive is read first and then unpacked on the local SSD; inference compilation still uses the local SSD. After warm-up new cache entries are repacked, and the old archive is replaced only after the temporary file uploads successfully. Unchanged content is not uploaded again, and an archive already restored on the same VM is not read again.

The first time a legacy Drive cache is encountered it is still migrated file by file, after which archives are generated automatically. Restore, migration and save progress is printed every 10 seconds. A corrupt archive falls back to the old directory; cache restore or save failures produce warnings, and the service keeps using the local cache or compiles on demand.

Different vLLM versions and GPU architectures use separate directories. For concurrent instances on the same architecture, the last successfully uploaded archive wins, and missing cache entries can be regenerated. Registered voices and generated audio stay ordinary Drive files.

This optimization reduces cache transfer time; it does not cache the Python virtual environments or model weights. A new VM still installs the vLLM environment and the auxiliary Qwen3-TTS environment, downloads the models and loads/warms them up. The new logic takes effect on the next `up`.
