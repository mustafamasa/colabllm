<!-- Generated from docs/indextts.md; edit the source and run tools/sync_runtime_docs.py. -->

# IndexTTS-2.5: cloning, emotion and speed control

Defaults to **L4**; two-stage vLLM-Omni speech synthesis with native Bearer authentication; supports reference voices, registered voices, multiple languages and emotion control.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and basic synthesis

```bash
colabllm indextts up
colabllm indextts status --json
colabllm indextts voices --json
colabllm indextts tts 'Hello, this is a speech synthesis test.' -o ./speech.wav
colabllm indextts tts --text-file ./script.txt --voice default --lang zh -o ./narration.wav
```

Without `--voice/--ref-audio` the service's default voice `default` is used. `--ref-audio` takes precedence over a voice name.
Text can be given as a positional argument, with `--text` or as a UTF-8 text file. Keep long texts and the generated token limit under control.

## Cloning and persistent voices

```bash
colabllm indextts tts 'Say this sentence in the reference voice.' --ref-audio ./reference.wav -o ./clone.wav
colabllm indextts tts 'Use the reference audio on Drive directly.' --ref-audio 'drive://voices/ref.wav' -o ./drive-clone.wav
colabllm indextts upload-voice --name narrator --audio ./reference.wav --description 'Narrator reference with permission to use'
colabllm indextts upload-voice --name drivevoice --drive 'drive://voices/ref.wav'
colabllm indextts voices --json
colabllm indextts tts 'Continue the narration.' --voice narrator -o ./registered.wav
colabllm indextts ls-drive 'voices' --json
```

Check the name before registering; delete with `delete-voice NAME`. Local reference audio is uploaded; Drive inputs are read remotely.
Reference audio can be a URL supported by the service, or forced to be interpreted as a Drive path with `--ref-drive`.
Registering and deleting voices are persistent changes, not temporary generation parameters.

## Emotion, language and speed

```bash
colabllm indextts tts 'What a wonderful surprise!' --voice narrator --emo-preset happy --emo-alpha 0.8 -o ./happy.wav
colabllm indextts tts 'The story is about to begin.' --voice narrator --emo-text 'gentle and slightly expectant' -o ./emotion.wav
colabllm indextts tts 'Please listen to me slowly.' --voice narrator --speed 0.8 --lang en -o ./slow.wav
```

The native speed range is 0.5-2.0; languages are `zh/en/zhen/ja/es/ar/yue`, and when omitted the CLI picks one automatically from the Chinese/Latin characters in the text.
`--emo-preset` accepts happy, angry, sad, afraid, disgusted, melancholic, surprised and calm.
`--emo-vector` uses the eight dimensions above in that order, and `--emo-alpha` controls blending (0-1); when an emotion is given without a blend value the default is 0.6.
`--emo-audio` / `--emo-audio-drive` set an emotion reference. Pick one main emotion source at a time to avoid unclear precedence between mixed parameters.
`--no-text-norm` disables text normalization, `--use-random` enables random sampling, and `--seed` helps reproduce experiments.

## Output, authentication and startup configuration

`-o` is a local file; the default output is a timestamped WAV in the current directory. `--format` supports WAV, MP3, FLAC, Opus and PCM;
make sure the file extension matches the format. `--save-drive` also saves the server-side output; the returned path takes precedence over directory examples in older help text.
The current notebook persists voices/outputs in `MyDrive/colab-llm/indextts/`; weights live on the SSD and build caches are archived to Drive per GPU and environment.
`--timeout` defaults to 300 seconds and is an HTTP request limit; TTS does not return persistent job IDs like OCR/Avatar.

Select the service with `--url` / `INDEXTTS_URL` and pass explicit credentials with `--api-token` / `INDEXTTS_API_TOKEN`.
The main APIs are `POST /v1/audio/speech`, `GET /v1/audio/voices` and the voice register/delete endpoints.
`up --model/--alias/--stage0-gpu-mem/--stage1-gpu-mem/--max-model-len/--max-seqs` configure the next start.
Check `up --help` first, and do not mistake them for reconfiguring a running model.

```bash
colabllm indextts up -i 2
colabllm indextts tts 'Testing the second instance.' --voice default -i 2 -o ./instance2.wav
colabllm indextts logs -i 2 -n 100
colabllm indextts down -i 2
```

## Startup and build cache

Setup restores the caches and registered voices first, then prepares the Python environment and model in parallel. After the service starts it loads the two-stage model and warms up; JIT caches are reused when they hit, otherwise generated on demand.

Build caches always run on the local SSD; Drive only keeps the archives. For L4:

```text
MyDrive/colab-llm/indextts/
  cache-archives/vllm-0.32.0-py312-sm89/
    flashinfer-cache.tar
    triton-cache.tar
    vllm-cache.tar
```

- At startup each cache type reads a single tar file from Drive and unpacks it locally, reducing per-file Drive operations.
- Legacy `flashinfer-cache/`, `triton-cache/` and `vllm-cache/` still need a one-time migration, after which archives are saved automatically; the first migration may still be slow. The old directories are kept for fallback.
- Restore, migration and save progress is printed every 10 seconds; archives already restored on the same VM are not read again, and unchanged content is not uploaded again.
- After warm-up the cache is packed locally, uploaded as a temporary file and swapped in for the real archive; a failed upload keeps the old archive. Cache errors produce warnings and the service keeps using the local cache or regenerates it.
- Different vLLM versions and GPU architectures use different directories. For concurrent instances on the same architecture, the last successfully saved archive wins; missing entries can be recompiled.
- Registered voices and generated audio stay ordinary Drive files and are not put into the build cache archives.

Archiving only shortens build cache restore/save time; it does not remove dependency installation, model download, model loading or the necessary compile warm-up on a new VM. A running older notebook is not hot-swapped; the new restore logic takes effect on the next `up`.

Verification record: in a real L4 Colab session, 577 Triton files (about 54 MB of data) were saved as a ~55 MB tar, then read back from Drive into a separate local directory and checked file by file with SHA-256; all matched. Reading, unpacking and checking took about 0.65 s; the test ran in the same session right after the archive was written, so it is a warm-cache check and does not represent cold-start time on a new VM. The FlashInfer and vLLM archives were saved as well.
