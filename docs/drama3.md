# Fish Audio Drama 3: multi-reference cloning, emotion/event tags and long-form novel narration

Defaults to **G4** (96 GB VRAM, loading the `fishaudio/drama-3` BF16 39 GB weights; GPUs with less than 72 GiB VRAM or `--variant int4` switch to `fishaudio/drama-3-int4` automatically). Built on the `sglang-omni[fishaudio]==0.2.0` three-stage pipeline (`preprocessing` → `tts_engine` → `vocoder`), it outputs **44.1 kHz** mono audio with native Bearer API key authentication.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and single-sentence synthesis

```bash
colabllm drama3 up
colabllm drama3 up --gpu H100
colabllm drama3 up --gpu A100 --variant int4
colabllm drama3 status --json
colabllm drama3 voices --json
colabllm drama3 tts 'What a lovely day. [sighs] Finally some quiet time to read.' --voice alloy -o ./speech.wav
colabllm drama3 tts 'Streaming synthesis test: receive audio while it is being generated.' --voice alloy --stream -o ./stream.wav
```

After startup the reference voice `alloy` is pre-registered by default; without `--voice` and `--ref-audio` the model samples a voice freely. English square-bracket tone/paralinguistic tags (such as `[sighs]`, `[laughs]`, `[whispers]`, `[gasp]`, `[pause]`) can be embedded directly in the text to guide pauses and emotional delivery.

## Single/multi-reference voice cloning and persistent voices

```bash
colabllm drama3 tts 'Zero-shot cloning from a single local reference clip.' --ref-audio ./ref1.wav --ref-text 'Transcript of the first reference clip.' -o ./clone1.wav
colabllm drama3 tts 'Blend the timbre and delivery of several reference clips.' --ref-audio ./calm.wav --ref-text 'A calm narration reference sentence.' --ref-audio ./excited.wav --ref-text 'An excited reference sentence.' -o ./multi-ref.wav
colabllm drama3 tts 'Read the reference audio directly from Google Drive.' --ref-audio 'drive://voices/narrator.wav' -o ./drive-clone.wav
colabllm drama3 upload-voice --name narrator --audio ./narrator.wav --ref-text 'Exact verbatim transcript of the reference audio.' --description 'Main audiobook narrator'
colabllm drama3 upload-voice --name heroine --drive 'drive://voices/heroine.wav' --ref-text 'Character voice reference text.'
colabllm drama3 voices --json
colabllm drama3 delete-voice heroine
colabllm drama3 ls-drive 'voices' --json
```

`--ref-audio` can be repeated to build a multi-reference `references` cloning list; `--ref-text` values map one-to-one in the same order (accurate transcripts improve onset stability and prosody). Named voices registered with `upload-voice` are stored on the local SSD as `speakers/*.safetensors` and synced automatically to `MyDrive/colab-llm/drama3/voices/`, so they are restored after an instance restart and the reference audio does not have to be uploaded and encoded again for every request.

## Audiobook generation for novels and long texts (`narrate`)

Drama 3 generates at most **2048 frames (about 95.1 seconds of audio)** per request; putting a whole chapter into a single request is truncated at around 95 seconds. Always use the built-in `narrate` command for long novels:

```bash
colabllm drama3 narrate ./novel.txt --voice narrator -O ./novel-audiobook
colabllm drama3 narrate ./novel.txt --voice narrator --chapters '1-5' --batch-size 8 --max-chars 180 --format mp3 -O ./novel-mp3
colabllm drama3 narrate ./novel.md --voice narrator --cast ./cast.json --turn-pause-ms 180 --pause-ms 280 --para-pause-ms 650 -O ./novel-cast
colabllm drama3 narrate ./novel.txt --cast ./cast.json --llm-service qwen-flash --llm-tags -O ./novel-drama
```

`narrate` runs the whole book production pipeline automatically:

1. **Automatic chapter detection and sentence splitting**: recognizes common Chinese and English chapter headings (Chinese `第X章/节/回/卷`, `楔子` prologue, `尾声` epilogue, `Chapter N`, Markdown `#` headings; `--single-chapter` treats the whole text as one chapter) and greedily packs sentences within a paragraph at full stops/question marks/exclamation marks/semicolons into natural chunks of at most `--max-chars` (default `180` characters, about 35–45 seconds of speech, far below the 95-second hard limit).
2. **Automatic multi-character dialogue splitting and speaker attribution**: with `--cast` (or explicitly `--split-dialogue`), narration and quoted dialogue mixed in the same paragraph are split into separate narrator lines and character lines, while inline emphasis quotes (scare quotes) are not split by mistake. Supports leading and trailing speech attributions, interrupted quotes by the same speaker, character alias matching and automatic alternation of consecutive unattributed two-person dialogue; the tighter `--turn-pause-ms` (default `180 ms`) is used where narration turns into dialogue. The attribution heuristics are tuned for Chinese prose (`“…”`, `「…」`, `『…』` quotes and Chinese speech verbs); plain `"…"` quotes are also recognized.
3. **Optional LLM script pre-annotation (`--llm-service` / `--llm-url`)**: for complex passages with omitted names, crowded scenes or many pronouns, point to a running `colabllm` text model service (such as `--llm-service qwen-flash` or `--llm-service deepseek-coder`) or any OpenAI-compatible API (`--llm-url` + `--llm-token` / `--llm-model`). Before synthesis the text is converted window by window into a `[role:旁白]` (narrator) / `[role:character name]` script and cached at `<output-dir>/annotated/<chapter>.txt`. A built-in verbatim skeleton check guarantees no characters are dropped or changed (windows that fail the check fall back to rule-based splitting); `--llm-tags` also lets the model sparingly insert `[sighs]`, `[laughs]` and `[gasp]` where emotions are obvious.
4. **Dynamic frame limit and trailing-silence trimming**: `max_new_tokens` is estimated from each chunk's length, speaking rate, punctuation and tone tags, and every 44.1 kHz WAV chunk is checked locally for abnormal trailing silence and trimmed (`--no-trim-silence` disables this), avoiding the occasional long silent tail of autoregressive generation.
5. **High-throughput batched concurrency**: by default calls the server's `POST /v1/audio/speech/batch` with `--batch-size 8` (chunks in the same batch can each carry their own character `voice`, `speed` or `ref_audio`; with 8-way concurrency on a G4 GPU the measured combined throughput is about **22x real time**, so a 100,000-character novel finishes in roughly 15–20 minutes); single failures are retried automatically with degradation.
6. **Resume and chapter concatenation**: each chunk is first written to `<output-dir>/chunks/<chapter>/chunk_XXXX.wav` and `<output-dir>/manifest.json` is updated (including per-chapter voice chunk statistics `voice_chunks`); after a network interruption or a pause, rerun the same command to continue from the checkpoint, and finished chapters and chunks are skipped automatically (`--force` reruns everything; `--clean-chunks` deletes chunk files after each chapter is concatenated). By default a `280 ms` pause is inserted between sentences, `180 ms` between speaker turns and `650 ms` between paragraphs.

The character voice map `--cast cast.json` accepts plain voice names as well as per-character `voice`, `aliases` (aliases/nicknames), `speed` or zero-shot `ref_audio`/`ref_text` (you can also embed `[role:Lin Wan]` anywhere in the text to set the speaker explicitly):

```json
{
  "narrator": "narrator",
  "dialogue": "heroine",
  "characters": {
    "Lin Wan": {
      "voice": "heroine",
      "aliases": ["Wan", "Miss Lin"],
      "speed": 1.02
    },
    "Chen Mo": {
      "voice": "male_lead",
      "aliases": ["Old Chen"],
      "speed": 0.96
    }
  }
}
```

## Model limits, output and startup parameters

- **Single-speaker limit**: the current Drama 3 implementation in `sglang-omni` only supports single-speaker generation (`<|speaker:0|>`) and does not support alternating multi-speaker tags `<|speaker:1|>`…`<|speaker:15|>` within one request (the server returns HTTP 400); the CLI strips unsupported `<|speaker:1..|>` tags automatically. For multi-character audiobooks switch voices per segment (`--cast`).
- **Unsupported generic TTS fields**: the Drama 3 pipeline does not accept `seed`, `language`, `instructions` or a `repetition_penalty` other than `1.0`; the built-in middleware strips `seed/language/instructions` sent by generic clients and maps model aliases (`drama3`, `tts-1`, etc.) to the loaded model.
- **Output formats and Drive persistence**: `tts --format` supports `wav`, `mp3`, `flac`, `pcm`, `aac` and `opus`; `narrate --format` supports `wav` (default, concatenated with the standard library only, no extra dependencies) as well as `mp3`, `flac` and `opus` (transcoded with the local `ffmpeg` after concatenation). `tts --save-drive` also writes the output audio to `MyDrive/colab-llm/drama3/outputs/`.
- **Connection and startup overrides**: `--url` / `DRAMA3_URL` and `--api-token` / `DRAMA3_API_TOKEN` override the target service and credentials. `up` supports `--variant auto|bf16|int4`, `--model`, `--alias`, `--engine-gpu-mem`, `--vocoder-gpu-mem`, `--max-running-reqs`, `--context-length`, `--batch-max-items` and `--max-ref-sec`, which only apply to the next startup.

```bash
colabllm drama3 up -i 2
colabllm drama3 tts 'Connectivity check for the second instance.' --voice alloy -i 2 -o ./instance2.wav
colabllm drama3 down -i 2
```

## Startup and compilation cache

Model weights (`/content/drama3/models/`) and the S2-Pro codec (`/content/drama3/codec/codec.pth`) are always downloaded to Colab's local NVMe SSD (`HF_XET_CHUNK_CACHE_SIZE_BYTES=0`), avoiding the FUSE Drive small-file read bottleneck and temporary caches filling the disk. FlashInfer, Triton and TorchInductor compilation caches also live on the local SSD and are archived to Drive per environment and GPU architecture after warm-up:

```text
MyDrive/colab-llm/drama3/
  voices/
  outputs/
  cache-archives/sglomni-0.2.0-py312-bf16-sm120/
    flashinfer-cache.tar
    triton-cache.tar
    inductor-cache.tar
```
