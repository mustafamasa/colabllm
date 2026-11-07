<!-- Generated from docs/nemotron-transcribe.md; edit the source and run tools/sync_runtime_docs.py. -->

# Nemotron Transcribe: audio transcription, subtitles and batch processing

Defaults to **G4** with native Bearer authentication; supports local/Drive audio, media URLs, batch lists, speakers and timestamps.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and locating data

```bash
colabllm nemotron-transcribe up
colabllm nemotron-transcribe status --json
colabllm nemotron-transcribe ls-drive 'audio' --json
colabllm nemotron-transcribe prompts
colabllm nemotron-transcribe transcribe --help
```

The BF16 model is used by default; `up --precision fp8` or `nvfp4` selects the matching model variant, `--repo` overrides the repository and `--mtp` overrides MTP.
These parameters only affect startup and never hot-swap a running model. The first download and compilation may take a while.
`ls-drive` queries Drive directly through Colab without public HTTP; it lists only the current level by default, add `-r` to recurse.

## Local files, Drive and subtitles

```bash
colabllm nemotron-transcribe transcribe ./meeting.wav --local --preset verbatim --json -O ./transcripts
colabllm nemotron-transcribe transcribe 'drive://audio/interview.wav' --drive \
  --preset speaker --timestamps --save-drive 'audio/transcripts' --json -O ./transcripts
colabllm nemotron-transcribe transcribe ./english.wav --local --preset translate --lang Chinese -o ./translation.txt
colabllm nemotron-transcribe transcribe 'audio/lecture.wav' --drive \
  --preset context --context 'project name, proper nouns, people names' --json -O ./transcripts
```

`--local` / `--drive` remove ambiguity about input paths; do not use both. `drive://` is a MyDrive path.
`--preset speaker` enables timestamps by default; `--timestamps` works for other tasks. `--preset clean` removes filler words and `verbatim` keeps the exact wording.
A custom `--prompt` / `--prompt-file` can be used for Q&A or summaries of a recording; consider `--reasoning` only for tasks that need reasoning.
Do not add Q&A instructions to plain transcription by accident. `prompts` lists all presets of this version, including translation, language identification, speaker and context hints.

## Directories, task lists and multiple instances

```bash
# Directories cover only the current level by default; recursion needs -r. Quote globs so the remote side expands them
colabllm nemotron-transcribe transcribe 'meetings/*.mp3' --drive --preset clean --json -O ./transcripts
colabllm nemotron-transcribe transcribe 'meetings' --drive -r --preset verbatim --json -O ./transcripts
colabllm nemotron-transcribe transcribe --batch-file ./tasks.jsonl --json -O ./transcripts -j 2
```

`tasks.jsonl` has one task per line, for example:

```jsonl
{"audio":"meetings/call.wav","preset":"speaker","timestamps":true}
{"audio":"meetings/lecture.wav","preset":"context","context":"project name, proper nouns"}
```

JSON, CSV, TSV and TXT lists are also accepted; whether an input is local or on Drive is decided by the field contents and task parameters.
`-j` sets concurrency against one service, but GPU memory is still limited. With several existing instances, `--instances 1,2` or `--all-instances` spreads the requests.
They never create more VMs. `-i 2` selects a single instance.

```bash
colabllm nemotron-transcribe up -i 2
colabllm nemotron-transcribe transcribe ./meeting.wav --local -i 2 --json -O ./instance2
colabllm nemotron-transcribe ps --json
colabllm nemotron-transcribe down -i 2
```

## Media URLs and long audio

```bash
colabllm nemotron-transcribe download 'https://www.youtube.com/watch?v=VIDEO_ID' --download-dir 'audio/media' --json
colabllm nemotron-transcribe transcribe 'https://www.youtube.com/watch?v=VIDEO_ID' \
  --preset verbatim --download-dir 'audio/media' --json -O ./transcripts
```

`VIDEO_ID` is a placeholder. `--download-video` saves the video and `--playlist` processes playlists.
For sources that require login use `--cookies-drive`; force local yt-dlp with `--local-ytdlp`, optionally with `--cookies` or `--cookies-from-browser`.
The local download path needs yt-dlp/FFmpeg on the local machine; cookies and media access credentials must never be written into shared logs.
Cached downloads are usually reused; `--force-download` forces a fresh download.

By default audio longer than 40 minutes is split at silences and the text/timestamps are merged; `--chunk-minutes 0` disables this.
`--normalize-audio` asks the remote side to normalize; chunking does not mean the client can resume automatically from an arbitrary position.

## Output and failure semantics

`-O` saves each file's `.txt/.nemotron.json/.srt` (when subtitles exist) and `batch_summary.jsonl` locally;
`-o` sets a single text, SRT, JSON or batch summary file. `--save-drive` without a value saves next to the source audio, and with a value sets a Drive directory.
With `--json`, stdout contains only JSON and progress goes to stderr; a batch may partially succeed, so check `errors` and the exit code before declaring it done.

`--timeout` defaults to 3600 seconds and limits each request/audio task, not the whole batch.
This command is not a persistent job API like OCR/Avatar; do not assume `job/resume` can recover after a network drop. Save completed results and retry the failed items.
`NEMOTRON_URL` / `--url` and `NEMOTRON_API_TOKEN` / `--api-token` support remote calls; when changing origin, pass the token explicitly.
