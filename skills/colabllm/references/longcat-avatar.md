<!-- Generated from docs/longcat-avatar.md; edit the source and run tools/sync_runtime_docs.py. -->

# LongCat Avatar: audio-driven person video

Defaults to **G4** with Caddy Bearer/Basic Auth; a separate Avatar 1.5 model supporting local or Drive audio/images and persistent background jobs.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and local generation

```bash
colabllm longcat-avatar up
colabllm longcat-avatar status --json
colabllm longcat-avatar generate --audio ./speech.wav --image ./person.png \
  --prompt 'A person speaking naturally in a bright room.' -o ./avatar.mp4
```

Give the description with either `--prompt` or a UTF-8 `--prompt-file`. Audio is required; the image is optional (then the Audio + Prompt mode is used).
Local assets are uploaded to the server; local paths are never handed to the model directly. `-o` downloads to a local file and makes generate/job wait for completion.
Generation has content, VRAM and time limits; confirm composition and motion with a short clip first.

## Drive inputs and asynchronous jobs

```bash
colabllm longcat-avatar ls-drive 'materials'
colabllm longcat-avatar generate --audio-drive 'materials/speech.wav' --image-drive 'materials/person.png' \
  --prompt-file ./prompt.txt --drive-output 'videos/avatar'
colabllm longcat-avatar job JOB_ID
colabllm longcat-avatar job JOB_ID --wait --timeout 600 -o ./avatar.mp4
colabllm longcat-avatar download JOB_ID -o ./avatar.mp4
```

Without `-o/--wait`, generate returns a `job_id` in JSON, which does not mean the video is finished.
Drive inputs can be written as `drive://`, `gdrive://` or a full remote path; `--audio-drive/--image-drive` also accept MyDrive-relative paths.
`--drive-output` (`--output-drive`) is a remote output directory, different from the local download of `-o`; both can be used together.

Jobs keep a snapshot of their inputs, and the returned `result.path` / metadata give the actual Drive file. Videos go to the service's Drive `output/` by default.
`--timeout 0` waits without limit by default and `--poll` controls the query interval; Ctrl+C/timeouts only stop waiting, and submitted jobs keep running.
Do not regenerate because the client disconnected; query or download with the original ID. Report failed/cancelled states as failures.

## Duration and sampling

```bash
# Explicit short preview: 93 frames at 25 fps is about 3.72 seconds
colabllm longcat-avatar generate --audio ./speech.wav --image ./person.png --prompt 'A person speaking.' \
  --frames 93 --segments 1 -o ./preview.mp4

# Sampling steps and CFG are only available in standard mode
colabllm longcat-avatar generate --audio ./speech.wav --image ./person.png --prompt 'A person speaking.' \
  --sampling-mode standard --steps 50 --text-cfg 4 --audio-cfg 3 -o ./standard.mp4
```

Without `--frames/--segments` the full audio is covered by default; setting either one switches to manual duration mode (the other defaults to 93 frames or 1 segment).
Frames per segment are `4n+1` between 17 and 161, and segments 1-20; overlap stitching removes duplicate frames, so total length cannot be estimated as "frames x segments".
`distilled` is the fixed 8-step fast mode, and CFG can only be adjusted in standard mode; `--audio-cfg` is not a mouth-amplitude slider.
`--resolution` is 480p/720p; `--seed -1` is random. Higher resolutions and more segments increase VRAM and time.

## API, web UI and management

```bash
colabllm longcat-avatar up -i 2
colabllm longcat-avatar ps --json
colabllm longcat-avatar info --show-secrets
colabllm longcat-avatar logs -n 100
colabllm longcat-avatar down -i 2
```

The web UI uses Basic Auth and the CLI/API use Bearer. The endpoints are `POST /jobs`, `GET /jobs/ID`, `GET /jobs/ID/video` and the Drive directory listing endpoint.
Only generated results and allowed files are exposed; never expose the whole Drive as a static directory just to download files.
Across machines, set `LONGCAT_AVATAR_URL` / `LONGCAT_AVATAR_TOKEN`, or use `--url/--token`.
The base `longcat-video` has none of the generate/job/download CLI commands here; do not mix them up.
