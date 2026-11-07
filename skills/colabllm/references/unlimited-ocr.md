<!-- Generated from docs/unlimited-ocr.md; edit the source and run tools/sync_runtime_docs.py. -->

# Unlimited-OCR 2: batch PDF and image to Markdown

Defaults to **A100** with Caddy Bearer/Basic Auth; the model lives on the Colab SSD, and jobs and conversion results are persisted on Drive.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and the shortest conversion

```bash
colabllm unlimited-ocr up
colabllm unlimited-ocr status --json
colabllm unlimited-ocr ocr ./document.pdf -O ./ocr-results
colabllm unlimited-ocr ocr --help
```

`ocr` has the aliases `convert/run`. Local files are uploaded to Drive in chunks and then processed on the GPU; `-O` downloads the results back to the local machine.
Readiness requires model loading and a real inference warm-up; a reachable domain does not mean OCR is available.

## Input and output directories

```bash
colabllm unlimited-ocr ocr ./documents --local -O ./ocr-results
colabllm unlimited-ocr ocr 'drive://scans' --output-dir 'OCR/results' -O ./ocr-results
colabllm unlimited-ocr ocr ./document.pdf 'drive://scans' -O ./ocr-results
colabllm unlimited-ocr ls-drive 'scans'
```

PDF, PNG, JPEG, WEBP, BMP and TIFF can be processed; directories are recursive by default, and `--no-recursive` limits to the current level.
Existing local paths are treated as local automatically; `drive://`, `MyDrive/` or a full remote MyDrive path are treated as Drive.
Other relative paths may be interpreted as Drive; prefix local files with `./` or use `--local`. A Drive directory is not a web share link or a file ID.
`--output-dir` is the **Drive** root; `-O/--local-output-dir` is the **local** root; both can be given.

The default Drive output is `MyDrive/colab-llm/unlimited-ocr/outputs/`. Uploaded copies of local files are kept in the service's Drive `uploads/`;
clean them up only once no job depends on them. Local folders keep their folder name/relative subdirectories; output name conflicts among local files in one batch are errors.

## Background jobs, queries and recovery

```bash
colabllm unlimited-ocr ocr 'drive://scans' --no-wait
colabllm unlimited-ocr job JOB_ID
colabllm unlimited-ocr job JOB_ID --wait --timeout 600 -O ./ocr-results
colabllm unlimited-ocr resume JOB_ID --wait --timeout 600 -O ./ocr-results
```

Once the upload has finished and a job ID was returned, a client disconnect does not cancel the server-side job. By default `ocr` waits for completion; `--no-wait` only submits.
`job --wait` with `--timeout 0` waits without limit by default; `--poll-interval` changes the query interval. A timeout exits with 124 while the job keeps running in the background; keep querying the original ID.
`resume` is for failed/interrupted jobs; queued, running or completed jobs cannot be resumed again.
A service restart re-queues the queued/running jobs from the persistent records and reuses completed page groups; it never blindly reruns from scratch.

## Parameters and result acceptance

```bash
colabllm unlimited-ocr ocr ./documents --local --dpi 200 --pages-per-chunk 10 \
  --tokens-per-page 8000 -O ./ocr-results
colabllm unlimited-ocr ocr ./document.pdf --overwrite -O ./ocr-results
```

DPI is 72-400, page groups 1-100 and the per-page token budget 32-20000. Large page groups use more VRAM and keep cross-page context within a group,
but bigger is not always better; paragraphs/tables are not guaranteed to be merged across groups. The default is 10 pages per group and the budget has a per-group cap; when output is truncated, adjust parameters and check the output.
`ocr --help` is authoritative for advanced inference parameters; custom prompts or layout parameters may change the result structure, so validate on a small sample before processing a whole batch.
Completed, unchanged files are skipped by default; `--overwrite` forces reconversion.

Each file's directory usually contains `result.md`, `result.raw.txt`, cropped images under `chunk-*/` and `complete.json`.
Image references in the Markdown depend on the subdirectories, so keep the structure when delivering. Jobs can partially fail; check each file's status and errors. The CLI returns 1 for failed/interrupted.
Job records are kept in `MyDrive/colab-llm/unlimited-ocr/jobs/`; the output paths in the job response are authoritative.

## Multiple instances and remote calls

```bash
colabllm unlimited-ocr up -i 2
colabllm unlimited-ocr ocr ./document.pdf -i 2 -O ./instance2
colabllm unlimited-ocr logs -i 2 -n 100
colabllm unlimited-ocr down -i 2
```

`UNLIMITED_OCR_URL` / `--url` and `UNLIMITED_OCR_API_TOKEN` / `--api-token` connect to an existing service.
The API includes `POST /api/jobs`, `GET /api/jobs/ID`, `POST /api/jobs/ID/resume`, plus upload and result endpoints; large files are transferred in chunks with small requests.
Several instances may share the Drive directory holding jobs and outputs; do not treat them as isolated data accounts.
