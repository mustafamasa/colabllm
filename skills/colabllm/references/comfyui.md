<!-- Generated from docs/comfyui.md; edit the source and run tools/sync_runtime_docs.py. -->

# ComfyUI: workflows, model downloads and generation

Defaults to **L4** and includes ComfyUI-Manager; Caddy provides Basic Auth for the web UI and Bearer auth for the API.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and web UI

```bash
colabllm comfyui up
colabllm comfyui status --json
colabllm comfyui info --show-secrets
colabllm comfyui logs -n 100
```

Open the URL returned by `info` and log in with the browser username and password. Import a workflow, make sure the nodes and models are available, then queue the generation.
API clients use the Bearer token; the API cannot be called with the URL alone.

## Downloading workflow models before or during a run

```bash
colabllm comfyui up --workflow video_example.json
colabllm comfyui pull-workflow 'https://example.com/workflow.json' --timeout 600
colabllm comfyui pull-model 'https://example.com/model.safetensors' --dir loras --name example.safetensors --timeout 600
colabllm comfyui pull-workflow --help
```

Replace `example.com` and the sample template names with real, reachable workflows/models.
`up --workflow` can be repeated or take comma-separated values; `pull-workflow` downloads the models a workflow references to the SSD of a running VM.
These commands **do not submit generation jobs**. Do not assume that workflows without a model manifest or with unsupported references can always be inferred; act on the download results and missing-node hints.
`pull-model --dir` should be a model category such as `loras/checkpoints/diffusion_models/vae/text_encoders`, and the name must match the workflow's reference.
The download target is the remote ComfyUI, not the local current directory. `--timeout 0` waits without limit by default, and a timeout does not guarantee that the remote download is cancelled.

## Running workflows and getting results

UI workflow files and API prompt JSON are different formats. For automation, export the API format from ComfyUI and replace inputs, seeds, text and model references;
submit the prompt through the authenticated ComfyUI API, track the queue/history, then download the outputs. Never submit UI JSON as a prompt as-is.
Usually `/prompt` is used to submit and `/history/<prompt_id>` to query; files are fetched using the returned filename/subfolder/type.
The endpoints and custom node parameters actually supported by the running service are authoritative; query `/object_info` when needed instead of guessing node parameters.

Keep the prompt ID and do not re-queue while the job is still queued or running. Download output files to the local directory the user asked for,
or report the Drive path after confirming it was persisted. Imported custom nodes may need installation and a restart; do that only after the user's active jobs finish.

## Storage, multiple instances and recovery

```bash
colabllm comfyui up -i 2
colabllm comfyui pull-workflow video_example.json -i 2 --timeout 600
colabllm comfyui pull-model 'https://example.com/model.safetensors' --dir loras --instances 1,2 --timeout 600
colabllm comfyui ps --json
colabllm comfyui down -i 2
```

`--instances` / `-A/--all-instances` download to several existing instances; confirm the targets first to avoid wasting disk.
Inputs, outputs, user settings and custom_nodes are stored in the service's Drive directory; model files live on the local SSD at `/content/ComfyUI/models` by default.
Models must be downloaded again after the VM is released, so save each workflow's model links, file names and node versions. A shared Drive does not mean queued jobs resume automatically on a new VM.
For missing models/wrong file names, check the download results first; for a 502, first determine whether the model or service is still loading. Read the logs before deciding to restart.
