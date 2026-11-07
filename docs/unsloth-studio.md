# Unsloth Studio: training, inference and model export

Defaults to **L4**; uses Unsloth Studio's own login, listens on internal Colab port 8888 and publishes the web UI through the selected tunnel.

Shared prerequisites, domains, authentication, timeouts and stop behavior are in [Common runtime operations](runtime-common.md). All commands below run on the local machine.

## Startup and account

```bash
colabllm unsloth-studio up
colabllm unsloth-studio status --json
colabllm unsloth-studio info --json
colabllm unsloth-studio info --show-secrets
colabllm unsloth-studio logs -n 100
```

Open Studio from the returned URL and use the actual `login_user/login_pass` (if the app reported them) or its UI login flow.
Do not assume the account fields of other Caddy services or the random `api_token` are API keys that Studio will accept.
The current notebook may only produce app credentials after Studio starts, and the start marker and login fields appear at different times; if the UI is unavailable, keep checking the logs.

## Fine-tuning workflow

1. Choose the base model and training task in the UI; gated models need a Hugging Face account with access.
2. Import the dataset and check the column mapping, chat template, train/validation split and a few samples, so a wrong format does not go into a long training run.
3. Set LoRA/QLoRA, sequence length, batch size, gradient accumulation and output location; memory needs depend on the model and settings, and L4 does not suit every model.
4. Validate loss, VRAM and output with a short training run first, then run the full training as the user asked.
5. Save checkpoints, adapters and configuration and confirm they were written to a persistent directory; choose the export format explicitly in the UI when merging or quantizing.

Tokens for gated repositories can be read by the launcher from a configured `HF_TOKEN`; avoid writing the actual token into command history or documents.
When installs/downloads fail, check permissions, network and free disk first instead of creating new VMs repeatedly.

## Inference and export

Studio's inference and training workflows are operated in the web UI. If its version exposes automatable APIs, read the actual API documentation first;
the project has no `colabllm unsloth-studio train/chat/export` subcommands.
The inference model, adapter and tokenizer must match the training configuration; a LoRA adapter alone cannot be used as a full base model.
Export to a user directory under `MyDrive/colab-llm/unsloth-studio/` and record the base model version, adapter, tokenizer and training configuration.
The default working directory or temporary UI downloads are not guaranteed to persist; before stopping, confirm the needed files actually exist on Drive, and download local copies when necessary.

## Instances, larger GPUs and stopping

```bash
colabllm unsloth-studio up -i 2 --gpu A100
colabllm unsloth-studio info -i 2 --json
colabllm unsloth-studio ps --json
colabllm unsloth-studio logs -i 2 -f
colabllm unsloth-studio down -i 2
```

GPU overrides must match the user's quota and the model's needs. Separate VMs run independently but may share the same Drive service directory,
so give different experiments different output subdirectories to avoid overwriting checkpoints.
The local runner and the VM must both stay alive until training/export finishes; closing the web page does not cancel training. `down` releases the VM,
and recoverability depends on the checkpoints actually saved and what the UI supports; automatic training resumption cannot be promised.
