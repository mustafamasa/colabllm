# Notebook build rules (`templates/build_notebooks.py`)

All service notebooks (`src/colabllm/services/*.ipynb`) and `templates/service.ipynb` are **generated**.
Never hand-edit a generated `.ipynb`: `tests/test_notebooks_sync.py` regenerates them in a temporary
copy and fails if any committed notebook differs.

```bash
python3 templates/build_notebooks.py          # regenerate every notebook
python3 -m unittest tests/test_notebooks_sync.py
```

## Where code lives

| Kind of code | Location | How it reaches the VM |
| --- | --- | --- |
| Cell skeletons (config, Caddy, keep-alive, per-service setup/service cells) | string constants in `build_notebooks.py` | become notebook cells |
| Shared transport | `templates/tunnel_client.py` | pasted as cell 4 |
| Compiler cache archives | `templates/colab_cache.py` | written to `<work>/colab_cache.py` and imported |
| ASGI security / Drive helpers | `templates/colab_asgi_common.py` | written next to each middleware |
| Service middlewares / workers | `templates/nemotron_middleware.py`, `indextts_middleware.py`, `qwen3_tts_middleware.py`, `qwen3_tts_aux_worker.py`, `drama3_middleware.py`, `tangoflux_service.py`, `flux_klein_middleware.py`, `qwen_image_middleware.py`, `minimax_h3_image_middleware.py` | written to the service work dir |
| ComfyUI helper, custom node, browser JS | `templates/comfyui_workflow_helper.py`, `comfyui_workflow_downloader.py`, `comfyui_ssd_downloader.js` | written under `/content/ComfyUI` |
| Code shared with the local CLI | `src/colabllm/lib/` (e.g. `nemotron_audio.py`, `avatar_sampling.py`, `ocr_parameters.py`) | embedded the same way |

Rules:

1. Non-trivial logic (> ~30 lines, or anything security-relevant) goes in a standalone module, not an
   inline string. Modules must import with the standard library alone so local tests can exercise them;
   guard heavy/runtime-only imports (`starlette`, `nemo`, `torch`) with `try/except ImportError` fallbacks
   when the module is security-tested locally.
2. Embed files with a placeholder plus `repr()` of the source, resolved when the builder runs:
   ```python
   SERVICE_CELL = r'''...
   (WORK / "colab_asgi_common.py").write_text(__ASGI_COMMON_SOURCE__, encoding="utf-8")
   ...'''.replace("__ASGI_COMMON_SOURCE__", repr((ROOT / "templates/colab_asgi_common.py").read_text()))
   ```
   Placeholders use `__UPPER_SNAKE__`; the sync test fails on any that remain unresolved.
3. Do not derive one cell from another with `.replace()` / `.index()` surgery. Use the parameterized
   builders instead:
   - `render_caddy_cell(rewrite_host=True, extra_proxy_lines="")` for the auth cell;
   - `render_keepalive(cfg, before_ready="")` for the standard keep-alive cell (code runs after the
     health check, before `CT_READY`);
   - `build_longcat_setup(...)`, `build_longcat_service(...)`, `embed_source(dir_var, filename, path)`
     as the pattern for families of similar services.
4. `write()` runs `ast.parse` on every code cell (lines starting with `%`/`!` are treated as comments);
   a syntax error aborts generation.

## Helpers provided by cell 0 (`CONFIG`)

Use these rather than re-implementing them per service:

| Helper | Purpose |
| --- | --- |
| `ct_spawn(name, args, env=, cwd=)` | start a process group, log to `/content/ct/<name>.log` |
| `ct_stop_proc(proc, timeout=15)` | SIGTERM the group, escalate to SIGKILL after a bounded wait |
| `ct_wait_http(url, timeout, ok=)` | wait for a local port/health status |
| `ct_tail(name, n)`, `ct_log_size(name)`, `ct_log_since(name, offset)` | read logs; offsets avoid re-reading old error lines |
| `ct_gpu_stat(query)` | `nvidia-smi` summary that never raises (`cpu/no-gpu` fallback) |
| `ct_fetch_binary(path, url)` | download with timeout + atomic rename |
| `ct_public_gate(url, probe, verify_auth, log, timeout=120)` | public readiness: network errors retry, auth failures and timeout raise |
| `ct_watchdog(health, start_service, log, *, log_name, grace=900, dead_markers=(), heartbeat=None)` | supervise `PROCS` forever, 900 s grace after restarts, 6 consecutive failures trigger a restart |
| `MOUNT_DRIVE`, `DATA_DIR` | `CT_MOUNT_DRIVE=0` (`up --no-drive`) moves `DATA_DIR` to `/content/data/<service>` |

Custom keep-alive cells (vLLM/llama.cpp services) must: wait for local health, run `auth_check` locally,
warm up, save caches, call `ct_public_gate`, print `CT_READY`, then end in `ct_watchdog(...)`. Never print
`CT_READY` when the public auth check failed or timed out.

## Security rules for embedded middlewares

- Authenticate with `colab_asgi_common.verify_bearer_auth` (constant-time, **fail-closed** when the token is
  empty). No unauthenticated loopback `/health` bypass; local probes send the Bearer token.
- Resolve every user-supplied path (`drive://`, `gdrive://`, `file://`, `MyDrive/...`, relative) with
  `resolve_drive_path(..., data_dir=DATA_DIR, default_subdir=...)`; it raises `ValueError` outside
  MyDrive / `DATA_DIR` (respecting `--no-drive`). Map `ValueError` to HTTP 400.
- Server-internal files (e.g. default reference WAV) are selected by server-side flags
  (`use_default_ref: true`), never by client-provided paths.
- Validate custom voice names with `validate_voice_name`; build multipart bodies with
  `build_multipart_form`; sanitize response header values with `sanitize_header_param`.
- Scope vLLM `--allowed-local-media-path` to the service work dir, never `/content`.
- Give every `subprocess.run` / `urlopen` a timeout.
- ComfyUI model destinations go through `validate_model_destination` (confined to `models/`).

## Compiler cache standard (`colab_cache.py`)

Compiler caches (FlashInfer, Triton, TorchInductor/vLLM) live on local NVMe SSD and persist to Drive as
**one tar archive per cache directory, keyed by environment + GPU**:

```python
(WORK / "colab_cache.py").write_text(__CACHE_SOURCE__, encoding="utf-8")
sys.path.insert(0, str(WORK))
from colab_cache import restore_cache, save_cache
CACHE_KEY = "vllm-0.32.0-py312-sm" + ct_gpu_stat("compute_cap").splitlines()[0].replace(".", "")
CACHE_ARCHIVES = Path(DATA_DIR) / "cache-archives"
for local, legacy in CACHE_PAIRS:                       # setup cell, before starting the server
    try:
        restore_cache(local, legacy, CACHE_ARCHIVES, CACHE_KEY, log)
    except (OSError, ValueError) as err:
        log(f"Cache restore warning ({local.name}): {err}")
...
for local, _ in CACHE_PAIRS:                            # keep-alive cell, after health + warmup
    try:
        save_cache(local, CACHE_ARCHIVES, CACHE_KEY, log)
    except (OSError, ValueError) as err:
        log(f"Cache archive save warning ({local.name}): {err}")
```

Point `FLASHINFER_WORKSPACE_BASE`, `TRITON_CACHE_DIR`, `VLLM_CACHE_ROOT`, `TORCHINDUCTOR_CACHE_DIR`,
`SGLANG_JIT_CACHE_DIR`, and `SGLANG_DG_CACHE_DIR` at the local SSD directories. Never point compiler caches
directly at `/content/drive` and do not use `cp -rn` / `cp -ru` directory mirroring. Run a real warmup request
before `save_cache` so lazily compiled kernels are included. Reference implementations: `NEMOTRON_TRANSCRIBE_*`,
`INDEXTTS_*`, `QWEN3_TTS_*`, `MINIMAX_H3_IMAGE_*`, `LONGCAT_SETUP_TEMPLATE`, `templates/unlimited_ocr_setup.py`.

## Checklist when changing notebooks

1. Edit `build_notebooks.py` or the module under `templates/` / `src/colabllm/lib/`.
2. `python3 templates/build_notebooks.py`.
3. Add or extend local tests for the module (see `tests/test_notebooks_sync.py`).
4. `python3 -m unittest discover -s tests`.
5. Commit the generator, the modules and every regenerated `.ipynb` together.
