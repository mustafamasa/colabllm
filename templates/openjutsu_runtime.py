"""Self-contained Colab setup and verification helpers for Openjutsu (ComfyUI + comfy-api-proxy + Next.js)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

NODE_VERSION = "22.14.0"
PROXY_VERSION = "0.1.6"
DEFAULT_OPENJUTSU_REPO = "https://github.com/D3voz/openjutsu.git"
HF_RESOLVE_URL_RE = re.compile(r"^https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+?)(?:\?.*)?$")

CUSTOM_NODES: tuple[tuple[str, str], ...] = (
    ("ComfyUI-MiniMaxH3-Context-Loop", "https://github.com/ethanfel/ComfyUI-MiniMaxH3-Context-Loop.git"),
    ("comfyui-h3-oneclick-swap", "https://github.com/D3voz/comfyui-h3-oneclick-swap.git"),
    ("ComfyUI-Fantastic-MiniMaxH3-PromptBuilder", "https://github.com/Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder.git"),
    ("ComfyUI-KJNodes", "https://github.com/kijai/ComfyUI-KJNodes.git"),
    ("ComfyUI-VideoHelperSuite", "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"),
    ("comfyui-websocket-reconnect-fix", "https://github.com/D3voz/comfyui-websocket-reconnect-fix.git"),
    ("SolarRenderer", "https://github.com/Enderfga/SolarRenderer.git"),
    ("comfyui_controlnet_aux", "https://github.com/Fannovel16/comfyui_controlnet_aux.git"),
    ("ComfyUI-H3-FaceRefine", "https://github.com/Carasibana/ComfyUI-H3-FaceRefine.git"),
    ("ComfyUI-Impact-Pack", "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"),
    ("comfyui-image-detail-contrast", "https://github.com/D3voz/comfyui-image-detail-contrast.git"),
    ("Orion4D_FXMax", "https://github.com/orion4d/Orion4D_FXMax.git"),
    ("ComfyUI-Spectrum-MiniMax-H3", "https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3.git"),
)

CORE_MODELS: tuple[dict[str, object], ...] = (
    {
        "directory": "diffusion_models",
        "name": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "approx_gb": 20.0,
        "urls": (
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        ),
    },
    {
        "directory": "text_encoders",
        "name": "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
        "approx_gb": 15.0,
        "urls": (
            "https://huggingface.co/Merserk/MiniMax-H3-INT4-ConvRot/resolve/main/qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
            "https://huggingface.co/Abiray/MiniMax-H3-GGUF/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
        ),
    },
    {
        "directory": "text_encoders",
        "name": "qwen3vl_8b_int8_convrot.safetensors",
        "approx_gb": 8.9,
        "urls": (
            "https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors",
        ),
    },
    {
        "directory": "vae",
        "name": "minimax_h3_video_vae_int8_convrot.safetensors",
        "approx_gb": 2.8,
        "urls": (
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_int8_convrot.safetensors",
        ),
    },
    {
        "directory": "vae",
        "name": "minimax_h3_audio_vae_fp32.safetensors",
        "approx_gb": 0.6,
        "urls": (
            "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors",
        ),
    },
    {
        "directory": "loras",
        "name": "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors",
        "approx_gb": 0.55,
        "urls": (
            "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/loras/minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors",
        ),
    },
    {
        "directory": "checkpoints",
        "name": "sam3.1_multiplex_fp16.safetensors",
        "approx_gb": 1.7,
        "urls": (
            "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors",
        ),
    },
)

EXTRA_FULL_MODELS: tuple[dict[str, object], ...] = (
    {
        "directory": "text_encoders",
        "name": "qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors",
        "approx_gb": 9.8,
        "urls": (
            "https://huggingface.co/ml-resources/flux-3-image-klein-comfy/resolve/main/text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors",
            "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/text_encoders/qwen3.5_9b_flux3_prompt_upsampler_int8_convrot.safetensors",
        ),
    },
    {
        "directory": "diffusion_models",
        "name": "flux3_image_klein_sd_int8_convrot.safetensors",
        "approx_gb": 9.4,
        "urls": (
            "https://huggingface.co/ml-resources/flux-3-image-klein-comfy/resolve/main/diffusion_models/flux3-image-klein-int8-convrot.safetensors",
            "https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy/resolve/main/diffusion_models/flux3-image-klein-int8-convrot.safetensors",
        ),
    },
    {
        "directory": "text_encoders",
        "name": "qwen3vl_4b_fp8_scaled.safetensors",
        "approx_gb": 5.24,
        "urls": (
            "https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
        ),
    },
    {
        "directory": "loras",
        "name": "solarrenderer_4step_lora.safetensors",
        "approx_gb": 4.15,
        "urls": (
            "https://huggingface.co/Enderfga/SolarRenderer/resolve/main/solarrenderer_4step_lora.safetensors",
        ),
    },
    {
        "directory": "loras",
        "name": "solarrenderer_control_lora.safetensors",
        "approx_gb": 4.15,
        "urls": (
            "https://huggingface.co/Enderfga/SolarRenderer/resolve/main/solarrenderer_control_lora.safetensors",
        ),
    },
    {
        "directory": "sams",
        "name": "sam_vit_b_01ec64.pth",
        "approx_gb": 0.38,
        "urls": (
            "https://huggingface.co/comfyuistudio/sam_vit_b_01ec64.pth/resolve/main/sam_vit_b_01ec64.pth",
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        ),
    },
    {
        "directory": "vae",
        "name": "flux2-vae.safetensors",
        "approx_gb": 0.34,
        "urls": (
            "https://huggingface.co/Comfy-Org/Flux-2-Klein-9B/resolve/main/split_files/vae/flux2-vae.safetensors",
            "https://huggingface.co/ml-resources/flux-3-image-klein-comfy/resolve/main/vae/flux2-vae.safetensors",
        ),
    },
    {
        "directory": "ultralytics/segm",
        "name": "person_yolov8m-seg.pt",
        "approx_gb": 0.06,
        "urls": (
            "https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt",
        ),
    },
    {
        "directory": "ultralytics/bbox",
        "name": "face_yolov8m.pt",
        "approx_gb": 0.05,
        "urls": (
            "https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt",
        ),
    },
)

MODEL_PROFILES: dict[str, tuple[dict[str, object], ...]] = {
    "core": CORE_MODELS,
    "full": CORE_MODELS + EXTRA_FULL_MODELS,
}

EXTRA_NODE_PACKAGES: tuple[str, ...] = (
    "huggingface_hub",
    "hf_xet",
    "av>=17.0.0",
    "imageio-ffmpeg",
    "color-matcher",
    "mss",
    "segment-anything",
    "scikit-image",
    "scikit-learn",
    "piexif",
    "dill",
    "matplotlib",
    "ultralytics",
    "scipy",
    "insightface",
    "onnxruntime-gpu",
    "scenedetect>=0.7",
    "mediapipe>=0.8.0",
    "omegaconf",
    "addict",
    "yacs",
    "trimesh",
    "albumentationsx",
)


def olog(msg: str) -> None:
    print(time.strftime("[%F %T] [openjutsu] ") + msg, flush=True)


def validate_model_destination(directory: str, name: str, comfy_dir: Path) -> Path:
    """Resolve and verify that <comfy_dir>/models/<directory>/<name> stays strictly inside models/."""
    models_root = (comfy_dir / "models").resolve()
    rel_dir = (directory or "").strip().strip("/")
    rel_name = (name or "").strip().lstrip("/")
    if not rel_dir or not rel_name or "\x00" in rel_dir or "\x00" in rel_name:
        raise ValueError(f"Invalid model destination: directory={directory!r}, name={name!r}")
    if ".." in Path(rel_dir).parts or ".." in Path(rel_name).parts or "/" in rel_name or "\\" in rel_name:
        raise ValueError(f"Refusing path traversal in model destination: {rel_dir}/{rel_name}")
    dest = (models_root / rel_dir / rel_name).resolve()
    if not dest.is_relative_to(models_root) or dest == models_root:
        raise ValueError(f"Refusing model path outside {models_root}: {rel_dir}/{rel_name}")
    return dest


def models_for_profile(profile: str) -> tuple[dict[str, object], ...]:
    key = (profile or "full").strip().lower()
    if key not in MODEL_PROFILES:
        raise ValueError(f"Unsupported Openjutsu model profile {profile!r}; expected one of {sorted(MODEL_PROFILES)}")
    return MODEL_PROFILES[key]


def render_extra_model_paths(comfy_dir: Path) -> str:
    """Register `ultralytics` and `sams` model folders before custom nodes are imported."""
    base = str(comfy_dir.resolve())
    return (
        "openjutsu:\n"
        f"  base_path: {json.dumps(base)}\n"
        "  ultralytics: models/ultralytics\n"
        "  ultralytics_bbox: models/ultralytics/bbox\n"
        "  ultralytics_segm: models/ultralytics/segm\n"
        "  sams: models/sams\n"
    )


VRAM_MODES = ("auto", "highvram", "dynamic", "gpu-only")
CHAINS_STORAGE_MODES = ("ssd", "drive")


def detect_gpu_vram_mib() -> int:
    """Return total GPU0 VRAM in MiB, or 0 if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=15,
        )
        first = out.splitlines()[0].strip()
        return int(float(first))
    except Exception:
        return 0


def select_comfy_vram_flags(vram_total_mib: int, mode: str = "auto") -> list[str]:
    """Select ComfyUI VRAM flags; enables `--highvram` automatically on >=76 GiB GPUs (e.g. G4 96 GB)."""
    clean = (mode or "auto").strip().lower()
    if clean not in VRAM_MODES:
        raise ValueError(f"Unsupported VRAM mode {mode!r}; expected one of {VRAM_MODES}")
    if clean == "highvram":
        return ["--highvram"]
    if clean == "gpu-only":
        return ["--gpu-only"]
    if clean == "dynamic":
        return []
    return ["--highvram"] if vram_total_mib >= 76000 else []


def configure_chains_storage(work_dir: Path, data_dir: Path, mode: str = "ssd") -> str:
    """Route `output/h3_chains` per-clip checkpoints (~450 MB/clip) to local NVMe SSD via VFS bind mount.

    Using `mount --bind` (instead of a symlink) keeps `os.path.realpath` inside `output/`,
    satisfying Context-Loop's `CheckpointGraphManager` path-confinement check while avoiding
    multi-GB synchronous FUSE writes and Google Drive quota bloat.
    """
    clean = (mode or "ssd").strip().lower()
    if clean not in CHAINS_STORAGE_MODES:
        raise ValueError(f"Unsupported chains storage mode {mode!r}; expected one of {CHAINS_STORAGE_MODES}")
    drive_chains = data_dir / "output" / "h3_chains"
    drive_chains.mkdir(parents=True, exist_ok=True)
    if clean == "drive":
        if os.path.ismount(drive_chains):
            subprocess.run(["umount", str(drive_chains)], check=False, timeout=30)
            olog(f"Unmounted SSD bind mount; h3_chains stored on Drive: {drive_chains}")
        return "drive"
    if not str(data_dir).startswith("/content/drive/"):
        return "ssd-native"
    local_chains = work_dir / "h3_chains"
    local_chains.mkdir(parents=True, exist_ok=True)
    if os.path.ismount(drive_chains):
        return "ssd-bind"
    res = subprocess.run(["mount", "--bind", str(local_chains), str(drive_chains)],
                         capture_output=True, text=True, check=False, timeout=30)
    if res.returncode == 0:
        olog(f"Bind-mounted local NVMe SSD for h3_chains checkpoints: {local_chains} -> {drive_chains}")
        return "ssd-bind"
    olog(f"Bind-mount unavailable ({res.stderr.strip() or res.returncode}); keeping h3_chains on {drive_chains}")
    return "drive-fallback"


def normalize_linux_workflows(app_dir: Path) -> list[str]:
    """Normalize Windows backslash relative model paths (e.g. face.api.json) for POSIX ComfyUI."""
    wf_root = app_dir / "workflows"
    updated: list[str] = []
    if not wf_root.is_dir():
        return updated
    for wf_path in sorted(wf_root.rglob("*.json")):
        raw = wf_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        changed = False
        if isinstance(data, dict):
            for node in data.values():
                if not isinstance(node, dict) or node.get("class_type") != "H3FaceTrackCrop":
                    continue
                inputs = node.get("inputs")
                if not isinstance(inputs, dict):
                    continue
                for field in ("detector", "fallback_detector"):
                    val = inputs.get(field)
                    if isinstance(val, str) and "\\" in val:
                        inputs[field] = val.replace("\\", "/")
                        changed = True
        if changed:
            wf_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            updated.append(str(wf_path.relative_to(app_dir)))
    return updated


def run_logged(cmd: list[str], log_path: Path, *, env: dict[str, str] | None = None,
               cwd: Path | str | None = None, check: bool = True, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(str(x) for x in cmd)}\n")
        handle.flush()
        res = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True, timeout=timeout)
    if check and res.returncode != 0:
        tail = "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(True)[-45:])
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(str(x) for x in cmd[:5])}\n{tail}")
    return res


def ensure_nodejs(work_dir: Path, log_path: Path) -> Path:
    node_dir = work_dir / f"node-v{NODE_VERSION}"
    node_bin = node_dir / "bin" / "node"
    if node_bin.is_file():
        return node_dir / "bin"
    arch = {"x86_64": "x64", "aarch64": "arm64"}.get(platform.machine())
    if not arch:
        raise RuntimeError(f"Unsupported architecture for Node.js binary: {platform.machine()}")
    tar_name = f"node-v{NODE_VERSION}-linux-{arch}.tar.xz"
    base_url = f"https://nodejs.org/dist/v{NODE_VERSION}/"
    olog(f"Installing Node.js v{NODE_VERSION} ({arch}) -> {node_dir}")
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        archive = Path(tmp) / tar_name
        req_sums = urllib.request.Request(base_url + "SHASUMS256.txt", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_sums, timeout=30) as resp:
            shasums = resp.read().decode("utf-8")
        expected = next(
            (line.split()[0] for line in shasums.splitlines() if line.split() and line.split()[-1] == tar_name),
            None,
        )
        if not expected:
            raise RuntimeError(f"Could not find SHA256 for {tar_name} in Node.js SHASUMS256.txt")
        req_tar = urllib.request.Request(base_url + tar_name, headers={"User-Agent": "Mozilla/5.0"})
        h = hashlib.sha256()
        with urllib.request.urlopen(req_tar, timeout=120) as src, archive.open("wb") as dst:
            while chunk := src.read(4 * 1024 * 1024):
                dst.write(chunk)
                h.update(chunk)
        if h.hexdigest() != expected:
            raise RuntimeError(f"Node.js archive SHA256 mismatch: expected {expected}, got {h.hexdigest()}")
        extract_root = Path(tmp) / "extracted"
        extract_root.mkdir()
        with tarfile.open(archive, "r:xz") as tf:
            tf.extractall(extract_root, filter="data")
        extracted_sub = extract_root / f"node-v{NODE_VERSION}-linux-{arch}"
        if node_dir.exists():
            shutil.rmtree(node_dir)
        extracted_sub.replace(node_dir)
    ver = subprocess.check_output([str(node_bin), "--version"], text=True, timeout=15).strip()
    olog(f"Node.js ready: {ver}")
    return node_dir / "bin"


def _symlink_dir(local_path: Path, target_path: Path) -> None:
    target_path.mkdir(parents=True, exist_ok=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.is_symlink():
        if os.readlink(local_path) != str(target_path):
            local_path.unlink()
            local_path.symlink_to(target_path)
    elif local_path.is_dir():
        subprocess.run(["cp", "-rn", f"{local_path}/.", f"{target_path}/"], check=False, timeout=120)
        shutil.rmtree(local_path)
        local_path.symlink_to(target_path)
    else:
        local_path.symlink_to(target_path)


def prepare_comfy_tree(work_dir: Path, comfy_dir: Path, data_dir: Path) -> Path:
    """Ensure ComfyUI repo, local SSD models/ root, and Drive symlinks exist before parallel tasks start."""
    env_log = work_dir / "env-setup.log"
    comfy_repo = os.environ.get("CT_COMFY_REPO", "https://github.com/comfyanonymous/ComfyUI.git")
    if not (comfy_dir / ".git").is_dir():
        if comfy_dir.exists():
            shutil.rmtree(comfy_dir)
        olog(f"Cloning ComfyUI -> {comfy_dir}")
        run_logged(["git", "clone", "--depth", "1", comfy_repo, str(comfy_dir)], env_log, timeout=600)
    else:
        run_logged(["git", "-C", str(comfy_dir), "pull", "--ff-only"], env_log, check=False, timeout=300)

    if (comfy_dir / "models").is_symlink():
        (comfy_dir / "models").unlink()
    (comfy_dir / "models" / "loras").mkdir(parents=True, exist_ok=True)
    (comfy_dir / "custom_nodes").mkdir(parents=True, exist_ok=True)
    for sub in ("output", "input", "user"):
        _symlink_dir(comfy_dir / sub, data_dir / sub)
    _symlink_dir(comfy_dir / "models" / "loras" / "drive", data_dir / "loras")
    configure_chains_storage(work_dir, data_dir, os.environ.get("CT_OPENJUTSU_CHAINS_STORAGE", "ssd"))

    extra_paths_file = work_dir / "extra_model_paths.yaml"
    extra_paths_file.write_text(render_extra_model_paths(comfy_dir), encoding="utf-8")
    return extra_paths_file


def _normalize_git_url(url: str) -> str:
    return url.strip().rstrip("/").removesuffix(".git")


def sync_app_checkout(app_dir: Path, app_repo: str, env_log: Path) -> None:
    repo = (app_repo or DEFAULT_OPENJUTSU_REPO).strip() or DEFAULT_OPENJUTSU_REPO
    if (app_dir / ".git").is_dir():
        origin_res = subprocess.run(
            ["git", "-C", str(app_dir), "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        current_origin = origin_res.stdout.strip() if origin_res.returncode == 0 else ""
        if current_origin and _normalize_git_url(current_origin) == _normalize_git_url(repo):
            run_logged(["git", "-C", str(app_dir), "pull", "--ff-only"], env_log, check=False, timeout=300)
            return
        olog(f"Switching Openjutsu checkout ({current_origin or 'unknown'}) -> {repo}")
        shutil.rmtree(app_dir)
    elif app_dir.exists():
        shutil.rmtree(app_dir)
    olog(f"Cloning Openjutsu ({repo}) -> {app_dir}")
    run_logged(["git", "clone", "--depth", "1", repo, str(app_dir)], env_log, timeout=300)


def torch_supports_sm120(version_file: Path | None = None) -> bool:
    """Check Blackwell sm_120 support via `torch/version.py` without cold-loading 2.5 GB of CUDA `.so`s
    while `hf_xet` saturates NVMe disk writes; falls back to a 180s subprocess check if needed.
    """
    vf = version_file
    if vf is None:
        importlib.invalidate_caches()
        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.origin:
            return False
        vf = Path(spec.origin).parent / "version.py"
    if vf.is_file():
        text = vf.read_text(encoding="utf-8", errors="replace")
        if re.search(r"^\s*cuda(?:\s*:\s*[^=]+)?\s*=\s*None\b", text, flags=re.MULTILINE):
            return False
        m = re.search(r"""^\s*cuda(?:\s*:\s*[^=]+)?\s*=\s*['"](\d+)\.(\d+)""", text, flags=re.MULTILINE)
        if m:
            return (int(m.group(1)), int(m.group(2))) >= (12, 8)
    try:
        arch_check = subprocess.run(
            [sys.executable, "-c", "import sys, torch; sys.exit(0 if 'sm_120' in torch.cuda.get_arch_list() else 1)"],
            capture_output=True,
            text=True,
            timeout=180,
        )
        return arch_check.returncode == 0
    except subprocess.TimeoutExpired:
        olog("[warn] Timed out probing torch.cuda.get_arch_list() after 180s; treating sm_120 as unsupported")
        return False


def _modules_installed(*names: str) -> bool:
    importlib.invalidate_caches()
    return all(importlib.util.find_spec(name) is not None for name in names)


def setup_environment(work_dir: Path, comfy_dir: Path, data_dir: Path) -> dict[str, str]:
    env_log = work_dir / "env-setup.log"
    extra_paths_file = work_dir / "extra_model_paths.yaml"
    if not extra_paths_file.is_file():
        extra_paths_file = prepare_comfy_tree(work_dir, comfy_dir, data_dir)

    if not shutil.which("ffmpeg"):
        subprocess.run("apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null 2>&1",
                       shell=True, check=False, timeout=300)
    if not shutil.which("uv"):
        run_logged([sys.executable, "-m", "pip", "install", "-q", "-U", "uv"], env_log, timeout=300)
    uv_bin = shutil.which("uv") or "uv"

    # 1. Ensure Blackwell sm_120 PyTorch support if running on G4
    try:
        cap = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], text=True, timeout=15
        ).splitlines()[0].strip()
    except Exception:
        cap = ""
    if cap.startswith("12.") and not torch_supports_sm120():
        olog("Upgrading PyTorch to CUDA 13.0 build for Blackwell sm_120...")
        run_logged([
            uv_bin, "pip", "install", "--system", "-U",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu130",
        ], env_log, timeout=900)

    # 3. Clone Openjutsu repo + Node.js 22 + build production Next.js app
    app_dir = work_dir / "app"
    app_repo = (os.environ.get("CT_OPENJUTSU_REPO") or DEFAULT_OPENJUTSU_REPO).strip()
    sync_app_checkout(app_dir, app_repo, env_log)

    patched_wfs = normalize_linux_workflows(app_dir)
    if patched_wfs:
        olog(f"Normalized POSIX detector paths in workflows: {', '.join(patched_wfs)}")

    # 4. Clone all 13 Openjutsu custom node packs onto local SSD
    olog(f"Installing {len(CUSTOM_NODES)} ComfyUI custom node packs...")
    for node_name, node_repo in CUSTOM_NODES:
        node_dir = comfy_dir / "custom_nodes" / node_name
        if not (node_dir / ".git").is_dir():
            if node_dir.exists():
                shutil.rmtree(node_dir)
            run_logged(["git", "clone", "--depth", "1", node_repo, str(node_dir)], env_log, timeout=300)
        else:
            run_logged(["git", "-C", str(node_dir), "pull", "--ff-only"], env_log, check=False, timeout=300)

    # Apply Openjutsu's SolarRenderer extended-silence patch idempotently
    solar_dir = comfy_dir / "custom_nodes" / "SolarRenderer"
    solar_patch = app_dir / "patches" / "solarrenderer-long-clips.patch"
    solar_nodes_py = solar_dir / "comfyui" / "nodes.py"
    if solar_patch.is_file() and solar_nodes_py.is_file():
        if "the held silence covers 170 frames" in solar_nodes_py.read_text(encoding="utf-8", errors="replace"):
            chk = subprocess.run(["git", "-C", str(solar_dir), "apply", "--check", str(solar_patch)],
                                 capture_output=True, text=True, timeout=30)
            if chk.returncode == 0:
                run_logged(["git", "-C", str(solar_dir), "apply", str(solar_patch)], env_log, timeout=30)
                olog("Applied SolarRenderer extended-silence patch for long clips")

    # 5. Install Python dependencies for ComfyUI + all custom nodes
    deps_marker = work_dir / ".comfy_nodes_deps_v1.ok"
    required_mods = ("comfy_kitchen", "insightface", "ultralytics", "segment_anything", "color_matcher")
    if not deps_marker.is_file() or not _modules_installed(*required_mods):
        olog("Installing ComfyUI and custom-node Python dependencies via uv...")
        run_logged([uv_bin, "pip", "install", "--system", "-r", str(comfy_dir / "requirements.txt")],
                   env_log, timeout=900)
        run_logged([uv_bin, "pip", "install", "--system", *EXTRA_NODE_PACKAGES], env_log, timeout=900)
        # sam2 for Impact-Pack (if not already importable)
        if not _modules_installed("sam2"):
            sam2_env = {**os.environ, "SAM2_BUILD_CUDA": "0"}
            run_logged([uv_bin, "pip", "install", "--system", "sam2", "--no-build-isolation"],
                       env_log, env=sam2_env, check=False, timeout=300)
            if not _modules_installed("sam2"):
                run_logged([uv_bin, "pip", "install", "--system", "--no-build-isolation",
                            "git+https://github.com/facebookresearch/sam2"],
                           env_log, env=sam2_env, check=False, timeout=600)
        for node_name, _ in CUSTOM_NODES:
            req = comfy_dir / "custom_nodes" / node_name / "requirements.txt"
            if req.is_file() and node_name != "ComfyUI-Impact-Pack":
                run_logged([uv_bin, "pip", "install", "--system", "-r", str(req)], env_log, check=False, timeout=600)
        deps_marker.write_text("ok\n", encoding="utf-8")

    # 6. Dedicated virtualenv for comfy-api-proxy==0.1.6 + Openjutsu's live-progress patch_proxy.py
    proxy_venv = work_dir / "venv-proxy"
    proxy_py = proxy_venv / "bin" / "python"
    proxy_bin = proxy_venv / "bin" / "comfy-api-proxy"
    if not proxy_bin.is_file():
        olog(f"Installing comfy-api-proxy=={PROXY_VERSION} -> {proxy_venv}")
        run_logged([uv_bin, "venv", "--python", sys.executable, str(proxy_venv)], env_log, timeout=120)
        run_logged([uv_bin, "pip", "install", "--python", str(proxy_py), f"comfy-api-proxy=={PROXY_VERSION}"],
                   env_log, timeout=300)
    run_logged([str(proxy_py), str(app_dir / "patch_proxy.py")], env_log, timeout=60)

    # 7. Node.js 22 + Next.js production build
    node_bin_dir = ensure_nodejs(work_dir, env_log)
    node_env = {**os.environ, "PATH": f"{node_bin_dir}:{os.environ.get('PATH', '')}", "NEXT_TELEMETRY_DISABLED": "1"}
    (app_dir / ".env.local").write_text(
        "COMFY_BASE_URL=http://127.0.0.1:8189\n"
        "COMFY_API_KEY=\n"
        "COMFYUI_URL=http://127.0.0.1:8188\n"
        "OPENJUTSU_PASSWORD=\n",
        encoding="utf-8",
    )
    head_sha = subprocess.check_output(["git", "-C", str(app_dir), "rev-parse", "HEAD"], text=True, timeout=15).strip()
    build_marker = work_dir / ".next_build_sha"
    if not (app_dir / ".next" / "BUILD_ID").is_file() or not build_marker.is_file() or build_marker.read_text().strip() != head_sha:
        olog("Building Openjutsu Next.js production app (`npm ci && npm run build`)...")
        run_logged([str(node_bin_dir / "npm"), "ci", "--no-audit", "--no-fund"], env_log,
                   env=node_env, cwd=app_dir, timeout=600)
        run_logged([str(node_bin_dir / "npm"), "run", "build"], env_log,
                   env=node_env, cwd=app_dir, timeout=600)
        build_marker.write_text(head_sha + "\n", encoding="utf-8")

    return {
        "node_bin_dir": str(node_bin_dir),
        "app_dir": str(app_dir),
        "proxy_bin": str(proxy_bin),
        "extra_model_paths": str(extra_paths_file),
    }


def parse_hf_resolve_url(url: str) -> tuple[str, str, str] | None:
    """Parse `https://huggingface.co/<repo_id>/(resolve|blob)/<revision>/<filename>`."""
    m = HF_RESOLVE_URL_RE.match((url or "").strip())
    if not m:
        return None
    return m.group(1), m.group(2), urllib.parse.unquote(m.group(3))


def _detect_system_ram_gib() -> float:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return 0.0


def ensure_hf_downloader(work_dir: Path) -> str | None:
    """Provision an isolated venv with `huggingface_hub` + `hf_xet` so parallel model downloads
    saturate multi-Gbps links from t=0 without racing concurrent `--system` package installs.
    """
    if os.environ.get("CT_OPENJUTSU_DISABLE_XET") == "1":
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    dl_venv = work_dir / "venv-hf-dl"
    dl_py = dl_venv / "bin" / "python"
    env_log = work_dir / "hf-dl-setup.log"
    if dl_py.is_file():
        chk = subprocess.run([str(dl_py), "-c", "import huggingface_hub, hf_xet"], capture_output=True, timeout=20)
        if chk.returncode == 0:
            return str(dl_py)
    uv_bin = shutil.which("uv")
    try:
        if uv_bin:
            if not dl_py.is_file():
                run_logged([uv_bin, "venv", "--python", sys.executable, str(dl_venv)], env_log, timeout=120)
            run_logged([uv_bin, "pip", "install", "--python", str(dl_py), "-U", "huggingface_hub", "hf_xet"],
                       env_log, timeout=180)
        else:
            if not dl_py.is_file():
                run_logged([sys.executable, "-m", "venv", str(dl_venv)], env_log, timeout=120)
            run_logged([str(dl_py), "-m", "pip", "install", "-q", "-U", "huggingface_hub", "hf_xet"],
                       env_log, timeout=300)
        return str(dl_py)
    except Exception as err:
        olog(f"[model] Isolated hf_xet venv unavailable ({err}); checking system Python...")
        chk = subprocess.run([sys.executable, "-c", "import huggingface_hub, hf_xet"], capture_output=True, timeout=20)
        return sys.executable if chk.returncode == 0 else None


def _stage_dir_progress_bytes(stage_dir: Path) -> int:
    best = 0
    if not stage_dir.is_dir():
        return 0
    for p in stage_dir.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                best = max(best, p.stat().st_size)
        except OSError:
            continue
    return best


def _download_via_hf_xet(url: str, dest: Path, name: str, dl_python: str, hf_tok: str | None) -> int:
    parsed = parse_hf_resolve_url(url)
    if not parsed:
        raise ValueError(f"Not a Hugging Face resolve URL: {url}")
    repo_id, revision, filename = parsed
    stage_dir = dest.parent / f".hf-stage-{dest.name}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    dl_env = {
        **os.environ,
        "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
    high_perf = os.environ.get("CT_HF_XET_HIGH_PERFORMANCE")
    if high_perf == "1" or (high_perf != "0" and _detect_system_ram_gib() >= 24.0):
        dl_env["HF_XET_HIGH_PERFORMANCE"] = "1"
    else:
        dl_env.pop("HF_XET_HIGH_PERFORMANCE", None)
    if hf_tok:
        dl_env["HF_TOKEN"] = hf_tok
        dl_env["HUGGING_FACE_HUB_TOKEN"] = hf_tok

    code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from huggingface_hub import hf_hub_download\n"
        "repo_id, filename, revision, stage_dir, dest = sys.argv[1:6]\n"
        "tok = os.environ.get('HF_TOKEN') or None\n"
        "p = Path(hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=stage_dir, token=tok)).resolve()\n"
        "os.replace(p, dest)\n"
    )
    t0 = last_log = time.monotonic()
    proc = subprocess.Popen(
        [dl_python, "-c", code, repo_id, filename, revision, str(stage_dir), str(dest)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=dl_env,
    )
    try:
        while proc.poll() is None:
            now = time.monotonic()
            if now - t0 > 1800:
                proc.kill()
                raise TimeoutError(f"hf_xet timed out after 1800s for {name}")
            if now - last_log >= 15:
                cur = _stage_dir_progress_bytes(stage_dir)
                if cur > 0:
                    mb_s = cur / 1e6 / max(now - t0, 0.1)
                    olog(f"[model]   {name}: {cur/1e9:.2f} GB @ {mb_s:.0f} MB/s (hf_xet)")
                last_log = now
            time.sleep(1.0)
        out = (proc.stdout.read() if proc.stdout else "").strip()
        if proc.returncode != 0 or not (dest.is_file() and dest.stat().st_size > 0):
            tail = out.splitlines()[-1] if out else f"exit code {proc.returncode}"
            raise RuntimeError(f"hf_xet failed ({tail})")
        shutil.rmtree(stage_dir, ignore_errors=True)
        return dest.stat().st_size
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def _probe_http_range(url: str, headers: dict[str, str]) -> tuple[int, bool]:
    probe_headers = {**headers, "Range": "bytes=0-0"}
    req = urllib.request.Request(url, headers=probe_headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status == 206:
            cr = resp.headers.get("Content-Range") or ""
            m = re.search(r"/(\d+)\s*$", cr)
            if m:
                return int(m.group(1)), True
        total = int(resp.headers.get("Content-Length") or 0)
        return total, False


def _download_via_http(
    url: str,
    dest: Path,
    headers: dict[str, str],
    name: str,
    *,
    parallel_threshold: int = 32 * 1024 * 1024,
    range_workers: int = 8,
) -> int:
    partial = dest.with_name(f".{dest.name}.{os.getpid()}.partial")
    t0 = time.monotonic()

    # Fast path 1: Multi-connection parallel HTTP Range (`os.pwrite`) for large files
    if not partial.exists():
        try:
            total, supports_range = _probe_http_range(url, headers)
        except Exception:
            total, supports_range = 0, False
        if supports_range and total >= parallel_threshold and range_workers > 1 and hasattr(os, "pwrite"):
            fd = os.open(partial, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
            try:
                os.ftruncate(fd, total)
                part_size = (total + range_workers - 1) // range_workers
                slices = [(i * part_size, min(total - 1, (i + 1) * part_size - 1))
                          for i in range(range_workers) if i * part_size < total]
                lock = threading.Lock()
                progress = [0, t0]  # [copied_bytes, last_log_monotonic]

                def _fetch_slice(bounds: tuple[int, int]) -> None:
                    start, end = bounds
                    cur = start
                    for sub_try in range(1, 4):
                        try:
                            req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={cur}-{end}"})
                            with urllib.request.urlopen(req, timeout=120) as resp:
                                if resp.status != 206:
                                    raise IOError(f"Expected HTTP 206 for Range request, got {resp.status}")
                                while cur <= end:
                                    chunk = resp.read(min(4 * 1024 * 1024, end - cur + 1))
                                    if not chunk:
                                        break
                                    os.pwrite(fd, chunk, cur)
                                    cur += len(chunk)
                                    with lock:
                                        progress[0] += len(chunk)
                                        now = time.monotonic()
                                        if now - progress[1] >= 15:
                                            mb_s = progress[0] / 1e6 / max(now - t0, 0.1)
                                            olog(f"[model]   {name}: {progress[0]/1e9:.2f}/{total/1e9:.2f} GB "
                                                 f"({progress[0]*100/total:.0f}%) @ {mb_s:.0f} MB/s (http-x{len(slices)})")
                                            progress[1] = now
                            if cur != end + 1:
                                raise IOError(f"Incomplete Range slice {cur}/{end + 1}")
                            return
                        except Exception:
                            if sub_try == 3:
                                raise
                            time.sleep(sub_try * 2)

                with ThreadPoolExecutor(max_workers=len(slices)) as pool:
                    futs = [pool.submit(_fetch_slice, b) for b in slices]
                    for fut in as_completed(futs):
                        fut.result()
            except Exception:
                os.close(fd)
                partial.unlink(missing_ok=True)
                raise
            else:
                os.close(fd)
                partial.replace(dest)
                return total

    # Fast path 2: Resumable single-stream HTTP download (never discards partial progress on reset)
    existing = partial.stat().st_size if partial.is_file() else 0
    req_headers = {**headers, "Range": f"bytes={existing}-"} if existing > 0 else dict(headers)
    req = urllib.request.Request(url, headers=req_headers)
    last_log = t0
    with urllib.request.urlopen(req, timeout=120) as resp:
        resumed = existing > 0 and resp.status == 206
        mode = "ab" if resumed else "wb"
        base_bytes = existing if resumed else 0
        rem = int(resp.headers.get("Content-Length") or 0)
        total = (base_bytes + rem) if rem > 0 else 0
        copied = base_bytes
        with partial.open(mode) as out:
            while True:
                chunk = resp.read(16 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                copied += len(chunk)
                now = time.monotonic()
                if now - last_log >= 15:
                    mb_s = (copied - base_bytes) / 1e6 / max(now - t0, 0.1)
                    pct = f" ({copied*100/total:.0f}%)" if total > 0 else ""
                    olog(f"[model]   {name}: {copied/1e9:.2f}/{total/1e9:.2f} GB{pct} @ {mb_s:.0f} MB/s")
                    last_log = now
    if total > 0 and copied != total:
        raise IOError(f"Incomplete read for {name}: got {copied} of {total} bytes")
    if copied == 0:
        raise IOError(f"Empty download for {name}")
    partial.replace(dest)
    return copied


def _download_single_model(item: dict[str, object], comfy_dir: Path, dl_python: str | None = None) -> dict[str, object]:
    directory = str(item["directory"])
    name = str(item["name"])
    urls = tuple(str(u) for u in item["urls"])  # type: ignore[arg-type]
    dest = validate_model_destination(directory, name, comfy_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        sz = dest.stat().st_size
        olog(f"[model] Cached ({sz/1e9:.2f} GB): {directory}/{name}")
        return {"model": f"{directory}/{name}", "status": "cached", "size": sz}

    hf_tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    last_err: Exception | None = None
    partial = dest.with_name(f".{dest.name}.{os.getpid()}.partial")
    stage_dir = dest.parent / f".hf-stage-{dest.name}"

    for url in urls:
        partial.unlink(missing_ok=True)
        shutil.rmtree(stage_dir, ignore_errors=True)
        headers = {"User-Agent": "Mozilla/5.0"}
        if hf_tok and "huggingface.co" in url:
            headers["Authorization"] = f"Bearer {hf_tok}"
        use_xet = bool(dl_python and parse_hf_resolve_url(url))
        for attempt in range(1, 4):
            t0 = time.monotonic()
            try:
                if use_xet and attempt <= 2:
                    assert dl_python is not None
                    copied = _download_via_hf_xet(url, dest, name, dl_python, hf_tok)
                    engine = "hf_xet"
                else:
                    copied = _download_via_http(url, dest, headers, name)
                    engine = "http"
                dt = max(time.monotonic() - t0, 0.1)
                olog(f"[model] Downloaded ({copied/1e9:.2f} GB in {dt:.1f}s, {copied/1e6/dt:.0f} MB/s, {engine}): {directory}/{name}")
                return {"model": f"{directory}/{name}", "status": "downloaded", "size": copied}
            except Exception as err:
                last_err = err
                olog(f"[model] Attempt {attempt}/3 failed for {name} ({url}): {err}")
                time.sleep(min(3 * attempt, 10))
        partial.unlink(missing_ok=True)
        shutil.rmtree(stage_dir, ignore_errors=True)
    raise RuntimeError(f"Failed to download {directory}/{name}: {last_err}")


def stage_models(
    comfy_dir: Path,
    profile: str = "full",
    workers: int = 3,
    work_dir: Path | None = None,
) -> list[dict[str, object]]:
    # Schedule largest multi-GB weights first (LPT) so tail stragglers never hold up readiness.
    items = sorted(models_for_profile(profile), key=lambda m: float(m.get("approx_gb", 0.0)), reverse=True)  # type: ignore[arg-type]
    free_gb = shutil.disk_usage(comfy_dir.parent if not comfy_dir.exists() else comfy_dir).free / 1e9
    olog(f"Staging {len(items)} model files (profile={profile}, free SSD={free_gb:.1f} GB)...")

    needs_hf = any(
        not (validate_model_destination(str(m["directory"]), str(m["name"]), comfy_dir).is_file())
        and any(parse_hf_resolve_url(str(u)) for u in m["urls"])  # type: ignore[attr-defined]
        for m in items
    )
    dl_python = ensure_hf_downloader(work_dir or (comfy_dir.parent / "openjutsu")) if needs_hf else None

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(_download_single_model, item, comfy_dir, dl_python): item for item in items}
        for fut in as_completed(future_map):
            results.append(fut.result())
    for p in ("/root/.cache/huggingface/xet", "/root/.cache/huggingface/hub"):
        shutil.rmtree(p, ignore_errors=True)
    total_gb = sum(int(r["size"]) for r in results) / 1e9
    olog(f"All {len(results)} model files ready on local SSD ({total_gb:.2f} GB total)")
    return results


DEFAULT_MODEL_SETTINGS: dict[str, object] = {
    "model": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "loras": [{"name": "minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors", "strength": 1}],
    "steps": 6,
    "textEncoder": "qwen3vl_32b_minimax_h3_int4_convrot.safetensors",
    "videoVae": "minimax_h3_video_vae_int8_convrot.safetensors",
    "audioVae": "minimax_h3_audio_vae_fp32.safetensors",
    "acceleration": "none",
}


def setup_check_payloads(profile: str = "full") -> dict[str, dict[str, object]]:
    """Return representative `/api/setup` payloads covering Openjutsu's workflows."""
    payloads: dict[str, dict[str, object]] = {
        "mask": {"kind": "mask", "hatFix": True, "margin": "normal", "target": "person"},
        "people": {"kind": "people", "resolution": "736p"},
        "fit": {"kind": "fit", "resolution": "736p"},
        "color": {
            "kind": "color",
            "settings": {
                "detail": -0.2,
                "tone": {
                    "mode": "Reinhard",
                    "exposure": 0.09,
                    "contrast": 0.84,
                    "pivot": 0.69,
                    "highlight_rolloff": 0.27,
                    "shadow_lift": 0.15,
                    "black_floor": 0.03,
                    "saturation": 1.05,
                    "strength": 0.2,
                },
            },
        },
        "simple": {
            "kind": "simple",
            "settings": {
                "role": "",
                "resolution": "736p",
                "motion": 768,
                "hatFix": True,
                "noLipSync": True,
                "margin": "normal",
                "target": "person",
                "seed": 0,
                "clipSeconds": 6,
            },
            "model": DEFAULT_MODEL_SETTINGS,
        },
        "advanced": {
            "kind": "advanced",
            "pictures": 2,
            "settings": {
                "roles": ["", ""],
                "resolution": "736p",
                "motion": 768,
                "hatFix": True,
                "noLipSync": True,
                "margin": "normal",
                "seed": 0,
                "clipSeconds": 6,
            },
            "model": DEFAULT_MODEL_SETTINGS,
        },
        "object": {
            "kind": "object",
            "picture": True,
            "settings": {
                "mode": "change",
                "target": "jacket",
                "changeTo": "leather jacket",
                "biggestOnly": True,
                "keepOutline": False,
                "resolution": "736p",
                "motion": 768,
                "margin": "normal",
                "seed": 0,
                "clipSeconds": 6,
            },
            "model": DEFAULT_MODEL_SETTINGS,
        },
        "montage": {
            "kind": "montage",
            "pictures": 2,
            "review": None,
            "settings": {
                "roles": ["", ""],
                "resolution": "736p",
                "motion": 768,
                "hatFix": True,
                "noLipSync": True,
                "margin": "normal",
                "seed": 0,
                "clipSeconds": 6,
            },
            "model": DEFAULT_MODEL_SETTINGS,
        },
    }
    if (profile or "full").strip().lower() == "full":
        payloads["face"] = {
            "kind": "face",
            "settings": {"style": "mouth", "seed": 0, "chunkSeconds": 6.5},
            "model": DEFAULT_MODEL_SETTINGS,
        }
    return payloads
