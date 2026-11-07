"""Single-GPU LongCat-Video UI; embedded into the self-contained Colab notebook."""
import json
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def install_sdpa():
    """Use native fused SDPA instead of the upstream CUDA-12.4 FlashAttention build.

    Cross attention uses packed, variable-length text; split each sample before
    calling SDPA so padding and neighbouring prompts never enter its attention.
    This adapter is intentionally single GPU, matching the Colab deployment.
    """
    from longcat_video.modules.attention import Attention, MultiHeadCrossAttention

    def self_attention(self, q, k, v, shape):
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)

    def cross_attention(self, x, cond, kv_seqlen):
        batch, tokens, channels = x.shape
        q = self.q_norm(self.q_linear(x).view(1, -1, self.num_heads, self.head_dim))
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        k = self.k_norm(k)
        outputs, offset = [], 0
        for i, length in enumerate(kv_seqlen):
            qi = q[:, i * tokens:(i + 1) * tokens].transpose(1, 2)
            ki = k[:, offset:offset + length].transpose(1, 2)
            vi = v[:, offset:offset + length].transpose(1, 2)
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                out = F.scaled_dot_product_attention(qi, ki, vi, dropout_p=0.0)
            outputs.append(out.transpose(1, 2))
            offset += length
        return self.proj(torch.cat(outputs, dim=1).reshape(batch, tokens, channels))

    Attention._process_attn = self_attention
    MultiHeadCrossAttention._process_cross_attn = cross_attention


def load_pipeline():
    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel

    if not torch.cuda.is_available():
        raise RuntimeError("LongCat-Video requires a CUDA GPU")
    install_sdpa()
    model = os.environ["CT_LONGCAT_MODEL_DIR"]
    common = dict(torch_dtype=torch.bfloat16, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, subfolder="tokenizer", local_files_only=True)
    text_encoder = UMT5EncoderModel.from_pretrained(model, subfolder="text_encoder", **common)
    vae = AutoencoderKLWan.from_pretrained(model, subfolder="vae", **common)
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        model, subfolder="dit", cp_split_hw=[1, 1],
        enable_flashattn2=False, enable_flashattn3=False, enable_xformers=False,
        **common,
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder="scheduler")
    for module in (text_encoder, vae, dit):
        module.eval().requires_grad_(False)
    pipe = LongCatVideoPipeline(tokenizer, text_encoder, vae, scheduler, dit)
    pipe.dit.load_lora(str(Path(model) / "lora/cfg_step_lora.safetensors"), "cfg_step_lora")
    pipe.to("cuda")
    return pipe


def main():
    import gradio as gr
    import uvicorn
    from fastapi import FastAPI
    from diffusers.utils import export_to_video, load_image, load_video

    output_dir = Path(os.environ["CT_LONGCAT_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline()
    lock = threading.Lock()
    # A real inference pass is the readiness gate, not just an HTTP listener.
    with torch.inference_mode():
        smoke = pipe.generate_t2v(
            prompt="A small orange cat walking through a sunny garden.",
            height=256, width=256, num_frames=9, num_inference_steps=1,
            guidance_scale=1.0, generator=torch.Generator(device="cuda").manual_seed(42),
        )[0]
    smoke_path = Path(os.environ["CT_LONGCAT_WORK_DIR"]) / "smoke.mp4"
    export_to_video(smoke, str(smoke_path), fps=15)
    print(f"LONGCAT_INFERENCE_OK frames={len(smoke)} output={smoke_path}", flush=True)
    del smoke
    torch.cuda.empty_cache()

    def generate(task, prompt, negative, image, video, resolution, width, height,
                 frames, steps, guidance, distill, seed, segments):
        if not prompt.strip():
            raise gr.Error("Enter a video description")
        if task == "Image to Video" and not image:
            raise gr.Error("Upload a reference image")
        if task == "Video Continuation" and not video:
            raise gr.Error("Upload a reference video")
        width, height, frames = int(width), int(height), int(frames)
        if not (256 <= width <= 1024 and 256 <= height <= 1024 and 17 <= frames <= 161):
            raise gr.Error("Width/height must be 256-1024 and frames 17-161")
        if width % 16 or height % 16 or frames % 4 != 1:
            raise gr.Error("Width/height must be multiples of 16 and frames must be 4n+1")
        if task not in ("Text to Video", "Image to Video", "Video Continuation") or resolution not in ("480p", "720p"):
            raise gr.Error("Choose a valid generation mode and resolution")
        if not (1 <= int(segments) <= 10 and 1 <= int(steps) <= 100 and 1 <= float(guidance) <= 10):
            raise gr.Error("Segments, steps or guidance scale out of range")
        seed = int(seed)
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "little")
        started = time.time()
        with lock, torch.inference_mode():
            try:
                if distill:
                    pipe.dit.enable_loras(["cfg_step_lora"])
                generator = torch.Generator(device="cuda").manual_seed(seed)
                kwargs = dict(prompt=prompt, negative_prompt=None if distill else negative,
                              num_frames=frames, num_inference_steps=16 if distill else int(steps),
                              guidance_scale=1.0 if distill else float(guidance),
                              use_distill=distill, generator=generator)
                if task == "Text to Video":
                    if int(segments) > 1:
                        from PIL import Image
                        # Continuation uses upstream aspect buckets: select the same
                        # bucket for the first segment so every clip can be joined.
                        height, width = pipe.get_condition_shape(
                            Image.new("RGB", (width, height)), resolution,
                            scale_factor_spatial=pipe.vae_scale_factor_spatial * 2,
                        )
                    result = pipe.generate_t2v(height=height, width=width, **kwargs)[0]
                elif task == "Image to Video":
                    result = pipe.generate_i2v(image=load_image(image), resolution=resolution, **kwargs)[0]
                else:
                    import cv2
                    cap = cv2.VideoCapture(video)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 15
                    cap.release()
                    source = load_video(video)[::max(1, round(fps / 15))]
                    if len(source) < 13:
                        raise gr.Error("The reference video needs at least 13 frames (sampled at 15 fps)")
                    result = pipe.generate_vc(video=source, resolution=resolution, num_cond_frames=13,
                                              use_kv_cache=True, offload_kv_cache=True,
                                              enhance_hf=not distill, **kwargs)[0]
                clips = [result]
                for _ in range(1, int(segments)):
                    from PIL import Image
                    source = [Image.fromarray((frame * 255).clip(0, 255).astype(np.uint8)) for frame in result[-13:]]
                    result = pipe.generate_vc(video=source, resolution=resolution, num_cond_frames=13,
                                              use_kv_cache=True, offload_kv_cache=True,
                                              enhance_hf=not distill, **kwargs)[0]
                    clips.append(result[13:])
                result = np.concatenate(clips, axis=0)
                name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
                path = output_dir / f"{name}.mp4"
                export_to_video(result, str(path), fps=15)
                metadata = dict(task=task, prompt=prompt, negative_prompt=negative, seed=seed,
                                frames=len(result), fps=15, resolution=resolution,
                                width=int(result.shape[2]), height=int(result.shape[1]),
                                steps=16 if distill else int(steps), guidance=1.0 if distill else guidance,
                                distill=distill, segments=int(segments), elapsed_seconds=round(time.time()-started, 1))
                path.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
                return str(path), str(path), metadata
            finally:
                pipe.dit.disable_all_loras()
                torch.cuda.empty_cache()

    with gr.Blocks(title="LongCat-Video") as demo:
        gr.Markdown("# LongCat-Video\nText to video · image to video · video continuation. Results are saved to Google Drive automatically.")
        with gr.Row():
            with gr.Column():
                task = gr.Dropdown(["Text to Video", "Image to Video", "Video Continuation"], value="Text to Video", label="Generation mode")
                prompt = gr.Textbox(label="Video description", lines=4, placeholder="Describe the scene, character actions and camera movement (English recommended)")
                negative = gr.Textbox(label="Negative prompt", value="blurry, low quality, static, subtitles")
                image = gr.Image(type="filepath", label="Reference image (image to video)")
                video = gr.Video(label="Reference video (video continuation)")
                distill = gr.Checkbox(value=True, label="Fast mode (16 steps)")
                with gr.Accordion("Generation parameters", open=False):
                    resolution = gr.Dropdown(["480p", "720p"], value="480p", label="Resolution for image / continuation / multi-segment generation")
                    width = gr.Slider(256, 1024, value=832, step=16, label="Text-to-video width")
                    height = gr.Slider(256, 1024, value=480, step=16, label="Text-to-video height")
                    frames = gr.Slider(17, 161, value=93, step=4, label="Frames per segment (15 fps)")
                    steps = gr.Slider(1, 100, value=50, step=1, label="Sampling steps (standard mode)")
                    guidance = gr.Slider(1, 10, value=4, step=0.5, label="Guidance scale (standard mode)")
                    seed = gr.Number(value=42, precision=0, label="Random seed (-1 for random)")
                    segments = gr.Slider(1, 10, value=1, step=1, label="Number of consecutive segments")
                submit = gr.Button("Generate video", variant="primary")
            with gr.Column():
                result = gr.Video(label="Result")
                download = gr.File(label="Download video")
                metadata = gr.JSON(label="Generation info")
        submit.click(generate, [task, prompt, negative, image, video, resolution, width, height,
                               frames, steps, guidance, distill, seed, segments],
                     [result, download, metadata], api_name="generate", concurrency_limit=1)
    api = FastAPI()

    @api.get('/health')
    def health():
        return {"status": "ok", "model": "LongCat-Video", "gpu": torch.cuda.get_device_name(0),
                "inference_verified": True}

    demo.queue(max_size=8, default_concurrency_limit=1)
    public_host = os.environ.get("CT_PUBLIC_HOSTNAME", "")
    api = gr.mount_gradio_app(
        api, demo, path="/", allowed_paths=[str(output_dir)],
        root_path=f"https://{public_host}" if public_host else None,
    )
    uvicorn.run(api, host="127.0.0.1", port=int(os.environ.get("CT_SERVICE_PORT", "7860")))


if __name__ == '__main__':
    main()
