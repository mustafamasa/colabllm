"""Single-person LongCat-Video-Avatar 1.5 with authenticated Gradio hosting."""
import inspect
import json
import math
import os
import subprocess
import textwrap
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from longcat_video_app import install_sdpa
from avatar_sampling import sampling_settings, apply_sampling_mode, plan_segments


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None):
    """Upstream BSHD convention, native fused SDPA on Blackwell."""
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                            v.transpose(1, 2), dropout_p=dropout_p,
                                            scale=softmax_scale)
    return out.transpose(1, 2)


def install_avatar_sdpa():
    install_sdpa()
    from longcat_video.modules.attention import Attention as BaseAttention
    from longcat_video.modules.avatar import attention as avatar
    avatar.Attention._process_attn = BaseAttention._process_attn
    # Keep the upstream audio projection, frame grouping, RoPE, and output
    # projection intact; replace only its external FlashAttention call.
    source = textwrap.dedent(inspect.getsource(avatar.SingleStreamAttention._process_cross_attn))
    old = '        from flash_attn import flash_attn_func\n'
    if source.count(old) != 1:
        raise RuntimeError('Pinned Avatar attention source changed; SDPA adapter needs review')
    source = source.replace(old, '        flash_attn_func = _avatar_sdpa\n')
    namespace = {**vars(avatar), '_avatar_sdpa': flash_attn_func}
    exec(compile(source, '<avatar-audio-sdpa>', 'exec'), namespace)
    avatar.SingleStreamAttention._process_cross_attn = namespace['_process_cross_attn']


def load_pipeline():
    from transformers import AutoTokenizer, UMT5EncoderModel, WhisperModel, AutoFeatureExtractor
    from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    install_avatar_sdpa()
    root = Path(os.environ['CT_LONGCAT_MODEL_DIR'])
    base, model = root / 'LongCat-Video', root / 'LongCat-Video-Avatar-1.5'
    common = dict(torch_dtype=torch.bfloat16, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(base, subfolder='tokenizer', local_files_only=True)
    text_encoder = UMT5EncoderModel.from_pretrained(base, subfolder='text_encoder', **common)
    vae = AutoencoderKLWan.from_pretrained(base, subfolder='vae', **common)
    dit = LongCatVideoAvatarTransformer3DModel.from_pretrained(
        model, subfolder='base_model', cp_split_hw=[1, 1],
        enable_flashattn2=True, enable_flashattn3=False, enable_xformers=False, **common)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model, subfolder='scheduler')
    audio_encoder = WhisperModel.from_pretrained(model / 'whisper-large-v3', local_files_only=True).cuda().eval()
    extractor = AutoFeatureExtractor.from_pretrained(model / 'whisper-large-v3', local_files_only=True)
    for module in (text_encoder, vae, dit, audio_encoder):
        module.eval().requires_grad_(False)
    pipe = LongCatVideoAvatarPipeline(tokenizer, text_encoder, vae, scheduler, dit,
                                      audio_encoder, extractor, model_type='avatar-v1.5')
    dit.load_lora(str(model / 'lora/dmd_lora.safetensors'), 'dmd')
    pipe.to('cuda')
    dit.enable_loras(['dmd'])
    return pipe


def window_audio(embeddings, start, frames):
    offsets = torch.arange(5, device=embeddings.device) - 2
    indices = torch.arange(start, start + frames, device=embeddings.device)[:, None] + offsets[None]
    return embeddings[indices.clamp(0, len(embeddings)-1)][None].cuda()


def main():
    import gradio as gr
    import imageio_ffmpeg
    import soundfile as sf
    import uvicorn
    from PIL import Image
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from longcat_avatar_drive import DRIVE_ROOT, resolve_drive, resolve_input, list_drive
    from diffusers.utils import export_to_video, load_image

    work = Path(os.environ['CT_LONGCAT_WORK_DIR'])
    output_dir = Path(os.environ['CT_LONGCAT_OUTPUT_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(os.environ['CT_LONGCAT_REPO_DIR'])
    sample_audio, sample_image = repo / 'assets/avatar/single/man.mp3', repo / 'assets/avatar/single/man.png'
    sample_prompt = json.loads((repo / 'assets/avatar/single_example_1.json').read_text())['prompt']
    pipe = load_pipeline()
    lock = threading.Lock()
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    def prepare_audio(audio, seconds, target):
        subprocess.run([ffmpeg, '-y', '-loglevel', 'error', '-i', str(audio), '-vn',
                        '-ac', '1', '-ar', '16000', '-af', 'apad', '-t', str(seconds), str(target)], check=True)
        waveform, sr = sf.read(target, dtype='float32')
        if not np.isfinite(waveform).all() or np.max(np.abs(waveform)) < 1e-5:
            raise ValueError('Audio is empty or silent; upload clear speech')
        emb = pipe.get_audio_embedding(waveform, fps=25, device='cuda', sample_rate=sr, model_type='avatar-v1.5')
        if not torch.isfinite(emb).all():
            raise ValueError('Audio features contain invalid values')
        return emb

    # Test the actual Whisper -> audio-conditioned DiT -> VAE path before readiness.
    with torch.inference_mode():
        emb = prepare_audio(sample_audio, 1.0, work / 'smoke-audio.wav')
        smoke = pipe.generate_at2v(prompt=sample_prompt, height=256, width=256,
                                   num_frames=9, num_inference_steps=8, text_guidance_scale=1,
                                   audio_guidance_scale=1, use_distill=True,
                                   audio_emb=window_audio(emb, 0, 9),
                                   generator=torch.Generator(device='cuda').manual_seed(42))[0]
        export_to_video(smoke, str(work / 'smoke.mp4'), fps=25)
    del emb, smoke
    torch.cuda.empty_cache()
    print('LONGCAT_AVATAR_INFERENCE_OK whisper + 8-step audio-conditioned video', flush=True)

    from longcat_avatar_jobs import JobQueue

    def run_job(job_id, folder, request, update):
        audio, image, prompt = request['audio'], request['image'], request['prompt']
        mode, resolution = request['mode'], request['resolution']
        frames, segments, seed = request['frames'], request['segments'], request['seed']
        auto_duration = request['auto_duration']
        sampling = sampling_settings(request.get('sampling_mode','distilled'), request.get('steps'),
                                     request.get('text_guidance_scale'), request.get('audio_guidance_scale'))
        started = time.time()
        temp = work / 'jobs' / job_id
        temp.mkdir(parents=True,exist_ok=True)
        with lock, torch.inference_mode():
            try:
                apply_sampling_mode(pipe, sampling)
                source=audio
                audio_seconds=None
                if auto_duration:
                    source=temp/'source.wav'
                    subprocess.run([ffmpeg,'-y','-loglevel','error','-i',str(audio),
                                    '-vn','-ac','1','-ar','16000',str(source)],check=True)
                    audio_seconds=sf.info(source).duration
                parts,target_frames,duration=plan_segments(audio_seconds,frames,segments,auto_duration)
                segments=len(parts)
                update(segments=segments,duration_seconds=duration)
                covered=parts[0]+sum(n-13 for n in parts[1:])
                emb=prepare_audio(source,covered/25,temp/'speech.wav')
                generator=torch.Generator(device='cuda').manual_seed(seed)
                checkpoint=folder/'checkpoint.pt'
                next_segment=0
                latent=ref_latent=result=None
                if checkpoint.exists():
                    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    next_segment=saved['next_segment']
                    latent=saved['latent'].cuda()
                    ref_latent=saved['ref_latent'].cuda()
                    generator.set_state(saved['generator'])
                    from diffusers.utils import load_video
                    previous=load_video(saved['full_video'])
                for segment in range(next_segment,segments):
                    update(completed_segments=segment,message=f'Generating segment {segment+1}/{segments}')
                    audio_start=sum(parts[:segment])-13*segment
                    kwargs=dict(prompt=prompt,num_frames=parts[segment],num_inference_steps=sampling['steps'],
                                text_guidance_scale=sampling['text_guidance_scale'],
                                audio_guidance_scale=sampling['audio_guidance_scale'],use_distill=sampling['use_distill'],
                                output_type='both',generator=generator,
                                audio_emb=window_audio(emb,audio_start,parts[segment]))
                    if segment==0:
                        if mode=='Audio + Image + Prompt':
                            output,latent=pipe.generate_ai2v(image=load_image(image),resolution=resolution,**kwargs)
                        else:
                            height,width=(480,832) if resolution=='480p' else (768,1280)
                            output,latent=pipe.generate_at2v(height=height,width=width,**kwargs)
                        ref_latent=latent[:,:,:1].clone()
                    else:
                        if result is not None:
                            previous=[Image.fromarray((f*255).clip(0,255).astype(np.uint8)) for f in result]
                        output,latent=pipe.generate_avc(video=previous,video_latent=latent,
                            height=previous[0].height,width=previous[0].width,
                            num_cond_frames=13,use_kv_cache=True,offload_kv_cache=True,enhance_hf=False,
                            ref_latent=ref_latent,ref_img_index=10,mask_frame_range=3,**kwargs)
                    result=output[0]
                    full=folder/f'full-{segment:04d}.mp4'
                    export_to_video(result,str(full),fps=25)
                    emitted=sum(parts[:segment])-13*max(0,segment-1) if segment else 0
                    novel=result if segment==0 else result[13:]
                    novel=novel[:max(0,target_frames-emitted)]
                    export_to_video(novel,str(folder/f'clip-{segment:04d}.mp4'),fps=25)
                    temporary=folder/'checkpoint.tmp.pt'
                    torch.save(dict(next_segment=segment+1,latent=latent.cpu(),ref_latent=ref_latent.cpu(),
                                    generator=generator.get_state(),full_video=str(full)),temporary)
                    temporary.replace(checkpoint)
                    update(completed_segments=segment+1,message=f'Saved checkpoint for segment {segment+1}/{segments}')
                    for old in folder.glob('full-*.mp4'):
                        if old!=full:old.unlink()
                listing=temp/'clips.txt'
                listing.write_text(''.join("file '"+str(folder/f'clip-{i:04d}.mp4')+"'\n" for i in range(segments)))
                destination=resolve_drive(request['output_dir'],directory=True,create=True) if request.get('output_dir') else output_dir
                path=destination/(job_id+'.mp4')
                subprocess.run([ffmpeg,'-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),
                                '-i',str(temp/'speech.wav'),'-map','0:v:0','-map','1:a:0',
                                '-c:v','copy','-c:a','aac','-t',str(duration),str(path)],check=True)
                import av
                with av.open(str(path)) as media:
                    width,height=media.streams.video[0].width,media.streams.video[0].height
                metadata=dict(job_id=job_id,model='LongCat-Video-Avatar-1.5',precision='BF16',mode=mode,
                              prompt=prompt,seed=seed,**sampling,fps=25,frames=target_frames,width=width,height=height,
                              segments=segments,auto_duration=bool(auto_duration),audio_duration_seconds=audio_seconds,
                              duration_seconds=round(duration,2),elapsed_seconds=round(time.time()-started,1),drive_output=str(path))
                path.with_suffix('.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
                return dict(path=str(path),metadata=metadata)
            finally:
                pipe._clear_cache()
                torch.cuda.empty_cache()
                import shutil
                shutil.rmtree(temp,ignore_errors=True)

    jobs=JobQueue(output_dir.parent/'jobs',run_job)

    def submit(audio,image,prompt,mode,resolution,frames,segments,seed,auto_duration=True,output_folder=None,
               sampling_mode='distilled',steps=None,text_guidance_scale=None,audio_guidance_scale=None):
        if not audio or not prompt.strip():raise gr.Error('Upload audio and enter a prompt')
        if mode not in ('Audio + Image + Prompt','Audio + Prompt') or resolution not in ('480p','720p'):
            raise gr.Error('Invalid generation mode or resolution')
        if mode=='Audio + Image + Prompt' and not image:raise gr.Error('Upload a reference image')
        frames,segments,seed=int(frames),int(segments),int(seed)
        if not (17<=frames<=161 and frames%4==1 and 1<=segments<=20):raise gr.Error('Invalid manual duration parameters')
        if seed<0:seed=int.from_bytes(os.urandom(4),'little')
        sampling=sampling_settings(sampling_mode,steps,text_guidance_scale,audio_guidance_scale)
        request=dict(**sampling,prompt=prompt,mode=mode,resolution=resolution,frames=frames,segments=segments,
                     seed=seed,auto_duration=bool(auto_duration))
        if output_folder:
            request['output_dir']=str(resolve_drive(output_folder,directory=True,create=True))
        try:job_id=jobs.submit(request,audio,image)
        except ValueError as error:raise gr.Error(str(error))
        return job_id,'Job submitted; you can close this page. Job ID: '+job_id

    def output_path(state):
        path=Path(state['result']['path']).resolve()
        if not any(path.is_relative_to(root.resolve()) for root in (output_dir,DRIVE_ROOT)):
            raise ValueError('Output path is outside the allowed directories')
        if not path.is_file():raise ValueError('Output video file does not exist')
        return path

    def display_path(state):
        path=output_path(state)
        if path.is_relative_to(output_dir.resolve()):return str(path)
        # Publish only this job's output to Gradio's allowlisted local cache,
        # rather than exposing the whole mounted Drive through its file API.
        import shutil
        cache=work/'published'/path.name
        cache.parent.mkdir(exist_ok=True)
        if not cache.exists() or cache.stat().st_size!=path.stat().st_size:
            temporary=cache.with_name(cache.name+'.'+uuid.uuid4().hex+'.tmp')
            shutil.copy2(path,temporary)
            temporary.replace(cache)
        return str(cache)

    def inspect_job(job_id):
        if not job_id:return 'No job selected yet',gr.skip(),gr.skip(),gr.skip()
        try:state=jobs.get(job_id)
        except (ValueError,FileNotFoundError):return 'Job not found',gr.skip(),gr.skip(),gr.skip()
        if state['status']=='completed':
            result=state['result']
            try:
                published=display_path(state)
            except (OSError,ValueError,KeyError) as error:
                return f'Output file unavailable: {error}',gr.skip(),gr.skip(),result.get('metadata')
            return 'Generation complete',published,published,result['metadata']
        progress=state.get('message',state['status'])
        if state['status']=='failed':progress='Job failed: '+progress
        return progress,gr.skip(),gr.skip(),state

    with gr.Blocks(title='LongCat Avatar 1.5') as demo:
        gr.Markdown('# LongCat Avatar 1.5\nUpload vocal audio and the person speaks or sings along with it. Defaults to 8 steps at 25 fps; results are saved to Google Drive automatically.')
        with gr.Row():
            with gr.Column():
                audio = gr.Audio(type='filepath', sources=['upload'], label='Vocal audio (clean dry vocals recommended)')
                image = gr.Image(type='filepath', label='Reference person image')
                prompt = gr.Textbox(label='Person, scene and action description', lines=4, value=sample_prompt)
                mode = gr.Dropdown(['Audio + Image + Prompt', 'Audio + Prompt'], value='Audio + Image + Prompt', label='Generation mode')
                resolution = gr.Dropdown(['480p', '720p'], value='480p', label='Resolution')
                auto_duration = gr.Checkbox(value=True, label='Generate for the full audio length (automatic segments)')
                duration_info = gr.Markdown('Upload audio to see its full length and the estimated number of segments.')
                with gr.Accordion('Manual duration and random seed', open=False):
                    frames = gr.Slider(17, 161, value=93, step=4, label='Manual mode: frames per segment', interactive=False)
                    segments = gr.Slider(1, 20, value=1, step=1, label='Manual mode: number of segments', interactive=False)
                    seed = gr.Number(value=42, precision=0, label='Random seed (-1 for random)')
                with gr.Accordion('Lip sync and sampling settings', open=False):
                    sampling_mode = gr.Dropdown(choices=[('Fast - 8-step distilled','distilled'),('Standard - non-distilled, adjustable CFG','standard')],
                                               value='distilled', label='Sampling mode')
                    steps = gr.Slider(8,100,value=8,step=1,label='Inference steps',interactive=False)
                    text_cfg = gr.Slider(1,10,value=1,step=0.1,label='Text CFG',interactive=False)
                    audio_cfg = gr.Slider(1,10,value=1,step=0.1,label='Audio CFG',interactive=False)
                    gr.Markdown('Standard mode is slower. Audio CFG controls audio guidance; lowering it may hurt sync. It does not directly control mouth amplitude.')
                button = gr.Button('Submit generation job', variant='primary')
            with gr.Column():
                job_id = gr.Textbox(label='Job ID (paste it here after reopening the page)')
                history = gr.Dropdown(label='Job history', choices=jobs.recent())
                status = gr.Markdown('After submission the server generates in the background; closing the page does not affect the job.')
                video = gr.Video(label='Result')
                download = gr.File(label='Download video with audio')
                metadata = gr.JSON(label='Generation info')
        gr.Examples([[str(sample_audio), str(sample_image), sample_prompt]], [audio, image, prompt], label='Official example')
        def describe_duration(path, automatic):
            if not automatic:
                return 'Manual mode trims the audio to the configured frames and segments.'
            if not path:
                return 'Upload audio to see its full length and the estimated number of segments.'
            try:
                seconds = sf.info(path).duration
                parts, _, _ = plan_segments(seconds)
                return f'Audio is about **{seconds:.2f} s**; **{len(parts)} segments** will be generated automatically, covering the full audio.'
            except Exception as error:
                return str(error)
        audio.change(describe_duration, [audio,auto_duration], duration_info)
        auto_duration.change(describe_duration, [audio,auto_duration], duration_info)
        auto_duration.change(lambda enabled: [gr.update(interactive=not enabled)]*2,
                             auto_duration, [frames,segments])
        def change_sampling(value):
            settings=sampling_settings(value)
            return [gr.update(value=settings[key],interactive=value=='standard')
                    for key in ('steps','text_guidance_scale','audio_guidance_scale')]
        def submit_ui(audio,image,prompt,mode,resolution,frames,segments,seed,auto_duration=True,
                      sampling_mode='distilled',steps=None,text_cfg=None,audio_cfg=None):
            return submit(audio,image,prompt,mode,resolution,frames,segments,seed,auto_duration,
                          None,sampling_mode,steps,text_cfg,audio_cfg)
        button.click(submit_ui,
                     [audio,image,prompt,mode,resolution,frames,segments,seed,auto_duration,sampling_mode,steps,text_cfg,audio_cfg],
                     [job_id,status],api_name='submit',concurrency_limit=4)
        timer=gr.Timer(3)
        timer.tick(inspect_job,job_id,[status,video,download,metadata],api_name=False,queue=False)
        history.change(lambda value:value,history,job_id)
        demo.load(lambda:gr.update(choices=jobs.recent(),value=jobs.recent()[0] if jobs.recent() else None),
                  outputs=history)
        sampling_mode.change(change_sampling,sampling_mode,[steps,text_cfg,audio_cfg])
    api = FastAPI()

    @api.get('/health')
    def health():
        return dict(status='ok', model='LongCat-Video-Avatar-1.5', precision='BF16',
                    gpu=torch.cuda.get_device_name(0), inference_verified=True,
                    active_job=jobs.active,draining=jobs.draining,sampling_modes=['distilled','standard'])

    @api.post('/jobs',status_code=202)
    def create_job(body: dict):
        try:
            audio=resolve_input(body.get('audio'))
            image=resolve_input(body['image']) if body.get('image') else None
            automatic=body.get('auto_duration',True)
            if not isinstance(automatic,bool):raise ValueError('auto_duration must be a boolean')
            job_id,_=submit(str(audio),str(image) if image else None,str(body.get('prompt','')),
                'Audio + Image + Prompt' if image else 'Audio + Prompt',body.get('resolution','480p'),
                body.get('frames',93),body.get('segments',1),body.get('seed',42),automatic,body.get('output_dir'),
                body.get('sampling_mode','distilled'),body.get('steps'),
                body.get('text_guidance_scale'),body.get('audio_guidance_scale'))
            return {'job_id':job_id}
        except (ValueError,OSError,gr.Error) as error:raise HTTPException(400,str(error))

    @api.get('/drive/files')
    def drive_files(path: str='drive://'):
        try:return {'files':list_drive(path)}
        except (ValueError,OSError) as error:raise HTTPException(400,str(error))

    @api.get('/jobs/{job_id}')
    def get_job(job_id: str):
        try:return jobs.get(job_id)
        except (ValueError,FileNotFoundError):raise HTTPException(404,'Job not found')

    @api.get('/jobs/{job_id}/video')
    def job_video(job_id: str):
        state=get_job(job_id)
        if state['status']!='completed':raise HTTPException(409,'Job not finished yet')
        try:path=output_path(state)
        except ValueError as error:raise HTTPException(404,str(error))
        return FileResponse(path,media_type='video/mp4',filename=path.name)

    demo.queue(max_size=8, default_concurrency_limit=1)
    host = os.environ.get('CT_PUBLIC_HOSTNAME','')
    api = gr.mount_gradio_app(api, demo, path='/', allowed_paths=[str(output_dir), str(work/'published'), str(sample_audio), str(sample_image)],
                             root_path=f'https://{host}' if host else None)
    # SIGTERM drains accepted jobs before shutting down the HTTP server. A hard
    # stop can resume from the most recent per-segment checkpoint on startup.
    import signal
    from contextlib import contextmanager
    class JobServer(uvicorn.Server):
        @contextmanager
        def capture_signals(self):
            def shutdown(signum,frame):
                jobs.drain()
                if jobs.shutdown_ready.is_set():self.should_exit=True
            previous={sig:signal.signal(sig,shutdown) for sig in (signal.SIGINT,signal.SIGTERM)}
            try:yield
            finally:
                for sig,handler in previous.items():signal.signal(sig,handler)
    server=JobServer(uvicorn.Config(api,host='127.0.0.1',port=int(os.environ.get('CT_SERVICE_PORT','7860'))))
    def await_shutdown():
        jobs.shutdown_ready.wait()
        server.should_exit=True
    jobs.start()
    threading.Thread(target=await_shutdown,daemon=True).start()
    try:server.run()
    finally:
        jobs.stop.set()
        jobs.thread.join()



if __name__ == '__main__':
    main()
