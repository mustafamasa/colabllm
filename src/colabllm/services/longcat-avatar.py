#!/usr/bin/env python3
"""LongCat-Video-Avatar 1.5 on Colab G4, with Caddy-protected Gradio UI.

  colabllm longcat-avatar up --domain
  colabllm longcat-avatar info | status | logs -f | down
  colabllm longcat-avatar up -i 2 --domain --gpu G4
  colabllm longcat-avatar generate --audio speech.wav --image person.png --prompt "A person speaking."
  colabllm longcat-avatar job JOB_ID --wait -o result.mp4
  colabllm longcat-avatar ls-drive "assets"
  colabllm longcat-avatar generate --audio-drive "assets/speech.wav" --image-drive "assets/person.png" --prompt "A person speaking." --drive-output "renders"
"""
import argparse
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from colabsvc import Service, State, main  # noqa: E402
from avatar_sampling import sampling_settings

SERVICE = Service(
    name="longcat-avatar",
    notebook="longcat-avatar.ipynb",
    service_port=7860,
    public_port=18080,
    auth="caddy",
    gpu="G4",
    ready_timeout=5400,
)


def is_drive_path(value):
    return bool(value and (value == "/content/drive/MyDrive" or value.startswith(("drive://", "gdrive://", "/content/drive/MyDrive/"))))


def drive_spec(value):
    return value if is_drive_path(value) else "drive://" + value.lstrip("/")


from cli_support import Deadline, positive_seconds, nonnegative_seconds, SameOriginRedirect, resolve_endpoint


class AvatarClient:
    def __init__(self, url, token):
        self.url = url.rstrip("/")
        self.headers = {"Authorization": "Bearer " + token, "User-Agent": "Mozilla/5.0"}
        self.opener = urllib.request.build_opener(SameOriginRedirect())

    def open(self, path, data=None, headers=None, timeout=180):
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={**self.headers, **(headers or {})})
        return self.opener.open(req, timeout=timeout)

    def upload(self, filename):
        path = Path(filename).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"File not found: {path}")
        boundary = "avatar-" + uuid.uuid4().hex
        name = path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as body:
            body.write((f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
                        f'filename="{name}"\r\nContent-Type: {mime}\r\n\r\n').encode())
            with path.open("rb") as source:
                shutil.copyfileobj(source, body)
            body.write(f"\r\n--{boundary}--\r\n".encode())
            length = body.tell()
            body.seek(0)
            with self.open("/gradio_api/upload", body, {
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Content-Length": str(length),
            }) as response:
                uploaded = json.load(response)
        if not isinstance(uploaded, list) or not uploaded or not isinstance(uploaded[0], str):
            raise RuntimeError("Unexpected upload response format")
        return {"path": uploaded[0], "orig_name": path.name, "meta": {"_type": "gradio.FileData"}}

    def source(self, value):
        return {"drive": value} if is_drive_path(value) else {"upload": self.upload(value)["path"]}

    def submit(self, audio, image, prompt, resolution, frames, segments, seed, full_audio, output_dir=None, sampling=None):
        data = dict(audio=self.source(audio), image=self.source(image) if image else None,
                    prompt=prompt, resolution=resolution, frames=frames, segments=segments,
                    seed=seed, auto_duration=full_audio)
        if sampling:
            data.update(sampling)
        if output_dir:
            data["output_dir"] = drive_spec(output_dir)
        with self.open("/jobs", json.dumps(data).encode(), {"Content-Type": "application/json"}) as response:
            result = json.load(response)
        if not isinstance(result.get("job_id"), str):
            raise RuntimeError("Submit response is missing the job ID")
        return result["job_id"]

    def list_drive(self, path):
        endpoint = "/drive/files?" + urllib.parse.urlencode({"path": drive_spec(path)})
        for attempt in range(3):
            try:
                with self.open(endpoint, timeout=15) as response:
                    return json.load(response)
            except (urllib.error.URLError, OSError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 502, 503, 504, 530):
                    raise
                if attempt == 2:
                    raise
                print("Drive directory query failed temporarily; retrying...", file=sys.stderr, flush=True)
                time.sleep(1)

    def job(self, job_id, timeout=30):
        with self.open("/jobs/" + urllib.parse.quote(job_id, safe=""), timeout=timeout) as response:
            return json.load(response)

    def wait(self, job_id, interval=3, timeout=0):
        deadline = Deadline(timeout, f"Timed out waiting; the server-side job keeps running: {job_id}")
        previous = None
        while True:
            try:
                state = self.job(job_id, timeout=deadline.remaining(30))
            except (urllib.error.URLError, OSError) as error:
                if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 502, 503, 504, 530):
                    raise
                message = "Connection interrupted, reconnecting; the server-side job is not affected"
            else:
                message = state.get("message", state["status"])
                if state["status"] == "completed":
                    return state
                if state["status"] in ("failed", "cancelled"):
                    raise RuntimeError(f"Job {state['status']}: {message}")
            if message != previous:
                print(message, file=sys.stderr, flush=True)
                previous = message
            deadline.sleep(interval)

    def download(self, state, destination):
        if state["status"] != "completed":
            raise ValueError("Job not finished yet; use --wait to wait for completion before downloading")
        output = Path(destination).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        path = "/jobs/" + urllib.parse.quote(state["id"], safe="") + "/video"
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix="."+output.name,
                                             suffix=".part", delete=False) as target:
                temporary = Path(target.name)
                with self.open(path, timeout=180) as response:
                    shutil.copyfileobj(response, target)
            temporary.replace(output)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return str(output)


def custom_parser(parser=None, commands=None):
    parser = parser or argparse.ArgumentParser(prog="colabllm longcat-avatar", description="Avatar 1.5: submit generations, query background jobs, download videos")
    commands = commands or parser.add_subparsers(dest="cmd", required=True)
    generate = commands.add_parser("generate", help="submit a generation job; by default generates for the full audio length")
    audio = generate.add_mutually_exclusive_group(required=True)
    audio.add_argument("--audio", help="local audio file or drive:// path")
    audio.add_argument("--audio-drive", help="audio path inside MyDrive, no local download needed")
    image = generate.add_mutually_exclusive_group()
    image.add_argument("--image", help="local reference image or drive:// path; optional")
    image.add_argument("--image-drive", help="reference image path inside MyDrive")
    generate.add_argument("--drive-output", "--output-drive", help="save the generated video to this directory inside MyDrive")
    prompt = generate.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="description of the person, scene and actions")
    prompt.add_argument("--prompt-file", help="read the description from a UTF-8 text file")
    generate.add_argument("--resolution", choices=("480p", "720p"), default="480p")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--sampling-mode", choices=("distilled", "standard"), default="distilled",
                          help="distilled: fast 8-step mode; standard: DMD LoRA off, CFG adjustable")
    generate.add_argument("--steps", type=int, help="inference steps; fixed at 8 in fast mode, default 50 in standard mode")
    generate.add_argument("--text-cfg", type=float, help="text CFG; default 4.0 in standard mode")
    generate.add_argument("--audio-cfg", type=float, help="audio CFG; default 3.0 in standard mode, not a mouth-amplitude parameter")
    generate.add_argument("--frames", type=int, help="frames per segment in manual mode (4n+1 between 17 and 161)")
    generate.add_argument("--segments", type=int, help="number of segments in manual mode (1-20); omit to cover the full audio")
    job = commands.add_parser("job", help="query a job; --wait keeps waiting")
    job.add_argument("job_id")
    download = commands.add_parser("download", help="download the video of a finished job")
    download.add_argument("job_id")
    listing = commands.add_parser("ls-drive", help="list a Google Drive directory")
    listing.add_argument("path", nargs="?", default="drive://")
    for command in (generate, job, download):
        command.add_argument("--wait", action="store_true", help="wait for completion; Ctrl+C only stops waiting")
        command.add_argument("-o", "--output", help="download to a local file; generate/job wait for completion automatically")
        command.add_argument("--poll", type=positive_seconds, default=3, help="polling interval in seconds, default 3")
        command.add_argument("--timeout", type=nonnegative_seconds, default=0, help="maximum wait in seconds; 0 means unlimited")
    for command in (generate, job, download, listing):
        command.add_argument("--url", help="override the service URL; LONGCAT_AVATAR_URL also works")
        command.add_argument("--api-token", "--token", dest="token", help="API token; LONGCAT_AVATAR_API_TOKEN or LONGCAT_AVATAR_TOKEN also work")
    return parser


def run_custom(args):
    if getattr(args, "poll", 3) <= 0 or getattr(args, "timeout", 0) < 0:
        raise ValueError("--poll must be greater than 0 and --timeout cannot be negative")
    explicit_token = args.token or os.environ.get('LONGCAT_AVATAR_API_TOKEN') or os.environ.get('LONGCAT_AVATAR_TOKEN')
    url, token = resolve_endpoint(State(SERVICE).load(), args.url, explicit_token,
                                  url_env='LONGCAT_AVATAR_URL')
    client = AvatarClient(url, token)
    if args.cmd == "ls-drive":
        print(json.dumps(client.list_drive(args.path), ensure_ascii=False, indent=2))
        return
    if args.output and is_drive_path(args.output):
        raise ValueError("-o is for local downloads; use --drive-output to choose a Drive directory when generating")
    if args.cmd == "generate":
        args.audio = drive_spec(args.audio_drive) if args.audio_drive else args.audio
        args.image = drive_spec(args.image_drive) if args.image_drive else args.image
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8") if args.prompt_file else args.prompt
        if not prompt.strip():
            raise ValueError("The description cannot be empty")
        for filename in (args.audio, args.image):
            if filename and not is_drive_path(filename) and not Path(filename).expanduser().is_file():
                raise ValueError(f"File not found: {filename}")
        frames, segments = args.frames if args.frames is not None else 93, args.segments if args.segments is not None else 1
        if not (17 <= frames <= 161 and frames % 4 == 1 and 1 <= segments <= 20):
            raise ValueError("Manual frame count must be 4n+1 between 17 and 161, and segments must be 1-20")
        sampling = sampling_settings(args.sampling_mode, args.steps, args.text_cfg, args.audio_cfg)
        job_id = client.submit(args.audio, args.image, prompt, args.resolution, frames, segments,
                               args.seed, args.frames is None and args.segments is None, args.drive_output, sampling)
        if not args.wait and not args.output:
            print(json.dumps({"job_id": job_id}, ensure_ascii=False))
            return
        print(f"Job submitted: {job_id}", file=sys.stderr, flush=True)
    else:
        job_id = args.job_id
    should_wait = args.wait or (args.output and args.cmd != "download") or args.cmd == "generate"
    state = client.wait(job_id, args.poll, args.timeout) if should_wait else client.job(job_id)
    if args.output or args.cmd == "download":
        state["local_output"] = client.download(state, args.output or (job_id+".mp4"))
    print(json.dumps(state, ensure_ascii=False, indent=2))


def configure_cli(parser, commands, up):
    custom_parser(parser, commands)


if __name__ == "__main__":
    from cli_support import run_cli
    run_cli(lambda: main(SERVICE, configure=configure_cli,
        handlers={name: run_custom for name in ("generate", "job", "download", "ls-drive")}))
