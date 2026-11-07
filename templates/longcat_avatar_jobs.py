"""Persistent, browser-independent serial jobs for LongCat Avatar."""
import json
import shutil
import threading
import time
import uuid
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


class JobQueue:
    def __init__(self, root, runner):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.lock = threading.RLock()
        self.active = None
        self.draining = False
        self.shutdown_ready = threading.Event()
        self.stop = threading.Event()
        for path in self.root.glob('*/status.json'):
            try:
                value = json.loads(path.read_text())
                if value.get('status')=='running':
                    value.update(status='queued', message='Resuming interrupted job from the latest checkpoint')
                    write_json(path,value)
            except (OSError, ValueError, KeyError, TypeError) as error:
                print(f'Warning: skipping unreadable job status {path}: {error}', flush=True)

    def submit(self, request, audio, image=None):
        with self.lock:
            if self.draining:
                raise ValueError('The service is updating; submit new jobs later')
            job_id = time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            folder = self.root/job_id
            folder.mkdir()
            audio_path = folder/('input-audio'+Path(audio).suffix)
            shutil.copy2(audio,audio_path)
            request = {**request, 'audio':str(audio_path), 'image':None}
            if image:
                image_path=folder/('input-image'+Path(image).suffix)
                shutil.copy2(image,image_path)
                request['image']=str(image_path)
            write_json(folder/'request.json',request)
            write_json(folder/'status.json',dict(id=job_id,status='queued',message='Queued',created=time.time(),attempts=0))
            return job_id

    def get(self, job_id):
        if not job_id or Path(job_id).name!=job_id:
            raise ValueError('Invalid job ID')
        return json.loads((self.root/job_id/'status.json').read_text())

    def recent(self):
        return sorted([p.name for p in self.root.iterdir() if (p/'status.json').exists()],reverse=True)

    def update(self, job_id, **values):
        with self.lock:
            state=self.get(job_id)
            state.update(values,updated=time.time())
            write_json(self.root/job_id/'status.json',state)
            return state

    def drain(self):
        with self.lock:
            self.draining=True

    def start(self):
        self.thread=threading.Thread(target=self._run,name='avatar-job-worker',daemon=False)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            with self.lock:
                pending=[]
                for job in reversed(self.recent()):
                    try:
                        if self.get(job).get('status')=='queued':pending.append(job)
                    except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
                        # Completed validation/history folders may be archived
                        # or unreadable while scanning the shared Drive.
                        continue
                if not pending:
                    if self.draining:
                        self.shutdown_ready.set()
                        return
                    job_id=None
                else:
                    job_id=pending[0]
                    state=self.get(job_id)
                    self.active=job_id
                    self.update(job_id,status='running',attempts=state['attempts']+1,message='Generating')
            if not job_id:
                self.stop.wait(.5)
                continue
            try:
                folder=self.root/job_id
                request=json.loads((folder/'request.json').read_text())
                result=self.runner(job_id,folder,request,lambda **kw:self.update(job_id,**kw))
                self.update(job_id,status='completed',message='Generation complete',result=result)
            except Exception as error:
                self.update(job_id,status='failed',message=str(error))
                import traceback
                traceback.print_exc()
            finally:
                with self.lock:self.active=None
