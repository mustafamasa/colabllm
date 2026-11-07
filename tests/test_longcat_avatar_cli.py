"""CLI integration against a local implementation of the service's HTTP contract."""
import email.parser
import email.policy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer

SCRIPT=Path(__file__).resolve().parents[1]/'src/colabllm/services/longcat-avatar.py'
VIDEO=b'\x00\x00\x00\x18ftypmp42-test-video'

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def send(self,code,body,content_type='application/json'):
        self.send_response(code);self.send_header('Content-Type',content_type)
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_POST(self):
        if self.headers.get('Authorization')!='Bearer test-token':
            self.send(401,b'{}');return
        raw=self.rfile.read(int(self.headers['Content-Length']))
        if self.path=='/gradio_api/upload':
            message=email.parser.BytesParser(policy=email.policy.default).parsebytes(
                ('Content-Type: '+self.headers['Content-Type']+'\r\nMIME-Version: 1.0\r\n\r\n').encode()+raw)
            part=list(message.iter_parts())[0]
            self.server.uploads.append((part.get_filename(),part.get_payload(decode=True)))
            self.send(200,json.dumps(['/tmp/upload/'+part.get_filename()]).encode())
        elif self.path=='/jobs':
            self.server.submitted=json.loads(raw)
            self.send(202,b'{"job_id":"job-42"}')
        else:self.send(404,b'{}')
    def do_GET(self):
        if self.headers.get('Authorization')!='Bearer test-token':
            self.send(401,b'{}');return
        if self.path=='/gradio_api/call/submit/submit-event':
            body='event: heartbeat\ndata: null\n\nevent: complete\ndata: ["job-42", "queued"]\n\n'
            self.send(200,body.encode(),'text/event-stream')
        elif self.path=='/jobs/job-42':
            self.server.polls+=1
            status='running' if self.server.polls==1 else self.server.final_status
            data={'id':'job-42','status':status,'message':'Generating' if status=='running' else 'test failure',
                  'result':{'path':'/drive/output/video with space.mp4','metadata':{'frames':21}}}
            self.send(200,json.dumps(data).encode())
        elif self.path=='/jobs/job-42/video':
            self.send(200,VIDEO,'video/mp4')
        elif self.path.startswith('/drive/files?'):
            if self.server.list_failures:
                self.server.list_failures-=1
                self.send(503,b'{}');return
            self.send(200,json.dumps({'files':[{'name':'ses_kaydı.wav','path':'drive://çekimler/ses_kaydı.wav'}]}).encode())
        else:self.send(404,b'{}')

class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.audio=self.root/'voice.wav';self.audio.write_bytes(b'voice-bytes')
        self.image=self.root/'person.png';self.image.write_bytes(b'picture-bytes')
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.server.uploads=[];self.server.polls=0;self.server.final_status='completed';self.server.list_failures=0
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.endpoint=['--url',f'http://127.0.0.1:{self.server.server_port}','--token','test-token']
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.tmp.cleanup()
    def cli(self,*args):
        return subprocess.run([sys.executable,str(SCRIPT),*args,*self.endpoint],capture_output=True,text=True,timeout=10)
    def test_async_submission_uploads_files_and_defaults_to_full_audio(self):
        p=self.cli('generate','--audio',str(self.audio),'--image',str(self.image),'--prompt','A person speaks.','-i','demo')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout),{'job_id':'job-42'})
        self.assertEqual(self.server.polls,0)
        self.assertEqual(self.server.uploads,[('voice.wav',b'voice-bytes'),('person.png',b'picture-bytes')])
        data=self.server.submitted
        self.assertEqual(data['audio'],{'upload':'/tmp/upload/voice.wav'})
        self.assertEqual(data['image'],{'upload':'/tmp/upload/person.png'})
        self.assertTrue(data['auto_duration'])
    def test_wait_download_and_manual_duration(self):
        prompt=self.root/'prompt.txt';prompt.write_text('Kişi konuşuyor',encoding='utf-8')
        output=self.root/'nested/result.mp4'
        p=self.cli('generate','--audio',str(self.audio),'--prompt-file',str(prompt),
                   '--frames','17','--segments','2','--poll','.01','-o',str(output))
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(output.read_bytes(),VIDEO)
        self.assertEqual(json.loads(p.stdout)['status'],'completed')
        self.assertIn('Job submitted: job-42',p.stderr)
        self.assertEqual(self.server.submitted['image'],None)
        self.assertEqual((self.server.submitted['frames'],self.server.submitted['segments']),(17,2))
        self.assertFalse(self.server.submitted['auto_duration'])
    def test_existing_job_download_does_not_resubmit(self):
        output=self.root/'download.mp4'
        p=self.cli('download','job-42','--wait','--poll','.01','-o',str(output))
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(output.read_bytes(),VIDEO)
        self.assertEqual(self.server.uploads,[])
    def test_failed_job_returns_error_and_keeps_existing_output(self):
        self.server.final_status='failed'
        output=self.root/'keep.mp4';output.write_bytes(b'old-file')
        p=self.cli('job','job-42','--wait','--poll','.01','-o',str(output))
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(output.read_bytes(),b'old-file')
        self.assertIn('test failure',p.stderr)
    def test_invalid_frames_do_not_upload_anything(self):
        p=self.cli('generate','--audio',str(self.audio),'--prompt','test','--frames','18')
        self.assertNotEqual(p.returncode,0)
        self.assertEqual(self.server.uploads,[])

    def test_standard_sampling_reaches_server(self):
        p=self.cli('generate','--audio',str(self.audio),'--prompt','test',
                   '--sampling-mode','standard','--steps','30','--text-cfg','4','--audio-cfg','1')
        self.assertEqual(p.returncode,0,p.stderr)
        data=self.server.submitted
        self.assertEqual((data['sampling_mode'],data['steps'],data['text_guidance_scale'],data['audio_guidance_scale']),
                         ('standard',30,4,1))
        self.assertFalse(data['use_distill'])

    def test_incompatible_sampling_fails_before_upload(self):
        for flags in [('--audio-cfg','3'),('--sampling-mode','standard','--audio-cfg','0.7')]:
            with self.subTest(flags=flags):
                p=self.cli('generate','--audio',str(self.audio),'--prompt','test',*flags)
                self.assertNotEqual(p.returncode,0)
                self.assertEqual(self.server.uploads,[])

    def test_drive_inputs_and_output_do_not_upload_local_files(self):
        p=self.cli('generate','--audio-drive','çekimler/ses_kaydı.wav','--image','drive://çekimler/portre_ğ.png',
                   '--prompt','test','--drive-output','çıktılar/eser')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(self.server.uploads,[])
        self.assertEqual(self.server.submitted['audio'],{'drive':'drive://çekimler/ses_kaydı.wav'})
        self.assertEqual(self.server.submitted['image'],{'drive':'drive://çekimler/portre_ğ.png'})
        self.assertEqual(self.server.submitted['output_dir'],'drive://çıktılar/eser')
    def test_mixed_drive_and_local_inputs(self):
        p=self.cli('generate','--audio','drive://voice.wav','--image',str(self.image),'--prompt','test')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(self.server.uploads,[('person.png',b'picture-bytes')])
        self.assertEqual(self.server.submitted['audio'],{'drive':'drive://voice.wav'})
    def test_drive_listing_recovers_from_transient_failure(self):
        self.server.list_failures=1
        p=self.cli('ls-drive','çekimler')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('retrying',p.stderr)
        self.assertEqual(json.loads(p.stdout)['files'][0]['name'],'ses_kaydı.wav')
    def test_drive_listing(self):
        p=self.cli('ls-drive','çekimler')
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(json.loads(p.stdout)['files'][0]['name'],'ses_kaydı.wav')

if __name__=='__main__':unittest.main()
