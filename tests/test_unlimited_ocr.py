import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/colabllm/lib'))
spec = importlib.util.spec_from_file_location('unlimited_ocr_app', ROOT / 'templates/unlimited_ocr_app.py')
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.fail = False

    def convert(self, source, destination, opts, progress):
        self.calls.append(source)
        if self.fail or source.name == 'broken.pdf':
            raise RuntimeError('bad PDF')
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'result.md').write_text('converted')
        progress(2, 2)
        return {'pages': 2, 'tokens': 10, 'chunks': 1}


class DriveJobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.patch = patch.object(app, 'DRIVE', self.root)
        self.patch.start()
        self.engine = FakeEngine()
        self.jobs = app.Jobs(self.engine, self.root / 'service')

    def tearDown(self):
        self.jobs.queue.join()
        self.patch.stop()
        self.tmp.cleanup()

    def submit(self, inputs, **kwargs):
        record = self.jobs.submit(dict(inputs=inputs, output_dir='out', **kwargs))
        self.jobs.queue.join()
        return self.jobs.get(record['id'])

    def test_path_aliases_and_escape(self):
        for value in ['drive://docs/a.pdf', 'MyDrive/docs/a.pdf', str(self.root / 'docs/a.pdf')]:
            self.assertEqual(app.drive_path(value), self.root / 'docs/a.pdf')
        for value in ['../escape', '/etc/passwd', 'drive://../../etc/passwd']:
            with self.assertRaises(ValueError):
                app.drive_path(value)
        (self.root / 'link').symlink_to('/tmp')
        with self.assertRaises(ValueError):
            app.drive_path('link/file.pdf')

    def test_batch_deduplicates_preserves_hierarchy_and_continues_on_error(self):
        for name in ['a/same.pdf', 'b/same.pdf', 'a/broken.pdf', 'a/ignore.txt']:
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.write_text('source')
        record = self.submit(['a', 'b', 'a/same.pdf'])
        self.assertEqual(record['status'], 'failed')
        self.assertEqual(len(record['files']), 3)
        self.assertTrue((self.root / 'out/a/same.pdf/result.md').is_file())
        self.assertTrue((self.root / 'out/b/same.pdf/result.md').is_file())
        self.assertEqual([f['status'] for f in record['files']], ['failed', 'completed', 'completed'])

    def test_skip_changed_parameters_and_failed_overwrite(self):
        source = self.root / 'a.pdf'
        source.write_text('v1')
        self.assertEqual(self.submit(['a.pdf'])['status'], 'completed')
        self.assertEqual(self.submit(['a.pdf'])['files'][0]['status'], 'skipped')
        source.write_text('changed')
        self.assertEqual(self.submit(['a.pdf'])['files'][0]['status'], 'completed')
        self.assertEqual(self.submit(['a.pdf'], dpi=300)['files'][0]['status'], 'completed')
        record = self.submit(['a.pdf'], inference_params={'temperature': 0.1})
        self.assertEqual(record['files'][0]['status'], 'completed')
        self.assertEqual(record['options']['inference_params'], {'temperature': 0.1})
        self.assertEqual(self.submit(['a.pdf'], inference_params={'temperature': 0.1})['files'][0]['status'], 'skipped')
        self.engine.fail = True
        self.assertEqual(self.submit(['a.pdf'], overwrite=True)['status'], 'failed')
        self.assertFalse((self.root / 'out/a.pdf/complete.json').exists())
        self.engine.fail = False
        self.assertEqual(self.submit(['a.pdf'])['files'][0]['status'], 'completed')

    def test_discovery_excludes_outputs_and_nonrecursive(self):
        for name in ['a.pdf', 'sub/b.pdf', 'out/generated.png']:
            path = self.root / name
            path.parent.mkdir(exist_ok=True)
            path.touch()
        paths = app.discover(['.'], output=self.root / 'out')
        self.assertEqual([p.relative_to(self.root).as_posix() for p in paths], ['a.pdf', 'sub/b.pdf'])
        self.assertEqual(app.discover(['.'], recursive=False), [self.root / 'a.pdf'])
        with self.assertRaises(ValueError):
            app.discover(['missing'])

    def test_restart_requeues_valid_jobs_and_job_ids_are_checked(self):
        source = self.root / 'resume.pdf'
        source.write_text('source')
        record = dict(id='a' * 32, status='running', options=app.options({}),
                      output_dir=str(self.root / 'out'), overwrite=False,
                      files=[dict(source=str(source), status='running')])
        app.atomic_json(self.jobs.root / ('a' * 32 + '.json'), record)
        recovered = app.Jobs(self.engine, self.root / 'service', start_worker=False)
        self.assertEqual(recovered.get('a' * 32)['status'], 'queued')
        self.assertEqual(recovered.get('a' * 32)['resume_count'], 1)
        resumed = recovered.queue.get_nowait()
        recovered.run(resumed)
        recovered.queue.task_done()
        self.assertEqual(recovered.get('a' * 32)['status'], 'completed')
        with self.assertRaises(ValueError):
            recovered.get('../secret')

    def test_manual_resume_retains_id_and_rejects_duplicate(self):
        source = self.root / 'broken.pdf'
        source.write_text('source')
        record = self.submit(['broken.pdf'])
        stopped = app.Jobs(self.engine, self.root / 'service', start_worker=False)
        resumed = stopped.resume(record['id'])
        self.assertEqual(resumed['id'], record['id'])
        self.assertEqual(resumed['files'][0]['status'], 'queued')
        with self.assertRaises(ValueError):
            stopped.resume(record['id'])
        queued = stopped.queue.get_nowait()
        stopped.queue.task_done()

    def test_overwrite_job_restart_keeps_already_completed_files(self):
        for name in ('first.pdf', 'second.pdf'):
            (self.root / name).write_text('source')
        record = self.submit(['first.pdf', 'second.pdf'], overwrite=True)
        record['status'] = 'running'
        record['files'][1]['status'] = 'running'
        record['files'][1].pop('completed_signature')
        (Path(record['files'][1]['output']) / 'complete.json').unlink()
        app.atomic_json(self.jobs.root / (record['id'] + '.json'), record)
        self.engine.calls.clear()
        recovered = app.Jobs(self.engine, self.root / 'service', start_worker=False)
        queued = recovered.queue.get_nowait()
        recovered.run(queued)
        recovered.queue.task_done()
        self.assertEqual(self.engine.calls, [self.root / 'second.pdf'])
        self.assertEqual(recovered.get(record['id'])['status'], 'completed')

    def test_corrupt_job_record_does_not_prevent_startup(self):
        (self.jobs.root / 'bad.json').write_text('{broken')
        recovered = app.Jobs(self.engine, self.root / 'service', start_worker=False)
        self.assertTrue(recovered.queue.empty())

    def test_invalid_options_rejected_before_queue(self):
        for values in [{'dpi': 1}, {'pages_per_chunk': 0}, {'tokens_per_page': -1}]:
            with self.assertRaises(ValueError):
                self.jobs.submit(dict(inputs=['a.pdf'], **values))

class LocalTransferTest(unittest.TestCase):
    def test_chunk_upload_checksum_and_result_download(self):
        import base64
        import hashlib
        with tempfile.TemporaryDirectory() as d, patch.object(app, 'DRIVE', Path(d).resolve()), patch.object(app, 'DATA', Path(d).resolve() / 'svc'):
            content = b'pdf contents' * 100
            body = dict(sha256=hashlib.sha256(content).hexdigest(), name='sözleşme.pdf', offset=0,
                        data=base64.b64encode(content[:100]).decode(), final=False)
            self.assertFalse(app.upload_chunk(body)['complete'])
            with self.assertRaises(ValueError):
                app.upload_chunk(dict(body, offset=99))
            result = app.upload_chunk(dict(body, offset=100, data=base64.b64encode(content[100:]).decode(), final=True))
            self.assertEqual(Path(result['path']).read_bytes(), content)
            with self.assertRaises(ValueError):
                app.upload_chunk(dict(body, name='../escape.pdf'))
            with self.assertRaises(ValueError):
                app.upload_chunk(dict(body, sha256='0' * 64, final=True))
            engine = FakeEngine()
            jobs = app.Jobs(engine, Path(d) / 'svc')
            job = jobs.submit(dict(inputs=[result['path']], output_dir='results', relative_outputs={result['path']: 'books/sözleşme.pdf'}))
            jobs.queue.join()
            self.assertEqual(jobs.get(job['id'])['files'][0]['relative_output'], 'books/sözleşme.pdf')
            self.assertIn('result.md', app.result_files(jobs, job['id'], 0)['files'])
            part = app.result_files(jobs, job['id'], 0, 'result.md')
            self.assertEqual(base64.b64decode(part['data']), b'converted')
            with self.assertRaises(ValueError):
                app.result_files(jobs, job['id'], 0, '../../outside')

    def test_cli_local_folder_and_forced_drive(self):
        spec = importlib.util.spec_from_file_location('ocr_cli', ROOT / 'src/colabllm/services/unlimited-ocr.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        import argparse
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / 'books'
            (folder / 'sub').mkdir(parents=True)
            (folder / 'sub/a.pdf').write_bytes(b'pdf')
            (folder / 'ignore.txt').write_text('ignore')
            args = argparse.Namespace(inputs=[str(folder)], local=False, drive=False, no_recursive=False, local_output_dir=None)
            with patch.object(cli, 'upload_local', return_value='/content/drive/MyDrive/upload/a.pdf') as upload:
                inputs, mapping = cli.prepare_inputs(args)
                self.assertEqual(list(mapping.values()), ['books/sub/a.pdf'])
                self.assertEqual(upload.call_count, 1)
            args.drive = True
            inputs, mapping = cli.prepare_inputs(args)
            self.assertEqual(inputs, [str(folder)])
            self.assertFalse(mapping)
            args.drive = False
            args.inputs = [str(folder / 'missing.pdf')]
            with self.assertRaises(ValueError):
                cli.prepare_inputs(args)


class InferenceParametersTest(unittest.TestCase):
    def test_all_parameters_reach_model_and_both_return_modes_work(self):
        from contextlib import nullcontext
        from unittest.mock import Mock
        params = dict(prompt='Recognize this.', base_size=512, image_size=320, crop_mode=True,
                      test_compress=True, save_results=True, eval_mode=False, max_length=8192,
                      max_new_tokens=123, tps_interval=10, no_repeat_ngram_size=0, ngram_window=20,
                      ngram_whitelist_token_ids=[1, 128827], temperature=0.1, page_count_guard=False,
                      max_page_tokens=0, dash_topm=8, max_crops=4, return_dict=False)
        self.assertEqual(set(params), set(app.validate_parameters(params)))
        engine = object.__new__(app.Engine)
        engine.torch = Mock()
        engine.torch.inference_mode.side_effect = nullcontext
        engine.tokenizer = Mock()
        engine.tokenizer.encode.return_value = [1, 2]
        engine.model = Mock()
        engine.model.infer.return_value = 'recognized'
        opts = dict(tokens_per_page=8000, inference_params=params)
        self.assertEqual(engine.infer_pages([Path('a.png')], Path('native'), opts), ('recognized', 2))
        engine.model.infer.assert_called_once_with(engine.tokenizer, image_file=['a.png'], output_path='native', **params)
        engine.model.infer.return_value = dict(text='layout', token_ids=[1, 2, 3])
        opts['inference_params'] = dict(return_dict=True, max_new_tokens=None, crop_mode=None)
        self.assertEqual(engine.infer_pages([Path('a.png')], Path('native'), opts), ('layout', 3))
        self.assertIsNone(engine.model.infer.call_args.kwargs['max_new_tokens'])
        self.assertIsNone(engine.model.infer.call_args.kwargs['crop_mode'])

    def test_rejects_unknown_reserved_and_malformed_parameters(self):
        for value in [[], {'typo': 1}, {'output_path': '/tmp'}, {'tokenizer': 'x'},
                      {'temperature': float('nan')}, {'temperature': -1}, {'max_new_tokens': False},
                      {'crop_mode': 'false'}, {'ngram_whitelist_token_ids': [-1]}, {'max_crops': 0}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                app.options(dict(inference_params=value))
        self.assertEqual(app.options(dict(inference_params={'max_page_tokens': -1}))['inference_params'], {'max_page_tokens': -1})

    def test_cli_precedence_and_bool_switches(self):
        from ocr_parameters import add_inference_arguments, cli_parameters
        import argparse
        parser = argparse.ArgumentParser()
        add_inference_arguments(parser)
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / 'params.json'
            file.write_text(json.dumps(dict(temperature=0.1, image_size=512)))
            args = parser.parse_args(['--params-file', str(file), '--params-json', '{"temperature":0.2}',
                                      '--temperature', '0.3', '--no-page-count-guard', '--save-results',
                                      '--param', 'temperature=0.4', '--param', 'crop_mode=null',
                                      '--ngram-whitelist-token-ids', '[1,2]'])
            self.assertEqual(cli_parameters(args), dict(temperature=0.4, image_size=512,
                             page_count_guard=False, save_results=True, crop_mode=None,
                             ngram_whitelist_token_ids=[1, 2]))


class CheckpointResumeTest(unittest.TestCase):
    def test_page_groups_resume_after_crash_and_invalid_artifacts_recompute(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        class Crash(BaseException):
            pass
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / 'book.pdf'
            source.write_bytes(b'pdf')
            destination = root / 'result'
            pdf = Mock()
            pdf.__len__ = Mock(return_value=3)
            page = Mock()
            page.get_pixmap.return_value.save.side_effect = lambda path: Path(path).write_bytes(b'png')
            pdf.__getitem__ = Mock(return_value=page)
            modules = {'fitz': SimpleNamespace(open=lambda path: pdf),
                       'PIL': SimpleNamespace(Image=Mock()),
                       'process_unlimitedOCR2': SimpleNamespace(convert=lambda raw: raw)}
            engine = object.__new__(app.Engine)
            calls = []
            crash = [True]
            def infer(pages, native, opts):
                name = pages[0].name
                calls.append(name)
                if name == 'page2.png' and crash[0]:
                    raise Crash()
                native.mkdir(parents=True, exist_ok=True)
                (native / 'figure.jpg').write_bytes(b'figure')
                return name, 10
            engine.infer_pages = infer
            opts = app.options({'pages_per_chunk': 1})
            original_tempdir = tempfile.TemporaryDirectory
            def tempdir(**kwargs):
                return original_tempdir(dir=root)
            with patch.dict(sys.modules, modules), patch.object(app.tempfile, 'TemporaryDirectory', side_effect=tempdir):
                with self.assertRaises(Crash):
                    engine.convert(source, destination, opts, lambda *args: None)
                self.assertEqual(calls, ['page1.png', 'page2.png'])
                crash[0] = False
                calls.clear()
                result = engine.convert(source, destination, opts, lambda *args: None)
                self.assertEqual(calls, ['page2.png', 'page3.png'])
                self.assertEqual(result['resumed_chunks'], 1)
                self.assertEqual(result['tokens'], 30)
                self.assertEqual((destination / 'result.raw.txt').read_text(), 'page1.png\npage2.png\npage3.png')
                (destination / 'native-0001/figure.jpg').unlink()
                calls.clear()
                result = engine.convert(source, destination, opts, lambda *args: None)
                self.assertEqual(calls, ['page1.png'])
                self.assertEqual(result['resumed_chunks'], 2)
                (destination / '.checkpoints/000001.json').write_text('{bad json')
                calls.clear()
                result = engine.convert(source, destination, opts, lambda *args: None)
                self.assertEqual(calls, ['page2.png'])
                self.assertEqual(result['resumed_chunks'], 2)
                # Changing source, parameters or overwrite generation invalidates every group.
                for changed in [dict(opts, dpi=300), dict(opts, _checkpoint_generation='new-job')]:
                    calls.clear()
                    engine.convert(source, destination, changed, lambda *args: None)
                    self.assertEqual(len(calls), 3)
                source.write_bytes(b'changed pdf')
                calls.clear()
                engine.convert(source, destination, opts, lambda *args: None)
                self.assertEqual(len(calls), 3)


if __name__ == '__main__':
    unittest.main()
