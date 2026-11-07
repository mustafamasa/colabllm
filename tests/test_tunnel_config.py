"""Tunnel selection, persisted defaults and quick-mode isolation."""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import runpy
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/colabllm/lib'))
import colabsvc as svc


class TunnelConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(svc, 'STATE_ROOT', self.root))
        self.stack.enter_context(patch.dict(os.environ, {}, clear=True))
        self.service = svc.Service('avatar','avatar.ipynb',7860)

    def args(self, **kwargs):
        return argparse.Namespace(domain=None, base_domain=None, quick_tunnel=False, **kwargs)

    def test_no_domain_defaults_to_dynamic_even_with_old_tunnel_state(self):
        svc.State(self.service).save(hostname='avatar.old.example',tunnel_id='old',env={'CT_TUNNEL_TOKEN':'old'})
        self.assertIsNone(svc.tunnel_hostname(self.service,self.args()))
        args=self.args();args.domain=True
        self.assertIsNone(svc.tunnel_hostname(self.service,args))

    def test_persisted_config_can_be_shared_and_cleared(self):
        with contextlib.redirect_stdout(io.StringIO()):
            svc.cmd_tunnel_config(argparse.Namespace(base_domain='Example.COM.',clear=False))
        self.assertEqual(svc.tunnel_hostname(self.service,self.args()),'avatar.example.com')
        other=svc.Service('ollama','ollama.ipynb',11434)
        self.assertEqual(svc.tunnel_hostname(other,self.args()),'ollama.example.com')
        with contextlib.redirect_stdout(io.StringIO()):
            svc.cmd_tunnel_config(argparse.Namespace(base_domain=None,clear=True))
        self.assertIsNone(svc.tunnel_hostname(other,self.args()))

    def test_precedence_and_alias_instance(self):
        (self.root/'tunnel-config.json').write_text(json.dumps({'base_domain':'saved.example'}))
        service=svc.Service('qwen-flash','qwen.ipynb',8000,default_hostname='qwen.legacy.example')
        svc.apply_instance(service,'2')
        args=self.args()
        self.assertEqual(svc.tunnel_hostname(service,args),'qwen-2.saved.example')
        os.environ['COLABSVC_BASE_DOMAIN']='env.example'
        self.assertEqual(svc.tunnel_hostname(service,args),'qwen-2.env.example')
        args.base_domain='cli.example'
        self.assertEqual(svc.tunnel_hostname(service,args),'qwen-2.cli.example')
        args.domain='exact.example'
        self.assertEqual(svc.tunnel_hostname(service,args),'exact.example')
        args.quick_tunnel=True
        self.assertIsNone(svc.tunnel_hostname(service,args))

    def test_invalid_hostnames_rejected(self):
        for domain in ['https://example.com','example.com/path','example.com:443','*.example.com',
                       '127.0.0.1','localhost','bad_.example','-bad.example','example..com','']:
            with self.subTest(domain=domain),self.assertRaises(ValueError):svc.validate_domain(domain)
        self.assertEqual(svc.validate_domain('örnek.çay'),'xn--rnek-4qa.xn--ay-3ia')

    def test_real_up_clears_previous_kernel_env_without_cf(self):
        script=self.root/'avatar.py';script.write_text('')
        (self.root/'avatar.ipynb').write_text(json.dumps({'cells':[]}))
        self.service.env={'CT_TUNNEL_TOKEN':'stale','CT_PUBLIC_HOSTNAME':'stale.example'}
        args=self.args()
        args.__dict__.update(gpu=None,skip='',dry_run=False,new_credentials=False,hf_token=None,
                             no_drive=True,foreground=True)
        captured={}
        def runner(service, notebook):
            captured.update(svc.State(service).load()['env']);return 0
        with patch.object(svc.shutil,'which',return_value='/bin/colab'), \
             patch.object(svc,'ensure_session'),patch.object(svc,'ensure_named_tunnel') as named, \
             patch.object(svc,'exec_runner',side_effect=runner), \
             contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            svc.cmd_up(self.service,script,args)
        named.assert_not_called()
        self.assertEqual(captured['CT_TUNNEL_TOKEN'],'')
        self.assertEqual(captured['CT_PUBLIC_HOSTNAME'],'')

    def test_named_up_provisions_selected_host_and_passes_token(self):
        script=self.root/'avatar.py';script.write_text('')
        (self.root/'avatar.ipynb').write_text(json.dumps({'cells':[]}))
        args=self.args()
        args.__dict__.update(base_domain='example.com',gpu=None,skip='',dry_run=False,
                             new_credentials=False,hf_token=None,no_drive=True,foreground=True)
        info=dict(hostname='avatar.example.com',token='test-token',tunnel_id='test-id')
        with patch.object(svc.shutil,'which',return_value='/bin/tool'), \
             patch.object(svc,'ensure_session'),patch.object(svc,'ensure_named_tunnel',return_value=info) as named, \
             patch.object(svc,'exec_runner',return_value=0), \
             contextlib.redirect_stdout(io.StringIO()),self.assertRaises(SystemExit):
            svc.cmd_up(self.service,script,args)
        named.assert_called_once_with(self.service,'avatar.example.com')
        env=svc.State(self.service).load()['env']
        self.assertEqual(env['CT_TUNNEL_TOKEN'],'test-token')
        self.assertEqual(env['CT_PUBLIC_HOSTNAME'],'avatar.example.com')

    def test_parser_bare_domain_and_base_domain_are_supported(self):
        for flags,expected in [(['--domain'],None),(['--base-domain','example.com'],'avatar.example.com')]:
            with self.subTest(flags=flags),patch.object(sys,'argv',['avatar.py','up',*flags]), \
                 patch.object(svc,'cmd_up') as up:
                svc.main(self.service)
                self.assertEqual(svc.tunnel_hostname(self.service,up.call_args.args[2]),expected)

    def test_project_entry_configures_all_runtime_services(self):
        entry = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'src/colabllm/control.py'))
        with contextlib.redirect_stdout(io.StringIO()):
            entry['main'](['tunnel-config','--base-domain','example.com'])
        services = Path(__file__).resolve().parents[1] / 'src/colabllm/services'
        for script in services.glob('*.py'):
            with self.subTest(service=script.stem):
                service=runpy.run_path(str(script))['SERVICE']
                hostname=svc.tunnel_hostname(service,self.args())
                self.assertTrue(hostname.endswith('.example.com'),hostname)
        with contextlib.redirect_stdout(io.StringIO()):
            entry['main'](['tunnel-config','--clear'])
        self.assertIsNone(svc.tunnel_hostname(self.service,self.args()))


if __name__=='__main__':unittest.main()
