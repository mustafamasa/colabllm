import io
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'templates'))
import colab_cache as cache


class CacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.local=self.root/'ssd'/'triton-cache'
        self.local.mkdir(parents=True)
        self.legacy=self.root/'drive'/'triton-cache'
        self.legacy.mkdir(parents=True)
        self.archives=self.root/'drive'/'cache-archives'
        self.key='vllm-0.32.0-py312-sm89'
        self.logs=[]

    def restore(self):
        return cache.restore_cache(self.local,self.legacy,self.archives,self.key,self.logs.append)

    def save(self):
        return cache.save_cache(self.local,self.archives,self.key,self.logs.append)

    def test_legacy_migration_then_archive_only_restore(self):
        nested=self.legacy/'.cache'/'kernel';nested.mkdir(parents=True)
        (nested/'modül.cubin').write_bytes(b'compiled-kernel')
        self.assertEqual(self.restore(),'legacy')
        self.assertTrue((self.archives/self.key/'triton-cache.tar').is_file())
        fresh=self.root/'fresh'/'triton-cache'
        with patch.object(cache.shutil,'copytree',side_effect=AssertionError('must not walk Drive')):
            result=cache.restore_cache(fresh,self.legacy,self.archives,self.key,self.logs.append)
        self.assertEqual(result,'archive')
        self.assertEqual((fresh/'.cache/kernel/modül.cubin').read_bytes(),b'compiled-kernel')
        self.assertFalse(cache.save_cache(fresh,self.archives,self.key,self.logs.append))
        with patch.object(cache.shutil,'copyfile',side_effect=AssertionError('local cache should be reused')):
            self.assertEqual(cache.restore_cache(fresh,self.legacy,self.archives,self.key,self.logs.append),'local')

    def test_unchanged_cache_skips_upload_and_changed_cache_updates(self):
        p=self.local/'kernel.so';p.write_bytes(b'one')
        self.assertTrue(self.save())
        self.assertFalse(self.save())
        p.write_bytes(b'two-two')
        self.assertTrue(self.save())
        with tarfile.open(self.archives/self.key/'triton-cache.tar') as tar:
            self.assertEqual(tar.extractfile('kernel.so').read(),b'two-two')

    def test_corrupt_archive_falls_back_and_is_replaced(self):
        folder=self.archives/self.key;folder.mkdir(parents=True)
        (folder/'triton-cache.tar').write_bytes(b'interrupted-data')
        (self.legacy/'kernel').write_bytes(b'legacy')
        self.assertEqual(self.restore(),'legacy')
        self.assertTrue(any('archive unavailable' in line for line in self.logs))
        with tarfile.open(folder/'triton-cache.tar') as tar:
            self.assertEqual(tar.extractfile('kernel').read(),b'legacy')

    def test_failed_upload_preserves_previous_archive(self):
        (self.local/'kernel').write_bytes(b'original')
        self.save()
        archive=self.archives/self.key/'triton-cache.tar';original=archive.read_bytes()
        (self.local/'kernel').write_bytes(b'new')
        def broken(source,target):
            Path(target).write_bytes(b'partial')
            raise OSError('Drive disconnected')
        with patch.object(cache.shutil,'copyfile',side_effect=broken),self.assertRaises(OSError):self.save()
        self.assertEqual(archive.read_bytes(),original)
        self.assertEqual(list(archive.parent.glob('*.tmp')),[])

    def test_unsafe_archive_is_rejected_before_any_extraction(self):
        p=self.root/'unsafe.tar'
        for name,kind in [('../escaped',tarfile.REGTYPE),('link',tarfile.SYMTYPE)]:
            with self.subTest(name=name):
                with tarfile.open(p,'w:') as tar:
                    good=tarfile.TarInfo('good');good.size=2;tar.addfile(good,io.BytesIO(b'ok'))
                    member=tarfile.TarInfo(name);member.type=kind;member.linkname='/tmp'
                    tar.addfile(member)
                with self.assertRaises(ValueError):cache._unpack(p,self.local)
                self.assertFalse((self.local/'good').exists())
                self.assertFalse((self.local.parent/'escaped').exists())

    def test_empty_cache_and_gpu_namespace_isolation(self):
        self.assertEqual(self.restore(),'miss')
        self.assertFalse(self.save())
        (self.local/'kernel').write_bytes(b'gpu89');self.save()
        other=self.root/'other'/'triton-cache'
        self.assertEqual(cache.restore_cache(other,self.legacy,self.archives,'sm120',self.logs.append),'miss')
        self.assertFalse((other/'kernel').exists())


if __name__=='__main__':unittest.main()
