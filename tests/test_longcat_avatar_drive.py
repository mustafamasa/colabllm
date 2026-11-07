import sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'templates'))
from longcat_avatar_drive import resolve_drive,resolve_input,list_drive

class DriveTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()/'MyDrive';self.root.mkdir()
        (self.root/'çekimler').mkdir();(self.root/'çekimler/ses_kaydı.wav').write_bytes(b'voice')
    def tearDown(self):self.temp.cleanup()
    def test_drive_paths_unicode_and_absolute(self):
        target=self.root/'çekimler/ses_kaydı.wav'
        for value in ['drive://çekimler/ses_kaydı.wav','gdrive://çekimler/ses_kaydı.wav','drive://MyDrive/çekimler/ses_kaydı.wav',str(target)]:
            self.assertEqual(resolve_drive(value,root=self.root),target)
    def test_rejects_parent_paths_and_symlinks_outside_drive(self):
        outside=self.root.parent/'private';outside.write_text('private')
        (self.root/'escape').symlink_to(outside)
        for value in ['drive://../private',str(outside),'drive://escape']:
            with self.assertRaises(ValueError):resolve_drive(value,root=self.root)
    def test_output_directory_creation_and_listing(self):
        target=resolve_drive('drive://çıktı/film',root=self.root,directory=True,create=True)
        self.assertTrue(target.is_dir())
        files=list_drive('drive://çekimler',root=self.root)
        self.assertEqual(files[0]['path'],'drive://çekimler/ses_kaydı.wav')
        self.assertEqual(files[0]['size'],5)
    def test_upload_input_cannot_read_server_files(self):
        cache=self.root.parent/'uploads';cache.mkdir();p=cache/'image.png';p.write_bytes(b'image')
        self.assertEqual(resolve_input({'upload':str(p)},upload_root=cache),p)
        with self.assertRaises(ValueError):resolve_input({'upload':str(self.root/'çekimler/ses_kaydı.wav')},upload_root=cache)
        self.assertEqual(resolve_input({'drive':'drive://çekimler/ses_kaydı.wav'},drive_root=self.root),self.root/'çekimler/ses_kaydı.wav')

if __name__=='__main__':unittest.main()
