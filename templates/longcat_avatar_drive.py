"""Resolve mounted Google Drive paths and validated Gradio uploads."""
import os
from pathlib import Path
from urllib.parse import unquote

DRIVE_ROOT=Path('/content/drive/MyDrive')
UPLOAD_ROOT=Path(os.environ.get('GRADIO_TEMP_DIR','/tmp/gradio'))


def resolve_drive(spec, *, directory=False, create=False, root=DRIVE_ROOT):
    root=Path(root).resolve()
    if not root.is_dir():raise ValueError('Google Drive is not mounted')
    text=str(spec)
    for prefix in ('drive://','gdrive://'):
        if text.startswith(prefix):text=unquote(text[len(prefix):]);break
    if text.startswith('MyDrive/'):text=text[len('MyDrive/'):]
    candidate=Path(text)
    if not candidate.is_absolute():candidate=root/candidate
    candidate=candidate.resolve()
    if not candidate.is_relative_to(root):raise ValueError('The path must be inside MyDrive on Google Drive')
    if directory:
        if create:candidate.mkdir(parents=True,exist_ok=True)
        if not candidate.is_dir():raise ValueError(f'Drive directory does not exist: {spec}')
    elif not candidate.is_file():raise ValueError(f'Drive file does not exist: {spec}')
    return candidate


def resolve_input(reference, *, drive_root=DRIVE_ROOT, upload_root=UPLOAD_ROOT):
    if not isinstance(reference,dict) or len(reference)!=1:
        raise ValueError('The input must specify a drive or upload path')
    if 'drive' in reference:return resolve_drive(reference['drive'],root=drive_root)
    if 'upload' in reference:
        path=Path(reference['upload']).resolve()
        if not path.is_relative_to(Path(upload_root).resolve()) or not path.is_file():
            raise ValueError('Uploaded files must be inside the service upload cache')
        return path
    raise ValueError('Unsupported input path type')


def list_drive(spec='drive://', *, root=DRIVE_ROOT):
    folder=resolve_drive(spec,directory=True,root=root)
    base=Path(root).resolve()
    return [{'name':p.name,'type':'directory' if p.is_dir() else 'file',
             'path':'drive://'+p.relative_to(base).as_posix(),
             'size':p.stat().st_size if p.is_file() else None}
            for p in sorted(folder.iterdir(),key=lambda p:(not p.is_dir(),p.name))
            if p.resolve().is_relative_to(base)]
