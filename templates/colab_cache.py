"""Stage optional compiler caches using one Drive archive per environment/GPU."""
import hashlib
import os
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path


def _progress(log, message, operation):
    stopped = threading.Event()
    started = time.monotonic()

    def heartbeat():
        while not stopped.wait(10):
            log(f'{message}, elapsed {time.monotonic()-started:.0f} s')

    log(message)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        return operation()
    finally:
        stopped.set()
        thread.join()


def _files(root):
    return sorted(p for p in root.rglob('*') if p.is_file() and not p.is_symlink()
                  and not p.name.endswith(('.lock', '.tmp')))


def _fingerprint(root, files):
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(f'{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n'.encode())
    return digest.hexdigest()


def _unpack(archive, local):
    # Validate the whole archive before extracting; compiler caches contain
    # regular files/directories only. Never follow archive or local symlinks.
    with tarfile.open(archive, 'r:') as tar:
        members = tar.getmembers()
        root = local.resolve()
        for member in members:
            destination = local / member.name
            if (not (member.isfile() or member.isdir()) or Path(member.name).is_absolute()
                    or not destination.resolve().is_relative_to(root)):
                raise ValueError('Cache archive contains unsafe paths or file types')
            current = destination
            while current != local:
                if current.is_symlink():
                    raise ValueError('Cache target contains a symbolic link')
                current = current.parent
        tar.extractall(local, members=members, filter='data')


def restore_cache(local, legacy, archive_dir, key, log=print):
    local, legacy, archive_dir = map(Path, (local, legacy, archive_dir))
    local.mkdir(parents=True, exist_ok=True)
    archive = archive_dir / key / (local.name + '.tar')
    stamp = local.parent / ('.' + local.name + '-' + key + '.restored')
    if archive.is_file():
        stat = archive.stat()
        signature = f'{stat.st_size}:{stat.st_mtime_ns}'
        if stamp.exists() and stamp.read_text() == signature and _files(local):
            log(f'{local.name}: already restored locally, skipping duplicate transfer')
            return 'local'
        try:
            with tempfile.TemporaryDirectory(dir=local.parent, prefix='.cache-restore-') as tmp:
                staged = Path(tmp) / 'cache.tar'
                _progress(log, f'{local.name}: reading archive from Drive ({stat.st_size/1e6:.1f} MB)',
                          lambda: shutil.copyfile(archive, staged))
                _progress(log, f'{local.name}: unpacking on local SSD', lambda: _unpack(staged, local))
            stamp.write_text(signature)
            (local.parent / ('.' + local.name + '-' + key + '.saved')).write_text(
                _fingerprint(local, _files(local)))
            log(f'{local.name}: archive restore complete')
            return 'archive'
        except (OSError, ValueError, tarfile.TarError) as error:
            log(f'{local.name}: archive unavailable, falling back to the legacy cache ({error})')
    if legacy.is_dir():
        try:
            def migrate():
                shutil.copytree(legacy, local, dirs_exist_ok=True, symlinks=True)
            _progress(log, f'{local.name}: first-time migration of the legacy Drive cache (slow when there are many small files)', migrate)
            if _files(local):
                save_cache(local, archive_dir, key, log)
                return 'legacy'
        except (OSError, ValueError, tarfile.TarError) as error:
            log(f'{local.name}: legacy cache migration failed, falling back to on-demand generation ({error})')
    log(f'{local.name}: no persistent cache; the service will generate it on demand')
    return 'miss'


def save_cache(local, archive_dir, key, log=print):
    local, archive_dir = Path(local), Path(archive_dir)
    files = _files(local)
    if not files:
        log(f'{local.name}: no cache files, skipping archive')
        return False
    fingerprint = _fingerprint(local, files)
    saved = local.parent / ('.' + local.name + '-' + key + '.saved')
    destination = archive_dir / key / (local.name + '.tar')
    if saved.exists() and saved.read_text() == fingerprint and destination.is_file():
        log(f'{local.name}: content unchanged, skipping upload')
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tempfile.TemporaryDirectory(dir=local.parent, prefix='.cache-save-') as tmp:
            staged = Path(tmp) / 'cache.tar'
            def pack():
                with tarfile.open(staged, 'w:') as tar:
                    for path in files:
                        tar.add(path, arcname=str(path.relative_to(local)), recursive=False)
            _progress(log, f'{local.name}: packing {len(files)} files locally', pack)
            _progress(log, f'{local.name}: saving archive to Drive ({staged.stat().st_size/1e6:.1f} MB)',
                      lambda: shutil.copyfile(staged, temporary))
            os.replace(temporary, destination)
        saved.write_text(fingerprint)
        stat = destination.stat()
        (local.parent / ('.' + local.name + '-' + key + '.restored')).write_text(
            f'{stat.st_size}:{stat.st_mtime_ns}')
        log(f'{local.name}: Drive archive updated')
        return True
    finally:
        temporary.unlink(missing_ok=True)
