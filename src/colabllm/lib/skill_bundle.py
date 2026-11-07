"""Read or export the complete user skill bundled with a wheel or source checkout."""
from importlib.metadata import distribution, PackageNotFoundError
from pathlib import Path
import shutil
import tempfile

from cli_support import CliError


def bundle_root():
    checkout = Path(__file__).resolve().parents[3] / 'skills/colabllm'
    if (checkout / 'SKILL.md').is_file():
        return checkout
    try:
        dist = distribution('colabllm')
        for file in dist.files or ():
            if str(file).endswith('share/colabllm/skills/colabllm/SKILL.md'):
                root = Path(dist.locate_file(file)).resolve().parent
                if (root / 'SKILL.md').is_file():
                    return root
    except PackageNotFoundError:
        pass
    raise CliError('The installed package is missing the skill resources; install a newer colabllm distribution that includes skills')


def export_skill(output):
    source = bundle_root()
    parent = Path(output).expanduser().resolve()
    target = parent / 'colabllm'
    if target.exists() or target.is_symlink():
        raise CliError(f'Target already exists, not overwritten: {target}; export to a new directory and compare before updating')
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.colabllm-skill-', dir=parent) as temporary:
        staged = Path(temporary) / 'colabllm'
        shutil.copytree(source, staged)
        # Recheck before rename; the exporter never merges with an existing skill.
        if target.exists() or target.is_symlink():
            raise CliError(f'Target already exists, not overwritten: {target}')
        staged.rename(target)
    return target
