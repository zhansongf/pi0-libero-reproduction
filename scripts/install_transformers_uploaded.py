"""Safely extract and build the uploaded, pinned LeRobot Transformers fork."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

root = Path('/root/autodl-tmp')
commit = 'dcddb970176382c0fcf4521b0c0e6fc15894dfe0'
archive = root / f'transformers-{commit}.zip'
sources = root / 'lerobot-src'
destination = sources / f'transformers-{commit}'
records = root / 'lerobot-env-records'
python = str(root / 'envs/lerobot/bin/python')
with zipfile.ZipFile(archive) as bundle:
    prefix = f'transformers-{commit}'
    safe_members = []
    for member in bundle.infolist():
        path = Path(member.filename)
        assert not path.is_absolute() and '..' not in path.parts, member.filename
        assert path.parts[0] == prefix, member.filename
        if ((member.external_attr >> 16) & 0o170000) == 0o120000:
            continue  # Documentation symlinks are unnecessary for building the package.
        safe_members.append(member)
    assert not destination.exists(), f'Source folder already exists: {destination}'
    bundle.extractall(sources, members=safe_members)
version_source = (destination / 'src/transformers/__init__.py').read_text()
assert '__version__ = "4.53.3"' in version_source
check_source = destination / 'src/transformers/models/siglip/check.py'
assert 'check_whether_transformers_replace_is_installed_correctly' in check_source.read_text()
wheels = records / 'transformers-wheels'
wheels.mkdir(exist_ok=True)
subprocess.run([python, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                '--wheel-dir', str(wheels), str(destination)], check=True)
wheel_files = list(wheels.glob('transformers-4.53.3-*.whl'))
assert len(wheel_files) == 1, wheel_files
wheel = wheel_files[0]
subprocess.run([python, '-m', 'pip', 'install', '--no-cache-dir', '--index-url',
                'https://pypi.tuna.tsinghua.edu.cn/simple', str(wheel)], check=True)
record = {'repo': 'https://github.com/huggingface/transformers',
          'branch': 'fix/lerobot_openpi', 'commit': commit, 'version': '4.53.3',
          'source_path': str(destination), 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
          'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
(records / 'transformers-source.json').write_text(json.dumps(record, indent=2))
subprocess.run([python, '-m', 'pip', 'check'], check=True)
with (records / 'requirements-lock.txt').open('w') as stream:
    subprocess.run([python, '-m', 'pip', 'freeze'], stdout=stream, check=True)
print('BUILD AND INSTALL COMPLETE', flush=True)
