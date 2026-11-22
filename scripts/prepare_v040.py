"""Extract LeRobot v0.4.0 dependencies while deferring all Transformers installation."""
from pathlib import Path
import tomllib

root = Path('/root/autodl-tmp')
source = root / 'lerobot-src/lerobot'
metadata = tomllib.loads((source / 'pyproject.toml').read_text())
assert metadata['project']['version'] == '0.4.0'
requirements = metadata['project']['dependencies']
# The training dependencies are in the base package in v0.4.0.
# The libero extra adds Transformers and a Git LIBERO reference; both are handled separately.
excluded = ('transformers', 'lerobot', 'libero')
requirements = [r for r in requirements if not r.lower().startswith(excluded)]
requirements += [
    'torch==2.7.1', 'torchvision==0.22.1', 'torchcodec==0.5.0',
    'numpy==2.2.6', 'opencv-python-headless==4.11.0.86',
    'opencv-python==4.11.0.86', 'matplotlib==3.10.7',
    'hydra-core>=1.2,<1.4', 'robomimic==0.2.0', 'robosuite==1.4.0',
    'bddl==1.0.1', 'mujoco==3.3.2', 'hf-egl-probe', 'easydict', 'thop', 'cloudpickle>=2',
    'gym>=0.25,<0.27', 'future>=0.18.2', 'sentencepiece',
]
target = root / 'lerobot-env-records/requirements-no-transformers.in'
target.write_text('\n'.join(requirements) + '\n')
print(target)
