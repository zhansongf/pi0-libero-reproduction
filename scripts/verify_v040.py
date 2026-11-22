"""Verify the Python/LIBERO stack without Transformers, model downloads, or a GPU."""
import importlib.metadata as md
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['NUMBA_DISABLE_JIT'] = '1'
root = Path('/root/autodl-tmp')
config_dir = root / 'lerobot-env-records/libero-config'
config_dir.mkdir(exist_ok=True)
os.environ['LIBERO_CONFIG_PATH'] = str(config_dir)
base = root / 'lerobot-src/LIBERO/libero/libero'
import yaml
config = {'benchmark_root': str(base), 'bddl_files': str(base / 'bddl_files'),
          'init_states': str(base / 'init_files'), 'assets': str(base / 'assets'),
          'datasets': str(root / 'datasets/libero')}
(config_dir / 'config.yaml').write_text(yaml.safe_dump(config))
report = {'python': sys.version, 'packages': {}, 'errors': {}}
for name in ['lerobot', 'libero', 'torch', 'torchvision', 'torchcodec', 'numpy', 'mujoco', 'robosuite']:
    try:
        report['packages'][name] = md.version(name)
    except Exception as error:
        report['errors'][name] = str(error)
for module in ['torch', 'torchvision', 'torchcodec', 'lerobot', 'robosuite']:
    try:
        __import__(module)
    except Exception as error:
        report['errors'][module] = f'{type(error).__name__}: {error}'
try:
    from libero.libero import benchmark
    suites = benchmark.get_benchmark_dict()
    report['libero_tasks'] = {name: suites[name]().n_tasks for name in (
        'libero_spatial', 'libero_object', 'libero_goal', 'libero_10')}
except Exception as error:
    report['errors']['libero_benchmark'] = f'{type(error).__name__}: {error}'
report['libero_step_status'] = 'Pending: full robot scene exceeded the 2 GiB no-GPU instance memory limit.'
try:
    import mujoco
    model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size="0.1"/></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=32, width=32) as renderer:
        renderer.update_scene(data)
        report['cpu_render_shape'] = list(renderer.render().shape)
except Exception as error:
    report['errors']['cpu_render'] = f'{type(error).__name__}: {error}'
report['transformers_installed'] = importlib.util.find_spec('transformers') is not None
import torch
report['cuda_runtime'] = torch.version.cuda
report['cuda_available'] = torch.cuda.is_available()
print(json.dumps(report, indent=2))
(root / 'lerobot-env-records/verification.json').write_text(json.dumps(report, indent=2))
sys.exit(bool(report['errors']) or report['transformers_installed'])
