"""Stage 1 baseline smoke test: official evaluator, task 0 per suite, without the proposed smoother."""
from functools import partial
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
os.environ['LIBERO_CONFIG_PATH'] = '/root/autodl-tmp/lerobot-env-records/libero-config'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '10'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
import lerobot.envs.libero as libero_env
original_create = libero_env.create_libero_envs
def create_one_task(*args, **kwargs):
    """Limit the environment factory through its documented gym_kwargs hook."""
    kwargs['gym_kwargs'] = {**(kwargs.get('gym_kwargs') or {}), 'task_ids': [0]}
    return original_create(*args, **kwargs)
libero_env.create_libero_envs = create_one_task
from lerobot.scripts.lerobot_eval import main
if __name__ == '__main__':
    main()
