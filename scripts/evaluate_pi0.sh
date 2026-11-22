#!/usr/bin/env bash
# Evaluate the pinned local pretrained checkpoint across the four standard suites.
set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1
export OMP_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=/root/autodl-tmp/lerobot-env-records/libero-config
/root/autodl-tmp/envs/lerobot/bin/lerobot-eval \
  --seed=1000 \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --policy.path=/root/autodl-tmp/models/pi0_libero_finetuned_v044 \
  --policy.n_action_steps=10 \
  --policy.device=cuda \
  --output_dir="${1:-/root/autodl-tmp/outputs/eval/pi0_v44s1ka10}" \
  --env.max_parallel_tasks=1
