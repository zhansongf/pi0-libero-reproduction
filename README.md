# Pi0 / LIBERO reproduction on AutoDL

This repository collects the **existing pretrained-policy evaluation work** for
`lerobot/pi0_libero_finetuned_v044` on the LIBERO benchmark. It contains environment
records, download/build scripts, compatibility patches, evaluation commands, and
the actual small smoke-test results. No model was trained or fine-tuned in this work.

**Status:** four-suite smoke evaluation passed at the program level. A full
40-task / 400-episode benchmark and final report have not been provided. This is
an intermediate code snapshot, not a claim of completed benchmark reproduction.
Missing work is listed at the end and is intentionally left for the project author.

## Existing files

- `scripts/`: installation, source build, downloads, compatibility fix and evaluation.
- `requirements.txt`: resolved third-party dependencies from the actual environment;
  LeRobot, LIBERO and the special Transformers source build are installed separately.
- `environment/requirements-lock.raw.txt`: original `pip freeze`, including local
  source/wheel references. This is a record, not a portable one-command installer.
- `patches/pi0-eval-compatibility.patch`: actual changes against the pinned LeRobot.
- `provenance/`: source commits, file manifests/checksums and environment verification.
- `results/`: original smoke-test JSON metrics and five small recorded videos.

The scripts preserve the tested AutoDL absolute paths under `/root/autodl-tmp`.
They are not generic installers for arbitrary operating systems. Commands below
describe reconstruction of the recorded setup; a clean-machine reconstruction has
not yet been tested.

## Recorded environment and fixed resources

| Component | Recorded version / revision |
|---|---|
| OS | Ubuntu 22.04.5 LTS, Linux x86_64 |
| Python | 3.10.22 in the project environment |
| PyTorch / CUDA runtime | 2.7.1 / 12.6 |
| torchvision / torchcodec | 0.22.1 / 0.5 |
| LeRobot | v0.4.0, `f25ac02e6c8fa9c467ab8462289e5f4aed3a2e85`, plus this repository's patch |
| LIBERO integration | `huggingface/lerobot-libero`, `14fd6f022ff6647defc24e73563b95aa36670f43` |
| Transformers | 4.53.3, special `fix/lerobot_openpi` source, `dcddb970176382c0fcf4521b0c0e6fc15894dfe0` |
| MuJoCo / robosuite | 3.3.2 / 1.4.0 |
| Final smoke-test GPU | AutoDL NVIDIA vGPU-48GB, 49,140 MiB, driver 580.105.08 |
| Model revision | `45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f` |
| Dataset revision | `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |

An earlier single-task diagnostic ran on an AutoDL RTX 4090 with 48 GB reported
VRAM. The committed completed smoke tests were run after reconnecting to the vGPU.

## Reconstruct the recorded AutoDL setup

Prerequisites: an AutoDL Ubuntu instance, NVIDIA driver, Git, and the existing
Miniconda base at `/root/miniconda3` (Python 3.12 was used for the dependency
extraction helper). Use a GPU instance for evaluation. The 2 GiB no-GPU instance
could install packages and run lightweight checks but could not run a full robot scene.

Clone this public project and stage its existing scripts at their tested location:

```bash
git clone https://github.com/zhansongf/pi0-libero-reproduction.git
cd pi0-libero-reproduction
mkdir -p /root/autodl-tmp/lerobot-env-records /root/autodl-tmp/lerobot-src
cp scripts/* /root/autodl-tmp/lerobot-env-records/

/root/miniconda3/bin/conda create -y -p /root/autodl-tmp/envs/lerobot python=3.10.22 pip
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/envs/lerobot
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libegl1 libgl1 libglib2.0-0 libosmesa6

git clone https://github.com/huggingface/lerobot.git /root/autodl-tmp/lerobot-src/lerobot
git -C /root/autodl-tmp/lerobot-src/lerobot checkout f25ac02e6c8fa9c467ab8462289e5f4aed3a2e85
git clone https://github.com/huggingface/lerobot-libero.git /root/autodl-tmp/lerobot-src/LIBERO
git -C /root/autodl-tmp/lerobot-src/LIBERO checkout 14fd6f022ff6647defc24e73563b95aa36670f43

python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install --no-deps --no-build-isolation -e /root/autodl-tmp/lerobot-src/lerobot
python -m pip install --no-deps --no-build-isolation -e /root/autodl-tmp/lerobot-src/LIBERO
# Run BEFORE installing Transformers. Also creates the LIBERO path configuration.
python /root/autodl-tmp/lerobot-env-records/verify_v040.py
```

The alternate original installation helpers `prepare_v040.py` and `install_v040.sh`
are preserved as records of the initial installation. The pinned `requirements.txt`
above is preferable when reconstructing the resolved environment.

Download the exact [Transformers source archive](https://github.com/huggingface/transformers/archive/dcddb970176382c0fcf4521b0c0e6fc15894dfe0.zip)
and upload it as `/root/autodl-tmp/transformers-dcddb970176382c0fcf4521b0c0e6fc15894dfe0.zip`.
The following existing script safely extracts it, builds a wheel, installs it and
records checksums. It expects that the destination source directory does not yet exist.

```bash
python /root/autodl-tmp/lerobot-env-records/install_transformers_uploaded.py
python /root/autodl-tmp/lerobot-env-records/verify_transformers.py
python -m pip check
# Apply ONCE to an unmodified pinned LeRobot checkout; backs up both changed files.
python /root/autodl-tmp/lerobot-env-records/patch_pi0_eval.py
```

Expected: Transformers `4.53.3`, patch check `true`, successful `PI0Policy` import,
and `No broken requirements found`. The compatibility changes allow configured
virtual cameras to be generated by PI0 and restore actual tied-parameter aliases
omitted by safetensors. Weight-loading errors are made fatal. Model weights are
not altered. The original source backups and generated run files are not committed.

## Model and data preparation

- [Pinned Pi0 checkpoint](https://huggingface.co/lerobot/pi0_libero_finetuned_v044/tree/45dcc8fc0e02601c8ccf0554fbd1d26a55070c1f):
  nine files, 7,011,579,889 bytes including metadata.
- [Pinned LIBERO demonstration dataset](https://huggingface.co/datasets/lerobot/libero/tree/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4):
  457 files, 1,935,512,060 bytes; 40 tasks, 1,693 episodes. This is training data;
  simulation evaluation of the pretrained checkpoint does not require it.

```bash
python /root/autodl-tmp/lerobot-env-records/download_model.py
# Optional fallback used when whole-weight downloads stalled; resumes saved ranges.
bash /root/autodl-tmp/lerobot-env-records/run_model_download.sh
# Optional demonstrations, not required for pretrained simulation evaluation:
python /root/autodl-tmp/lerobot-env-records/download_dataset.py
```

Download scripts use `hf-mirror.com` for these public resources, fix revisions,
and validate sizes plus SHA-256 / Git blob hashes. They disable implicit token use.
Expected destinations are `/root/autodl-tmp/models/pi0_libero_finetuned_v044` and
`/root/autodl-tmp/datasets/libero`. Manifests are committed in `provenance/`.
The dataset download/verification took about 302 seconds in the recorded completed
download. Model download time varied substantially with network failures and
retries; its post-download verification took about 21 seconds. A reliable total
installation/model-download time has not been recorded.

Pi0 preprocessing also needs the gated `google/paligemma-3b-pt-224` tokenizer.
Using your own Hugging Face account, review the conditions and obtain access on
the [official model page](https://huggingface.co/google/paligemma-3b-pt-224).
On AutoDL, use its official network acceleration when needed, then authenticate
interactively. Credentials are local only; no credentials are in this repository.

```bash
source /etc/network_turbo
hf auth login
hf auth whoami
```

## Reproduce the existing smoke tests

These are setup checks, not benchmark scores. The wrapper calls the official
evaluator and limits each requested suite to task 0 through its `gym_kwargs`
filter. Compilation is disabled for these short tests. Use a new output directory
if the specified one already exists. Activate the project environment first.

```bash
source /etc/network_turbo
export TOKENIZERS_PARALLELISM=false
python /root/autodl-tmp/lerobot-env-records/eval_smoke.py \
  --seed=1000 --env.type=libero --env.task=libero_spatial \
  --eval.batch_size=1 --eval.n_episodes=1 \
  --policy.path=/root/autodl-tmp/models/pi0_libero_finetuned_v044 \
  --policy.n_action_steps=10 --policy.device=cuda --policy.compile_model=false \
  --output_dir=/root/autodl-tmp/outputs/eval/pi0_smoke_authorized_20261122 \
  --env.max_parallel_tasks=1

python /root/autodl-tmp/lerobot-env-records/eval_smoke.py \
  --seed=1000 --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 --eval.n_episodes=1 \
  --policy.path=/root/autodl-tmp/models/pi0_libero_finetuned_v044 \
  --policy.n_action_steps=10 --policy.device=cuda --policy.compile_model=false \
  --output_dir=/root/autodl-tmp/outputs/eval/pi0_four_suite_smoke_20261122 \
  --env.max_parallel_tasks=1
```

Expected outputs in each run directory: `eval_info.json` and
`videos/<suite>_0/eval_episode_0.mp4`. Original outputs are under `results/`.

| Smoke test | Episodes | Recorded successes | Evaluator-reported time |
|---|---:|---:|---:|
| Spatial task 0 only | 1 | 0 / 1 | 20.88 s |
| Four suites, task 0 in each | 4 | 2 / 4 | 56.81 s |

In the four-suite test, Object and Goal succeeded; Spatial and Long did not.
These times exclude part of model/environment initialization and are not total
shell wall-clock times. Success can vary with hardware/numerics; the raw results
are the observations from the recorded run, not guaranteed outcomes.

## Full benchmark command (prepared, NOT yet run)

```bash
bash /root/autodl-tmp/lerobot-env-records/evaluate_pi0.sh
```

This preserves seed 1000, four suites, ten episodes per task, batch size 1,
`n_action_steps=10`, CUDA, and one parallel task: 40 tasks / 400 episodes.
Output defaults to `/root/autodl-tmp/outputs/eval/pi0_v44s1ka10/`; an alternate
directory may be supplied as the script's first argument. Unlike the smoke tests,
this command does not override the checkpoint's compilation setting. Full-run
duration, success rates, and compilation behavior have not been measured.

## Missing items intentionally left for manual addition

- The final report and its exact chosen experiments/comparisons.
- Full benchmark results, logs, total wall-clock runtime and any multi-seed runs.
- Plotting/analysis scripts and report figures; none existed at collection time.
- Training/data preprocessing code and training outputs, if training becomes part
  of the project. The current work only evaluates a pretrained policy.
- Fresh-machine end-to-end validation and any portability improvements.
- Additional module/function comments or documentation required by the final rubric.
- A final submission tag after the author completes the missing work.

The course's WebVPN page could not be read automatically; only the requirements
provided in the conversation were used. Review any additional course instructions
before final submission. Cite the public repository URL and the exact submitted
commit hash (or a final tag) in Canvas. This intermediate snapshot is not marked final.

## Upstream sources and licenses

This work builds on [LeRobot](https://github.com/huggingface/lerobot),
[LIBERO integration](https://github.com/huggingface/lerobot-libero), and
[Transformers](https://github.com/huggingface/transformers). Source implementations
are obtained from their pinned upstream repositories instead of duplicated here.
Their original license notices and the Gemma model/tokenizer conditions continue
to apply. No new project-wide license has been selected in this snapshot.
