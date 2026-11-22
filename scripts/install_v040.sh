#!/usr/bin/env bash
# Install the requested editable LeRobot stack, deliberately leaving Transformers to the user.
set -euo pipefail
ROOT=/root/autodl-tmp
ENV="$ROOT/envs/lerobot"
RECORDS="$ROOT/lerobot-env-records"
/root/miniconda3/bin/python "$RECORDS/prepare_v040.py"
"$ENV/bin/python" -m pip install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r "$RECORDS/requirements-no-transformers.in"
cd "$ROOT/lerobot-src/lerobot"
"$ENV/bin/python" -m pip install --no-deps --no-build-isolation -e '.[libero]'
# pi in v0.4.0 only adds the custom Transformers fork, intentionally skipped here.
"$ENV/bin/python" -m pip install --no-deps --no-build-isolation --index-url https://pypi.org/simple -e '.[pi]'
cd "$ROOT/lerobot-src/LIBERO"
"$ENV/bin/python" -m pip install --no-deps --no-build-isolation --index-url https://pypi.org/simple -e .
# LIBERO declares Transformers as required; this is intentionally pending user installation.
"$ENV/bin/python" -m pip check > "$RECORDS/pip-check.txt" 2>&1 || true
if grep -vE '^libero 0\.1\.4 requires transformers, which is not installed\.$|^No broken requirements found\.$' "$RECORDS/pip-check.txt" | grep -q .; then
  cat "$RECORDS/pip-check.txt"
  exit 1
fi
"$ENV/bin/python" -m pip freeze > "$RECORDS/requirements-lock.txt"
git -C "$ROOT/lerobot-src/lerobot" rev-parse HEAD > "$RECORDS/lerobot-commit.txt"
git -C "$ROOT/lerobot-src/LIBERO" rev-parse HEAD > "$RECORDS/libero-commit.txt"
printf 'Installation complete; Transformers intentionally deferred.\n'
