#!/usr/bin/env bash
# Recycle stalled connections while preserving every completed or partial range.
set -euo pipefail
PY=/root/autodl-tmp/envs/lerobot/bin/python
RECORDS=/root/autodl-tmp/lerobot-env-records
for attempt in $(seq 1 40); do
  echo "Download round $attempt"
  if timeout --signal=TERM 90 "$PY" -u "$RECORDS/download_weight_ranges.py"; then
    "$PY" -u "$RECORDS/download_model.py"
    exit 0
  fi
  echo 'Reconnecting from saved ranges'
  sleep 3
done
echo 'Download did not complete within retry limit' >&2
exit 1
