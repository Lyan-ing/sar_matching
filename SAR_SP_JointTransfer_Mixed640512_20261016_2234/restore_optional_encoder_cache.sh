#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
if [ -L cache/train ]; then
  target=$(readlink cache/train)
  case "$target" in /dev/shm/sar_joint_mixed_train_*) ;; *) echo 'Unexpected cache target; refusing changes.' >&2; exit 2;; esac
  if [ -d cache/train ]; then echo 'Verified RAM encoder cache still exists; no regeneration needed.'; exit 0; fi
  unlink cache/train
fi
mkdir -p cache/train cache/train_supervision
: "${CUDA_VISIBLE_DEVICES:?Set one truly free authorized GPU0 or GPU1 first}"
exec conda run --no-capture-output -n match python -B code/cache_training_inputs.py
