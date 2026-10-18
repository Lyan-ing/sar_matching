#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
variant=${1:?m1 or m2}; shift
case "$variant" in m1) gpu=0;; m2) gpu=1;; *) exit 2;; esac
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu PYTHONDONTWRITEBYTECODE=1
mkdir -p "$root/runs/$variant"
log="$root/runs/$variant/train_${BASHPID}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M).log"
# Exact launch creates a unique timestamped log; never truncate an existing log.
set -o noclobber
set +e
/home/yl/miniconda3/bin/conda run --no-capture-output -n match python -B "$root/code/train.py" "$variant" "$@" > "$log" 2>&1
result=$?
set -e
printf '{"exit_status":%s,"physical_gpu":%s,"ended_at":"%s"}\n' "$result" "$gpu" "$(TZ=Asia/Shanghai date -Iseconds)" > "$root/runs/$variant/wrapper_exit_${BASHPID}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M).json"
exit "$result"
