#!/usr/bin/env bash
set -euo pipefail
cd /home/yl/matching/sar_matching
TASK_ROOT=/home/yl/matching/sar_matching/SAR_SP_LG_V22_20261015_1525
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
until [[ -f "$TASK_ROOT/runs/m2_core/selection_locked_20261015_1525.json" && -f "$TASK_ROOT/runs/b_core_evidence/selection_locked_20261015_1525.json" ]]; do sleep 15; done
for version in v1 v21 v22; do
 RUN_TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
 conda run --no-capture-output -n match python -B "$TASK_ROOT/code/final_evidence22.py" --gpu 0 --group main --version "$version" >"$TASK_ROOT/logs/evaluate_main_${version}_${RUN_TAG}.log" 2>&1
done
