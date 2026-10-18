#!/usr/bin/env bash
set -euo pipefail
cd /home/yl/matching/sar_matching
TASK_ROOT=/home/yl/matching/sar_matching/SAR_SP_LG_V22_20261015_1525
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
while true; do
 complete=1
 for name in m2_core b_core_evidence attr_old attr_local attr_roi; do
  if [[ ! -f "$TASK_ROOT/runs/$name/selection_locked_20261015_1525.json" ]]; then complete=0; fi
 done
 if (( complete )); then break; fi
 sleep 30
done
RUN_TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
TRAIN=(conda run --no-capture-output -n match python -B "$TASK_ROOT/code/train_ship_evidence22.py")
"${TRAIN[@]}" --gpu 0 --name ship_off --kind b --steps 1000 --initial "$TASK_ROOT/runs/b_core_evidence/best.ckpt" >"$TASK_ROOT/logs/ship_off_${RUN_TAG}.log" 2>&1 &
off_pid=$!
"${TRAIN[@]}" --gpu 0 --name ship_on --kind b --steps 1000 --initial "$TASK_ROOT/runs/b_core_evidence/best.ckpt" --ship-aux >"$TASK_ROOT/logs/ship_on_${RUN_TAG}.log" 2>&1 &
on_pid=$!
wait "$off_pid"
wait "$on_pid"
for group in main attribution ship; do
 for version in v1 v21 v22; do
  if [[ "$group" != main && "$version" != v22 ]]; then continue; fi
  RUN_TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
  conda run --no-capture-output -n match python -B "$TASK_ROOT/code/final_evidence22.py" --gpu 0 --group "$group" --version "$version" >"$TASK_ROOT/logs/evaluate_${group}_${version}_${RUN_TAG}.log" 2>&1
 done
done
RUN_TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
conda run --no-capture-output -n match python -B "$TASK_ROOT/code/report22.py" >"$TASK_ROOT/logs/report_${RUN_TAG}.log" 2>&1
echo 'Controlled training and evaluations completed.'
