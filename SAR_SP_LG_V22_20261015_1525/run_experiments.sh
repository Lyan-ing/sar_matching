#!/usr/bin/env bash
set -euo pipefail
cd /home/yl/matching/sar_matching
TASK_ROOT=/home/yl/matching/sar_matching/SAR_SP_LG_V22_20261015_1525
exec 9>"$TASK_ROOT/.driver.lock"
flock -n 9
export CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
TRAIN=(conda run --no-capture-output -n match python -B "$TASK_ROOT/code/train22.py")
DATA_ROOT=/home/yl/matching/sar_matching/SyntheticSAR_V22_20261015_1352
RUN_TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
pids=()
launch() {
 local name=$1; shift
 "${TRAIN[@]}" --gpu 0 --name "$name" "$@" >"$TASK_ROOT/logs/${name}_${RUN_TAG}.log" 2>&1 &
 pids+=("$!")
}
launch m2_core --kind m2
launch b_core --kind b
launch attr_old --kind m2 --fixed --attribution --manifest "$TASK_ROOT/attribution_old_strict_20261015_1536.jsonl"
launch attr_local --kind m2 --fixed --attribution --manifest "$DATA_ROOT/ABLATION_train_core_same_pixels_old_selector_local_20261015_1434.jsonl"
launch attr_roi --kind m2 --fixed --attribution --manifest "$DATA_ROOT/ABLATION_train_core_new_ROI_local_20261015_1434.jsonl"
failed=0
for task_pid in "${pids[@]}"; do wait "$task_pid" || failed=1; done
if (( failed )); then echo 'At least one controlled run failed; inspect preserved logs.'; exit 1; fi
RUN_TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
"${TRAIN[@]}" --gpu 0 --name ship_off --kind b --steps 1000 --initial "$TASK_ROOT/runs/b_core/best.ckpt" >"$TASK_ROOT/logs/ship_off_${RUN_TAG}.log" 2>&1 &
off_pid=$!
"${TRAIN[@]}" --gpu 0 --name ship_on --kind b --steps 1000 --initial "$TASK_ROOT/runs/b_core/best.ckpt" --ship-aux >"$TASK_ROOT/logs/ship_on_${RUN_TAG}.log" 2>&1 &
on_pid=$!
wait "$off_pid"
wait "$on_pid"
for group in main attribution ship; do
 for version in v1 v21 v22; do
  if [[ "$group" != main && "$version" != v22 ]]; then continue; fi
  RUN_TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
  conda run --no-capture-output -n match python -B "$TASK_ROOT/code/final22.py" --gpu 0 --group "$group" --version "$version" >"$TASK_ROOT/logs/evaluate_${group}_${version}_${RUN_TAG}.log" 2>&1
 done
done
RUN_TAG=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
conda run --no-capture-output -n match python -B "$TASK_ROOT/code/report22.py" >"$TASK_ROOT/logs/report_${RUN_TAG}.log" 2>&1
echo 'Controlled training and evaluations completed.'
