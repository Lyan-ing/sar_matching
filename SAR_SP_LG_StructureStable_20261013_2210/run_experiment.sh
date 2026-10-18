#!/usr/bin/env bash
set -euo pipefail
cd /home/yl/matching/sar_matching/SAR_SP_LG_StructureStable_20261013_2210
mkdir -p logs
for variant in a b control; do
  stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
  CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n match python -B code/train.py --gpu 1 --variant "$variant" > "logs/train_${variant}_${stamp}.log" 2>&1
done
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n match python -B code/finalize.py --gpu 1 > "logs/final_evaluation_${stamp}.log" 2>&1
