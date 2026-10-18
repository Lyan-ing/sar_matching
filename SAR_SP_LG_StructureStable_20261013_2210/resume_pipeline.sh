#!/usr/bin/env bash
set -euo pipefail
cd /home/yl/matching/sar_matching/SAR_SP_LG_StructureStable_20261013_2210
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n match python -B code/train_corrected.py --gpu 1 --variant b --resume > "logs/train_b_resume_${stamp}.log" 2>&1
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n match python -B code/train.py --gpu 1 --variant control > "logs/train_control_${stamp}.log" 2>&1
stamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n match python -B code/finalize.py --gpu 1 > "logs/final_evaluation_${stamp}.log" 2>&1
