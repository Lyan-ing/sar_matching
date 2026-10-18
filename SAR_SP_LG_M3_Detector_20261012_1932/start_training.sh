#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
test -f preflight/result.json
test -f baseline_summary.json
test ! -f runs/m3/selection_locked.json
if tmux has-session -t sar_m3_train 2>/dev/null; then echo 'sar_m3_train already exists'; exit 1; fi
mode=""
if [ "${1:-}" = '--resume' ]; then mode='--resume'; test -f runs/m3/last.ckpt; fi
logfile="logs/train_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M).log"
tmux new-session -d -s sar_m3_train "cd '$PWD' && CUDA_VISIBLE_DEVICES=5 /home/yl/miniconda3/bin/conda run --no-capture-output -n match python -B code/train_m3.py --gpu 5 $mode > '$logfile' 2>&1"
if ! tmux has-session -t sar_m3_monitor 2>/dev/null; then
 tmux new-session -d -s sar_m3_monitor "cd '$PWD' && CUDA_VISIBLE_DEVICES='' /home/yl/miniconda3/bin/conda run --no-capture-output -n match python -B code/watch_m3.py > 'logs/monitor_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M).log' 2>&1"
fi
echo "Started sar_m3_train on physical GPU5; log $logfile"
