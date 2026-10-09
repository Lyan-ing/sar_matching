#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${SAR_PHASE1_ROOT:-/home/yl/matching/sar_matching/MaritimeSARRegDataset/output/phase1_formal_20261009_1846}"
INDEX="${SAR_PHASE1_INDEX:-/home/yl/matching/sar_matching/MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl}"
STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
OUT="${SAR_PHASE2_OUT:-$PWD/output/phase2_classic_$STAMP}"
conda run --no-capture-output -n match python scripts/run_logged.py --log "reports/phase2_execution_$STAMP.log" -- conda run --no-capture-output -n match python -m sarbench run --root "$ROOT" --source-index "$INDEX" --methods sift orb phase --model homography --seed 42 --threads 1 --out "$OUT"
