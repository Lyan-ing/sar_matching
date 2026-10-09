#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${SAR_PHASE1_ROOT:-/home/yl/matching/sar_matching/MaritimeSARRegDataset/output/phase1_formal_20261009_1846}"
INDEX="${SAR_PHASE1_INDEX:-/home/yl/matching/sar_matching/MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl}"
STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
RUN="${SAR_PHASE2_OUT:-$PWD/output/phase2_reproduce_$STAMP}"
# mkdir without -p deliberately refuses an existing experiment directory.
mkdir "$RUN"
export CUDA_VISIBLE_DEVICES="${SAR_GPU:-7}"
LOG="$RUN/execution_$STAMP.log"
COMMON=(--root "$ROOT" --source-index "$INDEX" --model homography --seed 42 --threads 1 --max-features 2048 --ratio 0.8 --reproj-px 3 --min-matches 8 --min-inliers 8 --min-inlier-ratio 0.3 --min-coverage 0.0625 --max-inlier-rmse 3 --min-phase-response 0.20)
run_logged() {
  conda run --no-capture-output -n match python scripts/run_logged.py --log "$LOG" -- conda run --no-capture-output -n match python "$@"
}
run_logged -m sarbench inspect --root "$ROOT"
run_logged -m sarbench run "${COMMON[@]}" --methods sift orb phase --out "$RUN/classic"
run_logged -m sarbench run "${COMMON[@]}" --methods lightglue --device cuda --out "$RUN/lightglue"
run_logged -m sarbench compare --runs "$RUN/classic" "$RUN/lightglue" --out "$RUN/comparison" --seed 42
run_logged scripts/finalize_phase2.py --comparison "$RUN/comparison" --log "$LOG"
conda run --no-capture-output -n match python - "$RUN/comparison" "$LOG" <<'PY'
import json, shutil, sys
from pathlib import Path
for f in sorted(Path('reports').glob('phase2_delivery_*.json'),reverse=True):
    d=json.loads(f.read_text())
    if d['comparison']==str(Path(sys.argv[1]).resolve()):
        shutil.copy2(sys.argv[2],d['execution_log']);print(d['report']);break
else:raise RuntimeError('Final delivery manifest missing')
PY
printf 'Completed experiment: %s\n' "$RUN"
