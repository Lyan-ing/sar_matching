#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PHASE1_DATA_ROOT="$(realpath "${1:-../dataset}")"
PHASE1_PYTHON="${PHASE1_PYTHON:-$PWD/.venv/bin/python}"
PHASE1_STAMP="${PHASE1_STAMP:-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)}"
PHASE1_INDEX="data/index/phase1_${PHASE1_STAMP}"
PHASE1_OUTPUT="output/phase1_${PHASE1_STAMP}"
PHASE1_LOG="reports/phase1_execution_${PHASE1_STAMP}.log"
if [[ -e "$PHASE1_INDEX" || -e "$PHASE1_OUTPUT" || -e "$PHASE1_LOG" ]]; then
  echo "Run already exists for $PHASE1_STAMP; preserve it and use another minute timestamp." >&2
  exit 2
fi
mkdir -p reports
exec > >(tee "$PHASE1_LOG") 2>&1
trap 'status=$?; echo "[$(TZ=Asia/Shanghai date -Is)] exit=$status"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 PYTHONUNBUFFERED=1
set -x
TZ=Asia/Shanghai date -Is
df -h . "$PHASE1_DATA_ROOT"
git rev-parse HEAD
"$PHASE1_PYTHON" -m pytest -q | tee "reports/phase1_pytest_${PHASE1_STAMP}.log"
"$PHASE1_PYTHON" -m sarreg prepare --dataset hrsid --root "$PHASE1_DATA_ROOT/HRSID" --split-policy group --seed 42 --out "$PHASE1_INDEX/hrsid.jsonl"
"$PHASE1_PYTHON" -m sarreg prepare --dataset ssdd --root "$PHASE1_DATA_ROOT/SSDD" --sea-mask-root "$PHASE1_DATA_ROOT/SL-SSDD" --split-policy official --seed 42 --out "$PHASE1_INDEX/ssdd.jsonl"
"$PHASE1_PYTHON" -m sarreg prepare --dataset background --root "$PHASE1_DATA_ROOT/HRSID/pure background" --split-policy group --seed 42 --out "$PHASE1_INDEX/background.jsonl"
"$PHASE1_PYTHON" -m sarreg merge-index --indexes "$PHASE1_INDEX/hrsid.jsonl" "$PHASE1_INDEX/ssdd.jsonl" "$PHASE1_INDEX/background.jsonl" --out "$PHASE1_INDEX/maritime.jsonl"
# A detected exact-byte split leak is recorded and repaired as a NEW explicit
# experiment protocol. Any image, annotation or mask error stops the pipeline.
set +e
"$PHASE1_PYTHON" scripts/audit_real_data.py --index "$PHASE1_INDEX/maritime.jsonl" --out "reports/phase1_data_audit_${PHASE1_STAMP}.json"
audit_status=$?
set -e
if (( audit_status > 1 )); then exit "$audit_status"; fi
"$PHASE1_PYTHON" scripts/make_leakage_safe_index.py --index "$PHASE1_INDEX/maritime.jsonl" --audit "reports/phase1_data_audit_${PHASE1_STAMP}.json" --out "$PHASE1_INDEX/maritime_leakage_safe.jsonl" --report "reports/phase1_split_repair_${PHASE1_STAMP}.json"
sources=(ssdd hrsid background maritime)
counts=(20 20 10 24)
seeds=(42 43 44 45)
previews=(10 10 5 12)
for i in 0 1 2 3; do
  source="${sources[$i]}"
  target="$PHASE1_OUTPUT/smoke_${source}_${counts[$i]}"
  "$PHASE1_PYTHON" -m sarreg generate --index "$PHASE1_INDEX/$source.jsonl" --config configs/easy.yaml --split all --count "${counts[$i]}" --seed "${seeds[$i]}" --out "$target"
  "$PHASE1_PYTHON" -m sarreg validate --root "$target"
  "$PHASE1_PYTHON" -m sarreg visualize --root "$target" --out "$PHASE1_OUTPUT/previews_${source}_${counts[$i]}" --count "${previews[$i]}"
done
"$PHASE1_PYTHON" scripts/build_phase1_examples.py --index "$PHASE1_INDEX/maritime_leakage_safe.jsonl" --out "$PHASE1_OUTPUT/coverage_examples_8" --seed 46
"$PHASE1_PYTHON" -m sarreg validate --root "$PHASE1_OUTPUT/coverage_examples_8"
"$PHASE1_PYTHON" -m sarreg visualize --root "$PHASE1_OUTPUT/coverage_examples_8" --out "$PHASE1_OUTPUT/previews_coverage_8" --count 8
"$PHASE1_PYTHON" -m sarreg generate --index "$PHASE1_INDEX/maritime_leakage_safe.jsonl" --config configs/phase1_three_view.yaml --split all --count 6 --seed 47 --out "$PHASE1_OUTPUT/smoke_three_view_6"
"$PHASE1_PYTHON" -m sarreg validate --root "$PHASE1_OUTPUT/smoke_three_view_6"
"$PHASE1_PYTHON" scripts/summarize_phase1.py --root "$PHASE1_OUTPUT" --index "$PHASE1_INDEX/maritime.jsonl" --stamp "$PHASE1_STAMP"
