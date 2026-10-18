#!/usr/bin/env bash
set -euo pipefail
SAR_BASE=/home/yl/matching/sar_matching
SAR_FIXED_RUN=/home/yl/matching/sar_matching/real_sar_pilot/multisource512_20261010_2352
SAR_CODE="$SAR_FIXED_RUN/code_snapshot_final"
SAR_STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
SAR_NEW_RUN="$SAR_BASE/real_sar_pilot/replay512_$SAR_STAMP"
mkdir "$SAR_NEW_RUN"
conda run --no-capture-output -n match python "$SAR_CODE/replay_released_windows.py" --window-manifest "$SAR_FIXED_RUN/released_windows.jsonl" --output "$SAR_NEW_RUN/data" > "$SAR_NEW_RUN/replay_$SAR_STAMP.log" 2>&1
conda run --no-capture-output -n match python "$SAR_CODE/validate_release.py" --run "$SAR_FIXED_RUN" --sources-manifest "$SAR_FIXED_RUN/downloaded_acquisitions.jsonl" --data "$SAR_NEW_RUN/data" --out "$SAR_NEW_RUN/quality" > "$SAR_NEW_RUN/quality_$SAR_STAMP.log" 2>&1
