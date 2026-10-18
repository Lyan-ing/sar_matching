#!/usr/bin/env bash
set -euo pipefail
SAR_BASE=/home/yl/matching/sar_matching
SAR_FIXED_RUN=/home/yl/matching/sar_matching/real_sar_pilot/multisource512_20261010_2352
SAR_CODE="$SAR_FIXED_RUN/code_snapshot_final"
SAR_MAIN_BUILDER="$SAR_FIXED_RUN/code_snapshot/build_real_pairs.py"
SAR_STAMP=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
SAR_NEW_RUN="$SAR_BASE/real_sar_pilot/reproduce512_$SAR_STAMP"
mkdir "$SAR_NEW_RUN"
conda run --no-capture-output -n match python "$SAR_MAIN_BUILDER" --manifest "$SAR_FIXED_RUN/verified_sources.csv" --output "$SAR_NEW_RUN/main" --chip-size 512 --chips-per-pair 3 --seed 260512 --max-attempts 150 > "$SAR_NEW_RUN/main_$SAR_STAMP.log" 2>&1
conda run --no-capture-output -n match python "$SAR_CODE/validate_release.py" --run "$SAR_FIXED_RUN" --data "$SAR_NEW_RUN/main" --out "$SAR_NEW_RUN/quality" > "$SAR_NEW_RUN/quality_$SAR_STAMP.log" 2>&1
conda run --no-capture-output -n match python "$SAR_MAIN_BUILDER" --manifest "$SAR_FIXED_RUN/s1_sf_extension/verified_sources.csv" --output "$SAR_NEW_RUN/san_francisco" --chip-size 512 --chips-per-pair 3 --seed 260512 --max-attempts 150 > "$SAR_NEW_RUN/san_francisco_$SAR_STAMP.log" 2>&1
conda run --no-capture-output -n match python "$SAR_CODE/validate_release.py" --run "$SAR_FIXED_RUN/s1_sf_extension" --data "$SAR_NEW_RUN/san_francisco" --out "$SAR_NEW_RUN/san_francisco_quality" > "$SAR_NEW_RUN/san_francisco_quality_$SAR_STAMP.log" 2>&1
conda run --no-capture-output -n match python "$SAR_CODE/frozen_image_diagnostics.py" --run "$SAR_NEW_RUN" --data "$SAR_NEW_RUN/main" --project "$SAR_BASE/MaritimeSARRegistrationPhase2" > "$SAR_NEW_RUN/diagnostics_$SAR_STAMP.log" 2>&1
conda run --no-capture-output -n match python "$SAR_CODE/build_real_pairs.py" --manifest "$SAR_FIXED_RUN/melbourne_zero_aware/verified_sources.csv" --output "$SAR_NEW_RUN/melbourne_zero_aware" --chip-size 512 --chips-per-pair 3 --seed 260513 --max-attempts 30 --allow-valid-zero > "$SAR_NEW_RUN/melbourne_$SAR_STAMP.log" 2>&1
conda run --no-capture-output -n match python "$SAR_CODE/validate_release.py" --run "$SAR_FIXED_RUN" --data "$SAR_NEW_RUN/melbourne_zero_aware" --out "$SAR_NEW_RUN/melbourne_quality" > "$SAR_NEW_RUN/melbourne_quality_$SAR_STAMP.log" 2>&1
