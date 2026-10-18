#!/usr/bin/env bash
set -euo pipefail
cd /home/yl/matching/sar_matching/SAR_SP_LG_StructureStable_20261013_2210
for variant in a b control; do
  if [ -d "runs/$variant" ]; then
    latest=$(ls -t "runs/$variant"/status_*.json 2>/dev/null | head -1)
    if [ -n "$latest" ]; then cat "$latest"; fi
  fi
done
