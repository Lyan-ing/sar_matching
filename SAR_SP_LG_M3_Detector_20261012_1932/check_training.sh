#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
cat runs/m3/status.json
if [ -f runs/m3/monitor_status.json ]; then cat runs/m3/monitor_status.json; fi
if [ -f runs/m3/metrics.csv ]; then tail -4 runs/m3/metrics.csv; fi
