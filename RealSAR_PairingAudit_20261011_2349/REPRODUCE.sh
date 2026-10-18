#!/usr/bin/env bash
set -euo pipefail
kit_dir=$(cd -- "$(dirname -- "$0")" && pwd)
task_root=$(dirname -- "$kit_dir")
run_stamp=$(date +%Y%m%d_%H%M)
new_output=${1:-"$task_root/RealSAR_PairingAuditReplay_$run_stamp"}
if [ "$(dirname -- "$new_output")" != "$task_root" ]; then
  echo 'Replay output must be a new sibling directory in the existing project root.' >&2
  exit 2
fi
mkdir -- "$new_output"
mkdir -- "$new_output/code" "$new_output/logs" "$new_output/audit" "$new_output/figures" "$new_output/pairs"
cp -- "$kit_dir"/code/*.py "$new_output/code/"
cp -- "$kit_dir/protocol.json" "$kit_dir/selection.json" "$new_output/"
export MPLCONFIGDIR="$new_output/matplotlib_cache"
conda run --no-capture-output -n match python "$new_output/code/audit.py" scan > "$new_output/logs/scan_$(date +%Y%m%d_%H%M).log" 2>&1
conda run --no-capture-output -n match python "$new_output/code/render.py" > "$new_output/logs/render_$(date +%Y%m%d_%H%M).log" 2>&1
conda run --no-capture-output -n match python "$new_output/code/finish.py" > "$new_output/logs/finish_$(date +%Y%m%d_%H%M).log" 2>&1
conda run --no-capture-output -n match python "$new_output/code/test_audit.py" > "$new_output/logs/tests_$(date +%Y%m%d_%H%M).log" 2>&1
conda run --no-capture-output -n match python "$new_output/code/report.py" > "$new_output/logs/report_draft_$(date +%Y%m%d_%H%M).log" 2>&1
# Uses already retained Playwright/Chromium dependencies under /tmp; installs nothing.
conda run --no-capture-output -n match python "$new_output/code/test_gallery_browser.py" > "$new_output/logs/browser_$(date +%Y%m%d_%H%M).log" 2>&1
conda run --no-capture-output -n match python "$new_output/code/report.py" > "$new_output/logs/report_final_$(date +%Y%m%d_%H%M).log" 2>&1
echo "Replayed audit and gallery: $new_output"
