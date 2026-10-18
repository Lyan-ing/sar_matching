#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
conda run --no-capture-output -n match python -B - <<'PY'
import os,json,signal
from pathlib import Path
r=json.loads(Path('runs/m3/status.json').read_text());pid=r['pid']
if r['state'] not in ['training','validation']:raise SystemExit('Task is not running; no signal sent')
assert Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]==r['start_ticks'],'PID has been reused'
assert str(Path.cwd()).encode() in Path(f'/proc/{pid}/cmdline').read_bytes() or b'code/train_m3.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
os.kill(pid,signal.SIGTERM);print('Graceful stop requested; checkpoint saves after current step/validation')
PY
