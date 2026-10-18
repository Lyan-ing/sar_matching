#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
conda run --no-capture-output -n match python -B - <<'PY'
from pathlib import Path
import json
r=Path('SAR_SP_LG_V22_20261015_1525')
for p in sorted(r.glob('runs/*/status_*.json')):
 s=json.loads(p.read_text());pid=s.get('pid');q=Path(f'/proc/{pid}/cmdline');cmd=q.read_bytes().replace(b'\0',b' ').decode() if q.exists() else '';live='SAR_SP_LG_V22_20261015_1525' in cmd and ('--name '+p.parent.name) in cmd;print(p.parent.name,s['state'],'step',s['step'],'calls',s['consumed_calls'],'PID',pid,'live',live,'best',s.get('best',{}).get('step') if s.get('best') else None,'at',s['at'])
PY
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
