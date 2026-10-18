#!/usr/bin/env bash
set -euo pipefail
cd /home/yl/matching/sar_matching
while tmux has-session -t sar_v22_train 2>/dev/null; do
 conda run --no-capture-output -n match python -B - <<'PY'
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
r=Path('SAR_SP_LG_V22_20261015_1525');t=datetime.now(ZoneInfo('Asia/Shanghai'));rr=[]
for p in r.glob('runs/*/status_*.json'):
 s=json.loads(p.read_text());pid=s.get('pid');q=Path(f'/proc/{pid}/cmdline');s['run']=p.parent.name;s['live_cmdline']=q.read_bytes().replace(b'\0',b' ').decode() if q.exists() else None;rr.append(s)
p=r/'logs'/('monitor_'+t.strftime('%Y%m%d_%H%M')+'.json')
if not p.exists():p.write_text(json.dumps(dict(at=t.isoformat(),runs=rr),indent=2))
PY
 sleep 60
done
