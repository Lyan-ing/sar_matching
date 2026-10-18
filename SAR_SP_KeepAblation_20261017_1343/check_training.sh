#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
conda run --no-capture-output -n match python -B - <<'PY'
from pathlib import Path
import json
for p in [Path('QUEUE_STATUS_20261017_1343.json'),Path('runs/formal/status_20261017_1343.json')]:
 if p.exists():
  s=json.loads(p.read_text()); print(p); print({k:s[k] for k in ['state','step','total_steps','updated','pid','physical_gpu','best','occupancy','exit_code'] if k in s})
PY
