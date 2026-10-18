#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
conda run --no-capture-output -n match python -B - <<'PY'
from pathlib import Path
import json
for p in [Path('mechanism_status_20261017_0908.json'),Path('runs/formal/status_20261017_0908.json'),Path('postprocess_status_20261017_0908.json')]:
 if p.exists():
  s=json.loads(p.read_text());print(p,{k:s[k] for k in ['stage','state','step','total_steps','updated','best','qualified'] if k in s})
validation=sorted(Path('runs/formal').glob('validation_step*.json'))
if validation:
 s=json.loads(validation[-1].read_text());print('Latest completed Val:',s['step'],'Natural S5:',s['metrics']['natural']['all']['Success@5px'],'Equal S5:',s['metrics']['equal']['all']['Success@5px'],'qualified:',s['qualified'])
done=sorted(Path('.').glob('TASK_COMPLETED_*.json'))
if done:
 s=json.loads(done[-1].read_text());print('Overall completed:',s['completed']);print('Report:',s['report']);print('PDF:',s['pdf']);print('Do not restart completed training.')
PY
