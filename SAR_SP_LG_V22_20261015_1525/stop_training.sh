#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
conda run --no-capture-output -n match python -B - "$1" <<'PY'
from pathlib import Path
import json,os,signal,sys
name=sys.argv[1];assert name.isidentifier()
p=Path('SAR_SP_LG_V22_20261015_1525/runs')/name/'status_20261015_1525.json';s=json.loads(p.read_text());assert s['state'] in ['training','validation'];pid=s['pid'];cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert any(v in cmd for v in ['train22.py','train_evidence22.py','train_ship_evidence22.py']) and ('--name '+name) in cmd and 'SAR_SP_LG_V22_20261015_1525' in cmd
os.kill(pid,signal.SIGTERM);print('Graceful stop requested',name,pid)
PY
