import sys
sys.dont_write_bytecode=True
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
root=Path(__file__).resolve().parent
stage=sys.argv[1];stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
(root/'logs').mkdir(exist_ok=True)
with (root/'logs'/f'{stage}_{stamp}.log').open('x') as f:
    cmd=[sys.executable,str(root/(stage+'.py'))] if stage in ['analyze','confirm','visualize','finish_confirmation','timing_validate','review_guide'] else [sys.executable,str(root/'experiment.py'),stage];f.write(f'START {datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}\nCOMMAND {cmd!r}\n');f.flush()
    p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    for line in p.stdout:print(line,end='',flush=True);f.write(line);f.flush()
    rc=p.wait();f.write(f'END {datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()} EXIT {rc}\n')
sys.exit(rc)
