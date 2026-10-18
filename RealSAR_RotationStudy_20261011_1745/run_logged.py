"""Timestamp actual commands/output; run with match Python."""
import sys
sys.dont_write_bytecode=True
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    (ROOT/'logs').mkdir(exist_ok=True)
    for stage in sys.argv[1:]:
        stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');command=[sys.executable,'-u',str(ROOT/'study.py'),stage]
        with (ROOT/'logs'/f'{stage}_{stamp}.log').open('x') as log:
            log.write(f'START {datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}\nCOMMAND {command}\n');log.flush();print('RUN',stage,flush=True)
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait();log.write(f'EXIT {code}\nEND {datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()}\n')
            if code:sys.exit(code)
