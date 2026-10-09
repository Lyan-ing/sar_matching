"""Run authorized commands with a timestamped, streaming experiment log."""
import argparse,datetime,os,shlex,subprocess,sys
from pathlib import Path
from zoneinfo import ZoneInfo
p=argparse.ArgumentParser();p.add_argument('--log',required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if not cmd:raise SystemExit('Missing command')
now=lambda:datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
log=Path(a.log);log.parent.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1',MPLCONFIGDIR=str(Path('cache/matplotlib').resolve()),TORCH_HOME=str(Path('cache/torch').resolve()),PYTHONUNBUFFERED='1',PIP_CACHE_DIR='/tmp/sarbench-pip-cache')
with log.open('a') as f:
 def emit(s):f.write(s);f.flush();print(s,end='',flush=True)
 emit(f'\n[{now()}] cwd={Path.cwd()}\n$ {shlex.join(cmd)}\n')
 proc=subprocess.Popen(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
 for line in proc.stdout:emit(line)
 code=proc.wait();emit(f'[{now()}] exit={code}\n')
sys.exit(code)
