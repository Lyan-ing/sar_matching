"""One finite run, only when an authorized GPU is entirely free; no automatic restart."""
from pathlib import Path
import subprocess,os,time,json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];TAG='20261017_1343'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def status(obj):
 p=ROOT/f'QUEUE_STATUS_{TAG}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(updated=now(),**obj),ensure_ascii=False,indent=2));os.replace(tmp,p)
def main():
 assert not (ROOT/'runs/formal'/f'status_{TAG}.json').exists(),'Refuse existing run'
 checks=sorted(ROOT.glob('PREFLIGHT_CPU_*.json'));assert checks and json.loads(checks[-1].read_text())['passed']
 deadline=time.monotonic()+12*3600
 while time.monotonic()<deadline:
  if (ROOT/'STOP_QUEUE').exists():status(dict(state='stopped_without_launch'));return
  found=None;occupancy=[]
  for g in [0,1]:
   q=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip();free,util=map(int,q.split(','))
   pids=subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
   occupancy.append(dict(gpu=g,free_MiB=free,utilization=util,pids=pids))
   if not pids and free>=10000 and util<=20:found=g;break
  if found is None:
   status(dict(state='waiting_for_entirely_free_GPU0_or_1',occupancy=occupancy,no_training_started=True));time.sleep(60);continue
  env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(found);cmd=['conda','run','--no-capture-output','-n','match','python','-B',str(ROOT/'code/keep_train.py')]
  status(dict(state='launching_one_finite_run',physical_gpu=found,command=cmd))
  with (ROOT/f'logs/formal_lambda0_{TAG}.log').open('x') as f:p=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT);status(dict(state='training',pid=p.pid,physical_gpu=found,command=cmd));code=p.wait()
  status(dict(state='training_completed' if code==0 else 'training_failed_no_restart',exit_code=code,physical_gpu=found));return
 status(dict(state='resource_wait_expired',training_started=False,no_restart=True))
if __name__=='__main__':main()
