"""One user-authorized shared-GPU run; no retry and no numerical protocol change."""
from pathlib import Path
import os,subprocess,json
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];TAG='20261017_1343'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(tmp,p)
def main():
 assert not (ROOT/'runs/formal').exists(),'No duplicate training'
 output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
 cards=[]
 for line in output.strip().splitlines():
  idx,uuid,free,util=[s.strip() for s in line.split(',')];cards.append(dict(index=int(idx),uuid=uuid,free_MiB=int(free),utilization=int(util)))
 best=max(cards,key=lambda q:q['free_MiB']);assert best['free_MiB']>=8192
 ts=stamp();state=ROOT/f'QUEUE_STATUS_{TAG}.json'
 if state.exists():(ROOT/f'PREVIOUS_QUEUE_STATE_{ts}.json').write_bytes(state.read_bytes())
 env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(best['index']);cmd=['conda','run','--no-capture-output','-n','match','python','-B',str(ROOT/'code/keep_train.py')]
 write(ROOT/f'ACTUAL_LAUNCH_{ts}.json',dict(updated=now(),selection='greatest free memory across all eight authorized GPUs',cards=cards,selected=best,torch_allocator_cap_GiB=4,command=cmd,prior_training_steps=0))
 with (ROOT/f'logs/formal_lambda0_authorized_{ts}.log').open('x') as log:
  p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);write(state,dict(state='training',updated=now(),pid=p.pid,physical_gpu=best['index'],command=cmd));code=p.wait()
 write(state,dict(state='training_completed' if code==0 else 'training_failed_no_restart',updated=now(),exit_code=code,physical_gpu=best['index']))
if __name__=='__main__':main()
