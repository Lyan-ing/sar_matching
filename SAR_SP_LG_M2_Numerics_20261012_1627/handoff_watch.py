"""CPU-only status publication; never starts, stops or evaluates any model."""
import json,sys,time,fcntl,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from common import now,atomic_json
from control import running
BASE=ROOT.parent;DOCUMENT=BASE/'PROJECT_HANDOFF.md'
BEGIN='<!-- TRAINING_STATUS_BEGIN -->';END='<!-- TRAINING_STATUS_END -->'
def update():
 jobs={};lines=[f'自动刷新：{now()}；实时权威记录见下列status路径。','', '|模型|状态|step|Python PID / 核验存活|物理GPU|权威状态路径|','|---|---|---:|---|---:|---|']
 for variant in ['m1','m2']:
  path=ROOT/'runs'/variant/'status.json';actual=path.resolve()
  s=json.loads(path.read_text()) if path.exists() else {'status':'prepared_not_started'}
  # M1 is completed in its original experiment; do not apply new-root command identity.
  alive=running(s) if variant=='m2' else False
  entry=dict(s,process_verified_running=alive,status_path=str(actual));jobs[variant]=entry
  state=s['status']
  if state=='running' and not alive:state='running_state_but_process_not_verified'
  gpu=s.get('gpu',{}).get('physical_gpu',0 if variant=='m1' else 1)
  lines.append(f"|{variant}|{state}|{s.get('global_step','—')}|{s.get('pid','—')} / {alive}|{gpu}|`{actual.relative_to(BASE)}`|")
 lines+=['',f'查询：`cd {ROOT}; bash check_training.sh`。运行状态以PID出生时间和命令核验为准；非运行行PID仅为历史身份。','M1已完成；M2新目录只修复描述子头数值精度，原失败记录不覆盖。GPU可能还有无关任务，勿停止无关PID。']
 with Path('/tmp/sar_project_handoff.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  text=DOCUMENT.read_text();assert text.count(BEGIN)==text.count(END)==1
  first,rest=text.split(BEGIN);_,last=rest.split(END)
  temporary=DOCUMENT.with_name(DOCUMENT.name+'.tmp.'+str(os.getpid()))
  with temporary.open('w') as f:f.write(first+BEGIN+'\n'+'\n'.join(lines)+'\n'+END+last);f.flush();os.fsync(f.fileno())
  os.replace(temporary,DOCUMENT)
 atomic_json(BASE/'PROJECT_TRAINING_STATUS.json',dict(at=now(),watcher_pid=os.getpid(),runtime=str(ROOT),jobs=jobs,refresh_seconds=60))
 return jobs
if __name__=='__main__':
 while True:
  try:update()
  except Exception as e:print(now(),repr(e),flush=True)
  if '--once' in sys.argv:break
  time.sleep(60)
