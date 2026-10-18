"""Detached tmux timer: lightweight every15min, summary30min; total retries<=1."""
import os,sys,json,time,subprocess,fcntl
from pathlib import Path
from common import ROOT,now,atomic_json
from control import running,start
def restart_eligible(s,retries,checkpoint):
 return s.get('status')=='failed' and s.get('failure_type') in ['oom','exception','unexpected_process_exit'] and retries<1 and checkpoint and not s.get('user_stop')
def snapshot(recover=False):
 result=dict(at=now(),monitor_pid=os.getpid(),interval_seconds=900,summary_interval_seconds=1800,jobs={},notification_channel=None)
 for variant in ['m1','m2']:
  folder=ROOT/'runs'/variant;statefile=folder/'status.json'
  if not statefile.exists():result['jobs'][variant]=dict(status='not_started');continue
  s=json.loads(statefile.read_text());alive=running(s);last=folder/'last.ckpt';best=folder/'best.ckpt';entry=dict(s,process_verified_running=alive,last_checkpoint_exists=last.exists(),best_checkpoint_exists=best.exists(),last_checkpoint_bytes=last.stat().st_size if last.exists() else 0)
  age=time.time()-s.get('heartbeat_unix',time.time());entry['heartbeat_age_seconds']=age
  if alive and age>2700:entry['warning']='No heartbeat45min; inspect current validation/save phase and log before stopping. No automatic kill.'
  if not alive and s.get('status')=='running':
   entry.update(status='failed',failure_type='unexpected_process_exit')
   if recover:
    s.update(status='failed',failure_type='unexpected_process_exit',error='Owned process exited without completion state');atomic_json(statefile,s)
  if s.get('phase')=='failed':entry['warning']=s.get('error')
  retryfile=folder/'recovery.json';retry=json.loads(retryfile.read_text()) if retryfile.exists() else dict(attempts=0)
  requested=(folder/'stop_requested.json').exists()
  entry['automatic_recovery_attempts']=retry['attempts'];entry['user_stop_request']=requested
  if recover and not alive and not requested and restart_eligible(entry,retry['attempts'],last.exists()):
   retry.update(attempts=retry['attempts']+1,at=now(),reason=entry['failure_type'],parameters_changed=False);atomic_json(retryfile,retry)
   try:start(variant,True);entry['recovery']='one bounded explicit checkpoint resume requested'
   except Exception as e:entry['recovery_error']=repr(e)
  result['jobs'][variant]=entry
 try:result['gpu01']=subprocess.check_output(['nvidia-smi','-i','0,1','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv'],text=True,timeout=10)
 except Exception as e:result['gpu_query_error']=repr(e)
 return result
def print_status(result):
 print('SAR training status',result['at'])
 for v,s in result['jobs'].items():print(v,'PID',s.get('pid'),'physicalGPU',s.get('gpu',{}).get('physical_gpu'),'step',s.get('global_step'),'status',s.get('status'),'phase',s.get('phase'),'alive',s.get('process_verified_running'),'loss',s.get('latest_loss'),'best',s.get('best_validation',{}),'last/best',s.get('last_checkpoint_exists'),s.get('best_checkpoint_exists'))
 print(result.get('gpu01',result.get('gpu_query_error','')))
def main(once=False):
 folder=ROOT/'monitor';folder.mkdir(exist_ok=True)
 if once:print_status(snapshot(False));return
 with (folder/'monitor.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);last_summary=0
  while True:
   try:
    result=snapshot(True);atomic_json(folder/'status.json',result)
    if time.time()-last_summary>=1800:
     atomic_json(folder/'summary.json',result);last_summary=time.time();print_status(result)
    atomic_json(folder/'heartbeat.json',dict(pid=os.getpid(),at=now(),next_check_unix=time.time()+900))
    if all(s.get('status')=='completed' for s in result['jobs'].values()):
     atomic_json(folder/'completed.json',dict(at=now(),reason='both trainings completed; no auto benchmark or restart'));break
   except Exception as e:atomic_json(folder/'monitor_error.json',dict(at=now(),error=repr(e)))
   sys.stdout.flush();time.sleep(900)
if __name__=='__main__':main('--once' in sys.argv)
