"""Launch/query/stop owned tmux jobs with PID-reuse protection."""
import os,sys,json,subprocess,shlex,signal,time
from common import ROOT,proc_start,now,atomic_json
def running(s):
 p=s.get('pid');birth=s.get('process_start_ticks')
 if not p or not birth or proc_start(p)!=birth:return False
 try:
  command=open(f'/proc/{p}/cmdline','rb').read().replace(b'\0',b' ').decode()
  return str(ROOT/'code/train.py') in command and s['variant'] in command
 except OSError:return False
def start(variant,resume=False):
 assert variant in ['m1','m2'];session='sar_'+variant+'_train';folder=ROOT/'runs'/variant;status=folder/'status.json'
 if status.exists() and running(json.loads(status.read_text())):raise RuntimeError('Owned training is already running')
 if not (ROOT/'preflight'/variant/'result.json').exists() or not json.loads((ROOT/'preflight'/variant/'result.json').read_text())['passed']:raise RuntimeError('Real preflight not passed')
 if (folder/'last.ckpt').exists() and not resume:raise RuntimeError('Existing checkpoint; explicit --resume required')
 if resume and not (folder/'last.ckpt').exists():raise RuntimeError('No last.ckpt to restore')
 check=subprocess.run(['tmux','has-session','-t',session],capture_output=True)
 if check.returncode==0:
  dead=subprocess.check_output(['tmux','display-message','-p','-t',session,'#{pane_dead}'],text=True).strip()
  if dead!='1':raise RuntimeError('Live tmux session name already in use; never overwrite')
  command=subprocess.check_output(['tmux','display-message','-p','-t',session,'#{pane_start_command}'],text=True)
  if str(ROOT) not in command:raise RuntimeError('Existing dead tmux session not owned by this experiment')
  subprocess.run(['tmux','kill-session','-t',session],check=True)
 args=['bash',str(ROOT/'run_job.sh'),variant]+(['--resume'] if resume else [])
 subprocess.run(['tmux','new-session','-d','-s',session,shlex.join(args)],check=True)
 subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
 stop_marker=folder/'stop_requested.json'
 if resume and stop_marker.exists():stop_marker.rename(folder/('stop_request_resumed_'+str(time.time_ns())+'.json'))
 atomic_json(folder/'launch.json',dict(session=session,launched_at=now(),resume=resume,command=args,physical_gpu=0 if variant=='m1' else 1))
 print('LAUNCHED',session,'resume='+str(resume),flush=True)
def stop(variant):
 p=ROOT/'runs'/variant/'status.json'
 if not p.exists():print('No running status');return
 s=json.loads(p.read_text())
 if not running(s):print('Process not running or identity mismatch; nothing signalled');return
 atomic_json(p.parent/'stop_requested.json',dict(at=now(),pid=s['pid'],process_start_ticks=s['process_start_ticks'],user_requested=True))
 os.kill(s['pid'],signal.SIGTERM);print('SIGTERM sent only to owned verified PID',s['pid'],'; checkpoint saved at the next safe step boundary')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop']);p.add_argument('variant',choices=['m1','m2']);p.add_argument('--resume',action='store_true');a=p.parse_args();start(a.variant,a.resume) if a.action=='start' else stop(a.variant)
