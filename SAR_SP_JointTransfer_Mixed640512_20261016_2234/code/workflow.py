"""Detached finite workflow: singles -> audited joint -> one seed repeat. No auto tuning."""
from mixed_io import *
from joint_train import rank,code_identity
from joint_sampling import make_plan
import subprocess

def available():
 rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
 rr=[tuple(map(int,s.split(','))) for s in rows]
 free=[(mem,i) for i,mem,util in rr if i in [0,1] and mem>=10000 and util<=20]
 if free:return max(free)[1]
 # Sharing is allowed only with this workflow's own training, never another job.
 for i,mem,util in rr:
  if i not in [0,1] or mem<10000:continue
  pids=subprocess.check_output(['nvidia-smi','-i',str(i),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
  if pids and all((ROOT.name+'/code/joint_train.py').encode() in Path(f'/proc/{pid.strip()}/cmdline').read_bytes() for pid in pids):return i
 return None

def status(name):
 p=ROOT/'runs'/name/f'status_{TAG}.json';return json.loads(p.read_text()) if p.exists() else {}
def start(kind,name,seed,plan):
 gpu=available()
 while gpu is None:write(ROOT/f'workflow_status_{TAG}.json',dict(stage='waiting_for_free_authorized_GPU',next=name,updated=now(),pid=os.getpid()));time.sleep(30);gpu=available()
 log=ROOT/'logs'/f'train_{name}_{stamp()}.log';f=log.open('x');cmd=[sys.executable,'-B',str(ROOT/'code/joint_train.py'),'--kind',kind,'--run',name,'--seed',str(seed),'--plan',str(plan)]
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu));proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
 write(ROOT/f'launch_{name}_{stamp()}.json',dict(pid=proc.pid,physical_gpu=gpu,command=cmd,log=str(log),started=now()));print(now(),'START',name,proc.pid,gpu,flush=True);return proc,f

def wait(group):
 while any(p.poll() is None for p,f in group.values()):
  write(ROOT/f'workflow_status_{TAG}.json',dict(stage='running',runs={n:status(n) for n in group},updated=now(),pid=os.getpid()));time.sleep(30)
 for name,(p,f) in group.items():
  f.close();assert p.returncode==0,(name,p.returncode);s=status(name);assert s['state']=='completed' and s['step']==6000,(name,s)
  ck=torch.load(s['best_checkpoint'],map_location='cpu',weights_only=False);assert ck['global_step']>0 and ck['frozen_parameters_unchanged'];assert ck['initialization_sha']==sha(WEIGHTS['b']);assert all(torch.isfinite(v).all() for v in ck['model_state'].values());assert sha(s['best_checkpoint'])!=sha(WEIGHTS['b'])
  print(now(),'COMPLETE_AUDITED',name,s['best'],flush=True)

if __name__=='__main__':
 assert len(list(ROOT.glob('training_smoke_*.json'))) and len(list(ROOT.glob('complete_resume_check_*.json')))
 plan=sorted((ROOT/'plans').glob('sampling_seed42_*.jsonl'))[-1];frozen=code_identity();write(ROOT/f'formal_code_freeze_{stamp()}.json',dict(code_sha=frozen,plan_sha=sha(plan),initialization_sha=sha(WEIGHTS['b']),seed=42,steps=6000,effective_batch=8,sequence=['t1+t2','t3','one_best_seed43_repeat'],selection_fixed_before_formal=True))
 group={'t1':start('t1','t1',42,plan)}
 # Wait for first actual update before allowing a second job on the same otherwise-free GPU.
 while status('t1').get('step',0)<1:
  assert group['t1'][0].poll() is None,'First transfer failed before its first update';time.sleep(3)
 group['t2']=start('t2','t2',42,plan);wait(group);assert code_identity()==frozen
 wait({'t3':start('t3','t3',42,plan)});assert code_identity()==frozen
 candidates=[]
 for k in ['t1','t2','t3']:
  s=status(k);candidates.append((bool(s['qualified']),tuple(s['best']['rank']),k))
 chosen=max(candidates)[2];write(ROOT/f'repeat_choice_{stamp()}.json',dict(scope=chosen,criterion='prefer calibrated candidate, then fixed Natural S5/anchor/repeat rank',candidates=candidates,seed=43,no_hyperparameter_search=True))
 repeat_plan=make_plan(6000,43);wait({'repeat':start(chosen,'repeat',43,repeat_plan)})
 write(ROOT/f'workflow_status_{TAG}.json',dict(stage='training_and_selection_completed_pending_final_common_budget_evaluation',runs={n:status(n) for n in ['t1','t2','t3','repeat']},updated=now(),pid=os.getpid()))
 print(now(),'ALL_CONTROLLED_TRAINING_COMPLETED',flush=True)
