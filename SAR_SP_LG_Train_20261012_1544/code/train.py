"""Independent single-GPU jobs, deterministic online pairs, atomic full-state resume."""
import os,sys,json,random,time,signal,traceback,csv,math
from pathlib import Path
import numpy as np,torch,cv2
from common import ROOT,now,stamp,sha,readrows,atomic_json,ensure_gpu,proc_start
from model import SARModel,loss
from data import pair
from evaluate import model_validation
from checkpoint import pack,atomic_save,restore

def run(variant,resume=False):
 cfg=json.loads((ROOT/'configs'/f'{variant}.json').read_text());run=ROOT/'runs'/variant;run.mkdir(exist_ok=True)
 freeze=json.loads((ROOT/'code_freeze.json').read_text())
 for p,h in freeze['sha256'].items():
  if sha(ROOT/p)!=h:raise RuntimeError('Training code/config changed: '+p)
 for p,h in freeze['external_sha256'].items():
  if sha(p)!=h:raise RuntimeError('Frozen reused inference code changed: '+p)
 for p,key in [('train_sources.jsonl','training_sources_sha256'),('validation_plan.json','validation_plan_sha256'),('experiment_identity.json','model_identity_sha256')]:
  if sha(ROOT/p)!=freeze[key]:raise RuntimeError('Frozen data/model identity changed: '+p)
 gpu=ensure_gpu(cfg['physical_gpu']);torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.backends.cudnn.benchmark=False
 model=SARModel(variant).cuda();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg['max_steps'],eta_min=cfg['learning_rate']*.1);scaler=torch.amp.GradScaler('cuda',enabled=True)
 sources=readrows(ROOT/'train_sources.jsonl');plan=json.loads((ROOT/'validation_plan.json').read_text());identity=json.loads((ROOT/'experiment_identity.json').read_text());best=None;current=None;step=0;unimproved=0
 if resume:
  c=restore(run/'last.ckpt',model,optimizer,scheduler,scaler,cfg);step=c['global_step'];best=c['best_validation'];current=c['current_validation'];unimproved=int(c.get('unimproved',0))
 elif (run/'last.ckpt').exists():raise RuntimeError('Existing training run must explicitly resume')
 started=now();pid=os.getpid();stop=[False];signal.signal(signal.SIGTERM,lambda *a:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *a:stop.__setitem__(0,True))
 state=dict(variant=variant,pid=pid,process_start_ticks=proc_start(pid),process_started_at=started,gpu=gpu,run_id=cfg['run_id'],config_sha256=sha(ROOT/'configs'/f'{variant}.json'),checkpoint=str(run/'last.ckpt'),status='running',phase='initial_validation',global_step=step,current_validation=current,best_validation=best)
 def heartbeat(phase,**kwargs):
  state.update(phase=phase,heartbeat_at=now(),heartbeat_unix=time.time(),global_step=step,peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,**kwargs);atomic_json(run/'status.json',state)
 def save_ckpt(path):
  heartbeat('saving');content=pack(model,optimizer,scheduler,scaler,step,cfg,identity,best,current);content['unimproved']=unimproved;atomic_save(path,content)
 def metrics(values):
  p=run/'metrics.csv';fields=['time','step','train_loss','match_loss','preserve_loss','train_recall1','validation_recall1','validation_recall5','validation_success3','validation_success5','validation_success10','validation_false_accept_none','validation_repeatability3','learning_rate','GPU_peak_GiB']
  with p.open('a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields)
   if f.tell()==0:w.writeheader()
   row=dict(time=now(),step=step,learning_rate=optimizer.param_groups[0]['lr'],GPU_peak_GiB=torch.cuda.max_memory_allocated()/2**30,**values);w.writerow(row);f.flush();os.fsync(f.fileno())
 def validate():
  nonlocal best,current,unimproved
  heartbeat('validating');result=model_validation(model,plan,output=run/f'validation_step_{step:06d}.json');current=result['summary'];improved=best is None or tuple(current['selection_score'])>tuple(best['selection_score'])
  if improved:best=dict(current,best_step=step);unimproved=0;save_ckpt(run/'best.ckpt')
  else:unimproved+=1
  metrics(dict(validation_recall1=current['descriptor_recall1'],validation_recall5=current['descriptor_recall5'],validation_success3=current['Success@3px'],validation_success5=current['Success@5px'],validation_success10=current['Success@10px'],validation_false_accept_none=current['none_false_accept_rate'],validation_repeatability3=current['detector_repeatability_3px']))
  from curves import plot_metrics
  plot_metrics(run)
  heartbeat('training',current_validation=current,best_validation=best);print('VALIDATION',step,json.dumps(current),flush=True)
 pools={}
 for r in sources:
  group='nearshore' if r['scene_type']=='inshore' else 'background' if r['dataset']=='background' else 'offshore';pools.setdefault(group,{}).setdefault(r['training_component'],[]).append(r)
 try:
  if step==0:
   baseline=model_validation(model,plan,output=run/'M0_initial_validation.json');current=baseline['summary'];heartbeat('training',M0_validation=current);save_ckpt(run/'initial.ckpt')
  while step<cfg['max_steps'] and not stop[0]:
   model.train();optimizer.zero_grad(set_to_none=True);totals=[]
   for micro in range(cfg['batch_accumulation']):
    for attempt in range(20):
     seed=cfg['seed']+step*200+micro*20+attempt;rng=np.random.default_rng(seed);category=str(rng.choice(['nearshore','offshore','background'],p=[.55,.35,.10]));groups=pools[category];group=sorted(groups)[int(rng.integers(len(groups)))];r=groups[group][int(rng.integers(len(groups[group])))];regime='easy' if step<600 else ('medium' if step<1800 else str(rng.choice(['easy','medium','low'],p=[.2,.6,.2])))
     z=pair(r,seed,regime,dynamic_occlusion=step>=1800 and bool(r['annotations']) and rng.random()<.15);item=loss(model,z,'cuda',cfg['temperature'],cfg['preserve_weight'])
     if item is not None:break
    else:raise RuntimeError('Cannot construct valid positive geometry after20 attempts')
    value,stats=item
    if not torch.isfinite(value):raise FloatingPointError('Nonfinite training loss')
    scaler.scale(value/cfg['batch_accumulation']).backward();totals.append((float(value.detach()),stats))
   scaler.unscale_(optimizer);grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
   if not torch.isfinite(grad):raise FloatingPointError('Nonfinite gradient')
   scaler.step(optimizer);scaler.update();scheduler.step();step+=1
   if step%10==0:
    av=float(np.mean([v[0] for v in totals]))
    metrics(dict(train_loss=av,match_loss=float(np.mean([v[1]['match'] for v in totals])),preserve_loss=float(np.mean([v[1]['preserve'] for v in totals])),train_recall1=float(np.mean([v[1]['recall1'] for v in totals]))))
    heartbeat('training',latest_loss=av,grad_norm=float(grad));print('STEP',step,'loss',av,'grad',float(grad),flush=True)
   if step==10 or step%cfg['checkpoint_every']==0:save_ckpt(run/'last.ckpt');heartbeat('training')
   if step==25 or step%cfg['validate_every']==0:validate();save_ckpt(run/'last.ckpt');heartbeat('training')
   if step>=cfg['early_stop_min_steps'] and unimproved>=cfg['early_stop_patience']:state['completion_reason']='validation_plateau';break
  save_ckpt(run/'last.ckpt')
  if stop[0]:heartbeat('stopped',status='stopped',user_stop=True)
  else:
   if not current or best is None or current.get('best_step')!=step:
    if not (run/f'validation_step_{step:06d}.json').exists():validate()
   save_ckpt(run/'last.ckpt');heartbeat('completed',status='completed',completed_at=now(),selection_locked=True,best_validation=best)
   atomic_json(run/'selection_locked.json',dict(best_checkpoint=str(run/'best.ckpt'),best_sha256=sha(run/'best.ckpt'),best_validation=best,locked_at=now(),frozen900_runs=0,formal_test_deferred_to_second_delivery=True))
 except Exception as e:
  typ='nonfinite' if isinstance(e,FloatingPointError) else 'oom' if isinstance(e,torch.OutOfMemoryError) else 'exception'
  traceback.print_exc();heartbeat('failed',status='failed',failure_type=typ,error=repr(e),failed_at=now());raise

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('variant',choices=['m1','m2']);p.add_argument('--resume',action='store_true');a=p.parse_args();run(a.variant,a.resume)
