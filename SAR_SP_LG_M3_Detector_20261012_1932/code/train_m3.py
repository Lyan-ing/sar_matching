"""Single GPU detector-only FP32 training; bounded, evidence-based extensions and resumable state."""
import argparse,csv,time,traceback,signal
from collections import defaultdict
from m3_common import *
from m3_model import load_m3,detector_loss
from m3_evaluate import Evaluator,summarize,selection
RUN=ROOT/'runs/m3';STOP=False
def stop(*_):
 global STOP
 STOP=True
def optimizer(m,c):return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=c['learning_rate'],weight_decay=c['weight_decay'],foreach=False)
def pack(m,o,s,scaler,step,c,best,current,patience,elapsed,budget):
 return dict(variant='m3',model_state=m.state_dict(),optimizer_state=o.state_dict(),scheduler_state=s.state_dict(),amp_scaler=scaler.state_dict(),global_step=step,epoch=step*c['batch_accumulation']/5111,random_state=rng_state(),config=c,data_identity=json.loads((ROOT/'data_identity.json').read_text()),code_identity=json.loads((ROOT/'code_freeze.json').read_text()),best_validation=best,current_validation=current,early_stop_patience_used=patience,elapsed_training_seconds=elapsed,current_step_budget=budget,model_structure='M2 pretrained frozen encoder/descriptor; original convPa/convPb trainable only',created=now())
def restore(p,m,o,s,scaler,c):
 x=torch.load(p,map_location='cuda',weights_only=False);assert x['variant']=='m3' and x['config']==c
 assert x['code_identity']==json.loads((ROOT/'code_freeze.json').read_text()),'Running code identity changed'
 m.load_state_dict(x['model_state']);o.load_state_dict(x['optimizer_state'])
 # Non-capturable AdamW creates CPU step counters; undo map_location CUDA for exact arithmetic replay.
 for state in o.state.values():
  if 'step' in state:state['step']=state['step'].cpu()
 s.load_state_dict(x['scheduler_state']);scaler.load_state_dict(x['amp_scaler']);restore_rng(x['random_state']);return x
def train_pair_pool():
 pool=defaultdict(lambda:defaultdict(list))
 for r in rows(OLD/'train_sources.jsonl'):
  category='nearshore' if r['scene_type']=='inshore' else ('background' if r['dataset']=='background' else 'offshore')
  pool[category][component(r)].append(r)
 return pool
def training_pair(pool,seed,step):
 rng=np.random.default_rng(seed);category=rng.choice(['nearshore','offshore','background'],p=[.55,.35,.10]);groups=pool[category];cs=sorted(groups);sources=groups[cs[int(rng.integers(len(cs)))]];r=sources[int(rng.integers(len(sources)))]
 regime='easy' if step<500 else ('medium' if step<1500 else rng.choice(['easy','medium','low'],p=[.2,.6,.2]))
 return pair(r,seed,regime,dynamic_occlusion=step>=1500 and bool(r['annotations']) and rng.random()<.15)
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 c=json.loads((ROOT/'config.json').read_text());assert a.gpu==c['physical_gpu'];assert not (RUN/'selection_locked.json').exists(),'Completed model must not retrain'
 for f,h in json.loads((ROOT/'code_freeze.json').read_text()).items():assert sha(ROOT/f)==h,'Source identity changed'
 process=gpu(a.gpu);atomic_json(RUN/'launch.json',process);signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
 m=load_m3();o=optimizer(m,c);s=torch.optim.lr_scheduler.LambdaLR(o,lambda step:1.0);scaler=torch.amp.GradScaler('cuda',enabled=False);pool=train_pair_pool();ev=Evaluator()
 baseline=json.loads((ROOT/'baseline_summary.json').read_text())['m2'];initial=None;best=None;current=None;patience=0;step=0;elapsed0=0;budget=c['initial_step_budget'];history=[]
 if a.resume:
  x=restore(RUN/'last.ckpt',m,o,s,scaler,c);step=x['global_step'];best=x['best_validation'];current=x['current_validation'];patience=x['early_stop_patience_used'];elapsed0=x['elapsed_training_seconds'];budget=x['current_step_budget']
  history=[json.loads(f.read_text())['summary'] for f in sorted((RUN/'validation').glob('step_*.json'))]
 frozen={k:v.detach().cpu().clone() for k,v in m.state_dict().items() if not k.startswith(('sp.convPa.','sp.convPb.'))};start=time.perf_counter()
 mf=(RUN/'metrics.csv').open('a',newline='');writer=csv.DictWriter(mf,fieldnames=['step','elapsed_seconds','loss','peak','background','equivariance','preserve','anchors','grad_norm','max_allocated_MiB'])
 if mf.tell()==0:writer.writeheader()
 def elapsed():return elapsed0+time.perf_counter()-start
 def status(state,reason=''):
  atomic_json(RUN/'status.json',dict(**process,state=state,global_step=step,step_budget=budget,heartbeat=now(),elapsed_seconds=elapsed(),best_validation=best,current_validation=current,reason=reason,max_gpu_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,max_gpu_reserved_MiB=torch.cuda.max_memory_reserved()/2**20))
 reason=''
 try:
  while not STOP and elapsed()<c['max_training_seconds']:
   status('training');m.train();o.zero_grad(set_to_none=True);metrics=[]
   for micro in range(c['batch_accumulation']):
    z=training_pair(pool,c['seed']+step*20+micro,step);loss,info=detector_loss(m,z,c);assert torch.isfinite(loss),'Nonfinite loss';(loss/c['batch_accumulation']).backward();metrics.append(info)
   for name,p in m.named_parameters():
    if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),f'Nonfinite or missing gradient {name}'
    else:assert p.grad is None,f'Unexpected frozen gradient {name}'
   gn=float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.));o.step();s.step();step+=1
   if step%10==0 or step==1:
    row={k:float(np.mean([r[k] for r in metrics])) for k in ['loss','peak','background','equivariance','preserve','anchors']};row.update(step=step,elapsed_seconds=elapsed(),grad_norm=gn,max_allocated_MiB=torch.cuda.max_memory_allocated()/2**20);writer.writerow(row);mf.flush();print(f'{now()} M3 step={step} loss={row["loss"]:.4f} anchors={row["anchors"]:.1f}',flush=True)
   if step==100 or step%c['validate_every']==0:
    status('validation');m.eval();rr=ev.evaluate(m,'m3',f'step_{step:06d}',save=False);current=summarize(rr);current['step']=step;history.append(current)
    vp=RUN/'validation'/f'step_{step:06d}.json';atomic_json(vp,dict(summary=current,rows=rr));print(f'{now()} M3 validation '+json.dumps(current),flush=True)
    if best is None or selection(current)>selection(best):
     best=current.copy();patience=0;save_ckpt(RUN/'best.ckpt',pack(m,o,s,scaler,step,c,best,current,patience,elapsed(),budget))
    else:patience+=1
    if step>=c['early_stop_min_steps'] and patience>=c['early_stop_patience']:reason='validation_plateau';break
   if step%100==0:save_ckpt(RUN/'last.ckpt',pack(m,o,s,scaler,step,c,best,current,patience,elapsed(),budget))
   if step>=budget:
    recent=history[-3:];effective=best is not None and best['category_component_S5']>baseline['category_component_S5'] and best['no_overlap_false_accept']<=baseline['no_overlap_false_accept']
    improving=len(recent)==3 and recent[-1]['category_component_S5']-recent[0]['category_component_S5']>=.005 and recent[-1]['correct_LG']>recent[0]['correct_LG']
    if effective and improving:
     old=budget;budget+=c['extension_steps'];atomic_json(RUN/'extensions'/f'step_{step:06d}.json',dict(previous_budget=old,new_budget=budget,baseline=baseline,recent=recent,condition_effective=effective,condition_improving=improving,created=now()));print(f'VALIDATION EVIDENCE EXTENSION {old}->{budget}',flush=True)
    else:reason='step_budget_no_evidence_for_extension';break
  reason=reason or ('user_stop' if STOP else 'four_hour_budget')
  save_ckpt(RUN/'last.ckpt',pack(m,o,s,scaler,step,c,best,current,patience,elapsed(),budget))
  unchanged=all(torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in frozen.items());assert unchanged,'Frozen model changed'
  atomic_json(RUN/'frozen_parameter_verification.json',dict(passed=unchanged,checked_tensors=len(frozen),trainable=[n for n,p in m.named_parameters() if p.requires_grad],trainable_params=sum(p.numel() for p in m.parameters() if p.requires_grad)))
  if not STOP:
   assert best is not None;atomic_json(RUN/'selection_locked.json',dict(best_step=best['step'],best_sha256=sha(RUN/'best.ckpt'),last_sha256=sha(RUN/'last.ckpt'),validation=best,rule=c['selection_rule'],effective_vs_M2=best['category_component_S5']>baseline['category_component_S5'],reason=reason,locked=now()))
  status('stopped' if STOP else 'completed',reason)
 except BaseException as e:
  status('failed',repr(e));print(traceback.format_exc(),flush=True);raise
 finally:mf.close()
if __name__=='__main__':main()
