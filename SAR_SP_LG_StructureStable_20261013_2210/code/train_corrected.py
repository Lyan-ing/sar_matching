"""Same seed/budget protocol: continuation control, structure-only A, jointly stable B."""
import argparse,time,signal,traceback
from collections import defaultdict
from evaluation import *
from network_corrected import joint_loss,load
class RawDensityEvaluator(FairEvaluator):
 def evaluate(self,*args,**kwargs):
  self.native_counts=[]
  return super().evaluate(*args,**kwargs)
 @torch.no_grad()
 def predict(self,model,a,b,tiled=False):
  self.native_counts.append(len(model.extract(torch.tensor(a.copy(),device='cuda').float()[None,None]/255)['keypoints'][0]))
  return super().predict(model,a,b,tiled)
def corrected_key(s,sanity,baseline):
 old=quality_key(s,sanity,baseline)
 return (int(bool(old[0]) and s['native_keypoints0']<=2*baseline['keypoints0']),)+old[1:]
STOP=False
def stopping(*args):
 global STOP
 STOP=True

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',choices=['a','b','control'],required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
 c=json.loads(output('config').read_text());run=ROOT/'runs'/a.variant;run.mkdir(parents=True,exist_ok=True)
 assert not output('selection_locked',folder=run).exists(),'Frozen completed model cannot retrain'
 ident=gpu(a.gpu);torch.manual_seed(c['seed']);random.seed(c['seed']);np.random.seed(c['seed']);event('launch',dict(process=ident,variant=a.variant),run)
 signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
 m=load(a.variant);teacher=load('m2');params=[p for p in m.parameters() if p.requires_grad];groups=[]
 for prefix,lr in [('structure',c['structure_lr']),('sp.convP',c['detector_lr']),('sp.convD',c['descriptor_control_lr'])]:
  ps=[p for n,p in m.named_parameters() if p.requires_grad and n.startswith(prefix)]
  if ps:groups.append(dict(params=ps,lr=lr))
 o=torch.optim.AdamW(groups,weight_decay=c['weight_decay'],foreach=False);sched=torch.optim.lr_scheduler.LambdaLR(o,lambda s:1.);scaler=torch.amp.GradScaler('cuda',enabled=False)
 plan=json.loads(output('plan_select').read_text());ev=RawDensityEvaluator(plan);baseline_records=json.loads(output('baseline_records').read_text());ids={s['id'] for s in plan};baseline=summarize([r for r in baseline_records if r['id'] in ids and r['method']=='m2'])
 frozen={k:v.cpu().clone() for k,v in m.state_dict().items() if k not in {n for n,p in m.named_parameters() if p.requires_grad}}
 pool=defaultdict(lambda:defaultdict(list))
 for r in rows(OLD/'train_sources.jsonl'):
  category='nearshore' if r['scene_type']=='inshore' else ('background' if r['dataset']=='background' else 'offshore');pool[category][component(r)].append(r)
 step=0;patience=0;best=None;key=None;elapsed0=0;start=time.perf_counter();history=[]
 if a.resume:
  ck=torch.load(run/'last.ckpt',map_location='cuda',weights_only=False);assert ck['config']==c;m.load_state_dict(ck['model_state']);o.load_state_dict(ck['optimizer_state'])
  for st in o.state.values():
   if 'step' in st:st['step']=st['step'].cpu()
  sched.load_state_dict(ck['scheduler_state']);scaler.load_state_dict(ck['amp_scaler']);restore_rng(ck['random_state']);step=ck['global_step'];best=ck['best'];key=tuple(ck['selection_key']);patience=ck['patience'];elapsed0=ck['elapsed_seconds']
 if a.resume:
  # Recompute acceptance of the previous best using the corrected native-count gate; no threshold/hyperparameter change.
  oldbest=load(a.variant,run/'best.ckpt');oldrows=ev.evaluate(oldbest,a.variant,'density_gate_correction',save=False);bs=summarize(oldrows);bs['native_keypoints0']=float(np.mean([n for n,r in zip(ev.native_counts,oldrows) if not r['no_overlap']]));san=probes(oldbest);key=corrected_key(bs,san,baseline);best=dict(step=best['step'],summary=bs,probes=san,hard_gate_pass=bool(key[0]));patience=0
  event('acceptance_gate_correction',dict(best=best,selection_key=key,resume_step=step,no_loss_hyperparameter_change=True),run);del oldbest
 metrics=output('metrics',ext='csv',folder=run)
 if metrics.exists():metrics=run/f'metrics_resume_{stamp()}.csv'
 fields=['step','elapsed_seconds','loss','match','preserve','detector','calibration','blank','equivariance','n_points','gradient_norm','gpu_allocated_MiB'];mf=metrics.open('x',newline='');w=csv.DictWriter(mf,fieldnames=fields);w.writeheader()
 def ckpt():return dict(variant=a.variant,model_state=m.state_dict(),optimizer_state=o.state_dict(),scheduler_state=sched.state_dict(),amp_scaler=scaler.state_dict(),global_step=step,epoch=step*c['accumulation']/5111,random_state=rng_state(),config=c,best=best,selection_key=key,patience=patience,elapsed_seconds=elapsed0+time.perf_counter()-start,code_identity=json.loads(output('code_identity').read_text()),data_identity=json.loads(output('data_identity').read_text()))
 def state(s,reason=''):event('status',dict(state=s,reason=reason,variant=a.variant,global_step=step,best=best,elapsed_seconds=elapsed0+time.perf_counter()-start,process=ident),run)
 reason=''
 try:
  while step<c['steps'] and not STOP:
   m.train();o.zero_grad(set_to_none=True);infos=[]
   for micro in range(c['accumulation']):
    seed=c['seed']+step*20+micro;rng=np.random.default_rng(seed);category=rng.choice(['nearshore','offshore','background'],p=[.55,.35,.1]);cs=sorted(pool[category]);rr=pool[category][cs[int(rng.integers(len(cs)))]];r=rr[int(rng.integers(len(rr)))];regime='easy' if step<500 else ('medium' if step<1500 else rng.choice(['easy','medium','low'],p=[.2,.6,.2]));z=pair(r,seed,regime,dynamic_occlusion=step>=1500 and bool(r['annotations']) and rng.random()<.15)
    loss,info=joint_loss(m,teacher,z);assert torch.isfinite(loss),'NaN/Inf loss';(loss/c['accumulation']).backward();infos.append(info)
   for n,p in m.named_parameters():
    if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all(),n
    else:assert p.grad is None,n
   gn=float(torch.nn.utils.clip_grad_norm_(params,1.));o.step();sched.step();step+=1
   if step==1 or step%20==0:
    row={k:float(np.mean([r[k] for r in infos])) for k in fields if k in infos[0]};row.update(step=step,elapsed_seconds=elapsed0+time.perf_counter()-start,gradient_norm=gn,gpu_allocated_MiB=torch.cuda.max_memory_allocated()/2**20);w.writerow(row);mf.flush();print(f'{now()} {a.variant} step={step} loss={row["loss"]:.4f}',flush=True)
   if step==100 or step%c['validate_every']==0:
    m.eval();state('validation');rr=ev.evaluate(m,a.variant,f'step{step}',save=False);summ=summarize(rr);summ['native_keypoints0']=float(np.mean([n for n,r in zip(ev.native_counts,rr) if not r['no_overlap']]));san=probes(m);newkey=corrected_key(summ,san,baseline);history.append(dict(step=step,summary=summ,probe=san,selection_key=newkey));event(f'validation_step{step}',history[-1],run)
    print(f'{now()} {a.variant} VALIDATION '+json.dumps(history[-1]),flush=True)
    if key is None or newkey>key:key=newkey;best=dict(step=step,summary=summ,probes=san,hard_gate_pass=bool(newkey[0]));patience=0;save_ckpt(run/'best.ckpt',ckpt())
    else:patience+=1
    if step>=c['early_stop_min'] and patience>=c['patience']:reason='validation_plateau';break
   if step%100==0:save_ckpt(run/'last.ckpt',ckpt());state('training')
   if elapsed0+time.perf_counter()-start>14400:reason='four_hour_cap';break
  reason=reason or ('user_stop' if STOP else 'fixed_3000_step_budget');save_ckpt(run/'last.ckpt',ckpt())
  intact=all(torch.equal(v,m.state_dict()[k].cpu()) for k,v in frozen.items());assert intact
  event('frozen_parameters',dict(passed=True,frozen_tensors=len(frozen),trainable_params=sum(p.numel() for p in params),trainable=[n for n,p in m.named_parameters() if p.requires_grad]),run)
  if not STOP:write_new(output('selection_locked',folder=run),dict(best=best,best_sha256=sha(run/'best.ckpt'),last_sha256=sha(run/'last.ckpt'),final_step=step,reason=reason,elapsed_seconds=elapsed0+time.perf_counter()-start,selection_only='179 development selection pairs, no holdout/real/900'))
  state('stopped' if STOP else 'completed',reason)
 except BaseException as ex:state('failed',repr(ex));print(traceback.format_exc(),flush=True);raise
 finally:mf.close()
if __name__=='__main__':main()
