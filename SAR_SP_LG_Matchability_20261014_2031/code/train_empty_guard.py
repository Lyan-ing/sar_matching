from common import *
from net import load_model,joint_loss
from pairdata import pair
from evaluate import Evaluator,summary,probes
import signal,traceback
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=['control','joint','frozen'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();identity=gpu(a.gpu);c=json.loads(output('config').read_text());run=ROOT/'runs'/a.variant;run.mkdir(parents=True,exist_ok=True)
 assert not output('selection_locked',folder=run).exists(),'Completed model is frozen'
 freeze=json.loads((ROOT/'code_freeze_empty_guard_20261014_2134.json').read_text())
 for path,h in freeze['sha256'].items():assert sha(path)==h,path
 torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed']);m=load_model(a.variant);teacher=load_model('m2');ev=Evaluator();plan=[s for s in json.loads(output('validation_plan').read_text()) if s['split']=='select'];sources=readrows(OLD/'train_sources.jsonl');schedule=json.loads(output('source_schedule').read_text());counts=np.zeros(len(sources),int)
 groups=[]
 for prefix,lr in [('sp.convP',c['detector_lr']),('sp.convD',c['descriptor_lr']),('sp.conv4',c['shared_lr'])]:
  ps=[p for n,p in m.named_parameters() if p.requires_grad and n.startswith(prefix)]
  if ps:groups.append(dict(params=ps,lr=lr,name=prefix))
 opt=torch.optim.AdamW(groups,weight_decay=c['weight_decay'],foreach=False);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=c['steps'],eta_min=0);scaler=torch.amp.GradScaler('cuda',enabled=False)
 frozen={n:p.detach().cpu().clone() for n,p in m.named_parameters() if not p.requires_grad};step=0;best=None;key=None;elapsed0=0;start=time.perf_counter();stop=[False]
 signal.signal(signal.SIGTERM,lambda *args:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *args:stop.__setitem__(0,True))
 if a.resume:
  ck=torch.load(run/'last.ckpt',map_location='cuda',weights_only=False);assert ck['config']==c;m.load_state_dict(ck['model_state']);opt.load_state_dict(ck['optimizer_state']);sched.load_state_dict(ck['scheduler_state']);scaler.load_state_dict(ck['amp_scaler']);restore_rng(ck['random_state']);step=ck['global_step'];best=ck['best'];key=tuple(ck['selection_key']) if ck['selection_key'] is not None else None;counts=np.array(ck['sampling_counts']);elapsed0=ck['elapsed_seconds']
  for s in opt.state.values():
   if 'step' in s:s['step']=s['step'].cpu()
 else:assert not (run/'last.ckpt').exists()
 logpath=run/f'metrics_{stamp()}.csv';mf=logpath.open('x',newline='');mw=csv.DictWriter(mf,fieldnames=['step','elapsed_seconds','loss','descriptor','preserve','ranking','empty','equiv','repeatable_pairs','kp0','kp1','grad_norm','GPU_peak_MiB']);mw.writeheader();samplepath=run/f'sampling_{stamp()}.csv';sf=samplepath.open('x',newline='');sw=csv.DictWriter(sf,fieldnames=['step','micro','source_index','source_id','component','seed','regime','size','overlap']);sw.writeheader()
 def state(s,reason=''):atomic_json(output('status',folder=run),dict(state=s,reason=reason,global_step=step,process=identity,best=best,heartbeat=now(),elapsed_seconds=elapsed0+time.perf_counter()-start));print(now(),a.variant,s,step,reason,flush=True)
 def pack():return dict(variant=a.variant,model_state=m.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=sched.state_dict(),amp_scaler=scaler.state_dict(),global_step=step,random_state=rng_state(),config=c,best=best,selection_key=key,sampling_counts=counts.tolist(),elapsed_seconds=elapsed0+time.perf_counter()-start,data_identity=json.loads(output('data_identity').read_text()),code_identity=freeze)
 def validate():
  nonlocal key,best
  state('validation');m.eval();rr=ev.run({'m2':teacher,a.variant:m},plan,f'{a.variant}_step{step}',save=False,modes=('equal','natural'));current={mode:summary([r for r in rr if r['method']==a.variant and r['protocol']==mode]) for mode in ['equal','natural']};baseline={mode:summary([r for r in rr if r['method']=='m2' and r['protocol']==mode]) for mode in ['equal','natural']};san=probes(m);tsan=probes(teacher);hard=all(x['n_keypoints']==0 for x in san if x['probe']!='noise') and san[-1]['n_keypoints']<=tsan[-1]['n_keypoints'] and current['natural']['keypoints0']<=1.5*baseline['natural']['keypoints0'] and current['natural']['no_overlap_FPR']<=.05
  s=current['equal'];newkey=(int(hard),s['category_component_S5'],-s['wrong_accept'],s['correct_matches'],s['repeatability'] or 0);event(f'validation_step{step}',dict(step=step,summary=current,baseline=baseline,probes=san,hard_gate=hard,key=newkey),run)
  if key is None or newkey>key:key=newkey;best=dict(step=step,summary=current,probes=san,hard_gate_pass=hard);save_ckpt(run/'best.ckpt',pack())
  print(now(),a.variant,'VALIDATION',json.dumps(best),flush=True);m.train()
 try:
  state('training')
  if not a.resume:save_ckpt(run/'initial.ckpt',pack())
  while step<c['steps'] and not stop[0]:
   opt.zero_grad(set_to_none=True);infos=[]
   for micro in range(c['accumulation']):
    k=step*c['accumulation']+micro;index=schedule[k];r=sources[index];seed=c['seed']+k;rg=np.random.default_rng(seed);regime='easy' if step<500 else ('medium' if step<1500 else str(rg.choice(['easy','medium','low'],p=[.2,.5,.3])))
    if k>=5111 and min(r['native_width'],r['native_height'])>=768 and rg.random()<.05:regime='none'
    size=128 if min(r['native_width'],r['native_height'])<256 else (384 if k%4==0 and min(r['native_width'],r['native_height'])>=384 else 256)
    z=pair(r,seed,regime,size=size,dynamic_occlusion=bool(r['annotations']) and step>=1500 and k%7==0);loss,info=joint_loss(m,teacher,z,c);assert torch.isfinite(loss),'NaN/Inf loss';(loss/c['accumulation']).backward();infos.append(info);counts[index]+=1;sw.writerow(dict(step=step+1,micro=micro,source_index=index,source_id=r['source_id'],component=component(r),seed=seed,regime=regime,size=size,overlap=z['overlap']))
   for n,p in m.named_parameters():
    if p.requires_grad:
     if p.grad is None:assert n.startswith('sp.convD') and not any(x['repeatable_pairs'] for x in infos),(n,infos)
     else:assert torch.isfinite(p.grad).all(),n
    else:assert p.grad is None,n
   grad=float(torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.));opt.step();sched.step();step+=1
   if step==1 or step%20==0:
    row={k:float(np.mean([r[k] for r in infos])) for k in infos[0]};row.update(step=step,elapsed_seconds=elapsed0+time.perf_counter()-start,grad_norm=grad,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20);mw.writerow(row);mf.flush();sf.flush();state('training',f'loss={row["loss"]:.4f}')
   if step%c['checkpoint_every']==0:save_ckpt(run/'last.ckpt',pack())
   if step%c['validate_every']==0:
    validate();save_ckpt(run/'last.ckpt',pack());cp=run/'checkpoints';cp.mkdir(exist_ok=True);target=output(f'step{step}',ext='ckpt',folder=cp);assert not target.exists();save_ckpt(target,pack())
   if elapsed0+time.perf_counter()-start>c['wall_seconds']:raise RuntimeError('Pre-registered four-hour resource cap reached')
  save_ckpt(run/'last.ckpt',pack());assert all(torch.equal(x,m.state_dict()[n].cpu()) for n,x in frozen.items())
  if not stop[0]:
   assert step==c['steps'] and int(counts.sum())==10000 and (counts>0).all();write_new(output('selection_locked',folder=run),dict(best=best,final_step=step,steps_actual={n:int(opt.state[p]['step']) for n,p in m.named_parameters() if p in opt.state},best_sha=sha(run/'best.ckpt'),last_sha=sha(run/'last.ckpt'),unique_sources=int((counts>0).sum()),paired_calls=int(counts.sum()),view_instances=int(counts.sum()*2),frozen_parameters_intact=True,LightGlue_frozen=all(not p.requires_grad for p in ev.lg.parameters())));export_csv(output('source_coverage',ext='csv',folder=run),[dict(source_index=i,source_id=r['source_id'],component=component(r),dataset=r['dataset'],scene=r['scene_type'],calls=int(counts[i])) for i,r in enumerate(sources)])
  state('stopped' if stop[0] else 'completed')
 except BaseException as ex:state('failed',repr(ex));print(traceback.format_exc(),flush=True);raise
 finally:mf.close();sf.close()
if __name__=='__main__':main()
