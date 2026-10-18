from support22 import *
from graded_loss import loss22
from stream22 import Stream,loader
from evaluate22 import Evaluation,summarize,probes
import signal,traceback
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--kind',choices=['m2','b'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--manifest');p.add_argument('--fixed',action='store_true');p.add_argument('--attribution',action='store_true');p.add_argument('--ship-aux',action='store_true');p.add_argument('--initial');p.add_argument('--steps',type=int);p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true');p.add_argument('--stop-at',type=int);a=p.parse_args()
 identity=gpu(a.gpu);run=ROOT/'runs'/a.name;run.mkdir(parents=True,exist_ok=True);assert not out('selection_locked',folder=run).exists(),'Frozen completed run'
 cfg=json.loads(out('config_precheck' if a.smoke else 'config').read_text());cfg.update(kind=a.kind,manifest=a.manifest or E['recommended_train_reference'],online=not a.fixed,attribution=a.attribution,ship_aux=a.ship_aux,initial=a.initial,steps=a.steps or cfg['steps'])
 if a.smoke:cfg.update(steps=a.steps or 400,validate_every=200,checkpoint_every=100,workers=2)
 torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
 m=Model(a.kind,a.initial,train=True).cuda();teacher=Model('m2').cuda().eval();ev=Evaluation()
 groups=[]
 for prefix,lr in [('sp.convD',cfg['lr']),('sp.convP',cfg['detector_lr']),('structure.',cfg['structure_lr'])]:
  ps=[v for n,v in m.named_parameters() if v.requires_grad and n.startswith(prefix)]
  if ps:groups.append(dict(params=ps,lr=lr,name=prefix))
 opt=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay'],foreach=False);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['steps']);scaler=torch.amp.GradScaler('cuda',enabled=False)
 frozen={n:v.detach().cpu().clone() for n,v in m.named_parameters() if not v.requires_grad};initial={n:v.detach().cpu().clone() for n,v in m.named_parameters() if v.requires_grad};step=0;call=0;best=None;key=None;idle=0;elapsed0=0;start=time.perf_counter();stop=[False]
 signal.signal(signal.SIGTERM,lambda *x:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *x:stop.__setitem__(0,True))
 baseline_probes=probes(m);reference_probes=probes(teacher);code_identity={str(f):sha(f) for f in (ROOT/'code').glob('*.py')}
 if not a.resume:write(out('run_identity',folder=run),dict(created=now(),config=cfg,process=identity,trainable={n:v.numel() for n,v in m.named_parameters() if v.requires_grad},init_sha=sha(a.initial or (M2 if a.kind=='m2' else B)),code_sha=code_identity,entry_sha=sha(ENTRY),manifest_sha=sha(cfg['manifest']),LightGlue_trainable=0))
 def state(s,reason=''):
  write(out('status',folder=run),dict(at=now(),state=s,step=step,consumed_calls=call,best=best,pid=identity['pid'],process=identity,reason=reason,elapsed_seconds=elapsed0+time.perf_counter()-start),replace=True);print(now(),a.name,s,step,call,reason,flush=True)
 def pack():return dict(model_state=m.state_dict(),optimizer_state=opt.state_dict(),scheduler_state=sch.state_dict(),amp_scaler=scaler.state_dict(),global_step=step,next_call=call,random_state=rng(),config=cfg,best=best,selection_key=key,idle=idle,elapsed_seconds=elapsed0+time.perf_counter()-start,baseline_probes=baseline_probes,data_identity=dict(entry_sha=sha(ENTRY),manifest_sha=sha(cfg['manifest'])),code_identity=code_identity)
 if a.resume:
  ck=torch.load(run/'last.ckpt',map_location='cuda',weights_only=False);assert ck['config']==cfg;m.load_state_dict(ck['model_state']);opt.load_state_dict(ck['optimizer_state']);sch.load_state_dict(ck['scheduler_state']);scaler.load_state_dict(ck['amp_scaler']);restore(ck['random_state']);step=ck['global_step'];call=ck['next_call'];best=ck['best'];key=tuple(ck['selection_key']) if ck['selection_key'] else None;idle=ck['idle'];elapsed0=ck['elapsed_seconds'];baseline_probes=ck['baseline_probes']
  for s in opt.state.values():
   if 'step' in s:s['step']=s['step'].cpu()
 else:assert not (run/'last.ckpt').exists()
 protocol=json.loads(out('protocol').read_text());selected=set(protocol['selection_ids']);val=[r for r in rows(E['recommended_val_all']) if r['relation_id'] in selected]
 def validate():
  nonlocal best,key,idle
  state('validation');m.eval();rr=ev.run({a.name:m,'referenceM2':teacher},val,'v22',a.name+f'_step{step}',equal=True);summary=summarize([r for r in rr if r['method']==a.name and r['protocol']=='equal']);natural=summarize([r for r in rr if r['method']==a.name and r['protocol']=='natural']);reference=summarize([r for r in rr if r['method']=='referenceM2' and r['protocol']=='natural']);pp=probes(m)
  hard=all(q['n_keypoints']==0 for q in pp[:2]) and pp[2]['n_keypoints']<=reference_probes[2]['n_keypoints'] and (natural['calibration_keypoints0'] or 0)<=1.2*(reference['calibration_keypoints0'] or 0)+1 and (natural['no_overlap_FPR'] or 0)<=(reference['no_overlap_FPR'] or 0)
  new=(int(hard),summary['S5'] or 0,-(natural['no_overlap_FPR'] or 0),summary['local_R1'] or 0);improve=key is None or new>key
  write(out(f'validation_step{step}',folder=run),dict(step=step,summary=summary,natural=natural,reference=reference,probes=pp,hard_gate=hard,selection_key=new,rows=rr))
  if improve:key=new;best=dict(step=step,summary=summary,natural=natural,probes=pp,hard_gate=hard);idle=0;save(run/'best.ckpt',pack())
  else:idle+=1
  print(now(),a.name,'VAL',json.dumps(dict(current=summary,best_step=best['step'],probes=pp,hard=hard)),flush=True);m.train()
 version=stamp();mf=(run/f'metrics_from{step}_{version}.csv').open('x',newline='');fields=['step','calls','loss','ground_loss','preserve','detector','calibration','negative_loss','ship_loss','ground_points_used','ship_points_used','active_pairs','attempted_pairs','grad_norm','elapsed_seconds','GPU_peak_MiB'];mw=csv.DictWriter(mf,fieldnames=fields);mw.writeheader();sf=(run/f'sampling_from{step}_{version}.jsonl').open('x');stream=None
 try:
  if step==0:validate();save(run/'last.ckpt',pack())
  state('training');mode='attribution' if a.attribution else ('M2' if a.kind=='m2' else 'B')
  stream=iter(loader(Stream(cfg['manifest'],mode,cfg['seed'],run,start=call,online=cfg['online'],ship_aux=a.ship_aux,limit=(cfg['steps']-step)*32+128),workers=cfg['workers']))
  while step<cfg['steps'] and not stop[0]:
   opt.zero_grad(set_to_none=True);infos=[];attempts=0
   while len(infos)<cfg['effective_batch']:
    item=next(stream);call+=1;attempts+=1;trace=dict(item['trace'],step=step+1,consumed_at=now(),error=item['error'])
    assert trace['call_index']==call-1
    if item['pair'] is not None:
     z=item['pair'];loss,info=loss22(m,teacher,z,ship_aux=a.ship_aux);assert torch.isfinite(loss),'NaN/Inf loss';trace.update(actual_role=z['loss_role'],size=z['images'][0].shape[0],ground_available=len(z['points0']),ship_available=len(z['ship_points0']),**info)
     if info['active']:(loss/cfg['effective_batch']).backward();infos.append(info)
    sf.write(json.dumps(clean(trace),ensure_ascii=False)+'\n')
    if attempts>128:raise RuntimeError('128 calls without filling effective8; stop rather than silently resample')
   # Do not let zero-valued placeholder paths trigger AdamW decay/momentum on
   # descriptor/structure parameters when the entire effective batch has no labels.
   if all(r['skip_descriptor_update'] for r in infos):
    for n,v in m.named_parameters():
     if n.startswith(('sp.convD','structure.')):v.grad=None
   gradnames=[]
   for n,v in m.named_parameters():
    if v.requires_grad:
     if v.grad is not None:assert torch.isfinite(v.grad).all(),n;gradnames.append(n)
    else:assert v.grad is None,n
   assert gradnames,'No trainable gradients';grad=float(torch.nn.utils.clip_grad_norm_([v for v in m.parameters() if v.requires_grad],1.));assert np.isfinite(grad) and grad>0,'No effective gradient update'
   opt.step();sch.step();step+=1
   metric={k:float(np.mean([r[k] for r in infos])) for k in ['loss','ground_loss','preserve','detector','calibration','negative_loss','ship_loss','ground_points_used','ship_points_used']};metric.update(step=step,calls=call,active_pairs=len(infos),attempted_pairs=attempts,grad_norm=grad,elapsed_seconds=elapsed0+time.perf_counter()-start,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20);mw.writerow(metric)
   if step%20==0 or step==1:mf.flush();sf.flush();state('training',f'loss={metric["loss"]:.5f}; active={len(infos)}/{attempts}')
   if step%cfg['checkpoint_every']==0:save(run/'last.ckpt',pack())
   if step%cfg['validate_every']==0 or step==cfg['steps']:
    validate();save(run/'last.ckpt',pack());save(run/'checkpoints'/f'step{step}.ckpt',pack())
   if a.stop_at and step>=a.stop_at:stop[0]=True
   if not a.smoke and step>=cfg['min_steps'] and idle>=cfg['early_stop_validations']:break
   if elapsed0+time.perf_counter()-start>cfg['wall_seconds']:raise RuntimeError('Fixed wall budget exceeded')
  save(run/'last.ckpt',pack());unchanged=all(torch.equal(v,m.state_dict()[n].cpu()) for n,v in frozen.items());assert unchanged
  updated={n:float((m.state_dict()[n].cpu()-v).abs().max()) for n,v in initial.items()};assert any(v>0 for v in updated.values())
  if not stop[0]:write(out('selection_locked',folder=run),dict(completed=now(),best=best,actual_steps=step,consumed_calls=call,best_sha=sha(run/'best.ckpt'),last_sha=sha(run/'last.ckpt'),frozen_intact=unchanged,parameter_max_changes=updated,optimizer_steps={n:int(opt.state[v]['step']) for n,v in m.named_parameters() if v in opt.state},lightglue_frozen=all(not v.requires_grad for v in ev.lg.parameters()),peak_MiB=torch.cuda.max_memory_allocated()/2**20,elapsed_seconds=elapsed0+time.perf_counter()-start))
  state('stopped' if stop[0] else 'completed')
 except BaseException as ex:
  state('failed',repr(ex));print(traceback.format_exc(),flush=True);raise
 finally:
  mf.close();sf.close()
  if stream is not None:del stream
if __name__=='__main__':main()
