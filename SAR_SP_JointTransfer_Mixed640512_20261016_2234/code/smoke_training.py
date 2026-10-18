"""Actual 64-pair gradients, small-pool overfit, optimizer accumulation/resume tests."""
from joint_train import *

def step(m,opt,teacher,original,bank,ids,amp=True):
 opt.zero_grad(set_to_none=True);infos=[]
 for j in range(0,len(ids),2):
  x,rs,items=bank.batch(ids[j:j+2]);loss,inf=terms(m,teacher,original,x,rs,items,amp);(loss/(len(ids)/2)).backward();infos+=inf
 grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in m.named_parameters() if p.requires_grad};assert all(g is not None and np.isfinite(g) for g in grads.values());torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],5);opt.step()
 return float(np.mean([i['loss'] for i in infos])),infos,grads

def main():
 identity=setup();rows=records('train');plan=[json.loads(s) for s in sorted((ROOT/'plans').glob('sampling_seed42_*.jsonl'))[-1].read_text().splitlines()]
 ids=[i for q in plan[:8] for i in q['active']];bank=Bank(rows,ids);teacher=Model('b').cuda();original=Model('m0').cuda();result={};start=time.perf_counter()
 for kind in ['t1','t2','t3']:
  m=Model('b').cuda();scope(m,kind);initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};opt=optimizer(m);losses=[];grads=None;sizes=set()
  for s in plan[:8]:
   loss,info,grads=step(m,opt,teacher,original,bank,s['active']);losses.append(loss);sizes.add(s['size'])
  assert sizes=={512,640};trainable={n for n,p in m.named_parameters() if p.requires_grad};changed=[k for k,v in m.state_dict().items() if not torch.equal(v.detach().cpu(),initial[k])];assert changed and set(changed)<=trainable
  # Resume verifies frozen state, parameters, optimizer state and identical continuation input.
  save=ROOT/'smoke'/f'{kind}_resume_{TAG}.ckpt';state=dict(model_state=m.state_dict(),optimizer_state=opt.state_dict(),rng=rng(),next_pair_ids=[rows[i]['pair_id'] for i in plan[0]['active']]);atomic_checkpoint(save,state)
  ref=Model('b').cuda();scope(ref,kind);ck=torch.load(save,map_location='cpu',weights_only=False);ref.load_state_dict(ck['model_state']);ro=optimizer(ref);ro.load_state_dict(ck['optimizer_state']);restore_rng(ck['rng']);lr_before=[g['lr'] for g in ro.param_groups]
  a,_,_=step(m,opt,teacher,original,bank,plan[0]['active'],amp=False);restore_rng(ck['rng']);b,_,_=step(ref,ro,teacher,original,bank,plan[0]['active'],amp=False)
  delta=max(float((v-ref.state_dict()[k]).abs().max()) for k,v in m.state_dict().items());assert abs(a-b)<1e-5 and delta<1e-5,(a,b,delta)
  # A single fixed training mini-pool; never Val. Short, bounded overfit diagnostic.
  curve=[]
  for j in range(32):curve.append(step(m,opt,teacher,original,bank,plan[0]['active'])[0])
  assert np.mean(curve[-4:])<np.mean(curve[:4]),(kind,curve)
  result[kind]=dict(smoke_pairs=64,sizes=sorted(sizes),losses=losses,trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad),changed_parameters=changed,frozen_parameters_unchanged=True,last_gradient_norms=grads,resume_max_parameter_difference=delta,resume_loss_difference=abs(a-b),resume_learning_rates=lr_before,overfit_first4=float(np.mean(curve[:4])),overfit_last4=float(np.mean(curve[-4:])),overfit_curve=curve)
  print(now(),kind,json.dumps(clean(result[kind])),flush=True);del m,ref,opt,ro;torch.cuda.empty_cache()
 # Explicit gradient path check: descriptor matching can update detector logits through local coordinates.
 m=Model('b').cuda();scope(m,'t3');x,rs,items=bank.batch(plan[0]['active'][:1]);l,d=m.cached_maps(x,rs)
 with torch.no_grad():_,td=teacher.cached_maps(x,rs);sp=original.sp;tl=sp.convPb(sp.relu(sp.convPa(x)))
 loss,info,parts=objective(l,d,tl,td,items[0],int(x.shape[-1]*8),True);p=[v for n,v in m.named_parameters() if n.startswith('sp.convP')];gd=torch.autograd.grad(parts['descriptor'],p,retain_graph=True,allow_unused=True);gp=torch.autograd.grad(parts['detector'],p,retain_graph=True,allow_unused=True)
 av=torch.cat([g.flatten() for g in gd if g is not None]);bv=torch.cat([g.flatten() for g in gp if g is not None]);assert av.norm()>0 and bv.norm()>0
 gradient=dict(descriptor_to_detector_norm=float(av.norm()),detector_to_detector_norm=float(bv.norm()),cosine=float(F.cosine_similarity(av,bv,dim=0)),scope='one fixed Train pair; local differentiable response coordinates; not NMS gradient')
 # Valid singleton supervision has no fabricated contrastive classes; empty supervision is zero.
 single={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in items[0].items()};single['points']=single['points'][:,:1];single['patch_legal']=single['patch_legal'][:,:1]
 _,one=objective(l,d,tl,td,single,int(x.shape[-1]*8));assert one['contrastive_queries']==0 and one['positive_only']==1
 empty=dict(single,points=single['points'][:,:0],patch_legal=single['patch_legal'][:,:0]);zero,none=objective(l,d,tl,td,empty,int(x.shape[-1]*8));assert float(zero)==0 and not none['active']
 # Effective-batch accumulation vs full microbatch on the same eight images, FP32.
 ma=Model('b').cuda();mb=Model('b').cuda();scope(ma,'t3');scope(mb,'t3');oa=optimizer(ma);ob=optimizer(mb)
 step(ma,oa,teacher,original,bank,plan[0]['active'],False)
 xx,rr,ss=bank.batch(plan[0]['active']);ob.zero_grad();ll,_=terms(mb,teacher,original,xx,rr,ss,False);ll.backward();torch.nn.utils.clip_grad_norm_([p for p in mb.parameters() if p.requires_grad],5);ob.step();accum=max(float((v-mb.state_dict()[k]).abs().max()) for k,v in ma.state_dict().items());assert accum<2e-5,accum
 lg=LightGlue(features='superpoint').cuda().eval().requires_grad_(False);before={k:v.clone() for k,v in lg.state_dict().items()};row=rows[plan[0]['active'][0]];ims=images(row);fs=[m.extract(torch.tensor(im,device='cuda').float()[None,None]/255) for im in ims];pr,e,di,conf=predict(lg,fs,ims[0].shape);assert all(torch.equal(v,before[k]) for k,v in lg.state_dict().items())
 report=dict(passed=True,identity=identity,groups=result,gradient_scale=gradient,singleton=one,empty=none,accumulation_max_parameter_difference=accum,lightglue_frozen=True,lightglue_called=True,seconds=time.perf_counter()-start,peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20)
 write(ROOT/f'training_smoke_{stamp()}.json',report);print('ALL_CHECKS_PASSED',json.dumps(clean(gradient)),accum,flush=True)
if __name__=='__main__':main()
