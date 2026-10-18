"""Real CUDA gradients/overfit/reload-resume and geometric data checks before jobs."""
import sys,os,json,time,random
from pathlib import Path
import cv2,numpy as np,torch
from common import ROOT,BASE,ensure_gpu,now,stamp,atomic_json,readrows,sha
from model import SARModel,loss
from data import pair,warp,valid_at
from evaluate import model_validation,image_matches
from checkpoint import pack,atomic_save,restore
def run(variant):
 physical=0 if variant=='m1' else 1;gpu=ensure_gpu(physical);torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42);random.seed(42)
 cfg=json.loads((ROOT/'configs'/f'{variant}.json').read_text());folder=ROOT/'preflight'/variant;folder.mkdir(parents=True,exist_ok=False);start=time.perf_counter()
 sources=readrows(ROOT/'train_sources.jsonl');r=next(r for r in sources if r['scene_type']=='inshore' and r['dataset']=='hrsid');z=pair(r,314159,'medium');assert len(z['points0'])>=32
 np.testing.assert_allclose(warp(z['H_B']@np.linalg.inv(z['H_A']),z['points0']),z['points1'],atol=2e-5)
 for i in [0,1]:assert valid_at(z['static'][i],z['points'+str(i)]).all()
 h=z['H_AB'];np.testing.assert_allclose(warp(np.linalg.inv(h),warp(h,z['points0'])),z['points0'],atol=1e-5)
 vv=np.ones((20,20),bool);vv[4,4]=False;assert not valid_at(vv,np.array([[3.5,3.5]]))[0]
 m=SARModel(variant).cuda();baseline=SARModel('m0').cuda();trainable=[(n,p) for n,p in m.named_parameters() if p.requires_grad];frozen={n:p.detach().clone() for n,p in m.named_parameters() if not p.requires_grad};initial={n:p.detach().clone() for n,p in trainable};opt=torch.optim.AdamW([p for _,p in trainable],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg['max_steps']);scaler=torch.amp.GradScaler('cuda')
 image=torch.from_numpy(z['images'][0]).float().cuda()[None,None]/255
 with torch.no_grad():
  f=m.extract(image);fbase=baseline.extract(image);torch.testing.assert_close(f['keypoints'],fbase['keypoints'],rtol=0,atol=0);torch.testing.assert_close(f['descriptors'],fbase['descriptors'],rtol=0,atol=1e-6)
  sampled=m.descriptors(image,f['keypoints'])[0];torch.testing.assert_close(sampled,f['descriptors'],rtol=0,atol=1e-6)
 initial_loss,stats0=loss(m,z,'cuda',amp=False);before=dict(loss=float(initial_loss.detach()),**stats0);history=[]
 for step in range(160):
  m.train();opt.zero_grad(set_to_none=True);value,stats=loss(m,z,'cuda');scaler.scale(value).backward();scaler.unscale_(opt)
  norm=torch.nn.utils.clip_grad_norm_([p for _,p in trainable],1.);assert torch.isfinite(norm)
  if step==0:assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for _,p in trainable)
  scaler.step(opt);scaler.update();sched.step();history.append(dict(step=step+1,loss=float(value.detach()),**stats))
  if (step+1)%40==0:print(variant,'OVERFIT',step+1,history[-1],flush=True)
 m.eval();final_loss,stats1=loss(m,z,'cuda',amp=False);after=dict(loss=float(final_loss.detach()),**stats1);assert after['loss']<before['loss']-.001,'No real overfit improvement'
 delta=sum(float((p-initial[n]).square().sum()) for n,p in trainable);assert delta>0
 assert all(torch.equal(p,frozen[n]) for n,p in m.named_parameters() if n in frozen),'Frozen parameters changed'
 with torch.no_grad():
  changed=m.extract(image);torch.testing.assert_close(changed['keypoints'],fbase['keypoints'],rtol=0,atol=0);assert torch.isfinite(changed['descriptors']).all();torch.testing.assert_close(changed['descriptors'].norm(dim=-1),torch.ones_like(changed['keypoint_scores']),atol=1e-5,rtol=0)
 ckpt=folder/'trained.ckpt';atomic_save(ckpt,pack(m,opt,sched,scaler,160,cfg,{},None,None));expected={k:v.clone() for k,v in m.state_dict().items()};saved_rng=torch.get_rng_state().clone();reference_loss=loss(m,z,'cuda',amp=False)[0];opt.zero_grad(set_to_none=True);reference_loss.backward();opt.step();reference={n:p.clone() for n,p in m.named_parameters() if p.requires_grad}
 c=restore(ckpt,m,opt,sched,scaler,cfg);assert c['global_step']==160;assert all(torch.equal(v,expected[k]) for k,v in m.state_dict().items());assert torch.equal(torch.get_rng_state(),saved_rng)
 opt.zero_grad(set_to_none=True);loss(m,z,'cuda',amp=False)[0].backward();opt.step();assert all(torch.equal(p,reference[n]) for n,p in m.named_parameters() if p.requires_grad),'Optimizer resume not exact'
 plan=json.loads((ROOT/'validation_plan.json').read_text());small=[plan[i] for i in [0,1,12,24,36,48]]
 m0_result=model_validation(baseline,small,output=folder/'M0_validation_smoke.json');trained=model_validation(m,small,output=folder/'trained_validation_smoke.json')
 b0,b1,o,e,d=image_matches(m,__import__('lightglue').LightGlue(features='superpoint').eval().cuda(),*z['images']);assert o['matches'][0].ndim==2
 # Cached independent test fixture: native area rather than blank out-of-frame no-overlap.
 negative=pair(plan[48]['source'],plan[48]['seed'],'none');assert negative['overlap']==0 and all(v.mean()>.9 for v in negative['valid'])
 cv2.imwrite(str(folder/'overfit_view0.png'),z['images'][0]);cv2.imwrite(str(folder/'overfit_view1.png'),z['images'][1]);atomic_json(folder/'overfit_curve.json',history)
 result=dict(passed=True,variant=variant,gpu=gpu,trainable_names=[n for n,_ in trainable],trainable_parameters=sum(p.numel() for _,p in trainable),real_optimizer_steps=160,before=before,after=after,parameter_squared_delta=delta,frozen_weights_unchanged=True,detector_keypoints_exactly_unchanged=True,resume_model_optimizer_rng_exact=True,LightGlue_loaded_new_descriptors=True,model0_validation=m0_result['summary'],trained_validation=trained['summary'],synthetic_not_real_GT=True,elapsed_s=time.perf_counter()-start,peak_GPU_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),finished_at=now())
 atomic_json(folder/'result.json',result);print('PREFLIGHT PASSED',json.dumps(result),flush=True)
if __name__=='__main__':run(sys.argv[1])
