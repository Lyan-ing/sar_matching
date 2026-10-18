"""Focused actual CUDA tests, train-only overfit, exact checkpoint/RNG replay."""
from m3_common import *
from m3_model import *
from m3_evaluate import Evaluator
from train_m3 import optimizer,pack,restore
import argparse,time
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();process=gpu(a.gpu);start=time.perf_counter();c=json.loads((ROOT/'config.json').read_text());m=load_m3();ev=Evaluator()
 train=rows(OLD/'train_sources.jsonl');assert not {component(r) for r in train}&{component(s['source']) for s in json.loads((ROOT/'validation_plan.json').read_text())}
 eligible=sorted([r for r in train if r['scene_type']=='inshore' and r['native_width']>=384],key=lambda r:r['source_id'])
 for i,r in enumerate(eligible[:50]):
  z=pair(r,93100+i,'easy');anchors=structure_targets(z,m)
  if len(anchors[0])>=16:break
 assert len(anchors[0])>=16,'No structured training pair found for overfit'
 # Pixel shuffle = official 8x8 channel-to-pixel order; exact H_A/H_B relation.
 t=torch.randn(1,65,5,7,device='cuda');soft=t.softmax(1)[:,:64];official=soft.permute(0,2,3,1).reshape(1,5,7,8,8).permute(0,1,3,2,4).reshape(1,40,56);assert torch.equal(score_map(t),official)
 q=np.array([[20.,30.],[80.,90.],[190.,160.]]);assert np.allclose(warp(z['H_AB'],warp(z['H_A'],q)),warp(z['H_B'],q))
 for i,pts in enumerate(anchors):assert valid_at(z['static'][i],pts).all() and not valid_at(z['ship_masks'][i],pts).any()
 frozen={k:v.detach().cpu().clone() for k,v in m.state_dict().items() if not k.startswith(('sp.convPa.','sp.convPb.'))};initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items() if k.startswith(('sp.convPa.','sp.convPb.'))}
 im=torch.tensor(z['images'][0].copy(),device='cuda',dtype=torch.float32)[None,None]/255;f0=m.extract(im);fm=ev.models['m2'].extract(im);assert torch.equal(f0['keypoints'],fm['keypoints']) and torch.equal(f0['descriptors'],fm['descriptors'])
 dd,_=m.descriptors(im,f0['keypoints']);assert torch.allclose(dd,f0['descriptors'],atol=2e-6)
 # Check exact tiling against full official extraction at odd image dimensions, including tile seams.
 test=torch.tensor(np.tile(z['images'][0],(3,3))[:517,:531].copy()).float()[None,None]/255
 full=m.extract(test.cuda());tile=tiled_extract(m.sp,test,tile=256,halo=64)
 def indexed(f):
  kp=f['keypoints'][0].cpu().numpy();order=np.lexsort((kp[:,0],kp[:,1]));return kp[order],f['descriptors'][0].cpu().numpy()[order]
 ka,da=indexed(full);kb,db=indexed(tile);assert np.array_equal(ka,kb),f'Tile keypoints differ {len(ka)}/{len(kb)}';assert np.allclose(da,db,atol=2e-5),'Tile descriptors differ'
 o=optimizer(m,c);s=torch.optim.lr_scheduler.LambdaLR(o,lambda _:1.);scaler=torch.amp.GradScaler('cuda',enabled=False);curve=[];grads={}
 def update():
  o.zero_grad(set_to_none=True);loss,info=detector_loss(m,z,c,anchors);loss.backward()
  for n,p in m.named_parameters():
   if p.requires_grad:assert p.grad is not None and torch.isfinite(p.grad).all();grads[n]=float(p.grad.norm())
   else:assert p.grad is None
  torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1);o.step();s.step();return info
 for step in range(100):
  info=update();curve.append(dict(step=step+1,**info))
 assert np.mean([x['loss'] for x in curve[-10:]])<np.mean([x['loss'] for x in curve[:10]]), 'Train-only overfit does not improve loss'
 assert all(torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in frozen.items())
 assert all(not torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in initial.items())
 out=ROOT/'preflight';out.mkdir(exist_ok=True);save_ckpt(out/'resume.ckpt',pack(m,o,s,scaler,100,c,None,None,0,0,3000));update();expected={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};random_expected=torch.rand(5,device='cuda')
 saved=torch.load(out/'resume.ckpt',map_location='cpu',weights_only=False)
 restore(out/'resume.ckpt',m,o,s,scaler,c)
 assert all(torch.equal(v,m.state_dict()[k].detach().cpu()) for k,v in saved['model_state'].items()),'Checkpoint restoration changed tensors'
 update()
 differences={k:float((v-m.state_dict()[k].detach().cpu()).abs().max()) for k,v in expected.items() if not torch.equal(v,m.state_dict()[k].detach().cpu())}
 print('resume differences',differences,flush=True)
 # CUDA grid_sample backward uses atomic accumulation (documented in installed torch).
 # State restoration must be bit-exact; a subsequent floating-point backward allows <=1e-8 absolute.
 assert 'nondeterministic' in F.grid_sample.__doc__
 replay_delta=max(differences.values(),default=0.);assert replay_delta<=1e-8
 assert torch.equal(torch.rand(5,device='cuda'),random_expected)
 pp,ee,diag,conf,ms,ff=ev.predict(m,*z['images']);dd,_=m.descriptors(im,ff[0]['keypoints']);assert torch.allclose(dd,ff[0]['descriptors'],atol=2e-6)
 atomic_json(out/'overfit_curve.json',curve);atomic_json(out/'result.json',dict(passed=True,process=process,source=r['source_id'],source_component=component(r),train_only=True,anchors=len(anchors[0]),gradients=grads,trainable_params=sum(p.numel() for p in m.parameters() if p.requires_grad),frozen_tensors=len(frozen),frozen_unchanged=True,detector_initial_identical_to_M2=True,descriptor_sampling_passed=True,pixel_alignment_passed=True,static_dynamic_exclusion_passed=True,tiled_odd_dimensions_equivalent=True,tile_descriptor_max_abs=float(np.max(np.abs(da-db))),exact_state_restore=True,exact_rng_restore=True,resume_one_step_max_abs=replay_delta,resume_one_step_tolerance=1e-8,resume_caveat='CUDA grid_sample backward atomic summation is not bitwise deterministic; exact checkpoint restore, <=1e-8 single-update delta',LightGlue_matches=len(pp.points0),overfit_initial_loss=curve[0]['loss'],overfit_final_loss=curve[-1]['loss'],elapsed_seconds=time.perf_counter()-start))
 print(json.dumps(json.loads((out/'result.json').read_text()),indent=2))
if __name__=='__main__':main()
