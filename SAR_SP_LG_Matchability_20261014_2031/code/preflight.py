from common import *
from net import load_model,joint_loss
from evaluate import Evaluator,probes
from pairdata import pair
def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True);args=ap.parse_args();ident=gpu(args.gpu);c=json.loads(output('config').read_text());torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
 train=readrows(OLD/'train_sources.jsonl');r=next(r for r in train if r['scene_type']=='inshore' and min(r['native_width'],r['native_height'])>=500);z=pair(r,990001,'easy',size=256);teacher=load_model('m2');lg=Evaluator().lg;report=[]
 for variant in ['control','joint','frozen']:
  m=load_model(variant);before={n:v.detach().cpu().clone() for n,v in m.state_dict().items()};groups=[]
  for prefix,lr in [('sp.convP',c['detector_lr']),('sp.convD',c['descriptor_lr']),('sp.conv4',c['shared_lr'])]:
   ps=[p for n,p in m.named_parameters() if p.requires_grad and n.startswith(prefix)]
   if ps:groups.append(dict(params=ps,lr=lr))
  opt=torch.optim.AdamW(groups,foreach=False);losses=[]
  for i in range(30):
   opt.zero_grad();loss,stats=joint_loss(m,teacher,z,c);assert torch.isfinite(loss);loss.backward();grads={n:float(p.grad.norm()) if p.grad is not None else None for n,p in m.named_parameters()}
   assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad);assert all(p.grad is None for p in m.parameters() if not p.requires_grad);torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1);opt.step();losses.append(stats)
  frozen=[n for n,p in m.named_parameters() if not p.requires_grad];assert all(torch.equal(before[n],m.state_dict()[n].cpu()) for n in frozen)
  changed={n:float((v.detach().cpu()-before[n]).abs().max()) for n,v in m.state_dict().items() if n not in frozen};assert max(changed.values())>0
  checkpoint=ROOT/f'smoke_{variant}_{stamp()}.ckpt';assert not checkpoint.exists();save_ckpt(checkpoint,dict(model_state=m.state_dict(),optimizer_state=opt.state_dict(),random_state=rng_state(),global_step=30,variant=variant));reloaded=load_model(variant,checkpoint)
  im=torch.tensor(z['images'][0],device='cuda').float()[None,None]/255
  with torch.no_grad():a=m.extract(im);b=reloaded.extract(im);assert torch.equal(a['keypoints'],b['keypoints']) and torch.equal(a['descriptors'],b['descriptors']);out=lg(dict(image0=a,image1=b))
  # Full optimizer/RNG resume of one identical optimization step, not just inference reload.
  state=torch.load(checkpoint,map_location='cpu',weights_only=False);g2=[]
  for prefix,lr in [('sp.convP',c['detector_lr']),('sp.convD',c['descriptor_lr']),('sp.conv4',c['shared_lr'])]:
   ps=[p for n,p in reloaded.named_parameters() if p.requires_grad and n.startswith(prefix)]
   if ps:g2.append(dict(params=ps,lr=lr))
  o2=torch.optim.AdamW(g2,foreach=False);o2.load_state_dict(state['optimizer_state'])
  for model,optimizer in [(m,opt),(reloaded,o2)]:
   restore_rng(state['random_state']);optimizer.zero_grad();loss,_=joint_loss(model,teacher,z,c);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step()
  delta=max(float((v-reloaded.state_dict()[n]).abs().max()) for n,v in m.state_dict().items());assert delta<1e-7,delta
  density=losses[-1]['kp0']/losses[0]['kp0'];assert density<=1.5,(variant,density)
  report.append(dict(variant=variant,trainable=[n for n,p in m.named_parameters() if p.requires_grad],trainable_params=sum(p.numel() for p in m.parameters() if p.requires_grad),gradient_norms=grads,updated_maxima=changed,frozen_intact=True,LG_parameters_frozen=all(not p.requires_grad for p in lg.parameters()),initial=losses[0],final=losses[-1],smoke_density_ratio=density,reload_exact=True,resume_max_error=delta,n_LG_identity_matches=len(out['matches'][0]),checkpoint=str(checkpoint)))
  print('PREFLIGHT',variant,losses[0],losses[-1],flush=True)
 # Native sizes and analytical forward/backward coordinate masks.
 sizes=[]
 for size in [128,256,384,512]:
  a=pair(r,991000+size,'medium',size=size);assert np.allclose(a['H_AB'],a['H_B']@np.linalg.inv(a['H_A']));pts=np.array([[50.,50.],[70.,65.]]);err=np.linalg.norm(warp(np.linalg.inv(a['H_AB']),warp(a['H_AB'],pts))-pts,axis=1);assert err.max()<1e-8;sizes.append(size)
 assert pair(r,992000,'none')['overlap']==0
 event('preflight',dict(passed=True,process=ident,variants=report,coordinate_sizes=sizes,no_overlap=True,teacher_requires_grad=False,LightGlue_frozen=True,known_GPU_determinism_caveat='grid_sample CUDA backward uses atomic accumulation; tolerance 1e-7, not claimed byte-exact entire training'))
if __name__=='__main__':main()
