from support22 import *
from graded_loss import loss22
from evaluate22 import Evaluation,predict,features
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();identity=gpu(a.gpu);torch.manual_seed(202622)
 rr=rows(E['recommended_train_reference']);teacher=Model('m2').cuda();results=[]
 for kind in ['m2','b']:
  m=Model(kind,train=True).cuda();frozen={n:v.detach().clone() for n,v in m.named_parameters() if not v.requires_grad};base=next(r for r in rr if r['structural_anchor_count']>=128)
  for N in [0,1,3,4,8,16,128]:
   z=load_pair(base);z['points0']=z['points0'][:N];z['points1']=z['points1'][:N];z['loss_role']='limited_structure' if N else 'no_structural_positive';z['local_weight']=min(1,N/16);z['local_descriptor_eligible']=N>=4
   m.zero_grad(set_to_none=True);loss,info=loss22(m,teacher,z);loss.backward();assert info['ground_points_used']==(N if N>=4 else 0)
   gradients={n:float(v.grad.abs().max()) if v.grad is not None else None for n,v in m.named_parameters() if v.requires_grad};assert all(v.grad is None for n,v in m.named_parameters() if not v.requires_grad);results.append(dict(kind=kind,N=N,info=info,gradients=gradients))
  # Actual variable-size materialization, no padding/implicit resize.
  for size in sorted({r['width'] for r in rr if r['structural_anchor_count']>=4}):
   r=next(r for r in rr if r['width']==size and r['structural_anchor_count']>=4);z=load_pair(r);loss,info=loss22(m,teacher,z);assert torch.isfinite(loss);results.append(dict(kind=kind,size=size,info=info))
  # All-unknown must not turn into dustbin calibration, all-blank is an explicit artificial probe only.
  z=load_pair(base);z['points0']=np.empty((0,2),np.float32);z['points1']=z['points0'].copy();z['static']=[np.zeros_like(v) for v in z['valid']];z['calibration_mask']=[np.zeros_like(v) for v in z['valid']];z['supervision_ignore']=[v.copy() for v in z['valid']];z['local_weight']=0
  loss,info=loss22(m,teacher,z);assert info['ground_points_used']==0 and info['calibration_cells']==0
  results.append(dict(kind=kind,unknown_only=info))
  # Two equal micro losses /2 have the same gradient as one full loss.
  z=load_pair(base);m.zero_grad(set_to_none=True);loss22(m,teacher,z)[0].backward();g={n:v.grad.clone() for n,v in m.named_parameters() if v.requires_grad and v.grad is not None};m.zero_grad(set_to_none=True)
  for _ in range(2):(loss22(m,teacher,z)[0]/2).backward()
  delta=max(float((v.grad-g[n]).abs().max()) for n,v in m.named_parameters() if n in g);assert delta<1e-6
  opt=torch.optim.AdamW([v for v in m.parameters() if v.requires_grad],lr=1e-5,foreach=False);opt.step();assert all(torch.equal(v,m.state_dict()[n]) for n,v in frozen.items())
  path=ROOT/'preflight'/f'{kind}_resume.ckpt';save(path,dict(model_state=m.state_dict(),optimizer_state=opt.state_dict(),random_state=rng()));ck=torch.load(path,map_location='cuda',weights_only=False);reload=Model(kind,path,train=True).cuda();assert all(torch.equal(v,reload.state_dict()[n]) for n,v in m.state_dict().items())
  ev=Evaluation();fs=[features(reload,im) for im in z['images']];pred,est,diag,conf=predict(ev.lg,fs,[im.shape for im in z['images']]);results.append(dict(kind=kind,accumulation_max_error=delta,checkpoint_exact=True,LG_matches=len(conf),frozen_intact=True))
  del m,reload,ev;torch.cuda.empty_cache()
 # Frozen delivered B bridge evidence on a real 1--3 point record.
 from loss22 import b_loss as delivered_b_loss
 b=Model('b',train=True).cuda()
 for r in [r for r in rr if 0<r['structural_anchor_count']<4][:12]:
  z=load_pair(r);loss,info=delivered_b_loss(b,teacher,z,device='cuda');results.append(dict(delivered_bridge=True,sample_id=r['sample_id'],saved_anchors=len(z['points0']),info=info))
 write(out('preflight_results'),dict(at=now(),process=identity,passed=True,checks=results,peak_MiB=torch.cuda.max_memory_allocated()/2**20))
 print('PREFLIGHT_PASS',len(results),flush=True)
if __name__=='__main__':main()
