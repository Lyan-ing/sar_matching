"""Training-only local response/descriptor coupling on immutable reliable anchors.

q_i=sum_{delta in [-2,2]^2} softmax(log(P_i(p_i+delta))/0.5) * (p_i+delta)
L = w[CE_bidir(D(qA),D(qB)) + positive_only_if_no_legal_neg + .1 distill_D
      + .05 anchor_detector_CE + .1 symmetric_H_coordinate_error/9]
    + .25 KL(original_detector || detector) on evidenced static cells
    + 1.0 dustbin_CE on explicitly calibrated legal cells only.
Descriptor matching gradients reach P via q and differentiable grid sampling,
not via NMS/top-k. Structure remains the historical descriptor-only branch.
No negative labels from unknown pixels, no positive labels from ship proxies.
"""
from runtime import *
OFF=torch.tensor(np.stack(np.meshgrid(np.arange(-2,3),np.arange(-2,3)),axis=-1).reshape(-1,2),dtype=torch.float32)

def scope(m,kind):
 m.requires_grad_(False)
 prefixes={'t1':('structure.','sp.convP'),'t2':('sp.convD',),'t3':('structure.','sp.convP','sp.convD')}[kind]
 for n,p in m.named_parameters():p.requires_grad_(n.startswith(prefixes))
 return [p for p in m.parameters() if p.requires_grad]

def transform(H,p):
 t=torch.cat([p,torch.ones_like(p[:,:1])],1)@H.T;return t[:,:2]/t[:,2:]
def sample_score(score,xy,size):
 grid=xy.reshape(1,1,-1,2)/(size-1)*2-1
 return F.grid_sample(score[None,None],grid,align_corners=True)[0,0,0].reshape(xy.shape[:-1])
def local_coordinates(logits,item,size):
 scores=score_map(logits.float());p=torch.as_tensor(item['points'],device=logits.device,dtype=torch.float32);legal=torch.as_tensor(item['patch_legal'],device=logits.device,dtype=torch.bool);result=[]
 for i in [0,1]:
  xy=p[i,:,None]+OFF.to(logits.device)[None];assert legal[i].any(1).all()
  value=sample_score(scores[i],xy,size).clamp_min(1e-8).log()/.5;weight=value.masked_fill(~legal[i],-1e4).softmax(-1)
  result.append((xy*weight[...,None]).sum(1))
 return result

def objective(logits,dense,teacher_logits,teacher_dense,item,size,return_parts=False):
 device=dense.device;dense=F.normalize(dense.float(),dim=1);teacher_dense=F.normalize(teacher_dense.float(),dim=1)
 p=torch.as_tensor(item['points'],device=device,dtype=torch.float32);n=p.shape[1];zero=dense.sum()*0+logits.sum()*0
 if not n:return zero,dict(active=False,points=0,contrastive_queries=0,positive_only=0)
 q=local_coordinates(logits,item,size)
 ds=[sample_descriptors(x[None],dense[i:i+1],8)[0].T for i,x in enumerate(q)]
 with torch.no_grad():ts=[sample_descriptors(x.detach()[None],teacher_dense[i:i+1],8)[0].T for i,x in enumerate(q)]
 near=(torch.cdist(p[0],p[0])<8)|(torch.cdist(p[1],p[1])<8);near.fill_diagonal_(False)
 allowed=~near;valid_negative=allowed.clone();valid_negative.fill_diagonal_(False);hasneg=valid_negative.any(1)
 sim=(ds[0]@ds[1].T/.1).masked_fill(~allowed,-1e4);labels=torch.arange(n,device=device)
 ce=(F.cross_entropy(sim,labels,reduction='none')[hasneg].mean()+F.cross_entropy(sim.T,labels,reduction='none')[hasneg].mean())/2 if hasneg.any() else zero
 positive=1-(ds[0]*ds[1]).sum(1);align=positive[~hasneg].mean() if (~hasneg).any() else zero
 preserve=sum((d-t).square().sum(1).mean() for d,t in zip(ds,ts))/2
 lp=logits.float().log_softmax(1);det=[]
 for i in [0,1]:
  pixel=p[i].round().long();label=(pixel[:,1]%8)*8+pixel[:,0]%8
  det.append(-lp[i,label,pixel[:,1]//8,pixel[:,0]//8].mean())
 det=sum(det)/2;keep=torch.as_tensor(item['keep'],device=device,dtype=torch.bool);cal=torch.as_tensor(item['calibration'],device=device,dtype=torch.bool)
 # An anchor cell cannot simultaneously become a dustbin training target.
 for i in [0,1]:
  pixel=p[i].round().long();cal[i,pixel[:,1]//8,pixel[:,0]//8]=False
 tl=teacher_logits.float();kl=(tl.softmax(1)*(tl.log_softmax(1)-lp)).sum(1)[keep].mean() if keep.any() else zero
 calibration=-lp[:,64][cal].mean() if cal.any() else zero
 H=torch.as_tensor(item['H_AB'],device=device,dtype=torch.float32);project=transform(H,q[0]);back=transform(torch.linalg.inv(H),q[1]);equiv=((project-q[1]).square().sum(1).mean()+(back-q[0]).square().sum(1).mean())/18
 w=float(item['local_weight']);match=ce+align+.1*preserve;detector=.05*det+.1*equiv
 total=w*(match+detector)+.25*kl+calibration
 info=dict(active=True,loss=float(total.detach()),descriptor=float(match.detach()),contrastive=float(ce.detach()),positive_only_loss=float(align.detach()),preserve=float(preserve.detach()),detector_anchor=float(det.detach()),coordinate_equivariance=float(equiv.detach()),teacher_KL=float(kl.detach()),calibration=float(calibration.detach()),points=n,raw_points=int(item['raw_count']),contrastive_queries=int(hasneg.sum()),positive_only=int((~hasneg).sum()),local_R1=float((sim.argmax(1)==labels).float().mean()),mean_offset=float(torch.stack([torch.linalg.vector_norm(x-p[i],dim=1).mean() for i,x in enumerate(q)]).mean().detach()),calibration_cells=int(cal.sum()),prior_cells=int(keep.sum()))
 return (total,info,dict(descriptor=w*match,detector=w*detector+.25*kl+calibration)) if return_parts else (total,info)

def load_item(row):
 path=ROOT/'cache/train_supervision'/f"{row['pair_id']}_{TAG}.npz"
 with np.load(path) as f:return {k:f[k].copy() for k in f.files}

def teacher_maps(teacher,x):
 with torch.no_grad():return teacher.cached_maps(x,teacher._responses)
