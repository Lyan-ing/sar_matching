"""V2.2 role bridge, exact saved anchors; never synthesize labels from unknown pixels.

This bounded correction replaces B's resampled-point CE with the SAME saved-anchor
CE as M2. It prevents the delivered bridge computing CE from <4 reliable anchors.
Existing B detector terms are preserved, with explicit evidence-only masks.
"""
from support22 import *
def correspondence(dense,teacher,points0,points1):
 n=len(points0);zero=dense.sum()*0
 if n<4:return zero,zero,0,None
 xy=[torch.as_tensor(p,device=dense.device,dtype=torch.float32)[None] for p in [points0,points1]]
 ds=[sample_descriptors(p,dense[i:i+1],8)[0].T.float() for i,p in enumerate(xy)]
 ts=[sample_descriptors(p,teacher[i:i+1],8)[0].T.float() for i,p in enumerate(xy)]
 sim=ds[0]@ds[1].T/.1;near=(torch.cdist(xy[0][0],xy[0][0])<8)|(torch.cdist(xy[1][0],xy[1][0])<8);near.fill_diagonal_(False);sim=sim.masked_fill(near,-1e4);target=torch.arange(n,device=dense.device)
 ce=(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2
 preserve=sum((d-t).square().sum(1).mean() for d,t in zip(ds,ts))/2
 return ce,preserve,n,float((sim.argmax(1)==target).float().mean())
def loss22(m,teacher,z,device='cuda',ship_aux=False):
 ims=torch.as_tensor(np.stack(z['images']).copy(),device=device,dtype=torch.float32)[:,None]/255
 logs,d=m.maps(ims)
 with torch.no_grad():tl,td=teacher.maps(ims)
 zero=d.sum()*0;ce,preserve,n,recall=correspondence(d,td,z['points0'],z['points1']);ground=z['local_weight']*(ce+.1*preserve)
 positive=0*logs.sum();calibration=positive;distill=positive;blank=positive;equiv=positive;poscells=0;negcells=0
 if m.kind=='b':
  lp=logs.log_softmax(1);shape=lp.shape[0:1]+lp.shape[2:];pos=torch.zeros(shape,device=device,dtype=torch.bool);target=torch.full(shape,64,device=device,dtype=torch.long)
  # Only accepted ground anchors create detector positives. Unknown cells are never negatives.
  for i,p in enumerate([z['points0'],z['points1']]):
   q=torch.as_tensor(np.rint(p),device=device,dtype=torch.long)
   if len(q):pos[i,q[:,1]//8,q[:,0]//8]=True;target[i,q[:,1]//8,q[:,0]//8]=(q[:,1]%8)*8+q[:,0]%8
  cm=torch.as_tensor(np.stack(z['calibration_mask']),device=device).float()[:,None]
  neg=(F.avg_pool2d(cm,8,8)[:,0]>.999)&~pos
  # All 64 pixels of a cell must be evidenced, valid and non-dynamic for calibration.
  eligible=np.stack([v&~i&~s&~u for v,i,s,u in zip(z['valid'],z['ignore'],z['ship_masks'],z['supervision_ignore'])])
  keep=F.avg_pool2d(torch.as_tensor(eligible,device=device).float()[:,None],8,8)[:,0]>.999
  if pos.any():positive=z['local_weight']*F.nll_loss(lp,target,reduction='none')[pos].mean()
  if neg.any():calibration=-lp[:,64][neg].mean()
  if keep.any():distill=(tl.softmax(1)*(tl.log_softmax(1)-lp)).sum(1)[keep].mean()
  if len(z['points0']):
   vals=[];scores=score_map(logs);h,w=ims.shape[-2:]
   for i,p in enumerate([z['points0'],z['points1']]):
    grid=torch.as_tensor(p,device=device)[None,None]/torch.tensor([w-1,h-1],device=device)*2-1
    vals.append(F.grid_sample(scores[i:i+1,None],grid,align_corners=True)[0,0,0])
   equiv=(vals[0]-vals[1]).square().mean()
  null=torch.cat([torch.zeros_like(ims[:1]),torch.full_like(ims[:1],.5)],0)
  blank=-m.maps(null)[0].log_softmax(1)[:,64].mean()
  poscells=int(pos.sum());negcells=int(neg.sum())
 detector=.05*positive+distill+calibration+blank+.2*equiv
 negative=zero;negative_pairs=0
 if z['loss_role']=='no_overlap':
  pp=z['negative_probe_points']
  if min(map(len,pp))>=2:
   dd=[sample_descriptors(torch.as_tensor(p,device=device,dtype=torch.float32)[None],d[i:i+1],8)[0].T for i,p in enumerate(pp)]
   sim=dd[0]@dd[1].T;negative=F.relu(sim-.8).square().mean();negative_pairs=sim.numel()
 aux=zero;ns=0
 if ship_aux:
  sc,sp,ns,_=correspondence(d,td,z['ship_points0'],z['ship_points1']);aux=sc+.1*sp
  if m.kind=='b' and len(z['ship_points0']):
   terms=[];lp=logs.log_softmax(1)
   for i,p in enumerate([z['ship_points0'],z['ship_points1']]):
    q=torch.as_tensor(np.rint(p),device=device,dtype=torch.long);lab=(q[:,1]%8)*8+q[:,0]%8
    terms.append(-lp[i,lab,q[:,1]//8,q[:,0]//8].mean())
   aux=aux+.05*sum(terms)/2
 total=ground+detector+.1*negative+(.15*aux if ship_aux else zero)
 active=bool(n or ns or (negative.detach()>0) or m.kind=='b')
 info=dict(loss=float(total.detach()),ground_loss=float(ce.detach()),preserve=float(preserve.detach()),detector=float(detector.detach()),positive_detector=float(positive.detach()),calibration=float(calibration.detach()),distill=float(distill.detach()),blank=float(blank.detach()),equiv=float(equiv.detach()),negative_loss=float(negative.detach()),ship_loss=float(aux.detach()),ground_points_used=n,ground_points_available=len(z['points0']),ship_points_used=ns,negative_pairs=negative_pairs,ground_weight=z['local_weight'],positive_cells=poscells,calibration_cells=negcells,local_R1=recall,active=active,skip_descriptor_update=not(n or ns or negative.detach()>0))
 return total,info
