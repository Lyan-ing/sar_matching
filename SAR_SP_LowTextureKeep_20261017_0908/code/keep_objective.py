"""One reliable-structure descriptor objective plus same-input preservation."""
from runtime import *
from old_objective import local_coordinates


def configure(model):
    model.requires_grad_(False)
    for n,p in model.named_parameters():p.requires_grad_(n.startswith('sp.convD'))
    return [p for p in model.parameters() if p.requires_grad]


def pair_losses(logits,dense,teacher_dense,item,size,role):
    dense=F.normalize(dense.float(),dim=1);teacher_dense=F.normalize(teacher_dense.detach().float(),dim=1)
    zero=dense.sum()*0;desc=zero;keep=zero;p=torch.as_tensor(item['points'],device=dense.device,dtype=torch.float32);n=p.shape[1]
    info=dict(structure_pairs=0,keep_pairs=0,points=0,keep_positions=0,contrastive_queries=0,positive_only=0,contrastive=0.,positive_alignment=0.,descriptor_R1=None)
    if role=='desc' and n:
        with torch.no_grad():q=local_coordinates(logits.detach(),item,size)
        ds=[sample_descriptors(x[None],dense[i:i+1],8)[0].T for i,x in enumerate(q)]
        near=(torch.cdist(p[0],p[0])<8)|(torch.cdist(p[1],p[1])<8);near.fill_diagonal_(False);allowed=~near;negative=allowed.clone();negative.fill_diagonal_(False);hasneg=negative.any(1)
        sim=(ds[0]@ds[1].T/.1).masked_fill(~allowed,-1e4);labels=torch.arange(n,device=dense.device)
        ce=(F.cross_entropy(sim,labels,reduction='none')[hasneg].mean()+F.cross_entropy(sim.T,labels,reduction='none')[hasneg].mean())/2 if hasneg.any() else zero
        positive=1-(ds[0]*ds[1]).sum(1);align=positive[~hasneg].mean() if (~hasneg).any() else zero
        desc=float(item['local_weight'])*(ce+align)
        info.update(structure_pairs=1,points=n,contrastive_queries=int(hasneg.sum()),positive_only=int((~hasneg).sum()),contrastive=float(ce.detach()),positive_alignment=float(align.detach()),descriptor_R1=float((sim.argmax(1)[hasneg]==labels[hasneg]).float().mean()) if hasneg.any() else None)
    if role=='keep':
        values=[]
        for i in [0,1]:
            xy=torch.as_tensor(item['keep'+str(i)],device=dense.device,dtype=torch.float32)
            if not len(xy):continue
            ds=sample_descriptors(xy[None],dense[i:i+1],8)[0].T
            with torch.no_grad():dt=sample_descriptors(xy[None],teacher_dense[i:i+1],8)[0].T
            values.append((1-(ds*dt).sum(1)).clamp_min(0).mean());info['keep_positions']+=len(xy)
        if values:keep=torch.stack(values).mean();info['keep_pairs']=1
    info.update(desc=float(desc.detach()),keep=float(keep.detach()))
    return desc,keep,info


def terms(model,teacher,x,rs,items,roles,amp=True):
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):
        logits,dense=model.cached_maps(x,rs)
        with torch.no_grad():_,td=teacher.cached_maps(x,rs)
    desc=[];keep=[];infos=[]
    with torch.autocast('cuda',enabled=False):
        for j,(item,role) in enumerate(zip(items,roles)):
            d,k,info=pair_losses(logits[2*j:2*j+2].float(),dense[2*j:2*j+2].float(),td[2*j:2*j+2].float(),item,int(x.shape[-1]*8),role)
            if role=='desc':desc.append(d)
            else:keep.append(k)
            infos.append(info)
    zero=dense.sum()*0
    return (torch.stack(desc).mean() if desc else zero),(torch.stack(keep).mean() if keep else zero),infos
