"""Role-aware data/loss bridge. No model architecture or historical file changes."""
from common22 import *
import ast
_B=None

def _zero(m):
    return next(p for p in m.parameters() if p.requires_grad).sum()*0

def _positive(m,z,p,q,device,minimum=4):
    import torch
    import torch.nn.functional as F
    if len(p)<minimum:return _zero(m),0
    ims=[torch.as_tensor(a.copy(),device=device,dtype=torch.float32)[None,None]/255 for a in z['images']]
    xy=[torch.as_tensor(a,device=device,dtype=torch.float32)[None] for a in [p,q]]
    ds=[m.descriptors(im,pt) for im,pt in zip(ims,xy)]
    d0,d1=[a[0][0].float() for a in ds];b0,b1=[a[1][0].float() for a in ds]
    sim=d0@d1.T/.1;near=(torch.cdist(xy[0][0],xy[0][0])<8)|(torch.cdist(xy[1][0],xy[1][0])<8);near.fill_diagonal_(False);sim=sim.masked_fill(near,-1e4)
    target=torch.arange(len(p),device=device)
    ce=(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2
    preserve=((d0-b0).square().sum(1).mean()+(d1-b1).square().sum(1).mean())/2
    return ce+.1*preserve,len(p)

def _nonoverlap(m,z,device):
    import torch
    import torch.nn.functional as F
    if z['loss_role']!='no_overlap':return _zero(m),0
    p,q=z['negative_probe_points']
    if min(len(p),len(q))<2:return _zero(m),0
    # These are provable NONcorrespondences from disjoint source FOV, not positive anchors.
    out=[]
    for im,xy in zip(z['images'],[p,q]):
        a=torch.as_tensor(im.copy(),device=device,dtype=torch.float32)[None,None]/255
        b=torch.as_tensor(xy,device=device,dtype=torch.float32)[None]
        out.append(m.descriptors(a,b))
    d0,d1=[a[0][0] for a in out];sim=d0@d1.T
    loss=F.relu(sim-.8).square().mean()
    return loss,int(sim.numel())

def m2_loss(model,z,device='cpu',ship_weight=.15,negative_weight=.1):
    main,n=_positive(model,z,z['points0'],z['points1'],device)
    aux,ns=_positive(model,z,z['ship_points0'],z['ship_points1'],device) if z['use_ship_aux'] else (_zero(model),0)
    neg,nn=_nonoverlap(model,z,device)
    total=z['local_weight']*main+ship_weight*aux+negative_weight*neg
    return total,dict(ground_points_used=n,ship_points_used=ns,negative_pairs=nn,local_weight=z['local_weight'],
        calibration_pixels=sum(int(a.sum()) for a in z['calibration_mask']),
        detector_calibration_trainable=False,skip_descriptor_update=not(n or ns or nn),
        skip_reason='M2 detector frozen; no fabricated matching positives for calibration/unknown regions' if not(n or ns or nn) else None,
        ground_loss=float(main.detach()),ship_loss=float(aux.detach()),negative_loss=float(neg.detach()))

def _compile_B():
    import torch
    import torch.nn.functional as F
    from lightglue.superpoint import sample_descriptors
    path=BASE/'SAR_SP_LG_StructureStable_20261013_2210/code/network_corrected.py';source=path.read_text()
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='joint_loss');code=ast.get_source_segment(source,node)
    changes={
      'def joint_loss(m,teacher,z):':"def joint_loss(m,teacher,z,device='cpu'):",
      'from m3_model import structure_targets':'',
      "device='cuda'":'device=device',
      '/255*2-1':'/torch.tensor([ims.shape[-1]-1,ims.shape[-2]-1],device=device)*2-1',
      'if len(p0)>=2:':'if len(p0)>=4:',
      'positive &= cells & structured':'positive = positive  # V2.2 explicitly validated anchor cells, not global spatial gates',
      'negative=(~structured | ~cells)&nondynamic':"negative=(F.avg_pool2d(torch.tensor(np.stack(z['calibration_mask']),device=device).float()[:,None],8,8)[:,0]>.999)&~positive",
      'det=weak[positive].mean() if positive.any() else logs.sum()*0':"det=z['local_weight']*weak[positive].mean() if positive.any() else logs.sum()*0",
      'total=match+.1*preserve+det':"total=z['local_weight']*(match+.1*preserve)+det"}
    for before,after in changes.items():
        assert before in code,before
        code=code.replace(before,after)
    def score_map(logits):
        s=logits.softmax(1)[:,:64].permute(0,2,3,1);b,h,w,_=s.shape
        return s.reshape(b,h,w,8,8).permute(0,1,3,2,4).reshape(b,h*8,w*8)
    ns=dict(torch=torch,np=np,F=F,sample_descriptors=sample_descriptors,score_map=score_map,warp=transform_points,valid_at=bilinear_at,
        structure_targets=lambda pair,teacher:(pair['points0'],pair['points1']))
    exec(compile(code,str(path)+'::V22_ROLE_BRIDGE','exec'),ns);return ns['joint_loss']

def b_loss(model,teacher,z,device='cpu',ship_weight=.15,negative_weight=.1):
    global _B
    import torch
    import torch.nn.functional as F
    if _B is None:_B=_compile_B()
    # The existing bridge's exclusion rule protects all unknown regions and all ships.
    from loss_bridge import legacy_view
    ground,info=_B(model,teacher,legacy_view(z),device=device)
    aux,ns=_positive(model,z,z['ship_points0'],z['ship_points1'],device) if z['use_ship_aux'] else (_zero(model),0)
    aux_detect=_zero(model)
    if z['use_ship_aux'] and len(z['ship_points0']):
        ims=torch.as_tensor(np.stack(z['images']).copy(),device=device,dtype=torch.float32)[:,None]/255
        lp=model.maps(ims)[0].log_softmax(1);terms=[]
        for i,p in enumerate([z['ship_points0'],z['ship_points1']]):
            xy=torch.as_tensor(np.rint(p),device=device,dtype=torch.long);lab=(xy[:,1]%8)*8+xy[:,0]%8
            terms.append(-lp[i,lab,xy[:,1]//8,xy[:,0]//8].mean())
        aux_detect=sum(terms)/2
    neg,nn=_nonoverlap(model,z,device);total=ground+ship_weight*(aux+.05*aux_detect)+negative_weight*neg
    info.update(ground_label_points=len(z['points0']),ship_points_used=ns,ship_detector_points=len(z['ship_points0']) if z['use_ship_aux'] else 0,
        negative_pairs=nn,calibration_pixels=sum(int(m.sum()) for m in z['calibration_mask']),unknown_pixels=sum(int(m.sum()) for m in z['supervision_ignore']),
        ship_aux_weight=ship_weight if z['use_ship_aux'] else 0.,negative_weight=negative_weight,graded_local_weight=z['local_weight'])
    return total,info
