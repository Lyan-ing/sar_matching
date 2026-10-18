"""Graded local/background supervision and separately opt-in synthetic ship labels."""
from common22 import *
import os,copy
SAMPLE21=v21.sample_geometry

def features(r):
    with np.load(cachepath(r)) as z:f={k:z[k] for k in z.files}
    return f

def ground_points(r,f,cfg):
    decision=cfg.get('semantic_overrides',{}).get(r['image_path'])
    if not decision:return f['ground']
    p=f['proposal'];keep=np.zeros(len(p),bool)
    for x0,y0,x1,y1 in decision['boxes_xyxy']:
        keep|=(p[:,0]>=x0)&(p[:,0]<=x1)&(p[:,1]>=y0)&(p[:,1]<=y1)
    return p[keep]

def _support(shape,p,safe):
    a=np.zeros(shape,np.uint8)
    for xy in p:cv2.circle(a,tuple(np.rint(xy).astype(int)),8,255,-1)
    return (a>0)&safe

def labels(r,row,cfg,f=None,arrays=None):
    f=features(r) if f is None else f;Hs=[np.asarray(h) for h in row['H_source_to_view']];n=row['width']
    if arrays is None:
        valid=[read_gray(p)>0 for p in row['valid_masks']];safe=[read_gray(p)>0 for p in row['safe_static_masks']];ignore=[read_gray(p)>0 for p in row['ignore_masks']]
    else:valid=arrays['valid'];safe=arrays['safe_static'];ignore=arrays['ignore']
    psrc=ground_points(r,f,cfg);source_support=_support(f['source_safe'].shape,psrc,f['source_safe'])
    masks=[(warp(source_support.astype('uint8')*255,h,n,mask=True)>0)&v&~i for h,v,i in zip(Hs,safe,ignore)]
    pp=[transform_points(h,psrc) for h in Hs]
    ok=bilinear_at(masks[0],pp[0])&bilinear_at(masks[1],pp[1]);p,q=thin(pp[0][ok].astype('float32'),pp[1][ok].astype('float32'))
    nooverlap=row['profile']=='none'
    if nooverlap:p=np.empty((0,2),np.float32);q=p.copy();masks=[np.zeros_like(x) for x in valid]
    role='no_overlap' if nooverlap else 'structural_positive' if global_ok(p,q,n) else 'limited_structure' if len(p) else 'no_structural_positive'
    if not len(p):masks=[np.zeros_like(x) for x in masks]
    calib=[(warp(f['calibration'].astype('uint8')*255,h,n,mask=True)>0)&v&~ig&~dilate(g,4) for h,v,ig,g in zip(Hs,safe,ignore,masks)]
    # Local positives are NEVER interpreted as enough global Homography support.
    ssrc=f['ship'];sp=[transform_points(h,ssrc) for h in Hs]
    shipm=[(warp(f['segmented_ship'].astype('uint8')*255,h,n,mask=True)>0)&v&~ig for h,v,ig in zip(Hs,valid,ignore)]
    sk=bilinear_at(shipm[0],sp[0])&bilinear_at(shipm[1],sp[1]);sp0,sp1=sp[0][sk],sp[1][sk];sid=f['ship_instance_index'][sk]
    if nooverlap:sp0=np.empty((0,2));sp1=sp0.copy();sid=np.empty(0,int)
    # Descriptor alternatives closer than 8px are not independent negatives.
    chosen=[]
    for k,(a,b) in enumerate(zip(sp0,sp1)):
        if all(np.linalg.norm(a-sp0[j])>=8 and np.linalg.norm(b-sp1[j])>=8 for j in chosen):chosen.append(k)
        if len(chosen)>=64:break
    sp0=sp0[chosen].astype('float32');sp1=sp1[chosen].astype('float32');sid=sid[chosen]
    if not len(sp0):shipm=[np.zeros_like(x) for x in shipm]
    # Unknown is an explicit supervision ignore, distinct from synthetic appearance occlusion.
    unknown=[v&~g&~c for v,g,c in zip(valid,masks,calib)]
    negative=[]
    for i in range(2):
        yy,xx=np.mgrid[12:n-12:24,12:n-12:24];xy=np.c_[xx.ravel(),yy.ravel()].astype('float32')
        xy=xy[bilinear_at(safe[i]&~ignore[i],xy)] if nooverlap else np.empty((0,2),np.float32)
        negative.append(xy[:64])
    return dict(points0=p,points1=q,static=masks,calibration_mask=calib,supervision_ignore=unknown,
        ship_points0=sp0,ship_points1=sp1,ship_aux_masks=shipm,ship_instance_index=sid,negative_probe_points=negative,
        loss_role=role,global_eligible=role=='structural_positive',local_descriptor_eligible=len(p)>=4,local_detector_eligible=len(p)>0,
        local_weight=1. if role=='structural_positive' else min(1.,len(p)/16.),
        ship_descriptor_eligible=len(sp0)>=4,ship_detector_eligible=len(sp0)>0,
        spatial_support=[spatial_support(x,n) for x in [p,q]],source_ground_points=len(psrc),calibration_pixels=[int(x.sum()) for x in calib])

def materialize_labels(r,row,cfg,origin,folder,a=None):
    a=labels(r,row,cfg) if a is None else a;folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);out=dict(row)
    def masks(name,arr):
        paths=[]
        for i,m in enumerate(arr):
            p=folder/f'{name}_{i}.png';assert cv2.imwrite(str(p),m.astype('uint8')*255,[cv2.IMWRITE_PNG_COMPRESSION,2]);paths.append(str(p))
        return paths
    npz=folder/'local_and_ship_labels.npz';np.savez_compressed(npz,points0=a['points0'],points1=a['points1'],ship_points0=a['ship_points0'],ship_points1=a['ship_points1'],
        ship_instance_index=a['ship_instance_index'],negative0=a['negative_probe_points'][0],negative1=a['negative_probe_points'][1])
    out.update(version='2.2',origin=origin,old_loss_role=row.get('loss_role'),old_static_correspondences=row['correspondences'],correspondences=str(npz),
        static_masks=masks('ground_local',a['static']),calibration_masks=masks('calibration_evidence',a['calibration_mask']),supervision_ignore_masks=masks('unknown_supervision',a['supervision_ignore']),
        ship_aux_masks=masks('synthetic_ship_aux',a['ship_aux_masks']),
        structural_anchor_count=len(a['points0']),ship_aux_count=len(a['ship_points0']),loss_role=a['loss_role'],global_eligible=a['global_eligible'],
        source_structural_candidates_v21=row.get('source_structural_candidates'),source_structural_candidates=a['source_ground_points'],
        is_low_texture_or_no_structure=a['loss_role']=='no_structural_positive',
        local_descriptor_eligible=a['local_descriptor_eligible'],local_detector_eligible=a['local_detector_eligible'],local_weight=a['local_weight'],
        ship_descriptor_eligible=a['ship_descriptor_eligible'],ship_aux_enabled_by_default=False,ship_supervision='same-source synthetic auxiliary ONLY; real static GT excludes all ships',
        spatial_support=a['spatial_support'],source_ground_points=a['source_ground_points'],calibration_pixels=a['calibration_pixels'],
        scene_semantics_override=cfg.get('semantic_overrides',{}).get(r['image_path']),
        global_evaluation_thresholds_unchanged=True,physical_multiview_GT=False,structure_extracted_before_appearance_noise=True,
        relation_id=('old:' if origin=='reused_V21' else 'new:')+row['sample_id'])
    return out

def load_pair(row,use_ship_aux=False):
    z=v21.load_training_pair(row,max_points=None)
    with np.load(row['correspondences']) as f:
        z.update(ship_points0=f['ship_points0'] if use_ship_aux else np.empty((0,2),np.float32),ship_points1=f['ship_points1'] if use_ship_aux else np.empty((0,2),np.float32),
                 negative_probe_points=[f['negative0'],f['negative1']])
    z.update(supervision_ignore=[read_gray(p)>0 for p in row['supervision_ignore_masks']],ship_aux_masks=[read_gray(p)>0 for p in row['ship_aux_masks']],
        global_eligible=row['global_eligible'],local_descriptor_eligible=row['local_descriptor_eligible'],local_weight=row['local_weight'],
        loss_role=row['loss_role'],relation_id=row['relation_id'],use_ship_aux=use_ship_aux)
    return z

def tensors22(z,device='cpu'):
    import torch
    out=v21.tensors(z,device)
    for key in ['supervision_ignore','ship_aux_masks']:
        out[key]=torch.as_tensor(np.stack(z[key]),device=device,dtype=torch.bool)[:,None]
    for key in ['ship_points0','ship_points1']:out[key]=torch.as_tensor(z[key],device=device,dtype=torch.float32)
    return out

def sample_geometry22(shape,points,spec,cfg):
    """Same angle/scale/overlap constraints, with a frozen native ROI center hint."""
    if 'roi_center' not in spec:return SAMPLE21(shape,points,spec,cfg)
    rng=np.random.default_rng(spec['geometry_seed']);h,w=shape;g=cfg['geometry'];negative=spec['profile']=='none'
    size=min(spec['requested_size'],int(min(h,w)/(2.6 if negative else 1.48)/32)*32)
    if size<64:raise ValueError('source too small')
    valid_src=np.ones(shape,np.float32);lo,hi=g['low_overlap'] if spec['profile']=='low' else g['normal_overlap'];b=spec['angle_bin'];bounds=g['positive_angle_bins'][b:b+2];history=[]
    for trial in range(g['max_attempts']):
        theta=float(rng.uniform(bounds[0]+.4,bounds[1]-.4))*rng.choice([-1,1]) if not negative else 0.;orient=float(rng.uniform(-5,5))
        scales=rng.uniform(*(g['high_angle_scale'] if b>=3 else g['small_angle_scale']),2)
        center=np.array(spec['roi_center'],float)+rng.uniform(-.02,.02,2)*size
        if negative:c0=np.array([w*.25,h*.5]);c1=np.array([w*.75,h*.5]);orient=theta=0.;scales[:]=1.
        else:
            shift=rng.uniform(.55,.85)*size if spec['profile']=='low' else rng.uniform(.03,.25)*size
            direction=rng.uniform(0,2*np.pi);d=np.array([np.cos(direction),np.sin(direction)])*shift;c0=center-d/2;c1=center+d/2
        perspective=g['perspective_fraction'] if spec['profile']=='projective' else 0.
        Hs=[view_matrix(c,size,a,s,0.,perspective,rng) for c,a,s in [(c0,orient,scales[0]),(c1,orient+theta,scales[1])]]
        valid=[erode(cv2.warpPerspective(valid_src,H,(size,size),flags=cv2.INTER_LINEAR)>1-1e-6,1) for H in Hs]
        H=Hs[1]@np.linalg.inv(Hs[0]);H/=H[2,2];ov=overlap_mask(*(m.astype('uint8')*255 for m in valid),H);fr=[ratio(o,v) for o,v in zip(ov,valid)];angle=angle_of(H,size)
        if min(float(v.mean()) for v in valid)<g['min_valid']:history.append('valid');continue
        if negative and max(fr)>0:history.append('negative_overlap');continue
        if not negative and not lo<=min(fr)<=hi:history.append('overlap');continue
        if not negative and angle_bin(angle)!=['00_15','15_30','30_60','60_90','90_120'][b]:history.append('angle');continue
        return dict(Hs=Hs,H=H,valid=valid,overlap=fr,angle=angle,size=size,trial=trial,rejected=history)
    raise ValueError('frozen ROI geometry infeasible '+str({k:history.count(k) for k in set(history)}))

def generate_new(r,spec,cfg):
    saved=v21.sample_geometry
    # In-process dependency injection only; imported historical source is never edited.
    v21.sample_geometry=sample_geometry22
    try:return v21.generate_arrays(r,spec,cfg)
    finally:v21.sample_geometry=saved

def online_pair(r,spec,cfg,arm='V2-R',use_ship_aux=False):
    geom,arms,meta=generate_new(r,spec,cfg)
    failed=[name+':'+reason for name,a in arms.items() for reason in a['qa_failures']]
    if failed:raise ValueError('R/S shared input gate '+repr(failed))
    a=arms[arm];row=dict(H_source_to_view=[h.tolist() for h in geom['Hs']],width=geom['size'],profile=spec['profile'])
    lab=labels(r,row,cfg,arrays=a)
    return dict(images=a['images'],valid=a['valid'],static=lab['static'],ship_masks=a['ship_masks'],ignore=a['ignore'],visibility=a['visibility'],safe_static=a['safe_static'],
        structure_support=lab['static'],calibration_mask=lab['calibration_mask'],supervision_ignore=lab['supervision_ignore'],ship_aux_masks=lab['ship_aux_masks'],
        points0=lab['points0'],points1=lab['points1'],ship_points0=lab['ship_points0'] if use_ship_aux else np.empty((0,2),np.float32),ship_points1=lab['ship_points1'] if use_ship_aux else np.empty((0,2),np.float32),
        negative_probe_points=lab['negative_probe_points'],H_A=geom['Hs'][0],H_B=geom['Hs'][1],H_AB=geom['H'],overlap=min(geom['overlap']),
        source=r,source_origin_xy=[0,0],regime=spec['profile'],seed=spec['geometry_seed'],dynamic_occlusion=bool(meta['dynamic_events']),
        loss_role=lab['loss_role'],global_eligible=lab['global_eligible'],local_descriptor_eligible=lab['local_descriptor_eligible'],local_weight=lab['local_weight'],
        use_ship_aux=use_ship_aux,arm=arm,sampling_spec=spec,appearance=a['appearance'])
