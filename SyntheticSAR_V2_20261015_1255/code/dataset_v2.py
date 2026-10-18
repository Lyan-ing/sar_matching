"""Replayable native-source synthetic SAR display-domain augmentation and pair adapter.

No SAR physical calibration is inferred. `static` is deliberately the conservative
STRUCTURAL supervision mask consumed by old losses, not merely valid minus ships.
"""
from common_v2 import *
from sarreg.synthesis import _apply_augment
import os, copy

def source_material(r,cfg):
    im=read_gray(r['image_path']); h,w=im.shape
    union=np.zeros_like(im); instances=[]
    for a in r['annotations']:
        m,method=annotation_mask(im.shape,a); union |= m
        instances.append((a['id'],m,method))
    sea=read_gray(r['sea_mask_path'])>0 if r.get('sea_mask_path') else None
    if sea is not None and sea.shape!=im.shape: raise ValueError('sea/source shape mismatch')
    safe=~dilate(union, cfg['structure']['source_ship_guard'])
    safe &= erode(np.ones_like(im,bool),10)
    category=r['scene_category']
    enabled=category in ['nearshore_vessels','port_quay_dense_vessels','other_land_structures']
    if sea is not None: safe &= ~dilate(sea,2)
    x=im.astype('float32')/255.; a=cv2.GaussianBlur(x,(0,0),2.); b=cv2.GaussianBlur(x,(0,0),4.)
    localstd=np.sqrt(np.maximum(cv2.GaussianBlur(b*b,(0,0),7)-cv2.GaussianBlur(b,(0,0),7)**2,0))
    eig=cv2.cornerMinEigenVal(b,7,3)
    eligible=safe & (localstd>=cfg['structure']['source_local_std_min']) & (eig>=cfg['structure']['source_corner_min_eigenvalue']) if enabled else np.zeros_like(safe)
    corners=cv2.goodFeaturesToTrack(a,1000,.015,12,mask=eligible.astype('uint8')*255,blockSize=7)
    coarse=cv2.goodFeaturesToTrack(b,1000,.015,10,mask=eligible.astype('uint8')*255,blockSize=7)
    p=corners[:,0] if corners is not None else np.empty((0,2),np.float32)
    q=coarse[:,0] if coarse is not None else np.empty((0,2),np.float32)
    if len(p) and len(q): p=p[np.min(np.sum((p[:,None]-q[None])**2,axis=2),axis=1)<=cfg['structure']['multiscale_tolerance']**2]
    else: p=np.empty((0,2),np.float32)
    support=np.zeros_like(im)
    for xy in p: cv2.circle(support,tuple(np.rint(xy).astype(int)),cfg['structure']['support_radius'],255,-1)
    support=(support>0)&safe
    return im,union,instances,sea,p,support

def sample_geometry(shape,points,spec,cfg):
    rng=np.random.default_rng(spec['geometry_seed']); h,w=shape
    g=cfg['geometry']; requested=spec['requested_size']; negative=spec['profile']=='none'
    headroom=2.6 if negative else 1.48
    size=min(requested,int(min(h,w)/headroom/32)*32)
    if size<64: raise ValueError('source too small without preupsampling')
    source_valid=np.ones(shape,np.float32)
    lo,hi=g['low_overlap'] if spec['profile']=='low' else g['normal_overlap']
    targetbin=spec['angle_bin']; bounds=g['positive_angle_bins'][targetbin:targetbin+2]
    history=[]
    for trial in range(g['max_attempts']):
        theta=float(rng.uniform(bounds[0]+.4,bounds[1]-.4))*rng.choice([-1,1]) if not negative else 0.
        orient=float(rng.uniform(-5,5))
        scales=rng.uniform(*(g['high_angle_scale'] if targetbin>=3 else g['small_angle_scale']),2)
        # Source corner centered sampling is image-only, independent of any learned matcher.
        center=points[int(rng.integers(len(points)))].copy() if len(points) and rng.random()<.8 else np.array([w/2,h/2])+rng.uniform(-.15,.15,2)*[w,h]
        margin=size*.70/min(scales)+4
        center=np.clip(center,[min(margin,w/2),min(margin,h/2)],[max(w-margin,w/2),max(h-margin,h/2)])
        if negative:
            center0=np.array([w*.25,h*.5]);center1=np.array([w*.75,h*.5]);orient=theta=0.;scales[:]=1.
        else:
            shift=rng.uniform(.55,.85)*size if spec['profile']=='low' else rng.uniform(.03,.25)*size
            direction=rng.uniform(0,2*np.pi);d=np.array([np.cos(direction),np.sin(direction)])*shift
            center0=center-d*.5;center1=center+d*.5
        perspective=g['perspective_fraction'] if spec['profile']=='projective' else 0.
        Hs=[view_matrix(c,size,a,s,0.,perspective,rng) for c,a,s in [(center0,orient,scales[0]),(center1,orient+theta,scales[1])]]
        # Full bilinear image support, not just nearest-neighbor source-one raster.
        valid=[erode(cv2.warpPerspective(source_valid,H,(size,size),flags=cv2.INTER_LINEAR)>1-1e-6,1) for H in Hs]
        H=Hs[1]@np.linalg.inv(Hs[0]);H/=H[2,2]
        ov=overlap_mask(*(v.astype('uint8')*255 for v in valid),H)
        overlap=[ratio(o,v) for o,v in zip(ov,valid)]; angle=angle_of(H,size)
        score=min(overlap);vf=[float(v.mean()) for v in valid]
        if min(vf)<g['min_valid']:history.append('valid_fraction');continue
        if negative:
            if max(overlap)>0:history.append('negative_overlap');continue
        elif not lo<=score<=hi:history.append('overlap');continue
        elif angle_bin(angle)!=['00_15','15_30','30_60','60_90','90_120'][targetbin]:history.append('angle_bin');continue
        return dict(Hs=Hs,H=H,valid=valid,overlap=overlap,angle=angle,size=size,trial=trial,rejected=history)
    raise ValueError('geometry constraints infeasible: '+str({k:history.count(k) for k in set(history)}))

def display_augment(im,valid,arm,seed,cfg):
    rng=np.random.default_rng(seed); c=cfg['appearance']; ignore=np.zeros_like(valid);events=[]
    if arm=='V1-matched':
        x,p=_apply_augment(im,dict(enabled=True,gamma_min=.85,gamma_max=1.15,contrast_min=.85,contrast_max=1.15,brightness_max=.08,speckle_sigma=.18,blur_probability=.18,blur_sigma_max=.85),rng)
        x[~valid]=0
        return x,ignore,dict(domain='display_DN',legacy_medium=p,seed=seed),events
    x=im.astype('float32')/255.;base=x.copy()
    alpha=float(rng.uniform(*c['smooth_residual_blend']))
    replacement=c.get('fine_texture_replacement',False)
    if replacement:
        # Approximate DISPLAY structure/texture separation, not latent SAR reflectivity.
        # Protect coarse boundaries; partially replace fine residual using independent
        # random texture with local measured DN variance, avoiding arbitrary extra noise.
        coarse=cv2.GaussianBlur(x,(0,0),2.)
        gx=cv2.Sobel(coarse,-1,1,0,ksize=3,scale=.125);gy=cv2.Sobel(coarse,-1,0,1,ksize=3,scale=.125)
        texture_weight=np.clip(1-np.hypot(gx,gy)/c['coarse_edge_protection_gradient'],.25,1.)
        smooth=cv2.GaussianBlur(x,(0,0),c['texture_lowpass_sigma'])
        residual=x-smooth
        residual_rms=np.sqrt(np.maximum(cv2.GaussianBlur(residual*residual,(0,0),4),0))
        x=x-alpha*texture_weight*residual
    else:
        smooth=cv2.bilateralFilter(x,5,c['bilateral_sigma_color'],c['bilateral_sigma_space'])
        x=(1-alpha)*x+alpha*smooth
    gamma=float(rng.uniform(*c['gamma']));gain=float(rng.uniform(*c['gain']));contrast=float(rng.uniform(*c['contrast']));offset=float(rng.uniform(*c['offset']))
    # Display-domain monotone mappings only; no inverse physical intensity is assumed.
    x=gain*(np.maximum(x,0)**gamma); x=(x-.5)*contrast+.5+offset
    shape=float(rng.uniform(*c['gamma_field_shape_proxy']));sigma=float(rng.uniform(*c['display_noise_sigma']))
    field=(rng.gamma(shape,1/shape,x.shape)-1)*np.sqrt(shape)
    corr=float(rng.uniform(*c['correlated_noise_mix']));rho=float(rng.uniform(*c['correlation_sigma']))
    blurred=cv2.GaussianBlur(field.astype('float32'),(0,0),rho);blurred=(blurred-blurred.mean())/max(float(blurred.std()),1e-6)
    field=(1-corr)*field+corr*blurred;field=(field-field.mean())/max(float(field.std()),1e-6)
    if replacement:
        noise_amplitude=np.clip(sigma+alpha*texture_weight*residual_rms,.008,c['display_noise_sigma_cap'])
        x+=noise_amplitude*field
    else:
        noise_amplitude=sigma*(.25+.75*np.sqrt(np.clip(x,0,1)))
        x+=noise_amplitude*field
    # Smooth, non-periodic local clutter; no tiled noise or random rectangle boundaries.
    clutter=cv2.GaussianBlur(rng.normal(size=x.shape).astype('float32'),(0,0),max(8,x.shape[0]/16))
    clutter=clutter/max(float(clutter.std()),1e-6)*c['clutter_sigma'];x+=np.clip(clutter,-.03,.03)
    sx,sy=map(float,rng.uniform(*c['anisotropic_blur_sigma'],2))
    x=cv2.sepFilter2D(x,-1,cv2.getGaussianKernel(5,sx),cv2.getGaussianKernel(5,sy))
    if arm=='V2-S' and rng.random()<c['local_scattering_probability']:
        yy,xx=np.indices(x.shape);bright=(x>max(.25,float(np.quantile(x[valid],.92))))&erode(valid,20)
        ys,xs=np.where(bright)
        n=int(rng.integers(c['local_blobs'][0],c['local_blobs'][1]+1))
        for _ in range(n if len(xs) else 0):
            k=int(rng.integers(len(xs)));cx,cy=int(xs[k]),int(ys[k]);rad=float(rng.uniform(*c['blob_sigma']))
            blob=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*rad*rad)).astype('float32')
            factor=float(rng.uniform(*c['blob_gain']));drop=bool(rng.random()<c['visibility_dropout_probability'])
            if drop:factor=float(rng.uniform(.35,.55))
            changed=(blob>.05)&valid
            if float((ignore|changed).mean())>c['local_change_max_fraction']:continue
            x*=1+(factor-1)*blob;ignore|=changed
            events.append(dict(kind='response_missing' if drop else 'scatter_response_proxy',center_xy=[cx,cy],sigma=rad,gain=factor,ignore_pixels=int(changed.sum())))
    raw=x[valid];p=dict(seed=seed,domain='display_DN',residual_blend=alpha,gamma=gamma,gain=gain,contrast=contrast,offset=offset,
        gamma_shape_proxy_not_ENL=shape,noise_sigma=sigma,correlated_mix=corr,correlation_sigma=rho,blur_sigma_xy=[sx,sy],clutter_sigma=c['clutter_sigma'],
        preclip_low_fraction=float((raw<0).mean()),preclip_high_fraction=float((raw>1).mean()),local_events=events,
        texture_residual_replaced=replacement,display_noise_amplitude_median=float(np.median(noise_amplitude[valid])),
        display_noise_amplitude_max=float(noise_amplitude[valid].max()),
        latent_reflectivity_estimated=False,shared_original_speckle_remaining=True)
    x=np.round(np.clip(x,0,1)*255).astype('uint8');x[~valid]=0
    return x,ignore,p,events

def generate_arrays(r,spec,cfg):
    im,union,instances,sea,points,structure=source_material(r,cfg)
    geom=sample_geometry(im.shape,points,spec,cfg);size=geom['size'];Hs=geom['Hs'];H=geom['H'];valid=geom['valid']
    native_views=[warp(im,h,size) for h in Hs]
    vessels=[warp(union,h,size,mask=True)>0 for h in Hs]
    visible_counts=[];visible_partial=[]
    for vi,h in enumerate(Hs):
        items=[]
        for id,m,method in instances:
            area=int(np.count_nonzero((warp(m,h,size,mask=True)>0)&valid[vi]))
            if area:items.append(dict(id=str(id),visible_pixels=area,method=method))
        visible_counts.append(len(items));visible_partial.append(items)
    dynamic=[np.zeros((size,size),bool),np.zeros((size,size),bool)];dynamic_events=[]
    rng=np.random.default_rng(spec['dynamic_seed'])
    # Soft attenuation of actual annotated ship response is a dynamic visibility proxy.
    # No made-up displacement labels. Every instance, including partial ships, stays excluded.
    if rng.random()<cfg['appearance']['dynamic_ship_probability'] and vessels[1].any():
        m=vessels[1].astype('float32');weight=cv2.GaussianBlur(m,(0,0),1.)
        dynamic[1]=dilate(weight>.01,3)&valid[1]
        native_views[1]=np.round(native_views[1]*(1-.65*weight)).astype('uint8')
        dynamic_events.append(dict(kind='annotated_ship_visibility_attenuation',pixels=int(dynamic[1].sum())))
    safe=[v&~dilate(m,cfg['structure']['view_ship_guard'])&~d for v,m,d in zip(valid,vessels,dynamic)]
    structural=[(warp(structure.astype('uint8')*255,h,size,mask=True)>0)&s for h,s in zip(Hs,safe)]
    pp=[transform_points(h,points) for h in Hs]
    sea_masks=[(warp(sea.astype('uint8')*255,h,size,mask=True)>0)&v for h,v in zip(Hs,valid)] if sea is not None else None
    common_ignore=[d.copy() for d in dynamic]
    arm_values={}
    for arm in cfg['arms']:
        images=[];ignored=[];params=[];local=[]
        for i in range(2):
            seed=stable_seed(spec['appearance_seeds'][i], 'V1' if arm=='V1-matched' else 'V2-common-RS')
            x,m,p,events=display_augment(native_views[i],valid[i],arm,seed,cfg)
            images.append(x);ignored.append(dilate(m,5)|common_ignore[i]);params.append(p);local.extend(events)
        # BOTH coordinates of a destroyed/hidden response are removed from supervision.
        projected=[warp(ignored[1].astype('uint8')*255,np.linalg.inv(H),size,mask=True)>0,
                   warp(ignored[0].astype('uint8')*255,H,size,mask=True)>0]
        ignored=[(a|b)&v for a,b,v in zip(ignored,projected,valid)]
        static=[s&~m for s,m in zip(structural,ignored)]
        keep=bilinear_at(static[0],pp[0])&bilinear_at(static[1],pp[1])
        p,q=pp[0][keep].astype('float32'),pp[1][keep].astype('float32')
        # At most 128 spatially ordered, persistent source-corner anchors; never fill from sea.
        if len(p)>128:
            order=np.lexsort((p[:,0],p[:,1]));ids=order[np.linspace(0,len(order)-1,128).astype(int)];p,q=p[ids],q[ids]
        support=[spatial_support(x,size) for x in [p,q]]
        c=cfg['structure']; adequate=all(s['count']>=c['pair_min_points'] and s['cells4x4']>=c['pair_min_cells4x4'] and s['hull_fraction']>=c['pair_min_hull'] and s['eigen_ratio']>=c['pair_min_eigen_ratio'] for s in support)
        role='no_overlap' if spec['profile']=='none' else 'structural_positive' if adequate else 'limited_structure' if len(p)>=2 else 'no_structural_positive'
        # Old losses use `static`; all low/no-support samples get zero match supervision.
        if not adequate:
            p=np.empty((0,2),np.float32);q=p.copy();static=[np.zeros_like(m) for m in static]
        calibration=[]
        for i in range(2):
            m=sea_masks[i]&safe[i]&~ignored[i] if sea_masks is not None else safe[i]&~ignored[i] if r['scene_category']=='offshore_vessels_low_texture_sea' else np.zeros_like(safe[i])
            calibration.append(m)
        intens=[];fail=[]
        for i,x in enumerate(images):
            z=x[valid[i]];saturated=float((z==255).mean());std=float(z.std())
            intens.append(dict(min=int(z.min()),max=int(z.max()),std=std,saturated_fraction=saturated,zero_fraction=float((z==0).mean()),q01_q50_q99=np.quantile(z,[.01,.5,.99]).tolist()))
            if std<cfg['acceptance']['valid_std_min']:fail.append(f'view{i}:constant_or_collapsed')
            if saturated>cfg['acceptance']['saturated_fraction_max']:fail.append(f'view{i}:saturated')
        # Source-space safe mask differs from trainable static support, retain both explicitly.
        arm_values[arm]=dict(images=images,valid=valid,static=static,ship_masks=vessels,ignore=ignored,visibility=[v&~m for v,m in zip(valid,ignored)],
            safe_static=safe,structure_support=structural,calibration=calibration,points0=p,points1=q,appearance=params,local_events=local,
            support=support,role=role,intensity=intens,qa_failures=fail)
    return geom,arm_values,dict(source_shape=list(im.shape),source_structural_candidates=len(points),visible_ship_counts=visible_counts,visible_ship_instances=visible_partial,
        sea_fraction=[float(m[v].mean()) for m,v in zip(sea_masks,valid)] if sea_masks is not None else [None,None],
        dynamic_events=dynamic_events,source_units=r['data_units'],source_preupsampling=False)

def array_pair(r,spec,cfg,arm='V2-R'):
    geom,values,meta=generate_arrays(r,spec,cfg);a=values[arm]
    if cfg['acceptance']['common_arm_gate']:
        failures=[f'{name}:{why}' for name,item in values.items() for why in item['qa_failures']]
        if failures:raise ValueError('common-arm appearance QA: '+repr(failures))
    if a['qa_failures']:raise ValueError('appearance QA: '+repr(a['qa_failures']))
    return dict(images=a['images'],valid=a['valid'],static=a['static'],ship_masks=a['ship_masks'],ignore=a['ignore'],visibility=a['visibility'],
        safe_static=a['safe_static'],structure_support=a['structure_support'],calibration_mask=a['calibration'],
        points0=a['points0'],points1=a['points1'],H_A=geom['Hs'][0],H_B=geom['Hs'][1],H_AB=geom['H'],overlap=min(geom['overlap']),
        source=r,source_origin_xy=[0,0],regime=spec['profile'],dynamic_occlusion=bool(meta['dynamic_events']),seed=spec['geometry_seed'],
        loss_role=a['role'],appearance=a['appearance'],sampling_spec=spec,metadata=meta,arm=arm)

def load_training_pair(row,max_points=128):
    if row['qa_state']!='usable':raise ValueError('quarantined sample')
    read=lambda key:[read_gray(p)>0 for p in row[key]]
    with np.load(row['correspondences']) as z:p=z['points0'];q=z['points1']
    static=read('static_masks');keep=bilinear_at(static[0],p)&bilinear_at(static[1],q);p,q=p[keep],q[keep]
    if max_points is not None and len(p)>max_points:
        ids=np.linspace(0,len(p)-1,max_points).astype(int);p,q=p[ids],q[ids]
    return dict(images=[read_gray(row[k]) for k in ['image0','image1']],valid=read('valid_masks'),static=static,ship_masks=read('ship_masks'),
        ignore=read('ignore_masks'),visibility=read('visibility_masks'),safe_static=read('safe_static_masks'),structure_support=read('structure_masks'),calibration_mask=read('calibration_masks'),
        points0=p,points1=q,H_A=np.asarray(row['H_source_to_view'][0]),H_B=np.asarray(row['H_source_to_view'][1]),H_AB=np.asarray(row['H_AB']),
        overlap=min(row['overlap_fractions']),source=dict(dataset=row['source_dataset'],source_id=row['source_id'],source_group=row['known_component'],image_path=row['source_path']),
        source_origin_xy=[0,0],regime=row['profile'],seed=row['geometry_seed'],dynamic_occlusion=bool(row['dynamic_events']),loss_role=row['loss_role'],arm=row['arm'])

def load_inference_images(row):
    return [read_gray(row['image0']),read_gray(row['image1'])]

def tensors(pair,device='cpu'):
    """Only images are network inputs. Masks, H and points are explicit supervision."""
    import torch
    return dict(images=torch.from_numpy(np.stack(pair['images']).copy()).to(device=device,dtype=torch.float32)[:,None]/255.,
        H_AB=torch.as_tensor(pair['H_AB'],device=device,dtype=torch.float64),
        **{key:torch.from_numpy(np.stack(pair[key]).copy()).to(device=device,dtype=torch.bool)[:,None] for key in ['valid','static','ignore','visibility','calibration_mask']},
        points0=torch.as_tensor(pair['points0'],device=device),points1=torch.as_tensor(pair['points1'],device=device))

def make_spec(index,seed,cfg,forced=None):
    rng=np.random.default_rng(seed);u=rng.random()
    profile='none' if u<.04 else 'low' if u<.14 else 'projective' if u<.44 else 'affine'
    b=int(rng.choice(5,p=cfg['geometry']['positive_angle_weights']))
    result=dict(index=index,geometry_seed=int(seed),profile=profile,angle_bin=b,requested_size=int(rng.choice(cfg['geometry']['requested_sizes'])),
        appearance_seeds=[stable_seed(seed,'view0'),stable_seed(seed,'view1')],dynamic_seed=stable_seed(seed,'dynamic'))
    if forced:result.update(forced)
    return result

class OnlinePairs:
    """Stateless scheduling by (source index, epoch, visit); append-only per-worker audit.

    Explicit visits avoid a hidden RNG dependent on DataLoader worker scheduling.
    Use a new log path per run/worker. Validation never uses this class.
    """
    def __init__(self,config_path,sources_path,log_path,arm='V2-R',seed=20261015):
        self.cfg=json.loads(Path(config_path).read_text());self.sources=rows(sources_path)
        assert self.sources and all(r['split']=='train' for r in self.sources),'online TRAIN sources only'
        assert arm in self.cfg['arms'];self.arm=arm;self.seed=seed;self.log_path=Path(log_path)
        # Exclusive creation prevents accidental append across independent experiments.
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        with self.log_path.open('x') as f:f.write(json.dumps(dict(type='run',at=now(),arm=arm,seed=seed,config=str(config_path),config_sha=sha(config_path),sources=str(sources_path),sources_sha=sha(sources_path),code_sha=sha(__file__)))+'\n')
    def __len__(self):return len(self.sources)
    def get(self,index,epoch=0,visit=0):
        r=self.sources[index];seed=stable_seed(self.seed,f'{index}:{epoch}:{visit}')
        spec=make_spec(index,seed,self.cfg);record=dict(type='sample',index=index,epoch=epoch,visit=visit,source_id=r['source_id'],known_component=r['known_component'],source_path=r['image_path'],source_sha=sha(r['image_path']),arm=self.arm,spec=spec,at=now())
        try:
            z=array_pair(r,spec,self.cfg,self.arm)
            record.update(state='usable',H_AB=z['H_AB'].tolist(),appearance=z['appearance'],loss_role=z['loss_role'],points=len(z['points0']),image_sha=[digest_array(x) for x in z['images']],mask_sha={k:[digest_array(x) for x in z[k]] for k in ['valid','static','ignore']})
        except Exception as ex:
            record.update(state='rejected',reason=repr(ex))
            with self.log_path.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
            raise
        with self.log_path.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
        return z

def replay(record,source,cfg):
    assert sha(source['image_path'])==record['source_sha'],'source changed'
    z=array_pair(source,record['spec'],cfg,record['arm'])
    assert [digest_array(x) for x in z['images']]==record['image_sha']
    assert np.array_equal(z['H_AB'],np.asarray(record['H_AB']))
    for k,hashes in record['mask_sha'].items():assert [digest_array(x) for x in z[k]]==hashes
    return z
