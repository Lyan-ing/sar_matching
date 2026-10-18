from common import *
from functools import lru_cache
from sarreg.synthesis import _apply_augment

CFG22=json.loads((OLD/'CONFIG_V22_20261015_1404.json').read_text())
SOURCES={r['image_path']:r for r in raw_sources()}

@lru_cache(maxsize=24)
def tile(p):
    r=SOURCES[p];im=read_gray(p);f=features(r)
    return im,source_valid(im),f,ground_points(r,f,CFG22)

@lru_cache(maxsize=4)
def material(serialized):
    d=json.loads(serialized);x0,y0,x1,y1=d['bounds_xyxy'];shape=(y1-y0,x1-x0)
    im=np.zeros(shape,np.uint8);valid=np.zeros(shape,bool);safe=valid.copy();ship=valid.copy();cal=valid.copy();owner=np.full(shape,-1,np.int16);conflict=valid.copy();points=[]
    # Deterministic ownership. Existing pixels are never averaged, resized or inpainted.
    for k,p in enumerate(d['members']):
        a,v,f,pp=tile(p);r=SOURCES[p];c=coords(r);xx,yy=(c[0]-x0,c[1]-y0) if c else (0,0)
        sl=np.s_[yy:yy+a.shape[0],xx:xx+a.shape[1]];occupied=owner[sl]>=0
        conflict[sl]|=occupied&(im[sl]!=a)
        take=~occupied;im[sl][take]=a[take];valid[sl][take]=v[take];safe[sl][take]=f['source_safe'][take];cal[sl][take]=f['calibration'][take];owner[sl][take]=k
        ship[sl]|=f['ship_union'];points.extend((pp+np.array([xx,yy])).tolist())
    valid &= ~dilate(conflict,2)
    seam=np.zeros(shape,bool);seam[:,1:]|=owner[:,1:]!=owner[:,:-1];seam[1:]|=owner[1:]!=owner[:-1]
    seam=dilate(seam,5)&valid
    safe &= erode(valid,5)&~dilate(ship,10)&~seam;cal &= safe
    pp=np.asarray(points,np.float32).reshape(-1,2)
    if len(pp):
        pp=pp[bilinear_at(safe,pp)]
        # Same source corners can be present in several overlapping publisher tiles.
        pp=np.unique(np.round(pp,3),axis=0)
        keep=[]
        for p in pp:
            if not keep or np.min(np.linalg.norm(np.asarray(keep)-p,axis=1))>=8:keep.append(p)
        pp=np.asarray(keep,np.float32).reshape(-1,2)
    return dict(image=im,valid=valid,safe=safe,ship=ship,calibration=cal,seam=seam,owner=owner,points=pp,conflict_pixels=int(conflict.sum()))

def domain_arrays(d):return material(json.dumps(d,sort_keys=True))

def footprint(H,n):
    p=np.array([[-.5,-.5],[n-.5,-.5],[n-.5,n-.5],[-.5,n-.5]],np.float64)
    return transform_points(np.linalg.inv(H),p).astype('float32')

def polygon_ratio(p,q):
    p,q=p.astype('float32'),q.astype('float32')
    area,_=cv2.intersectConvexConvex(p,q)
    return max(0.,float(area))/max(cv2.contourArea(p),cv2.contourArea(q),1e-9)

def sample_geometry(d,spec,cfg,force_size=None,probe=False):
    """Reuse original view_matrix/warp/overlap semantics, changing only the sampler.

    A convex-footprint bisection proposes the requested overlap; final acceptance
    always uses actual raster validity and the original bidirectional mask code.
    """
    rng=np.random.default_rng(spec['geometry_seed']);data=domain_arrays(d);h,w=data['image'].shape
    bi=OVERLAPS.index(spec['overlap_bin']);lo,hi=BOUNDS[bi];ab=spec['angle_bin'];al,au=ANGLES[ab:ab+2]
    rejections=Counter();fallback=None
    for n in ([force_size] if force_size else [640,512]):
        before=sum(rejections.values())
        if min(h,w)<n+4:
            rejections[f'{n}:native_dimensions']+=1
            if n==640:fallback='native_domain_dimensions_below_640_bilinear_support'
            continue
        for trial in range(cfg['geometry']['attempts_per_size']):
            theta=float(rng.uniform(al+.10,min(au-.10,179.9)))*rng.choice([-1,1])
            # Squared fields cannot have >90% mutual overlap at arbitrary rotation.
            # The rejection is retained; high-overlap feasible angles are not faked.
            orient=-theta/2+rng.choice([-90,0,90,180])+rng.uniform(-1,1)
            kind=spec['geometry_kind']
            scales=np.ones(2) if kind=='rigid' else rng.uniform(.96,1.,2)
            shear=[0.,0.] if kind!='projective' else rng.uniform(-1.,1.,2)
            persp=cfg['geometry']['perspective_fraction'] if kind=='projective' else 0.
            Hs=[view_matrix((0,0),n,a,s,sh,persp,rng) for a,s,sh in zip([orient,orient+theta],scales,shear)]
            p,q=[footprint(H,n) for H in Hs];target=0 if bi==0 else float(rng.uniform(lo+.008,min(hi-.008,.985)))
            if bi and polygon_ratio(p,q)<target:
                rejections[f'{n}:rotation_overlap_incompatible']+=1;continue
            phi=rng.uniform(0,2*np.pi);direction=np.array([np.cos(phi),np.sin(phi)])
            left,right=0.,n*2.5
            for _ in range(22):
                mid=(left+right)/2
                if polygon_ratio(p,q+direction*mid)>target:left=mid
                else:right=mid
            delta=direction*(right+(4. if bi==0 else 0.))
            allpts=np.vstack([p,q+delta]);mn=allpts.min(0);mx=allpts.max(0)
            low=2.-mn;high=np.array([w-3.,h-3.])-mx
            if np.any(high<low):rejections[f'{n}:native_joint_field_insufficient']+=1;continue
            center=rng.uniform(low,high)
            centers=[center,center+delta]
            local=[H@np.array([[1.,0.,-c[0]],[0.,1.,-c[1]],[0.,0.,1.]]) for H,c in zip(Hs,centers)]
            H=local[1]@np.linalg.inv(local[0]);H/=H[2,2]
            angle=angle_of(H,n)
            if angle_bin(angle)!=ab:rejections[f'{n}:actual_angle_bin']+=1;continue
            # Small proxy is only a cheap rejection; a full raster is mandatory below.
            proxy=np.array([[95/(n-1),0,0],[0,95/(n-1),0],[0,0,1.]])
            vv=[cv2.warpPerspective(data['valid'].astype('uint8'),proxy@A,(96,96),flags=cv2.INTER_NEAREST)>0 for A in local]
            if min(v.mean() for v in vv)<.98:rejections[f'{n}:source_gaps_proxy']+=1;continue
            if probe:
                return dict(size=n,trial=trial,rejected=dict(rejections),angle=angle,estimated_overlap=target)
            valid=[erode(cv2.warpPerspective(data['valid'].astype('float32'),A,(n,n),flags=cv2.INTER_LINEAR)>1-1e-6,1) for A in local]
            if min(v.mean() for v in valid)<cfg['geometry']['min_valid_fraction']:
                rejections[f'{n}:full_bilinear_support']+=1;continue
            ov=[o>0 for o in overlap_mask(*(v.astype('uint8')*255 for v in valid),H)]
            fr=[ratio(o,v) for o,v in zip(ov,valid)]
            if (bi==0 and max(fr)!=0) or (bi!=0 and overlap_bin(min(fr))!=spec['overlap_bin']):
                rejections[f'{n}:actual_overlap_bin']+=1;continue
            polys=[footprint(A,n) for A in local];intersection=float(cv2.intersectConvexConvex(*polys)[0])
            if bi==0 and intersection>1e-4:rejections[f'{n}:nonempty_source_intersection']+=1;continue
            O=np.array([[1.,0.,-d['bounds_xyxy'][0]],[0.,1.,-d['bounds_xyxy'][1]],[0.,0.,1.]])
            return dict(size=n,local=local,T=[A@O for A in local],H=H,valid=valid,overlap=ov,fractions=fr,angle=angle,
                rejected=dict(rejections),trial=trial,geometry_kind=kind,nominal_theta=theta,scales=scales.tolist(),shear=list(map(float,shear)),
                polygon_common_source_area=intersection,reason_if_512=fallback if n==512 else None)
        if n==640:fallback='640_finite_native_joint_field_or_validity_rejections:'+json.dumps({k:v for k,v in rejections.items() if k.startswith('640:')},sort_keys=True)
    raise ValueError('geometry:'+json.dumps(dict(rejections),sort_keys=True))

def appearance(im,valid,profile,seed,cfg,side=0):
    """Optional composition of the existing display-DN mechanisms, not raw SAR simulation."""
    rng=np.random.default_rng(seed);effective=profile
    if profile=='P4':effective='P0' if side==cfg.get('asymmetric_clean_side',0) else 'P'+str(int(rng.choice([2,3])))
    changed=np.zeros_like(valid);ignore=changed.copy();ops=[];params=dict(seed=int(seed),domain='published_display_DN',effective_profile=effective,ordered_ops=ops,skipped=[])
    if effective=='P0':return im.copy(),ignore,changed,params
    moderate=effective=='P3';ac=cfg['appearance'];radiation=ac['radiometry_moderate' if moderate else 'radiometry_light']
    # The exact executed V1 helper is reused, without its mandatory noise or blur.
    x,rp=_apply_augment(im,dict(enabled=True,gamma_min=radiation['gamma'][0],gamma_max=radiation['gamma'][1],
        contrast_min=radiation['contrast'][0],contrast_max=radiation['contrast'][1],brightness_max=radiation['offset'],speckle_sigma=0,blur_probability=0),rng)
    x=x.astype('float32')/255.;ops.append('radiometry');params['radiometry']=rp
    candidates=['fine_texture','correlated_field','multiplicative_speckle','directional_blur','local_scatter','smooth_clutter']
    count=0 if effective=='P1' else int(rng.integers(2,4) if moderate else rng.integers(1,3))
    selected=list(rng.choice(candidates,size=count,replace=False)) if count else []
    params['skipped']=[op for op in candidates if op not in selected]
    for op in selected:
        if op=='fine_texture':
            sigma=float(rng.uniform(.65,1.1));alpha=float(rng.uniform(.12,.30) if moderate else rng.uniform(.04,.16))
            smooth=cv2.GaussianBlur(x,(0,0),sigma);coarse=cv2.GaussianBlur(x,(0,0),2)
            gx=cv2.Sobel(coarse,-1,1,0,ksize=3,scale=.125);gy=cv2.Sobel(coarse,-1,0,1,ksize=3,scale=.125)
            weight=np.clip(1-np.hypot(gx,gy)/.035,.25,1.)
            x-=alpha*weight*(x-smooth);params[op]=dict(lowpass_sigma=sigma,residual_attenuation=alpha,edge_gradient=.035)
        elif op=='correlated_field':
            shape=float(rng.uniform(8,24));sigma=float(rng.uniform(.008,.018) if moderate else rng.uniform(.003,.010));rho=float(rng.uniform(.35,.8));mix=float(rng.uniform(.15,.55))
            field=(rng.gamma(shape,1/shape,x.shape)-1)*np.sqrt(shape);blur=cv2.GaussianBlur(field.astype('float32'),(0,0),rho);blur=(blur-blur.mean())/max(float(blur.std()),1e-6)
            field=(1-mix)*field+mix*blur;field=(field-field.mean())/max(float(field.std()),1e-6)
            x+=sigma*(.25+.75*np.sqrt(np.clip(x,0,1)))*field
            params[op]=dict(gamma_shape_proxy_not_ENL=shape,sigma=sigma,correlation_sigma=rho,mix=mix,zero_mean_unit_std=True)
        elif op=='multiplicative_speckle':
            sigma=float(rng.uniform(.06,.12) if moderate else rng.uniform(.02,.06));s=np.sqrt(np.log1p(sigma*sigma))
            x*=rng.lognormal(-.5*s*s,s,size=x.shape);params[op]=dict(coefficient_of_variation=sigma,unit_mean_in_expectation=True)
        elif op=='directional_blur':
            sx,sy=map(float,rng.uniform(.2,.65 if moderate else .45,2));x=cv2.sepFilter2D(x,-1,cv2.getGaussianKernel(5,sx),cv2.getGaussianKernel(5,sy));params[op]=dict(kernel_size=5,sigma_xy=[sx,sy],centered_kernel=True)
        elif op=='smooth_clutter':
            sigma=float(rng.uniform(.005,.012) if moderate else rng.uniform(.002,.006));rho=max(8,x.shape[0]/16)
            f=cv2.GaussianBlur(rng.normal(size=x.shape).astype('float32'),(0,0),rho);f-=f.mean();f/=max(float(f.std()),1e-6);x+=np.clip(f*sigma,-.025,.025);params[op]=dict(sigma=sigma,correlation_sigma=rho,clip=.025)
        elif op=='local_scatter':
            yy,xx=np.indices(x.shape);bright=(x>max(.25,float(np.quantile(x[valid],.90))))&erode(valid,20);ys,xs=np.where(bright);events=[]
            for _ in range(int(rng.integers(1,4) if moderate else rng.integers(1,3)) if len(xs) else 0):
                k=int(rng.integers(len(xs)));cx,cy=int(xs[k]),int(ys[k]);rad=float(rng.uniform(3,6));gain=float(rng.uniform(.65,1.25) if moderate else rng.uniform(.85,1.15))
                blob=np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*rad*rad)).astype('float32');region=(blob>.05)&valid
                if (changed|region).mean()>.025:continue
                x*=1+(gain-1)*blob;changed|=region;ignore|=region
                events.append(dict(center_xy=[cx,cy],sigma=rad,gain=gain,pixels=int(region.sum())))
            params[op]=dict(events=events,max_fraction=.025)
        ops.append(op)
    params['preclip_low_fraction']=float((x[valid]<0).mean());params['preclip_high_fraction']=float((x[valid]>1).mean())
    x=np.round(np.clip(x,0,1)*255).astype('uint8');x[~valid]=0
    return x,dilate(ignore,5)&valid,changed,params

def grid_checks(valid,over,ignore,H,n):
    step=max(12,n//32);yy,xx=np.mgrid[step//2:n:step,step//2:n:step];p=np.c_[xx.ravel(),yy.ravel()].astype('float64');q=transform_points(H,p)
    k=bilinear_at(valid[0]&over[0]&~ignore[0],p)&bilinear_at(valid[1]&over[1]&~ignore[1],q)
    return p[k],q[k]

def generate(d,spec,cfg):
    source=domain_arrays(d);g=sample_geometry(d,spec,cfg,force_size=spec.get('validation_force_size'));n=g['size'];local=g['local'];H=g['H'];valid=g['valid']
    native=[warp(source['image'],A,n) for A in local]
    for a,v in zip(native,valid):a[~v]=0
    ship=[warp(source['ship'].astype('uint8'),A,n,mask=True)>0 for A in local]
    seam=[warp(source['seam'].astype('uint8'),A,n,mask=True)>0 for A in local]
    clean_side=int(stable_seed(spec['seed'],'clean_side')%2);acfg=dict(cfg,asymmetric_clean_side=clean_side)
    output=[appearance(a,v,spec['enhancement_profile'],stable_seed(spec['seed'],f'appearance{i}'),acfg,i) for i,(a,v) in enumerate(zip(native,valid))]
    images=[o[0] for o in output];rawignore=[(o[1]|s)&v for o,s,v in zip(output,seam,valid)]
    # Uncertain changes on either side remove BOTH corresponding locations.
    projected=[warp(rawignore[1].astype('uint8'),np.linalg.inv(H),n,mask=True)>0,warp(rawignore[0].astype('uint8'),H,n,mask=True)>0]
    ignore=[(a|b|dilate(s,5))&v for a,b,s,v in zip(rawignore,projected,ship,valid)]
    safe=[(warp(source['safe'].astype('uint8'),A,n,mask=True)>0)&v&~ig for A,v,ig in zip(local,valid,ignore)]
    pp=[transform_points(A,source['points']) for A in local];k=bilinear_at(safe[0],pp[0])&bilinear_at(safe[1],pp[1])
    p,q=thin(pp[0][k].astype('float64'),pp[1][k].astype('float64'))
    if spec['overlap_bin']=='zero':p=np.empty((0,2));q=p.copy()
    static=[];calibration=[]
    for pts,s,A,v,ig in zip([p,q],safe,local,valid,ignore):
        m=np.zeros((n,n),np.uint8)
        for point in pts:cv2.circle(m,tuple(np.rint(point).astype(int)),8,1,-1)
        static.append((m>0)&s)
        calibration.append((warp(source['calibration'].astype('uint8'),A,n,mask=True)>0)&v&~ig&~dilate(m,4))
    k=bilinear_at(static[0],p)&bilinear_at(static[1],q);p,q=p[k],q[k]
    checks=grid_checks(valid,g['overlap'],ignore,H,n)
    no=spec['overlap_bin']=='zero';level='no_overlap' if no else 'sufficient_structure' if global_ok(p,q,n) else 'limited_structure' if len(p) else 'weak_or_unknown_structure'
    intensity=[]
    for a,v,b in zip(images,valid,native):
        z=a[v];std=float(z.std());sat=float((z==255).mean());zero=float((z==0).mean())
        if std<cfg['acceptance']['min_std'] or sat>cfg['acceptance']['max_saturated'] or zero>cfg['acceptance']['max_zero']:raise ValueError(f'appearance_QA:{std},{sat},{zero}')
        ga=cv2.GaussianBlur(a.astype('float32')/255,(0,0),2);gb=cv2.GaussianBlur(b.astype('float32')/255,(0,0),2)
        intensity.append(dict(std=std,saturated_fraction=sat,zero_fraction=zero,coarse_RMSE=float(np.sqrt(np.mean((ga[v]-gb[v])**2)))))
    return dict(g=g,images=images,native=native,ignore=ignore,appearance_masks=[o[2] for o in output],appearance=[o[3] for o in output],
        ship=ship,safe=safe,static=static,calibration=calibration,unknown=[v&~s&~c for v,s,c in zip(valid,static,calibration)],
        points=[p,q],checks=checks,level=level,intensity=intensity,source=source)
