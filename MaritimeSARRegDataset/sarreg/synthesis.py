from __future__ import annotations
from pathlib import Path
import hashlib
import json
from itertools import combinations
import cv2
import numpy as np
import yaml
from .geometry import relative_matrix, overlap_mask, ratio, sample_views, shared_points, warp
from .io import read_gray, write_png, save_json, jsonl_read, jsonl_write
from .labels import instance_views


def load_cfg(path):
    with Path(path).open('r',encoding='utf8') as f:
        c=yaml.safe_load(f)
    if not isinstance(c,dict):
        raise ValueError('Config must be YAML object')
    if not (0<=float(c['min_overlap'])<=float(c['max_overlap'])<=1):
        raise ValueError('Invalid overlap limits')
    if int(c['output_size'])<64:
        raise ValueError('output_size must be >=64')
    return c


def stable_seed(seed, string):
    return int.from_bytes(hashlib.sha256(f'{seed}:{string}'.encode()).digest()[:8],'big')%(2**32)


def _read_and_resize(row,cfg):
    src=read_gray(row['image_path'])
    H,W=src.shape
    required=float(cfg.get('min_source_side_factor',1.35))*float(cfg['output_size'])
    scale=max(1.0,required/max(1,min(H,W))) if cfg.get('allow_source_upsampling',True) else 1.
    if scale>1:
        src=cv2.resize(src,(round(W*scale),round(H*scale)),interpolation=cv2.INTER_CUBIC)
        # Scale is computed from integer image dimensions below for exact mask resizing.
    return src, (src.shape[1]/W,src.shape[0]/H)


def _apply_augment(img, cfg, rng):
    if not cfg or not cfg.get('enabled',True):
        return img.copy(), {'enabled':False}
    x=img.astype('float32')/255.
    gamma=float(rng.uniform(cfg.get('gamma_min',1),cfg.get('gamma_max',1)))
    contrast=float(rng.uniform(cfg.get('contrast_min',1),cfg.get('contrast_max',1)))
    offset=float(rng.uniform(-cfg.get('brightness_max',0),cfg.get('brightness_max',0)))
    x=np.maximum(x,0)**gamma
    x=(x-.5)*contrast+.5+offset
    sigma=float(cfg.get('speckle_sigma',0))
    if sigma>0:
        # Independent multiplicative synthetic noise per view, NOT full SAR raw-data simulation.
        x=x*rng.lognormal(mean=-.5*np.log(1+sigma*sigma),sigma=np.sqrt(np.log(1+sigma*sigma)),size=x.shape)
    x=np.clip(x,0,1)
    blur_sigma=0.
    if rng.random()<cfg.get('blur_probability',0):
        blur_sigma=float(rng.uniform(.1,max(.11,cfg.get('blur_sigma_max',.8))))
        x=cv2.GaussianBlur(x.astype('float32'),(0,0),blur_sigma)
    return (255*np.clip(x,0,1)).astype('uint8'),{
        'enabled':True,'gamma':round(gamma,4),'contrast':round(contrast,4),
        'brightness':round(offset,4),'speckle_sigma':sigma,'blur_sigma':round(blur_sigma,4)}


def _dynamic_move(img, entries, config, rng, valid=None, events=None):
    """Approximate ship shift for robustness tests. Background replacement is synthetic.
    Changes are excluded from static correspondence GT.
    """
    h,w=img.shape
    changed=np.zeros_like(img,dtype='uint8')
    if not config.get('enabled',False):
        return img,entries,changed
    out=img.copy()
    shifts=[]
    for entry,m in entries:
        if rng.random()>config.get('move_probability',.35) or np.count_nonzero(m)<config.get('min_area',12):
            shifts.append((entry,m))
            continue
        maxshift=int(config.get('max_shift_pixels',12))
        dx=int(rng.integers(-maxshift,maxshift+1));dy=int(rng.integers(-maxshift,maxshift+1))
        if dx==0 and dy==0:
            shifts.append((entry,m));continue
        yy,xx=np.where(m>0)
        if len(xx)==0:
            continue
        x0,x1=max(0,int(xx.min()-2)),min(w,int(xx.max()+3))
        y0,y1=max(0,int(yy.min()-2)),min(h,int(yy.max()+3))
        patch=out[y0:y1,x0:x1].copy()
        patchmask=m[y0:y1,x0:x1]
        # Remove origin with local inpainting (not physically accurate SAR reconstruction).
        layer=np.zeros_like(img)
        layermask=np.zeros_like(m)
        layer[y0:y1,x0:x1]=np.where(patchmask>0,patch,0)
        layermask[y0:y1,x0:x1]=patchmask
        movedm=cv2.warpAffine(layermask,np.float32([[1,0,dx],[0,1,dy]]),(w,h),flags=cv2.INTER_NEAREST)
        if valid is not None:
            movedm[valid==0]=0
        other=np.zeros_like(m)
        for other_entry,other_mask in entries:
            if other_entry['instance_id']!=entry['instance_id']:
                other=cv2.bitwise_or(other,other_mask)
        for previous_entry,previous_mask in shifts:
            if previous_entry['instance_id']!=entry['instance_id']:
                other=cv2.bitwise_or(other,previous_mask)
        if np.any((other>0)&(movedm>0)):
            shifts.append((entry,m));continue
        out=cv2.inpaint(out,m,3,cv2.INPAINT_TELEA)
        movedv=cv2.warpAffine(layer,np.float32([[1,0,dx],[0,1,dy]]),(w,h),flags=cv2.INTER_LINEAR)
        out[movedm>0]=movedv[movedm>0]
        changed=cv2.bitwise_or(changed,cv2.bitwise_or(m,movedm))
        if events is not None:
            events.append({'instance_id':entry['instance_id'],'shift_xy':[dx,dy],
                           'origin_area_pixels':int(np.count_nonzero(m)),
                           'destination_area_pixels':int(np.count_nonzero(movedm))})
        yy2,xx2=np.where(movedm>0)
        if len(xx2):
            upd=dict(entry)
            upd['bbox_xywh']=[int(xx2.min()),int(yy2.min()),int(xx2.max()+1-xx2.min()),int(yy2.max()+1-yy2.min())]
            upd['area_pixels']=int(len(xx2)); upd['synthetic_movement_xy']=[dx,dy]
            upd['movement_retained_fraction']=round(len(xx2)/max(1,np.count_nonzero(m)),6)
            upd['visible_fraction_estimate']=round(entry['visible_fraction_estimate']*min(1.,upd['movement_retained_fraction']),5)
            shifts.append((upd,movedm))
    return out,shifts,changed


def synthesize(row, cfg, output_path, seed=42):
    output_path=Path(output_path)
    rng=np.random.default_rng(seed)
    requested_size=int(cfg['output_size'])
    if not cfg.get('allow_source_upsampling',True):
        cfg=dict(cfg)
        native=read_gray(row['image_path'])
        multiple=int(cfg.get('output_size_multiple',32))
        cfg['output_size']=min(requested_size,int(min(native.shape)/float(cfg.get('min_source_side_factor',1.35))/multiple)*multiple)
        if cfg['output_size']<64:
            raise ValueError('Native source too small for the configured crop headroom (minimum output 64)')
    src,scales=_read_and_resize(row,cfg)
    # labels expects isotropic scaling; actual axes can differ slightly due to int rounding.
    size=int(cfg['output_size'])
    mode=cfg.get('center_mode','default')
    if mode!='default':
        cfg=dict(cfg)
        margin=size/(2*float(cfg['scale_min']))*1.1
        if mode=='ship':
            anns=row.get('annotations',[])
            if not anns:raise ValueError('Ship-centered crop requested without source ship labels')
            x,y,w,h=anns[int(rng.integers(len(anns)))]['bbox_xywh']
            cx,cy=(x+w/2)*scales[0],(y+h/2)*scales[1]
        elif mode=='uniform':
            cx,cy=rng.uniform(0,src.shape[1]),rng.uniform(0,src.shape[0])
        else:raise ValueError(f'Unknown center_mode {mode}')
        cfg['reference_center']=[float(np.clip(cx,min(margin,src.shape[1]/2),max(src.shape[1]-margin,src.shape[1]/2))),
                                 float(np.clip(cy,min(margin,src.shape[0]/2),max(src.shape[0]-margin,src.shape[0]/2)))]
    Hs,valid=sample_views(src.shape,cfg,rng)
    annotations=row.get('annotations',[])
    # Rather than rasterize COCO annotations on resized image, rasterize original then resize:
    # instance_views needs original shape, output affine H coordinates are resized-source coords.
    old_image=read_gray(row['image_path'])
    ship_rows,ship_masks,instance_lists=instance_views(old_image.shape,annotations,Hs,size,src.shape,
         min_visibility=cfg.get('min_ship_visible_fraction',.35), min_pixels=cfg.get('min_ship_pixels',8))
    imgs=[]; dynamic_masks=[]; augment_params=[]; dynamic_events=[[] for _ in Hs]
    for i,H in enumerate(Hs):
        im=warp(src,H,size)
        im,params=_apply_augment(im,cfg.get('augmentation',{}),rng)
        im=np.where(valid[i]>0,im,0).astype('uint8')
        if i>0:
            im,instance_lists[i],dyn=_dynamic_move(im,instance_lists[i],cfg.get('dynamic_ships',{}),rng,valid[i],dynamic_events[i])
            if cfg.get('dynamic_ships',{}).get('enabled',False):
                dyn=cv2.dilate(dyn,np.ones((3,3),np.uint8))
                dyn[valid[i]==0]=0
        else:
            dyn=np.zeros_like(im)
        ship_rows[i]=[e for e,_ in instance_lists[i]]
        ship_masks[i]=np.zeros_like(im)
        for e,m in instance_lists[i]:
            ship_masks[i]=cv2.bitwise_or(ship_masks[i],m)
        imgs.append(im);dynamic_masks.append(dyn);augment_params.append(params)
    meta={'format_version':'1.0','kind':'synthetic_2d_registration_proxy_not_physical_multiview',
          'registration_type':'synthetic_2d_registration_proxy',
          'dataset':row['dataset'],'source_id':row['source_id'],'source_group':row['source_group'],
          'source_image_path':row['image_path'],'source_split':row['split'],
          'official_split':row.get('official_split'),'scene_type':row.get('scene_type','unknown'),
          'annotation_source':row.get('annotation_source'),
          'split_protocol':row.get('split_protocol','prepared_index_split'),
          'source_shape_hw':list(old_image.shape),'resized_source_shape_hw':list(src.shape),
          'source_resize_factors_xy':list(scales),'num_views':len(Hs),'output_size':size,
          'requested_output_size':requested_size,'scale_policy':'native_adaptive' if not cfg.get('allow_source_upsampling',True) else 'legacy_resize',
          'center_mode':mode,'dynamic_events':dynamic_events,
          'dynamic_exclusion_guard_pixels':1 if cfg.get('dynamic_ships',{}).get('enabled',False) else 0,
          'seed':int(seed), 'H_source_to_view':[H.tolist() for H in Hs],
          'augmentations':augment_params,'pairs':[],
          'annotation_note':'COCO polygons used when available; VOC boxes are box-shaped proxies, not real instance masks.',
          'ship_annotations':ship_rows,'files':{}}
    for i in range(len(Hs)):
        write_png(output_path/f'view_{i}.png',imgs[i])
        write_png(output_path/f'valid_{i}.png',valid[i])
        write_png(output_path/f'ships_{i}.png',ship_masks[i])
        write_png(output_path/f'dynamic_{i}.png',dynamic_masks[i])
    if row.get('sea_mask_path'):
        sea=read_gray(row['sea_mask_path'])
        # SL-SSDD's published coding: 0=land, 255=sea; no unknown classes.
        u=np.unique(sea)
        if not np.all(np.isin(u, (0,255))):
            raise ValueError(f'SL-SSDD mask is not binary 0/255: {row["sea_mask_path"]}, values={u[:16].tolist()}')
        if sea.shape!=old_image.shape:
            # fail visibly rather than generating subtly wrong masks
            raise ValueError(f'Sea mask shape mismatch: {row["sea_mask_path"]} {sea.shape} vs {old_image.shape}')
        if sea.shape!=src.shape:
            sea=cv2.resize(sea,(src.shape[1],src.shape[0]),interpolation=cv2.INTER_NEAREST)
        for i,H in enumerate(Hs):
            seaout=warp(np.where(sea>=128,255,0).astype('uint8'),H,size,mask=True)
            seaout[valid[i]==0]=0
            write_png(output_path/f'sea_{i}.png',seaout)
        meta['files']['sea_masks']=[f'sea_{i}.png' for i in range(len(Hs))]
        meta['sea_mask_provenance']={'source_path':row['sea_mask_path'],
            'source_encoding':{'land':0,'sea':255},
            'outside_valid_region':0,
            'interpretation':'Use alongside valid_{i}.png to distinguish land from padding'}
    for i,j in combinations(range(len(Hs)),2):
        Hij=relative_matrix(Hs[i],Hs[j]); Hji=relative_matrix(Hs[j],Hs[i])
        o_i,o_j=overlap_mask(valid[i],valid[j],Hij)
        write_png(output_path/f'overlap_{i}_{j}_in_{i}.png',o_i)
        write_png(output_path/f'overlap_{i}_{j}_in_{j}.png',o_j)
        pi,pj=shared_points(Hij,o_i,o_j,int(cfg.get('correspondences_per_pair',512)),rng,
                            dynamic_masks[i],dynamic_masks[j])
        np.savez_compressed(output_path/f'matches_{i}_{j}.npz',points_i=pi,points_j=pj)
        meta['pairs'].append({'views':[i,j],'H_i_to_j':Hij.tolist(),'H_j_to_i':Hji.tolist(),
            'overlap_fraction_i':round(ratio(o_i,valid[i]),6),
            'overlap_fraction_j':round(ratio(o_j,valid[j]),6),
            'match_count':len(pi),'overlap_mask_i':f'overlap_{i}_{j}_in_{i}.png',
            'overlap_mask_j':f'overlap_{i}_{j}_in_{j}.png',
            'matches':f'matches_{i}_{j}.npz'})
    save_json(output_path/'meta.json',meta)
    return {'folder':str(output_path),'split':row['split'],'dataset':row['dataset'],
            'source_id':row['source_id'],'source_group':row['source_group'],
            'min_overlap':min(min(p['overlap_fraction_i'],p['overlap_fraction_j']) for p in meta['pairs']),
            'ships':sum(len(a) for a in ship_rows),'num_views':len(Hs)}


def generate(index, config, out, split='train', count=100, seed=42, overwrite=False):
    cfg=load_cfg(config)
    rows=jsonl_read(index)
    choices=[r for r in rows if split=='all' or r['split']==split]
    if not choices:
        raise ValueError(f'No indexed image in split={split}. Available splits: {sorted(set(r["split"] for r in rows))}')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    from tqdm import tqdm
    produced=[];failures=[]
    rng=np.random.default_rng(seed)
    # Source images sampled with replacement: same source may yield multiple differently seeded views.
    indices=rng.integers(0,len(choices),size=count)
    existing=[out/choices[int(k)]['split']/f'pair_{idx:06d}' for idx,k in enumerate(indices)
              if (out/choices[int(k)]['split']/f'pair_{idx:06d}').exists()]
    if not overwrite and (existing or (out/f'generation_report_{split}.json').exists()):
        raise FileExistsError(f'Existing generation under {out}: choose a new --out or explicitly --overwrite; '
                              'config, manifest and prior samples preserved')
    run_config={'index':str(Path(index).resolve()),'config':cfg,
                'requested_split':split,'requested_count':count,'global_seed':seed}
    save_json(out/f'generation_config_{split}.json',run_config)
    if not (out/'generation_config.json').exists():
        save_json(out/'generation_config.json',run_config)
    for idx,k in enumerate(tqdm(indices,desc='Generating pairs')):
        r=choices[int(k)]
        folder=out/r['split']/f'pair_{idx:06d}'
        if folder.exists() and (folder/'meta.json').exists() and not overwrite:
            failures.append({'index':idx,'reason':'existing directory: use --overwrite to regenerate'})
            continue
        try:
            smseed=stable_seed(seed,f'{idx}:{r["image_path"]}')
            result=synthesize(r,cfg,folder,smseed)
            produced.append(result)
        except Exception as exc:
            # don't silently claim requested output count was reached
            import shutil
            if folder.exists() and not (folder/'meta.json').exists():
                shutil.rmtree(folder)
            failures.append({'index':idx,'source':r['image_path'],'reason':str(exc)})
    jsonl_write(out/f'manifest_{split}.jsonl',produced)
    save_json(out/f'generation_report_{split}.json',{'requested':count,'generated':len(produced),
        'failures':failures[:200], 'failure_count':len(failures)})
    return {'requested':count,'generated':len(produced),'failed':len(failures),'output':str(out.resolve())}
