"""Numerical checks do not compare SAR intensities (view-specific augmentation differs)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
from .io import read_gray, load_json, save_json, jsonl_read, write_png
from .geometry import overlap_mask, transform_points, relative_matrix


def validate_pair(pair_path, tolerance=1e-3):
    path=Path(pair_path)
    meta=load_json(path/'meta.json')
    issues=[]
    n=meta['num_views']; sz=meta['output_size']
    valid=[]; dynamic=[]
    for i in range(n):
        for f in (f'view_{i}.png',f'valid_{i}.png',f'ships_{i}.png',f'dynamic_{i}.png'):
            if not (path/f).is_file():
                issues.append(f'Missing: {f}')
            else:
                im=read_gray(path/f)
                if im.shape!=(sz,sz):
                    issues.append(f'Invalid image/mask shape: {f}')
                elif not f.startswith('view_') and not np.all(np.isin(np.unique(im),(0,255))):
                    issues.append(f'Non-binary mask: {f}')
        if (path/f'valid_{i}.png').exists():
            v=read_gray(path/f'valid_{i}.png')
            if v.shape!=(sz,sz): issues.append(f'Invalid mask shape: view {i}')
            valid.append(v)
        if (path/f'dynamic_{i}.png').exists():
            dynamic.append(read_gray(path/f'dynamic_{i}.png'))
    if len(valid)!=n:
        return {'path':str(path),'passed':False,'issues':issues}
    if 'sea_masks' in meta.get('files', {}):
        files = meta['files']['sea_masks']
        if len(files) != n:
            issues.append('Incorrect number of sea/land masks')
        else:
            for i, f in enumerate(files):
                if not (path/f).exists():
                    issues.append(f'Missing sea/land mask {f}')
                    continue
                mask = read_gray(path/f)
                if mask.shape != (sz, sz):
                    issues.append(f'Invalid sea/land mask shape in view {i}')
                elif not np.all(np.isin(np.unique(mask), (0,255))):
                    issues.append(f'Non-binary sea/land mask in view {i}')
                elif np.any((mask > 0) & (valid[i] == 0)):
                    issues.append(f'Sea/land mask outside valid footprint in view {i}')
    srcH=[np.asarray(H,dtype='float64') for H in meta['H_source_to_view']]
    if len(srcH)!=n or any(H.shape!=(3,3) or not np.isfinite(H).all() for H in srcH):
        issues.append('Invalid or non-finite source-to-view matrices')
        return {'path':str(path),'passed':False,'issues':issues}
    for p in meta['pairs']:
        i,j=p['views']
        Hij=np.asarray(p['H_i_to_j'],dtype='float64')
        Hji=np.asarray(p['H_j_to_i'],dtype='float64')
        if any(H.shape!=(3,3) or not np.isfinite(H).all() for H in (Hij,Hji)):
            issues.append(f'Invalid or non-finite relative matrix {i}<->{j}')
            continue
        expected=relative_matrix(srcH[i],srcH[j])
        if np.max(np.abs(Hij/Hij[2,2]-expected))>tolerance:
            issues.append(f'Bad relative matrix {i}->{j}')
        composed=Hji@Hij
        composed/=composed[2,2]
        if np.max(np.abs(composed-np.eye(3)))>tolerance:
            issues.append(f'Bad inverse {i}<->{j}')
        ref_i,ref_j=overlap_mask(valid[i],valid[j],Hij)
        got_i=read_gray(path/p['overlap_mask_i']);got_j=read_gray(path/p['overlap_mask_j'])
        if not np.array_equal(ref_i,got_i) or not np.array_equal(ref_j,got_j):
            issues.append(f'Overlap mask mismatch {i},{j}')
        fraction_i=np.count_nonzero(got_i)/max(1,np.count_nonzero(valid[i]))
        if abs(p['overlap_fraction_i']-fraction_i)>2e-6:
            issues.append(f'Overlap fraction error in view {i}')
        fraction_j=np.count_nonzero(got_j)/max(1,np.count_nonzero(valid[j]))
        if abs(p['overlap_fraction_j']-fraction_j)>2e-6:
            issues.append(f'Overlap fraction error in view {j}')
        with np.load(path/p['matches']) as pairs:
            pi,pj=pairs['points_i'],pairs['points_j']
        if pi.ndim!=2 or pj.ndim!=2 or pi.shape[1:]!=(2,) or pj.shape[1:]!=(2,):
            issues.append(f'Invalid correspondence shape {i}-{j}')
        elif not np.isfinite(pi).all() or not np.isfinite(pj).all():
            issues.append(f'Non-finite correspondences {i}-{j}')
        elif len(pi)!=len(pj):
            issues.append(f'Length mismatch in {i}-{j} correspondences')
        elif len(pi)!=p['match_count']:
            issues.append(f'Incorrect match count {i}-{j}')
        elif len(pi):
            err=np.linalg.norm(transform_points(Hij,pi)-pj,axis=1)
            if max(err)>tolerance:
                issues.append(f'Correspondence reprojection max error {max(err):.6f}px')
            for idx,xx,over in ((i,pi,got_i),(j,pj,got_j)):
                coords=np.rint(xx).astype(int)
                if np.any(coords<0) or np.any(coords>=sz):
                    issues.append(f'Match outside {idx}')
                else:
                    x,y=coords.T
                    if np.any(valid[idx][y,x]==0) or np.any(over[y,x]==0):
                        issues.append(f'Match outside valid overlap in view {idx}')
                    if len(dynamic)==n and np.any(dynamic[idx][y,x]>0):
                        issues.append(f'Match in dynamic region in view {idx}')
        for k in (i,j):
            for ship in meta['ship_annotations'][k]:
                x,y,w,h=ship['bbox_xywh']
                if x<0 or y<0 or x+w>sz or y+h>sz or w<=0 or h<=0:
                    issues.append(f'Invalid ship bbox view {k}')
    return {'path':str(path),'passed':not issues,'issues':issues,'n_views':n,
            'match_total':sum(p['match_count'] for p in meta['pairs'])}


def validate_generated(root, max_pairs=None):
    root=Path(root)
    metas=sorted(root.glob('*/pair_*/meta.json'))
    if max_pairs:metas=metas[:max_pairs]
    if not metas:
        raise ValueError(f'No pair_*/meta.json found under {root}')
    results=[]
    for p in metas:
        try:
            results.append(validate_pair(p.parent))
        except (OSError,ValueError,KeyError,TypeError,IndexError,EOFError,cv2.error,np.linalg.LinAlgError) as exc:
            results.append({'path':str(p.parent),'passed':False,
                            'issues':[f'{type(exc).__name__}: {exc}']})
    report={'checked':len(results),'passed':sum(x['passed'] for x in results),
        'failed':sum(not x['passed'] for x in results),
        'errors':[{'pair':x['path'],'issues':x['issues']} for x in results if not x['passed']][:100]}
    save_json(root/'validation_report.json',report)
    return report


def visualize_pair(path,out):
    path=Path(path);meta=load_json(path/'meta.json')
    if not meta['pairs']:
        raise ValueError('No view pairs')
    p=meta['pairs'][0]
    i,j=p['views']; sz=meta['output_size']
    a=read_gray(path/f'view_{i}.png'); b=read_gray(path/f'view_{j}.png')
    Hji=np.asarray(p['H_j_to_i'])
    bw=cv2.warpPerspective(b,Hji,(sz,sz))
    va=read_gray(path/f'valid_{i}.png')>0
    ov=read_gray(path/p['overlap_mask_i'])>0
    # tri-color overlay has meaningful colors only in overlap. red=ref, cyan=warped secondary.
    overlay=np.stack([a,a,a],axis=2)
    # OpenCV writes BGR: reference in red, aligned secondary in cyan.
    overlay[ov,2]=a[ov];overlay[ov,0]=bw[ov];overlay[ov,1]=bw[ov]
    rows=[np.stack([a,a,a],axis=-1),np.stack([b,b,b],axis=-1),overlay]
    annot=meta.get('ship_annotations',[[],[]])
    for k,view in zip((i,j),(0,1)):
        for ship in annot[k]:
            x,y,w,h=map(int,ship['bbox_xywh'])
            cv2.rectangle(rows[view],(x,y),(x+w,y+h),(0,240,200),1)
    with np.load(path/p['matches']) as matches:
        pi,pj=matches['points_i'],matches['points_j']
    # show a few matches in coordinate-aligned overlay (projected points in view i).
    for x,y in pi[:40]:
        cv2.circle(overlay,(int(round(x)),int(round(y))),2,(255,255,0),-1)
    def panel(im, title):
        if im.ndim==2:
            im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
        caption=np.zeros((28,sz,3),dtype='uint8')
        cv2.putText(caption,title,(7,19),cv2.FONT_HERSHEY_SIMPLEX,.43,(240,240,240),1,cv2.LINE_AA)
        return np.concatenate([caption,im],axis=0)
    top=np.concatenate([panel(im,title) for im,title in zip(rows,
                        ('View 0 + ship boxes','View 1 + ship boxes','GT alignment: red ref / cyan other'))],axis=1)
    oi=read_gray(path/p['overlap_mask_i']);oj=read_gray(path/p['overlap_mask_j'])
    bottom=np.concatenate([panel(read_gray(path/f'valid_{i}.png'),'Valid 0: white = real source'),
                           panel(oi,'Overlap in view 0'),panel(oj,'Overlap in view 1')],axis=1)
    display=np.concatenate([top,bottom],axis=0)
    if 'sea_masks' in meta.get('files',{}):
        def sea_panel(k):
            valid=read_gray(path/f'valid_{k}.png')>0
            sea=read_gray(path/f'sea_{k}.png')>0
            colored=np.zeros((sz,sz,3),dtype='uint8')
            colored[valid & sea]=(170,100,25)  # blue sea
            colored[valid & ~sea]=(40,150,70)  # green land
            return panel(colored,f'Sea/land {k}: blue sea, green land')
        maskrow=np.concatenate([sea_panel(i),sea_panel(j),
                                panel(read_gray(path/f'valid_{j}.png'),'Valid 1: black = padding')],axis=1)
        display=np.concatenate([display,maskrow],axis=0)
    write_png(out,display)
    return out


def visualize_generated(root,out,count=8,seed=42):
    root=Path(root)
    mets=sorted(root.glob('*/pair_*/meta.json'))
    if not mets:
        raise ValueError('No generated pairs found')
    rng=np.random.default_rng(seed)
    take=rng.choice(len(mets),size=min(count,len(mets)),replace=False)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    paths=[]
    for i,idx in enumerate(take):
        p=out/f'preview_{i:03d}.png'
        visualize_pair(mets[int(idx)].parent,p)
        paths.append(str(p))
    return paths
