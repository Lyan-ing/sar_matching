from __future__ import annotations
import math
import cv2
import numpy as np


def transform_points(H:np.ndarray, points:np.ndarray)->np.ndarray:
    pts=np.asarray(points,dtype=np.float64).reshape(-1,2)
    if len(pts)==0:
        return np.zeros((0,2),dtype=np.float64)
    homog=np.column_stack([pts,np.ones(len(pts))]) @ np.asarray(H,dtype=np.float64).T
    den=homog[:,2]
    den=np.where(abs(den)<1e-12,np.nan,den)
    return homog[:,:2]/den[:,None]


def warp(im,H,size,mask=False):
    return cv2.warpPerspective(im,np.asarray(H,dtype=np.float64),(int(size),int(size)),
                               flags=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def view_matrix(center, out_size, angle, scale, shear=0., persp=0., rng=None):
    """Pixel convention: source pixel coordinates -> view pixel coordinates.
    Projective jitter is relative to the output square and applied AFTER affine.
    """
    cx,cy=center
    mat=cv2.getRotationMatrix2D((float(cx),float(cy)),float(angle),float(scale))
    mat[0,2]+= (out_size-1)*0.5-cx
    mat[1,2]+= (out_size-1)*0.5-cy
    A=np.vstack([mat,[0.,0.,1.]])
    if abs(shear)>1e-8:
        sh=math.tan(math.radians(shear))
        C=(out_size-1)/2
        shearH=np.array([[1.,sh,-sh*C],[0.,1.,0.],[0.,0.,1.]])
        A=shearH@A
    if persp>0:
        if rng is None:
            raise ValueError('rng required for perspective jitter')
        c=float(out_size-1)
        corners=np.float32([[0,0],[c,0],[c,c],[0,c]])
        offsets=rng.uniform(-persp*out_size,persp*out_size,(4,2))
        P=cv2.getPerspectiveTransform(corners,(corners+offsets).astype('float32'))
        A=P@A
    return A / A[2,2]


def relative_matrix(Hsrc_to_i, Hsrc_to_j):
    H=np.asarray(Hsrc_to_j)@np.linalg.inv(Hsrc_to_i)
    return H/H[2,2]


def overlap_mask(valid_i, valid_j, H_i_to_j):
    """Return overlaps in corresponding view-i and view-j coordinate frames."""
    h,w=valid_i.shape
    if h!=w or valid_j.shape!=(h,w):
        raise ValueError('Square, equal-size views required by this generator')
    warped_j_in_i=warp(valid_j,np.linalg.inv(H_i_to_j),w,mask=True)
    warped_i_in_j=warp(valid_i,H_i_to_j,w,mask=True)
    over_i=((valid_i>0)&(warped_j_in_i>0)).astype('uint8')*255
    over_j=((valid_j>0)&(warped_i_in_j>0)).astype('uint8')*255
    return over_i,over_j


def ratio(over,valid):
    return float(np.count_nonzero(over))/max(1,np.count_nonzero(valid))


def shared_points(H_i_to_j, over_i, over_j, n, rng, exclude_i=None, exclude_j=None):
    valid=(over_i>0)
    if exclude_i is not None:
        valid &= (exclude_i==0)
    ys,xs=np.where(valid)
    if len(xs)==0:
        return np.zeros((0,2),dtype='float32'),np.zeros((0,2),dtype='float32')
    # draw more initially as discrete masks and perspective interpolation can disagree at boundaries
    take=min(len(xs),max(10*n,n))
    inds=rng.choice(len(xs),size=take,replace=False)
    pi=np.column_stack([xs[inds],ys[inds]]).astype('float64')
    pj=transform_points(H_i_to_j,pi)
    h,w=over_j.shape
    keep=np.isfinite(pj).all(axis=1)
    x=np.rint(np.clip(pj[:,0],0,w-1)).astype('int32')
    y=np.rint(np.clip(pj[:,1],0,h-1)).astype('int32')
    keep &= (pj[:,0]>=0)&(pj[:,0]<=w-1)&(pj[:,1]>=0)&(pj[:,1]<=h-1)
    keep &= over_j[y,x]>0
    if exclude_j is not None:
        keep &= exclude_j[y,x]==0
    pi,pj=pi[keep][:n],pj[keep][:n]
    return pi.astype('float32'),pj.astype('float32')


def sample_views(src_shape, cfg, rng, views=None):
    """Rejection sample geometries until requested pairwise overlap is feasible.
    Geometric masks are obtained by warping a source-one-mask, so edge and
    partial-cover effects are fully represented. Raises rather than mislabels GT.
    """
    h,w=src_shape[:2]
    size=int(cfg['output_size'])
    views=int(views or cfg.get('views',2))
    if views<2 or views>6:
        raise ValueError('views must be between 2 and 6')
    assert h>0 and w>0
    source_ones=np.ones((h,w),dtype='uint8')*255
    lo=float(cfg['min_overlap']); hi=float(cfg['max_overlap'])
    min_valid=float(cfg['min_valid_fraction'])
    max_tries=int(cfg.get('max_sampling_attempts',180))
    rot=float(cfg.get('rotation_degrees',15))
    minscale=float(cfg.get('scale_min',.9)); maxscale=float(cfg.get('scale_max',1.1))
    shr=float(cfg.get('shear_degrees',0.0))
    perspective=float(cfg.get('perspective_fraction',0.0))
    def create(center):
        return view_matrix(center,size,rng.uniform(-rot,rot),rng.uniform(minscale,maxscale),
                           rng.uniform(-shr,shr),perspective,rng)
    base=None
    for _ in range(max_tries):
        center=cfg.get('reference_center',(w/2,h/2))
        cen=(center[0]+rng.uniform(-.08,.08)*w,center[1]+rng.uniform(-.08,.08)*h)
        H0=create(cen)
        v0=warp(source_ones,H0,size,mask=True)
        if ratio(v0,np.ones_like(v0,dtype='uint8'))>=min_valid:
            base=(H0,v0,cen)
            break
    if base is None:
        raise RuntimeError('Cannot obtain adequately covered reference image. Try smaller output_size / lower min_valid_fraction.')
    Hs=[base[0]]; masks=[base[1]]
    for vidx in range(1,views):
        found=None
        for _ in range(max_tries):
            target=rng.uniform(lo,hi)
            # Shift source centers, roughly mapping overlap into area fraction.
            angle=rng.uniform(0,2*np.pi)
            shift=(1-target)*size/rng.uniform(minscale,maxscale)*rng.uniform(.65,1.5)
            cen=(base[2][0]+np.cos(angle)*shift,base[2][1]+np.sin(angle)*shift)
            H=create(cen)
            val=warp(source_ones,H,size,mask=True)
            if ratio(val,np.ones_like(val,dtype='uint8'))<min_valid:
                continue
            rH=relative_matrix(Hs[0],H)
            ov0,ov1=overlap_mask(masks[0],val,rH)
            score=min(ratio(ov0,masks[0]),ratio(ov1,val))
            if lo<=score<=hi:
                found=(H,val)
                break
        if found is None:
            raise RuntimeError(f'Could not sample view {vidx} with overlap [{lo},{hi}] in {max_tries} tries. Lower min_valid_fraction or change output_size.')
        Hs.append(found[0]);masks.append(found[1])
    return Hs,masks
