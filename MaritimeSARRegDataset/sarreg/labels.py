from __future__ import annotations
import cv2
import numpy as np
from .geometry import warp


def annotation_mask(shape,annotation):
    """Use COCO polygons when available, else honest bounding-box proxy.
    Compressed RLE needs optional pycocotools; if absent use bbox proxy.
    """
    h,w=shape[:2]
    m=np.zeros((h,w),dtype='uint8')
    seg=annotation.get('segmentation') or []
    method='bbox_proxy'
    if isinstance(seg,list) and seg:
        polycount=0
        for flat in seg:
            a=np.asarray(flat,dtype=np.float64)
            if a.size<6 or a.size%2:
                continue
            pts=np.rint(a.reshape(-1,2)).astype('int32')
            cv2.fillPoly(m,[pts],255)
            polycount+=1
        if polycount:
            method='polygon'
    elif isinstance(seg,dict):
        try:
            from pycocotools import mask as maskutils
            rrle=seg
            if isinstance(seg.get('counts'),list):
                rrle=maskutils.frPyObjects(seg,h,w)
            decoded=maskutils.decode(rrle)
            if decoded.ndim>2:
                decoded=np.any(decoded,axis=2)
            if decoded.shape==(h,w):
                m=decoded.astype('uint8')*255
                method='rle'
        except (ImportError,ValueError,TypeError):
            pass
    if not np.any(m):
        x,y,bw,bh=annotation['bbox_xywh']
        x0=max(0,int(np.floor(x)));y0=max(0,int(np.floor(y)))
        x1=min(w,int(np.ceil(x+bw)));y1=min(h,int(np.ceil(y+bh)))
        if x1>x0 and y1>y0:
            m[y0:y1,x0:x1]=255
    return m,method


def instance_views(source_shape,annotations,Hs,size,resized_shape,min_visibility=.35,min_pixels=8):
    """Propagate ship instance segmentation or box proxies to all views."""
    h,w=source_shape[:2]
    rows=[[] for _ in Hs]
    masks=[np.zeros((size,size),dtype='uint8') for _ in Hs]
    instances=[[] for _ in Hs]
    for a in annotations:
        src,method=annotation_mask((h,w),a)
        if tuple(src.shape)!=tuple(resized_shape):
            src=cv2.resize(src,(int(resized_shape[1]),int(resized_shape[0])),interpolation=cv2.INTER_NEAREST)
        original_area=np.count_nonzero(src)
        if original_area==0:
            continue
        for i,H in enumerate(Hs):
            m=warp(src,H,size,mask=True)
            area=np.count_nonzero(m)
            visible=area/max(1,original_area)
            # visible fraction also naturally decreases with scaling: use transformed full-area instead of source area
            # compute expected scale using H's local linear Jacobian at source center.
            center=np.array([src.shape[1]/2,src.shape[0]/2,1.0])
            z=H@center
            delta=1.0
            px=(H@(center+np.array([delta,0,0.])))
            py=(H@(center+np.array([0,delta,0.])))
            p0=z[:2]/z[2]; p1=px[:2]/px[2]; p2=py[:2]/py[2]
            jac=np.column_stack((p1-p0,p2-p0))
            expected=original_area*abs(np.linalg.det(jac))
            visible=float(min(1.,area/max(expected,1.)))
            if visible<min_visibility or area<min_pixels:
                continue
            ys,xs=np.where(m>0)
            x0,x1=int(xs.min()),int(xs.max()+1)
            y0,y1=int(ys.min()),int(ys.max()+1)
            row={'instance_id':str(a['id']),'category':'ship','bbox_xywh':[x0,y0,x1-x0,y1-y0],
                 'visible_fraction_estimate':round(visible,5),'area_pixels':int(area),'mask_source':method}
            rows[i].append(row)
            masks[i]=cv2.bitwise_or(masks[i],m)
            instances[i].append((row,m))
    return rows,masks,instances
