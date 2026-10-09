"""Shared robust geometry estimation and prediction-only reliability gate."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np
from .matchers import Prediction

@dataclass
class Estimated:
    H: np.ndarray | None
    accepted: bool
    reason: str
    n_matches: int
    n_inliers: int
    inlier_ratio: float
    inlier_rmse: float | None
    coverage: float | None
    confidence: float | None
    model_type: str
    candidate_H: np.ndarray | None = None


def perspective(H, xy):
    pts=np.asarray(xy,dtype=np.float64).reshape(-1,2)
    if not len(pts):
        return pts
    a=np.column_stack((pts,np.ones(len(pts)))) @ np.asarray(H,dtype=np.float64).T
    z=a[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        return a[:,:2]/z[:,None]


def _grid_coverage(points: np.ndarray, shape: tuple[int,int], cells=4) -> float:
    if len(points)==0:
        return 0.
    h,w=shape
    g=np.floor(points*np.array([cells/max(w,1),cells/max(h,1)])).astype(int)
    g=np.clip(g,0,cells-1)
    return float(len(set(map(tuple,g)))/(cells*cells))


def estimate(pred: Prediction, img_shape: tuple[int,int], *, model='homography',
             reproj_px=3.0, min_matches=8, min_inliers=8, min_inlier_ratio=.3,
             min_coverage=.0625, max_inlier_rmse=3., min_phase_response=.20,
             seed=42) -> Estimated:
    if pred.model_type=='translation':
        H=pred.precomputed_H
        conf=float(pred.confidence) if pred.confidence is not None else float('-inf')
        accepted=bool(np.isfinite(conf) and conf>=min_phase_response and H is not None and np.isfinite(H).all())
        return Estimated(H if accepted else None,accepted,'ok' if accepted else 'low_phase_response',
                         0,0,0.,None,None,conf,'translation',H)
    p0,p1=pred.points0,pred.points1
    n=len(p0)
    if n != len(p1):
        raise ValueError('Unequal match arrays')
    if n < max(min_matches,4 if model=='homography' else 3):
        return Estimated(None,False,'too_few_matches',n,0,0.,None,None,None,model)
    cv2.setRNGSeed(int(seed)%(2**31-1))
    if model=='homography':
        H,inliers=cv2.findHomography(p0,p1,cv2.RANSAC,reproj_px,maxIters=5000,confidence=.995)
    elif model=='affine':
        affine,inliers=cv2.estimateAffine2D(p0,p1,method=cv2.RANSAC,ransacReprojThreshold=reproj_px,
                                             maxIters=5000,confidence=.995,refineIters=10)
        H=np.vstack((affine,[0.,0.,1.])) if affine is not None else None
    else:
        raise ValueError(f'Unknown model {model}')
    if H is None or inliers is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-10:
        return Estimated(None,False,'estimation_failed',n,0,0.,None,None,None,model)
    inliers=np.asarray(inliers,dtype=bool).reshape(-1)
    count=int(np.count_nonzero(inliers))
    ratio=float(count/max(n,1))
    err=np.linalg.norm(perspective(H,p0[inliers])-p1[inliers],axis=1)
    rmse=float(np.sqrt(np.mean(err**2))) if len(err) else float('inf')
    coverage=_grid_coverage(p0[inliers],img_shape)
    # Gate uses predictions and raw image geometry ONLY, never GT masks or true H.
    reason='ok'
    if count<min_inliers: reason='few_inliers'
    elif ratio<min_inlier_ratio: reason='low_inlier_ratio'
    elif coverage<min_coverage: reason='spatially_degenerate'
    elif not np.isfinite(rmse) or rmse>max_inlier_rmse: reason='large_fit_residual'
    accepted=reason=='ok'
    return Estimated(H if accepted else None,accepted,reason,n,count,ratio,rmse,coverage,None,model,H)
