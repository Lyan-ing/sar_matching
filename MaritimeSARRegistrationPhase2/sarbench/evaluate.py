"""Evaluate predictions with held-out GT; special handling for zero overlap."""
from __future__ import annotations
import numpy as np
from .estimate import Estimated, perspective


def evaluate(est: Estimated, gt: dict, image_shape: tuple[int, int], *,
             success_thresholds=(3.,5.,10.)) -> dict:
    no_overlap=bool(gt['no_overlap'])
    result={'overlap':float(gt['overlap']), 'no_overlap':no_overlap,
            'has_estimate':bool(est.candidate_H is not None or est.H is not None),
            'decision_status':'incorrect_accept' if est.accepted else ('rejected' if est.candidate_H is not None else 'no_output'),
            'gt_points':int(gt['count']), 'correct_reject':bool(no_overlap and not est.accepted),
            'false_accept':bool(no_overlap and est.accepted),
            'failure_to_register':bool(not no_overlap and not est.accepted),
            'epe_mean':None,'epe_median':None,'epe_p90':None,
            'epe_normalized':None, 'gt_match_precision_3px':None, 'invalid_projection':False}
    for t in success_thresholds:
        result[f'success_{int(t)}px'] = None if no_overlap else False
    if no_overlap:
        return result
    if not est.accepted or est.H is None or gt['count'] == 0:
        return result
    # No training/decision ever uses these GT points; only post-hoc evaluation.
    pts=np.asarray(gt['points0'],np.float64)
    q=np.asarray(gt['points1'],np.float64)
    estimate_pts=perspective(est.H,pts)
    if not np.isfinite(estimate_pts).all():
        result['invalid_projection']=True
        return result
    errs=np.linalg.norm(estimate_pts-q,axis=1)
    if not np.isfinite(errs).all():
        result['invalid_projection']=True
        return result
    if not len(errs):
        return result
    mean=float(errs.mean())
    h,w=image_shape
    result.update(epe_mean=mean,epe_median=float(np.median(errs)),
                  epe_p90=float(np.percentile(errs,90)),
                  epe_normalized=mean/max(np.hypot(h,w),1),
                  gt_match_precision_3px=float(np.mean(errs<=3.)))
    for t in success_thresholds:
        result[f'success_{int(t)}px'] = bool(mean <= t)
    if result.get('success_5px'):
        result['decision_status']='success'
    return result
