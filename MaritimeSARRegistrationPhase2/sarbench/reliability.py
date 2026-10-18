"""Prediction-only geometric diagnostics. These functions cannot receive GT or masks."""
from dataclasses import replace
import time
import cv2
import numpy as np
from .estimate import perspective,estimate,_grid_coverage
from .matchers import Prediction

POLICIES={
 'baseline':{},'pole':{'pole':True},'pole_shape':{'pole':True,'shape':True},
 'pole_shape_support005':{'pole':True,'shape':True,'min_hull':.005},
 'pole_shape_support01':{'pole':True,'shape':True,'min_hull':.01},
 'pole_shape_support02':{'pole':True,'shape':True,'min_hull':.02},
 'pole_shape_bidirectional':{'pole':True,'shape':True,'cycle':True},
 'full':{'pole':True,'shape':True,'min_hull':.01,'cycle':True}}

def spatial(points,shape):
 p=np.asarray(points,np.float32).reshape(-1,2);h,w=shape
 hull=float(cv2.contourArea(cv2.convexHull(p)))/(h*w) if len(p)>=3 else 0.
 eig=np.linalg.eigvalsh(np.cov(p.T)) if len(p)>=3 else np.array([0.,0.])
 isotropy=float(max(0,eig[0])/max(eig[-1],1e-12))
 edge=float(np.mean((p[:,0]<31)|(p[:,0]>=w-31)|(p[:,1]<31)|(p[:,1]>=h-31))) if len(p) else None
 return {'grid_coverage':_grid_coverage(p,shape),'hull_fraction':hull,'isotropy':isotropy,'edge31_fraction':edge}

def grid(shape):
 h,w=shape;x,y=np.meshgrid(np.linspace(0,w-1,5),np.linspace(0,h-1,5));return np.c_[x.ravel(),y.ravel()]

def projection_diagnostics(H,shape):
 h,w=shape;p=np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]],float)
 H=np.asarray(H,float);z=np.c_[p,np.ones(4)]@H[2]
 finite=bool(np.isfinite(H).all() and np.isfinite(z).all())
 same_sign=bool(finite and (np.all(z>0) or np.all(z<0)))
 ratio=float(np.min(np.abs(z))/max(np.max(np.abs(z)),1e-30)) if finite else 0.
 with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
  q=perspective(H,p)
  area=float(cv2.contourArea(q.astype(np.float32),oriented=True)/max((w-1)*(h-1),1)) if np.isfinite(q).all() else None
  lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)/np.array([w-1,h-1,w-1,h-1])
  conds=[]
  for pt in p:
   v=perspective(H,np.array([pt,pt+[1,0],pt+[0,1]]));J=np.c_[v[1]-v[0],v[2]-v[0]]
   conds.append(float(np.linalg.cond(J)) if np.isfinite(J).all() else float('inf'))
 return {'denominator_same_sign':same_sign,'denominator_margin':ratio,
         'corner_area_ratio':area,'convex_corners':bool(np.isfinite(q).all() and cv2.isContourConvex(q.astype(np.float32))),
         'min_edge_scale':float(np.min(lengths)) if np.isfinite(lengths).all() else None,
         'max_edge_scale':float(np.max(lengths)) if np.isfinite(lengths).all() else None,
         'max_jacobian_condition':max(conds) if np.isfinite(conds).all() else None}

def geometry_diagnostics(est,pred,shape0,shape1,seed=42):
 """Reverse H is independently fitted to reversed image-only matches, not inv(H)."""
 start=time.perf_counter();H=est.candidate_H if est.candidate_H is not None else est.H
 if H is None:return {'extra_diagnostic_ms':1000*(time.perf_counter()-start),'has_candidate':False}
 result={'has_candidate':True};t=time.perf_counter()
 try:
  forward=projection_diagnostics(H,shape0);backward=projection_diagnostics(np.linalg.inv(H),shape1)
  result.update({f'{side}_{k}':v for side,d in [('forward',forward),('inverse',backward)] for k,v in d.items()})
 except (ValueError,np.linalg.LinAlgError,cv2.error):result['invalid_matrix']=True
 result['projection_diagnostic_ms']=1000*(time.perf_counter()-t)
 inliers=est.inlier_mask if est.inlier_mask is not None else np.zeros(len(pred.points0),bool)
 for side,pts,shape in [('inlier0',pred.points0[inliers],shape0),('inlier1',pred.points1[inliers],shape1)]:
  result.update({f'{side}_{k}':v for k,v in spatial(pts,shape).items()})
 if np.any(inliers):
  try:
   a=perspective(H,pred.points0[inliers])-pred.points1[inliers]
   b=perspective(np.linalg.inv(H),pred.points1[inliers])-pred.points0[inliers]
   symmetric=np.sqrt(np.mean((np.sum(a*a,axis=1)+np.sum(b*b,axis=1))/2))
   result['symmetric_transfer_rmse']=float(symmetric) if np.isfinite(symmetric) else None
  except np.linalg.LinAlgError:result['symmetric_transfer_rmse']=None
 t=time.perf_counter();reverse=estimate(Prediction(pred.method,pred.points1,pred.points0),shape1,model=est.model_type,seed=seed)
 result['reverse_estimation_ms']=1000*(time.perf_counter()-t)
 reverseH=reverse.candidate_H
 if reverseH is not None:
  errors=[]
  for C,shape in [(reverseH@H,shape0),(H@reverseH,shape1)]:
   xy=grid(shape);errors.extend(np.linalg.norm(perspective(C,xy)-xy,axis=1).tolist())
  result['bidirectional_cycle_mean_px']=float(np.mean(errors)) if np.isfinite(errors).all() else None
  result['bidirectional_cycle_max_px']=float(np.max(errors)) if np.isfinite(errors).all() else None
 else:result['bidirectional_cycle_mean_px']=None
 result['extra_diagnostic_ms']=1000*(time.perf_counter()-start)
 return result

def gate(est,diagnostics,policy):
 """Scale-invariant finite projective domain, broad shape prior, support and cycle."""
 if policy not in POLICIES:raise ValueError(policy)
 if not est.accepted or policy=='baseline':return est
 settings=POLICIES[policy];d=diagnostics;reason='ok'
 if settings.get('pole') and (d.get('invalid_matrix') or any(not d.get(f'{side}_denominator_same_sign') or d.get(f'{side}_denominator_margin',0)<.01 for side in ['forward','inverse'])):reason='projective_pole_or_near_pole'
 if reason=='ok' and settings.get('shape'):
  for side in ['forward','inverse']:
   area=d.get(f'{side}_corner_area_ratio');minimum=d.get(f'{side}_min_edge_scale');maximum=d.get(f'{side}_max_edge_scale');condition=d.get(f'{side}_max_jacobian_condition')
   if not d.get(f'{side}_convex_corners') or area is None or not .1<=area<=10 or minimum is None or minimum<.1 or maximum is None or maximum>10 or condition is None or condition>20:reason='unstable_corner_shape';break
 if reason=='ok' and 'min_hull' in settings:
  if any(d.get(f'inlier{side}_hull_fraction',0)<settings['min_hull'] or d.get(f'inlier{side}_isotropy',0)<.005 for side in [0,1]):reason='weak_spatial_support'
 if reason=='ok' and settings.get('cycle'):
  cycle=d.get('bidirectional_cycle_mean_px');symmetric=d.get('symmetric_transfer_rmse')
  if cycle is None or cycle>10 or symmetric is None or symmetric>3:reason='bidirectional_inconsistent'
 return est if reason=='ok' else replace(est,H=None,accepted=False,reason=reason)
