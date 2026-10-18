"""Prediction-only heterogeneous correspondence fusion. No data readers or labels."""
from dataclasses import replace
import time
import numpy as np,cv2
from sarbench.matchers import Prediction
from sarbench.estimate import estimate,perspective
from sarbench.reliability import geometry_diagnostics,gate,spatial,grid

def geometry(p,shapes):
 e=estimate(p,shapes[0],model='homography',seed=42)
 d=geometry_diagnostics(e,p,*shapes,42) if e.candidate_H is not None else {}
 return gate(e,d,'full'),d
def residual(H,p):
 try:
  a=np.linalg.norm(perspective(H,p.points0)-p.points1,axis=1)
  b=np.linalg.norm(perspective(np.linalg.inv(H),p.points1)-p.points0,axis=1)
  return np.maximum(a,b)
 except np.linalg.LinAlgError:return np.full(len(p.points0),np.inf)
def near(p,q):
 if not len(p.points0) or not len(q.points0):return np.zeros(len(p.points0),bool)
 return ((np.linalg.norm(p.points0[:,None]-q.points0[None],axis=2)<=3)&(np.linalg.norm(p.points1[:,None]-q.points1[None],axis=2)<=3)).any(1)
def joint(pools,points0,points1):
 ks=[np.concatenate([getattr(p,'keypoints'+str(i)) for p in pools]) for i in [0,1]]
 return Prediction('heterogeneous',np.asarray(points0,np.float32).reshape(-1,2),np.asarray(points1,np.float32).reshape(-1,2),n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1])
def spatial_order(p,shape):
 h,w=shape;groups={}
 for i,pt in enumerate(p):groups.setdefault(tuple(np.clip((pt*[4/w,4/h]).astype(int),0,3)),[]).append(i)
 order=[]
 while any(groups.values()):
  for cell in sorted(groups):
   if groups[cell]:order.append(groups[cell].pop(0))
 return np.array(order,int)
def clean_pool(p0,p1,origin):
 """Keep one exact spatial duplicate; remove BOTH sides of a one-to-many conflict."""
 keep=[];blocked=set();duplicates=0;conflicts=0
 for i in range(len(p0)):
  if i in blocked:continue
  for j in keep.copy():
   a=np.linalg.norm(p0[i]-p0[j])<=3;b=np.linalg.norm(p1[i]-p1[j])<=3
   if a and b:duplicates+=1;blocked.add(i);break
   if a!=b:conflicts+=1;blocked.update([i,j]);keep.remove(j)
  if i not in blocked:keep.append(i)
 return np.array(keep,int),duplicates,conflicts
def quality(e,d):
 return (min(d.get('inlier0_hull_fraction',0),d.get('inlier1_hull_fraction',0)),e.inlier_ratio,-(e.inlier_rmse if e.inlier_rmse is not None else np.inf))
def strategies(sp,orb,shapes):
 """Same fixed input match pools for F2, F3 and naive concatenation."""
 start=time.perf_counter();pools=[sp,orb];fits=[geometry(p,shapes) for p in pools]
 eligible=[i for i,(e,d) in enumerate(fits) if e.accepted]
 chosen=max(eligible,key=lambda i:(quality(*fits[i]),-i)) if eligible else 0
 base=pools[chosen];p=joint(pools,base.points0,base.points1);e,d=fits[chosen]
 fit_ms=1000*(time.perf_counter()-start)
 result={'F2':(p,e,d,dict(route='choose_sp' if chosen==0 else 'choose_orb',sources=np.full(len(p.points0),chosen,int),pool_indices=np.arange(len(p.points0)),sp_matches=len(sp.points0),orb_matches=len(orb.points0),geometry_ms=fit_ms))}
 concat_start=time.perf_counter()
 concat=joint(pools,np.concatenate([p.points0 for p in pools]),np.concatenate([p.points1 for p in pools]));ce,cd=geometry(concat,shapes)
 result['Concat']=(concat,ce,cd,dict(route='naive_concat',sources=np.r_[np.zeros(len(sp.points0),int),np.ones(len(orb.points0),int)],pool_indices=np.r_[np.arange(len(sp.points0)),np.arange(len(orb.points0))],geometry_ms=1000*(time.perf_counter()-concat_start)))
 fuse_start=time.perf_counter();candidates=[];attempts=[]
 for seed,(seed_e,_) in enumerate(fits):
  H=seed_e.candidate_H
  if H is None:attempts.append(dict(seed=seed,reason='no_seed_H'));continue
  other=1-seed;cross=(residual(H,pools[other])<=3)&~near(pools[other],pools[seed]);count=int(cross.sum());fraction=count/max(len(pools[other].points0),1)
  cells=spatial(pools[other].points0[cross],shapes[0])['grid_coverage']*16
  if count<4 or fraction<.3 or cells<2:attempts.append(dict(seed=seed,reason='insufficient_distinct_cross_support',cross=count,fraction=fraction,cells=cells));continue
  selected=[np.flatnonzero(residual(H,x)<=3) for x in pools]
  pp0=np.concatenate([x.points0[ids] for x,ids in zip(pools,selected)]);pp1=np.concatenate([x.points1[ids] for x,ids in zip(pools,selected)]);origin=np.r_[np.zeros(len(selected[0]),int),np.ones(len(selected[1]),int)];original_idx=np.concatenate(selected)
  keep,dups,conflicts=clean_pool(pp0,pp1,origin);pp0,pp1,origin,original_idx=pp0[keep],pp1[keep],origin[keep],original_idx[keep]
  per=[np.flatnonzero(origin==i) for i in [0,1]];cap=min(map(len,per))
  if cap<4:attempts.append(dict(seed=seed,reason='fewer4_each_after_conflict',duplicates=dups,conflicts=conflicts));continue
  selected=[ids[spatial_order(pp0[ids],shapes[0])[:cap]] for ids in per];ids=np.array([j for pair in zip(*selected) for j in pair],int)
  p=joint(pools,pp0[ids],pp1[ids]);e,d=geometry(p,shapes);src=origin[ids];rawidx=original_idx[ids]
  supports=[int(((residual(e.candidate_H,p)<=3)&(src==i)).sum()) if e.candidate_H is not None else 0 for i in [0,1]]
  meta=dict(route='cross_supported_balanced',seed_source=seed,sources=src,pool_indices=rawidx,cross_support=count,cross_fraction=fraction,duplicates_removed=dups,conflicts_removed=conflicts,balanced_cap_each=cap,support_each=supports)
  attempts.append(dict(seed=seed,reason=e.reason if min(supports)>=4 else 'lost_source_support',cross=count,accepted=e.accepted and min(supports)>=4))
  if e.accepted and min(supports)>=4:candidates.append((p,e,d,meta))
 if len(candidates)==2:
  distances=[]
  for side,shape in enumerate(shapes):
   hs=[x[1].candidate_H if side==0 else np.linalg.inv(x[1].candidate_H) for x in candidates]
   distances.extend(np.linalg.norm(perspective(hs[0],grid(shape))-perspective(hs[1],grid(shape)),axis=1))
  if not np.isfinite(distances).all() or np.mean(distances)>3:
   p,e,d,meta=result['F2'];result['F3']=(p,replace(e,H=None,accepted=False,reason='fusion_hypothesis_conflict'),d,dict(meta,route='reject_conflicting_fused_hypotheses',attempts=attempts,geometry_ms=fit_ms+1000*(time.perf_counter()-fuse_start)));return result
 if candidates:
  pick=max(candidates,key=lambda x:(min(x[3]['support_each']),quality(x[1],x[2]),-x[3]['seed_source']));p,e,d,meta=pick;result['F3']=(p,e,d,dict(meta,attempts=attempts))
 else:
  p,e,d,meta=result['F2'];result['F3']=(p,e,d,dict(meta,route='F2_fallback',attempts=attempts))
 p,e,d,meta=result['F3'];result['F3']=(p,e,d,dict(meta,geometry_ms=fit_ms+1000*(time.perf_counter()-fuse_start)))
 return result
