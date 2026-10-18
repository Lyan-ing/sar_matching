"""Fixed synthetic-validation diagnostics; predictor never gets GT/geography."""
import time,json,torch,cv2,numpy as np
from torch.nn import functional as F
from common import ROOT,atomic_json
from data import pair,warp,valid_at
from lightglue import LightGlue
from sarbench.matchers import Prediction,_mutual_ratio
from sarbench.estimate import estimate,perspective
from sarbench.reliability import gate,geometry_diagnostics,spatial

@torch.no_grad()
def image_matches(model,matcher,image0,image1):
 """Only raw arrays; no Homography/Mask/GeoTransform/manual data accepted."""
 device=next(model.parameters()).device;ims=[torch.from_numpy(im.copy()).to(device).float()[None,None]/255 for im in [image0,image1]]
 f0,f1=[model.extract(im) for im in ims];out=matcher(dict(image0=f0,image1=f1));ids=out['matches'][0]
 p0=f0['keypoints'][0][ids[:,0]].cpu().numpy();p1=f1['keypoints'][0][ids[:,1]].cpu().numpy()
 prediction=Prediction('SP_LG_'+model.variant,p0.astype(np.float32),p1.astype(np.float32));est=estimate(prediction,image0.shape,model='homography',seed=42)
 diag=geometry_diagnostics(est,prediction,image0.shape,image1.shape,42) if est.candidate_H is not None else {};est=gate(est,diag,'full')
 return f0,f1,out,est,diag

def model_validation(model,plan,matcher=None,output=None):
 model.eval();device=next(model.parameters()).device
 matcher=matcher or LightGlue(features='superpoint').eval().to(device);matcher.requires_grad_(False);rows=[]
 for spec in plan:
  z=pair(spec['source'],spec['seed'],spec['regime'],dynamic_occlusion=spec['dynamic_occlusion']);a,b=z['images'];torch.cuda.synchronize();start=time.perf_counter()
  with torch.no_grad():f0,f1,matched,e,diag=image_matches(model,matcher,a,b)
  torch.cuda.synchronize();ms=1000*(time.perf_counter()-start);ids=matched['matches'][0].cpu().numpy();kp=[f['keypoints'][0].cpu().numpy() for f in [f0,f1]];ds=[f['descriptors'][0].cpu().numpy() for f in [f0,f1]]
  pp=warp(z['H_AB'],kp[0]);visible=valid_at(z['static'][0],kp[0])&valid_at(z['static'][1],pp)
  repeat=float((np.linalg.norm(pp[visible,None,:]-kp[1][None,:,:],axis=2).min(1)<=3).mean()) if visible.any() and len(kp[1]) else None
  nn0,nn1=_mutual_ratio(ds[0].astype(np.float32),ds[1].astype(np.float32),cv2.BFMatcher(cv2.NORM_L2),.8)
  def correct(x,y):
   if not len(x):return 0
   return int(((np.linalg.norm(warp(z['H_AB'],x)-y,axis=1)<=3)&valid_at(z['static'][0],x)&valid_at(z['static'][1],y)).sum())
  gt_error=None
  # Geometry evaluation uses a separate uniform grid, not gradient-ranked
  # descriptor-training sample locations. All valid static overlap retained.
  gy,gx=np.mgrid[12:a.shape[0]-12:12,12:a.shape[1]-12:12];checks=np.c_[gx.ravel(),gy.ravel()].astype(float);targets=warp(z['H_AB'],checks);keep=valid_at(z['static'][0],checks)&valid_at(z['static'][1],targets);checks,targets=checks[keep],targets[keep]
  if e.candidate_H is not None and len(checks):gt_error=float(np.linalg.norm(perspective(e.candidate_H,checks)-targets,axis=1).mean())
  row=dict(id=spec['id'],dataset=spec['source']['dataset'],scene=spec['source']['scene_type'],source_component=spec['source']['training_component'],regime=spec['regime'],dynamic_occlusion=spec['dynamic_occlusion'],overlap=z['overlap'],n_keypoints=[len(x) for x in kp],detector_repeatability_3px=repeat,coverage=[spatial(x,a.shape) for x in kp],NN_matches=len(nn0),NN_correct_3px=correct(kp[0][nn0],kp[1][nn1]),LG_matches=len(ids),LG_correct_3px=correct(kp[0][ids[:,0]],kp[1][ids[:,1]]),n_inliers=e.n_inliers,accepted=e.accepted,reason=e.reason,decision='accepted' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),geometry=diag,gt_reprojection_mean_px=gt_error,elapsed_ms=ms,low_texture=float(a.std())<12)
  row['n_uniform_static_geometry_checks']=len(checks);row['GT_scope']='uniform grid in valid static synthetic overlap, not whole-image extrapolation'
  row.update({'Success@'+str(t)+'px':bool(spec['regime']!='none' and e.accepted and gt_error is not None and gt_error<=t) for t in [3,5,10]})
  row['incorrect_accept']=bool(e.accepted and (spec['regime']=='none' or gt_error is not None and gt_error>5))
  if len(z['points0'])>=16:
   with torch.no_grad():
    im=[torch.from_numpy(v.copy()).to(device).float()[None,None]/255 for v in [a,b]];pts=[torch.from_numpy(p).to(device)[None] for p in [z['points0'],z['points1']]];d=[model.descriptors(i,p)[0][0].float() for i,p in zip(im,pts)];sim=d[0]@d[1].T;target=torch.arange(sim.shape[0],device=device)
    # Near-positive alternatives excluded for retrieval exactly as in loss.
    near=(torch.cdist(pts[0][0],pts[0][0])<8)|(torch.cdist(pts[1][0],pts[1][0])<8);near.fill_diagonal_(False);sim=sim.masked_fill(near,-1e4)
    row['descriptor_recall1']=float((sim.argmax(1)==target).float().mean());row['descriptor_recall5']=float((sim.topk(min(5,len(target)),1).indices==target[:,None]).any(1).float().mean());row['positive_similarity']=float(sim.diag().mean());negative=~near&~torch.eye(len(target),device=device,dtype=torch.bool);row['negative_similarity']=float(sim[negative].mean())
  rows.append(row)
 positives=[r for r in rows if r['regime']!='none'];negatives=[r for r in rows if r['regime']=='none'];recalls=[r['descriptor_recall1'] for r in positives if 'descriptor_recall1' in r]
 summary=dict(n_pairs=len(rows),positive_pairs=len(positives),none_pairs=len(negatives),descriptor_recall1=float(np.mean(recalls)) if recalls else 0.,descriptor_recall5=float(np.mean([r['descriptor_recall5'] for r in positives if 'descriptor_recall5' in r])) if recalls else 0.,acceptance_rate=float(np.mean([r['accepted'] for r in rows])),incorrect_accept_rate=float(np.mean([r['incorrect_accept'] for r in rows])),none_false_accept_rate=float(np.mean([r['accepted'] for r in negatives])) if negatives else 0.,mean_ms=float(np.mean([r['elapsed_ms'] for r in rows])),detector_repeatability_3px=float(np.mean([r['detector_repeatability_3px'] for r in positives if r['detector_repeatability_3px'] is not None])) if any(r['detector_repeatability_3px'] is not None for r in positives) else 0.)
 summary.update({'Success@'+str(t)+'px':float(np.mean([r['Success@'+str(t)+'px'] for r in positives])) for t in [3,5,10]})
 summary['selection_score']=[summary['Success@5px'],-summary['none_false_accept_rate'],summary['descriptor_recall1']]
 result=dict(summary=summary,rows=rows,GT_kind='known single-image 2D synthetic geometry; not real different-look GT')
 if output is not None:atomic_json(output,result)
 return result
