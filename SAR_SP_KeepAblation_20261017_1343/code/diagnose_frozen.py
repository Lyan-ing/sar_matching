"""CPU post-hoc analysis of frozen outputs. GT-assisted fits never become predictions."""
from pathlib import Path
import json,csv,sys,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;REF=BASE/'SAR_SP_LowTextureKeep_20261017_0908'
sys.path.insert(0,str(BASE/'MaritimeSARRegistrationPhase2'))
from sarbench.matchers import Prediction
from sarbench.estimate import estimate,perspective
from sarbench.reliability import spatial,geometry_diagnostics,gate,projection_diagnostics
cv2.setNumThreads(1)
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def write(path,o):
 with Path(path).open('x') as f:json.dump(clean(o),f,ensure_ascii=False,indent=2,allow_nan=False)
def csvout(path,rr):
 with Path(path).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader();w.writerows(clean(rr))
def at(mask,p):
 p=np.asarray(p);x=np.floor(p[:,0]).astype(int);y=np.floor(p[:,1]).astype(int);x1=x+1;y1=y+1;h,w=mask.shape;inside=(x>=0)&(y>=0)&(x1<w)&(y1<h);v=np.zeros(len(p),bool)
 ids=np.flatnonzero(inside);v[ids]=mask[y[ids],x[ids]]&mask[y1[ids],x[ids]]&mask[y[ids],x1[ids]]&mask[y1[ids],x1[ids]];return v

def masks(gt,a,b):
 target=perspective(gt['H_AB'],a);reverse=perspective(gt['H_BA'],b);err=np.maximum(np.linalg.norm(target-b,axis=1),np.linalg.norm(reverse-a,axis=1));geom=np.isfinite(err)&(err<=3)
 static=[gt['static_'+L]&~gt['ignore_'+L]&gt['valid_'+L] for L in 'AB'];eligible=at(static[0],a)&at(static[1],b)&at(static[1],target);good=geom&eligible
 return geom,eligible,good,err

def oracle(a,b,good,gt,shape):
 p=Prediction('GT-filtered diagnostic ONLY',a[good],b[good]);e=estimate(p,shape,model='homography',seed=42)
 d=geometry_diagnostics(e,p,shape,shape,42) if e.candidate_H is not None else {};e=gate(e,d,'full');err=None
 if e.candidate_H is not None:
  q0,q1=gt['check_points_A'],gt['check_points_B'];H=e.candidate_H
  try:err=float(np.r_[np.linalg.norm(perspective(H,q0)-q1,axis=1),np.linalg.norm(perspective(np.linalg.inv(H),q1)-q0,axis=1)].mean())
  except np.linalg.LinAlgError:pass
 return dict(accepted=e.accepted,reason=e.reason,error=err,success5=bool(e.accepted and err is not None and err<=5),n_inliers=e.n_inliers)

def main():
 t=stamp();rr=json.loads((REF/'final_scores_20261017_0908.json').read_text());lookup={(r['pair_id'],r['method'],r['protocol']):r for r in rr};val=[json.loads(s) for s in (BASE/'SyntheticSAR_Mixed640512_Multicondition_20261016_2114/manifests/val_CANONICAL_20261016_2147.jsonl').read_text().splitlines()];rows={r['pair_id']:r for r in val}
 wrong=[r['pair_id'] for r in rr if r['method']=='keep' and r['protocol']=='natural' and r['geometry_eligible'] and r['accepted'] and not r['Success@5px']]
 lost=[r['pair_id'] for r in rr if r['method']=='keep' and r['protocol']=='natural' and r['geometry_eligible'] and r['scene']=='offshore_vessels_low_texture_sea' and not r['Success@5px'] and lookup[r['pair_id'],'b','natural']['Success@5px']]
 assert len(wrong)==10 and len(lost)==14
 ids=sorted(set(wrong+lost));out=[];evidence=[];alllabels=[]
 for pid in ids:
  row=rows[pid];shape=(row['height'],row['width'])
  with np.load(row['gt_path']) as z:gt={k:z[k] for k in z.files}
  for method in ['b','t2','keep']:
   for mode in ['natural','equal']:
    r=lookup[pid,method,mode];path=Path(r['frozen_prediction_path'])
    with np.load(path) as z:p={k:z[k] for k in z.files}
    a,b=p['matches_A'],p['matches_B'];ins=p['inliers'].astype(bool);geom,eligible,good,err=masks(gt,a,b);assert good.sum()==r['correct_structure_matches'];assert ins.sum()==r['inliers']
    oracle_all=oracle(a,b,geom,gt,shape);oracle_static=oracle(a,b,good,gt,shape)
    d=dict(pair_id=pid,method=method,protocol=mode,scene=r['scene'],component=r['component'],wrong_accept_case=pid in wrong,offshore_loss_case=pid in lost,structure_global_eligible=r['structure_global_eligible'],overlap_bin=r['overlap_bin'],angle_bin=r['angle_bin'],image_rotation_2D_deg=row['image_rotation_2D_deg'],accepted=r['accepted'],state=r['state'],reason=r['reason'],success5=r['Success@5px'],global_error=r['bidirectional_error_px'],matches=len(a),ransac_inliers=int(ins.sum()),supervisable_matches=int(eligible.sum()),correct_static_matches=int(good.sum()),wrong_static_matches=int((eligible&~geom).sum()),unknown_or_ignored_matches=int((~eligible).sum()),synthetic_H_consistent_matches=int(geom.sum()),synthetic_H_inconsistent_matches=int((~geom).sum()),correct_static_inliers=int((good&ins).sum()),synthetic_H_consistent_inliers=int((geom&ins).sum()),inconsistent_inliers=int((~geom&ins).sum()),median_inlier_trueH_error=float(np.median(err[ins])) if ins.any() else None,gt_visible_detection=r['visible_detected_structure_points'],gt_repeatable_detection=r['repeatable_structure_points'],oracle_synthetic_success5=oracle_all['success5'],oracle_static_success5=oracle_static['success5'],oracle_synthetic_reason=oracle_all['reason'],oracle_static_reason=oracle_static['reason'],oracle_synthetic_error=oracle_all['error'],oracle_static_error=oracle_static['error'],frozen_prediction=str(path),frozen_prediction_sha=hashlib.sha256(path.read_bytes()).hexdigest(),gt_assisted_oracles_are_diagnostics_only=True)
    for name,points in [('inlier',a[ins]),('synthetic_good',a[geom]),('static_good',a[good]),('moving_inlier',b[ins]),('moving_synthetic_good',b[geom]),('moving_static_good',b[good])]:d.update({name+'_'+k:v for k,v in spatial(points,shape).items()})
    if len(p['H_AB']):
     H=p['H_AB'];d.update({'candidate_forward_'+key:value for key,value in projection_diagnostics(H,shape).items()});d.update({'candidate_inverse_'+key:value for key,value in projection_diagnostics(np.linalg.inv(H),shape).items()});res=np.linalg.norm(perspective(H,a[ins])-b[ins],axis=1);d['candidate_inlier_rmse']=float(np.sqrt(np.mean(res**2))) if len(res) else None
    cp,cq=gt['check_points_A'],gt['check_points_B'];d['check_point_count']=len(cp)
    if len(p['H_AB']) and len(cp):
     def in_hull(points,checks):
      if len(points)<3:return np.zeros(len(checks),bool)
      hull=cv2.convexHull(points.astype(np.float32));return np.array([cv2.pointPolygonTest(hull,(float(x),float(y)),False)>=0 for x,y in checks])
     inside=in_hull(a[ins],cp)&in_hull(b[ins],cq);errors=(np.linalg.norm(perspective(p['H_AB'],cp)-cq,axis=1)+np.linalg.norm(perspective(np.linalg.inv(p['H_AB']),cq)-cp,axis=1))/2
     assert abs(float(errors.mean())-r['bidirectional_error_px'])<1e-5
     d.update(check_outside_inlier_hull_fraction=float((~inside).mean()),inside_check_error=float(errors[inside].mean()) if inside.any() else None,outside_check_error=float(errors[~inside].mean()) if (~inside).any() else None)
    if geom.sum()<8:category='insufficient_geometrically_correct_correspondences'
    elif oracle_all['success5'] and not r['Success@5px']:category='correct_pool_available_but_selected_geometry_failed'
    elif not oracle_all['success5']:category='correct_pool_local_or_unstable_global_support'
    else:category='success'
    d['failure_stage_diagnostic']=category
    out.append(d)
    if method=='keep' and mode=='natural':
     evidence.append(d);alllabels.append(dict(pair_id=pid,matches_A=a,matches_B=b,synthetic_H_consistent=geom,static_supervisable=eligible,static_correct=good,inliers=ins,trueH_error=err))
    print(pid,method,mode,len(a),int(geom.sum()),int(good.sum()),category,flush=True)
 csvout(ROOT/f'FROZEN_FAILURE_DETAILS_{t}.csv',out);write(ROOT/f'FROZEN_FAILURE_CASES_{t}.json',evidence)
 for e in alllabels:pid=e.pop('pair_id');np.savez_compressed(ROOT/f'predictions/{pid}_diagnostic_labels_{t}.npz',**e)
 drift={r['pair_id']:r for r in json.loads((REF/'DESCRIPTOR_DRIFT_20261017_0947.json').read_text())}
 for e in evidence:e.update({k:drift[e['pair_id']][k] for k in ['positions','old_drift','keep_drift','old_detected_drift','keep_detected_drift','old_detected_inside_drift','keep_detected_inside_drift','old_detected_outside_drift','keep_detected_outside_drift']})
 csvout(ROOT/f'FOCUSED_FAILURES_WITH_DRIFT_{t}.csv',evidence)
 from collections import Counter
 summaries={}
 for group,subset in [('wrong_accept',[e for e in evidence if e['wrong_accept_case']]),('offshore_loss',[e for e in evidence if e['offshore_loss_case']])]:
  summaries[group]=dict(n=len(subset),categories=dict(Counter(e['failure_stage_diagnostic'] for e in subset)),structure_qualified=sum(e['structure_global_eligible'] for e in subset),static_correct_total=sum(e['correct_static_matches'] for e in subset),synthetic_correct_total=sum(e['synthetic_H_consistent_matches'] for e in subset),median_matches=float(np.median([e['matches'] for e in subset])),median_inliers=float(np.median([e['ransac_inliers'] for e in subset])),median_H_correct=float(np.median([e['synthetic_H_consistent_matches'] for e in subset])),oracle_synthetic_success=sum(e['oracle_synthetic_success5'] for e in subset),oracle_static_success=sum(e['oracle_static_success5'] for e in subset),ids=[e['pair_id'] for e in subset])
 write(ROOT/f'FROZEN_DIAGNOSIS_SUMMARY_{t}.json',dict(groups=summaries,union_cases=len(ids),no_predictor_or_gate_changed=True,oracles='GT-filter same saved matches, then frozen estimator/full gate; diagnostics only, not a new algorithm',warning='Synthetic H consistency outside eligible static masks is not physical static correspondence evidence'))
 print(json.dumps(summaries,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
