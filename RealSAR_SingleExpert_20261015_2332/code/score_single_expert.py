"""Separate single-expert scoring; accepts a prediction, never fits/initializes it."""
import json
from pathlib import Path
import numpy as np

def checks(record):
 if record.get('evidence_type')!='single_expert_user_physical_confirmation' or record.get('GT_qualified') is not False:raise ValueError('Use the separate single-expert manifest, never relabel it as old two-person GT')
 ids=set(record['eligible_point_ids'])
 return [p for p in record['points'] if p['origin_observation_id'] in ids and p['role']=='check' and not p['physical_failures']]

def point_errors(H,xy0,xy1):
 H=np.asarray(H,float)
 if H.shape!=(3,3) or not np.isfinite(H).all():raise ValueError('A finite pure-image prediction3x3 is required')
 x=np.asarray(xy0,float).reshape(-1,2);y=np.asarray(xy1,float).reshape(-1,2);h=np.c_[x,np.ones(len(x))]@H.T
 with np.errstate(divide='ignore',invalid='ignore'):pred=h[:,:2]/h[:,2:]
 return np.linalg.norm(pred-y,axis=1)

def score_prediction(H,accepted,record):
 ps=checks(record);result=dict(sample_id=record['sample_id'],evidence_type='single_expert',scope=record['expert_gt_level'],checks=len(ps),origin_observation_ids=[p['origin_observation_id'] for p in ps],GT_qualified=False,mean_px=None,median_px=None,p90_px=None,p95_px=None,success={})
 thresholds=record['supported_success_thresholds_px'] if record['single_expert_geometry_qualified'] else []
 if H is not None and ps:
  e=point_errors(H,[p['xy0'] for p in ps],[p['xy1'] for p in ps])
  if np.isfinite(e).all():result.update(mean_px=float(e.mean()),median_px=float(np.median(e)),p90_px=float(np.quantile(e,.9)),p95_px=float(np.quantile(e,.95)),point_residuals_px=e.tolist())
 for threshold in thresholds:result['success'][str(threshold)]=bool(accepted and result['mean_px'] is not None and result['mean_px']<=threshold)
 return result
