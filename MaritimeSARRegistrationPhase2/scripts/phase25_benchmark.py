"""Image-only Phase 2.5 matching, cached same-match model/gate ablations and dev-only selection."""
import argparse, collections, dataclasses, hashlib, json, random, subprocess, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from sarbench.data import discover,load_images,load_gt
from sarbench.matchers import Prediction,LocalFeatureMatcher,SuperPointLightGlue
from sarbench.estimate import estimate,perspective
from sarbench.evaluate import evaluate
from sarbench.reliability import geometry_diagnostics,gate,spatial,POLICIES
from sarbench.reports import save_json,write_csv,summarize,bootstrap_group_success,_truth

VARIANTS={
 'sift_base':{'algorithm':'sift'},'orb_base':{'algorithm':'orb'},'lightglue_base':{'algorithm':'lightglue'},
 'sift_contrast01':{'algorithm':'sift','detector_options':{'contrastThreshold':.01}},
 'orb_edge16':{'algorithm':'orb','detector_options':{'edgeThreshold':16}},
 'lightglue_detect1e4':{'algorithm':'lightglue','detector_threshold':.0001},
 'lightglue_clahe':{'algorithm':'lightglue','preprocessing':'clahe_clip2_grid8'}}

def checksum(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sample(mask,points):
 xy=np.asarray(points,float).reshape(-1,2);h,w=mask.shape;valid=np.isfinite(xy).all(axis=1)
 q=np.rint(np.where(np.isfinite(xy),xy,-1)).astype(np.int64);inside=valid&(q[:,0]>=0)&(q[:,0]<w)&(q[:,1]>=0)&(q[:,1]<h);result=np.zeros(len(q),bool);result[inside]=mask[q[inside,1],q[inside,0]]>0;return result

def nearest(a,b):
 if not len(a) or not len(b):return np.zeros(len(a),bool)
 return np.array([m[0].distance<=3 for m in cv2.BFMatcher(cv2.NORM_L2).knnMatch(np.float32(a),np.float32(b),k=1)])

def offline_features(ref,pred,gt):
 """Post-hoc GT/mask diagnostics only; never used by predict_variant or a gate."""
 out={};shape=(ref.meta['output_size'],)*2
 for side,points in [(0,pred.keypoints0),(1,pred.keypoints1)]:out.update({f'keypoint{side}_{k}':v for k,v in spatial(points,shape).items()})
 for side,points in [(0,pred.points0),(1,pred.points1)]:out.update({f'match{side}_{k}':v for k,v in spatial(points,shape).items()})
 if gt['no_overlap']:return dict(out,static_match_precision3=None,static_matches_evaluable=0,keypoint_repeatability3=None,keypoints_repeatability_evaluable=0,gt_affine_floor_mean_px=None)
 valid=[];dynamic=[]
 for view in [ref.view_i,ref.view_j]:
  v=cv2.imread(str(ref.folder/f'valid_{view}.png'),0);d=cv2.imread(str(ref.folder/f'dynamic_{view}.png'),0);assert v is not None and d is not None;valid.append(v);dynamic.append(d)
 H=gt['H'];p0,p1=pred.points0,pred.points1;projected=perspective(H,p0)
 eligible=sample(valid[0],p0)&sample(valid[1],p1)&~sample(dynamic[0],p0)&~sample(dynamic[1],p1)&~sample(dynamic[1],projected)
 correct=sample(valid[1],projected)&(np.linalg.norm(projected-p1,axis=1)<=3)
 out['static_matches_evaluable']=int(eligible.sum());out['static_match_precision3']=float(correct[eligible].mean()) if np.any(eligible) else None
 repeat=[];denom=0
 for x,y,matrix,v0,v1,d0,d1 in [(pred.keypoints0,pred.keypoints1,H,valid[0],valid[1],dynamic[0],dynamic[1]),(pred.keypoints1,pred.keypoints0,np.linalg.inv(H),valid[1],valid[0],dynamic[1],dynamic[0])]:
  q=perspective(matrix,x);support=sample(v0,x)&sample(v1,q)&~sample(d0,x)&~sample(d1,q);target=y[sample(v1,y)&~sample(d1,y)]
  r=nearest(q[support],target);repeat.extend(r.tolist());denom+=int(support.sum())
 out['keypoints_repeatability_evaluable']=denom;out['keypoint_repeatability3']=float(np.mean(repeat)) if repeat else None
 X=np.c_[gt['points0'],np.ones(gt['count'])];fit=np.linalg.lstsq(X,gt['points1'],rcond=None)[0];err=np.linalg.norm(X@fit-gt['points1'],axis=1)
 out['gt_affine_floor_mean_px']=float(err.mean());out['gt_affine_floor_rmse_px']=float(np.sqrt(np.mean(err*err)))
 return out

def make_variant(name,max_features):
 d=VARIANTS[name]
 if d['algorithm']=='lightglue':return SuperPointLightGlue(device='cuda',max_features=max_features,detector_threshold=d.get('detector_threshold'))
 return LocalFeatureMatcher(d['algorithm'],max_features=max_features,detector_options=d.get('detector_options'))

def predict_variant(name,matcher,a,b):
 if VARIANTS[name].get('preprocessing'):
  clahe=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8));a=clahe.apply(a);b=clahe.apply(b)
 return matcher.predict(a,b)

def select_dev(summary,config):
 """Selection uses only dev metrics; validation never enters this function."""
 s=summary['by_method'];selected={};validation_variants=set()
 for algorithm in ['sift','orb','lightglue']:
  base=algorithm+'_base';variants=[v for v in VARIANTS if VARIANTS[v]['algorithm']==algorithm];validation_variants.add(base)
  feature=min(variants,key=lambda v:(-s[f'{v}@homography@baseline']['success_5px_rate'],s[f'{v}@homography@baseline']['false_accept_rate'],s[f'{v}@homography@baseline']['runtime_ms_mean']))
  validation_variants.add(feature);ref=s[f'{feature}@homography@baseline'];min_success=ref['success_5px_rate']-config['selection_max_success_loss'];options=[]
  for model in ['homography','affine']:
   for policy in config['geometry_policies']:
    stats=s[f'{feature}@{model}@{policy}']
    if stats['success_5px_rate']<min_success:continue
    wrong=stats['accepted_positive']-stats['decision_counts'].get('success',0);risk=wrong/stats['positive']+stats['false_accept_rate']
    options.append((risk,-stats['success_5px_rate'],stats['runtime_ms_mean'],model,policy))
  choice=min(options);selected[algorithm]={'variant':feature,'model':choice[3],'policy':choice[4],'dev_risk':choice[0],'dev_minimum_success':min_success,'reference':f'{feature}@homography@baseline'}
 return {'selected':selected,'validation_variants':sorted(validation_variants),'rule':config['reliability_selection'],'feature_rule':config['feature_selection'],'geometry_policies':config['geometry_policies'],'immutable_before_validation':True,'hybrid_rule':'ORB-first image-only fallback: run selected SIFT only when selected ORB is not accepted; no GT in selection.'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--split',choices=['dev','val'],required=True);p.add_argument('--config',default='configs/phase25.yaml');p.add_argument('--selection');p.add_argument('--out',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
 data=Path(a.data_root).resolve();root=data/a.split;out=Path(a.out).resolve();cfg=yaml.safe_load(Path(a.config).read_text());protocol=cfg['benchmark'];isolation=json.loads((data/'isolation.json').read_text());assert isolation['accepted'];assert json.loads((root/'ACCEPTED_phase25.json').read_text())['accepted']
 if out.exists():raise FileExistsError(out)
 selection=None
 if a.split=='val':
  if not a.selection:raise ValueError('Validation requires a sealed dev selection')
  selection=json.loads(Path(a.selection).read_text());assert selection['role']=='dev' and selection['immutable_before_validation'];assert selection['config_sha256']==checksum(a.config);assert selection['data_manifest_sha256']==checksum(data/'dev/manifest.jsonl')
  assert all(checksum(path)==digest for path,digest in selection['inference_code_sha256'].items()), 'Inference code changed after dev selection'
  variants=selection['validation_variants']
 else:
  if a.selection:raise ValueError('Dev cannot reuse or inspect validation selections')
  variants=protocol['variants']
 refs=discover(root,require_test=False);assert len(refs)==384 and all(r.test_source==('train' if a.split=='dev' else 'val') for r in refs)
 for record in json.loads((root/'file_checksums.json').read_text()):assert checksum(root/record['path'])==record['sha256']
 metadata={r['sample_id']:r for r in [json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]}
 if a.smoke:
  refs=[next(r for r in refs if r.profile==profile) for profile in sorted({r.profile for r in refs})]
 cv2.setNumThreads(1);torch.set_num_threads(1);torch.manual_seed(protocol['seed']);torch.cuda.manual_seed_all(protocol['seed']);random.seed(protocol['seed']);np.random.seed(protocol['seed']);assert torch.cuda.is_available()
 out.mkdir();(out/'matches').mkdir();provenance={'role':a.split,'smoke':a.smoke,'root':str(root),'manifest_sha256':checksum(root/'manifest.jsonl'),'source_index_sha256':isolation['source_index_sha256'],'config_sha256':checksum(a.config),'config':cfg,'variants':{v:VARIANTS[v] for v in variants},'policy_definitions':POLICIES,'selection':str(Path(a.selection).resolve()) if a.selection else None,'selection_sha256':checksum(a.selection) if a.selection else None,'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'code_files_sha256':{str(p):checksum(p) for p in [Path('sarbench/matchers.py'),Path('sarbench/estimate.py'),Path('sarbench/reliability.py'),Path(__file__)]},'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'opencv':cv2.__version__,'gpu_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],text=True),'dependencies':{d['name']:d['version'] for d in json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True))},'timing':'image decode, model initialization, one warmup per variant, cached-array write and offline GT diagnostics excluded; keypoint metadata transfer included. Model/gate comparisons reuse identical image-only cached matches. Projection diagnostics include both pole and shape, so standalone pole cost is conservatively charged the shared projection component.'}
 save_json(out/'provenance.json',provenance)
 rows=[];features=[]
 for variant in variants:
  matcher=make_variant(variant,protocol['max_features']);warm0,warm1=load_images(refs[0]);predict_variant(variant,matcher,warm0,warm1);torch.cuda.synchronize()
  for idx,ref in enumerate(refs):
   image0,image1=load_images(ref);torch.cuda.synchronize();start=time.perf_counter();pred=predict_variant(variant,matcher,image0,image1);torch.cuda.synchronize();match_ms=1000*(time.perf_counter()-start)
   # Persistence and evaluation occur only after image-only prediction.
   cachefile=out/'matches'/f'{variant}_{ref.folder.name}_{ref.view_i}_{ref.view_j}.npz';np.savez_compressed(cachefile,points0=pred.points0,points1=pred.points1,keypoints0=pred.keypoints0,keypoints1=pred.keypoints1)
   gt=load_gt(ref);diagnostic=offline_features(ref,pred,gt);m=metadata[ref.folder.name]
   feature={'id':ref.id,'variant':variant,'algorithm':VARIANTS[variant]['algorithm'],'profile':ref.profile,'source_scene':ref.meta.get('scene_type','unknown') if ref.dataset!='background' else 'background','scene':ref.scene,'texture_stratum':m['texture_stratum'],'source_texture_grad75':m['source_texture_grad75'],'dataset':ref.dataset,'source_component':ref.source_component,'source_split':ref.test_source,'image_size':image0.shape[0],'no_overlap':gt['no_overlap'],'overlap':gt['overlap'],'n_keypoints0':pred.n_keypoints0,'n_keypoints1':pred.n_keypoints1,'raw_n_matches':len(pred.points0),'feature_match_ms':match_ms,'prediction_reason':pred.reason,'cache_file':str(cachefile),'cache_sha256':checksum(cachefile),**diagnostic}
   features.append(feature)
   for model in ['homography','affine']:
    start=time.perf_counter();est=estimate(pred,image0.shape,model=model,seed=protocol['seed']+idx);estimate_ms=1000*(time.perf_counter()-start);geo=geometry_diagnostics(est,pred,image0.shape,image1.shape,seed=protocol['seed']+idx)
    for policy in protocol['geometry_policies']:
     start=time.perf_counter();filtered=gate(est,geo,policy);gate_ms=1000*(time.perf_counter()-start);settings=POLICIES[policy]
     # Conservative component cost: support/residual calculation charged with full diagnostics except reverse fit.
     extra=0.
     if policy!='baseline' and est.accepted:
      extra=geo.get('projection_diagnostic_ms',0)
      if settings.get('min_hull') or settings.get('cycle'):extra+=max(0,geo.get('extra_diagnostic_ms',0)-geo.get('projection_diagnostic_ms',0)-geo.get('reverse_estimation_ms',0))
      if settings.get('cycle'):extra+=geo.get('reverse_estimation_ms',0)
      extra+=gate_ms
     ev=evaluate(filtered,gt,image0.shape);rows.append({**feature,'folder':str(ref.folder),'method':f'{variant}@{model}@{policy}','model':model,'policy':policy,'view_i':ref.view_i,'view_j':ref.view_j,'num_views':2,'has_ship':any(ref.meta['ship_annotations']),'overlap_bin':('none' if gt['no_overlap'] else '0-0.3' if gt['overlap']<.3 else '0.3-0.5' if gt['overlap']<.5 else '0.5-0.7' if gt['overlap']<.7 else '0.7-1.0'),'accepted':filtered.accepted,'reason':filtered.reason,'error':None,'n_matches':est.n_matches,'n_inliers':est.n_inliers,'inlier_ratio':est.inlier_ratio,'inlier_rmse':est.inlier_rmse,'coverage':est.coverage,'geometry_estimation_ms':estimate_ms,'charged_extra_gate_ms':extra,'gate_decision_ms':gate_ms,'runtime_ms':match_ms+estimate_ms+extra,'candidate_H':json.dumps(est.candidate_H.tolist()) if est.candidate_H is not None else '', 'pred_H':json.dumps(filtered.H.tolist()) if filtered.H is not None else '',**geo,**ev})
   if (idx+1)%64==0:print(f'{a.split} {variant}: {idx+1}/{len(refs)}',flush=True)
  del matcher
  torch.cuda.empty_cache()
 # Strictly image-only ORB-first fallback; record also GT oracle union as a descriptive upper bound.
 if selection:
  lookup={(r['id'],r['variant'],r['model'],r['policy']):r for r in rows};hybrid=[]
  for ref in refs:
   opts={}
   for algorithm in ['orb','sift']:
    s=selection['selected'][algorithm];opts[algorithm]=lookup[(ref.id,s['variant'],s['model'],s['policy'])]
   first,second=opts['orb'],opts['sift'];chosen=first if first['accepted'] else second
   r={**chosen,'method':'orb_then_sift_selected','variant':'orb_then_sift_selected','policy':'selected_fallback','algorithm':'hybrid','runtime_ms':first['runtime_ms']+(second['runtime_ms'] if not first['accepted'] else 0),'feature_match_ms':first['feature_match_ms']+(second['feature_match_ms'] if not first['accepted'] else 0),'oracle_union_success5':bool(first['success_5px'] or second['success_5px']) if not first['no_overlap'] else None,'selected_from':'orb' if first['accepted'] else 'sift'};hybrid.append(r)
  rows.extend(hybrid)
 write_csv(out/'results.csv',rows);write_csv(out/'features.csv',features);summary=summarize(rows);summary['role']=a.split;summary['partial_smoke']=a.smoke;summary['relations_per_configuration']=len(refs);summary['models_reuse_same_feature_cache']=True
 save_json(out/'summary.json',summary)
 if a.split=='dev' and not a.smoke:
  selected=select_dev(summary,protocol);selected.update(role='dev',config_sha256=checksum(a.config),data_manifest_sha256=checksum(root/'manifest.jsonl'),dev_results_sha256=checksum(out/'results.csv'),code_commit=provenance['code_commit'],inference_code_sha256=provenance['code_files_sha256']);save_json(out/'selection.json',selected);print(json.dumps(selected,indent=2))
 if selection:summary['selection']=selection;save_json(out/'summary.json',summary)
 save_json(out/'RUN_COMPLETE.json',{'role':a.split,'partial_smoke':a.smoke,'relations':len(refs),'variants':variants,'rows':len(rows),'features':len(features),'results_sha256':checksum(out/'results.csv'),'feature_csv_sha256':checksum(out/'features.csv'),'exceptions':0})
 print(json.dumps({'out':str(out),'rows':len(rows),'features':len(features),'smoke':a.smoke},indent=2))

if __name__=='__main__':main()
