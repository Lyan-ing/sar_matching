"""Focused geometry/qualification checks; no fake-person submissions or real labels."""
from common import *
from reader import load_images,image_records,evaluation_checks
from gt_qa import support,qualified_documents

def main():
 rr=rows(latest('REAL_PAIR_MANIFEST_*.jsonl'));checks={}
 assert len(rr)==1036 and sum(r['qa_state']=='passed' for r in rr)==1033
 assert sum(r['qa_state']=='quarantined_historical_pair_resampling' for r in rr)==3
 assert not any(r['GT_qualified'] for r in rr)
 t=stamp();input_counts={}
 for split in ['Train','Val']:
  p=latest(f'IMAGE_ONLY_{split}_*.jsonl');rs=list(image_records(p));input_counts[split]=len(rs)
  for r in rs[:3]+rs[-3:]:
   images=load_images(r);assert len(images)==2 and all(im.shape==(512,512) for im in images)
 checks['image_only_whitelist_and_native_shapes']=True
 try:load_images(dict(sample_id='not-a-real-task',image0='',image1='',GeoTransform=[]));raise AssertionError('extra predictor input accepted')
 except ValueError:pass
 try:evaluation_checks(dict(GT_qualified=False));raise AssertionError('unqualified GT accepted')
 except ValueError:pass
 checks['geography_rejected_from_prediction_and_unqualified_GT_rejected']=True
 # Mathematical spatial-support unit arrays only; never written as human point files.
 xy=[[40,40],[470,40],[470,470],[40,470],[256,128],[128,256]]
 ps=[dict(xy0=p,xy1=p,role='check',physical_structure_group='math_group_'+str(i//2)) for i,p in enumerate(xy)]
 views=[dict(native_info=dict(width=512,height=512))]*2
 assert support(ps,views,'primary_patch_sparse')['passed']
 assert not support(ps[:5],views,'primary_patch_sparse')['passed']
 assert support(ps[:4],views,'secondary_local_sparse')['passed'] is False # only two groups
 try:
  from gt_qa import quality
  quality.validate_points([dict(sample_id='math-only',point_id='math-only',xy0=[1,2],xy1=[1,2],role='check',sigma0_px=None,sigma1_px=None)],dict(sample_id='math-only',height0=512,width0=512,height1=512,width1=512));raise AssertionError('missing uncertainty accepted')
 except ValueError:pass
 checks['frozen_512_support_and_no_default_uncertainty']=True
 d=qualified_documents({'package_id':'unassigned'},{'package_id':'unassigned'},{},{},None,None,{'component':'none','image_names':[],'image_sha256':[]})
 assert 'unassigned_real_person' in d;checks['no_unassigned_identity_qualification']=True
 protected=json.loads(latest('PROTECTED_HISTORY_*.json').read_text())
 for p,h in protected.items():assert sha(p)==h,p
 frozen=json.loads((FIXED/'FREEZE_20261014_1810.json').read_text())
 for name,d in frozen['files'].items():assert sha(FIXED/name)==d['sha256'],name
 checks['protected_history_and_C001_C002_C003_unchanged']=True
 # Check source-identity and whole-station isolation, not crop-random partition.
 pools={s:[r for r in rr if r['split']==s] for s in ['Train','Val']}
 for fn in [lambda r:[r['component']],lambda r:r['acquisition_ids'],lambda r:[r['relation_id']]]:
  assert not ({x for r in pools['Train'] for x in fn(r)}&{x for r in pools['Val'] for x in fn(r)})
 checks['relation_acquisition_station_isolation']=True
 out=ROOT/f'DELIVERY_CHECKS_{t}.json';save(out,dict(at=now(),passed=True,checks=checks,input_records=input_counts,real_human_points_created=0,geometry_test_arrays_are_unit_math_only=True,trained_models=0,CUDA_used=False,old_whole_rasters_rehashed=False));print(out,flush=True)

if __name__=='__main__':main()
