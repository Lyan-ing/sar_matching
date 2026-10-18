"""Versioned all-vessel static supervision; original generated assets remain intact."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json,sys,hashlib
import numpy as np,cv2
BASE=Path('/home/yl/matching/sar_matching');ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'MaritimeSARRegDataset'))
from sarreg.labels import annotation_mask
from sarreg.io import read_gray,write_png
from sarreg.geometry import transform_points
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def jsonl(p,rs):
 with p.open('x',encoding='utf-8') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)

def one(item):
 row,source,out=item;cv2.setNumThreads(1);meta=json.loads(Path(row['metadata']).read_text());size=row['width'];folder=Path(out)/row['split']/row['sample_id'];folder.mkdir(parents=True,exist_ok=False)
 union=np.zeros(tuple(meta['source_shape_hw']),np.uint8);kinds=Counter()
 for a in source['annotations']:
  mask,kind=annotation_mask(union.shape,a);union|=mask;kinds[kind]+=1
 masks=[];ships=[];changes=[]
 for i in [0,1]:
  H=np.asarray(meta['H_source_to_view'][i]);full=cv2.warpPerspective(union,H,(size,size),flags=cv2.INTER_NEAREST)
  full|=read_gray(Path(row['metadata']).parent/f'ships_{i}.png')
  valid=read_gray(row['valid_masks'][i])>0;dynamic=read_gray(Path(row['metadata']).parent/f'dynamic_{i}.png')>0
  full[~valid]=0
  # Full original support is retained even when instance visibility is below the old engine threshold.
  static=valid&~(cv2.dilate(full,np.ones((11,11),np.uint8))>0)&~dynamic
  prior=read_gray(row['static_supervision_masks'][i])>0
  assert not np.any(static&~prior) and not np.any(static&(full>0)) and not np.any(static&dynamic)
  changes.append(int((prior&~static).sum()));masks.append(static);ships.append(full)
  write_png(folder/f'static_all_vessels_{i}.png',static.astype('uint8')*255)
  write_png(folder/f'all_annotated_vessels_{i}.png',full)
 with np.load(Path(row['metadata']).parent/'matches_0_1.npz') as z:a=z['points_i'];b=z['points_j']
 keep=np.ones(len(a),bool)
 for xy,mask in zip([a,b],masks):
  ix=np.rint(xy).astype(int);keep&=mask[ix[:,1],ix[:,0]]
 a,b=a[keep],b[keep];H=np.asarray(row['H_0_to_1'])
 assert not len(a) or np.max(np.linalg.norm(transform_points(H,a)-b,axis=1))<1e-3
 for xy,mask in zip([a,b],masks):
  ix=np.rint(xy).astype(int);assert np.all(mask[ix[:,1],ix[:,0]])
 np.savez_compressed(folder/'matches_static_0_1.npz',points_i=a,points_j=b)
 r=dict(row,previous_static_supervision_masks=row['static_supervision_masks'],previous_static_correspondences=row['static_correspondences'],
  static_supervision_masks=[str(folder/f'static_all_vessels_{i}.png') for i in [0,1]],
  all_vessel_masks=[str(folder/f'all_annotated_vessels_{i}.png') for i in [0,1]],
  static_correspondences=str(folder/'matches_static_0_1.npz'),static_check_count=len(a),
  static_fractions=[float(m.mean()) for m in masks],all_source_annotation_support_including_partial_vessels_excluded=True,
  static_supervision_QA_passed=True,additional_static_pixels_excluded=changes,
  requires_variable_size_or_shape_bucketing=True,images_unchanged=True,
  allow_static_descriptor_positive_supervision=(len(a)>=16 and row['profile']!='none'),
  original_generation_QA_passed=row['QA_passed'])
 return r

def main():
 T=stamp();gen=next(ROOT.glob('synthetic_materialized_*'));source=rows(next(gen.glob('SOURCES_SELECTED_*.jsonl')));by={(r['dataset'],r['source_id']):r for r in source}
 data=rows(next(gen.glob('train_all_manifest_*.jsonl')))+rows(next(gen.glob('val_all_manifest_*.jsonl')))
 out=ROOT/f'synthetic_static_supervision_v2_{T}';out.mkdir(exist_ok=False)
 with ProcessPoolExecutor(max_workers=4) as pool:final=list(pool.map(one,[(r,by[(r['source_dataset'],r['source_id'])],str(out)) for r in data],chunksize=8))
 end=stamp()
 for split in ['train','val']:
  for kind,rr in [('parent_core',[r for r in final if r['split']==split and r['construction_pool']=='parent_component_core']),
                  ('ssdd_extension',[r for r in final if r['split']==split and r['construction_pool']!='parent_component_core']),
                  ('all_known_components',[r for r in final if r['split']==split])]:
   jsonl(ROOT/f'SYNTHETIC_{split}_{kind}_manifest_{end}.jsonl',rr)
 components={s:{r['known_component'] for r in final if r['split']==s} for s in ['train','val']}
 assert not components['train']&components['val'] and len(final)==5961
 summary=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),all_pairs=len(final),view_instances=2*len(final),
  parent_core_pairs=sum(r['construction_pool']=='parent_component_core' for r in final),
  parent_core_view_instances=2*sum(r['construction_pool']=='parent_component_core' for r in final),
  static_QA_passed_count=sum(r['static_supervision_QA_passed'] for r in final),
  pairs_with_additional_partial_vessel_exclusion=sum(any(r['additional_static_pixels_excluded']) for r in final),
  static_descriptor_positive_pairs=sum(r['allow_static_descriptor_positive_supervision'] for r in final),
  training_only_masks_never_image_input=True,train_val_components_overlap=[],old_data_and_old_masks_modified=False,
  actual_image_views_not_regenerated=True,manifest_stamp=end,supervision_folder=str(out),real_GT_created=0)
 save(ROOT/f'STATIC_SUPERVISION_VERIFICATION_{end}.json',summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
