"""Read-only input reuse, display thumbnails, and a separate single-expert protocol."""
import sys,json,hashlib,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from affine import Affine
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;DATA=BASE/'RealSAR_Multiview512_20261015_1534'
sys.path.insert(0,str(DATA/'code'))
from common import stamp,now,rows,sha,save,jsonl

def main():
 t=stamp();entry=DATA/'REAL_MULTIVIEW_512_ENTRYPOINTS_20261015_1942.json';e=json.loads(entry.read_text());pairs=[p for p in rows(e['pair_manifest']) if p['qa_state']=='passed'];assert len(pairs)==1033
 acqs={a['acquisition_id']:a for a in rows(e['acquisition_manifest'])};frames={};grids=defaultdict(list)
 def frame(cid,path,z,origin,display=None,valid=None,context=False):
  path=str(Path(path).resolve());assert Path(path).is_file() and origin is not None
  parent=tuple(Affine(*z['transform'])*Affine.translation(-origin[0],-origin[1]))[:6]
  grid=next((gid for crs,mat,gid in grids[cid] if crs==z['crs'] and np.allclose(mat,parent,rtol=0,atol=1e-8)),None)
  if grid is None:grid='G'+hashlib.sha256((cid+z['crs']+repr(parent)).encode()).hexdigest()[:16];grids[cid].append((z['crs'],parent,grid))
  fid='F'+hashlib.sha256((cid+path+repr(origin)).encode()).hexdigest()[:18]
  if fid not in frames:frames[fid]=dict(frame_id=fid,acquisition_id=cid,grid_id=grid,raw_path=path,display=str(display) if display and Path(display).is_file() else None,valid_mask=str(valid) if valid else None,width=z['width'],height=z['height'],origin_xy=origin,native_info=z,units=acqs[cid].get('input_units','linear_amplitude'),large_context=context,metadata=acqs[cid].get('metadata_path'),raw_read='native Window only; no reproject')
  return fid
 tasks=[];thumb=ROOT/f'thumbnails_{t}';thumb.mkdir()
 for p in pairs:
  fids=[frame(v['acquisition_id'],v['path'],v['native_info'],v['product_origin_xy'],p[f'image{i}'],v.get('valid_mask')) for i,v in enumerate(p['views'])]
  dst=thumb/f'{p["sample_id"]}_{t}.jpg';sheet=Image.new('L',(386,192),0)
  for side in [0,1]:
   with Image.open(p[f'image{side}']) as im:im=im.convert('L');im.thumbnail((192,192),Image.Resampling.NEAREST);sheet.paste(im,(side*194,0))
  sheet.save(dst,quality=84)
  tasks.append(dict(sample_id=p['sample_id'],relation_id=p['relation_id'],site=p['site'],component=p['component'],split=p['split'],angle=p['azimuth_delta_deg'],angle_bin=p['angle_bin'],scene=p['scene_class'],AI_class=p['annotatability'],AI_hint=p.get('annotatability_basis',''),AI_reviewed=bool(p.get('visible_vessels',{}).get('AI_reviewed')),frame_ids=fids,thumbnail=str(dst),source_manifest=e['pair_manifest'],acquisition_ids=p['acquisition_ids'],historical_exposure=p.get('prior_geographic_or_algorithm_exposure',False)))
 assets=rows(DATA/'LOCAL_NATIVE_ASSETS_20261015_1538.jsonl')+[json.loads(p.read_text()) for p in (DATA/'acquisitions').glob('*/ASSET_*.json')]
 for a in assets:
  if a['acquisition_id'] not in {c for p in tasks for c in p['acquisition_ids']}:continue
  path=Path(a['path']);display=a.get('display')
  if not display:
   for q in [path.parent/'native_db.png',path.with_name(path.stem.replace('_float','_db')+'.png')]:
    if q.is_file():
     with Image.open(q) as im:
      if im.size==(a['native_info']['width'],a['native_info']['height']):display=str(q);break
  frame(a['acquisition_id'],path,a['native_info'],a['product_origin_xy'],display,context=True)
 byacq=defaultdict(list)
 for f in frames.values():byacq[f['acquisition_id']].append(f)
 def native_default(crop):
  f=frames[crop];cx,cy=np.array(f['origin_xy'])+255.5
  available=[v for v in byacq[f['acquisition_id']] if v['grid_id']==f['grid_id'] and v['origin_xy'][0]<=cx<v['origin_xy'][0]+v['width'] and v['origin_xy'][1]<=cy<v['origin_xy'][1]+v['height']]
  return max(available,key=lambda v:v['width']*v['height'])['frame_id']
 for p in tasks:
  p['native_frame_ids']=[native_default(f) for f in p['frame_ids']]
  p['native_context_available']=all(frames[f]['width']>512 or frames[f]['height']>512 for f in p['native_frame_ids'])
 # Model-free review priority. Val B first, then other B; no matcher used.
 tasks.sort(key=lambda p:(p['AI_class']!='B',not(p['split']=='Val' and p['AI_class']=='B'),p['AI_class']=='C',not p['AI_reviewed'],p['split']!='Val',p['site'],p['angle'] if p['angle'] is not None else 999,p['sample_id']))
 pilot=[];seen=set()
 for p in tasks:
  if p['AI_class']=='B' and p['relation_id'] not in seen:pilot.append(p['sample_id']);seen.add(p['relation_id'])
  if len(pilot)==30:break
 assert len(pilot)==30
 for i,p in enumerate(tasks):p['priority']=i;p['pilot30']=p['sample_id'] in pilot
 oldprotocol=Path(e['original_frozen_protocol']);old=json.loads(oldprotocol.read_text())
 protocol=dict(protocol_id='REAL512_SINGLE_EXPERT_V1',frozen_before_actual_single_expert_annotation=True,created_at=now(),annotator='user only; no identity registration or second person required',evidence='single-expert physically confirmed correspondences; auxiliary metadata allowed and recorded',GT_qualified_always=False,independent_two_person_GT=False,coordinate_convention=old['coordinate_convention'],primary_patch_sparse=old['primary_patch_sparse'],secondary_local_sparse=old['secondary_local_sparse'],uncertainty=dict(thresholds_px=[3,5,10],fraction=1/3,rule='U=hypot(sigma0,sigma1); when actual repeat exists use max(original sigma,repeat sigma,measured repeat disagreement) per side; no automatic average/default sigma'),geometry=old['geometry'],delayed_repeat=dict(optional=True,minimum_hours=24,same_person=True,first_coordinates_hidden=True,independent_two_person=False),native_reuse=dict(same_acquisition_and_product_grid_only=True,both_crop_extents_and_masks=True,raw_support_values_must_match=True,no_geo_projection=True,no_new_human_click_credit=True),preserve_all_actual_points=True,physical_eligibility='explicit same static ground structure, visible both, nonempty type/group/reason, medium/high confidence, explicit finite uncertainty, all five risks explicitly no',planar_geometry='expert must explicitly confirm local2D applicability; not established by a fitted H or RANSAC',prediction='images only; check points never fit or initialize prediction; point/local/full/no_reliable/pending reported separately',old_protocol_path=str(oldprotocol),old_protocol_sha256=sha(oldprotocol),split_preserved=True)
 index=ROOT/f'TASK_INDEX_{t}.json';save(index,dict(at=now(),tasks=tasks,frames=frames,pilot30=pilot))
 protocolpath=ROOT/f'SINGLE_EXPERT_PROTOCOL_{t}.json';save(protocolpath,protocol)
 protected={str(p):sha(p) for p in [entry,Path(e['pair_manifest']),Path(e['acquisition_manifest']),oldprotocol,Path(e['old_freeze']),BASE/'SyntheticSAR_V22_20261015_1352/DATASET_ENTRYPOINTS_V22_20261015_1434.json']}
 save(ROOT/f'PROTECTED_INPUTS_{t}.json',protected)
 config=dict(created_at=now(),task_index=str(index),protocol=str(protocolpath),data_entrypoint=str(entry),old_source_pair_manifest=e['pair_manifest'],cache_source=str(BASE/'RealSAR_GeoAssist_20261012_1917/display_cache.json'),observations=str(ROOT/'observations'),exports=str(ROOT/'exports'),port=8774,operator='user_single_expert',training=False,GPU=False)
 save(ROOT/f'CONFIG_{t}.json',config)
 save(ROOT/f'PREPARATION_{t}.json',dict(at=now(),tasks=len(tasks),frames=len(frames),high_priority_B=sum(p['AI_class']=='B' for p in tasks),pilot30=pilot,Native_context_pairs=sum(p['native_context_available'] for p in tasks),Native_context_relations=len({p['relation_id'] for p in tasks if p['native_context_available']}),relations=len({p['relation_id'] for p in tasks}),split=dict(Counter(p['split'] for p in tasks)),new_SAR_crops=0,old_files_modified=0,actual_human_points=0))
 print('PREPARED',len(tasks),'B40 pilot30; large Native pairs',sum(p['native_context_available'] for p in tasks),'CONFIG',ROOT/f'CONFIG_{t}.json',flush=True)

if __name__=='__main__':main()
