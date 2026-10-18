"""Deduplicated actual acquisition relations and existing crops; no new cropping/GT."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter,defaultdict
from itertools import combinations
import json,csv,math,hashlib
import rasterio,cv2,numpy as np
from rasterio.warp import transform
BASE=Path('/home/yl/matching/sar_matching');ROOT=Path(__file__).resolve().parents[1]
T=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
OLD=BASE/'real_sar_pilot/multisource512_20261010_2352'
A=BASE/'RealSAR_AngleExpansion_20261012_1641'
FROZEN=BASE/'RealSAR_Fixed512_20261014_1743/release_20261014_1756'
BINS=['0-15','15-30','30-60','60-90','90-120','>120','unknown']
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def jsonl(p,rs):
 with p.open('x',encoding='utf-8') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def table(p,rs,fields):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rs)
def angle(a,b):
 if a is None or b is None:return None
 return abs((float(a)-float(b)+180)%360-180)
def binof(a):
 if a is None:return 'unknown'
 for lo,hi,label in [(0,15,'0-15'),(15,30,'15-30'),(30,60,'30-60'),(60,90,'60-90'),(90,120.0000001,'90-120')]:
  if lo<=a<hi:return label
 return '>120'
def canonical(site):
 if site in ['South_Dakota','S1_14TPN_PILOT']:return 's1_14TPN_south_dakota'
 return site.lower().replace(',','').replace(' ','_')
def component(site):return 'real_site_'+hashlib.sha256(canonical(site).encode()).hexdigest()[:12]
def marine(site):return site.startswith('Port of ') or site in ['Sunny_Isles','San_Francisco','Kingston_Australia','Melbourne, Australia']
def scene(site):
 if site.startswith('Port of '):return 'port_quay_dense_vessels_context'
 if marine(site):return 'nearshore_or_coastal_urban_context_vessel_presence_unverified'
 return 'other_land_structures'
def role(site):
 return 'research_development' if site in ['Port of Rotterdam, Netherlands','Port of Jebel Ali, United Arab Emirates',
  'Port of Long Beach, California, United States','Denver_Mountains','South_Dakota','Bingham Copper Mine','Melbourne, Australia'] else 'candidate_validation'

def main():
 old=rows(OLD/'dataset_index.jsonl');q=rows(A/'review/candidate_regions.jsonl');inventory=rows(A/'acquisition_inventory.jsonl')
 lookup={r['acquisition_id']:r for r in inventory};relations={};crops=[];header_failures=[]
 def relation(site,sensor,ids,views,az,aliases,original_role):
  key=tuple(sorted(ids))
  if key not in relations:
   h=hashlib.sha256('|'.join(key).encode()).hexdigest()[:16]
   relations[key]=dict(relation_id='R_'+h,site=site,sensor=sensor,canonical_component=component(site),
    original_component_aliases=[],acquisition_ids=list(key),view_metadata=views,azimuth_delta_deg=az,angle_bin=binof(az),
    relation_id_aliases=[],original_split_roles=[],new_manifest_role=role(site),
    scenario_context=scene(site),maritime_context=marine(site),vessel_presence_verified=False,
    physical_correspondence_status='no_independent_GT_in_workspace',independent_GT=False,GT_qualified=False,
    local_source_files=[],native_source_samples=[],fixed512_codes=[],existing512_samples=[])
  r=relations[key]
  for name in aliases:
   if name not in r['relation_id_aliases']:r['relation_id_aliases'].append(name)
  if original_role not in r['original_split_roles']:r['original_split_roles'].append(original_role)
  if r['azimuth_delta_deg'] is None and az is not None:r.update(azimuth_delta_deg=az,angle_bin=binof(az))
  return r
 for x in old:
  ids=[x['reference_acquisition_id'],x['moving_acquisition_id']]
  az=angle(x['reference_azimuth_degrees'],x['moving_azimuth_degrees'])
  views=[dict(acquisition_id=ids[i],utc=x[p+'_utc'],azimuth_degrees=x[p+'_azimuth_degrees'],
    incidence_degrees=x[p+'_incidence_degrees'],asset_url=x[p+'_asset_url'],platform=x[p+'_platform']) for i,p in enumerate(['reference','moving'])]
  r=relation(x['site_name'],x['sensor'],ids,views,az,[x['pair_id']],x['split'])
  if x['component'] not in r['original_component_aliases']:r['original_component_aliases'].append(x['component'])
  sid=x['pair_id']+'__'+x['chip_id'];r['existing512_samples'].append(sid)
  meta_path=Path(x['folder'])/'metadata.json';meta=json.loads(meta_path.read_text())
  paths=[x['reference_native_geotiff'],x['moving_native_geotiff']];imgs=[x['reference_native_png'],x['moving_native_png']]
  infos=[]
  for i,p in enumerate(paths):
   with rasterio.open(p) as d:
    assert (d.width,d.height)==(512,512)
    infos.append(dict(native_tif=p,width=d.width,height=d.height,crs=str(d.crs),affine=list(d.transform)[:6],
     parent_product_origin_xy=meta['reference_window_xy'] if i==0 else meta['moving_native_window_xy']))
  crops.append(dict(sample_id=sid,relation_id=r['relation_id'],canonical_component=r['canonical_component'],
   source_kind='existing191_native512',site=r['site'],new_manifest_role=r['new_manifest_role'],original_split=x['split'],
   azimuth_delta_deg=az,angle_bin=binof(az),scenario_context=r['scenario_context'],views=infos,
   image0=imgs[0],image1=imgs[1],source_metadata=str(meta_path),allow_supervised_training=False,
   independent_GT=False,GT_qualified=False,physical_correspondence_status='unconfirmed_existing_real_pair'))
 for x in q:
  views=x['acquisition_metadata']['views'];ids=[v['acquisition_id'] for v in views]
  r=relation(x['site_name'],x['sensor'],ids,views,angle(*[v.get('azimuth_degrees') for v in views]),[x['pair_id']],x['split'])
  if x['component'] not in r['original_component_aliases']:r['original_component_aliases'].append(x['component'])
  r['native_source_samples'].append(x['sample_id'])
  for v in x['views']:
   with rasterio.open(v['native_tif']) as d:assert (d.width,d.height)==(v['native_info']['width'],v['native_info']['height'])
   if v['native_tif'] not in r['local_source_files']:r['local_source_files'].append(v['native_tif'])
 # Three actual prior pilot relations, not copies of old191. Identity is in original SAFE metadata.
 pilot=rows(OLD/'pilot_512/candidates.jsonl')
 for x in pilot:
  ids=[x['reference_acquisition_id'],x['moving_acquisition_id']];views=[lookup.get(cid,dict(acquisition_id=cid,utc=x[p+'_utc'])) for cid,p in zip(ids,['reference','moving'])]
  r=relation('South_Dakota','Sentinel-1',ids,views,None,[x['pair_id']],x['split'])
  paths=[x['reference_path'],x['moving_path']]
  for p in paths:
   assert Path(p).is_file()
   with rasterio.open(p) as d:assert (d.width,d.height)==(512,512)
  sid=x['pair_id']+'__'+x['chip_id'];r['existing512_samples'].append(sid)
  crops.append(dict(sample_id=sid,relation_id=r['relation_id'],canonical_component=r['canonical_component'],site='South_Dakota',
   source_kind='prior_pilot_native512_provider_RTC_grid',new_manifest_role=role('South_Dakota'),original_split=x['split'],
   azimuth_delta_deg=None,angle_bin='unknown',scenario_context='other_land_structures',image0=paths[0],image1=paths[1],
   source_metadata=str(OLD/'pilot_512/candidates.jsonl'),allow_supervised_training=False,independent_GT=False,GT_qualified=False,
   physical_correspondence_status='same_orbit_repeat_pass_no_verified_azimuth_no_GT'))
 fixed= json.loads(next((FROZEN/'custodian').glob('candidate_sources_*.json')).read_text())
 for x in fixed:
  ids=[v['acquisition']['acquisition_id'] for v in x['views']];r=relations[tuple(sorted(ids))];r['fixed512_codes'].append(x['candidate_id'])
  crops.append(dict(sample_id=x['candidate_id'],relation_id=r['relation_id'],canonical_component=r['canonical_component'],
   site=x['site'],source_kind='existing_FROZEN_512_do_not_recrop_or_repackage',new_manifest_role=r['new_manifest_role'],
   original_split='diagnostic_development_seen',azimuth_delta_deg=x['azimuth_delta_deg'],angle_bin=binof(x['azimuth_delta_deg']),
   scenario_context=scene(x['site']),image0=x['views'][0]['raw_DN_png'],image1=x['views'][1]['raw_DN_png'],
   source_metadata=str(next((FROZEN/'custodian').glob('candidate_sources_*.json'))),views=x['views'],
   allow_supervised_training=False,independent_GT=False,GT_qualified=False,physical_correspondence_status='frozen_unannotated_candidate'))
 # Reuse existing pre-model human-screening shortlist; no new crops or blind packages.
 curated_ids={r['sample_id'] for r in json.loads((BASE/'RealSAR_PairingAudit_20261011_2349/selection.json').read_text())['candidates']}
 curated_ids|={'C001','C002','C003','Umbra_0032__chip_000'}
 curated=[]
 for c in crops:
  if c['sample_id'] in curated_ids:
   c=dict(c,candidate_evidence='frozen assisted preparation, not human point GT' if c['sample_id'].startswith('C0') else 'existing static-structure visual search, not independently confirmed',
    candidate_stage='fixed512_ready_for_real_people' if c['sample_id'].startswith('C0') else 'existing512_exploratory_search_only')
   curated.append(c)
   r=next(r for r in relations.values() if r['relation_id']==c['relation_id']);r.setdefault('curated512_samples',[]).append(c['sample_id'])
 allrel=list(relations.values());allrel.sort(key=lambda r:(r['site'],r['relation_id']))
 jsonl(ROOT/f'REAL_ACQUISITION_RELATIONS_{T}.jsonl',allrel)
 jsonl(ROOT/f'REAL512_EXISTING_CROPS_{T}.jsonl',crops)
 jsonl(ROOT/f'REAL512_CANDIDATES_{T}.jsonl',curated)
 for role_name in ['research_development','candidate_validation']:
  jsonl(ROOT/f'REAL512_{role_name}_UNLABELED_{T}.jsonl',[x for x in crops if x['new_manifest_role']==role_name])
  jsonl(ROOT/f'REAL512_{role_name}_SUPERVISED_ELIGIBLE_{T}.jsonl',[])
 # Identical site/acquisition group cannot cross the new descriptive manifest roles.
 ds={r['canonical_component'] for r in allrel if r['new_manifest_role']=='research_development'}
 vs={r['canonical_component'] for r in allrel if r['new_manifest_role']=='candidate_validation'};assert not ds&vs
 alias_sites=defaultdict(set)
 for r in allrel:
  for alias in r['original_component_aliases']:alias_sites[alias].add(r['site'])
 angular=[]
 for b in BINS:
  rr=[r for r in allrel if r['angle_bin']==b];mr=[r for r in rr if r['maritime_context']];cc=[c for c in curated if c['angle_bin']==b];ff=[c for c in cc if c['sample_id'] in ['C001','C002','C003']]
  angular.append(dict(angle_bin=b,actual_acquisition_relations=len(rr),all_source_components=len({r['canonical_component'] for r in rr}),
   maritime_relations=len(mr),maritime_source_components=len({r['canonical_component'] for r in mr}),
   existing512_crop_pairs=sum(c['angle_bin']==b for c in crops),curated_search_candidates=len(cc),
   curated_source_components=len({c['canonical_component'] for c in cc}),fixed512_prepared_candidates=len(ff),independent_GT_evaluable=0))
 table(ROOT/f'REAL_ANGLE_COVERAGE_{T}.csv',angular,list(angular[0]))
 # Only public catalog candidates: no raster access. Prefer missing maritime site/bin support.
 retained={cid for r in allrel for cid in r['acquisition_ids']};site_inventory=defaultdict(list)
 for r in inventory:
  if r['site_name'].startswith('Port of ') and r.get('geometry') and r.get('azimuth_degrees') is not None:site_inventory[r['site_name']].append(r)
 proposals=[]
 for site,rs in site_inventory.items():
  lon,lat=np.mean(np.asarray(rs[0]['geometry']['coordinates'][0])[:,:2],axis=0);zone=int((lon+180)//6)+1;epsg=(32600 if lat>=0 else 32700)+zone
  geoms={}
  for r in rs:
   try:
    assert r['geometry']['type']=='Polygon'
    coords=np.asarray(r['geometry']['coordinates'][0])[:,:2]
    x,y=transform('EPSG:4326',f'EPSG:{epsg}',coords[:,0].tolist(),coords[:,1].tolist())
    geom=np.c_[x,y];assert np.isfinite(geom).all();geoms[r['acquisition_id']]=geom
   except Exception:continue
  for a,b in combinations(rs,2):
   key=tuple(sorted([a['acquisition_id'],b['acquisition_id']]))
   if key in relations or any(cid not in geoms for cid in key):continue
   delta=angle(a['azimuth_degrees'],b['azimuth_degrees']);bucket=binof(delta)
   if bucket not in BINS[:5]:continue
   try:
    days=abs((datetime.fromisoformat(a['datetime'].replace('Z','+00:00'))-datetime.fromisoformat(b['datetime'].replace('Z','+00:00'))).total_seconds())/86400
   except ValueError:continue
   if days>2:continue
   pa,pb=[geoms[cid] for cid in key];origin=np.mean(pa,axis=0)
   hulls=[cv2.convexHull((poly-origin).astype('float32')) for poly in [pa,pb]]
   area,_=cv2.intersectConvexConvex(*hulls);den=min(cv2.contourArea(h) for h in hulls)
   if den<=0:continue
   frac=float(max(0,area)/den)
   if frac<.5:continue
   local=sum(cid in retained for cid in key)
   proposals.append(dict(site=site,angle_bin=bucket,azimuth_delta_deg=delta,time_gap_days=days,
    acquisition_ids=list(key),existing_local_acquisition_count=local,
    product_footprint_intersection_over_smaller_area=frac,
    metadata_only_common_scene_evidence=True,physical_common_structures_confirmed=False,
    assets=[dict(acquisition_id=r['acquisition_id'],utc=r['datetime'],asset_url=r['asset_url'],metadata_path=r['metadata_path'],
      azimuth_degrees=r['azimuth_degrees'],incidence_degrees=r.get('incidence_degrees'),task_id=r.get('task_id')) for r in [a,b]],
    action='reuse_local_assets_only' if local==2 else 'proposed_targeted_source_supplement_requires_separate_download_decision',
    no_download_performed=True))
 chosen=[]
 for bucket in BINS[:5]:
  pool=sorted([p for p in proposals if p['angle_bin']==bucket],key=lambda p:(p['time_gap_days'],p['existing_local_acquisition_count']))
  sites=set()
  for p in pool:
   if p['site'] in sites:continue
   chosen.append(p);sites.add(p['site'])
   if len(sites)>=2:break
 save(ROOT/f'TARGETED_SOURCE_PROPOSALS_{T}.json',chosen)
 summary=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),old512_pairs=len(old),native_relations=len(q),
  additional_prior_pilot_pairs=len(pilot),unique_acquisition_relations=len(allrel),unique_acquisitions=len(retained),
  canonical_site_components=len({r['canonical_component'] for r in allrel}),existing512_pairs_including_frozen=len(crops),
  crop_view_instances=2*len(crops),metadata_inventory_entries=len(inventory),
  metadata_only_inventory_flagged=sum(r.get('raster_status')=='METADATA_VERIFIED_RASTER_NOT_READ' for r in inventory),
  maritime_relations=sum(r['maritime_context'] for r in allrel),maritime512_pairs=sum(marine(c['site']) for c in crops),
  curated_search_candidates=len(curated),curated_maritime_context_candidates=sum(marine(c['site']) for c in curated),
  curated_source_components=len({c['canonical_component'] for c in curated}),
  fixed512_ready_for_people=3,independent_GT_evaluable=0,supervised_real_train_eligible=0,supervised_real_val_eligible=0,
  prospective_component_roles={name:dict(relations=sum(r['new_manifest_role']==name for r in allrel),components=len({r['canonical_component'] for r in allrel if r['new_manifest_role']==name}),crop_pairs=sum(c['new_manifest_role']==name for c in crops)) for name in ['research_development','candidate_validation']},
  role_component_intersection=[],roles_are_descriptive_no_prior_split_or_model_unseen_claim=True,
  original_component_alias_collisions={k:sorted(v) for k,v in alias_sites.items() if len(v)>1},angle_coverage=angular,
  fixed_candidates_unchanged=True,new_crops_created=0,new_blind_packages_created=0,new_real_GT_created=0,
  targeted_source_proposal_count=len(chosen),geographic_overlap_not_GT=True,no_downloads=True,
  no_vessel_presence_inferred_from_site_name=True)
 save(ROOT/f'REAL_RESOURCE_SUMMARY_{T}.json',summary)
 table(ROOT/f'REAL512_CANDIDATES_{T}.csv',[{k:c.get(k) for k in ['sample_id','site','relation_id','canonical_component','new_manifest_role','azimuth_delta_deg','angle_bin','candidate_stage','physical_correspondence_status','image0','image1','GT_qualified']} for c in curated],
  ['sample_id','site','relation_id','canonical_component','new_manifest_role','azimuth_delta_deg','angle_bin','candidate_stage','physical_correspondence_status','image0','image1','GT_qualified'])
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
