"""Release accounting only; no new imagery, annotations, or changed frozen samples."""
from common import *
from collections import Counter,defaultdict

def counts(rr):
 return dict(pairs=len(rr),relations=len({r['relation_id'] for r in rr}),acquisitions=len({c for r in rr for c in r['acquisition_ids']}),site_components=len({r['component'] for r in rr}),AI_local_candidates=sum(r['annotatability']=='B' for r in rr),independent_GT=0)

def main():
 t=stamp();pairpath=latest('REAL_PAIR_MANIFEST_*.jsonl');allrr=rows(pairpath);rr=[r for r in allrr if r['qa_state']=='passed']
 aa=rows(ROOT/'REAL_ACQUISITION_MANIFEST_20261015_1623.jsonl');amap={a['acquisition_id']:a for a in aa}
 views=defaultdict(list);relations=defaultdict(list)
 for r in rr:
  relations[r['relation_id']].append(r)
  for side,v in enumerate(r['views']):views[v['acquisition_id']].append((r,side,v))
 parent={c:c for c in views}
 def find(c):
  while parent[c]!=c:parent[c]=parent[parent[c]];c=parent[c]
  return c
 for r in rr:parent[find(r['acquisition_ids'][0])]=find(r['acquisition_ids'][1])
 graph=defaultdict(list)
 for c in views:graph[find(c)].append(c)
 graphid={c:'acqgraph_'+hashlib.sha256('|'.join(sorted(cs)).encode()).hexdigest()[:12] for cs in graph.values() for c in cs}
 duplicate=[];spatial=[];unique_windows=set();cross_grid=0
 for cid,vv in views.items():
  for i,(r,side,v) in enumerate(vv):
   z=v['native_info'];o=v['product_origin_xy'];ta=Affine(*z['transform'])*Affine.translation(-o[0],-o[1])
   # A footprint key detects exact reuse even when file paths/display versions differ.
   unique_windows.add((cid,z['crs'],*tuple(round(x,10) for x in z['transform']),z['width'],z['height']))
   for p,ps,pv in vv[:i]:
    if r['relation_id']==p['relation_id']:continue
    pz=pv['native_info'];po=pv['product_origin_xy'];tb=Affine(*pz['transform'])*Affine.translation(-po[0],-po[1])
    same=z['crs']==pz['crs'] and np.allclose(tuple(ta),tuple(tb),rtol=0,atol=1e-8)
    if same:overlap=roi_overlap([*o,z['width'],z['height']],[*po,pz['width'],pz['height']])
    else:
     cross_grid+=1;crs=metric_crs(z);pa,pb=polygon(z,crs),polygon(pz,crs);inter=poly_area(poly_intersection(pa,pb));overlap=inter/max(1e-12,poly_area(pa)+poly_area(pb)-inter)
    if overlap>.2:
     d=dict(acquisition_id=cid,sample_a=r['sample_id'],side_a=side,relation_a=r['relation_id'],sample_b=p['sample_id'],side_b=ps,relation_b=p['relation_id'],source_window_a=[*o,z['width'],z['height']],source_window_b=[*po,pz['width'],pz['height']],native_window_iou=overlap,same_product_grid=same,split=r['split'],same_split=r['split']==p['split'],meaning='shared acquisition across distinct relations; correlated observations, not independent examples')
     spatial.append(d)
     if overlap>1-1e-8:duplicate.append(d)
 jsonl(ROOT/f'CROSS_RELATION_WINDOW_DEPENDENCY_{t}.jsonl',spatial)
 angle=[]
 for split in ['All','Train','Val']:
  for b in ['0-15','15-30','30-60','60-90','90-120','>120','unknown']:
   z=[r for r in rr if (split=='All' or r['split']==split) and r['angle_bin']==b]
   angle.append(dict(split=split,angle_bin=b,**counts(z)))
 table(ROOT/f'ANGLE_COVERAGE_{t}.csv',angle,['split','angle_bin',*counts(rr)])
 site=[]
 for component_id in sorted({r['component'] for r in rr}):
  z=[r for r in rr if r['component']==component_id]
  site.append(dict(component=component_id,site=z[0]['site'],split=z[0]['split'],**counts(z),acquisition_graph_components=len({graphid[c] for r in z for c in r['acquisition_ids']}),angle_bins=';'.join(sorted({r['angle_bin'] for r in z})),maritime_source_context=any(r['maritime_source_context'] for r in z),human_confirmed_vessel_pairs=0))
 table(ROOT/f'SOURCE_COMPONENT_COUNTS_{t}.csv',site,list(site[0]))
 acq_rows=[]
 for c,vv in sorted(views.items()):
  a=amap[c];acq_rows.append(dict(acquisition_id=c,provider=a['provider'],site=a['site_name'],component=a['component'],split=a['split'],acquisition_graph_component=graphid[c],view_instances=len(vv),relations=len({r['relation_id'] for r,_,_ in vv}),image_SHA_identities=len({v['algorithm_input_sha256'] for _,_,v in vv}),acquired_at=a['datetime'],vendor_native_azimuth=a.get('vendor_native_azimuth_degrees'),target_LOS_azimuth=a.get('radar_los_azimuth_degrees'),platform_heading=a.get('platform_heading_degrees'),incidence=a.get('incidence_degrees'),orbit_direction=a.get('orbit_direction'),observation_direction=a.get('observation_direction'),polarization=json.dumps(a.get('polarizations')),resolution=json.dumps(a.get('resolution')),original_metadata=a.get('metadata_path'),product_url=a.get('asset_url')))
 table(ROOT/f'ACQUISITION_SOURCE_RELATION_{t}.csv',acq_rows,list(acq_rows[0]))
 relation_rows=[]
 for rid,z in sorted(relations.items()):
  r=z[0];a,b=[amap[c] for c in r['acquisition_ids']]
  relation_rows.append(dict(relation_id=rid,site=r['site'],component=r['component'],split=r['split'],acquisition_graph_component=graphid[a['acquisition_id']],acquisition0=a['acquisition_id'],acquisition1=b['acquisition_id'],time0=a['datetime'],time1=b['datetime'],time_gap_days=abs((utc(a['datetime'])-utc(b['datetime'])).total_seconds())/86400,angle_degrees=r['azimuth_delta_deg'],angle_bin=r['angle_bin'],pairs=len(z),AI_B_candidates=sum(p['annotatability']=='B' for p in z),GT_qualified=0))
 table(ROOT/f'REAL_RELATION_COUNTS_{t}.csv',relation_rows,list(relation_rows[0]))
 scene=[]
 for split in ['All','Train','Val']:
  z=[r for r in rr if split=='All' or r['split']==split]
  for sc in sorted({r['scene_class'] for r in rr}):
   q=[r for r in z if r['scene_class']==sc];scene.append(dict(split=split,scene_class=sc,**counts(q),evidence='AI preliminary imagery inspection' if any(p['visible_vessels'].get('AI_reviewed') for p in q) else 'source context only; content unreviewed',human_verified=0))
 table(ROOT/f'SCENE_COVERAGE_{t}.csv',scene,list(scene[0]))
 vessels=[]
 for split in ['All','Train','Val']:
  z=[r for r in rr if split=='All' or r['split']==split]
  def visible(r,s):return 'visible' in r['visible_vessels'][f'side{s}'] and 'none' not in r['visible_vessels'][f'side{s}'] and 'uncertain' not in r['visible_vessels'][f'side{s}']
  vessels.append(dict(split=split,pairs=len(z),AI_reviewed=sum(bool(r['visible_vessels'].get('AI_reviewed')) for r in z),AI_visible_at_least_one=sum(visible(r,0) or visible(r,1) for r in z),AI_visible_both=sum(visible(r,0) and visible(r,1) for r in z),human_verified=0,not_AI_reviewed=sum(not r['visible_vessels'].get('AI_reviewed') for r in z)))
 table(ROOT/f'VESSEL_VISIBILITY_{t}.csv',vessels,list(vessels[0]))
 states=[]
 for split in ['All','Train','Val']:
  z=[r for r in rr if split=='All' or r['split']==split]
  for category in ['A','B','C','D','E']:
   q=[r for r in z if r['annotatability']==category];states.append(dict(split=split,provisional_annotatability=category,**counts(q),human_annotation=0,independent_review=0,formal_evaluable=0,evidence='AI preliminary candidates, not actual GT; E includes unreviewed'))
 table(ROOT/f'GT_COVERAGE_{t}.csv',states,list(states[0]))
 # These subsets are independent of any algorithm's success, and never qualified GT.
 for name,keep in [('GT_FULL_QUALIFIED',[]),('GT_LOCAL_QUALIFIED',[]),('NO_OVERLAP_CONFIRMED',[]),('GT_REJECTED_ACTUAL',[]),('OVERLAP_WITHOUT_RELIABLE_SUPPORT_AI',[r for r in rr if r['annotatability']=='C']),('UNCERTAIN_OR_UNREVIEWED',[r for r in rr if r['annotatability']=='E'])]:jsonl(ROOT/f'{name}_{t}.jsonl',keep)
 for split in ['Train','Val']:
  keep=[r for r in rr if r['split']==split];jsonl(ROOT/f'REAL_{split}_UNLABELED_{t}.jsonl',keep)
  jsonl(ROOT/f'IMAGE_ONLY_{split}_{t}.jsonl',[dict(sample_id=r['sample_id'],image0=r['image0'],image1=r['image1']) for r in keep])
 assets=[json.loads(p.read_text()) for p in (ROOT/'acquisitions').glob('*/ASSET_*.json')];new_ids={a['acquisition_id'] for a in assets};plans=[]
 for name in ['DOWNLOAD_PLAN_20261015_1543.json','EXPANSION_SOURCE_PLAN_20261015_1545.json']:plans+=json.loads((ROOT/name).read_text())['assets']
 planned={p['acquisition_id'] for p in plans};missing=[]
 for c in sorted(planned-new_ids):
  fs=sorted((ROOT/'acquisitions'/c).glob('FAILURE_*.json'));d=json.loads(fs[-1].read_text()) if fs else {};missing.append(dict(acquisition_id=c,site=amap[c]['site_name'],metadata_only=True,last_failure_file=str(fs[-1]) if fs else None,error=json.dumps(d.get('attempts',[]),ensure_ascii=False)))
 table(ROOT/f'UNOBTAINED_PLANNED_SOURCES_{t}.csv',missing,['acquisition_id','site','metadata_only','last_failure_file','error'])
 for a in aa:
  if a.get('raster_status'):a['historical_inventory_raster_status']=a.pop('raster_status')
  a['raster_status']='LOCAL_NATIVE_PIXELS_AVAILABLE' if a['local_pixel_status']=='locally_available' else 'METADATA_ONLY_NOT_DOWNLOADED'
  a['native_source_assets']=[str(p) for p in (ROOT/'acquisitions'/a['acquisition_id']).glob('ASSET_*.json')]
  a['used_in_archived_or_active_pair']=a['used_in_512_pair'];a['used_in_512_pair']=a['acquisition_id'] in views
  if a['provider']=='Sentinel-1' and not a.get('metadata_path'):
   old=BASE/'real_sar_pilot/sentinel1_20261010_1731/triplet_verified/manifest.json';d=json.loads(old.read_text())
   original=next(p for p in d['scenes'] if p['raster_metadata']['source_tags']['SCENES']==a['acquisition_id'])
   a.update(metadata_path=str(old),metadata_sha256=sha(old),asset_url=original['url'],license=d['license'],license_url='https://www.indigoag.com/forms/atlas-sentinel-license',orbit_direction=original['orbit_direction'],legacy_processing=original['raster_metadata'].get('resampling'),native_512_protocol_status='archived_only_prior_pair_resampling')
 jsonl(ROOT/f'REAL_ACQUISITION_MANIFEST_{t}.jsonl',aa)
 target=json.loads((AUDIT/'TARGETED_SOURCE_PROPOSALS_20261014_2329.json').read_text());targetrows=[]
 for p in target:
  rid=relid(p['acquisition_ids']);z=relations.get(rid,[])
  targetrows.append(dict(relation_id=rid,site=amap[p['acquisition_ids'][0]]['site_name'],acquisition0=p['acquisition_ids'][0],acquisition1=p['acquisition_ids'][1],angle_degrees=delta(*[amap[c]['vendor_native_azimuth_degrees'] for c in p['acquisition_ids']]),both_rasters_available=all(amap[c]['local_pixel_status']=='locally_available' for c in p['acquisition_ids']),generated_512_pairs=len(z),qualified_GT=0))
 table(ROOT/f'PRIORITY_SOURCE_OUTCOMES_{t}.csv',targetrows,list(targetrows[0]))
 pilot=rows(ROOT/'annotation_20261015_1644/PACKAGE_INDEX_20261015_1644.jsonl');pmap={r['sample_id']:r for r in rr};pilotids=[p['sample_id'] for batch in pilot if batch['slot']=='A' and batch['batch_id'].startswith(('B001_','B002_')) for p in batch['tasks']]
 summary=dict(at=now(),statistics_version=t,pair_manifest=str(pairpath),acquisition_manifest=str(ROOT/f'REAL_ACQUISITION_MANIFEST_{t}.jsonl'),**counts(rr),archived_and_active_pairs=len(allrr),quarantined_prior_resampled_pairs=len(allrr)-len(rr),archived_and_active_relations=len({r['relation_id'] for r in allrr}),view_instances=2*len(rr),distinct_native_view_footprints=len(unique_windows),distinct_algorithm_image_SHA256=len({v['algorithm_input_sha256'] for r in rr for v in r['views']}),acquisition_graph_components=len(graph),site_component_count=len(site),same_relation_duplicate_pair_windows=0,same_relation_any_side_iou_above_point2=0,cross_relation_shared_acquisition_window_edges_above_point2=len(spatial),cross_relation_edges_above_point7=sum(r['native_window_iou']>.7 for r in spatial),cross_relation_exact_window_edges=len(duplicate),cross_product_grid_comparisons=cross_grid,all_dependency_edges_same_split=all(r['same_split'] for r in spatial),downloaded_new_acquisitions=len(new_ids),downloaded_but_unpaired=sorted(new_ids-set(views)),planned_new_acquisitions=len(planned),unobtained_planned_acquisitions=len(missing),new_geocoding_modes=dict(Counter(a['geocoding_diagnostic']['method'] for a in assets)),new_context_shapes=dict(Counter(str(a['native_info']['width']) for a in assets)),metadata_records=len(aa),metadata_only=sum(a['local_pixel_status']=='metadata_only' for a in aa),locally_available_acquisition_identities=sum(a['local_pixel_status']=='locally_available' for a in aa),angle=angle,site=site,vessels=vessels,pilot50=dict(pairs=len(pilotids),annotatability=dict(Counter(pmap[s]['annotatability'] for s in pilotids)),relations=len({pmap[s]['relation_id'] for s in pilotids}),sites=len({pmap[s]['component'] for s in pilotids})),unlabeled_only=True,actual_human_GT=0)
 save(ROOT/f'FINAL_DATASET_STATISTICS_{t}.json',summary)
 print(json.dumps({k:v for k,v in summary.items() if k not in ['angle','site']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
