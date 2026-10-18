"""Reuse prior audited identities; read local headers, no raster resampling or GT."""
from common import *
from collections import Counter,defaultdict

def main():
 t=stamp();acqs={r['acquisition_id']:r for r in rows(ANGLE/'acquisition_inventory.jsonl')}
 rels=rows(AUDIT/'REAL_ACQUISITION_RELATIONS_20261014_2329.jsonl');oldc=rows(AUDIT/'REAL512_EXISTING_CROPS_20261014_2329.jsonl')
 seen_sites={canonical(r['site']) for r in rels};assets=[];errors=[];paths=set()
 def add(cid,p,origin=None,original=None,metadata=None):
  if str(p) in paths:return
  try:assets.append(native_asset(cid,p,origin,original,metadata));paths.add(str(p))
  except Exception as e:errors.append(dict(path=str(p),error=repr(e)))
 for p in (ANGLE/'assets').glob('*/metadata.json'):
  d=json.loads(p.read_text());cid=d['acquisition']['acquisition_id'];acqs.setdefault(cid,d['acquisition'])
  w=d.get('original_window_xywh');add(cid,d['native_tif'],w[:2] if w else None,d.get('original_profile'),str(p))
 v2=BASE/'RealSAR_MultiViewV2_20261012_1259'
 for r in rows(v2/'regions.jsonl'):
  for i,v in enumerate(r['views']):
   cid=v['acquisition_id'];w=v['context_provenance'].get('original_window_xywh')
   add(cid,v2/'regions'/r['sample_id']/f'context{i}.tif',w[:2] if w else None,None,str(v2/'regions'/r['sample_id']/'metadata.json'))
 for r in rels:
  for v in r['view_metadata']:
   cid=v['acquisition_id']
   if cid not in acqs:acqs[cid]=dict(v,provider=r['sensor'],site_name=r['site'],datetime=v.get('utc'),geometry=None,input_units='linear_power')
 crops=[];pilot={r['pair_id']+'__'+r['chip_id']:r for r in rows(OLD/'pilot_512/candidates.jsonl')}
 fixed={r['candidate_id']:r for r in json.loads(next((FIXED/'custodian').glob('candidate_sources*')).read_text())}
 relation_lookup={r['relation_id']:r for r in rels}
 for r in oldc:
  rr=relation_lookup[r['relation_id']];c=dict(r,views=[],acquisition_ids=[],split='Train',split_reason='historical image/algorithm/human diagnostic exposure',gt_state='unannotated',annotatability='E',annotatability_basis='not yet independently physically confirmed')
  if r['sample_id'] in fixed:
   f=fixed[r['sample_id']]
   for v in f['views']:
    cid=v['acquisition']['acquisition_id'];c['acquisition_ids'].append(cid)
    a=native_asset(cid,v['raw_crop_tif'],v['crop_offset_in_original_product_xy'],None,r['source_metadata']);a['valid_mask']=v['valid_mask_png'];c['views'].append(a)
   c['frozen_external_package']=str(FIXED/'blind512_materials_20261014_1810.zip')
  elif r['sample_id'] in pilot:
   f=pilot[r['sample_id']]
   for side,prefix in enumerate(['reference','moving']):
    cid=f[prefix+'_acquisition_id'];c['acquisition_ids'].append(cid);a=native_asset(cid,f[prefix+'_path'],f['reference_window_xy'] if side==0 else f['moving_native_window_xy'],None,r['source_metadata']);c['views'].append(a)
  else:
   f=json.loads(Path(r['source_metadata']).read_text())
   for side,prefix in enumerate(['reference','moving']):
    cid=f[prefix+'_acquisition_id'];c['acquisition_ids'].append(cid);v=r['views'][side]
    a=native_asset(cid,v['native_tif'],v['parent_product_origin_xy'],None,r['source_metadata'])
    a['valid_mask']=str(Path(v['native_tif']).parent/('reference_valid.png' if side==0 else 'moving_native_valid.png'));c['views'].append(a)
  assert len(set(c['acquisition_ids']))==2
  c['relation_id']=relid(c['acquisition_ids']);c['component']=component(r['site']);c['image_only_input']=[r['image0'],r['image1']]
  c['prior_geographic_or_algorithm_exposure']=True;c['human_points_migrated']=False;c['GT_qualified']=False
  crops.append(c)
 for a in acqs.values():
  a.pop('source_tags',None)
  a['component']=component(a['site_name']);a['split']='Train' if canonical(a['site_name']) in seen_sites else 'Val'
  a['split_basis']='old observed station' if a['split']=='Train' else 'new actual imagery station; metadata search is not algorithm exposure'
  a['platform_heading_degrees']=None
  a['radar_los_azimuth_degrees']=a.get('azimuth_degrees');a['azimuth_original_field']=a.get('azimuth_source_field','view:azimuth' if a['provider']=='Capella' else 'angleAzimuthDegrees or view:azimuth' if a['provider']=='Umbra' else None)
  a['heading_not_inferred_from_look_or_orbit']=True
 protected={}
 for p in [AUDIT/'REAL_ACQUISITION_RELATIONS_20261014_2329.jsonl',AUDIT/'REAL512_EXISTING_CROPS_20261014_2329.jsonl',BASE/'SyntheticSAR_V22_20261015_1352/DATASET_ENTRYPOINTS_V22_20261015_1434.json',*FIXED.glob('FREEZE*.json'),*FIXED.glob('EVALUATION_PROTOCOL*.json')]:protected[str(p)]=sha(p)
 save(ROOT/f'PROTECTED_HISTORY_{t}.json',protected)
 protocol=next(FIXED.glob('EVALUATION_PROTOCOL_20261014_1754.json'));save(ROOT/f'FROZEN_512_PROTOCOL_REFERENCE_{t}.json',dict(path=str(protocol),sha256=sha(protocol),unaltered=True))
 jsonl(ROOT/f'ACQUISITION_INVENTORY_{t}.jsonl',sorted(acqs.values(),key=lambda a:a['acquisition_id']))
 jsonl(ROOT/f'LOCAL_NATIVE_ASSETS_{t}.jsonl',assets);jsonl(ROOT/f'EXISTING512_REFERENCES_{t}.jsonl',crops)
 save(ROOT/f'INITIAL_AUDIT_{t}.json',dict(at=now(),legacy_pairs=len(crops),relations=len(rels),actual_legacy_acquisitions=len({i for r in crops for i in r['acquisition_ids']}),native_queue=18,available_contexts=len(assets),context_acquisitions=len({a['acquisition_id'] for a in assets}),header_errors=errors,GT=0,read_only_old_assets=True))
 print(json.dumps(dict(crops=len(crops),old_relations=len(rels),local_contexts=len(assets),inventory=len(acqs),errors=errors),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
