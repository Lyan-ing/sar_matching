"""Keep historical products intact; exclude documented prior pair resampling."""
from common import *

def main():
 t=stamp();source=ROOT/'REAL_PAIR_MANIFEST_20261015_1642.jsonl';rr=rows(source)
 original=BASE/'real_sar_pilot/sentinel1_20261010_1731/triplet_verified/manifest.json'
 history=json.loads(original.read_text());evidence=[]
 for r in rr:
  if r['sample_id'].startswith('PILOT_'):
   r.update(qa_state='quarantined_historical_pair_resampling',qa_failures=['prior bilinear_to_first_scene_map_grid documented in original triplet manifest'],geometry_resampling_applied=True,native_pair_protocol_eligible=False,GT_qualified=False,independent_GT=False,allow_supervised_training=False,gt_state='quarantined_input_not_formal',historical_product_evidence=str(original),historical_product_evidence_sha256=sha(original))
   for v in r['views']:
    v['historical_reported_product_origin_xy']=v['product_origin_xy'];v['product_origin_frame_warning']='legacy offset is relative to already exported512 product, not an original unwarped acquisition grid'
   evidence.append(dict(sample_id=r['sample_id'],acquisition_ids=r['acquisition_ids'],evidence=str(original),actual_metadata_resampling=[s['raster_metadata'].get('resampling') for s in history['scenes']],status=r['qa_state'],original_files_modified=False))
 assert len(evidence)==3
 jsonl(ROOT/f'REAL_PAIR_MANIFEST_{t}.jsonl',rr);jsonl(ROOT/f'HISTORICAL_RESAMPLING_QUARANTINE_{t}.jsonl',evidence)
 packages=rows(ROOT/'annotation_20261015_1644/PACKAGE_INDEX_20261015_1644.jsonl');excluded={e['sample_id'] for e in evidence};anonymous=[]
 for pkg in packages:
  for task in pkg['tasks']:
   if task['sample_id'] in excluded:anonymous.append(dict(package_id=pkg['package_id'],task_id=task['task_id'],action='skip_this_archived_task_not_in_native_release'))
 table(ROOT/f'BLIND_PACKAGE_EXCLUSIONS_{t}.csv',anonymous,['package_id','task_id','action'])
 assert not any(x['package_id'].startswith(('B001_','B002_')) for x in anonymous)
 print('QUARANTINED',len(evidence),'kept unchanged; pilot50 unaffected; manifest',ROOT/f'REAL_PAIR_MANIFEST_{t}.jsonl',flush=True)

if __name__=='__main__':main()
