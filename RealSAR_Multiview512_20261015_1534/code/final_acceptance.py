"""Verify the completed delivery, including the legacy-resampling quarantine."""
from common import *
from collections import Counter
import subprocess,zipfile
from reader import image_records,load_images

def main():
 t=stamp();entrypath=latest('REAL_MULTIVIEW_512_ENTRYPOINTS_*.json');e=json.loads(entrypath.read_text());checks={}
 rr=rows(e['pair_manifest']);active=[p for p in rr if p['qa_state']=='passed'];bad=[p for p in rr if p['qa_state']!='passed']
 assert len(rr)==1036 and len(active)==1033 and len(bad)==3
 assert all(p['sample_id'].startswith('PILOT_') and p['geometry_resampling_applied'] and not p['GT_qualified'] for p in bad)
 assert len({p['relation_id'] for p in active})==137 and len({c for p in active for c in p['acquisition_ids']})==126 and len({p['component'] for p in active})==20
 assert all(len(set(p['acquisition_ids']))==2 and not p['geometry_resampling_applied'] and not p['model_selection_used'] for p in active if p['source_kind']=='new_real_native_crop')
 checks['active_native_population_and_archived_resampling_quarantine']=True
 populations={};ids={}
 for split,key,n in [('Train','train_image_only',747),('Val','val_image_only',286)]:
  inputs=list(image_records(e[key]));assert len(inputs)==n
  ids[split]={r['sample_id'] for r in inputs};populations[split]=[r for r in active if r['split']==split]
  assert ids[split]=={r['sample_id'] for r in populations[split]}
  for r in inputs[:3]+inputs[-3:]:assert all(im.shape==(512,512) for im in load_images(r))
 for key in ['component','relation_id']:
  assert not ({r[key] for r in populations['Train']}&{r[key] for r in populations['Val']})
 assert not ({c for r in populations['Train'] for c in r['acquisition_ids']}&{c for r in populations['Val'] for c in r['acquisition_ids']})
 checks['image_only_manifests_exclude_three_quarantined_and_are_source_isolated']=True
 stats=json.loads(Path(e['counts']).read_text());assert stats['pairs']==1033 and stats['all_dependency_edges_same_split']
 assert Counter(p['angle_bin'] for p in active)==Counter({'0-15':289,'15-30':152,'30-60':232,'60-90':88,'90-120':117,'>120':141,'unknown':14})
 assert Counter(p['annotatability'] for p in active)==Counter(B=40,C=37,E=956)
 assert stats['view_instances']==2066 and stats['distinct_native_view_footprints']==1882
 checks['angle_content_and_spatial_dependency_statistics']=True
 assert Path(e['GT_manifest']).stat().st_size==0 and not any(p['GT_qualified'] for p in rr)
 status=rows(e['GT_status']);assert Counter(r['state'] for r in status)==Counter(unannotated=1033,quarantined_input_not_formal=3)
 assert json.loads(Path(e['GT_actual_intake_QA']).read_text())['actual_return_versions']==0
 for key in ['full_GT_qualified','local_GT_qualified','actual_GT_rejected','no_overlap_confirmed']:assert Path(e[key]).stat().st_size==0
 checks['no_actual_human_submissions_and_no_fabricated_GT_or_negatives']=True
 pkg=rows(e['annotation_package_index']);assert len(pkg)==84
 active_byid={r['sample_id']:r for r in active};all_byid={r['sample_id']:r for r in rr}
 skip=list(csv.DictReader(Path(e['annotation_excluded_task_ids']).open(encoding='utf-8-sig')));assert len(skip)==6
 skipkeys={(x['package_id'],x['task_id']) for x in skip};pilot=[];seen=set();oldjs=FIXED/'blind_release'
 for p in pkg:
  assert p['person_id'] is None and Path(p['zip']).is_file() and Path(p['html']).is_file()
  html=Path(p['html']).read_text();d=re.search(r'<script>window.BLIND_TASKS=(.*?);window.BLIND_PACKAGE_ID=',html,re.S);tasks=json.loads(d[1]);assert len(tasks)==p['task_count']
  assert all(set(x)<={'sample_id','width0','height0','width1','height1','batch_id','batch_label','site_name','image0','image1'} and x['site_name']=='' for x in tasks)
  assert not any(k in d[1] for k in ['GeoTransform','azimuth','latitude','longitude','relation_id','acquisition_id','projection','candidate_points'])
  for js in ['coordinates_20261013_2227.js','offline_capture_20261013_2227.js','blind_app_20261013_2227.js']:assert sha(Path(p['html']).parent/js)==sha(oldjs/js)
  with zipfile.ZipFile(p['zip']) as z:assert Path(p['html']).name in z.namelist()
  for x in p['tasks']:
   assert x['sample_id'] not in ['C001','C002','C003']
   r=all_byid[x['sample_id']];assert x['image_sha256']==[v['algorithm_input_sha256'] for v in r['views']]
   if (p['package_id'],x['task_id']) in skipkeys:assert r in bad
   else:assert r['sample_id'] in active_byid
   if p['slot']=='A':seen.add(x['sample_id'])
   if p['slot']=='A' and p['batch_id'].startswith(('B001_','B002_')):pilot.append(r)
 assert len(seen)==1033 and len(pilot)==50 and Counter(p['annotatability'] for p in pilot)==Counter(B=40,E=10)
 assert all(p['qa_state']=='passed' for p in pilot)
 checks['84_existing_UI_packages_anonymous_1030_active_plus3_skipped_frozen3_referenced_separately']=True
 browser=json.loads(Path(e['browser_QA']).read_text());assert browser['passed'] and len(browser['checks'])==4 and not browser['actual_human_identity_entered'] and browser['point_clicks']==browser['save_calls']==0
 checks['actual_readonly_Chromium_first50_A_B_four_packages']=True
 for p,h in json.loads(latest('PROTECTED_HISTORY_*.json').read_text()).items():assert sha(p)==h
 frozen=json.loads(Path(e['old_freeze']).read_text())
 for name,v in frozen['files'].items():assert sha(FIXED/name)==v['sha256']
 checks['historical_indices_synthetic_V22_and_all36_fixed512_frozen_files_unchanged']=True
 # Check references and PDF structure, without repeating content review or image QA.
 def paths(v):
  if isinstance(v,dict):
   for x in v.values():yield from paths(x)
  elif isinstance(v,list):
   for x in v:yield from paths(x)
  elif isinstance(v,str) and v.startswith('/home/yl/matching/sar_matching/'):yield Path(v)
 for p in paths(e):assert p.exists(),str(p)
 pdfinfo=subprocess.check_output(['pdfinfo',e['report_pdf']],text=True);assert '595.28 x 841.89' in pdfinfo or '595.28 x 841.88' in pdfinfo or 'A4' in pdfinfo
 assert Path(e['report_md']).with_suffix('.pdf')==Path(e['report_pdf'])
 conversions=list(ROOT.glob('pdf_export_*/conversion_manifest_*.json'));conversion=next(json.loads(p.read_text()) for p in conversions if json.loads(p.read_text())['pdf']==e['report_pdf'])
 assert conversion['paper']=='A4 portrait' and conversion['continuous_layout'] and not conversion['forced_heading_pagebreaks'] and not conversion['segmented_pdf_merge']
 checks['all_delivery_paths_resolve_and_A4_continuous_PDF_parses']=True
 release_files={p:sha(p) for p in paths(e) if p.is_file() and p.suffix in ['.py','.md','.pdf','.csv','.json','.jsonl']}
 for p in (ROOT/'code').glob('*.py'):release_files[p]=sha(p)
 identity=ROOT/f'RELEASE_IDENTITY_{t}.json';save(identity,dict(at=now(),entrypoint=str(entrypath),files={str(p):h for p,h in release_files.items()},entrypoint_sha256=sha(entrypath),no_old_whole_raster_rehash=True,actual_human_GT=0))
 acceptance=ROOT/f'FINAL_ACCEPTANCE_{t}.json';save(acceptance,dict(at=now(),passed=True,checks=checks,active_native_pairs=1033,archived_pairs=1036,quarantined_historical_resampling=3,Train=747,Val=286,actual_GT=0,identity=str(identity),entrypoint=str(entrypath),limitations_not_completion_claim=['independent physical GT unavailable','vessel content validation sparse, Val vessel evidence missing','0-120 main878 not1000','Val90-120 has one site only']))
 save(ROOT/f'RELEASE_SELECTION_{t}.json',dict(at=now(),current_entrypoint=str(entrypath),current_report_md=e['report_md'],current_report_pdf=e['report_pdf'],acceptance=str(acceptance),identity=str(identity),superseded_draft_entrypoints=[str(p) for p in ROOT.glob('REAL_MULTIVIEW_512_ENTRYPOINTS_*.json') if p!=entrypath],reason='final source trace found 3 legacy Sentinel pair-resampled products; archived, never overwritten'))
 print('FINAL_ACCEPTANCE',acceptance,'PASSED',len(checks),'groups; GT remains0',flush=True)

if __name__=='__main__':main()
