"""Verify completed fixed-run artifacts, scores and frozen referenced identities."""
import csv,inspect,re
from collections import Counter
from core import *
def main():
 failures=[];checks={};identity=json.loads((ROOT/'identity.json').read_text())
 for path,digest in identity['files'].items():
  if sha(path)!=digest:failures.append('Changed referenced file: '+path)
 for manifest in ['formal_inputs.json','real_inputs.json']:
  rows=json.loads((ROOT/manifest).read_text())
  for r in rows:
   for path,digest in zip(r['paths'],r['sha256']):
    if sha(path)!=digest:failures.append('Changed pixel input: '+path)
 checks['frozen_referenced_files_and_pixels_unchanged']=not failures
 allresults={n:json.loads((ROOT/(n+'_results.json')).read_text()) for n in ['formal','validation','real']}
 assert len(allresults['formal'])==4500 and len(allresults['validation'])==168 and len(allresults['real'])==24
 for name,rs in allresults.items():
  methods=METHODS if name=='formal' else ['m0','m1','m2'];n=900 if name=='formal' else (56 if name=='validation' else 8)
  assert Counter(r['method'] for r in rs)==Counter(dict.fromkeys(methods,n))
  assert len({(r['id'],r['method']) for r in rs})==len(rs)
  assert all(not r.get('exception') and not r.get('prediction_reason','').startswith('exception') for r in rs)
  assert json.loads((ROOT/(name+'_status.json')).read_text())['state']=='completed'
 checks['complete_unique_results_no_exceptions']=True
 formal=allresults['formal'];modelsbyid={}
 for r in formal:
  assert r['n_keypoints0']<=1024 and r['n_keypoints1']<=1024
  folder=ROOT/'formal_predictions'/r['id'].replace(':','__');z=np.load(folder/(r['method']+'.npz'))
  assert len(z['points0'])==r['n_matches']==len(z['points1']) and len(z['inlier_mask'])==r['n_matches']
  assert (z['points0'][:,0]>=0).all() and (z['points0'][:,0]<r['width0']).all()
  assert (z['points0'][:,1]>=0).all() and (z['points0'][:,1]<r['height0']).all()
  assert (z['points1'][:,0]>=0).all() and (z['points1'][:,0]<r['width1']).all()
  assert (z['points1'][:,1]>=0).all() and (z['points1'][:,1]<r['height1']).all()
  if r['method']=='m0':modelsbyid[r['id']]=[z['keypoints0'].copy(),z['keypoints1'].copy()]
  if r['method'] in ['m1','m2']:
   assert all(np.array_equal(x,z[k]) for x,k in zip(modelsbyid[r['id']],['keypoints0','keypoints1']))
  for t in [3,5,10]:
   expected=None if r['no_overlap'] else bool(r['accepted'] and r['epe_mean'] is not None and r['epe_mean']<=t)
   assert r[f'success_{t}px']==expected
 checks['native_bounds_feature_caps_detection_identity_and_success_denominators']=True
 a=json.loads((ROOT/'analysis_summary.json').read_text())
 for x in a['formal']:
  rs=[r for r in formal if r['method']==x['method']]
  assert x['positive']==860 and x['none']==40 and x['accepted']+x['rejected']+x['no_output']==900
  assert x['success5']==sum(bool(r['success_5px']) for r in rs)
 for name,n in [('formal_per_pair.csv',4500),('validation_module_per_pair.csv',168),('real_native_per_pair.csv',24)]:
  with (ROOT/'tables'/name).open(encoding='utf-8-sig') as f:assert len(list(csv.DictReader(f)))==n
 checks['tables_complete_and_scores_recomputed']=True
 from PIL import Image
 for p in (ROOT/'figures').glob('*.png'):
  with Image.open(p) as im:im.verify()
 checks['visualizations_valid']=len(list((ROOT/'figures').glob('*.png')))
 report=json.loads((ROOT/'latest_report.json').read_text());md=(ROOT/report['markdown']).read_text()
 for match in re.findall(r'\]\(([^)]+)\)',md):
  if not match.startswith(('https://','http://')):assert (ROOT/match).exists(),match
 htmltext=(ROOT/report['html']).read_text();assert '<table>' in htmltext and '<img src=' in htmltext and '641' in htmltext
 checks['report_links_tables_and_images']=True
 for n in ['formal_process','diagnostics_process']:
  proc=json.loads((ROOT/(n+'.json')).read_text());assert proc['physical_gpu'] in [0,2]
 checks['authorized_physical_GPUs_only']=True
 assert not failures,failures
 atomic_json(ROOT/'acceptance.json',dict(passed=True,verified=now(),checks=checks,formal_run_count=1,inference_rows=4692,failures=failures,report=report,limitations=['Exact synthetic GT only','No independent real SAR GT','Observed fixed benchmark, not unseen real confirmation']))
 # Freeze the completed output identity; report hash and code sufficient for reuse.
 files=list((ROOT/'code').glob('*.py'))+[ROOT/report['markdown'],ROOT/report['html'],ROOT/'protocol.json',ROOT/'identity.json',ROOT/'formal_results.json',ROOT/'validation_results.json',ROOT/'validation_evaluated.json',ROOT/'real_results.json']+list((ROOT/'tables').glob('*.csv'))
 atomic_json(ROOT/'final_file_identity.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
 print(json.dumps(dict(passed=True,checks=checks),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
