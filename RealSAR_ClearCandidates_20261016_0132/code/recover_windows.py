"""Resume only missing public native windows; original data stays read-only."""
import os,sys,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
for key in list(os.environ):
 if key.lower() in ['http_proxy','https_proxy','all_proxy']:os.environ.pop(key,None)
os.environ['NO_PROXY']='*'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'RealSAR_Multiview512_20261015_1534/code'))
import fetch
from common import rows,jsonl,save,stamp,now

def main():
 fetch.ROOT=ROOT
 relations=rows(sorted(ROOT.glob('NEW_NATIVE_RELATIONS_*.jsonl'))[-1])+rows(sorted(ROOT.glob('CAPELLA_NATIVE_RELATIONS_*.jsonl'))[-1])
 source_rows=rows(sorted(ROOT.glob('SELECTED_SOURCE_METADATA_*.jsonl'))[-1])+rows(sorted(ROOT.glob('CAPELLA_SELECTED_SOURCE_METADATA_*.jsonl'))[-1])
 sources={a['acquisition_id']:a for a in source_rows}
 work=[(sources[c],r['geographic_search_center']) for r in relations for c in r['acquisition_ids'] if not list((ROOT/'acquisitions'/c).glob('ASSET_*.json'))]
 print('Missing windows only:',len(work),'completed assets reused:',6-len(work),flush=True)
 def get(item):
  a,center=item
  try:return fetch.fetch_one(a,center,side=4096,rpc_reference=a['provider']=='Umbra')
  except Exception as e:
   print('RECOVERY_ERROR',a['acquisition_id'],type(e).__name__,str(e)[:120],flush=True)
   return None
 with ThreadPoolExecutor(max_workers=2) as pool:
  for result in pool.map(get,work):
   if result:print('RECOVERED',result['acquisition_id'],flush=True)
 for r in relations:
  r['assets']=[]
  for cid in r['acquisition_ids']:
   files=sorted((ROOT/'acquisitions'/cid).glob('ASSET_*.json'))
   if files:r['assets'].append(json.loads(files[-1].read_text()))
  r['acquisition_complete']=len(r['assets'])==2
  r['classification']='await_actual_pixel_review' if r['acquisition_complete'] else 'download_incomplete_not_a_pair'
 t=stamp()
 jsonl(ROOT/f'RECOVERED_NATIVE_RELATIONS_{t}.jsonl',relations)
 save(ROOT/f'RECOVERY_RESULT_{t}.json',dict(at=now(),missing_windows_requested=len(work),completed_pairs=sum(r['acquisition_complete'] for r in relations),native_assets=sum(len(r['assets']) for r in relations),proxy_change='per-process only; global environment unchanged',new_512_crops=0,GT_created=0))
 print('COMPLETED_PAIRS',sum(r['acquisition_complete'] for r in relations),flush=True)

if __name__=='__main__':main()
