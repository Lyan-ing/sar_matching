"""Audit existing SAR files and SL-SSDD masks; no original edits or downloads."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json,sys,hashlib,csv
import numpy as np,cv2
from PIL import Image,ImageDraw
BASE=Path('/home/yl/matching/sar_matching');ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'MaritimeSARRegDataset'))
from sarreg.io import read_gray
from sarreg.indexer import _mask_lookup,_find_sea_mask
T=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
INDEX=BASE/'MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl'
AUDIT=BASE/'MaritimeSARRegDataset/reports/phase1_source_relationships_v2_20261009_1846.json'
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def jsonl(p,rows):
 with p.open('x',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def comp(r):return ('hrsid_parent:' if r['dataset'] in ['hrsid','background'] else 'ssdd_known_pixel_component:')+str(r['source_group'])
def main():
 cv2.setNumThreads(1)
 rows=[json.loads(l) for l in INDEX.read_text().splitlines()];old=json.loads(AUDIT.read_text())
 assert hashlib.sha256(INDEX.read_bytes()).hexdigest()==old['output_sha256']
 canonical={p:min(g) for g in old['decoded_duplicate_groups'] for p in g}
 masks=_mask_lookup('/data1/libo/yl/datasets/sar/SL-SSDD')
 def one(r):
  out=dict(r,known_component=comp(r),canonical_image_path=canonical.get(r['image_path'],r['image_path']))
  try:
   a=read_gray(r['image_path']);h,w=a.shape;out.update(readable=True,native_width=w,native_height=h,
    annotation_count=len(r['annotations']),low_texture_score=float(np.std(cv2.resize(a,(96,96),interpolation=cv2.INTER_AREA))))
   out['sea_land_evidence']='publisher coarse offshore/inshore only; no pixel sea mask' if r['dataset']=='hrsid' else 'no sea/land annotation'
   out['pixel_sea_fraction']=None
   if r['dataset']=='ssdd':
    p=_find_sea_mask(masks,r['official_split'],Path(r['image_path']).stem)
    if p is None:raise ValueError('SL-SSDD mask missing')
    m=read_gray(p)
    assert m.shape==a.shape and np.all(np.isin(m,[0,255]))
    out.update(sea_mask_path=str(p),pixel_sea_fraction=float((m==255).mean()),sea_land_evidence='SL-SSDD publisher binary 0=land/255=sea; stem+official-split+shape verified')
   scene=r['scene_type']
   if out['pixel_sea_fraction'] is not None:scene='offshore' if out['pixel_sea_fraction']>=.98 else 'inshore' if out['pixel_sea_fraction']>0 else 'land'
   if scene=='offshore':category='offshore_vessels_low_texture_sea'
   elif scene=='inshore':category='nearshore_vessels'
   elif scene=='land':category='other_land_structures'
   else:category='unclassified_background'
   out.update(scene_category=category,scene_category_status='publisher_coarse_or_mask_derived' if category!='unclassified_background' else 'unknown',
    port_dense_candidate=(scene=='inshore' and len(r['annotations'])>=8),
    port_structure_confirmed=False,independent_raw_acquisition_identity_known=False,
    isolation_strength='known_HRSID_filename_parent_plus_existing_pixel_joins' if r['dataset']!='ssdd' else 'known_pixel_component_only_raw_scene_unresolved')
   out['construction_pool']='parent_component_core' if r['dataset']!='ssdd' else 'ssdd_component_limited_extension'
  except Exception as e:out.update(readable=False,error=repr(e))
  return out
 with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,rows))
 selected=[r for r in results if r['readable'] and r['split'] in ['train','val'] and r['canonical_image_path']==r['image_path']]
 sets={s:{r['known_component'] for r in selected if r['split']==s} for s in ['train','val']}
 assert not sets['train']&sets['val']
 # Existing source test remains reserved, including every parent component.
 tests={comp(r) for r in rows if r['split']=='test'};assert not tests&(sets['train']|sets['val'])
 inventory=ROOT/f'SOURCE_INVENTORY_{T}.jsonl';jsonl(inventory,results)
 jsonl(ROOT/f'SYNTHETIC_SOURCE_POOL_{T}.jsonl',selected)
 summary=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),input_index=str(INDEX),input_index_sha256=old['output_sha256'],
  inventory=str(inventory),original_index_rows=len(rows),readable_rows=sum(r['readable'] for r in results),
  known_exact_duplicate_groups_reused=len(old['decoded_duplicate_groups']),
  known_exact_duplicates_removed=len(results)-len({r['canonical_image_path'] for r in results}),
  known_deduplicated_images=len({r['canonical_image_path'] for r in results if r['readable']}),
  independent_physical_acquisitions_count='not established; original files/known parents are counted separately',
  SL_SSDD_masks=len(masks),SL_SSDD_new_SAR_images=0,
  image_counts={f'{d}/{s}':sum(r['dataset']==d and r['split']==s for r in results if r['readable']) for d in ['hrsid','ssdd','background'] for s in ['train','val','test']},
  source_pool_count=len(selected),source_pool_counts=dict(Counter(r['split']+'/'+r['construction_pool'] for r in selected)),
  source_scene_counts=dict(Counter(r['split']+'/'+r['scene_category'] for r in selected)),
  known_component_counts={s:len(v) for s,v in sets.items()},train_val_component_intersection=[],test_component_intersection=[],
  source_pool_distinct_parent_core_components={s:len({r['known_component'] for r in selected if r['split']==s and r['construction_pool']=='parent_component_core'}) for s in ['train','val']},
  verified_pixel_sea_masks=sum(r['readable'] and r['dataset']=='ssdd' for r in results),
  dense_port_candidates=sum(r.get('port_dense_candidate',False) for r in selected),
  semantic_caveat='coarse inshore/offshore and mask fractions are not verified port labels; unknown background stays unknown',
  failures=[{k:r.get(k) for k in ['dataset','source_id','image_path','error']} for r in results if not r['readable']],
  original_bytes_hashed=False,no_downloads=True)
 save(ROOT/f'RESOURCE_AUDIT_{T}.json',summary)
 # Bounded scene review, no model predictions: diverse parent/component examples.
 chosen=[];used=Counter()
 for r in sorted(selected,key=lambda r:(r['port_dense_candidate'],r['low_texture_score']),reverse=True):
  if r['port_dense_candidate'] and len([q for q in chosen if q['dataset']!='background'])<24 and used[r['known_component']]<2:
   chosen.append(r);used[r['known_component']]+=1
 for r in sorted([r for r in selected if r['dataset']=='background'],key=lambda r:r['low_texture_score'],reverse=True)[:24]:chosen.append(r)
 im=Image.new('RGB',(1440,240*((len(chosen)+5)//6)),(20,20,20));draw=ImageDraw.Draw(im)
 for i,r in enumerate(chosen):
  x=(i%6)*240;y=(i//6)*240
  with Image.open(r['image_path']) as tile:tile=tile.convert('RGB');tile.thumbnail((232,205));im.paste(tile,(x,y+32))
  draw.text((x+3,y+2),f"{i:02d} {r['dataset']} {r['source_id'][:23]}",fill='white')
  draw.text((x+3,y+16),f"{r['split']} ships={r['annotation_count']} {r['known_component'][-12:]}",fill='white')
 im.save(ROOT/f'SCENE_SCREEN_{T}.png');jsonl(ROOT/f'SCENE_SCREEN_ENTRIES_{T}.jsonl',chosen)
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
