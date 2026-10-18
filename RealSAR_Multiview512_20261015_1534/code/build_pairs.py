"""Freeze new native crops after bounded, model-free common-FOV region search."""
from common import *
from itertools import combinations
from collections import defaultdict,Counter
import argparse
from functools import lru_cache

def assets():
 aa=rows(latest('LOCAL_NATIVE_ASSETS_*.jsonl'))+[json.loads(p.read_text()) for p in sorted((ROOT/'acquisitions').glob('*/ASSET_*.json'))]
 out=defaultdict(list)
 for a in aa:out[a['acquisition_id']].append(a)
 return out

def relations(bank,metadata):
 old=rows(AUDIT/'REAL_ACQUISITION_RELATIONS_20261014_2329.jsonl');out={r['relation_id']:dict(r,origin='legacy') for r in old}
 groups=defaultdict(list)
 for cid in bank:
  if cid in metadata:groups[metadata[cid]['site_name']].append(metadata[cid])
 new=[]
 for site,rr in groups.items():
  for a,b in combinations(rr,2):
   ids=[a['acquisition_id'],b['acquisition_id']];rid=relid(ids)
   if rid in out:continue
   ang=delta(a.get('azimuth_degrees'),b.get('azimuth_degrees'));days=abs((utc(a['datetime'])-utc(b['datetime'])).total_seconds())/86400
   if ang is None or not .5<=ang<=120 or days>30:continue
   pa,crs=footprint(a);pb,_=footprint(b,crs);inter=poly_intersection(pa,pb)
   if poly_area(inter)<.6*min(poly_area(pa),poly_area(pb)):continue
   new.append(dict(relation_id=rid,site=site,sensor=a['provider'],acquisition_ids=ids,view_metadata=[a,b],azimuth_delta_deg=ang,angle_bin=binof(ang),time_gap_days=days,origin='new_actual_acquisition_relation'))
 # Bounded relation expansion: up to six distinct identities per site and angular bin.
 counts=Counter()
 for r in sorted(new,key=lambda r:(r['time_gap_days'],r['relation_id'])):
  k=(r['site'],r['angle_bin'])
  if counts[k]>=6:continue
  out[r['relation_id']]=r;counts[k]+=1
 return list(out.values())

def preview_score(im,valid):
 f=cv2.GaussianBlur(im.astype(np.float32)/255,(0,0),3)
 std=[]
 for y in range(0,len(im),128):
  for x in range(0,im.shape[1],128):
   m=valid[y:y+128,x:x+128];std.append(float(f[y:y+128,x:x+128][m].std()) if m.any() else 0)
 return dict(coarse_contrast_std=float(f[valid].std()),coarse_supported_cells=sum(v>.035 for v in std),coarse_cell_std=std,proxy_only_not_static_semantics=True)

@lru_cache(maxsize=6)
def filedata(path,units,png):
 with rasterio.open(path) as ds:
  check_geo(ds);v=ds.read_masks(1)>0
  arr=None
  if ds.dtypes[0].startswith(('float','complex')) or not png:arr=ds.read(1);v&=np.isfinite(arr)
 if png:im=np.asarray(Image.open(png));settings=dict(reused_existing_native_display=True)
 else:im,settings=display(arr,v,units=units)
 return None,v,im,settings

def data(a,metadata):
 p=Path(a['path']);png=a.get('display')
 if not png:
  options=[p.parent/'native_db.png',p.parent/('context_db'+p.stem.replace('context','')+'.png')]
  png=next((str(q) for q in options if q.exists()),'')
 return filedata(a['path'],metadata.get('input_units','linear_amplitude'),png)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--max-new',type=int,default=90);ap.add_argument('--max-per-relation',type=int,default=12);ap.add_argument('--required-only',action='store_true');args=ap.parse_args()
 t=stamp();bank=assets();metadata={a['acquisition_id']:a for a in rows(latest('MERGED_SOURCE_INVENTORY_*.jsonl'))};rels=relations(bank,metadata)
 existing=rows(latest('EXISTING512_REFERENCES_*.jsonl'))
 for p in sorted(ROOT.glob('BATCH_PAIRS_*.jsonl')):existing+=rows(p)
 existing_by_rel=defaultdict(list)
 for r in existing:existing_by_rel[r['relation_id']].append(r)
 queue=[]
 priorities={relid(q['acquisition_ids']) for q in json.loads((AUDIT/'TARGETED_SOURCE_PROPOSALS_20261014_2329.json').read_text())}
 for r in rels:
  if not all(cid in bank for cid in r['acquisition_ids']):continue
  if len(existing_by_rel[r['relation_id']])>=args.max_per_relation:continue
  r=dict(r);ids=sorted(r['acquisition_ids']);r['acquisition_ids']=ids
  choices=[max(bank[cid],key=lambda a:a['native_info']['width']*a['native_info']['height']) for cid in ids]
  z=[a['native_info'] for a in choices];crs=metric_crs(z[0]);over=poly_intersection(native_polygon(choices[0],crs),native_polygon(choices[1],crs))
  if poly_area(over)<100:continue
  r.update(asset_choices=choices,component=component(r['site']),split=metadata[ids[0]]['split'],context_geographic_intersection_m2=poly_area(over),metric_crs=crs)
  assert metadata[ids[0]]['split']==metadata[ids[1]]['split']
  angle=r.get('azimuth_delta_deg') or 0
  r['priority']=r['relation_id'] in priorities or 74<angle<77 or 88<angle<89
  if args.required_only and not (r['priority'] or r['site']=='Port of Hamburg, Germany'):continue
  queue.append(r)
 # Round robin site/bin, do not let Rotterdam combinatorics dominate.
 bins=['0-15','15-30','30-60','60-90','90-120','>120','unknown'];group=defaultdict(list)
 for r in queue:group[(r['site'],r['angle_bin'])].append(r)
 for rs in group.values():rs.sort(key=lambda r:(not r['priority'],r.get('time_gap_days',999),r['relation_id']))
 ordered=[]
 while any(group.values()):
  for k in sorted(group,key=lambda k:(bins.index(k[1]),k[0])):
   if group[k]:ordered.append(group[k].pop(0))
 events=[];proposals=[]
 for ir,r in enumerate(ordered):
  cid0,cid1=r['acquisition_ids'];aa=r['asset_choices'];dd=[data(a,metadata[cid]) for a,cid in zip(aa,r['acquisition_ids'])];z=[a['native_info'] for a in aa]
  n=512
  if min(z[0]['width'],z[0]['height'],z[1]['width'],z[1]['height'])<n:continue
  step=640
  xs=list(range(0,z[0]['width']-n+1,step));ys=list(range(0,z[0]['height']-n+1,step))
  candidates=[];mapper=NativeMap(*aa)
  for y in ys:
   for x in xs:
    q=mapper([[x+255.5,y+255.5]])[0];x1,y1=np.rint(q-255.5).astype(int)
    if min(x1,y1)<0 or x1+n>z[1]['width'] or y1+n>z[1]['height']:continue
    wins=[[x,y,n,n],[int(x1),int(y1),n,n]];valid=[d[1][w[1]:w[1]+n,w[0]:w[0]+n] for d,w in zip(dd,wins)]
    if min(float(v.mean()) for v in valid)<.98:continue
    patches=[d[2][w[1]:w[1]+n,w[0]:w[0]+n] for d,w in zip(dd,wins)]
    if min(float(im[v].std()) for im,v in zip(patches,valid))<.5:continue
    origins=[a.get('product_origin_xy') for a in aa]
    if any(o is None for o in origins):continue
    product_wins=[[int(w[0]+o[0]),int(w[1]+o[1]),n,n] for w,o in zip(wins,origins)]
    dup=False;max_old_overlap=[0.,0.]
    for old in existing_by_rel[r['relation_id']]:
     oldmap={c:v for c,v in zip(old['acquisition_ids'],old['views'])}
     ov=[]
     for cid,w in zip(r['acquisition_ids'],product_wins):
      v=oldmap[cid];o=v.get('product_origin_xy');zz=v['native_info']
      ov.append(roi_overlap(w,[*o,zz['width'],zz['height']]) if o is not None else 0.)
     max_old_overlap=np.maximum(max_old_overlap,ov).tolist()
     if max(ov)>.2:dup=True;break
    if dup:continue
    stats=[preview_score(im,v) for im,v in zip(patches,valid)]
    # Coarse structure support ranks regions for review; never certifies physical points.
    score=min(s['coarse_supported_cells'] for s in stats)+min(s['coarse_contrast_std'] for s in stats)
    c=dict(relation=r,windows=wins,original_product_windows=product_wins,coarse_proxy=stats,score=score,max_overlap_with_existing_pair=max_old_overlap)
    candidates.append(c)
  candidates.sort(key=lambda c:(-c['score'],c['windows']))
  mapper.close()
  chosen=[]
  target=min(args.max_per_relation-len(existing_by_rel[r['relation_id']]),12)
  for c in candidates:
   if len(chosen)>=max(1,target-1):break
   if any(max(roi_overlap(w,ow) for w,ow in zip(c['windows'],p['windows']))>.2 for p in chosen):continue
   chosen.append(c)
  # Retain a low-structure tail for honest hard cases, rather than deleting all sea.
  if candidates and len(chosen)<target:
   for c in reversed(candidates):
    if all(max(roi_overlap(w,ow) for w,ow in zip(c['windows'],p['windows']))<=.2 for p in chosen):chosen.append(c);break
  for rank,c in enumerate(chosen):c['rank_in_relation']=rank;proposals.append(c)
  events.append(dict(relation_id=r['relation_id'],bounded_grid_proposals=len(xs)*len(ys),technical_distinct_candidates=len(candidates),chosen=len(chosen)))
  if (ir+1)%5==0:print('FROZEN_GRID_SEARCH',ir+1,'of',len(ordered),'planned distinct candidates',len(proposals),flush=True)
 # Fair coverage first: a few spatial regions per relation before many in one relation.
 proposals.sort(key=lambda c:(c['rank_in_relation'],not c['relation']['priority'],bins.index(c['relation']['angle_bin']),c['relation']['site'],c['relation']['relation_id']))
 proposals=proposals[:args.max_new]
 frozen=[]
 for i,c in enumerate(proposals):
  r=c['relation'];rid=r['relation_id'];digest=hashlib.sha256(json.dumps([rid,c['original_product_windows']]).encode()).hexdigest()[:14]
  sid='N512_'+digest;folder=ROOT/'pairs'/sid
  if folder.exists():raise FileExistsError('Frozen candidate exists; never regenerate '+sid)
  c['sample_id']=sid;frozen.append(dict(sample_id=sid,relation_id=rid,acquisition_ids=r['acquisition_ids'],native_assets=[a['path'] for a in r['asset_choices']],windows=c['windows'],product_windows=c['original_product_windows'],rank=c['rank_in_relation'],selection='fixed 640px native grid, two-view coarse texture support, validity and spatial separation; no matcher or labels; native geography only',annotatability='E',human_GT=0))
 plan=ROOT/f'FROZEN_CROP_PLAN_{t}.jsonl';jsonl(plan,frozen)
 output=[]
 for i,c in enumerate(proposals):
  r=c['relation'];folder=ROOT/'pairs'/c['sample_id'];folder.mkdir(parents=True);views=[];ims=[]
  for side,(a,w,cid) in enumerate(zip(r['asset_choices'],c['windows'],r['acquisition_ids'])):
   x,y,n,_=w
   with rasterio.open(a['path']) as ds:
    arr=ds.read(1,window=Window(*w));valid=(ds.read_masks(1,window=Window(*w))>0)&np.isfinite(arr);z=info(ds);z.update(width=n,height=n,transform=list(ds.window_transform(Window(*w)))[:6])
   # Preserve raw values. Display conversion is pointwise only, without image warping.
   im,settings=display(arr,valid,units=metadata[cid].get('input_units','linear_amplitude'));ims.append(im)
   raw=folder/f'side{side}_native.tif';png=folder/f'side{side}.png';mask=folder/f'side{side}_valid.png'
   write_tif(raw,arr,valid,z);Image.fromarray(im).save(png);Image.fromarray(valid.astype('uint8')*255).save(mask)
   views.append(dict(acquisition_id=cid,path=str(raw),native_info=z,geocoding_diagnostic=a.get('geocoding_diagnostic',{}),source_native_path=a['path'],source_local_window_xywh=w,product_origin_xy=c['original_product_windows'][side][:2],raw_pixel_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),raw_file_sha256=sha(raw),image=str(png),image_sha256=sha(png),valid_mask=str(mask),valid_fraction=float(valid.mean()),display_settings=settings,pixel_spacing=geo.metric_geometry(z),resolution_metadata=metadata[cid].get('resolution'),original_product=a.get('original_profile'),source_metadata=a.get('metadata'),unresampled_window=True))
  z=[v['native_info'] for v in views];crs=r['metric_crs'];p=[native_polygon(v,crs) for v in views];inter=poly_intersection(*p);area=poly_area(inter);frac=[area/poly_area(pp) for pp in p]
  pts=np.array([[0,0],[255.5,255.5],[511,511]],float);mapper=NativeMap(*views);err=np.linalg.norm(mapper.reverse()(mapper(pts))-pts,axis=1);mapper.close()
  assert err.max()<.001
  rec=dict(sample_id=c['sample_id'],relation_id=r['relation_id'],site=r['site'],sensor=r['sensor'],acquisition_ids=r['acquisition_ids'],component=r['component'],split=r['split'],split_reason='station/atomic-acquisition component isolation; old diagnosed station Train',source_kind='new_real_native_crop',azimuth_delta_deg=r['azimuth_delta_deg'],angle_bin=r['angle_bin'],time_gap_days=r.get('time_gap_days',abs((utc(metadata[r['acquisition_ids'][0]]['datetime'])-utc(metadata[r['acquisition_ids'][1]]['datetime'])).total_seconds())/86400),views=views,image0=views[0]['image'],image1=views[1]['image'],image_only_input=[views[0]['image'],views[1]['image']],candidate_status='frozen_before_human_annotation',annotatability='E',annotatability_basis='coarse structure proxy and geography require actual physical review; not A certification',coarse_structure_proxy=c['coarse_proxy'],geographic_FOV=dict(area_m2=area,fractions=frac,metric_crs=crs,NOT_physical_GT=True,numeric_roundtrip_max_px=float(err.max())),spatial_selection=dict(native_stride_px=640,max_existing_same_relation_iou=c['max_overlap_with_existing_pair'],max_allowed_same_relation_side_iou=.2),GT_qualified=False,independent_GT=False,gt_state='unannotated',allow_supervised_training=False,model_selection_used=False,geometry_resampling_applied=False,prior_geographic_or_algorithm_exposure=False)
  # Admin preview only; never distributed with blind task packages.
  preview=Image.new('RGB',(1040,550),(15,20,28));d=ImageDraw.Draw(preview);d.text((8,5),f"{rec['sample_id']} | {rec['site']} | {rec['azimuth_delta_deg']} deg | E: physical review pending",fill='white')
  for j,im in enumerate(ims):preview.paste(Image.fromarray(im).convert('RGB'),(j*528,30))
  preview.save(folder/f'ADMIN_PREVIEW_{t}.jpg');save(folder/f'PROVENANCE_{t}.json',rec);output.append(rec)
  if (i+1)%25==0:print('NEW_NATIVE_PAIRS',i+1,'of',len(proposals),flush=True)
 jsonl(ROOT/f'BATCH_PAIRS_{t}.jsonl',output);jsonl(ROOT/f'CROP_SEARCH_EVENTS_{t}.jsonl',events)
 jsonl(ROOT/f'RELATION_POOL_{t}.jsonl',[{k:v for k,v in r.items() if k!='asset_choices'} for r in ordered])
 save(ROOT/f'BATCH_SUMMARY_{t}.json',dict(at=now(),existing_pairs_before=len(existing),new_native_pairs=len(output),relations=len({r['relation_id'] for r in output}),sites=len({r['site'] for r in output}),angle_bins=dict(Counter(r['angle_bin'] for r in output)),independent_GT=0,plan=str(plan),frozen_no_recrop=True))
 print('BATCH_DONE',len(output),'PAIR_MANIFEST',str(ROOT/f'BATCH_PAIRS_{t}.jsonl'),flush=True)

if __name__=='__main__':main()
