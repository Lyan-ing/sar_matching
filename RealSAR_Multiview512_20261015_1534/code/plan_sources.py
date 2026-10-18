"""One bounded source expansion: temporal/angle/site coverage before pixel selection."""
from common import *
from collections import defaultdict,Counter
from itertools import combinations

def main():
 t=stamp();aa=rows(latest('ACQUISITION_INVENTORY_*.jsonl'))+rows(latest('NEW_METADATA_INVENTORY_*.jsonl'));acqs={a['acquisition_id']:a for a in aa}
 jsonl(ROOT/f'MERGED_SOURCE_INVENTORY_{t}.jsonl',sorted(acqs.values(),key=lambda a:a['acquisition_id']))
 local=rows(latest('LOCAL_NATIVE_ASSETS_*.jsonl'))+[json.loads(p.read_text()) for p in (ROOT/'acquisitions').glob('*/ASSET_*.json')]
 have={a['acquisition_id'] for a in local};chosen=set(have);priority=[]
 props=json.loads((AUDIT/'TARGETED_SOURCE_PROPOSALS_20261014_2329.json').read_text())
 for r in props:
  priority.append(dict(acquisition_ids=r['acquisition_ids'],site=r['site'],reason='explicit requested pending relationship',angle=r['azimuth_delta_deg']))
  chosen.update(r['acquisition_ids'])
 sites=defaultdict(list)
 for a in acqs.values():
  if a['provider']=='Umbra' and (a['site_name'].startswith('Port of ') or a['site_name']=='Panama Canal, Panama'):sites[a['site_name']].append(a)
 selection=[]
 for site,rr in sorted(sites.items()):
  candidates=[]
  for a,b in combinations(rr,2):
   d=delta(a.get('azimuth_degrees'),b.get('azimuth_degrees'));days=abs((utc(a['datetime'])-utc(b['datetime'])).total_seconds())/86400
   if d is None or not .5<=d<=120 or days>30:continue
   pa,crs=footprint(a);pb,_=footprint(b,crs);joint=poly_intersection(pa,pb)
   if poly_area(joint)<.6*min(poly_area(pa),poly_area(pb)):continue
   candidates.append(dict(ids=[a['acquisition_id'],b['acquisition_id']],delta=d,bin=binof(d),days=days))
  bins=Counter();used={x for x in chosen if acqs[x]['site_name']==site};cap=max(len(used),8 if next(iter(rr))['split']=='Val' else 8)
  # Cover missing bins first, then short temporal gaps; source budget remains fixed.
  selected=[]
  for _ in range(14):
   feasible=[r for r in candidates if r not in selected and len(used|set(r['ids']))<=cap]
   if not feasible:break
   def score(r):
    fresh=len(set(r['ids'])-used)
    return (bins[r['bin']]>0,bins[r['bin']],r['days']>2,r['days'],fresh,tuple(r['ids']))
   best=min(feasible,key=score);selected.append(best);used.update(best['ids']);bins[best['bin']]+=1
  chosen.update(used);selection.append(dict(site=site,metadata_acquisitions=len(rr),selected_ids=sorted(used),relation_seed_candidates=selected))
 centers={}
 for site in sites:
  rr=[acqs[x] for x in chosen if x in acqs and acqs[x]['site_name']==site]
  # A shared geographic context is only a window search hint, never a physical correspondence.
  points=[np.asarray(a['geometry']['coordinates'][0],float)[:,:2].mean(0) for a in rr if a.get('geometry')]
  if points:centers[site]=np.median(points,axis=0).tolist()
 plan=[]
 for cid in sorted(chosen-have):
  a=acqs[cid]
  if a['provider']!='Umbra' or a['site_name'] not in centers:continue
  plan.append(dict(acquisition_id=cid,center_lonlat=centers[a['site_name']],side=4096,reason='bounded source and angle coverage; original view orientation; no matching scores'))
 if len(plan)>85:raise ValueError('Bounded new acquisition cap exceeded')
 save(ROOT/f'EXPANSION_SOURCE_PLAN_{t}.json',dict(at=now(),assets=plan,estimated_uncompressed_window_bytes=sum(p['side']**2 for p in plan),max_new_assets=85,storage_cap_gib=20,priority_relations=priority,site_selection=selection,site_centers_lonlat=centers,selection_uses_only_identity_time_footprint_angles=True,GT=0))
 print('EXPANSION_PLAN',len(plan),'new assets','sites',len(selection),flush=True)
 for r in selection:print(r['site'],len(r['selected_ids']),dict(Counter(x['bin'] for x in r['relation_seed_candidates'])),flush=True)

if __name__=='__main__':main()
