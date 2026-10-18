"""True acquisition deduplication and candidate geometry inventory, never pixel GT."""
import sys
sys.dont_write_bytecode=True
import json, hashlib, itertools, math, re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from rasterio.warp import transform

ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
V2=BASE/'RealSAR_MultiViewV2_20261012_1259'
sys.path.insert(0,str(V2/'code'))
from geo_tools import poly_area,poly_intersection,save,jsonl,sha
from build import angular,gap,utc

def bin_angle(d):
    if d is None:return 'unknown'
    return '0-20' if d<20 else '20-40' if d<40 else '40-60' if d<60 else '60-90' if d<=90 else '>90'

def footprint(r):
    points=np.asarray(r['geometry']['coordinates'][0],float)[:,:2]
    lon,lat=points.mean(0); zone=int(np.clip(math.floor((lon+180)/6)+1,1,60))
    crs=f'EPSG:{(32600 if lat>=0 else 32700)+zone}'
    x,y=transform('EPSG:4326',crs,points[:,0].tolist(),points[:,1].tolist())
    return np.c_[x,y],crs

def key(r): return (r['provider'].lower(),r['acquisition_id'])

def pair_key(a,b):return tuple(sorted([key(a),key(b)]))

def run():
    paths=[BASE/'real_sar_pilot/multisource512_20261010_2352/sources.jsonl',
           BASE/'real_sar_pilot/multisource512_20261010_2352/downloaded_acquisitions.jsonl',
           BASE/'real_sar_pilot/multisource512_20261010_2352/umbra_entities_v2/inventory.jsonl',ROOT/'discovery/inventory.jsonl']
    old=json.loads((V2/'source_isolation.json').read_text())['pair_component']
    original=[json.loads(s) for s in (V2/'acquisition_pair_inventory.jsonl').read_text().splitlines()]
    old_pairs={}; components={}; old_sites=set()
    for r in original:
        provider={'Sentinel-1':'sentinel1','Capella':'capella','UmbraZero':'umbra'}.get(r['sensor'],r['sensor'].lower())
        ids=tuple(sorted((provider,v['acquisition_id']) for v in r['views']))
        old_pairs[ids]=r['pair_id']; old_sites.add(r['site_name'])
        components[r['site_name']]=old[r['pair_id']]
    records={}; aliases=defaultdict(list); conflicts=[]; excluded=[]
    now=datetime.now(ZoneInfo('Asia/Shanghai'))
    for p in paths:
        for s in p.read_text().splitlines():
            r=json.loads(s);k=key(r)
            if utc(r['datetime'])>now:
                excluded.append(dict(source=str(p),id=r['acquisition_id'],reason='future_acquisition_datetime'));continue
            aliases[k].append(str(p))
            if k in records:
                prior=records[k]
                for field in ['datetime','azimuth_degrees','incidence_degrees']:
                    if prior.get(field) is not None and r.get(field) is not None:
                        if field=='datetime': same=abs((utc(prior[field])-utc(r[field])).total_seconds())<2
                        else: same=abs(prior[field]-r[field])<1e-5
                        if not same:conflicts.append(dict(id=k,field=field,values=[prior[field],r[field]]))
                continue
            r=dict(r);r['metadata_inventory_source']=str(p);records[k]=r
    for k,r in records.items():
        r['inventory_aliases']=aliases[k]
        r['component']=components.get(r['site_name'],'angle_site_'+hashlib.sha256(r['site_name'].casefold().encode()).hexdigest()[:10])
        r['split']='diagnostic_development_seen' if r['site_name'] in old_sites else 'new_metadata_observed_development'
    groups=defaultdict(list)
    for r in records.values(): groups[r['site_name'],r['provider'].lower()].append(r)
    pairs=[]
    for (site,provider),rs in sorted(groups.items()):
        for a,b in itertools.combinations(sorted(rs,key=lambda r:r['datetime']),2):
            if a['acquisition_id']==b['acquisition_id']:continue
            if not a.get('geometry') or not b.get('geometry'):continue
            pa,crs=footprint(a);pts=np.asarray(b['geometry']['coordinates'][0],float)[:,:2]
            x,y=transform('EPSG:4326',crs,pts[:,0].tolist(),pts[:,1].tolist());pb=np.c_[x,y]
            intersection=poly_intersection(pa,pb);area=poly_area(intersection)
            fractions=[area/max(poly_area(p),1e-9) for p in [pa,pb]]
            if min(fractions)<.25:continue
            d=angular(a.get('azimuth_degrees'),b.get('azimuth_degrees'));days=gap(a['datetime'],b['datetime'])
            pk=pair_key(a,b);pid='ANGLE_'+hashlib.sha256(json.dumps(pk).encode()).hexdigest()[:12]
            incs=[r.get('incidence_degrees') for r in [a,b]]
            incidence_delta=abs(incs[0]-incs[1]) if all(i is not None for i in incs) else None
            states={f:[str(r.get(f) or 'unknown').casefold() for r in [a,b]] for f in ['orbit_direction','observation_direction']}
            pairs.append(dict(pair_id=pid,legacy_pair_id=old_pairs.get(pk),site_name=site,sensor=a['provider'],
                component=a['component'],split=a['split'],views=[a,b],days=days,azimuth_delta_deg=d,
                angle_bin=bin_angle(d),incidence_delta_deg=incidence_delta,orbit_look_states=states,
                orbit_changed=states['orbit_direction'][0]!=states['orbit_direction'][1],
                left_right_changed=states['observation_direction'][0]!=states['observation_direction'][1],
                look_pair_category='azimuth_difference_metadata_only',
                metric_crs=crs,intersection_metric=intersection.tolist(),product_overlap_fractions=fractions,
                geography_only=True,gt_qualified=False,raster_status='WINDOW_READ_REQUIRED',
                angle_definition='minimum circular azimuth difference, not a complete 3D viewpoint angle'))
    pairs.sort(key=lambda r:(r['days'],r['pair_id']))
    summary=dict(at=now.isoformat(),acquisitions=len(records),original_pairs=len(original),
        original_angle_bins=dict(Counter(bin_angle(r['azimuth_delta_deg']) for r in original)),
        all_metadata_pairs=len(pairs),all_metadata_bins=dict(Counter(r['angle_bin'] for r in pairs)),
        new_pair_identities=sum(r['legacy_pair_id'] is None for r in pairs),
        short_gap_14d_bins=dict(Counter(r['angle_bin'] for r in pairs if r['days']<=14)),
        harbor_short_gap_14d_bins=dict(Counter(r['angle_bin'] for r in pairs if r['days']<=14 and ('Port of' in r['site_name'] or 'Panama' in r['site_name']))),
        original_GT=0,new_GT=0,metadata_conflicts=conflicts,exclusions=excluded,
        raw_inventory_hashes={str(p):sha(p) for p in paths},no_chip_counts=True,
        footprint_method='projected provider footprints convex intersection; screening geography only')
    jsonl(ROOT/'acquisition_inventory.jsonl',sorted(records.values(),key=key));jsonl(ROOT/'pair_candidates.jsonl',pairs);save(ROOT/'angle_audit.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['raw_inventory_hashes','metadata_conflicts']},indent=2))
    for bn in ['0-20','20-40','40-60','60-90','>90']:
        selected=[r for r in pairs if r['angle_bin']==bn and ('Port of' in r['site_name'] or 'Panama' in r['site_name'])]
        print('BIN',bn)
        for r in selected[:8]:print(r['pair_id'],r['site_name'],round(r['azimuth_delta_deg'],3),round(r['days'],5),r['legacy_pair_id'])

if __name__=='__main__':run()
