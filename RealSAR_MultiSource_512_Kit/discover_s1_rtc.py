#!/usr/bin/env python3
"""Anonymous Sentinel-1 RTC listing and *candidate* real-acquisition pairing.

Use with Indigo RTC public COG catalog. Candidate scene IDs and acquisition timestamps must
be reconciled against the original productInfo.json/manifest.safe before dataset release.
"""
from __future__ import annotations
import argparse,csv,json
from datetime import datetime
from pathlib import Path
import requests
from download_s1_rtc import list_tile_keys,scene_from_key

COLUMNS=['pair_id','site_id','split','sensor','reference_acquisition_id','moving_acquisition_id',
 'reference_path','moving_path','reference_utc','moving_utc','input_units','product_level',
 'look_pair_category','source_metadata_status','license']


def distinct_observations(scenes):
    unique={}
    for scene in scenes:
        key=(scene['satellite'],scene['date'],scene['orbit_direction'])
        unique.setdefault(key,scene)
    return sorted(unique.values(),key=lambda x:(x['date'],x['scene_name']))


def candidate_pairs(scenes,max_pairs=12,min_days=12,max_days=366,seed=42):
    """Diversify temporal separations. Coverage still must be verified by crop builder."""
    import numpy as np
    dates=[datetime.strptime(s['date'],'%Y%m%d').date() for s in scenes]
    edges=[]
    for i in range(len(scenes)):
        for j in range(i+1,len(scenes)):
            delta=abs((dates[i]-dates[j]).days)
            if min_days<=delta<=max_days:
                variation=int(scenes[i]['orbit_direction']!=scenes[j]['orbit_direction'])
                edges.append((variation,delta,i,j))
    rng=np.random.default_rng(seed)
    rng.shuffle(edges)
    edges.sort(key=lambda t:(-t[0],-abs(t[1]-45)))
    selected=[]; counts={}
    for same,days,i,j in edges:
        if len(selected)>=max_pairs:break
        if counts.get(i,0)>=2 or counts.get(j,0)>=2:continue
        selected.append((scenes[i],scenes[j],days))
        counts[i]=counts.get(i,0)+1; counts[j]=counts.get(j,0)+1
    return selected


def discover(tiles,years,maximum,min_days,split,output,timeout,seed):
    s=requests.Session()
    s.headers['User-Agent']='RealSAR-MultiSource512/0.2 (open SAR research)'
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    allrows=[]; audit=[]
    for tile in tiles:
        scenes=[]
        for year in years:
            for key in list_tile_keys(s,tile,year,timeout):
                obj=scene_from_key(key,tile)
                if obj:scenes.append(obj)
        scenes=distinct_observations(scenes)
        pairs=candidate_pairs(scenes,maximum,min_days,366,seed)
        audit.append({'tile':tile,'years':years,'available_scenes':len(scenes),
                      'pairs_proposed':len(pairs),'orbit_directions':sorted(set(v['orbit_direction'] for v in scenes)),
                      'status':'SCENE_NAME_DATE_LEVEL_ONLY__REQUIRES_ORIGINAL_PRODUCT_ID_AND_COVERAGE_AUDIT'})
        (output/f'{tile}_scenes.json').write_text(json.dumps(scenes,ensure_ascii=False,indent=2))
        for seq,(a,b,delta) in enumerate(pairs):
            allrows.append(dict(pair_id=f'S1_{tile}_{seq:04d}',site_id=f'S1_{tile}',
             split=split,sensor='Sentinel-1',reference_acquisition_id=a['scene_name'],
             moving_acquisition_id=b['scene_name'],reference_path=a['url'],moving_path=b['url'],
             reference_utc=a['date']+'T00:00:00Z',moving_utc=b['date']+'T00:00:00Z',
             input_units='linear_power',product_level='RTC_Gamma0_VV',
             look_pair_category=('opposite_pass_candidate' if a['orbit_direction']!=b['orbit_direction']
                                 else 'repeat_pass_same_direction'),
             source_metadata_status='DATE_TIME_PLACEHOLDER__VERIFY_SAFE_IDS_UTC_ORBIT_BEFORE_RELEASE',
             license='Indigo_Atlas_Sentinel_License_verify_before_redistribution'))
    with (output/'pairs_manifest_CANDIDATE.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS)
        w.writeheader();w.writerows(allrows)
    (output/'discovery_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',nargs='+',default=['14TPN'],help='Verified MGRS tile codes; only 14TPN was confirmed in previous run')
    p.add_argument('--years',nargs='+',type=int,default=[2020])
    p.add_argument('--max-pairs-per-tile',type=int,default=12)
    p.add_argument('--min-days',type=int,default=12)
    p.add_argument('--split',choices=['train','dev','val','test','confirmation'],default='train')
    p.add_argument('--timeout',type=int,default=40)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    if len(set(a.tiles))!=len(a.tiles) or any(len(t)!=5 for t in a.tiles):p.error('MGRS tile codes must be 5 chars and unique')
    print(json.dumps(discover(a.tiles,a.years,a.max_pairs_per_tile,a.min_days,
                 a.split,a.out,a.timeout,a.seed),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
