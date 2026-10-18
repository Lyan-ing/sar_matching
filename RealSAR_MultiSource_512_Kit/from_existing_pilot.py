#!/usr/bin/env python3
"""Make a source manifest from an existing validated real Sentinel-1 RTC pilot download.

This is for a smoke run only: multiple cropped dates from one site are NOT independent
benchmark scenes and cannot support precise pixel GT by map alignment alone.
"""
import argparse,csv,json
from pathlib import Path
from itertools import combinations


def convert(manifest,output,site_id='S1_14TPN_PILOT',split='train'):
    src=Path(manifest).resolve()
    d=json.loads(src.read_text())
    scenes=d.get('scenes',[])
    if len(scenes)<2:raise ValueError('Need at least two verified dates')
    records=[]
    def identity(scene):
        tags=scene.get('raster_metadata',{}).get('source_tags',{})
        raw=tags.get('SCENE_1_METADATA')
        if not raw:raise ValueError('Verified source metadata with exact acquisition time required')
        info=json.loads(raw) if isinstance(raw,str) else raw
        return info['title'], info['start_time'].rstrip('Z')+'Z'
    for i,(a,b) in enumerate(combinations(scenes,2)):
        if a['scene_name']==b['scene_name']:continue
        ap=src.parent/a['geotiff'];bp=src.parent/b['geotiff']
        if not ap.exists() or not bp.exists():raise FileNotFoundError('Expected pilot GeoTIFF missing')
        aid,at=identity(a);bid,bt=identity(b)
        records.append(dict(pair_id=f'PILOT_{i:03d}',site_id=site_id,split=split,
          sensor='Sentinel-1',reference_acquisition_id=aid,
          moving_acquisition_id=bid,reference_path=str(ap),moving_path=str(bp),
          reference_utc=at,moving_utc=bt,
          input_units='linear_power',product_level='RTC_Gamma0_VV',
          look_pair_category='repeat_pass_same_look',
          source_metadata_status='PREVIOUS_PILOT_SAFE_VERIFIED',
          license='https://www.indigoag.com/forms/atlas-sentinel-license'))
    out=Path(output)
    if out.exists():raise FileExistsError('Refuse overwrite '+str(out))
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    print('Wrote',len(records),'candidates (NOT independent pixel GT)',out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-manifest',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--site-id',default='S1_14TPN_PILOT')
    p.add_argument('--split',default='train')
    a=p.parse_args()
    convert(a.pilot_manifest,a.out,a.site_id,a.split)

if __name__=='__main__':main()
