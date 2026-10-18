"""One disclosed path-style S3 recovery for empty TLS-failed short-gap assets."""
import sys
sys.dont_write_bytecode=True
import numpy as np
from rasterio.warp import transform
from build_windows import ROOT,SELECTED,fetch_window,event
from audit import footprint
from geo_tools import lines,poly_intersection,save

def run():
    ids=['d2524701-cd7a-4f1d-a368-13ee0b9a1d77','51994e66-c41c-465b-aebf-d4f9b865d43d','c808e026-5291-4dba-9eb8-4ee20443f2ce']
    events=lines(ROOT/'window_events.jsonl');failed={r.get('acquisition_id') for r in events if r['status']=='failed'}
    assert set(ids)<=failed,'Only previously failed sources; never race an active first attempt'
    rows={r['pair_id']:r for r in lines(ROOT/'pair_candidates.jsonl')};selected=[rows[k] for k in SELECTED]
    allv={v['acquisition_id']:v for r in selected for v in r['views']};results=[]
    save(ROOT/'evidence/one_path_style_retry_plan.json',dict(acquisition_ids=ids,max_additional_attempts=1,
        reason='TLS failure before any saved pixels; preserve short temporal baseline across angular strata',
        virtual_host='umbra-open-data-catalog.s3.us-west-2.amazonaws.com',path_style_host='s3.us-west-2.amazonaws.com/umbra-open-data-catalog',
        invariant='same bucket, object key and true acquisition; no new identity or alternate easier pair'))
    for aid in ids:
        folder=ROOT/'assets'/aid
        assert folder.exists() and not list(folder.iterdir()),'Failed folder must contain no saved pixels'
        v=allv[aid];vs={a['acquisition_id']:a for r in selected for a in r['views'] if r['site_name']==v['site_name']};joint=None;crs=None
        for a in vs.values():
            if crs is None:p,crs=footprint(a)
            else:
                q=np.asarray(a['geometry']['coordinates'][0])[:,:2];x,y=transform('EPSG:4326',crs,q[:,0].tolist(),q[:,1].tolist());p=np.c_[x,y]
            joint=p if joint is None else poly_intersection(joint,p)
        center=(joint.min(0)+joint.max(0))/2
        alias=v['asset_url'].replace('https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/',
            'https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog/')
        assert alias!=v['asset_url']
        try:
            meta=fetch_window(dict(v,canonical_asset_url=v['asset_url'],asset_url=alias,same_bucket_key_alias=True),center,crs)
            results.append(dict(acquisition_id=aid,recovered=True,raw_sha256=meta['raw_sha256']))
        except Exception as e:
            result=dict(acquisition_id=aid,recovered=False,error=repr(e));results.append(result);event(dict(status='path_style_retry_failed',**result));print(result,flush=True)
    save(ROOT/'path_style_retry_results.json',results)

if __name__=='__main__':run()
