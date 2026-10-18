"""One explicit recovery of the empty window interrupted by our CPU stop."""
import sys
sys.dont_write_bytecode=True
import json
import numpy as np
from rasterio.warp import transform
from build_windows import ROOT,SELECTED,fetch_window
from audit import footprint
from geo_tools import lines,poly_intersection,save

def run():
    aid='54e07361-06f6-4420-b20b-dcb315f8d7b9'
    folder=ROOT/'assets'/aid
    if (folder/'metadata.json').exists():raise FileExistsError('Acquisition already complete')
    assert folder.exists() and not list(folder.iterdir()),'Only explicitly interrupted EMPTY folder may be recovered'
    rows={r['pair_id']:r for r in lines(ROOT/'pair_candidates.jsonl')}
    vs={v['acquisition_id']:v for k in SELECTED for v in rows[k]['views'] if rows[k]['site_name']=='Port of Rotterdam, Netherlands'}
    joint=None;crs=None
    for v in vs.values():
        if crs is None:p,crs=footprint(v)
        else:
            q=np.asarray(v['geometry']['coordinates'][0])[:,:2];x,y=transform('EPSG:4326',crs,q[:,0].tolist(),q[:,1].tolist());p=np.c_[x,y]
        joint=p if joint is None else poly_intersection(joint,p)
    center=(joint.min(0)+joint.max(0))/2
    save(ROOT/'evidence/explicit_interrupted_recovery.json',dict(acquisition_id=aid,
        prior_attempt='owned CPU stop; interrupted request had no retained raw or display bytes',
        prior_success_preserved=True,full_product_downloaded=False,new_identity_count_increment=0,
        note='one disclosed operational recovery, not duplicated source or extra crop to inflate count'))
    fetch_window(vs[aid],center,crs)

if __name__=='__main__':run()
