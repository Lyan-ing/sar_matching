"""Focused real-asset evidence and gate checks; no matcher, no synthetic GT."""
import sys
sys.dont_write_bytecode=True
import json,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,rasterio
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'RealSAR_MultiViewV2_20261012_1259/code'))
from geo_tools import lines,sha,save
from readability_server import Reviews
from prepare_gt import prepare
sys.path.insert(0,str(ROOT/'code'))
from audit import bin_angle,pair_key
sys.path.insert(0,str(BASE/'RealSAR_IndependentBenchmark_20261011_2015'))
from benchmark_quality import quality

def run():
    checks=[]
    assert [bin_angle(x) for x in [0,19.999,20,39.999,40,59.999,60,90,90.001,None]]==['0-20','0-20','20-40','20-40','40-60','40-60','60-90','60-90','>90','unknown']
    checks.append('half-open angle boundaries, 90 included in 60-90, unknown retained')
    allpairs=lines(ROOT/'pair_candidates.jsonl');seen=set()
    for r in allpairs:
        k=pair_key(*r['views']);assert k not in seen;seen.add(k)
        assert r['views'][0]['acquisition_id']!=r['views'][1]['acquisition_id'] and not r['gt_qualified']
    checks.append(f'{len(allpairs)} unordered true-ID pairs unique; no metadata GT')
    regions=lines(ROOT/'candidate_regions_complete.jsonl');assets={}
    for r in regions:
        assert not r['gt_qualified'] and not r['independent_GT']
        for v in r['views']:
            assets[v['acquisition_id']]=v
            p=Path(v['native_tif']);assert sha(p)==v['native_sha256']
            with rasterio.open(p) as ds:
                assert (ds.width,ds.height)==(v['native_info']['width'],v['native_info']['height'])
                assert np.allclose(list(ds.transform)[:6],v['native_info']['transform'],atol=1e-10,rtol=0)
                assert ds.crs.to_string()==v['native_info']['crs']
                # Validate actual stored mask support in four native blocks.
                for x,y in [(0,0),(ds.width-32,0),(0,ds.height-32),(ds.width//2,ds.height//2)]:
                    a=ds.read(1,window=((y,y+32),(x,x+32)));m=ds.read_masks(1,window=((y,y+32),(x,x+32)))>0
                    assert np.isfinite(a[m]).all()
            for kind in ['db','linear']:
                with Image.open(ROOT/'regions'/r['sample_id']/f'native_{kind}{v["side"]}.png') as im:
                    assert im.size==(v['native_info']['width'],v['native_info']['height'])
    checks.append(f'{len(assets)} unique native rasters hashed; dimensions/affines/CRS/mask blocks and paired displays agree')
    hashes=[json.loads((ROOT/'assets'/k/'metadata.json').read_text())['pixel_value_sha256'] for k in assets]
    assert len(hashes)==len(set(hashes));checks.append('no pixel-identical new raw windows')
    review=Reviews(ROOT/'review')
    for sid in review.tasks:assert review.load(sid)['revision']==0
    assert prepare(ROOT/'review') is None
    assert not (ROOT/'review/geometry_review').exists()
    checks.append('no real A/B reviews: prepare_gt creates no geometry tasks/points')
    for t in lines(ROOT/'gt_staging_manifest.jsonl'):
        q=quality(dict(sample_id=t['sample_id'],points=[],disposition='pending',evaluation_scope='local'),t)
        assert q['status']=='unverified' and not q['formal_errors_allowed'] and q['supported_success_thresholds_px']==[]
    checks.append('inactive native-sized staging fails independent GT qualification and computes no Success')
    prior=json.loads((ROOT/'evidence/protected_before.json').read_text());changed=[]
    for p,h in prior.items():
        if sha(p)!=h:changed.append(p)
    assert not changed,changed;checks.append(f'{len(prior)} directly protected old indexes/annotation versions/training source files unchanged')
    result=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),passed=True,checks=checks,
        real_regions=len(regions),unique_acquisitions=len(assets),GT=0,real_accuracy_computed=False,
        synthetic_point_generation=False,GPU_used=False)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');save(ROOT/f'reports/verification_{stamp}.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':run()
