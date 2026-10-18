"""Verify actual frozen windows before building UI. Never produces physical GT."""
import sys,json
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,rasterio
from affine import Affine
from PIL import Image,ImageDraw
from geometry import ROOT,SOURCE,BASE,queue,View,mapped,point_result,overlap,save,sha

PRIORITY=['ANGLE_e76cc299fa7b','ANGLE_e3bfffe1139e','ANGLE_75c261e08eae','ANGLE_5baa3ae0a7c1','ANGLE_5da40401a853','ANGLE_cc7e4d5fcb78']

def window_check(row,v):
    z=row['views'][v.side];asset=SOURCE/'assets'/z['acquisition_id']/'metadata.json'
    if str(v.path).startswith(str(SOURCE/'assets')):
        meta=json.loads(asset.read_text());parent=meta['original_profile'];win=meta['original_window_xywh']
    else:
        parent=row['acquisition_metadata']['views'][v.side].get('profile')
        win=z.get('original_native_window_xywh')
        if not parent:parent=z.get('context_provenance',{}).get('original_profile')
    if parent and win:
        expected=Affine(*parent['transform'])*Affine.translation(*win[:2])
        err=max(abs(a-b) for a,b in zip(list(expected)[:6],list(v.affine)[:6]))
        assert err<1e-8,(row['sample_id'],v.side,err)
        return dict(status='original_affine_plus_integer_window_matches',max_coefficient_difference=err,parent_window=win)
    # Local context chain is independent of any uncertain parent_window_xywh tag.
    if z.get('context_info') and z.get('context_native_window_xywh'):
        t=Affine(*z['context_info']['transform'])*Affine.translation(*z['context_native_window_xywh'][:2])
        err=max(abs(a-b) for a,b in zip(list(t)[:6],list(v.affine)[:6]));assert err<1e-8
        return dict(status='frozen_native_matches_local_context_window',max_coefficient_difference=err,original_parent_not_available=True)
    return dict(status='original_parent_not_available_local_header_and_manifest_only',physical_accuracy_verified=False)

def illustration(row,views,cover,tests,folder):
    panels=[]
    for side,v in enumerate(views):
        with Image.open(v.png['db']) as im:
            im=im.convert('RGB');im.thumbnail((1000,1000),Image.Resampling.NEAREST)
        sx,sy=im.width/v.width,im.height/v.height;d=ImageDraw.Draw(im)
        poly=cover['native_polygons'][side]
        if poly:d.line([((p[0]+.5)*sx,(p[1]+.5)*sy) for p in poly+[poly[0]]],fill='#39ff92',width=3)
        for i,test in enumerate(tests):
            xy=test['source_xy'] if side==test['source_side'] else test['target_xy']
            x,y=(xy[0]+.5)*sx,(xy[1]+.5)*sy
            color='#ffd36b' if side==test['source_side'] else '#ff62d2'
            d.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2);d.text((x+7,y+3),str(i+1),fill=color)
        im.save(folder/f'native{side}_geographic_reference.png');panels.append(im)
    canvas=Image.new('RGB',(2040,1100),'#15202b');d=ImageDraw.Draw(canvas)
    d.text((15,10),row['pair_id']+' | geographic reference ONLY; green=crop overlap, gold=source, pink=prediction',fill='white')
    for i,im in enumerate(panels):canvas.paste(im,(15+i*1020,45))
    canvas.save(folder/'bidirectional_mapping_preview.png')

def run():
    summaries=[];tasks=[]
    for row in queue():
        folder=ROOT/'validation'/row['sample_id'];folder.mkdir(exist_ok=False)
        try:
            views=[View(row,i) for i in [0,1]];cover=overlap(views);tests=[];checks=[]
            for side,a in enumerate(views):
                parent=window_check(row,a);p=np.array([[(a.width-1)*x,(a.height-1)*y] for x in [.1,.5,.9] for y in [.1,.5,.9]])
                q=mapped(a,views[1-side],p);back=mapped(views[1-side],a,q);err=np.linalg.norm(back-p,axis=1)
                assert np.isfinite(q).all() and max(err)<1e-4
                with rasterio.open(a.path) as d:
                    # Independent rasterio.xy oracle for pixel centres, including rotated/non-north-up rasters.
                    x,y=d.xy(p[:,1],p[:,0]);assert np.max(np.abs(a.world(p)-np.c_[x,y]))<1e-8
                checks.append(dict(side=side,n=9,max_math_roundtrip_px=float(max(err)),window=parent,header=a.header))
                selected=[point_result(row,views,side,xy) for xy in p]
                # Prefer three inside common extent for diagrams, retaining all nine in evidence.
                good=[t for t in selected if t['source_valid'] and t['target_valid']]
                tests.extend((good or selected)[:3]);save(folder/f'point_tests{side}.json',selected)
            assert cover['area_m2']>0
            record=dict(sample_id=row['sample_id'],pair_id=row['pair_id'],passed=True,views=checks,coverage=cover,
                tests_for_diagram=tests,physical_scatterer_error_px=None,independent_GT=False,
                meaning='numerical geocoding consistency only; no corresponding physical landmark verified')
            save(folder/'mapping_verification.json',record)
            if row['pair_id'] in PRIORITY or views[0].crs!=views[1].crs:illustration(row,views,cover,tests,folder)
            tasks.append(dict(sample_id=row['sample_id'],pair_id=row['pair_id'],site_name=row['site_name'],component=row['component'],
                priority=PRIORITY.index(row['pair_id'])+1 if row['pair_id'] in PRIORITY else 7,
                azimuth_delta_deg=row.get('azimuth_delta_deg'),days=row.get('days'),
                views=[dict(native_tif=str(v.path),width=v.width,height=v.height,crs=str(v.crs),method=v.method,
                    images={k:str(p) for k,p in v.png.items()},acquisition_id=row['views'][i]['acquisition_id']) for i,v in enumerate(views)],
                coverage=cover,mapping_verified=True,GT=False))
            summaries.append(dict(sample_id=row['sample_id'],passed=True,max_math_roundtrip_px=max(c['max_math_roundtrip_px'] for c in checks),
                distinct_crs=views[0].crs!=views[1].crs,coverage_m2=cover['area_m2']))
        except Exception as e:
            save(folder/'mapping_failed.json',dict(sample_id=row['sample_id'],error=repr(e),affine_fallback=False))
            summaries.append(dict(sample_id=row['sample_id'],passed=False,error=repr(e)))
            raise
    save(ROOT/'mapping_validation.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),passed=True,pairs=summaries,
        actual_pairs=len(summaries),priority_pairs=6,mixed_CRS_pairs=sum(s['distinct_crs'] for s in summaries),
        mathematical_error_is_not_physical_error=True,real_GT=0,new_image_downloads=0,source_queue_sha256=sha(SOURCE/'review/candidate_regions.jsonl')))
    with (ROOT/'tasks.jsonl').open('x') as f:
        for t in sorted(tasks,key=lambda t:t['priority']):f.write(json.dumps(t,ensure_ascii=False,allow_nan=False)+'\n')
    print('PASS',len(tasks),'actual mappings; six primary previews; max math roundtrip',max(s['max_math_roundtrip_px'] for s in summaries))

if __name__=='__main__':run()
