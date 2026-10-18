"""Focused mathematics, display identity and assisted-record rejection checks, no human simulation."""
import sys,json,tempfile,io
sys.dont_write_bytecode=True
from pathlib import Path
import numpy as np,rasterio
from PIL import Image
from geometry import ROOT,SOURCE,queue,View,mapped,point_result,lines,save
from records import Records

def run():
    rows=queue();row=rows[0];views=[View(row,i) for i in [0,1]]
    for v in views:
        assert v.inside([[0,0],[v.width-1,v.height-1]]).all()
        assert not v.inside([[-1,0],[v.width,0],[0,v.height]]).any()
        # At least one out-of-target result: no clipping to manufacture overlap.
        outside=[]
        for p in [[0,0],[v.width-1,0],[0,v.height-1],[v.width-1,v.height-1]]:
            result=point_result(row,views,v.side,p)
            if not result['target_in_extent']:outside.append(result)
        assert outside,'rotated real windows should include non-overlapping corners'
    centre=np.array([[100.25,200.75]])
    direct=mapped(*views,centre)
    wrong_corner=views[1].pixels(views[0].world(centre-.5))
    assert np.linalg.norm(direct-wrong_corner)>.3,'half pixel convention is a meaningful check'
    cache=json.loads((ROOT/'display_cache.json').read_text());tested=set()
    for t in lines(ROOT/'tasks.jsonl'):
        for side,v in enumerate(t['views']):
            key=cache['task_view_index'][t['sample_id']+':'+str(side)]
            if key in tested:continue
            tested.add(key);e=cache['assets'][key];a=np.load(e['array'],mmap_mode='r',allow_pickle=False)
            with Image.open(e['source_png']) as im:
                for x,y in [(0,0),(v['width']//2,v['height']//2),(v['width']-1,v['height']-1)]:
                    assert a[y,x]==im.getpixel((x,y))
    tasks={t['sample_id']:t for t in lines(ROOT/'tasks.jsonl')}
    with tempfile.TemporaryDirectory(prefix='sar_geo_record_rejection_',dir='/tmp') as folder:
        records=Records(folder,tasks)
        for payload in [{},dict(sample_id=row['sample_id'],actor=''),
                dict(sample_id=row['sample_id'],actor='AUTOMATED_REJECTION_CHECK',GT_qualified=True),
                dict(sample_id=row['sample_id'],actor='AUTOMATED_REJECTION_CHECK',points=[[0,0]])]:
            try:records.commit(payload,'AUTOMATED_REJECTION_CHECK')
            except (ValueError,KeyError):pass
            else:raise AssertionError('unqualified/GT injection accepted')
        assert not list(Path(folder).iterdir()),'invalid automated requests must never create human records'
    save(ROOT/'reports/selfcheck.json',dict(passed=True,actual_native_boundaries=True,unclipped_out_of_target=True,
        pixel_centre_half_pixel_check=True,unique_display_caches_tested=len(tested),
        record_rejection_checks=4,simulated_human_judgments=0,real_GT=0))
    print('PASS focused checks; no human judgments or GT created')

if __name__=='__main__':run()
