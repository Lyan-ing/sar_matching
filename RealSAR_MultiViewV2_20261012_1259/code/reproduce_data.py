"""Rebuild published Native/Common-FOV from local frozen contexts, no downloads."""
import sys
sys.dont_write_bytecode=True
import argparse
from pathlib import Path
import numpy as np
from geo_tools import ROOT,lines,read_tif,write_tif,sample_grid,save,sha

def reproduce(out=None):
    destination=None if out is None else Path(out).resolve()
    if destination is not None:destination.mkdir(parents=True,exist_ok=False)
    results=[]
    for r in lines(ROOT/'candidate_regions.jsonl'):
        folder=ROOT/'regions'/r['sample_id'];side_results=[]
        if destination is not None:(destination/r['sample_id']).mkdir()
        for i,v in enumerate(r['views']):
            a,m,z=read_tif(folder/f'context{i}.tif');x,y,w,h=v['context_native_window_xywh']
            native=a[y:y+h,x:x+w];nv=m[y:y+h,x:x+w]
            old,ov,_=read_tif(folder/f'native{i}.tif');np.testing.assert_array_equal(native,old);np.testing.assert_array_equal(nv,ov)
            common,cv=sample_grid(a,m,z,v['common_info']);old,ov,_=read_tif(folder/f'common{i}.tif')
            np.testing.assert_array_equal(cv,ov);np.testing.assert_array_equal(common[cv],old[ov])
            if destination is not None:
                write_tif(destination/r['sample_id']/f'native{i}.tif',native,nv,v['native_info'])
                write_tif(destination/r['sample_id']/f'common{i}.tif',common,cv,v['common_info'])
            side_results.append(dict(context_sha256=sha(folder/f'context{i}.tif'),native_exact=True,common_valid_pixels_exact=True,valid_mask_exact=True))
        results.append(dict(sample_id=r['sample_id'],sides=side_results))
    result=dict(candidate_regions=len(results),passed=True,network_reads=0,source=str(ROOT),results=results)
    if destination is not None:save(destination/'reproduction_result.json',result)
    print('REPRODUCTION',len(results),'regions: native, common and masks exactly reproduced; no network')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);a=p.parse_args();reproduce(a.out)
