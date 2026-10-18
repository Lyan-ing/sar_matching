"""Decode frozen Native display PNG once; caches are 8-bit display, never new sources."""
import sys
sys.dont_write_bytecode=True
import json,hashlib,os
import numpy as np
from PIL import Image
from geometry import ROOT,lines,save,sha

def run():
    assert json.loads((ROOT/'mapping_validation.json').read_text())['passed']
    index={};seen={};bytes_total=0
    for task in lines(ROOT/'tasks.jsonl'):
        for side,v in enumerate(task['views']):
            source=v['images']['db'];source_sha=sha(source);key=source_sha[:24]
            if key not in seen:
                array_path=ROOT/'cache'/(key+'.npy');overview_path=ROOT/'cache'/(key+'.png')
                if array_path.exists() or overview_path.exists():raise FileExistsError('retain partial cache; no silent overwrite')
                with Image.open(source) as im:
                    assert im.mode=='L' and im.size==(v['width'],v['height'])
                    arr=np.asarray(im);np.save(array_path,arr,allow_pickle=False)
                    preview=im.copy();preview.thumbnail((1100,1100),Image.Resampling.NEAREST);preview.save(overview_path)
                seen[key]=dict(array=str(array_path),overview=str(overview_path),source_png=source,
                    source_sha256=source_sha,array_sha256=sha(array_path),width=v['width'],height=v['height'],
                    dtype='uint8',unresampled_display_array=True,geographic_resampling=False)
                bytes_total+=array_path.stat().st_size
                print('CACHE',key,v['width'],v['height'],flush=True)
            index[task['sample_id']+':'+str(side)]=key
    save(ROOT/'display_cache.json',dict(assets=seen,task_view_index=index,bytes=bytes_total,
        distinct_display_windows=len(seen),new_acquisition_identities=0,raw_TIFFs_changed=False))
    print('PASS cached',len(seen),'existing displays,',bytes_total,'bytes')

if __name__=='__main__':run()
