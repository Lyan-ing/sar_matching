"""Recover public-read failures from frozen local native context, no network."""
import sys
sys.dont_write_bytecode=True
import os,json,hashlib
from pathlib import Path
from build_windows import ROOT,V2,SELECTED,SIDE_M,event
from geo_tools import lines,read_tif,save,sha,geo

def run():
    assert (ROOT/'window_summary.json').exists(),'Wait until remote batch ends; never race a raw writer'
    rows={r['pair_id']:r for r in lines(ROOT/'pair_candidates.jsonl')}
    wanted={v['acquisition_id']:v for k in SELECTED for v in rows[k]['views']};available={}
    for r in lines(V2/'regions.jsonl'):
        for i,v in enumerate(r['views']):
            aid=v['acquisition_id']
            if aid not in available or v['context_info']['width']>available[aid][0]['views'][available[aid][1]]['context_info']['width']:
                available[aid]=(r,i)
    recovered=[];unavailable=[]
    for aid,acq in wanted.items():
        folder=ROOT/'assets'/aid
        if (folder/'metadata.json').exists():continue
        if aid not in available:unavailable.append(aid);continue
        assert not (folder/'native.tif').exists(),'Do not overwrite a retained raw/partial file'
        folder.mkdir(parents=True,exist_ok=True);r,i=available[aid];v=r['views'][i];source=V2/'regions'/r['sample_id'];src=source/f'context{i}.tif'
        os.link(src,folder/'native.tif');a,valid,z=read_tif(src)
        for kind in ['db','linear']:
            os.link(source/f'context_{kind}{i}.png',folder/f'native_{kind}.png')
            save(folder/f'display_{kind}.json',dict(v['display']['context_'+kind],reused_frozen_native_context=True))
        from PIL import Image
        im=Image.open(folder/'native_db.png');im.thumbnail((1100,1100),Image.Resampling.NEAREST);im.save(folder/'overview.png')
        meta=dict(acquisition=acq,native_info=z,original_profile=r['acquisition_metadata']['views'][i]['profile'],
            original_window_xywh=v['context_provenance']['original_window_xywh'],native_tif=str(folder/'native.tif'),
            source_method='existing_frozen_native_context_after_public_read_failure_no_network',unresampled=True,
            technical_valid_fraction=float(valid.mean()),requested_ground_side_m=SIDE_M,actual_context_smaller_than_requested=True,
            native_spacing=geo.metric_geometry(z),raw_sha256=sha(src),pixel_value_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            valid_mask_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),source_path=str(src),source_sha256=sha(src),
            source_region=r['sample_id'],geocoding='geographic screen only, not GT; small support must be declared')
        save(folder/'metadata.json',meta);recovered.append(dict(acquisition_id=aid,source=str(src),shape=list(a.shape),no_new_identity=True))
        event(dict(acquisition_id=aid,status='reused_frozen_local_after_network_failure',source=str(src),shape=list(a.shape)))
    save(ROOT/'local_fallbacks.json',dict(recovered=recovered,unavailable=unavailable,network_requests=0,
        warning='Different view FOV may differ substantially. Do not silently assert full-field correspondence.'))
    print('REUSED_LOCAL',len(recovered),'UNAVAILABLE',len(unavailable))

if __name__=='__main__':run()
