"""Rebuild exact released source windows, without resampling a random index.

Transient retrieval failures affect sequential RNG. Released window coordinates,
display limits and file hashes are therefore the reproduction authority. No GT.
"""
import argparse,json
from pathlib import Path
import numpy as np,rasterio
from PIL import Image
from rasterio.windows import Window
from build_real_pairs import read_common_window,read_moving_native_window,save_tiff,display,sha256_file

def replay(manifest,output):
    manifest=Path(manifest).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);entries=[]
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='40',GDAL_HTTP_MAX_RETRY='2',AWS_NO_SIGN_REQUEST='YES'):
        for n,line in enumerate(manifest.read_text().splitlines()):
            pinned=json.loads(line);original=manifest.parent/pinned['metadata_relative_path'];m=json.loads(original.read_text())
            assert not m.get('patches'),'Released real pairs must not contain provisional patch training labels'
            x,y=m['reference_window_xy'];w,h=m['size_px'];assert w==h==512
            units=m['input_units'];zero=m.get('allow_declared_valid_zero',False);lo,hi=m['display_db_range']
            with rasterio.open(m['reference_path']) as ref,rasterio.open(m['moving_path']) as mov:
                a,b,va,vb,t=read_common_window(ref,mov,Window(x,y,w,h),units,zero)
                bn,vbn,tn,origin=read_moving_native_window(ref,mov,Window(x,y,w,h),units,zero)
                assert origin==m['moving_native_window_xy'] and str(ref.crs)==m['crs'] and str(mov.crs)==m['native_moving_crs']
                assert np.allclose(list(t)[:6],m['map_transform_first6'],atol=1e-9,rtol=0)
                assert np.allclose(list(tn)[:6],m['native_moving_transform_first6'],atol=1e-9,rtol=0)
                folder=output/m['split']/m['pair_id']/m['chip_id'];folder.mkdir(parents=True,exist_ok=False)
                for name,arr,mask,crs,tr in [('reference_float.tif',a,va,ref.crs,t),('moving_float.tif',b,vb,ref.crs,t),('moving_native_float.tif',bn,vbn,mov.crs,tn)]:save_tiff(folder/name,arr,mask,crs,tr)
                for name,arr,mask in [('reference.png',a,va),('moving.png',b,vb),('moving_native.png',bn,vbn)]:Image.fromarray(display(arr,mask,lo,hi,units)).save(folder/name)
                for name,mask in [('reference_valid.png',va),('common_valid.png',va&vb),('moving_native_valid.png',vbn)]:Image.fromarray(mask.astype('uint8')*255).save(folder/name)
                (folder/'metadata.json').write_bytes(original.read_bytes())
                for name,digest in pinned['expected_file_sha256'].items():assert sha256_file(folder/name)==digest,('Reproduction mismatch: source bytes/dependency version changed',m['pair_id'],m['chip_id'],name)
            entries.append(m);print('Exact replay',n+1,m['pair_id'],m['chip_id'],flush=True)
    (output/'SUMMARY.json').write_text(json.dumps({'candidate_chip_pairs':len(entries),'source_windows':'Pinned actual release coordinates; no RNG','all_file_sha256_equal_originals':True,'GT_status':'unverified'},indent=2))
    (output/'candidates.jsonl').write_text(''.join(json.dumps(m)+'\n' for m in entries))
    return len(entries)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--window-manifest',required=True);p.add_argument('--output',required=True);a=p.parse_args();replay(a.window_manifest,a.output)
