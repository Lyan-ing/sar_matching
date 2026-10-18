"""Add the two catalogue-confirmed 12-day repeats around the verified San Francisco scene.

Keeps the original locked 72-pair manifest unchanged; same site stays in train.
Reuses the earlier read-only SAFE verifier, never a matcher or GT source.
"""
import argparse,csv,json,itertools,sys,subprocess
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np,rasterio
from rasterio.warp import transform
from rasterio.windows import Window,bounds
from prepare_sources import lines,raster_info

def main(run,base):
    out=run/'s1_sf_extension';out.mkdir(exist_ok=False);sys.path.insert(0,str(base/'MaritimeSARRegistrationPhase2'));sys.path.insert(0,str(base/'RealSAR_Sentinel1_DownloadKit'))
    from validate_pilot import supplemental
    first=next(r for r in lines(run/'s1_verified/inventory.jsonl') if r['site_name']=='San_Francisco');catalog=json.loads((run/'s1_discovery/10SEG_scenes.json').read_text());d=datetime.fromisoformat(first['datetime'][:19]);dates={(d+timedelta(days=v)).strftime('%Y%m%d') for v in [-12,12]};targets=[r for r in catalog if r['date'] in dates and r['satellite']==first['platform'] and r['orbit_direction']==first['orbit_direction']]
    (out/'protocol.json').write_text(json.dumps({'seed':260512,'chip_size':512,'chips_per_pair':3,'min_valid':.98,'new_source_selection':'catalogue-confirmed +-12 day repeats of metadata-verified relative orbit35 June14 scene; no registration-based selection','target_scene_names':[r['scene_name'] for r in targets],'component':'real_component_004','split':'train'},indent=2))
    sources=[first];failures=[]
    for rec in targets:
        folder=out/rec['scene_name'];folder.mkdir()
        try:
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='40',AWS_NO_SIGN_REQUEST='YES'):
                with rasterio.open(rec['url']) as ds:
                    info=raster_info(ds);tags=ds.tags();assert int(tags['NUMBER_SCENES'])==1
                    xx,yy=transform('EPSG:4326',ds.crs,[-122.4],[37.78]);row,col=ds.index(xx[0],yy[0]);win=Window(col-256,row-256,512,512);a=ds.read(1,window=win);assert (np.isfinite(a)&(a>0)).mean()>=.98
                    grid={'crs':str(ds.crs),'transform':list(ds.window_transform(win))[:6]};bb=bounds(win,ds.transform)
            meta=supplemental({**rec,'raster_metadata':{'source_tags':tags}},folder,grid,bb);p=meta['products'][0];assert p['relative_orbit']==first['relative_orbit'];safe=p['safe_metadata'];pid=p['product_id'];m=p['tile_product_metadata']
            r={**first,'datetime':safe['startTime'][0]['value'].rstrip('Z')+'Z','end_datetime':safe['stopTime'][0]['value'].rstrip('Z')+'Z','acquisition_id':pid,'original_products':[pid],'datatake_ids':[m['mission_id']+':'+m['mission_data_take_id']],
               'absolute_orbit':int(m['absolute_orbit_number']),'asset_url':rec['url'],'raster':info,'geometry':json.loads(m['footprint']),'bbox':info['bounds_wgs84'],'metadata_path':str((folder/'source.json').resolve()),'safe':[safe]}
            (folder/'source.json').write_text(json.dumps(r,indent=2));sources.append(r);print('S1 repeat verified',rec['scene_name'],pid,flush=True)
        except Exception as e:failures.append({'scene':rec['scene_name'],'failure':repr(e)})
    (out/'sources.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sources));(out/'source_checks.json').write_text(json.dumps({'sources':len(sources),'failures':failures},indent=2));pairs=[]
    for i,(a,b) in enumerate(itertools.combinations(sorted(sources,key=lambda r:r['datetime']),2)):
        pairs.append({'pair_id':f'S1SF_EXTENSION_{i:03d}','site_id':'real_component_004','split':'train','sensor':'Sentinel-1','reference_acquisition_id':a['acquisition_id'],'moving_acquisition_id':b['acquisition_id'],'reference_path':a['asset_url'],'moving_path':b['asset_url'],'reference_utc':a['datetime'],'moving_utc':b['datetime'],'input_units':'linear_power','product_level':'RTC_Gamma0_VV','site_name':'San_Francisco','scene_class':'urban_coast','look_pair_category':'repeat_pass_same_look','license':a['license'],'license_url':a['license_url'],'source_metadata_status':'SAFE_VERIFIED_RELATIVE_ORBIT35','reference_roi_xyxy':json.dumps(a['roi_xyxy']),'reference_metadata':a['metadata_path'],'moving_metadata':b['metadata_path'],'reference_relative_orbit':35,'moving_relative_orbit':35})
    if not pairs:raise RuntimeError('No extension source pair')
    with (out/'verified_sources.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
    commands=[[sys.executable,str(Path(__file__).parent/'build_real_pairs.py'),'--manifest',str(out/'verified_sources.csv'),'--output',str(out/'data'),'--chips-per-pair','3','--seed','260512'],[sys.executable,str(Path(__file__).parent/'validate_release.py'),'--run',str(out),'--data',str(out/'data'),'--out',str(out/'quality')]]
    for cmd in commands:
        print('COMMAND',cmd,flush=True);result=subprocess.run(cmd);assert result.returncode==0,result.returncode

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--base',required=True);a=p.parse_args();main(Path(a.run).resolve(),Path(a.base).resolve())
