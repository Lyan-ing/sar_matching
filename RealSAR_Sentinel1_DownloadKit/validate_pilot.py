"""Verify genuine acquisition metadata and chips; qualitative image-only matching.

No map-grid correspondence is used as evaluation GT, and no Success@px is computed.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from PIL import Image
import requests
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sarbench.data import source_component
from sarbench.matchers import LocalFeatureMatcher
from sarbench.estimate import estimate
from sarbench.reliability import geometry_diagnostics,gate

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def supplemental(scene,folder,grid,bounds):
    tags=scene['raster_metadata']['source_tags'];records=[];http=[];session=requests.Session()
    for i in range(1,int(tags['NUMBER_SCENES'])+1):
        meta=json.loads(tags[f'SCENE_{i}_METADATA']);title=meta['title'];raw=folder/title;raw.mkdir()
        values={};rooturl=scene['url'].rsplit('/',1)[0]+'/'+title
        for name in ['manifest.safe','productInfo.json']:
            url=rooturl+'/'+name
            try:
                r=session.get(url,timeout=35);http.append({'url':url,'status':r.status_code})
                if r.ok:
                    (raw/name).write_bytes(r.content)
                    if name=='manifest.safe':
                        tree=ET.fromstring(r.content)
                        for key in ['relativeOrbitNumber','orbitNumber','pass','startTime','stopTime']:
                            matches=[{'value':n.text,'attributes':n.attrib} for n in tree.iter() if n.tag.split('}')[-1]==key]
                            values[key]=matches
            except (requests.RequestException,ET.ParseError) as exc:http.append({'url':url,'error':str(exc)})
        records.append({'product_id':title,'tile_product_metadata':meta,'safe_metadata':values,
                        'relative_orbit':next((int(v['value']) for v in values.get('relativeOrbitNumber',[]) if v['attributes'].get('type')=='start'),None)})
    angle_url=scene['url'].rsplit('/',1)[0]+'/local_incident_angle.tif';angle={'url':angle_url,'definition':'terrain-local incidence; source integer/100 degrees, not a complete look-angle model'}
    try:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_HTTP_TIMEOUT='35',AWS_NO_SIGN_REQUEST='YES'):
            with rasterio.open(angle_url) as src:
                assert src.crs.to_string()==grid['crs']
                window=from_bounds(*bounds,transform=src.transform).round_offsets().round_lengths()
                a=src.read(1,window=window,masked=True).astype('float32').filled(np.nan)/100
                assert a.shape==(512,512)
                valid=np.isfinite(a)&(a>0)&(a<90)
                assert valid.mean()>.7
                angle.update(valid_fraction=float(valid.mean()),degree_min=float(a[valid].min()),degree_max=float(a[valid].max()),degree_mean=float(a[valid].mean()))
                profile={'driver':'GTiff','height':512,'width':512,'count':1,'dtype':'float32','crs':src.crs,'transform':rasterio.Affine(*grid['transform']),'nodata':-9999.,'compress':'deflate'}
                with rasterio.open(folder/'local_incidence_degrees.tif','w',**profile) as dst:dst.write(np.where(valid,a,-9999).astype('float32'),1);dst.update_tags(UNIT='degree',SOURCE_URL=angle_url,NOTE=angle['definition'])
    except (rasterio.errors.RasterioError,AssertionError,ValueError) as exc:angle['missing_reason']=str(exc)
    result={'products':records,'http_access':http,'local_incidence':angle,'missing':['look_azimuth','sensor_heading','independent_static_GCP_GT','per_view_ship_annotations']}
    save(folder/'supplemental_metadata.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--data',required=True);a=p.parse_args();run=Path(a.run).resolve();data=Path(a.data).resolve();out=run/'validation'
    if out.exists():raise FileExistsError(out)
    out.mkdir();manifest=json.loads((data/'manifest.json').read_text());scenes=manifest['scenes'];grid=manifest['common_grid'];validated=[];product_ids=[];images=[]
    for scene in scenes:
        path=data/scene['geotiff'];assert sha(path)==scene['geotiff_sha256'];assert sha(data/scene['preview'])==scene['preview_sha256']
        tags=scene['raster_metadata']['source_tags'];assert tags['DATE'].replace('-','')==scene['date'];assert tags['TILE_ID']==manifest['tile'];assert tags['MISSION_ID']==scene['satellite'];assert tags['ORBIT_DIRECTION'].lower()==scene['orbit_direction']
        with rasterio.open(path) as src:
            assert src.shape==(512,512) and src.count==1 and src.dtypes[0]=='float32';assert src.crs.to_string()==grid['crs'] and list(src.transform)[:6]==grid['transform']
            arr=src.read(1,masked=True).astype('float32').filled(np.nan);valid=np.isfinite(arr)&(arr>0);assert valid.mean()>=.7;db=10*np.log10(arr[valid]);assert db.std()>.5
            bounds=list(src.bounds);wgs=list(transform_bounds(src.crs,'EPSG:4326',*bounds,densify_pts=21));summary={'scene_name':scene['scene_name'],'date':scene['date'],'orbit_direction':scene['orbit_direction'],'crs':grid['crs'],'bounds_projected':bounds,'bounds_wgs84':wgs,'pixel_spacing':list(src.res),'valid_fraction':float(valid.mean()),'db_std':float(db.std()),'gamma0_p1_p50_p99':np.quantile(arr[valid],[.01,.5,.99]).tolist(),'geotiff_sha256':sha(path),'preview_sha256':sha(data/scene['preview'])}
        image=np.array(Image.open(data/scene['preview']));assert image.shape==(512,512) and image.dtype==np.uint8 and np.std(image)>1;images.append(image)
        folder=out/scene['scene_name'];folder.mkdir();summary['metadata']=supplemental(scene,folder,grid,bounds);validated.append(summary);product_ids.append({r['product_id'] for r in summary['metadata']['products']});print('Verified',scene['scene_name'],'valid',summary['valid_fraction'],'products',product_ids[-1],flush=True)
    assert len({s['date'] for s in scenes})==len(scenes)
    assert all(not product_ids[i]&product_ids[j] for i in range(len(scenes)) for j in range(i+1,len(scenes)))
    assert all(s['bounds_projected']==validated[0]['bounds_projected'] for s in validated)
    fig,axes=plt.subplots(1,len(images),figsize=(5*len(images),5.6),layout='constrained')
    for ax,image,s in zip(axes,images,validated):ax.imshow(image,cmap='gray',vmin=0,vmax=255);ax.set_title(s['scene_name']+'\n512×512, VV, ascending; dB [-25,0]',fontsize=10);ax.axis('off')
    fig.suptitle('Real repeat acquisitions; map-grid aligned, no independent pixel GT',fontsize=12);fig.text(.5,.01,'Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026',ha='center',fontsize=8);fig.savefig(out/'comparison_preview.png',dpi=160);plt.close(fig)
    selected=json.loads(Path('MaritimeSARRegistrationPhase2/output/phase25_20261010_0000/development/selection.json').read_text());cv2.setNumThreads(1);predictions=[]
    for j in range(1,len(images)):
        for algorithm,options in [('sift',{'contrastThreshold':.01}),('orb',{'edgeThreshold':16})]:
            settings=selected['selected'][algorithm];assert settings['model']=='affine' and settings['policy']=='full';matcher=LocalFeatureMatcher(algorithm,max_features=2048,detector_options=options)
            start=time.perf_counter();pred=matcher.predict(images[0],images[j]);base=estimate(pred,images[0].shape,model=settings['model'],seed=42);geo=geometry_diagnostics(base,pred,images[0].shape,images[j].shape,42) if base.accepted else {};est=gate(base,geo,'full');elapsed=1000*(time.perf_counter()-start)
            result={'method':algorithm,'pair':[scenes[0]['scene_name'],scenes[j]['scene_name']],'accepted_by_image_only_gate':est.accepted,'reason':est.reason,'keypoints':[pred.n_keypoints0,pred.n_keypoints1],'matches':est.n_matches,'inliers':est.n_inliers,'fit_inlier_rmse_px':est.inlier_rmse,'prediction_only_ms':elapsed,'pred_H':est.H.tolist() if est.H is not None else None,'candidate_H':est.candidate_H.tolist() if est.candidate_H is not None else None,'geometry':geo,'warning':'Gate acceptance and fit residual are not GT accuracy or formal registration success.'}
            np.savez_compressed(out/f'{algorithm}_0_{j}_matches.npz',points0=pred.points0,points1=pred.points1)
            H=est.H;warped=cv2.warpPerspective(images[j],np.linalg.inv(H),(512,512)) if H is not None else images[j];overlay=np.stack([images[0],warped,warped],-1)
            fig,ax=plt.subplots(figsize=(5,5.7),layout='constrained');ax.imshow(overlay);ax.axis('off');ax.set_title(f'{algorithm} 0→{j}: accepted={est.accepted}, {est.reason}\nred=view0, cyan=warped view{j}; no independent GT',fontsize=10);fig.text(.5,.01,'Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery, 2026',ha='center',fontsize=7);fig.savefig(out/f'{algorithm}_0_{j}_qualitative.png',dpi=150);plt.close(fig);predictions.append(result)
    save(out/'image_only_predictions.json',predictions)
    protected=json.loads((run/'protected_artifacts.json').read_text())
    for path,digest in protected.items():assert sha(path)==digest,path
    save(out/'acceptance.json',{'accepted':True,'scenes':validated,'distinct_dates':True,'distinct_original_product_ids':True,'common_geographic_bounds':True,'protected_files_unchanged':len(protected),'independent_gt_available':False,'formal_success_metrics_computed':False,'script_sha256':sha(__file__)})
    print('REAL PILOT ACCEPTED',len(scenes),'protected files',len(protected),flush=True)

if __name__=='__main__':main()
