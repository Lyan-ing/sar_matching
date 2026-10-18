"""Read-only native SAR pairing audit. No matcher or annotation imports."""
import sys
sys.dont_write_bytecode = True
import argparse, csv, hashlib, importlib.util, json, shutil, time
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2, rasterio
from rasterio.warp import transform
from affine import Affine
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
DATA = BASE/'real_sar_pilot/multisource512_20261010_2352'
CONTEXT = BASE/'RealSAR_RepresentationContext_20261011_1903'
spec = importlib.util.spec_from_file_location('geography_only', BASE/'RealSAR_AutoLabels_20261011_1149/geo.py')
geo = importlib.util.module_from_spec(spec); spec.loader.exec_module(geo)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
FIVE = ['S1SF_EXTENSION_000__chip_002','Capella_0003__chip_001','Umbra_0061__chip_001','Umbra_0071__chip_000','Umbra_0014__chip_002']

def dump(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def require_unsealed():
    if (ROOT/'DELIVERY_COMPLETE.json').exists():
        raise RuntimeError('This audit output is sealed; copy code/config to a new output directory before reproducing.')

def db_image(raw, valid, units):
    a = raw.astype(np.float64)
    good = valid & np.isfinite(a) & (a >= 0)
    positive = a[good & (a > 0)]
    if not len(positive):
        return np.zeros(a.shape), good, {'display_positive_floor':None,'constant_or_zero':True}
    floor = float(positive.min())
    db = (10 if units == 'linear_power' else 20)*np.log10(np.maximum(a, floor))
    return db, good, {'display_positive_floor':floor,'valid_zero_count':int((good & (a==0)).sum()),'zero_note':'valid zero shown at minimum observed positive DN; not removed from validity'}

def stretch(db, mask, limits):
    low, high = limits
    im = np.rint(np.nan_to_num(np.clip((db-low)/max(high-low,1e-12),0,1))*255).astype(np.uint8)
    im[~mask] = 0
    return im

def display_pair(arrays, valid, units):
    db, masks, notes = [], [], []
    for a,v,u in zip(arrays,valid,units):
        d,m,n = db_image(a,v,u); db.append(d); masks.append(m); notes.append(n)
    joint = np.quantile(np.concatenate([d[m] for d,m in zip(db,masks)]), [.01,.99]).tolist()
    ranges = [np.quantile(d[m],[.01,.99]).tolist() for d,m in zip(db,masks)]
    for d,m,n,lim in zip(db,masks,notes,ranges):
        n.update(independent_db_limits=lim,clipped_below_fraction=float((d[m]<lim[0]).mean()),clipped_above_fraction=float((d[m]>lim[1]).mean()))
    return [stretch(d,m,joint) for d,m in zip(db,masks)], [stretch(d,m,l) for d,m,l in zip(db,masks,ranges)], {'joint_db_limits':joint,'views':notes,'formula':'10log10(power) or 20log10(amplitude); arbitrary DN display, no absolute calibration; no smoothing'}

def metric_polygon(info, metric_crs):
    fp = geo.footprint(info)['coordinates'][0][:-1]
    x,y = transform('EPSG:4326',metric_crs,[p[0] for p in fp],[p[1] for p in fp])
    return np.c_[x,y]

def intersection(p,q):
    # Densified affine rectangles remain nearly convex locally. Subtract world
    # origin before OpenCV float32 clipping; never use degrees as metric area.
    origin = p.mean(axis=0)
    pp = cv2.convexHull((p-origin).astype(np.float32)); qq = cv2.convexHull((q-origin).astype(np.float32))
    area, poly = cv2.intersectConvexConvex(pp,qq)
    inter = np.empty((0,2)) if poly is None else poly.reshape(-1,2)+origin
    return float(area),inter,float(cv2.contourArea(pp)),float(cv2.contourArea(qq))

def read_sources():
    profiles = {}
    for folder in ['quality/source_rasters','quality_retry1/source_rasters',
                   'ports_recovery/quality/source_rasters','melbourne_zero_aware/quality/source_rasters',
                   'melbourne_zero_aware/quality_retry1/source_rasters','s1_sf_extension/quality_retry1/source_rasters']:
        for p in (DATA/folder).glob('*.json'):
            r=json.loads(p.read_text()); profiles[r['url']]=r
    return profiles

def inspect(row, profiles, contexts):
    sid=row['pair_id']+'__'+row['chip_id']; row=dict(row,sample_id=sid)
    arrays,valid,images,infos,mapper=geo.load_pair(row)
    masks,_,counts=geo.overlap(valid,mapper)
    metrics=[geo.metric_geometry(i) for i in infos]
    poly=[metric_polygon(i,metrics[0]['local_metric_crs']) for i in infos]
    area,ip,a,b=intersection(*poly)
    meta=json.loads((Path(row['folder'])/'metadata.json').read_text())
    checks=[]
    for i,prefix in enumerate(['reference','moving']):
        origin=meta['reference_window_xy' if i==0 else 'moving_native_window_xy']
        url=meta[prefix+'_path']; record=profiles.get(url)
        source_profile=record['profile'] if record else None
        if source_profile is None and sid in contexts:
            z=contexts[sid]['views'][i]['source_metadata'];source_profile=dict(z,transform=z['source_transform'])
        expected=Affine(*source_profile['transform'])*Affine.translation(*origin) if source_profile else None
        identity=bool(meta[prefix+'_acquisition_id']==row[prefix+'_acquisition_id'] and meta[prefix+'_utc']==row[prefix+'_utc'] and url==row[prefix+'_asset_url'])
        size_ok=bool(source_profile and origin[0]>=0 and origin[1]>=0 and origin[0]+512<=source_profile['width'] and origin[1]+512<=source_profile['height'])
        check={'side':prefix,'original_url':url,'acquisition_id':row[prefix+'_acquisition_id'],'utc':row[prefix+'_utc'],'native_origin_xy':origin,'native_info':infos[i],'source_profile':source_profile,'profile_evidence':str(record.get('collection_identity_evidence','local original raster QA profile')) if record else 'existing context acquisition record',
               'index_crop_metadata_identity_equal':identity,'window_inside_source':size_ok,
               'crop_transform_matches_source_window':bool(expected is not None and np.allclose(list(expected)[:6],infos[i]['transform'],atol=1e-8,rtol=0)),
               'crop_crs_matches_source':bool(source_profile and rasterio.crs.CRS.from_user_input(source_profile['crs'])==rasterio.crs.CRS.from_user_input(infos[i]['crs'])),
               'previous_native_pixel_equality':row.get('native_pixel_equality_passed'),
               'provider_metadata_file':row[prefix+'_provider_metadata'] if prefix+'_provider_metadata' in row else row['provider_metadata_'+prefix]}
        if sid in contexts:
            cx=contexts[sid]; cp=Path(cx['context_folder'])/f'view{i}_float.tif';off=np.array(origin)-np.array(cx['views'][i]['window_xy'])
            with rasterio.open(cp) as ds:
                x,y=map(int,off);central=ds.read(1,window=rasterio.windows.Window(x,y,512,512));cm=ds.read_masks(1,window=rasterio.windows.Window(x,y,512,512))>0
                check['context']={'path':str(cp),'shape':[ds.height,ds.width],'offset_xy':off.tolist(),'same_url':cx['views'][i]['source']==url,'pixels_equal_on_valid':bool(np.array_equal(central[valid[i]].astype(arrays[i].dtype),arrays[i][valid[i]])),
                    'validity_equal':bool(np.array_equal(cm,valid[i])),'transform_equal':bool(np.allclose(list(ds.transform*Affine.translation(x,y))[:6],infos[i]['transform'],atol=1e-8,rtol=0)),
                    'source_window_transform_equal':bool(np.allclose(list(Affine(*source_profile['transform'])*Affine.translation(*cx['views'][i]['window_xy']))[:6],list(ds.transform)[:6],atol=1e-8,rtol=0))}
        checks.append(check)
    centers=[]
    for i in infos:
        x,y=Affine(*i['transform'])*(i['width']/2,i['height']/2);lon,lat=transform(i['crs'],'EPSG:4326',[x],[y]);centers.append([lon[0],lat[0]])
    cxy=transform('EPSG:4326',metrics[0]['local_metric_crs'],[x[0] for x in centers],[x[1] for x in centers]);c=np.column_stack(cxy)
    joint,independent,disp=display_pair(arrays,valid,[row['reference_input_units'],row['moving_input_units']])
    record={'sample_id':sid,'pair_id':row['pair_id'],'site_name':row['site_name'],'sensor':row['sensor'],'scene_class':row['scene_class'],'component':row['component'],'split':row['split'],'look_pair_category':row['look_pair_category'],
        'distinct_acquisition_and_date':row['reference_acquisition_id']!=row['moving_acquisition_id'] and row['reference_utc']!=row['moving_utc'],
        'checks':checks,'valid_fraction':[float(v.mean()) for v in valid],'effective_overlap':counts,
        'metric_geometry':metrics,'center_lon_lat':centers,'geocoded_center_separation_m':float(np.linalg.norm(c[0]-c[1])),
        'geographic_polygon_area_m2':[a,b],'intersection_area_m2':area,'polygon_overlap_fractions':[area/a,area/b],
        'metric_polygons':[x.tolist() for x in poly],'intersection_metric_polygon':ip.tolist(),'numerical_roundtrip':geo.roundtrip(mapper,geo.regular_grid((512,512),16)),
        'display_record':disp,'static_common_physical_structure':'pending_human_confirmation','GT_qualified':False}
    engineering=record['distinct_acquisition_and_date'] and all(all(z.get(k,False) for k in ['index_crop_metadata_identity_equal','window_inside_source','crop_transform_matches_source_window','crop_crs_matches_source']) for z in checks)
    record['engineering_metadata_checks_passed']=bool(engineering)
    return record,(arrays,valid,images,infos,mapper,masks,joint,independent)

def panel(images, labels, title, output, width=512):
    h=max(round(im.shape[0]*width/im.shape[1]) for im in images)
    canvas=Image.new('RGB',(len(images)*width, h+95),'#171b22');d=ImageDraw.Draw(canvas);d.text((10,8),title,fill='white',font=FONT)
    for i,(im,label) in enumerate(zip(images,labels)):
        img=Image.fromarray(im).convert('RGB');img=img.resize((width,h),Image.Resampling.NEAREST);canvas.paste(img,(i*width,65));d.text((i*width+10,36),label,fill='white',font=FONT)
    canvas.save(output)

def scan():
    require_unsealed()
    if (ROOT/'audit'/'all191.json').exists():raise FileExistsError('Existing scan output; use a new audit directory.')
    rows=[json.loads(l) for l in (DATA/'dataset_index.jsonl').read_text().splitlines()]
    contexts={r['sample_id']:r for r in json.loads((CONTEXT/'contexts_resolved.json').read_text())['records'] if r['available']}
    profiles=read_sources(); results=[]; thumbs={}
    for row in rows:
        r,pack=inspect(row,profiles,contexts);results.append(r);joint,independent=pack[-2:];sid=r['sample_id']
        thumbs.setdefault(r['site_name'],[]).append((r,independent))
        print('AUDIT',sid,'pass',r['engineering_metadata_checks_passed'],'overlap',*[round(x['fraction_of_valid_pixels'],4) for x in r['effective_overlap']],flush=True)
        if sid in FIVE:
            dump(ROOT/'audit'/f'{sid}.json',r)
    dump(ROOT/'audit'/'all191.json',results)
    fields=['sample_id','pair_id','site_name','sensor','look_pair_category','engineering_metadata_checks_passed','geocoded_center_separation_m','overlap_reference','overlap_moving','side_reference_m','side_moving_m','context_available']
    with (ROOT/'audit'/'geography191.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:
            w.writerow({**{k:r[k] for k in fields if k in r},'overlap_reference':r['effective_overlap'][0]['fraction_of_valid_pixels'],'overlap_moving':r['effective_overlap'][1]['fraction_of_valid_pixels'],'side_reference_m':r['metric_geometry'][0]['coverage_edge_lengths_m'][0],'side_moving_m':r['metric_geometry'][1]['coverage_edge_lengths_m'][0],'context_available':'context' in r['checks'][0]})
    for site, items in thumbs.items():
        for page,start in enumerate(range(0,len(items),12)):
            subset=items[start:start+12];canvas=Image.new('RGB',(1536,((len(subset)+2)//3)*330+45),'#171b22');d=ImageDraw.Draw(canvas);d.text((8,8),site+' | metadata + native appearance screening ONLY',fill='white',font=FONT)
            for j,(r,ims) in enumerate(subset):
                x=(j%3)*512;y=(j//3)*330+45
                for k,im in enumerate(ims):canvas.paste(Image.fromarray(im).convert('RGB').resize((256,256)),(x+k*256,y+58))
                ov=[round(q['fraction_of_valid_pixels']*100,1) for q in r['effective_overlap']]
                d.text((x+6,y+3),r['sample_id'],fill='white',font=FONT)
                d.text((x+6,y+27),f"{r['look_pair_category'].replace('repeat_pass_same_look','same').replace('different_look_verified','different')} | overlap {ov}",fill='white',font=FONT)
            canvas.save(ROOT/'figures'/f"screen_{site.replace('/','_').replace(' ','_').replace(',','')}_{page}.png")
    dump(ROOT/'audit'/'scan_summary.json',{'total':len(results),'engineering_pass':sum(r['engineering_metadata_checks_passed'] for r in results),'distinct_acquisition_pairs':len({r['pair_id'] for r in results}),'sites':len(thumbs),'contexts_available':len(contexts),'current_raw_product_reaccess':False,'previous_original_pixel_evidence':'existing frozen release QA; selected five revalidated exactly against local 1024 raw context','human_qualification':0})

def refresh_profiles():
    """Complete locally saved split-specific QA evidence without rerunning pixels."""
    require_unsealed()
    profiles=read_sources(); results=json.loads((ROOT/'audit'/'all191.json').read_text())
    rows={r['pair_id']+'__'+r['chip_id']:r for r in [json.loads(l) for l in (DATA/'dataset_index.jsonl').read_text().splitlines()]}
    for r in results:
        for z in r['checks']:
            q=profiles[z['original_url']];p=q['profile'];t=Affine(*p['transform'])*Affine.translation(*z['native_origin_xy']);i=z['native_info'];x,y=z['native_origin_xy']
            z.update(source_profile=p,profile_evidence=q.get('collection_identity_evidence','local original raster QA profile'),window_inside_source=bool(0<=x and 0<=y and x+512<=p['width'] and y+512<=p['height']),crop_transform_matches_source_window=bool(np.allclose(list(t)[:6],i['transform'],atol=1e-8,rtol=0)),crop_crs_matches_source=bool(rasterio.crs.CRS.from_user_input(p['crs'])==rasterio.crs.CRS.from_user_input(i['crs'])))
        r['engineering_metadata_checks_passed']=r['distinct_acquisition_and_date'] and all(all(z[k] for k in ['index_crop_metadata_identity_equal','window_inside_source','crop_transform_matches_source_window','crop_crs_matches_source']) for z in r['checks'])
        if r['sample_id'] in FIVE:dump(ROOT/'audit'/f"{r['sample_id']}.json",r)
    dump(ROOT/'audit'/'all191.json',results)
    summary=json.loads((ROOT/'audit'/'scan_summary.json').read_text());summary.update(engineering_pass=sum(r['engineering_metadata_checks_passed'] for r in results),profile_records=len(profiles),initial_incomplete_QA_profile_scan='64 records initially had missing split-specific QA evidence; resolved using locally retained ports/zero-aware/S1 QA, no data repair was needed')
    dump(ROOT/'audit'/'scan_summary.json',summary)
    fields=['sample_id','pair_id','site_name','sensor','look_pair_category','engineering_metadata_checks_passed','geocoded_center_separation_m','overlap_reference','overlap_moving','side_reference_m','side_moving_m','context_available']
    with (ROOT/'audit'/'geography191.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in results:w.writerow({**{k:r[k] for k in fields if k in r},'overlap_reference':r['effective_overlap'][0]['fraction_of_valid_pixels'],'overlap_moving':r['effective_overlap'][1]['fraction_of_valid_pixels'],'side_reference_m':r['metric_geometry'][0]['coverage_edge_lengths_m'][0],'side_moving_m':r['metric_geometry'][1]['coverage_edge_lengths_m'][0],'context_available':'context' in r['checks'][0]})
    print('PROFILE REFRESH',summary,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['scan','refresh_profiles']);args=parser.parse_args()
    scan() if args.action=='scan' else refresh_profiles()
