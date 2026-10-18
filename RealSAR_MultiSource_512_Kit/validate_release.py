"""Independent candidate engineering QA; geographic overlap is NOT registration GT."""
import argparse,csv,hashlib,json,itertools,math
from collections import Counter,defaultdict
from contextlib import ExitStack
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
import cv2,requests,rasterio
from PIL import Image,ImageDraw,ImageFont
from rasterio.warp import reproject,transform,transform_bounds
from rasterio.enums import Resampling
from rasterio.windows import Window
from build_real_pairs import sha256_file,display,spatially_disjoint
from prepare_sources import lines,raster_info,parsed_utc

def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def verify_umbra_identity(src,info,url):
    embedded=info['tags'].get('COLLECT_ID')
    if embedded is not None:
        assert embedded==src['acquisition_id'],('original collect tag mismatch',src['acquisition_id'],embedded)
        return 'embedded_collect_id_verified'
    # Legacy vendor GEC files omit COLLECT_ID. Require the original single-collect
    # sidecar, its fixed hash, identical public folder, task UUID, and exact dimensions.
    path=Path(src['metadata_path']);assert sha256_file(path)==src['metadata_sha256']
    raw=json.loads(path.read_text());collects=raw.get('collects',[])
    assert len(collects)==1 and collects[0]['id']==src['acquisition_id']
    assert url==src['asset_url']
    assert urlsplit(url).path.rsplit('/',1)[0]==urlsplit(src['metadata_url']).path.rsplit('/',1)[0]
    assert '/'+collects[0]['taskId']+'/' in urlsplit(url).path
    assert any(p.get('numColumns')==info['width'] and p.get('numRows')==info['height'] for p in raw.get('derivedProducts',{}).get('GEC',[]))
    return 'legacy_co_located_hashed_sidecar_task_id_and_dimensions_verified__embedded_id_absent'

def qa(run,data,out,sources_manifest=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'source_rasters').mkdir();(out/'overlap_masks').mkdir();(out/'previews').mkdir()
    candidates=lines(data/'candidates.jsonl');sources=lines(Path(sources_manifest) if sources_manifest else run/'sources.jsonl');by_id={r['acquisition_id']:r for r in sources}
    failures=[];checks=[];raster_records=[];raw_hashes=defaultdict(list);splits=defaultdict(set);origin_groups=defaultdict(list)
    with ExitStack() as stack,rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='40',GDAL_HTTP_MAX_RETRY='2',AWS_NO_SIGN_REQUEST='YES'):
        opened={}
        for n,entry in enumerate(candidates):
            folder=data/entry['split']/entry['pair_id']/entry['chip_id'];cid=entry['pair_id']+'_'+entry['chip_id']
            try:
                assert entry['GT_STATUS']=='COARSE_GEO_ONLY__INDEPENDENT_GCP_REQUIRED'
                assert entry['reference_acquisition_id']!=entry['moving_acquisition_id']
                assert parsed_utc(entry['reference_utc'])<parsed_utc(entry['moving_utc'])
                native=[]
                for side,filename,origin_field,pngfile in [('reference','reference_float.tif','reference_window_xy','reference.png'),('moving','moving_native_float.tif','moving_native_window_xy','moving_native.png')]:
                    aid=entry[side+'_acquisition_id'];src=by_id[aid];splits[('acq',aid)].add(entry['split']);splits[('site',src['provider'],src['site_name'])].add(entry['split'])
                    for product in src.get('original_products',[])+src.get('datatake_ids',[]):splits[('original',product)].add(entry['split'])
                    url=entry[side+'_path']
                    if url not in opened:
                        ds=stack.enter_context(rasterio.open(url));info=raster_info(ds)
                        identity='original_sidecar_verified'
                        if src['provider']=='Umbra':identity=verify_umbra_identity(src,info,url)
                        if src['provider']=='Capella':
                            desc=json.loads(info['tags']['TIFFTAG_IMAGEDESCRIPTION']);assert desc['collect']['collect_id']==aid
                        head=requests.head(url,timeout=40);head.raise_for_status()
                        record={'acquisition_id':aid,'url':url,'metadata_path':src['metadata_path'],'provider':src['provider'],
                            'profile':info,'collection_identity_evidence':identity,'http_headers':{k:v for k,v in head.headers.items() if k.lower() in ['etag','last-modified','content-length','content-type','accept-ranges']},
                            'full_product_sha256':None,'download_scope':'original native windows only; no claim of full-product checksum'}
                        save(out/'source_rasters'/(hashlib.sha256(url.encode()).hexdigest()+'.json'),record);raster_records.append(record)
                        opened[url]=ds
                    ds=opened[url];x,y=entry[origin_field];win=Window(x,y,512,512)
                    expected=ds.read(1,window=win,masked=True).astype('float32');arr=expected.filled(-9999);valid=(~np.ma.getmaskarray(expected))&np.isfinite(arr)
                    if entry['input_units']!='db':valid&=arr>=0 if entry.get('allow_declared_valid_zero',False) else arr>0
                    with rasterio.open(folder/filename) as crop:
                        actual=crop.read(1);assert crop.width==crop.height==512 and crop.count==1 and crop.crs==ds.crs
                        assert np.allclose(tuple(crop.transform),tuple(ds.window_transform(win)),atol=1e-9,rtol=0)
                        assert np.array_equal(actual,np.where(valid,arr,-9999)),('source pixel mismatch',cid,side)
                        assert valid.mean()>=.98
                        transform6=list(crop.transform)[:6];native.append({'a':actual,'valid':valid,'transform':crop.transform,'crs':crop.crs})
                        lo,hi=entry['display_db_range'];png=np.asarray(Image.open(folder/pngfile));assert png.shape==(512,512)
                        assert np.array_equal(png,display(actual,valid,lo,hi,entry['input_units']))
                        valid_png='reference_valid.png' if side=='reference' else 'moving_native_valid.png'
                        assert np.array_equal(np.asarray(Image.open(folder/valid_png))>0,valid)
                        coords=[crop.transform*p for p in [(0,0),(512,0),(512,512),(0,512),(0,0)]]
                        lon,lat=transform(crop.crs,'EPSG:4326',[p[0] for p in coords],[p[1] for p in coords]);native[-1]['footprint']={'type':'Polygon','coordinates':[[list(p) for p in zip(lon,lat)]]}
                        # Ground sample distance near chip center, including rotated degree grids.
                        corners=[crop.transform*p for p in [(256,256),(257,256),(256,257)]]
                        px,py=transform(crop.crs,'EPSG:4326',[p[0] for p in corners],[p[1] for p in corners]);meters=[]
                        for j in [1,2]:
                            rad=math.pi/180;dl=(px[j]-px[0])*rad;dp=(py[j]-py[0])*rad;h=math.sin(dp/2)**2+math.cos(py[0]*rad)*math.cos(py[j]*rad)*math.sin(dl/2)**2;meters.append(2*6371008.8*math.asin(math.sqrt(h)))
                        native[-1]['spacing_m_at_center']=meters
                    raw_hashes[hashlib.sha256(actual.tobytes()).hexdigest()].append((cid,side,entry['split']))
                    origin_groups[(entry['pair_id'],side)].append((x,y))
                a,b=native
                masks=[]
                for source,target in [(b,a),(a,b)]:
                    target_mask=np.zeros((512,512),dtype='uint8')
                    reproject(source=source['valid'].astype('uint8'),destination=target_mask,
                        src_transform=source['transform'],src_crs=source['crs'],src_nodata=0,
                        dst_transform=target['transform'],dst_crs=target['crs'],dst_nodata=0,resampling=Resampling.nearest)
                    masks.append((target_mask>0)&target['valid'])
                for side,mask in zip(['reference','moving'],masks):Image.fromarray(mask.astype('uint8')*255).save(out/'overlap_masks'/f'{cid}_{side}_geographic_overlap.png')
                overlap=[float(m.mean()) for m in masks]
                assert entry['common_valid_fraction']>=.98
                common=np.asarray(Image.open(folder/'common_valid.png'))>0
                with rasterio.open(folder/'moving_float.tif') as warped:
                    wb=warped.read(1);wv=np.isfinite(wb)&(wb!=-9999)
                    if entry['input_units']!='db':wv&=wb>=0 if entry.get('allow_declared_valid_zero',False) else wb>0
                    assert np.array_equal(common,a['valid']&wv)
                    assert warped.crs==a['crs'] and warped.transform==a['transform']
                    assert warped.width==warped.height==512
                    lo,hi=entry['display_db_range']
                    assert np.array_equal(np.asarray(Image.open(folder/'moving.png')),display(wb,wv,lo,hi,entry['input_units']))
                attrs=entry['source_attributes'];r={'pair_id':entry['pair_id'],'chip_id':entry['chip_id'],'sensor':entry['sensor'],'site_name':attrs['site_name'],'component':entry['site_id'],'split':entry['split'],
                    'scene_class':attrs['scene_class'],'look_pair_category':attrs['look_pair_category'],'reference_acquisition_id':entry['reference_acquisition_id'],'moving_acquisition_id':entry['moving_acquisition_id'],
                    'native_pixel_equality_passed':True,'coarse_valid_fraction':entry['common_valid_fraction'],'native_geographic_overlap_reference':overlap[0],'native_geographic_overlap_moving':overlap[1],
                    'geographic_overlap_stratum':'low' if min(overlap)<.5 else 'medium' if min(overlap)<.8 else 'high',
                    'native_reference_spacing_m_x':a['spacing_m_at_center'][0],'native_reference_spacing_m_y':a['spacing_m_at_center'][1],
                    'native_moving_spacing_m_x':b['spacing_m_at_center'][0],'native_moving_spacing_m_y':b['spacing_m_at_center'][1],
                    'gt_status':'unverified','independent_check_points':0,'allow_declared_valid_zero':entry.get('allow_declared_valid_zero',False),'folder':str(folder.resolve()),'native_reference_footprint':a['footprint'],'native_moving_footprint':b['footprint'],
                    'geographic_overlap_mask_reference':str((out/'overlap_masks'/f'{cid}_reference_geographic_overlap.png').resolve()),'note':'These footprints/overlap masks derive from provider geocoding and exclude no layover; they are not independent pixel GT'}
                save(out/'overlap_masks'/f'{cid}_geographic_metadata.json',r);checks.append(r)
            except Exception as e:failures.append({'pair_id':entry['pair_id'],'chip_id':entry['chip_id'],'error':repr(e)})
            if (n+1)%15==0:print('QA',n+1,'/',len(candidates),'failures',len(failures),flush=True)
    leaks=[{'key':str(k),'splits':sorted(v)} for k,v in splits.items() if len(v)>1]
    duplicates=[v for v in raw_hashes.values() if len(v)>1]
    duplicate_leaks=[v for v in duplicates if len({x[2] for x in v})>1]
    redundant=[]
    for (pair,side),origins in origin_groups.items():
        for a,b in itertools.combinations(origins,2):
            if not spatially_disjoint(*a,*b,512,0):redundant.append({'pair_id':pair,'side':side,'origins':[a,b]})
    passed=bool(candidates) and len(checks)==len(candidates) and not failures and not leaks and not duplicate_leaks and not any(r['side']=='reference' for r in redundant)
    fields=[k for k in checks[0] if k not in ['native_reference_footprint','native_moving_footprint']] if checks else []
    with (out/'per_chip_quality.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(checks)
    (out/'release_index.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in checks))
    grouped=defaultdict(list)
    for r in checks:grouped[(r['sensor'],r['site_name'],r['scene_class'],r['split'])].append(r)
    statistics=[]
    for key,g in grouped.items():statistics.append({'sensor':key[0],'site':key[1],'scene_class':key[2],'split':key[3],'acquisition_pairs':len({r['pair_id'] for r in g}),'crop_pairs':len(g),'acquisitions':len({r[k] for r in g for k in ['reference_acquisition_id','moving_acquisition_id']}),'different_look_crop_pairs':sum(r['look_pair_category']=='different_look_verified' for r in g),'min_native_geographic_overlap':min(min(r['native_geographic_overlap_reference'],r['native_geographic_overlap_moving']) for r in g)})
    save(out/'by_source_site_scene.json',statistics)
    # Overview plus full 512-pixel panels. No matching-based alignment in these figures.
    panels=[]
    for key,g in grouped.items():
        r=g[0];folder=Path(r['folder']);canvas=Image.new('RGB',(1536,600),'white');draw=ImageDraw.Draw(canvas)
        for j,name in enumerate(['reference.png','moving_native.png','moving.png']):canvas.paste(Image.open(folder/name).convert('RGB'),(512*j,58));draw.text((512*j+5,35),['Reference native','Moving native','Moving coarse geocoded'][j],fill='black')
        draw.text((5,5),f"{r['sensor']} / {r['site_name']} / {r['look_pair_category']} / NO INDEPENDENT GT",fill='black')
        attribution='Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026' if r['sensor']=='Sentinel-1' else 'Umbra SAR Open Data, CC BY 4.0; cropped and display-stretched' if r['sensor']=='Umbra' else 'Copyright Capella Space Corp, All Rights Reserved. CC BY 4.0; cropped and display-stretched'
        draw.text((5,580),attribution,fill='black')
        canvas.save(out/'previews'/f"{r['sensor'].replace('-','')}_{r['pair_id']}.png");panels.append(canvas.resize((768,300)))
    contact=Image.new('RGB',(1536,300*math.ceil(len(panels)/2)+45),'white');d=ImageDraw.Draw(contact);d.text((5,5),'Distinct real SAR acquisitions; native / native / coarse. Umbra SAR, CC BY 4.0; Copyright Capella Space Corp, CC BY 4.0. NO INDEPENDENT GT.',fill='black');d.text((5,20),'Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026; cropped and display-stretched.',fill='black')
    for i,p in enumerate(panels):contact.paste(p,(768*(i%2),45+300*(i//2)))
    contact.save(out/'previews/contact_sheet.png')
    save(out/'ACCEPTANCE.json',{'engineering_QA_passed':passed,'checked_crop_pairs':len(candidates),'passed_crop_pairs':len(checks),'acquisition_pairs':len({r['pair_id'] for r in checks}),
       'unique_acquisitions':len({r[k] for r in checks for k in ['reference_acquisition_id','moving_acquisition_id']}),
       'sensors':dict(Counter(r['sensor'] for r in checks)),'sites':len(grouped),'look_crop_pairs':dict(Counter(r['look_pair_category'] for r in checks)),
       'geographic_overlap_crop_strata':dict(Counter(r['geographic_overlap_stratum'] for r in checks)),
       'cross_split_identity_leaks':leaks,'exact_native_duplicate_groups':duplicates,'duplicate_cross_split_leaks':duplicate_leaks,'spatially_overlapping_crops_within_pair':redundant,
       'failures':failures,'independent_GT_verified_pairs':0,'formal_pixel_accuracy_metrics_computed':False,
       'qa_scope':'Original provider collection identities, all native crop pixel equality, masks, georeferencing, timestamps, source splits. Independent static GCPs are pending.'})
    print('Engineering QA passed',passed,'crop pairs',len(checks),'failures',len(failures),flush=True)
    if not passed:raise SystemExit(2)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--sources-manifest',help='Optional combined acquisition manifest for replaying multiple accepted batches');a=p.parse_args();qa(Path(a.run).resolve(),Path(a.data).resolve(),Path(a.out).resolve(),a.sources_manifest)
