"""Single bounded native window per acquisition; paired references, no matcher/GT."""
import sys
sys.dont_write_bytecode=True
import os,json,time,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,rasterio
from affine import Affine
from rasterio.warp import transform
from rasterio.windows import Window
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
V2=BASE/'RealSAR_MultiViewV2_20261012_1259'
sys.path.insert(0,str(V2/'code'))
from geo_tools import (box, poly_inside, polygon, poly_intersection, info, roi_window,
    read_window, read_tif, write_tif, display, save, jsonl, sha, clean, geo, lines)
from audit import footprint

# Selection fixed on acquisition metadata, chronology and local reuse before pixels.
SELECTED=['ANGLE_e76cc299fa7b','ANGLE_00b88634c45f','ANGLE_e3bfffe1139e',
          'ANGLE_5baa3ae0a7c1','ANGLE_75c261e08eae','ANGLE_b28cf6a2cdc9',
          'ANGLE_5da40401a853','ANGLE_e17fe5f59130','ANGLE_325c0714a279','ANGLE_cc7e4d5fcb78']
SIDE_M=1500.; MAX_SIDE=8192

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def event(doc):
    with (ROOT/'window_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(at=now(),**clean(doc)),ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())

def fast_display(a,v,kind,units):
    """Bounded percentile sample for display only; raw intensity never changed."""
    stride=max(1,int(np.ceil(np.sqrt(a.size/1_000_000))))
    sample=a[::stride,::stride];mask=v[::stride,::stride]
    values=sample[mask].astype(np.float32)
    image=a.astype(np.float32)
    floor=None
    if kind=='db' and units!='db':
        positive=values[values>0];floor=float(positive.min()) if positive.size else 1.
        factor=10 if units=='linear_power' else 20
        values=factor*np.log10(np.maximum(values,floor));image=factor*np.log10(np.maximum(image,floor))
    low,high=np.quantile(values,[.01,.99]);out=np.zeros(a.shape,np.uint8)
    if high>low:out[v]=np.rint(np.clip((image[v]-low)/(high-low),0,1)*255).astype(np.uint8)
    return out,dict(kind=kind,units=units,percentiles=[1,99],limits=[float(low),float(high)],
        percentile_sample_stride=stride,percentile_sample_count=len(values),native_pixels_preserved=True,
        display_only=True,smoothing=False,calibrated_radiometry=False,zero_display_floor=floor)

def fetch_window(acq,center,crs):
    folder=ROOT/'assets'/acq['acquisition_id'];folder.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();url=acq['asset_url']; wanted=box(center[0]-SIDE_M/2,center[1]-SIDE_M/2,center[0]+SIDE_M/2,center[1]+SIDE_M/2)
    # Reuse a sufficient retained native context first. Smaller old chips are not upsampled.
    choices=[]
    for m in lines(V2/'regions.jsonl'):
        for i,v in enumerate(m['acquisition_metadata']['views']):
            if v['asset_url']==url:
                p=V2/'regions'/m['sample_id']/f'context{i}.tif'
                z=m['views'][i]['context_info'];poly=polygon(z,crs)
                if poly_inside(wanted,poly):choices.append(p)
    if (folder/'native.tif').exists():
        # Recover a completed raw TIFF after our controlled CPU stop; header only
        # retrieves the original affine. Never repeat the large remote pixel read.
        a,valid,z=read_tif(folder/'native.tif')
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='30',GDAL_HTTP_MAX_RETRY='0',AWS_NO_SIGN_REQUEST='YES'):
            with rasterio.open(url) as ds:
                original=info(ds);origin_xy=(~ds.transform)*(Affine(*z['transform'])*(0,0))
                origin=[int(round(origin_xy[0])),int(round(origin_xy[1])),z['width'],z['height']]
                assert np.allclose(list(ds.transform*Affine.translation(*origin[:2]))[:6],z['transform'],atol=1e-10,rtol=0)
                tags={ns:ds.tags(ns=ns) for ns in ['','RPC']}
        method='retained_complete_native_TIFF_after_own_CPU_stop_no_remote_pixel_reread'
    elif choices:
        src=min(choices,key=lambda p:p.stat().st_size);os.link(src,folder/'native.tif')
        a,valid,z=read_tif(src);origin=None;original=z;method='existing_native_context_hardlink'
    else:
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='45',GDAL_HTTP_CONNECTTIMEOUT='10',
            GDAL_HTTP_MAX_RETRY='0',AWS_NO_SIGN_REQUEST='YES',GDAL_CACHEMAX=128*1024**2,CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
            with rasterio.open(url) as ds:
                if ds.count!=1 or ds.crs is None:raise ValueError('Requires single georeferenced native detected band')
                original=info(ds);win=roi_window(original,wanted,crs)
                # Never silently download a full product; cap to 75% per axis.
                w=min(int(win.width),MAX_SIDE,int(ds.width*.75));h=min(int(win.height),MAX_SIDE,int(ds.height*.75))
                x,y=transform(crs,ds.crs.to_string(),[center[0]],[center[1]]);q=(~ds.transform)*(x[0],y[0])
                left=int(np.clip(round(q[0]-w/2),0,ds.width-w));top=int(np.clip(round(q[1]-h/2),0,ds.height-h));win=Window(left,top,w,h)
                a,valid,z=read_window(ds,win,acq['input_units'],allow_zero=False)
                origin=[left,top,w,h];tags={ns:ds.tags(ns=ns) for ns in ['','RPC']}
        # Encoding only: uncompressed native raster avoids expensive re-encoding.
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
            with rasterio.open(folder/'native.tif','w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,
                dtype=a.dtype,crs=z['crs'],transform=Affine(*z['transform']),tiled=True,blockxsize=512,blockysize=512) as dst:
                dst.write(a,1);dst.write_mask(valid.astype(np.uint8)*255)
        method='new_bounded_larger_native_window_not_full_product'
    if valid.mean()<.75:raise ValueError('Technical validity below .75, retain failure not pad')
    for kind in ['db','linear']:
        im,settings=fast_display(a,valid,kind,acq['input_units'])
        if not (folder/f'native_{kind}.png').exists():Image.fromarray(im).save(folder/f'native_{kind}.png',compress_level=1)
        if kind=='db':
            preview=Image.fromarray(im);preview.thumbnail((1100,1100),Image.Resampling.NEAREST)
            if not (folder/'overview.png').exists():preview.save(folder/'overview.png')
        if not (folder/f'display_{kind}.json').exists():save(folder/f'display_{kind}.json',settings)
    poly=polygon(z,crs)
    meta=dict(acquisition=acq,native_info=z,original_profile=original,original_window_xywh=origin,
        native_tif=str(folder/'native.tif'),source_method=method,unresampled=True,technical_valid_fraction=float(valid.mean()),
        requested_ground_side_m=SIDE_M,ground_bbox_spans_m=(poly.max(0)-poly.min(0)).tolist(),
        native_spacing=geo.metric_geometry(z),raw_sha256=sha(folder/'native.tif'),
        pixel_value_sha256=hashlib.sha256(a.tobytes()).hexdigest(),valid_mask_sha256=hashlib.sha256(valid.tobytes()).hexdigest(),
        elapsed_seconds=time.monotonic()-started,source_tags=tags if not choices else {},
        geocoding='product geocoding only; height errors/layover retained; not GT')
    save(folder/'metadata.json',meta);event(dict(acquisition_id=acq['acquisition_id'],status='built',method=method,shape=list(a.shape),seconds=meta['elapsed_seconds']))
    print('ASSET',acq['acquisition_id'],a.shape,'valid',round(valid.mean(),4),round(meta['elapsed_seconds'],2),flush=True)
    return meta

def run():
    if (ROOT/'regions.jsonl').exists():raise FileExistsError('Completed batch exists; no overwrite')
    pairs={r['pair_id']:r for r in lines(ROOT/'pair_candidates.jsonl')};selected=[pairs[k] for k in SELECTED]
    if not (ROOT/'window_plan.json').exists():save(ROOT/'window_plan.json',dict(at=now(),pair_ids=SELECTED,side_m=SIDE_M,max_native_side=MAX_SIDE,
        selection='metadata-only shortlist; existing acquisition reuse + all requested angle bins; no matching scores',
        GPU=[],CPU_threads=1,GT=0))
    sites={}
    for r in selected:
        for v in r['views']:sites.setdefault(r['site_name'],{})[v['acquisition_id']]=v
    assets={};failures=[]
    for site,vs in sites.items():
        polys=[];crs=None
        for v in vs.values():
            if crs is None:p,crs=footprint(v)
            else:
                q=np.asarray(v['geometry']['coordinates'][0])[:,:2];x,y=transform('EPSG:4326',crs,q[:,0].tolist(),q[:,1].tolist());p=np.c_[x,y]
            polys.append(p)
        joint=polys[0]
        for p in polys[1:]:joint=poly_intersection(joint,p)
        if len(joint)<3:raise ValueError('Selected station has no common geographic center: '+site)
        center=(joint.min(0)+joint.max(0))/2
        for aid,v in vs.items():
            saved=ROOT/'assets'/aid/'metadata.json'
            if saved.exists():assets[aid]=json.loads(saved.read_text());continue
            if saved.parent.exists() and not (saved.parent/'native.tif').exists():
                failures.append(dict(acquisition_id=aid,error='Incomplete previous attempt retained, no silent retry'));continue
            try:assets[aid]=fetch_window(v,center,crs)
            except Exception as e:
                fail=dict(acquisition_id=aid,site=site,error=repr(e));failures.append(fail);event(dict(status='failed',**fail));print('FAILED',fail,flush=True)
    results=[]
    for r in selected:
        ids=[v['acquisition_id'] for v in r['views']]
        if not all(k in assets for k in ids):continue
        sid=r['pair_id']+'__native_000';folder=ROOT/'regions'/sid;folder.mkdir(parents=True,exist_ok=False)
        views=[]
        for i,(aid,acq) in enumerate(zip(ids,r['views'])):
            asset=assets[aid]
            for kind in ['db','linear']:os.link(ROOT/'assets'/aid/f'native_{kind}.png',folder/f'native_{kind}{i}.png')
            views.append(dict(side=i,acquisition_id=aid,native_info=asset['native_info'],metric_geometry=asset['native_spacing'],
                original_native_window_xywh=asset['original_window_xywh'],native_tif=asset['native_tif'],
                native_sha256=asset['raw_sha256'],window_source=asset['source_method']))
        acquisition_metadata=dict(r,views=[dict(v,utc=v['datetime']) for v in r['views']])
        z=dict(sample_id=sid,pair_id=r['pair_id'],legacy_pair_id=r['legacy_pair_id'],sensor=r['sensor'],site_name=r['site_name'],
            component=r['component'],split=r['split'],look_pair_category='metadata_azimuth_'+r['angle_bin'],angle_bin=r['angle_bin'],
            days=r['days'],azimuth_delta_deg=r['azimuth_delta_deg'],incidence_delta_deg=r['incidence_delta_deg'],
            acquisition_metadata=acquisition_metadata,views=views,metric_crs=r['metric_crs'],ground_edge_m=SIDE_M,native_geographic_overlap='not scatterer GT; see acquisition footprints',
            human_readability='pending',independent_GT=False,gt_qualified=False,
            risks=['geographic overlap does not establish common scatterer identity','roofs, layover, shadows, ships and container stacks are unsafe GT',
                   'same site/shared collects remain same component','same-pass seconds-apart collects are correlated; not independent temporal observations'])
        save(folder/'metadata.json',z);results.append(z)
    jsonl(ROOT/'regions.jsonl',results);jsonl(ROOT/'candidate_regions.jsonl',results);save(ROOT/'window_failures.json',failures)
    save(ROOT/'window_summary.json',dict(at=now(),selected_pairs=len(selected),built_pairs=len(results),unique_acquisitions=len(assets),
        bins={b:sum(r['angle_bin']==b for r in results) for b in ['0-20','20-40','40-60','60-90','>90']},failures=len(failures),GT=0,
        full_products_downloaded=0,derived_chip_counts_as_new_sources=False,raw_bytes=sum((ROOT/'assets'/k/'native.tif').stat().st_size for k in assets)))
    atlas=Image.new('RGB',(2260,650*len(results)), '#18202a');draw=ImageDraw.Draw(atlas)
    for row,r in enumerate(results):
        draw.text((15,row*650+8),f"{r['pair_id']} | {r['site_name']} | {r['azimuth_delta_deg']:.2f} deg | {r['days']:.5f} days | AI screening only, GT=0",fill='white')
        for side,v in enumerate(r['views']):
            im=Image.open(ROOT/'assets'/v['acquisition_id']/'overview.png');im.thumbnail((1100,600),Image.Resampling.NEAREST);atlas.paste(im,(15+side*1120,row*650+40))
    atlas.save(ROOT/'candidate_atlas.png')
    print('BUILT_PAIRS',len(results),'ASSETS',len(assets),'FAILURES',len(failures),flush=True)

if __name__=='__main__':run()
