"""Bounded anonymous public raster window acquisition. No whole-product download."""
from common import *
import argparse,requests,time,os
from rasterio.transform import RPCTransformer
from urllib.parse import urlsplit,urlunsplit
from concurrent.futures import ThreadPoolExecutor

def alternatives(url):
 u=urlsplit(url);host=u.netloc
 if host=='umbra-open-data-catalog.s3.us-west-2.amazonaws.com':
  return [url,urlunsplit((u.scheme,'s3.us-west-2.amazonaws.com','/umbra-open-data-catalog'+u.path,u.query,u.fragment)),urlunsplit((u.scheme,'umbra-open-data-catalog.s3.amazonaws.com',u.path,u.query,u.fragment))]
 return [url]

def fetch_one(a,center,side=4096,rpc_reference=False):
 folder=ROOT/'acquisitions'/a['acquisition_id'];folder.mkdir(parents=True,exist_ok=True)
 complete=sorted(folder.glob('ASSET_*.json'))
 if complete:return json.loads(complete[-1].read_text())
 if (folder/'native.tif').exists():raise FileExistsError('Incomplete prior raw window retained; recover explicitly, no overwrite')
 if __import__('shutil').disk_usage(ROOT).free<35*1024**3:raise ValueError('35 GiB remaining-space guard')
 attempts=[];start=time.monotonic();t=stamp()
 for url in alternatives(a['asset_url']):
  try:
   h=requests.get(url,headers={'Range':'bytes=0-16383'},timeout=(10,25));h.raise_for_status()
   if h.status_code!=206 or len(h.content)>65536:raise ValueError('Product lacks bounded HTTP range support')
   byte_size=int(h.headers['Content-Range'].split('/')[-1]);remote=dict(url=url,status=h.status_code,content_range=h.headers.get('Content-Range'),etag=h.headers.get('ETag'),last_modified=h.headers.get('Last-Modified'),full_product_bytes=byte_size)
   with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_CONNECTTIMEOUT='10',GDAL_HTTP_TIMEOUT='35',GDAL_HTTP_MAX_RETRY='0',GDAL_CACHEMAX=64*1024**2,AWS_NO_SIGN_REQUEST='YES',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_NUM_THREADS='1'):
    with rasterio.open(url) as ds:
     gec=a['provider']=='Umbra' and '_GEC' in url
     check_geo(ds,allow_gec_rpc=gec);original=info(ds);p=world_pixel(original,[center],'EPSG:4326')[0]
     if ds.tags().get('COLLECT_ID') and ds.tags()['COLLECT_ID']!=a['acquisition_id']:raise ValueError('TIFF COLLECT_ID disagrees with actual sidecar')
     n=min(side,ds.width//2,ds.height//2);n=int(n//512)*512
     if n<1024:raise ValueError('Product too small for safe bounded context')
     x,y=np.rint(p-(n-1)/2).astype(int)
     x=int(np.clip(x,0,ds.width-n));y=int(np.clip(y,0,ds.height-n));win=Window(x,y,n,n)
     rpc_note=dict(RPC_present=ds.rpcs is not None,method='published GEC constant-height affine grid for offline coarse FOV only',orthorectified=False,height_accuracy_not_measured=True)
     if ds.rpcs is not None:
      rpc=ds.rpcs;sample=np.array([[x+n*f-.5,y+n*g-.5] for f in [.2,.5,.8] for g in [.2,.5,.8]])
      ll=pixel_world(original,sample)
      with RPCTransformer(rpc) as tr:
       ry,rx=tr.rowcol(ll[:,0],ll[:,1],zs=np.full(len(ll),rpc.height_off),op=lambda q:q)
      error=np.linalg.norm(np.c_[rx,ry]-.5-sample,axis=1)
      rpc_note.update(rpc_coefficients=rpc.to_gdal(),vendor_rpc_height_off_m=rpc.height_off,numeric_affine_vs_RPC_at_vendor_reference_height_px=error.tolist(),not_scatterer_error=True,official_spec='https://help.umbra.space/product-guide/umbra-products/umbra-product-specifications/geo-ellipsoid-corrected-gec-geotiff')
      if not np.isfinite(error).all():raise ValueError('Nonfinite GEC/RPC mapping')
      if max(error)>3:
       if not rpc_reference:raise ValueError('GEC affine/RPC plane disagreement >3px; explicit terrain review required')
       # Official GEC reference-plane model, not assumed terrain or orthorectification.
       with RPCTransformer(rpc,RPC_PIXEL_ERROR_THRESHOLD='0.00001',RPC_MAX_ITERATIONS='100') as tr:
        ry,rx=tr.rowcol([center[0]],[center[1]],zs=[rpc.height_off],op=lambda q:q)
       p=np.array([rx[0],ry[0]])-.5;x,y=np.rint(p-(n-1)/2).astype(int);x=int(np.clip(x,0,ds.width-n));y=int(np.clip(y,0,ds.height-n));win=Window(x,y,n,n)
       rpc_note.update(method='RPC_at_vendor_reference_height_for_offline_FOV_only',actual_terrain_height_unknown=True,geo_accuracy_not_GT=True)
     arr=ds.read(1,window=win);valid=(ds.read_masks(1,window=win)>0)&np.isfinite(arr)
     z=info(ds);z.update(width=n,height=n,transform=list(ds.window_transform(win))[:6]);blocks=ds.block_shapes
   if valid.mean()<.8:raise ValueError('Context technical valid fraction below .8')
   write_tif(folder/'native.tif',arr,valid,z)
   im,settings=display(arr,valid,units=a.get('input_units','linear_amplitude'));Image.fromarray(im).save(folder/'display.png')
   overview=Image.fromarray(im);overview.thumbnail((1024,1024),Image.Resampling.NEAREST);overview.save(folder/'overview.png')
   result=native_asset(a['acquisition_id'],folder/'native.tif',[x,y],original,str(folder/f'ASSET_{t}.json'))
   result.update(acquisition=a,geocoding_diagnostic=rpc_note,download=dict(remote,elapsed_seconds=time.monotonic()-start,window_xywh=[x,y,n,n],range_read_only=True,block_shapes=blocks),display=str(folder/'display.png'),display_settings=settings,actual_valid_fraction=float(valid.mean()),raw_sha256=sha(folder/'native.tif'),pixel_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),attempts=attempts,geographic_accuracy='vendor GEC only, NOT independently physical GT')
   save(folder/f'ASSET_{t}.json',result);print('FETCHED',a['acquisition_id'],a['site_name'],n,round(time.monotonic()-start,1),flush=True);return result
  except Exception as e:
   attempts.append(dict(url=url,error=repr(e)))
 save(folder/f'FAILURE_{t}.json',dict(at=now(),acquisition=a,attempts=attempts));print('FETCH_FAILED',a['acquisition_id'],str(attempts)[-400:],flush=True);return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--proposal-indices',nargs='+',type=int);ap.add_argument('--plan');ap.add_argument('--workers',type=int,default=2);ap.add_argument('--rpc-reference',action='store_true');args=ap.parse_args()
 inventory={a['acquisition_id']:a for a in rows(latest('ACQUISITION_INVENTORY_*.jsonl'))}
 for p in sorted(ROOT.glob('NEW_METADATA_INVENTORY_*.jsonl')):
  inventory.update({a['acquisition_id']:a for a in rows(p)})
 if args.plan:plan=json.loads(Path(args.plan).read_text())['assets']
 else:
  ps=json.loads((AUDIT/'TARGETED_SOURCE_PROPOSALS_20261014_2329.json').read_text());plan=[]
  for ix in args.proposal_indices:
   q=ps[ix];aa=[inventory[x] for x in q['acquisition_ids']];p,crs=footprint(aa[0]);p=poly_intersection(p,footprint(aa[1],crs)[0]);c=p.mean(0)
   lon,lat=transform(crs,'EPSG:4326',[c[0]],[c[1]])
   for a in aa:plan.append(dict(acquisition_id=a['acquisition_id'],center_lonlat=[lon[0],lat[0]],reason=f'priority source request {ix}; true different collects; metadata only geography'))
  plan=list({p['acquisition_id']:p for p in plan}.values());save(ROOT/f'DOWNLOAD_PLAN_{stamp()}.json',dict(at=now(),assets=plan,context_max_side=4096,estimated_raw_window_total_bytes=len(plan)*4096**2*2,whole_products=False,storage_cap_gib=20,GT=0))
 results=[]
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  futures=[pool.submit(fetch_one,inventory[p['acquisition_id']],p['center_lonlat'],p.get('side',4096),args.rpc_reference) for p in plan]
  for f in futures:results.append(f.result())
 save(ROOT/f'FETCH_SUMMARY_{stamp()}.json',dict(at=now(),requested=len(plan),obtained=sum(r is not None for r in results),assets=[r['metadata'] for r in results if r],GT=0))

if __name__=='__main__':main()
