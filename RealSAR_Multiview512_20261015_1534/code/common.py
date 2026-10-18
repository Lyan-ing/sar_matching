"""Real acquisitions only; isolated native-pixel preparation, never physical GT."""
import sys,json,csv,hashlib,re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,cv2,rasterio
from affine import Affine
from rasterio.warp import transform
from rasterio.windows import Window
from rasterio.transform import RPCTransformer
from rasterio.rpc import RPC
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
OLD=BASE/'real_sar_pilot/multisource512_20261010_2352'
AUDIT=BASE/'DualSAR_DatasetPrep_20261014_2131'
ANGLE=BASE/'RealSAR_AngleExpansion_20261012_1641'
FIXED=BASE/'RealSAR_Fixed512_20261014_1743/release_20261014_1756'
sys.path.insert(0,str(BASE/'RealSAR_MultiViewV2_20261012_1259/code'))
from geo_tools import PixelMap,poly_intersection,poly_area,polygon,metric_crs,info,display,write_tif,geo
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def save(p,d):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x') as f:json.dump(clean(d),f,ensure_ascii=False,indent=2,allow_nan=False)
def jsonl(p,rr):
 with Path(p).open('x') as f:
  for r in rr:f.write(json.dumps(clean(r),ensure_ascii=False,allow_nan=False)+'\n')
def table(p,rr,fields):
 with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rr)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def latest(glob):return sorted(ROOT.glob(glob))[-1]
def utc(s):return datetime.fromisoformat(re.sub(r'(\.\d{6})\d+',r'\1',s).replace('Z','+00:00'))
def delta(a,b):return None if a is None or b is None else abs((float(a)-float(b)+180)%360-180)
def binof(d):
 if d is None:return 'unknown'
 for hi,label in [(15,'0-15'),(30,'15-30'),(60,'30-60'),(90,'60-90'),(120.00000001,'90-120')]:
  if d<hi:return label
 return '>120'
def canonical(site):return 's1_14tpn_south_dakota' if site in ['South_Dakota','S1_14TPN_PILOT'] else site.casefold().replace(',','').replace(' ','_')
def component(site):return 'site_'+hashlib.sha256(canonical(site).encode()).hexdigest()[:12]
def relid(ids):return 'R_'+hashlib.sha256('|'.join(sorted(ids)).encode()).hexdigest()[:16]
def footprint(a,crs=None):
 p=np.asarray(a['geometry']['coordinates'][0],float)[:,:2]
 if crs is None:
  lon,lat=p.mean(0);crs=f'EPSG:{(32600 if lat>=0 else 32700)+min(60,max(1,int((lon+180)//6)+1))}'
 x,y=transform('EPSG:4326',crs,p[:,0].tolist(),p[:,1].tolist());return np.c_[x,y],crs
def pixel_world(z,p):
 t=Affine(*z['transform']);p=np.asarray(p,float).reshape(-1,2)
 return np.c_[t.a*(p[:,0]+.5)+t.b*(p[:,1]+.5)+t.c,t.d*(p[:,0]+.5)+t.e*(p[:,1]+.5)+t.f]
def world_pixel(z,p,crs):
 p=np.asarray(p,float).reshape(-1,2)
 x,y=transform(crs,z['crs'],p[:,0].tolist(),p[:,1].tolist());inv=~Affine(*z['transform'])
 return np.array([inv*(a,b) for a,b in zip(x,y)])-.5
def check_geo(ds,allow_gec_rpc=False):
 if ds.rpcs is not None and not allow_gec_rpc:raise ValueError('RPC needs verified terrain/elevation; affine fallback forbidden')
 if ds.tags(ns='GEOLOCATION'):raise ValueError('Nonlinear geolocation arrays unsupported by this bounded builder')
 if ds.gcps[0]:raise ValueError('GCP product requires explicit verified transformer; no affine fallback')
 if ds.crs is None or abs(ds.transform.determinant)<1e-20 or ds.transform==Affine.identity():raise ValueError('Missing/invertible georeferencing required')
 if ds.count!=1:raise ValueError('Single detected SAR band required')
def native_asset(cid,path,origin=None,original=None,metadata=None):
 with rasterio.open(path) as ds:check_geo(ds);z=info(ds)
 return dict(acquisition_id=cid,path=str(Path(path).resolve()),native_info=z,product_origin_xy=origin,
             original_profile=original,metadata=metadata,unresampled_window=True)
def roi_overlap(a,b):
 x=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]));y=max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]));i=x*y
 return i/max(1,a[2]*a[3]+b[2]*b[3]-i)

class NativeGeo:
 """Nonlinear vendor reference-plane mapping, never an estimate of physical GT."""
 def __init__(self,a):
  self.a=a;self.z=a['native_info'];self.d=a.get('geocoding_diagnostic',{});self.rpc=None
  if self.d.get('method')=='RPC_at_vendor_reference_height_for_offline_FOV_only':
   self.rpc=RPCTransformer(RPC.from_gdal(self.d['rpc_coefficients']),RPC_PIXEL_ERROR_THRESHOLD='0.00001',RPC_MAX_ITERATIONS='100')
   self.origin=np.asarray(a['product_origin_xy'],float);self.height=self.d['vendor_rpc_height_off_m']
 def world(self,p):
  p=np.asarray(p,float).reshape(-1,2)
  if self.rpc:
   x,y=self.rpc.xy(p[:,1]+self.origin[1],p[:,0]+self.origin[0],zs=np.full(len(p),self.height),offset='center');return np.c_[x,y]
  p=pixel_world(self.z,p);x,y=transform(self.z['crs'],'EPSG:4326',p[:,0].tolist(),p[:,1].tolist());return np.c_[x,y]
 def pixels(self,ll):
  ll=np.asarray(ll,float).reshape(-1,2)
  if self.rpc:
   r,c=self.rpc.rowcol(ll[:,0],ll[:,1],zs=np.full(len(ll),self.height),op=lambda q:q);return np.c_[c,r]-.5-self.origin
  return world_pixel(self.z,ll,'EPSG:4326')
 def close(self):
  if self.rpc:self.rpc.close()

class NativeMap:
 def __init__(self,a,b):self.a=NativeGeo(a);self.b=NativeGeo(b)
 def __call__(self,p):return self.b.pixels(self.a.world(p))
 def reverse(self):return lambda p:self.a.pixels(self.b.world(p))
 def close(self):self.a.close();self.b.close()

def native_polygon(a,crs):
 z=a['native_info'];points=[]
 for p,q in [((-.5,-.5),(z['width']-.5,-.5)),((z['width']-.5,-.5),(z['width']-.5,z['height']-.5)),((z['width']-.5,z['height']-.5),(-.5,z['height']-.5)),((-.5,z['height']-.5),(-.5,-.5))]:
  points.extend(np.asarray(p)+f*(np.asarray(q)-p) for f in np.linspace(0,1,12,endpoint=False))
 g=NativeGeo(a)
 try:ll=g.world(points)
 finally:g.close()
 x,y=transform('EPSG:4326',crs,ll[:,0].tolist(),ll[:,1].tolist());return np.c_[x,y]
