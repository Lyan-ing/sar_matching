"""Native windows and metric Common-FOV; geography is never scatterer GT."""
import sys
sys.dont_write_bytecode=True
import json,hashlib,importlib.util
from pathlib import Path
import numpy as np,cv2,rasterio
from affine import Affine
from rasterio.warp import transform
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
spec=importlib.util.spec_from_file_location('v2_frozen_geo',BASE/'RealSAR_AutoLabels_20261011_1149/geo.py')
geo=importlib.util.module_from_spec(spec);spec.loader.exec_module(geo)
PixelMap=geo.PixelMap;sample_valid=geo.sample_valid
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def jsonl(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(clean(r),ensure_ascii=False,allow_nan=False)+'\n')
def info(ds):return dict(crs=ds.crs.to_string(),transform=list(ds.transform)[:6],width=ds.width,height=ds.height,dtype=ds.dtypes[0],nodata=ds.nodata)
def polygon(z,crs):
    points=[];t=Affine(*z['transform']);w,h=z['width'],z['height']
    for a,b in [((0,0),(w,0)),((w,0),(w,h)),((w,h),(0,h)),((0,h),(0,0))]:
        points.extend([t*(a[0]+f*(b[0]-a[0]),a[1]+f*(b[1]-a[1])) for f in np.linspace(0,1,16,endpoint=False)])
    x,y=transform(z['crs'],crs,[p[0] for p in points],[p[1] for p in points]);return np.c_[x,y]
def box(x0,y0,x1,y1):return np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float)
def mapping(p):return dict(type='Polygon',coordinates=[np.vstack([p,p[0]]).tolist()])
def poly_intersection(p,q):
    origin=p.mean(0);a=cv2.convexHull((p-origin).astype(np.float32));b=cv2.convexHull((q-origin).astype(np.float32));area,out=cv2.intersectConvexConvex(a,b)
    return np.empty((0,2)) if out is None or area<=0 else out.reshape(-1,2)+origin
def poly_area(p):return abs(float(cv2.contourArea((p-p.mean(0)).astype(np.float32)))) if len(p)>=3 else 0.
def poly_inside(roi,p):
    origin=p.mean(0);contour=cv2.convexHull((p-origin).astype(np.float32))
    return all(cv2.pointPolygonTest(contour,tuple(map(float,v-origin)),False)>=0 for v in roi)
def metric_crs(z):
    t=Affine(*z['transform']);x,y=t*(z['width']/2,z['height']/2);lon,lat=transform(z['crs'],'EPSG:4326',[x],[y]);zone=int(np.clip(np.floor((lon[0]+180)/6)+1,1,60));return f'EPSG:{(32600 if lat[0]>=0 else 32700)+zone}'
def read_window(ds,win,units,allow_zero=False):
    x,y,w,h=map(int,[win.col_off,win.row_off,win.width,win.height])
    if min(x,y)<0 or w<=0 or h<=0 or x+w>ds.width or y+h>ds.height:raise ValueError('window outside native product')
    a=ds.read(1,window=win);v=(ds.read_masks(1,window=win)>0)&np.isfinite(a)
    if units!='db':v&=(a>=0 if allow_zero else a>0)
    z=info(ds);z.update(width=w,height=h,transform=list(ds.window_transform(win))[:6],parent_window_xywh=[x,y,w,h])
    return a,v,z
def write_tif(p,a,v,z):
    if Path(p).exists():raise FileExistsError(p)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(p,'w',driver='GTiff',height=a.shape[0],width=a.shape[1],count=1,dtype=a.dtype,crs=z['crs'],transform=Affine(*z['transform']),compress='deflate') as ds:ds.write(a,1);ds.write_mask(v.astype(np.uint8)*255)
def read_tif(p):
    with rasterio.open(p) as ds:return ds.read(1),(ds.read_masks(1)>0)&np.isfinite(ds.read(1)),info(ds)
def display(a,v,kind='db',units='linear_amplitude'):
    a=np.asarray(a,np.float64);v=np.asarray(v,bool)&np.isfinite(a)
    if not v.any():raise ValueError('no valid display samples')
    floor=None
    if kind=='db' and units!='db':
        positive=a[v&(a>0)];floor=float(positive.min()) if len(positive) else 1.
        a=(10 if units=='linear_power' else 20)*np.log10(np.maximum(a,floor))
    low,high=np.quantile(a[v],[.01,.99]);im=np.zeros(a.shape,np.uint8)
    if high>low:im[v]=np.rint(np.clip((a[v]-low)/(high-low),0,1)*255).astype(np.uint8)
    return im,dict(kind=kind,units=units,percentiles=[1,99],limits=[low,high],zero_display_floor=floor,independent_per_image=True,smoothing=False,calibrated_radiometry=False)
def sample_grid(a,v,z,dest,interpolation=cv2.INTER_LINEAR):
    h,w=dest['height'],dest['width'];result=np.zeros((h,w),np.float32);valid=np.zeros((h,w),bool)
    mapper=PixelMap(Affine(*dest['transform']),dest['crs'],Affine(*z['transform']),z['crs'])
    # Bound temporary coordinate buffers; no dense worldwide map.
    for first in range(0,h,128):
        last=min(h,first+128);y,x=np.mgrid[first:last,0:w];q=mapper(np.c_[x.ravel(),y.ravel()]);m=sample_valid(v,q).reshape(last-first,w)
        xx=q[:,0].reshape(last-first,w).astype(np.float32);yy=q[:,1].reshape(last-first,w).astype(np.float32)
        result[first:last]=cv2.remap(a.astype(np.float32),xx,yy,interpolation,borderMode=cv2.BORDER_CONSTANT,borderValue=0);valid[first:last]=m
    return result,valid
def roi_window(z,roi,crs,pad=1):
    pts=np.asarray(roi);x,y=transform(crs,z['crs'],pts[:,0].tolist(),pts[:,1].tolist());inv=~Affine(*z['transform']);p=np.array([inv*(a,b) for a,b in zip(x,y)])
    lo=np.maximum(0,np.floor(p.min(0)).astype(int)-pad);hi=np.minimum([z['width'],z['height']],np.ceil(p.max(0)).astype(int)+pad)
    if np.any(hi<=lo):raise ValueError('ROI has no native window')
    return Window(int(lo[0]),int(lo[1]),int(hi[0]-lo[0]),int(hi[1]-lo[1]))
def overlap_masks(v0,z0,v1,z1):
    out=[]
    for va,za,vb,zb in [(v0,z0,v1,z1),(v1,z1,v0,z0)]:
        h,w=va.shape;m=np.zeros_like(va);mapper=PixelMap(Affine(*za['transform']),za['crs'],Affine(*zb['transform']),zb['crs'])
        for first in range(0,h,128):
            last=min(h,first+128);y,x=np.mgrid[first:last,0:w];m[first:last]=va[first:last]&sample_valid(vb,mapper(np.c_[x.ravel(),y.ravel()])).reshape(last-first,w)
        out.append(m)
    return out
def center_metric(z,xy,crs):
    p=np.asarray(xy,float).reshape(-1,2);t=Affine(*z['transform']);x=t.a*(p[:,0]+.5)+t.b*(p[:,1]+.5)+t.c;y=t.d*(p[:,0]+.5)+t.e*(p[:,1]+.5)+t.f
    xx,yy=transform(z['crs'],crs,x.tolist(),y.tolist());return np.c_[xx,yy]
