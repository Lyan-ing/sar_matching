"""Offline product geocoding only. Native coordinates are zero-based pixel centres."""
import sys,json,hashlib,threading
from pathlib import Path
import numpy as np,cv2,rasterio
from affine import Affine
from rasterio.warp import transform
from rasterio.windows import Window
from rasterio.transform import GCPTransformer
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
SOURCE=BASE/'RealSAR_AngleExpansion_20261012_1641'
sys.path.insert(0,str(BASE/'RealSAR_AutoLabels_20261011_1149'))
from geo import PixelMap

def lines(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def project(xy,src,dst):
    p=np.asarray(xy,float).reshape(-1,2)
    if str(src)==str(dst):return p.copy()
    x,y=transform(src,dst,p[:,0].tolist(),p[:,1].tolist())
    out=np.c_[x,y]
    if not np.isfinite(out).all():raise ValueError('nonfinite CRS conversion')
    return out

class View:
    def __init__(self,row,side):
        self.row=row;self.side=side;v=row['views'][side]
        self.path=Path(v['native_tif']);self.lock=threading.RLock()
        self.png={kind:SOURCE/'review/regions'/row['sample_id']/f'native_{kind}{side}.png' for kind in ['db','linear']}
        with rasterio.open(self.path) as d:
            self.width,self.height=d.width,d.height
            self.affine=d.transform;self.crs=d.crs;gcps,gcp_crs=d.gcps
            self.header=dict(crs=str(d.crs),affine=list(d.transform)[:6],width=d.width,height=d.height,
                gcp_count=len(gcps),gcp_crs=str(gcp_crs),RPC_present=d.rpcs is not None,
                geolocation_array_tags=d.tags(ns='GEOLOCATION'),tags=d.tags())
            if d.rpcs is not None:raise ValueError('RPC requires a verified elevation model; affine fallback forbidden')
            if d.tags(ns='GEOLOCATION'):raise ValueError('geolocation arrays require a verified nonlinear transformer; affine fallback forbidden')
            if gcps:
                if not gcp_crs or len(gcps)<3:raise ValueError('insufficient GCP georeferencing')
                self.crs=gcp_crs;self.gcp=GCPTransformer(gcps);self.method='GCPTransformer'
            else:
                if not d.crs or d.transform==Affine.identity() or abs(d.transform.determinant)<1e-20:
                    raise ValueError('missing/invertible georeferencing required')
                self.gcp=None;self.method='Affine';self.inverse=~self.affine
        z=v['native_info']
        assert [self.width,self.height]==[z['width'],z['height']],'frozen TIFF dimensions disagree'
        assert np.allclose(list(self.affine)[:6],z['transform'],rtol=0,atol=1e-10),'frozen TIFF affine disagrees'
        assert str(self.crs)==z['crs'],'frozen TIFF CRS disagrees'
        for p in self.png.values():
            with Image.open(p) as im:assert im.size==(self.width,self.height),'PNG/TIFF coordinates disagree'
        acq=row['acquisition_metadata']['views'][side]
        self.header.update(method=self.method,acquisition_id=v['acquisition_id'],native_tif=str(self.path),
            source_product=acq.get('asset_url',acq.get('asset','unknown')),
            centre_convention='native x=column, y=row; centre (0,0) is affine*(0.5,0.5)',
            original_native_window_xywh=v.get('original_native_window_xywh'),
            physical_scatterer_accuracy='NOT measured; geocoding is not scatterer GT')

    def world(self,p):
        p=np.asarray(p,float).reshape(-1,2)
        if self.gcp:
            with self.lock:x,y=self.gcp.xy(p[:,1],p[:,0],offset='center')
            return np.c_[x,y]
        return np.c_[self.affine.a*(p[:,0]+.5)+self.affine.b*(p[:,1]+.5)+self.affine.c,
                     self.affine.d*(p[:,0]+.5)+self.affine.e*(p[:,1]+.5)+self.affine.f]

    def pixels(self,p):
        p=np.asarray(p,float).reshape(-1,2)
        if self.gcp:
            with self.lock:rows,cols=self.gcp.rowcol(p[:,0],p[:,1],op=lambda x:x)
            return np.c_[cols,rows]-.5
        return np.c_[self.inverse.a*p[:,0]+self.inverse.b*p[:,1]+self.inverse.c-.5,
                     self.inverse.d*p[:,0]+self.inverse.e*p[:,1]+self.inverse.f-.5]

    def inside(self,p):
        p=np.asarray(p,float).reshape(-1,2)
        return np.isfinite(p).all(1)&(p[:,0]>=-.5)&(p[:,1]>=-.5)&(p[:,0]<self.width-.5)&(p[:,1]<self.height-.5)

    def valid(self,p):
        # Read only the at-most 2x2 support. No full-image reads per click.
        x,y=map(float,p)
        if not(0<=x<=self.width-1 and 0<=y<=self.height-1):return False
        x0,y0=int(np.floor(x)),int(np.floor(y));x1,y1=int(np.ceil(x)),int(np.ceil(y))
        with rasterio.open(self.path) as d:
            win=Window(x0,y0,x1-x0+1,y1-y0+1)
            return bool((d.read_masks(1,window=win)>0).all() and np.isfinite(d.read(1,window=win)).all())

    def boundary(self,n=32):
        w,h=self.width,self.height;out=[]
        for a,b in [((-.5,-.5),(w-.5,-.5)),((w-.5,-.5),(w-.5,h-.5)),((w-.5,h-.5),(-.5,h-.5)),((-.5,h-.5),(-.5,-.5))]:
            out.extend(np.asarray(a)+f*(np.asarray(b)-a) for f in np.linspace(0,1,n,endpoint=False))
        return np.asarray(out)

def mapped(a,b,p):
    if a.method==b.method=='Affine':return PixelMap(a.affine,a.crs,b.affine,b.crs)(p)
    return b.pixels(project(a.world(p),a.crs,b.crs))

def overlap(views):
    a,b=views;lon,lat=project(a.world([[(a.width-1)/2,(a.height-1)/2]]),a.crs,'EPSG:4326')[0]
    zone=int(np.clip(np.floor((lon+180)/6)+1,1,60));metric=f'EPSG:{(32600 if lat>=0 else 32700)+zone}'
    rings=[project(v.world(v.boundary()),v.crs,metric) for v in views]
    origin=np.mean(rings[0],axis=0)
    # Reuse existing OpenCV convex intersection in local metric coordinates.
    hulls=[cv2.convexHull((p-origin).astype(np.float32)) for p in rings]
    area,poly=cv2.intersectConvexConvex(*hulls)
    world=np.empty((0,2)) if poly is None or area<=0 else poly.reshape(-1,2).astype(float)+origin
    native=[v.pixels(project(world,metric,v.crs)).tolist() if len(world) else [] for v in views]
    return dict(metric_crs=metric,area_m2=float(max(0,area)),native_polygons=native,
        geographic_polygon=project(world,metric,'EPSG:4326').tolist() if len(world) else [],
        fractions=[float(max(0,area)/cv2.contourArea(h)) for h in hulls],
        note='densified crop-extent convex intersection; validity holes checked per point; NOT common scatterer coverage')

def queue():return lines(SOURCE/'review/candidate_regions.jsonl')
def point_result(row,views,side,xy):
    a,b=views[side],views[1-side];p=np.asarray(xy,float).reshape(1,2)
    if not np.isfinite(p).all() or not a.inside(p)[0]:raise ValueError('source point outside native extent')
    q=mapped(a,b,p);back=mapped(b,a,q)
    if not np.isfinite(q).all():raise ValueError('mapping is not finite')
    lonlat=project(a.world(p),a.crs,'EPSG:4326')[0]
    return dict(sample_id=row['sample_id'],pair_id=row['pair_id'],source_side=side,source_xy=p[0].tolist(),
        target_xy=q[0].tolist(),lon_lat=lonlat.tolist(),source_valid=a.valid(p[0]),
        target_in_extent=bool(b.inside(q)[0]),target_valid=b.valid(q[0]),
        roundtrip_px=float(np.linalg.norm(back-p)),reference_only=True,GT=False,
        warning='mathematical roundtrip is NOT physical scatterer position error; do not force target into bounds')
