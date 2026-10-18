"""Rasterio CRS transformations and pixel-center maps, no independent GT."""
import json
from pathlib import Path
import numpy as np,cv2,rasterio
from rasterio.warp import transform
from PIL import Image

class PixelMap:
    def __init__(self,A,crsA,B,crsB):
        if not crsA or not crsB:raise ValueError('missing CRS')
        self.A=A;self.B=B;self.crsA=rasterio.crs.CRS.from_user_input(crsA);self.crsB=rasterio.crs.CRS.from_user_input(crsB)
        if not np.isfinite(tuple(A)).all() or not np.isfinite(tuple(B)).all() or abs(A.determinant)<1e-20 or abs(B.determinant)<1e-20:raise ValueError('invalid affine geotransform')
        self.inverseB=~B
    def __call__(self,xy):
        p=np.asarray(xy,np.float64).reshape(-1,2);x=p[:,0]+.5;y=p[:,1]+.5
        mx=self.A.a*x+self.A.b*y+self.A.c;my=self.A.d*x+self.A.e*y+self.A.f
        if self.crsA!=self.crsB:
            tx,ty=transform(self.crsA,self.crsB,mx.tolist(),my.tolist());mx=np.asarray(tx);my=np.asarray(ty)
        qx=self.inverseB.a*mx+self.inverseB.b*my+self.inverseB.c-.5;qy=self.inverseB.d*mx+self.inverseB.e*my+self.inverseB.f-.5
        return np.c_[qx,qy]
    def reverse(self):return PixelMap(self.B,self.crsB,self.A,self.crsA)
    def affine_matrix(self):
        if self.crsA!=self.crsB:return None # projections can be nonlinear; never force a global H
        basis=np.array([[0,0],[1,0],[0,1]],float);q=self(basis)
        return np.array([[q[1,0]-q[0,0],q[2,0]-q[0,0],q[0,0]],[q[1,1]-q[0,1],q[2,1]-q[0,1],q[0,1]],[0,0,1.]])

def sample_valid(mask,xy,tolerance=1e-6):
    """All bilinear supporting neighbors valid; nearest integer within tolerance."""
    p=np.asarray(xy,float).reshape(-1,2);h,w=mask.shape
    finite=np.isfinite(p).all(axis=1);inside=finite&(p[:,0]>=-tolerance)&(p[:,1]>=-tolerance)&(p[:,0]<=w-1+tolerance)&(p[:,1]<=h-1+tolerance)
    result=np.zeros(len(p),bool);idx=np.flatnonzero(inside)
    if len(idx):
        q=np.clip(p[idx],[0,0],[w-1,h-1]);rounded=np.rint(q);q=np.where(np.abs(q-rounded)<=tolerance,rounded,q)
        low=np.floor(q).astype(int);high=np.ceil(q).astype(int)
        result[idx]=mask[low[:,1],low[:,0]]&mask[high[:,1],low[:,0]]&mask[low[:,1],high[:,0]]&mask[high[:,1],high[:,0]]
    return result

def load_pair(row):
    infos=[];arrays=[];valid=[];images=[]
    for side,prefix in enumerate(['reference','moving']):
        with rasterio.open(row[prefix+'_native_geotiff']) as src:
            a=src.read(1);m=src.read_masks(1)>0
            info={'crs':src.crs.to_string() if src.crs else None,'transform':list(src.transform)[:6],'width':src.width,'height':src.height,'nodata':None if src.nodata is None or not np.isfinite(src.nodata) else src.nodata,'dtype':str(a.dtype),'input_units':row[prefix+'_input_units'],'mask_policy':'saved native validity AND GeoTIFF finite/read-mask; declared valid zero retained'}
            t=src.transform
        png=Path(row['folder'])/('reference_valid.png' if side==0 else 'moving_native_valid.png');published=np.array(Image.open(png))>0
        if published.shape!=a.shape:raise ValueError('native mask/image shape mismatch')
        v=published&m&np.isfinite(a);im=np.array(Image.open(row[prefix+'_native_png']))
        if im.shape!=a.shape or im.dtype!=np.uint8:raise ValueError('unexpected native PNG shape/dtype')
        arrays.append(a);valid.append(v);images.append(im);infos.append(info)
    from affine import Affine
    mapper=PixelMap(Affine(*infos[0]['transform']),infos[0]['crs'],Affine(*infos[1]['transform']),infos[1]['crs'])
    return arrays,valid,images,infos,mapper

def pixel_grid(shape):
    y,x=np.indices(shape,dtype=np.float64);return np.c_[x.ravel(),y.ravel()]
def regular_grid(shape,n=16):
    h,w=shape;x,y=np.meshgrid((np.arange(n)+.5)*w/n-.5,(np.arange(n)+.5)*h/n-.5);return np.c_[x.ravel(),y.ravel()]
def overlap(valid,mapper):
    masks=[];mapped=[];counts=[]
    for i,fn in enumerate([mapper,mapper.reverse()]):
        xy=pixel_grid(valid[i].shape);q=fn(xy);joint=valid[i].ravel()&sample_valid(valid[1-i],q);masks.append(joint.reshape(valid[i].shape));mapped.append(q);counts.append({'joint_pixels':int(joint.sum()),'fraction_of_all_pixels':float(joint.mean()),'fraction_of_valid_pixels':float(joint.sum()/max(1,valid[i].sum()))})
    return masks,mapped,counts
def roundtrip(mapper,points):
    q=mapper(points);r=mapper.reverse()(q);e=np.linalg.norm(r-points,axis=1)
    return {'n':len(points),'finite':bool(np.isfinite(e).all()),'max_px':float(e.max()) if len(e) and np.isfinite(e).all() else None,'p90_px':float(np.quantile(e,.9)) if len(e) and np.isfinite(e).all() else None,'note':'numerical coordinate consistency ONLY, not product geolocation accuracy'}

def footprint(info):
    from affine import Affine
    a=Affine(*info['transform']);w,h=info['width'],info['height'];edge=[]
    for p,q in [((0,0),(w,0)),((w,0),(w,h)),((w,h),(0,h)),((0,h),(0,0))]:
        for f in np.linspace(0,1,16,endpoint=False):edge.append(a*(p[0]+f*(q[0]-p[0]),p[1]+f*(q[1]-p[1])))
    xx,yy=transform(info['crs'],'EPSG:4326',[v[0] for v in edge],[v[1] for v in edge]);coords=list(map(list,zip(xx,yy)));coords.append(coords[0]);return {'type':'Polygon','coordinates':[coords]}

def metric_geometry(info):
    from affine import Affine
    a=Affine(*info['transform']);w,h=info['width'],info['height'];corners=np.array([a*(x,y) for x,y in [(0,0),(w,0),(w,h),(0,h),(w/2,h/2),(w/2+1,h/2),(w/2,h/2+1)]])
    lon,lat=transform(info['crs'],'EPSG:4326',corners[:,0].tolist(),corners[:,1].tolist());centerlon,centerlat=lon[4],lat[4];zone=int(np.clip(np.floor((centerlon+180)/6)+1,1,60));epsg=(32600 if centerlat>=0 else 32700)+zone
    xx,yy=transform(info['crs'],f'EPSG:{epsg}',corners[:,0].tolist(),corners[:,1].tolist());p=np.c_[xx,yy];J=np.c_[p[5]-p[4],p[6]-p[4]];spacing=np.linalg.norm(J,axis=0)
    return {'local_metric_crs':f'EPSG:{epsg}','spacing_m_x':float(spacing[0]),'spacing_m_y':float(spacing[1]),'grid_jacobian_m_per_px':J.tolist(),'coverage_edge_lengths_m':np.linalg.norm(np.roll(p[:4],-1,axis=0)-p[:4],axis=1).tolist(),'note':'provider-geocoding-derived local projected grid distances; not independently measured resolution/accuracy'}

def image_quality(raw,valid,image):
    values=image[valid].astype(float)/255;v=raw[valid];blur=cv2.GaussianBlur(image.astype(np.float32)/255,(0,0),2);gx=cv2.Sobel(blur,cv2.CV_32F,1,0);gy=cv2.Sobel(blur,cv2.CV_32F,0,1);grad=np.hypot(gx,gy)
    tiles=[]
    for y in range(0,image.shape[0],64):
        for x in range(0,image.shape[1],64):
            m=valid[y:y+64,x:x+64];patch=blur[y:y+64,x:x+64];std=float(patch[m].std()) if m.any() else None;g=float(grad[y:y+64,x:x+64][m].mean()) if m.any() else None
            tiles.append({'x':x,'y':y,'valid_fraction':float(m.mean()),'normalized_display_blur_std':std,'normalized_display_gradient_mean':g,'low_structure_proxy':std is None or std<.02,'shadow_or_uniform_proxy':bool(m.any() and np.mean(image[y:y+64,x:x+64][m]<=5)>.75)})
    return {'valid_fraction':float(valid.mean()),'display_std':float(values.std()) if len(values) else None,'normalized_display_gradient_mean':float(grad[valid].mean()) if valid.any() else None,'display_dark_fraction':float(np.mean(values<=5/255)) if len(values) else None,'display_saturated_fraction':float(np.mean(values>=250/255)) if len(values) else None,'raw_valid_zero_fraction':float(np.mean(v==0)) if len(v) else None,'raw_distinct_values':int(len(np.unique(v))),'low_structure_tile_fraction':float(np.mean([t['low_structure_proxy'] for t in tiles])),'tiles':tiles,'interpretation':'display/quantization structure proxies ONLY; DN/power/amplitude incomparable across sensors; shadow/layover/dynamics are unverified'}
