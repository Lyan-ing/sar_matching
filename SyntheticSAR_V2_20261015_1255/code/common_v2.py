"""Small shared utilities. All data products are new; historical inputs are read only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, hashlib, sys
import numpy as np
import cv2
cv2.setNumThreads(1)
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
V1 = BASE/'DualSAR_DatasetPrep_20261014_2131'
sys.path.insert(0, str(BASE/'MaritimeSARRegDataset'))
from sarreg.geometry import transform_points, view_matrix, warp, overlap_mask, ratio
from sarreg.labels import annotation_mask
from sarreg.io import read_gray
from sarreg.synthesis import stable_seed, _apply_augment
def stamp(): return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def rows(path): return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def save(path, obj):
    with Path(path).open('x', encoding='utf8') as f: json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
def jsonl(path, data):
    with Path(path).open('x', encoding='utf8') as f:
        for r in data: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest_array(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def dilate(m, radius): return cv2.dilate(m.astype('uint8'), np.ones((2*radius+1,)*2, np.uint8))>0
def erode(m, radius): return cv2.erode(m.astype('uint8'), np.ones((2*radius+1,)*2, np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)>0
def bilinear_at(mask, xy):
    xy=np.asarray(xy).reshape(-1,2); good=np.isfinite(xy).all(1)
    ij=np.floor(np.nan_to_num(xy)).astype(int); x,y=ij.T
    good &= (x>=0)&(y>=0)&(x+1<mask.shape[1])&(y+1<mask.shape[0])
    k=np.flatnonzero(good); xx,yy=x[k],y[k]
    good[k] &= mask[yy,xx]&mask[yy+1,xx]&mask[yy,xx+1]&mask[yy+1,xx+1]
    return good
def angle_of(H, size):
    x=y=(size-1)/2; n=H@np.array([x,y,1.]); z=n[2]
    J=(H[:2,:2]*z-n[:2,None]*H[2:3,:2])/(z*z)
    u,_,v=np.linalg.svd(J); a=u@v
    return abs(float(np.rad2deg(np.arctan2(a[1,0],a[0,0]))))
def angle_bin(a): return ['00_15','15_30','30_60','60_90','90_120'][min(4,int(np.searchsorted([15,30,60,90,120],a,side='right')))]
def spatial_support(p, size):
    if not len(p): return dict(count=0,cells4x4=0,hull_fraction=0.,eigen_ratio=0.)
    cells=np.clip((p/size*4).astype(int),0,3)
    hull=cv2.contourArea(cv2.convexHull(p.astype('float32')))/(size*size) if len(p)>2 else 0.
    eig=np.linalg.eigvalsh(np.cov(p.T)) if len(p)>2 else [0,1]
    return dict(count=len(p),cells4x4=len(set(map(tuple,cells))),hull_fraction=float(hull),eigen_ratio=float(eig[0]/max(eig[1],1e-8)))
