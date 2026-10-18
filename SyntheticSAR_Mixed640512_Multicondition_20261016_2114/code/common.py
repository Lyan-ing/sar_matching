"""Small adapter around the executed V2.1/V2.2 SAR pipeline; historical files read-only."""
import os, sys, json, hashlib, math, time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter, defaultdict
import numpy as np
import cv2
cv2.setNumThreads(1)
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
OLD=BASE/'SyntheticSAR_V22_20261015_1352'
sys.path.insert(0,str(OLD/'code'))
from common22 import (v21,CFG21,cachepath,transform_points,view_matrix,warp,
                      overlap_mask,ratio,bilinear_at,spatial_support,angle_of,
                      stable_seed,dilate,erode,read_gray,global_ok,thin)
from data22 import features,ground_points
TAG='20261016_2114'
SCENES=['offshore_vessels_low_texture_sea','nearshore_vessels','port_quay_dense_vessels','other_land_structures']
OVERLAPS=['zero','10_30','30_50','50_70','70_90','90_100']
BOUNDS=[(0,0),(.10,.30),(.30,.50),(.50,.70),(.70,.90),(.90,1.000001)]
ANGLES=[0,15,30,60,90,120,180.000001]
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ahash(a):return hashlib.sha256(str(a.shape).encode()+np.ascontiguousarray(a).tobytes()).hexdigest()
def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
def jsonl(path,rr):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        for r in rr:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def latest(pattern):return sorted(ROOT.glob(pattern))[-1]
def raw_sources():return rows(CFG21['source_pool'])
def coords(r):
    if r['dataset'] not in ['hrsid','background']:return None
    a=r['source_id'].split('_')
    if len(a)!=5:return None
    # Checked against overlapping pixels: filename order is ROW bounds, then COLUMN bounds.
    y0,y1,x0,x1=map(int,a[1:]);return x0,y0,x1,y1
def source_valid(im):
    # Published display DN has no nodata metadata. Only border-connected exactly-zero
    # regions >=1% of the source are conservatively treated as possible padding.
    n,l,stats,_=cv2.connectedComponentsWithStats((im==0).astype('uint8'),8)
    border=np.unique(np.r_[l[0],l[-1],l[:,0],l[:,-1]])
    bad=[int(k) for k in border if k and stats[k,cv2.CC_STAT_AREA]>=.01*im.size]
    return ~np.isin(l,bad) if bad else np.ones(im.shape,bool)
def overlap_bin(x):
    if x==0:return 'zero'
    for name,(lo,hi) in zip(OVERLAPS[1:],BOUNDS[1:]):
        if lo<=x<hi:return name
    return 'tiny_0_10'
def angle_bin(x):return int(np.searchsorted(ANGLES[1:],x,side='right'))
def png(path,a):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    ok,b=cv2.imencode('.png',a,[cv2.IMWRITE_PNG_COMPRESSION,2]);assert ok
    with path.open('xb') as f:f.write(b)
