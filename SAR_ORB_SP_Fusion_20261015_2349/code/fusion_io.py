"""Experiment-local IO with a distinct name; historical common modules stay intact."""
import os,sys,json,csv,time,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,cv2
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;OLD=BASE/'SAR_SP_LG_V22_20261015_1525'
sys.path.insert(0,str(OLD/'code'))
import evaluate22 as frozen
cv2.setNumThreads(1)
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def write(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True)
 with p.open('x') as f:json.dump(frozen.clean(x),f,ensure_ascii=False,indent=2,allow_nan=False)
def csvwrite(p,rr):
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in frozen.clean(r).items()})
def cache(version,sample,method,mode):
 return OLD/f'predictions/main_{version}/{version}/{sample}/{method}_{mode}_20261015_1525.npz'
def nearest(a,b):
 if not len(a):return np.empty(0)
 if not len(b):return np.full(len(a),np.inf)
 return np.array([x.distance for x in cv2.BFMatcher(cv2.NORM_L2).match(np.asarray(a,np.float32),np.asarray(b,np.float32))])
def overlap_pairs(a,b):
 if not len(a['points0']) or not len(b['points0']):return np.zeros(len(a['points0']),bool)
 return ((np.linalg.norm(a['points0'][:,None]-b['points0'][None],axis=2)<=3)&(np.linalg.norm(a['points1'][:,None]-b['points1'][None],axis=2)<=3)).any(1)
def cells(p,shape):
 h,w=shape
 return set(map(tuple,np.clip((np.asarray(p)*[4/w,4/h]).astype(int),0,3)))
