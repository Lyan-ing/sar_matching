"""Isolated V2.2, reusing frozen V2.1 geometry/appearance and dependencies."""
from pathlib import Path
import sys,json,hashlib
OLD=Path('/home/yl/matching/sar_matching/SyntheticSAR_V2_20261015_1255')
sys.path.insert(0,str(OLD/'code'))
from common_v2 import *
import dataset_v2 as v21
ROOT=Path(__file__).resolve().parents[1]
ENTRY=json.loads((OLD/'DATASET_ENTRYPOINTS_V2_20261015_1326.json').read_text())
CFG21=json.loads(Path(ENTRY['config']).read_text())
def sources():return rows(CFG21['source_pool'])
def old_rows(arm='V2-R'):
    return rows(ENTRY['fixed_manifests'][f'{arm}/train/all'])+rows(ENTRY['fixed_manifests'][f'{arm}/val/all'])
def cachepath(r):return ROOT/'features'/f"{r['dataset']}_{stable_seed(22,r['image_path']):010d}.npz"
def persistent(a,b,mask,dist=12,tolerance=4.5):
    p=cv2.goodFeaturesToTrack(a,1000,.015,dist,mask=mask.astype('uint8')*255,blockSize=7)
    q=cv2.goodFeaturesToTrack(b,1000,.015,max(4,dist-2),mask=mask.astype('uint8')*255,blockSize=7)
    p=p[:,0] if p is not None else np.empty((0,2),np.float32)
    q=q[:,0] if q is not None else np.empty((0,2),np.float32)
    if len(p) and len(q):return p[np.min(np.sum((p[:,None]-q[None])**2,axis=2),axis=1)<=tolerance*tolerance]
    return np.empty((0,2),np.float32)
def global_ok(p,q,n):
    support=[spatial_support(x,n) for x in [p,q]]
    return all(a['count']>=16 and a['cells4x4']>=4 and a['hull_fraction']>=.05 and a['eigen_ratio']>=.03 for a in support)
def thin(p,q,limit=128):
    if len(p)>limit:
        order=np.lexsort((p[:,0],p[:,1]));ids=order[np.linspace(0,len(p)-1,limit).astype(int)];return p[ids],q[ids]
    return p,q
