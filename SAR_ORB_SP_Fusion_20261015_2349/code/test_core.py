"""Small prediction-only regression checks; not scientific accuracy evidence."""
from pathlib import Path
import sys,inspect
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'MaritimeSARRegistrationPhase2'))
from fusion_core import *
def main():
 rng=np.random.default_rng(42);a=rng.uniform(30,220,(32,2)).astype(np.float32);b=a+[5,-7]
 sp=Prediction('SP',a[:16],b[:16],keypoints0=a[:16],keypoints1=b[:16]);orb=Prediction('ORB',a[16:],b[16:],keypoints0=a[16:],keypoints1=b[16:]);r=strategies(sp,orb,[(256,256)]*2)
 assert r['F3'][1].accepted and r['F3'][3]['route']=='cross_supported_balanced';assert len(r['F3'][0].keypoints0)==32
 for method in r:assert np.max(np.abs(perspective(r[method][1].candidate_H,a)-b))<1e-3
 empty=Prediction('empty');r=strategies(empty,empty,[(256,256)]*2);assert all(not x[1].accepted for x in r.values())
 k,d,c=clean_pool(np.array([[0,0],[0,0],[10,10]]),np.array([[1,1],[20,20],[11,11]]),np.array([0,1,1]));assert c==1 and k.tolist()==[2]
 k,d,c=clean_pool(np.array([[0,0],[0,0]]),np.array([[1,1],[1,1]]),np.array([0,1]));assert k.tolist()==[0] and d==1
 assert list(inspect.signature(strategies).parameters)==['sp','orb','shapes']
 print('PASS: image-coordinate geometry, balanced sources, empty pools, duplicate/conflict handling, label-free signature')
if __name__=='__main__':main()
