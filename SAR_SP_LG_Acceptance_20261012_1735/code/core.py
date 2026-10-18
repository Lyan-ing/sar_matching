"""Locked image-only inference; all GT evaluation is outside this module."""
import os, sys, json, hashlib, time, subprocess
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; BASE=ROOT.parent
FIX=BASE/'SAR_SP_LG_M2_Numerics_20261012_1627'
OLD=BASE/'SAR_SP_LG_Train_20261012_1544'
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
os.environ['TORCH_HOME']=str(BASE/'MaritimeSARRegistrationPhase2/cache/torch')
sys.path.insert(0,str(FIX/'code'));sys.path.insert(0,str(BASE/'MaritimeSARRegistrationPhase2'))
import torch, cv2
from model import load_model
from lightglue import LightGlue
from sarbench.matchers import Prediction,LocalFeatureMatcher,_mutual_ratio
from sarbench.estimate import estimate,perspective
from sarbench.reliability import geometry_diagnostics,gate,spatial
from common import atomic_json,sha,now
WEIGHTS={'m1':OLD/'runs/m1/best.ckpt','m2':FIX/'runs/m2/best.ckpt'}
EXPECTED={'m1':'9825f5457b3ca6df0cf8ddb80e0c70a40e81d1142c871185a2fc57489bf62197','m2':'69e671f743adaf134983a3c01c3f9f1a152c387f8a999218b089b422799ac1ef'}
METHODS=['sift','orb','m0','m1','m2']

def setup_gpu(physical):
 if physical not in (0,2) or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical):raise RuntimeError('Only physical GPU 0 or 2, explicitly isolated')
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 expected=subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-');assert expected==actual
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.cuda.set_per_process_memory_fraction(.75)
 return dict(physical_gpu=physical,uuid=actual,logical_gpu='cuda:0',name=torch.cuda.get_device_name(0),pid=os.getpid(),started=now())

def clean(value):
 if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
 if isinstance(value,(tuple,list)):return [clean(v) for v in value]
 if isinstance(value,np.ndarray):return clean(value.tolist())
 if isinstance(value,np.generic):return clean(value.item())
 if isinstance(value,float) and not np.isfinite(value):return None
 return value

def geometry(pred,shape0,shape1):
 est=estimate(pred,shape0,model='homography',seed=42)
 diag=geometry_diagnostics(est,pred,shape0,shape1,42) if est.candidate_H is not None else {}
 return gate(est,diag,'full'),diag

def summary_prediction(pred,est,diag,ms):
 return clean(dict(n_keypoints0=pred.n_keypoints0,n_keypoints1=pred.n_keypoints1,n_matches=est.n_matches,n_inliers=est.n_inliers,inlier_ratio=est.inlier_ratio,inlier_rmse=est.inlier_rmse,accepted=est.accepted,reason=est.reason,prediction_reason=pred.reason,decision='accepted' if est.accepted else ('rejected' if est.candidate_H is not None else 'no_output'),has_candidate=est.candidate_H is not None,elapsed_ms=ms,geometry=diag))

def arrays(pred,est,scores=None):
 return dict(points0=pred.points0,points1=pred.points1,keypoints0=pred.keypoints0,keypoints1=pred.keypoints1,inlier_mask=est.inlier_mask if est.inlier_mask is not None else np.zeros(len(pred.points0),bool),candidate_H=est.candidate_H if est.candidate_H is not None else np.empty((0,3)),scores=np.asarray(scores if scores is not None else [],np.float32))

class Engine:
 def __init__(self):
  for m,p in WEIGHTS.items():assert sha(p)==EXPECTED[m],f'Changed locked {m} checkpoint'
  self.models={m:load_model(m,WEIGHTS.get(m)) for m in ['m0','m1','m2']}
  self.matcher=LightGlue(features='superpoint').eval().cuda().requires_grad_(False)
  self.classic={m:LocalFeatureMatcher(m,1024,.8) for m in ['sift','orb']}
 @torch.no_grad()
 def features(self,method,image0,image1):
  ims=[torch.from_numpy(x.copy()).cuda().float()[None,None]/255 for x in [image0,image1]]
  return [self.models[method].extract(x) for x in ims]
 @torch.no_grad()
 def from_features(self,method,f0,f1):
  out=self.matcher(dict(image0=f0,image1=f1));ids=out['matches'][0]
  k0,k1=[f['keypoints'][0].cpu().numpy() for f in [f0,f1]]
  pp=Prediction(method,k0[ids[:,0].cpu().numpy()],k1[ids[:,1].cpu().numpy()],n_keypoints0=len(k0),n_keypoints1=len(k1),keypoints0=k0,keypoints1=k1)
  return pp,out['scores'][0].cpu().numpy()
 def classical(self,method,a,b):
  # Enforce a strict shared cap, including SIFT response ties at nfeatures.
  obj=self.classic[method];ks=[];ds=[]
  for im in [a,b]:
   k,d=obj.detector.detectAndCompute(im,None)
   order=sorted(range(len(k)),key=lambda i:(-k[i].response,k[i].pt[1],k[i].pt[0]))[:1024]
   ks.append(np.asarray([k[i].pt for i in order],np.float32).reshape(-1,2));ds.append(d[order] if d is not None else None)
  i,j=_mutual_ratio(ds[0],ds[1],cv2.BFMatcher(obj.norm),.8)
  return Prediction(method,ks[0][i],ks[1][j],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1],reason='no_descriptors' if ds[0] is None or ds[1] is None else ('no_ratio_mutual_matches' if not len(i) else ''))
 def predict(self,method,image0,image1):
  # Public API has only method + pixel arrays. No GT, masks, CRS, look direction.
  if method.startswith('m'):torch.cuda.synchronize()
  start=time.perf_counter();scores=[]
  if method.startswith('m'):pred,scores=self.from_features(method,*self.features(method,image0,image1))
  else:pred=self.classical(method,image0,image1)
  est,diag=geometry(pred,image0.shape,image1.shape)
  if method.startswith('m'):torch.cuda.synchronize()
  ms=1000*(time.perf_counter()-start)
  return pred,est,diag,scores,ms

def save_prediction(folder,method,pred,est,diag,scores,ms):
 folder.mkdir(parents=True,exist_ok=True);npz=folder/(method+'.npz');js=folder/(method+'.json')
 assert not npz.exists() and not js.exists(),'Prediction overwrite prohibited'
 with npz.open('xb') as f:np.savez_compressed(f,**arrays(pred,est,scores))
 atomic_json(js,summary_prediction(pred,est,diag,ms))

def load_prediction(folder,method):
 from sarbench.estimate import Estimated
 r=json.loads((folder/(method+'.json')).read_text());z=np.load(folder/(method+'.npz'),allow_pickle=False)
 H=z['candidate_H'];H=H if H.shape==(3,3) else None
 p=Prediction(method,z['points0'],z['points1'],keypoints0=z['keypoints0'],keypoints1=z['keypoints1'],n_keypoints0=r['n_keypoints0'],n_keypoints1=r['n_keypoints1'])
 e=Estimated(H if r['accepted'] else None,r['accepted'],r['reason'],r['n_matches'],r['n_inliers'],r['inlier_ratio'],r['inlier_rmse'],r['geometry'].get('inlier0_grid_coverage'),None,'homography',H,z['inlier_mask'])
 return p,e,r
