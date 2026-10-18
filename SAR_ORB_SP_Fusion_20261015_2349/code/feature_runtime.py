"""Image-only runtime: local frozen weights, no manifests, masks, geography or labels."""
import os,sys,time
from pathlib import Path
BASE=Path(__file__).resolve().parents[2]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ['TORCH_HOME']=str(BASE/'MaritimeSARRegistrationPhase2/cache/torch')
sys.path.extend([str(BASE/'SAR_SP_LG_StructureStable_20261013_2210/code'),str(BASE/'MaritimeSARRegistrationPhase2')])
import numpy as np,cv2,torch
from network_corrected import SARNet
from lightglue import LightGlue
from lightglue.superpoint import simple_nms,sample_descriptors
from sarbench.matchers import Prediction,_mutual_ratio
from fusion_core import strategies

WEIGHTS={'M2':BASE/'SAR_SP_LG_M2_Numerics_20261012_1627/runs/m2/best.ckpt','B':BASE/'SAR_SP_LG_StructureStable_20261013_2210/runs/b/best.ckpt','M2_V22':BASE/'SAR_SP_LG_V22_20261015_1525/runs/m2_core/best.ckpt'}
def initialize(gpu):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu) and torch.cuda.device_count()==1
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 models={}
 for method,path in WEIGHTS.items():
  m=SARNet('b' if method=='B' else 'control');state=torch.load(path,map_location='cpu',weights_only=False)['model_state']
  if method=='M2':
   # Historical training also saved frozen teacher heads; inference uses only sp.*.
   assert all(k.startswith(('sp.','teacherDa.','teacherDb.')) for k in state)
   state={k:v for k,v in state.items() if k.startswith('sp.')}
  m.load_state_dict(state,strict=True);models[method]=m.cuda().eval().requires_grad_(False)
 return models,LightGlue(features='superpoint').cuda().eval().requires_grad_(False)
@torch.no_grad()
def features(m,im):
 if m is None:
  k,d=cv2.ORB_create(nfeatures=1024,fastThreshold=12).detectAndCompute(im,None);d=np.empty((0,32),np.uint8) if d is None else d
  return dict(keypoints=torch.tensor(np.array([p.pt for p in k],np.float32).reshape(-1,2),device='cuda')[None],keypoint_scores=torch.tensor([p.response for p in k],device='cuda')[None],descriptors=torch.tensor(d,device='cuda')[None],image_size=torch.tensor([[im.shape[1],im.shape[0]]],device='cuda'))
 im=torch.as_tensor(im.copy(),device='cuda').float()[None,None]/255;logits,dense=m.maps(im);s=logits.softmax(1)[:,:64].permute(0,2,3,1);bs,h,w,_=s.shape;s=s.reshape(bs,h,w,8,8).permute(0,1,3,2,4).reshape(bs,h*8,w*8);s=simple_nms(s,4);s[:,:4]=-1;s[:,-4:]=-1;s[:,:,:4]=-1;s[:,:,-4:]=-1
 _,y,x=torch.where(s>.0005);v=s[0,y,x];ids=torch.argsort(v,descending=True,stable=True)[:1024];k=torch.stack([x[ids],y[ids]],1).float()
 return dict(keypoints=k[None],keypoint_scores=v[ids][None],descriptors=sample_descriptors(k[None],dense,8).transpose(1,2),image_size=torch.tensor([[im.shape[-1],im.shape[-2]]],device='cuda'))
def trim(f,n):
 ids=torch.argsort(f['keypoint_scores'][0],descending=True,stable=True)[:n]
 return {k:(v[:,ids] if k in ['keypoints','keypoint_scores','descriptors'] else v) for k,v in f.items()}
@torch.no_grad()
def match(lg,fs,orb=False):
 ks=[f['keypoints'][0].cpu().numpy() for f in fs]
 if orb:
  ds=[f['descriptors'][0].cpu().numpy() for f in fs];i,j=_mutual_ratio(*ds,cv2.BFMatcher(cv2.NORM_HAMMING),.8);ij=np.c_[i,j];confidence=np.ones(len(i))
 elif min(map(len,ks))==0:ij=np.empty((0,2),int);confidence=np.empty(0)
 else:
  out=lg(dict(image0=fs[0],image1=fs[1]));ij=out['matches'][0].cpu().numpy();confidence=out['scores'][0].cpu().numpy()
 return Prediction('ORB' if orb else 'SP+LG',ks[0][ij[:,0]],ks[1][ij[:,1]],keypoints0=ks[0],keypoints1=ks[1],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1])),confidence
@torch.no_grad()
def infer_images(images,models,lg):
 assert all(x is not None and x.ndim==2 and max(x.shape)<=512 for x in images)
 torch.cuda.synchronize();start=time.perf_counter();allf={m:[trim(features(net,im),1024) for im in images] for m,net in dict(models,ORB=None).items()};torch.cuda.synchronize();feature_ms=1000*(time.perf_counter()-start)
 budgets=[min(128,*[len(f[i]['keypoints'][0]) for f in allf.values()]) for i in [0,1]];out=[]
 for spquota in [64,96]:
  sn=[int(t*spquota//128) for t in budgets];on=[t-s for t,s in zip(budgets,sn)]
  start=time.perf_counter();orb,oc=match(lg,[trim(f,n) for f,n in zip(allf['ORB'],on)],True);orb_ms=1000*(time.perf_counter()-start)
  for method in models:
   torch.cuda.synchronize();start=time.perf_counter();sp,sc=match(lg,[trim(f,n) for f,n in zip(allf[method],sn)]);torch.cuda.synchronize();match_ms=1000*(time.perf_counter()-start)
   outputs=strategies(sp,orb,[im.shape for im in images]);cost=feature_ms+orb_ms+match_ms
   out.append(dict(method=method,split=f'{spquota}_{128-spquota}',budgets=budgets,sp_n=sn,orb_n=on,sp=sp,orb=orb,sp_confidence=sc,orb_confidence=oc,outputs=outputs,feature_ms=feature_ms,orb_ms=orb_ms,sp_match_ms=match_ms,base_ms=cost))
 return out,allf
