"""Image-only native-size runtime. No dataset, GT, CRS or evaluation imports."""
import os,sys,time,json,hashlib,random,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['TORCH_HOME']=str(BASE/'MaritimeSARRegistrationPhase2/cache/torch')
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
sys.path.insert(0,str(BASE/'SAR_SP_LG_StructureStable_20261013_2210/code'))
sys.path.insert(0,str(BASE/'MaritimeSARRegistrationPhase2'))
import numpy as np,cv2,torch
from torch.nn import functional as F
from network_corrected import SARNet
from lightglue import LightGlue
from lightglue.superpoint import sample_descriptors,simple_nms
from sarbench.matchers import Prediction,_mutual_ratio
from sarbench.estimate import estimate
from sarbench.reliability import geometry_diagnostics,gate,spatial
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
TAG='20261016_2234'
WEIGHTS={'m0':BASE/'MaritimeSARRegistrationPhase2/cache/torch/hub/checkpoints/superpoint_v1.pth','m2':BASE/'SAR_SP_LG_M2_Numerics_20261012_1627/runs/m2/best.ckpt','b':BASE/'SAR_SP_LG_StructureStable_20261013_2210/runs/b/best.ckpt','m2v22':BASE/'SAR_SP_LG_V22_20261015_1525/runs/m2_core/best.ckpt','lg':BASE/'MaritimeSARRegistrationPhase2/cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth'}
NAMES={'m0':'原始预训练特征','m2':'历史SAR描述子迁移','b':'历史SAR结构增强','m2v22':'局部监督SAR描述子迁移','orb':'ORB','t1':'SAR结构检测迁移','t2':'SAR描述迁移对照','t3':'SAR检测描述联合迁移','repeat':'最佳迁移方案重复种子'}
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(1048576),b''):h.update(a)
 return h.hexdigest()
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(clean(v),ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,p)
def csvwrite(p,rr):
 import csv
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rr for k in r)));w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in clean(r).items()})
def setup(seed=42):
 physical=int(os.environ['CUDA_VISIBLE_DEVICES']);assert physical in [0,1]
 free,util=map(int,subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
 pids=subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
 own_only=bool(pids) and all((str(ROOT/'code/joint_train.py').encode() in Path(f'/proc/{pid.strip()}/cmdline').read_bytes() or (ROOT.name+'/code/joint_train.py').encode() in Path(f'/proc/{pid.strip()}/cmdline').read_bytes()) for pid in pids)
 assert free>=10000 and (util<=20 or own_only),f'GPU{physical} is occupied by other work: free={free} MiB, utilization={util}%; do not interfere'
 assert torch.cuda.device_count()==1
 torch.set_num_threads(2);cv2.setNumThreads(1);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True,warn_only=True);torch.cuda.set_per_process_memory_fraction(.75)
 return dict(pid=os.getpid(),physical_gpu=physical,uuid=str(torch.cuda.get_device_properties(0).uuid),started=now(),process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],seed=seed)
def score_map(logits):
 s=logits.softmax(1)[:,:64].permute(0,2,3,1);b,h,w,_=s.shape
 return s.reshape(b,h,w,8,8).permute(0,1,3,2,4).reshape(b,h*8,w*8)
class Model(SARNet):
 def __init__(self,kind='b',checkpoint=None):
  super().__init__('b' if kind not in ['m0','m2','m2v22'] else 'm0' if kind=='m0' else 'control')
  self.kind=kind
  if kind!='m0':
   ck=torch.load(checkpoint or WEIGHTS[kind],map_location='cpu',weights_only=False);sd=ck['model_state']
   sd={k:v for k,v in sd.items() if k.startswith(('sp.','structure.'))};self.load_state_dict(sd,strict=True)
  self.requires_grad_(False);self.eval()
 def cached_maps(self,x,responses=None):
  s=self.sp;logits=s.convPb(s.relu(s.convPa(x)));d=s.convDa(x)
  if hasattr(self,'structure'):d=d+self.structure.encode(responses)
  return logits,F.normalize(s.convDb(s.relu(d)).float(),dim=1)
 def maps(self,image):
  x=self.shared(image);rs=F.avg_pool2d(self.structure.responses(image),8,8) if hasattr(self,'structure') else None
  return self.cached_maps(x,rs)
 @torch.no_grad()
 def extract_maps(self,logits,dense,shape):
  assert logits.shape[0]==1
  s=simple_nms(score_map(logits),4);s[:,:4]=-1;s[:,-4:]=-1;s[:,:,:4]=-1;s[:,:,-4:]=-1
  _,y,x=torch.where(s>.0005);v=s[0,y,x];ids=torch.argsort(v,descending=True,stable=True)[:1024];k=torch.stack([x[ids],y[ids]],1).float()
  return dict(keypoints=k[None],keypoint_scores=v[ids][None],descriptors=sample_descriptors(k[None],dense,8).transpose(1,2),image_size=torch.tensor([[shape[1],shape[0]]],device=dense.device))
 @torch.no_grad()
 def extract(self,image):
  assert image.shape[0]==1 and max(image.shape[-2:])<=640 and min(image.shape[-2:])>=64
  return self.extract_maps(*self.maps(image),image.shape[-2:])
def trim(f,n):
 ids=torch.argsort(f['keypoint_scores'][0],descending=True,stable=True)[:n]
 return {k:v[:,ids] if k in ['keypoints','keypoint_scores','descriptors'] else v for k,v in f.items()}
def orb_features(im):
 k,d=cv2.ORB_create(nfeatures=1024,fastThreshold=12).detectAndCompute(im,None)
 d=np.empty((0,32),np.uint8) if d is None else d
 f=dict(keypoints=torch.tensor(np.array([p.pt for p in k],np.float32).reshape(-1,2),device='cuda')[None],keypoint_scores=torch.tensor([p.response for p in k],device='cuda')[None],descriptors=torch.tensor(d,device='cuda')[None],image_size=torch.tensor([[im.shape[1],im.shape[0]]],device='cuda'))
 return trim(f,1024)
@torch.no_grad()
def predict(lg,fs,shape,orb=False):
 if min(len(f['keypoints'][0]) for f in fs)<2:ij=np.empty((0,2),int);conf=np.empty(0)
 elif orb:
  ds=[f['descriptors'][0].cpu().numpy() for f in fs];i,j=_mutual_ratio(*ds,cv2.BFMatcher(cv2.NORM_HAMMING),.8);ij=np.c_[i,j];conf=np.ones(len(i))
 else:
  r=lg(dict(image0=fs[0],image1=fs[1]));ij=r['matches'][0].cpu().numpy();conf=r['scores'][0].cpu().numpy()
 ks=[f['keypoints'][0].cpu().numpy() for f in fs];p=Prediction('ORB' if orb else 'SP+LG',ks[0][ij[:,0]],ks[1][ij[:,1]],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1])
 e=estimate(p,shape,model='homography',seed=42);diag=geometry_diagnostics(e,p,shape,shape,42) if e.candidate_H is not None else {};e=gate(e,diag,'full')
 pred=dict(accepted=e.accepted,H_AB=e.candidate_H,matches_A=p.points0,matches_B=p.points1,keypoints_A=ks[0],keypoints_B=ks[1])
 return pred,e,diag,conf
