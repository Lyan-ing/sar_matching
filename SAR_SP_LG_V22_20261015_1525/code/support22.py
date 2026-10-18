"""V2.2 experiment helpers. Historical modules are read-only dependencies."""
import os,sys,json,csv,time,random,hashlib,subprocess,copy
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;TAG='20261015_1525'
V22=BASE/'SyntheticSAR_V22_20261015_1352';V21=BASE/'SyntheticSAR_V2_20261015_1255'
OLD=BASE/'SAR_SP_LG_Matchability_20261014_2031'
os.environ['TORCH_HOME']=str(BASE/'MaritimeSARRegistrationPhase2/cache/torch')
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
sys.path.insert(0,str(V22/'code'))
from data22 import load_pair,online_pair,tensors22
from sampling22 import BalancedSampler,Online22
from common22 import bilinear_at,transform_points,stable_seed
sys.path.extend([str(BASE/'SAR_SP_LG_StructureStable_20261013_2210/code'),str(BASE/'MaritimeSARRegistrationPhase2'),str(BASE/'DualSAR_DatasetPrep_20261014_2131/code')])
import numpy as np,cv2,torch
from torch.nn import functional as F
from network_corrected import SARNet
from lightglue import LightGlue
from lightglue.superpoint import sample_descriptors,simple_nms
ENTRY=V22/'DATASET_ENTRYPOINTS_V22_20261015_1434.json';E=json.loads(ENTRY.read_text())
M2=BASE/'SAR_SP_LG_M2_Numerics_20261012_1627/runs/m2/best.ckpt'
B=BASE/'SAR_SP_LG_StructureStable_20261013_2210/runs/b/best.ckpt'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def out(name,ext='json',folder=ROOT):return Path(folder)/f'{name}_{TAG}.{ext}'
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,x,replace=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if not replace:assert not p.exists(),p
 t=p.with_name(p.name+f'.tmp.{os.getpid()}');t.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False));os.replace(t,p)
def csvwrite(p,rr):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);keys=list(dict.fromkeys(k for r in rr for k in r))
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in clean(r).items()})
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with t.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore(x):random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in x['cuda']])
def gpu(index):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(index) and torch.cuda.device_count()==1
 expected=subprocess.check_output(['nvidia-smi','-i',str(index),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-');assert actual==expected
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.use_deterministic_algorithms(True,warn_only=True);torch.cuda.set_per_process_memory_fraction(.65)
 return dict(physical_gpu=index,uuid=actual,pid=os.getpid(),started=now())
def score_map(logits):
 s=logits.softmax(1)[:,:64].permute(0,2,3,1);b,h,w,_=s.shape
 return s.reshape(b,h,w,8,8).permute(0,1,3,2,4).reshape(b,h*8,w*8)
class Model(SARNet):
 """Exactly the frozen M2 or B architecture; no extra inference layers."""
 def __init__(self,kind='m2',checkpoint=None,train=False):
  super().__init__('b' if kind=='b' else 'control' if kind=='m2' else 'm0');self.kind=kind
  if kind=='b':self.load_state_dict(torch.load(B,map_location='cpu',weights_only=False)['model_state'],strict=True)
  if checkpoint:
   ck=torch.load(checkpoint,map_location='cpu',weights_only=False);self.load_state_dict(ck['model_state'],strict=True)
  self.requires_grad_(False)
  if train:
   if kind=='m2':self.sp.convDa.requires_grad_(True);self.sp.convDb.requires_grad_(True)
   else:self.structure.requires_grad_(True);self.sp.convPa.requires_grad_(True);self.sp.convPb.requires_grad_(True)
 @torch.no_grad()
 def extract(self,image):
  assert max(image.shape[-2:])<=512,'No implicit resize / external prior'
  logits,d=self.maps(image);s=simple_nms(score_map(logits),4);s[:,:4]=-1;s[:,-4:]=-1;s[:,:,:4]=-1;s[:,:,-4:]=-1
  _,y,x=torch.where(s>.0005);v=s[0,y,x];ids=torch.argsort(v,descending=True,stable=True)[:1024];k=torch.stack([x[ids],y[ids]],1).float()
  return dict(keypoints=k[None],keypoint_scores=v[ids][None],descriptors=sample_descriptors(k[None],d,8).transpose(1,2),image_size=torch.tensor([[image.shape[-1],image.shape[-2]]],device=image.device))
