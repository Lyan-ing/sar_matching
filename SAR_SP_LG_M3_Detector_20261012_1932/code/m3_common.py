"""Isolated experiment identity and unchanged historical data/geometry helpers."""
import os,sys,json,hashlib,tempfile,random,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
FIX=BASE/'SAR_SP_LG_M2_Numerics_20261012_1627';OLD=BASE/'SAR_SP_LG_Train_20261012_1544'
os.environ['TORCH_HOME']=str(BASE/'MaritimeSARRegistrationPhase2/cache/torch')
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
sys.path.insert(0,str(FIX/'code'));sys.path.insert(0,str(BASE/'MaritimeSARRegistrationPhase2'))
from data import pair,warp,valid_at,component
import torch,cv2
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,np.generic):return clean(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
def atomic_json(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with tmp.open('w') as f:json.dump(clean(v),f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def gpu(physical,fraction=.55):
 assert physical in range(8) and os.environ.get('CUDA_VISIBLE_DEVICES')==str(physical),'Explicit current eight-GPU authorization and isolation required'
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 expected=subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-');assert expected==actual
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(fraction);torch.manual_seed(42);np.random.seed(42);random.seed(42)
 return dict(physical_gpu=physical,uuid=actual,logical_device='cuda:0',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19],started=now(),allocator_fraction=fraction)
def save_ckpt(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with tmp.open('wb') as f:torch.save(v,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
 random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])
