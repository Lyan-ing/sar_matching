import os,sys,json,hashlib,random,csv,time,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np,cv2,torch
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;TAG='20261014_2031'
OLD=BASE/'SAR_SP_LG_Train_20261012_1544';FIX=BASE/'SAR_SP_LG_M2_Numerics_20261012_1627';STRUCT=BASE/'SAR_SP_LG_StructureStable_20261013_2210'
os.environ['TORCH_HOME']=str(BASE/'MaritimeSARRegistrationPhase2/cache/torch');os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'mpl_cache'))
sys.path.extend([str(BASE/'MaritimeSARRegistrationPhase2'),str(FIX/'code'),str(BASE/'SAR_SP_LG_M3_Detector_20261012_1932/code'),str(STRUCT/'code')])
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def output(name,ext='json',folder=None):return (Path(folder) if folder else ROOT)/f'{name}_{TAG}.{ext}'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+f'.tmp.{os.getpid()}');temp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False));os.replace(temp,p)
def write_new(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(clean(x),f,ensure_ascii=False,indent=2,allow_nan=False)
def event(name,x,folder=None):
 p=(Path(folder) if folder else ROOT)/f'{name}_{stamp()}.json';version=1
 while p.exists():p=p.with_name(f'{name}_v{version}_{stamp()}.json');version+=1
 write_new(p,dict(created=now(),**x));return p
def readrows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def export_csv(p,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict,tuple)) else v for k,v in clean(row).items()})
def gpu(physical):
 assert physical in range(8) and os.environ.get('CUDA_VISIBLE_DEVICES')==str(physical)
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 expected=subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip();actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-');assert actual==expected
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 torch.cuda.set_per_process_memory_fraction(.75);torch.use_deterministic_algorithms(True,warn_only=True)
 return dict(physical_gpu=physical,uuid=actual,pid=os.getpid(),started=now(),logical='cuda:0')
def save_ckpt(p,x):
 p=Path(p);temp=p.with_name(p.name+f'.tmp.{os.getpid()}')
 with temp.open('wb') as f:torch.save(x,f);f.flush();os.fsync(f.fileno())
 os.replace(temp,p)
def rng_state():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(x):random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch'].cpu());torch.cuda.set_rng_state_all([s.cpu() for s in x['cuda']])
from data import component,warp,valid_at
