import os,sys,json,hashlib,time,tempfile
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'MaritimeSARRegistrationPhase2'))
os.environ['TORCH_HOME']=str(BASE/'MaritimeSARRegistrationPhase2/cache/torch')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'mpl_cache'))
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def atomic_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 with tmp.open('w') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def readrows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def write_new(p,x):
 with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def proc_start(pid):
 try:return Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
 except OSError:return None
def ensure_gpu(physical):
 import torch,subprocess
 if physical not in [0,1] or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(physical):raise RuntimeError('Only explicit physical GPU0/1 allowed')
 if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('One permitted CUDA device must be visible; CPU fallback forbidden for training')
 expected=subprocess.check_output(['nvidia-smi','-i',str(physical),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 actual='GPU-'+str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
 if actual!=expected:raise RuntimeError(f'Physical GPU mismatch: {actual} != {expected}')
 torch.cuda.set_per_process_memory_fraction(.22,0)
 return dict(physical_gpu=physical,logical_device='cuda:0',uuid=actual,name=torch.cuda.get_device_name(0),visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],per_process_allocator_fraction=.22)
