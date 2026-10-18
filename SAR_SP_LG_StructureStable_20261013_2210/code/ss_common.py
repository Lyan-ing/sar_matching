"""Reuse frozen helpers; timestamped append-only experiment records."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
M3ROOT=BASE/'SAR_SP_LG_M3_Detector_20261012_1932'
sys.path.insert(0,str(M3ROOT/'code'))
from m3_common import *
ROOT=Path(__file__).resolve().parents[1]
import csv
RUNSTAMP=ROOT.name.rsplit('_',2)[-2]+'_'+ROOT.name.rsplit('_',1)[-1]
def output(name,ext='json',folder=None):return (Path(folder) if folder else ROOT)/f'{name}_{RUNSTAMP}.{ext}'
def write_new(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(clean(v),f,ensure_ascii=False,indent=2,allow_nan=False)
def event(name,v,folder=None):
 p=(Path(folder) if folder else ROOT)/f'{name}_{stamp()}.json'
 if p.exists():
  for i in range(1,10000):
   q=p.with_name(f'{name}_v{i}_{stamp()}.json')
   if not q.exists():p=q;break
 write_new(p,dict(created=now(),**v));return p

def export_csv(p,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r));Path(p).parent.mkdir(parents=True,exist_ok=True)
 with Path(p).open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
