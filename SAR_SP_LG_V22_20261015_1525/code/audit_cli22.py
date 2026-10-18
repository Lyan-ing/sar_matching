"""Python-level file-access audit around the standalone image-only CLI."""
import sys,runpy,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
args=sys.argv[1:];allowed={str(Path(p).resolve()) for p in args[:2]};target=Path(__file__).with_name('predict_frozen22.py');output=Path(args[args.index('--output')+1]);project=str(target.parents[2]);reads=[]
def audit(event,a):
 if event!='open' or not isinstance(a[0],(str,bytes)):return
 p=str(Path(a[0]).resolve());mode=str(a[1]) if len(a)>1 else ''
 if p in allowed:return
 if project in p:
  if any(s in p for s in ['SyntheticSAR_V22_','SyntheticSAR_V2_','DualSAR_DatasetPrep_','MaritimeSARRegDataset/data/','RealSAR_']):raise RuntimeError('Forbidden dataset/GT access during image-only prediction: '+p)
  if '.jsonl' in p:raise RuntimeError('Forbidden manifest access: '+p)
  reads.append(p)
sys.addaudithook(audit);sys.argv=[str(target),*args];runpy.run_path(str(target),run_name='__main__')
t=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');(output/f'python_file_access_audit_{t}.json').write_text(json.dumps(dict(passed=True,scope='Python open audit plus pixel-only CLI source; not a system-call tracer',allowed_images=sorted(allowed),project_paths=sorted(set(reads))),indent=2))
