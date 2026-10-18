"""Read-only process/status monitor. Does not import torch or start validation."""
import json,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
root=Path(__file__).resolve().parents[1]
print(datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
for name in ['t1','t2','t3','repeat']:
 paths=list((root/'runs'/name).glob('status_*.json'))
 if not paths:print(name,'not started');continue
 p=max(paths,key=lambda x:x.stat().st_mtime);s=json.loads(p.read_text());pid=s.get('pid');alive='not visible here; verify host namespace' if s.get('state') in ['training','validation'] else False
 if pid and Path(f'/proc/{pid}/stat').exists():
  actual=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19];alive=str(actual)==str(s.get('process_start_ticks'))
 print(name,s.get('state'),'step',s.get('step'),'PID',pid,'alive',alive,'GPU',s.get('physical_gpu'),'updated',s.get('updated'),'best',s.get('best'),'qualified',s.get('qualified'))
workflow=root/'workflow_status_20261016_2234.json'
if workflow.exists():print('training workflow:',json.loads(workflow.read_text()).get('stage'))
postprocess=root/'postprocess_status_20261016_2234.json'
if postprocess.exists():
 s=json.loads(postprocess.read_text());print('postprocessing:',s.get('stage'),'updated',s.get('updated'))
completed=sorted(root.glob('TASK_COMPLETED_*.json'))
if completed:
 s=json.loads(completed[-1].read_text())
 print('overall: completed; do not restart training or benchmark')
 print('completed:',s.get('completed'),'record:',completed[-1])
 print('report:',s.get('report'));print('PDF:',s.get('pdf'))
