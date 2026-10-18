"""Finish statistics and PDF once, only after the one authorized run completes."""
from pathlib import Path
import subprocess,time,os,json,hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;TAG='20261017_1343'
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def put(p,o):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(o,ensure_ascii=False,indent=2));os.replace(tmp,p)
def main():
 state=ROOT/f'POSTPROCESS_STATUS_{TAG}.json';deadline=time.monotonic()+13*3600
 while time.monotonic()<deadline:
  p=ROOT/f'runs/formal/status_{TAG}.json';q=ROOT/f'QUEUE_STATUS_{TAG}.json'
  s=json.loads(p.read_text()) if p.exists() else {};queue=json.loads(q.read_text()) if q.exists() else {}
  if s.get('state')=='completed' and s.get('step')==6000:break
  if s.get('state') in ['failed','stopped'] or queue.get('state') in ['resource_wait_expired','stopped_without_launch','training_failed_no_restart']:
   put(state,dict(state='blocked_no_rerun',training=s,queue=queue,updated=now()));return
  put(state,dict(state='waiting_for_one_training_completion',updated=now()));time.sleep(30)
 else:put(state,dict(state='timed_out_without_new_training',updated=now()));return
 if list(ROOT.glob('TASK_COMPLETED_*.json')):return
 commands=[['conda','run','--no-capture-output','-n','match','python','-B',str(ROOT/'code/finalize_ablation.py')],['conda','run','--no-capture-output','-n','match','python','-B',str(ROOT/'code/ablation_report.py')]]
 for i,cmd in enumerate(commands):
  ts=stamp();put(state,dict(state='postprocessing',command=cmd,updated=now()))
  with (ROOT/f'logs/postprocess_{i}_{ts}.log').open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
  put(ROOT/f'command_exit_{i}_{ts}.json',dict(command=cmd,exit_code=r.returncode,completed=now()))
  if r.returncode:put(state,dict(state='failed_no_rerun',command=cmd,exit_code=r.returncode,updated=now()));return
 report=sorted(ROOT.glob('KEEP_CONTROLLED_ABLATION_REPORT_*.md'))[-1];ts=stamp()
 # Move only the new report to archival storage, retain the same logical path.
 physical=ROOT.joinpath('runs').resolve().parent/report.name;physical.write_bytes(report.read_bytes());report.unlink();report.symlink_to(physical)
 cmd=['conda','run','--no-capture-output','-n','match','python','-B',str(BASE/'tools/md_to_pdf.py'),str(report)]
 with (ROOT/f'logs/pdf_conversion_{ts}.log').open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 put(ROOT/f'command_pdf_exit_{ts}.json',dict(command=cmd,exit_code=r.returncode,completed=now()))
 if r.returncode:put(state,dict(state='pdf_failed_no_rerun',updated=now()));return
 pdf=physical.with_suffix('.pdf');logical=report.with_suffix('.pdf');logical.symlink_to(pdf);selection=sorted(ROOT.glob('SELECTED_CONTROL_*.json'))[-1];model=json.loads(selection.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 put(ROOT/f'TASK_COMPLETED_{stamp()}.json',dict(completed=now(),only_one_new_long_training=True,steps=6000,selected_control=model,report=str(report),pdf=str(logical),report_sha=sha(report),pdf_sha=sha(logical),training_stopped=True,no_additional_training=True))
 put(state,dict(state='completed',report=str(report),pdf=str(logical),updated=now()))
if __name__=='__main__':main()
