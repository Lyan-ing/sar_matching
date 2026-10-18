"""One-shot bounded postprocessing; never reruns matching or changes fusion settings."""
import os,sys,json,time,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
R=Path(__file__).resolve().parents[1];B=R.parent
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def main():
 started=time.monotonic();records=[];tag=stamp()
 while not all(list((R/'predictions'/('fusion_'+v)).glob('COMPLETED_*.json')) for v in ['v1','v21','v22']):
  if time.monotonic()-started>7200:raise RuntimeError('Bounded wait exceeded')
  if subprocess.run(['tmux','has-session','-t','sar_orb_sp_fusion'],capture_output=True).returncode:raise RuntimeError('Inference stopped before all completion markers')
  time.sleep(10)
 while subprocess.run(['tmux','has-session','-t','sar_orb_sp_fusion'],capture_output=True).returncode==0:
  assert time.monotonic()-started<7260;time.sleep(2)
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);gpu=None
 for candidate in [0,1]:
  uuid=subprocess.check_output(['nvidia-smi','-i',str(candidate),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
  if uuid not in apps:gpu=candidate;break
 records.append(dict(name='fresh_resource_check',exit_code=0 if gpu is not None else 1,selected_gpu=gpu,applications=apps))
 def run(name,args):
  log=R/f'logs/{name}_{stamp()}.log';command=[sys.executable,'-B',*map(str,args)]
  with log.open('x') as f:result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=B,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu if gpu is not None else 0)))
  records.append(dict(name=name,command=command,log=str(log),exit_code=result.returncode));print(name,result.returncode,flush=True);return result.returncode
 try:
  if gpu is not None:run('detector_swap',[R/'code/detector_swap.py','--gpu',gpu])
  data=json.loads((B/'SAR_SP_LG_V22_20261015_1525/main_v1_20261015_1525.json').read_text());row=json.loads(Path(data['manifest']).open().readline())
  if gpu is not None:run('standalone_cli',[R/'code/predict_pair.py',row['view0'],row['view1'],'--gpu',gpu,'--model','M2','--output',R/f'cli_fusion_{stamp()}'])
  assert run('reports',[R/'code/report_fusion.py'])==0
  for pattern in ['COMPLEMENTARITY_REPORT_*.md','FINAL_DECISION_*.md','FAILURE_CASES_*.md']:
   report=max(R.glob(pattern),key=lambda p:p.stat().st_mtime)
   if '![' in report.read_text():assert run('pdf_'+report.stem,[B/'tools/md_to_pdf.py',report])==0
 finally:
  with (R/f'POSTPROCESS_EXECUTION_{tag}.json').open('x') as f:json.dump(dict(at=stamp(),stages=records,complete=all(x['exit_code']==0 for x in records)),f,ensure_ascii=False,indent=2)
if __name__=='__main__':main()
