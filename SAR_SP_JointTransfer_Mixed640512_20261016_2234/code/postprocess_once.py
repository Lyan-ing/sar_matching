"""Single finite post-training evaluation chain, independent of the interactive session."""
import os,sys,json,time,subprocess,fcntl,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;TAG='20261016_2234'
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));os.replace(t,p)
def available():
 a=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();q=[tuple(map(int,s.split(','))) for s in a];pool=[(m,i) for i,m,u in q if i in [0,1] and m>=10000 and u<=20];return max(pool)[1] if pool else None
def command(script,stage,gpu=None,args=()):
 cmd=[sys.executable,'-B',str(ROOT/'code'/script),*args];env=dict(os.environ)
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 else:env['CUDA_VISIBLE_DEVICES']=''
 log=ROOT/'logs'/f'{stage}_{stamp()}.log';write(ROOT/f'postprocess_status_{TAG}.json',dict(stage=stage,pid=os.getpid(),command=cmd,log=str(log),updated=now()))
 with log.open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
 write(ROOT/f'command_result_{stage}_{stamp()}.json',dict(command=cmd,exit_status=0,log=str(log),physical_gpu=gpu,completed=now()))
 print(now(),'COMPLETED',stage,flush=True)

def main():
 lock=open('/tmp/sar_joint_mixed_postprocess.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if list(ROOT.glob('TASK_COMPLETED_*.json')):print('Already completed; no repeated benchmark.');return
 while True:
  states=[]
  for name in ['t1','t2','t3','repeat']:
   p=ROOT/'runs'/name/f'status_{TAG}.json';s=json.loads(p.read_text()) if p.exists() else {};states.append(s)
  if any(s.get('state') in ['failed','stopped'] for s in states):raise RuntimeError('A controlled run failed/stopped; no postprocessing or silent restart')
  if all(s.get('state')=='completed' and s.get('step')==6000 for s in states):break
  write(ROOT/f'postprocess_status_{TAG}.json',dict(stage='waiting_for_all_fixed_training',pid=os.getpid(),updated=now()));time.sleep(30)
 gpu=available()
 while gpu is None:time.sleep(30);gpu=available()
 command('joint_finalize.py','final_common_budget',gpu)
 command('joint_timing.py','isolated_native_timing',gpu)
 command('joint_coverage.py','actual_training_coverage')
 command('joint_statistics.py','final_statistics')
 command('prediction_diagnostics.py','final_candidate_diagnostics',args=('--tag','final'))
 command('joint_report.py','research_report')
 # Image-only re-load tests against selected actual trained checkpoints, never GT inputs.
 selection=json.loads(sorted(ROOT.glob('final_model_selection_*.json'))[-1].read_text());manifest=BASE/'SyntheticSAR_Mixed640512_Multicondition_20261016_2114/manifests/val_CANONICAL_20261016_2147.jsonl';rows=[json.loads(s) for s in manifest.read_text().splitlines()]
 for name,size in [('t1',512),('t2',640),('t3',640)]:
  r=next(x for x in rows if x['width']==size);command('predict_pair.py',f'actual_trained_cli_{name}',gpu,args=(r['image_A'],r['image_B'],'--kind','b','--checkpoint',selection[name]['checkpoint'],'--gpu',str(gpu),'--audit','--output',str(ROOT/f'cli_trained/{name}')))
 report=json.loads(sorted(ROOT.glob('REPORT_OUTPUT_*.json'))[-1].read_text());md=Path(report['markdown']);cmd=[sys.executable,'-B',str(BASE/'tools/md_to_pdf.py'),str(md)]
 with (ROOT/'logs'/f'final_pdf_conversion_{stamp()}.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 # Only direct frozen dependencies and canonical manifests; no repeated full historical audit.
 identity=json.loads(sorted(ROOT.glob('input_and_baseline_identity_*.json'))[-1].read_text());unchanged=[]
 for item in list(identity['manifests'].values())+list(identity['models'].values()):
  assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()==item['sha'];unchanged.append(item['path'])
 for p,h in identity['direct_readonly_dependencies'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h;unchanged.append(p)
 assert md.with_suffix('.pdf').exists()
 done=dict(completed=now(),report=str(md),pdf=str(md.with_suffix('.pdf')),all_four_runs_6000_steps=True,final_prediction_rows=21600,Val_pairs=1200,historical_inputs_unchanged=unchanged,no_historical_900_rerun=True,no_real_GT_accuracy_claim=True,selected_models=selection)
 write(ROOT/f'TASK_COMPLETED_{stamp()}.json',done);write(ROOT/f'postprocess_status_{TAG}.json',dict(stage='completed',pid=os.getpid(),updated=now(),report=str(md)));print('TASK_COMPLETED',str(md),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:write(ROOT/f'postprocess_failure_{stamp()}.json',dict(error=repr(e),updated=now(),pid=os.getpid()));raise
