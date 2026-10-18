"""One finite finalization after formal selection; never retrains or repeats old baselines."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import os,sys,json,time,subprocess,fcntl,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;TAG='20261017_0908'
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def write(p,x):
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2));os.replace(temp,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def free_gpu():
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    for line in lines:
        i,free,util=map(int,line.split(','))
        if i in [0,1] and free>=10000 and util<=20:
            pids=subprocess.check_output(['nvidia-smi','-i',str(i),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if not pids:return i
    return None
def run(script,stage,gpu=None,args=()):
    command=[sys.executable,'-B',str(ROOT/'code'/script),*args];env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu));log=ROOT/f'logs/{stage}_{stamp()}.log'
    write(ROOT/f'postprocess_status_{TAG}.json',dict(stage=stage,updated=now(),pid=os.getpid(),command=command,log=str(log)))
    with log.open('x') as f:subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(ROOT/f'command_result_{stage}_{stamp()}.json',dict(command=command,exit_status=0,log=str(log),physical_gpu=gpu,completed=now()))
    print(now(),'COMPLETED',stage,flush=True)


def main():
    lock=open('/tmp/sar_lowtexture_keep_postprocess.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if list(ROOT.glob('TASK_COMPLETED_*.json')):print('Already completed, no repeated benchmark');return
    deadline=time.monotonic()+7200
    while True:
        p=ROOT/f'runs/formal/status_{TAG}.json';state=json.loads(p.read_text()) if p.exists() else {}
        if state.get('state') in ['failed','stopped']:raise RuntimeError('Formal training failed/stopped; no automatic restart or model substitution')
        if state.get('state')=='completed':assert state['step']==6000;break
        if time.monotonic()>deadline:raise TimeoutError('Bounded two-hour wait reached')
        write(ROOT/f'postprocess_status_{TAG}.json',dict(stage='waiting_for_formal_completion',updated=now(),pid=os.getpid()));time.sleep(20)
    gpu=free_gpu()
    while gpu is None:
        if time.monotonic()>deadline:raise TimeoutError('No authorized free GPU for final diagnostics')
        time.sleep(20);gpu=free_gpu()
    run('keep_results.py','final_statistics_and_drift',gpu)
    run('keep_report.py','research_report')
    selected=json.loads(sorted(ROOT.glob('SELECTED_MODEL_*.json'))[-1].read_text());manifest=BASE/'SyntheticSAR_Mixed640512_Multicondition_20261016_2114/manifests/val_CANONICAL_20261016_2147.jsonl';rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    for size in [512,640]:
        row=next(r for r in rows if r['width']==size);run('predict_pair.py',f'pure_image_cli_{size}',gpu,args=(row['image_A'],row['image_B'],'--kind','b','--checkpoint',selected['checkpoint'],'--gpu',str(gpu),'--audit','--output',str(ROOT/f'predictions/cli_{size}')))
    output=json.loads(sorted(ROOT.glob('REPORT_OUTPUT_*.json'))[-1].read_text());md=Path(output['markdown']);archive=(ROOT/'figures').resolve().parent;physical_md=archive/md.name
    assert not physical_md.exists();shutil.move(str(md),str(physical_md));md.symlink_to(physical_md)
    command=[sys.executable,'-B',str(BASE/'tools/md_to_pdf.py'),str(md)];log=ROOT/f'logs/pdf_conversion_{stamp()}.log'
    with log.open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    assert physical_md.with_suffix('.pdf').exists();md.with_suffix('.pdf').symlink_to(physical_md.with_suffix('.pdf'))
    for folder in archive.glob('pdf_export*'):
        if not (ROOT/folder.name).exists():(ROOT/folder.name).symlink_to(folder,target_is_directory=True)
    config=json.loads((ROOT/f'FORMAL_PROTOCOL_{TAG}.json').read_text());assert sha(selected['checkpoint'])==selected['sha']
    for name,h in config['code_sha'].items():assert sha(ROOT/'code'/name)==h
    for x in config['data'].values():assert sha(x['path'])==x['sha']
    assert sha(config['initialization'])==config['initialization_sha']
    assert sha(BASE/'MaritimeSARRegistrationPhase2/cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth')==config['lightglue_sha']
    done=dict(completed=now(),formal_steps=6000,selected_model=selected,report=str(md),pdf=str(md.with_suffix('.pdf')),report_sha=sha(md),pdf_sha=sha(md.with_suffix('.pdf')),comparison_rows=9600,new_inference_rows=2400,old_cached_rows_reused=7200,qualified=output['qualified'],decision=output['decision'],no_ORB_rerun=True,no_historical_900_rerun=True,no_new_data_generated=True,no_real_accuracy_claim=True)
    write(ROOT/f'TASK_COMPLETED_{stamp()}.json',done);write(ROOT/f'postprocess_status_{TAG}.json',dict(stage='completed',updated=now(),pid=os.getpid(),report=str(md)));print(now(),'TASK_COMPLETED',str(md),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:write(ROOT/f'postprocess_failure_{stamp()}.json',dict(error=repr(e),updated=now()));raise
