"""Reproduce fixed pilot in a NEW sibling directory using the current match Python."""
import argparse,json,subprocess,sys,shutil,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    if Path(sys.executable).parent.parent.name!='match':raise RuntimeError('Use conda run -n match python')
    out=Path(a.out).resolve()
    if out.parent!=ROOT.parent:raise ValueError('New directory must be a sibling of this pilot')
    out.mkdir(exist_ok=False)
    for name in ['core.py','run.py','guarded_run.py','test_core.py','evaluate_sparse.py','analyze_visualize.py']:
        shutil.copy2(ROOT/name,out/name)
    # Verified official artifacts and local dependencies are reused, never written.
    for name in ['vendor','weights']:(out/name).symlink_to(ROOT/name,target_is_directory=True)
    commands=[[sys.executable,str(out/'run.py'),'prepare'],
        [sys.executable,'-m','pytest','-q',str(out/'test_core.py')],
        *[[sys.executable,str(out/'guarded_run.py'),s] for s in ['A','B','baseline']],
        [sys.executable,str(out/'analyze_visualize.py')]]
    journal=[]
    for i,cmd in enumerate(commands):
        name='prepare' if i==0 else ('tests' if i==1 else (['A','B','baseline'][i-2] if i<5 else 'analysis'))
        log=out/f'{name}_{stamp()}.log'
        env=dict(os.environ,MPLCONFIGDIR='/tmp/sar_pilot_mplconfig')
        with log.open('x') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=ROOT.parent)
        completed=out/f'{name}_{stamp()}.log'
        if completed!=log:log.rename(completed)
        journal.append(dict(command=cmd,exit_status=result.returncode,log=str(completed)))
        if result.returncode:break
        if i==0:
            original=json.loads((ROOT/'preregistration.json').read_text());fresh=json.loads((out/'preregistration.json').read_text())
            for key in ['config_sha256','samples_sha256']:assert original[key]==fresh[key],key
    (out/f'execution_{stamp()}.json').write_text(json.dumps(journal,indent=2))
    if journal[-1]['exit_status']:raise SystemExit(journal[-1]['exit_status'])
    print('Fresh reproduction complete:',out)
if __name__=='__main__':main()
