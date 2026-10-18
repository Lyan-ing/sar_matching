"""Replay locked diagnostic to a NEW workspace sibling; reuse verified contexts."""
import sys
sys.dont_write_bytecode=True
import argparse,subprocess,shutil,json,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--fetch-context',action='store_true',help='optional bounded product re-read; default uses sealed actual1024 windows');a=p.parse_args();out=a.out.resolve()
    if out.parent!=ROOT.parent or out.exists():raise ValueError('output must be a new, nonexistent workspace sibling')
    out.mkdir()
    for f in [*ROOT.glob('*.py'),ROOT/'TASK.txt']:shutil.copyfile(f,out/f.name)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',SAR_STUDY_REPLAY='1')
    def run(*cmd):subprocess.run([sys.executable,*map(str,cmd)],env=env,check=True)
    for stage in ['prepare','data']:run(out/'run_logged.py',stage)
    if a.fetch_context:run(out/'run_logged.py','acquire_context')
    else:
        source=json.loads((ROOT/'contexts_resolved.json').read_text());receipt=json.loads((ROOT/'ENGINEERING_ACCEPTANCE.json').read_text())['file_sha256'];records=[]
        for row in source['records']:
            if not row['available']:raise ValueError('frozen context unavailable')
            old=Path(row['context_folder']);target=out/'contexts'/row['sample_id'];target.mkdir(parents=True)
            for f in old.iterdir():
                if f.is_file():
                    assert digest(f)==receipt[str(f.relative_to(ROOT))],'sealed context changed'
                    shutil.copyfile(f,target/f.name)
            records.append(json.loads((target/'metadata.json').read_text()))
        (out/'contexts/SUMMARY.json').write_text(json.dumps({'records':records,'available':len(records),'failed':0,'reused_verified_context_receipt':digest(ROOT/'ENGINEERING_ACCEPTANCE.json')}))
    run(out/'run_logged.py','resolve_context')
    for stage in ['radiometry','context','geometry','analyze','confirm','timing_validate','visualize','review_guide']:run(out/'run_logged.py',stage)
    run(out/'finalize.py')
if __name__=='__main__':main()
