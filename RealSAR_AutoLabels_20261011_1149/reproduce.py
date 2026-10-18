"""Replay the fixed pipeline into a NEW sibling directory; refuse overwrite."""
import sys
sys.dont_write_bytecode=True
import argparse,shutil,subprocess
from pathlib import Path
from common import ROOT,BASE

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve()
    if out.parent!=BASE:raise ValueError('Replay output must be a new directory immediately under the project workspace')
    out.mkdir(exist_ok=False)
    for d in ['georeference_correspondences','pseudo_labels','predictions','quality','visualization','annotation_adapter','tests','reports','code_snapshot']:(out/d).mkdir()
    for file in [*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'TASK.txt',*ROOT.glob('tests/*.py'),ROOT/'annotation_adapter/serve.py']:
        relative=file.relative_to(ROOT);(out/relative).parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,out/relative)
    for script in ['generate_geo.py','infer.py','labels.py','prepare_review.py','analyze.py']:
        command=[sys.executable,'-u',str(out/script),'--out',str(out)]
        print('RUN',command,flush=True);subprocess.run(command,check=True)
    print('Replay generated. Run focused tests, browser adapter checks and finalize.py before accepting the new release.',flush=True)
