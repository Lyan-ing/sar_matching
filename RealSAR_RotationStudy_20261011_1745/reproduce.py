"""Reproduce the fixed study in a NEW sibling folder, never overwrite outputs."""
import sys
sys.dont_write_bytecode=True
import argparse,shutil,subprocess
from pathlib import Path
SOURCE=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);a=p.parse_args();target=Path(a.out).resolve()
    if target.parent!=SOURCE.parent:raise ValueError('Output must be a new sibling in the existing project workspace')
    target.mkdir(exist_ok=False)
    for file in [*SOURCE.glob('*.py'),SOURCE/'TASK.txt']:shutil.copyfile(file,target/file.name)
    def run(*parts):subprocess.run([sys.executable,*parts],check=True)
    run(str(target/'prepare.py'))
    run('-m','pytest','-q','-p','no:cacheprovider',str(target/'test_study.py'),'-k','integer_pixels or 512_centers or selection or fixed_protocol')
    run(str(target/'run_logged.py'),'diagnostic8')
    run('-m','pytest','-q','-p','no:cacheprovider',str(target/'test_study.py'),'-k','not complete_experiment')
    run(str(target/'run_logged.py'),'representative24','full191','strategies','negative')
    for script in ['analyze.py','geometry_details.py','visualize.py','visualize_H.py','finalize.py']:run(str(target/script))
