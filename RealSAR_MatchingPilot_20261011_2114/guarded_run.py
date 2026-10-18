"""Execute frozen predictors with a Python file-access guard against human GT."""
import sys,runpy,json
from pathlib import Path
from collections import Counter
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
stage=sys.argv[1]
if stage not in ['A','B','baseline']:raise ValueError('predictor stages only')
opened=Counter();blocked=[]
def guard(event,args):
    if event!='open' or not args or not isinstance(args[0],(str,bytes)):return
    name=str(args[0])
    if 'RealSAR_GeometryGT' in name or 'RealSAR_IndependentBenchmark_' in name:
        blocked.append(name)
        raise PermissionError('Human annotation/preparation directory forbidden to predictors')
    if str(ROOT.parent) in name:opened[name]+=1
sys.addaudithook(guard)
try:runpy.run_path(str(ROOT/'run.py'),run_name='__main__')
finally:
    with (ROOT/'logs'/('file_access_'+stage+'.json')).open('x') as f:
        json.dump(dict(stage=stage,blocked=blocked,workspace_Python_open_counts=opened,
            scope='Python audit open events; native GDAL accesses additionally constrained by inspected load_pair'),f,indent=2)
