"""Read-only inputs, new artifacts, explicit evidence types."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
SOURCE=BASE/'real_sar_pilot/multisource512_20261010_2352'
MANUAL=BASE/'RealSAR_GeometryGT_20261011_0220'
PROJECT=BASE/'MaritimeSARRegistrationPhase2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:f.write(json.dumps(clean(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def clean(obj):
    if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [clean(v) for v in obj]
    if isinstance(obj,np.ndarray):return clean(obj.tolist())
    if isinstance(obj,np.generic):return clean(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj):return None
    return obj
def source_rows():
    rows=[json.loads(s) for s in (SOURCE/'dataset_index.jsonl').read_text().splitlines()]
    if len(rows)!=191:raise ValueError('wrong release index')
    for r in rows:
        r['sample_id']=r['pair_id']+'__'+r['chip_id']
        for k in ['reference_native_png','moving_native_png','reference_native_geotiff','moving_native_geotiff']:
            if not Path(r[k]).is_file():raise FileNotFoundError(r[k])
    return rows
def configure(out):
    out=Path(out).resolve();cfg=json.loads((out/'config.json').read_text())
    prereg=out/'preregistration.json'
    if prereg.exists():
        p=json.loads(prereg.read_text());assert p['config_sha256']==sha(out/'config.json');assert p['source_index_sha256']==sha(SOURCE/'dataset_index.jsonl')
    return cfg
