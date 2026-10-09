import json
from pathlib import Path
from sarreg.toy import make_toy
from sarreg.indexer import prepare_index
from sarreg.synthesis import generate
from sarreg.qa import validate_generated
from sarreg.io import jsonl_read

def test_end_to_end(tmp_path):
    root=tmp_path/'toy'
    make_toy(root,12,12)
    ix=tmp_path/'index.jsonl'
    inv=prepare_index('hrsid',root,ix)
    assert inv['total_images']==12
    assert inv['ships']>12
    assert sum(inv['splits'].values())==12
    data=jsonl_read(ix)
    assert data and all(d['source_group'].startswith('DEMO') for d in data)
    out=tmp_path/'results'
    summary=generate(ix,'configs/easy.yaml',out,split='all',count=4,seed=12)
    assert summary['generated']==4, summary
    report=validate_generated(out)
    assert report['failed']==0, report
    meta=next(out.glob('*/pair_*/meta.json'))
    d=json.loads(meta.read_text())
    assert d['kind']=='synthetic_2d_registration_proxy_not_physical_multiview'
    assert len(d['pairs'])==1

def test_split_reproducibility(tmp_path):
    root=tmp_path/'toy';make_toy(root,20,1)
    a=tmp_path/'a.jsonl'; b=tmp_path/'b.jsonl'
    prepare_index('hrsid',root,a,seed=3)
    prepare_index('hrsid',root,b,seed=3)
    assert a.read_bytes()==b.read_bytes()
