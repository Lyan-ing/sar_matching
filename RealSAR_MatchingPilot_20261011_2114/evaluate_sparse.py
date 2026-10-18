"""Later human-check-only evaluation of already frozen global predictions; no fitting."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'RealSAR_GeometryGT_20261011_0220/gt_quality'))
from quality import quantitative,quality
from core import save,sha
def main():
    p=argparse.ArgumentParser();p.add_argument('--human-export',required=True);p.add_argument('--out');a=p.parse_args()
    source=Path(a.human_export);receipt=json.loads((source/'EXPORT_COMPLETE.json').read_text())
    for rel,h in receipt['sha256'].items():
        if sha(source/rel)!=h:raise ValueError('human export hash mismatch')
    tasks={r['sample_id']:r for r in [json.loads(s) for s in (source/'tasks.jsonl').read_text().splitlines()]}
    out=Path(a.out) if a.out else ROOT/'human_evaluation'/datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    out.mkdir(parents=True,exist_ok=False);results=[]
    for directory in ['route_A','route_B','baselines']:
        for f in sorted((ROOT/directory).glob('*/record.json')):
            r=json.loads(f.read_text());sid=r['sample_id']
            doc=json.loads((source/'documents'/f'{sid}.json').read_text());q=quality(doc,tasks[sid])
            item=dict(sample_id=sid,method=r['method'],protocol=r['protocol'],frozen_prediction_sha256=sha(f),human_status=q['status'],prediction_refit=False)
            if q['status']=='gt_qualified':item['sparse_evaluation']=quantitative(r['candidate_H'],r['accepted'],doc,tasks[sid])
            else:item.update(sparse_evaluation=None,reason='no qualified independent check evidence')
            results.append(item)
    save(out/'results.json',results)
    save(out/'provenance.json',dict(human_export_sha256=sha(source/'EXPORT_COMPLETE.json'),at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        no_prediction_refit=True,local_models_not_scored_as_full_image=True))
    print('Evaluated frozen global predictions only; local supports require a separately specified coverage-aware evaluator')
if __name__=='__main__':main()
