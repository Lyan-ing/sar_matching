"""Export current evidence, never infer manual decisions from automatic matches."""
import argparse,sys,json,csv
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'annotation_app'))
from store import Store
from quality import POLICY
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);store=Store(ROOT);rows=[]
    for sid,t in store.tasks.items():
        d=store.load(sid);q=d['gt_quality'];rows.append({'sample_id':sid,'pair_id':t['pair_id'],'sensor':t['sensor'],'site_name':t['site_name'],'component':t['component'],'split':t['split'],'reference_acquisition_id':t['reference_acquisition_id'],'moving_acquisition_id':t['moving_acquisition_id'],'annotation_readiness':t['annotation_readiness'],'status':q['status'],'revision':d['revision'],'content_sha256':d.get('content_sha256'),'point_count':len(d['points']),'eligible_checks':q['eligible_checks'],'eligible_controls':q['eligible_controls'],'scope':q['scope'],'uncertainty_max_px':q['max_combined_uncertainty_px'],'supported_thresholds_px':q['supported_success_thresholds_px'],'disposition':d['disposition'],'disposition_note':d['disposition_note']})
    (out/'index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));
    with (out/'index.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'policy.json').write_text(json.dumps(POLICY,ensure_ascii=False,indent=2));print(json.dumps({'samples':len(rows),'gt_qualified':sum(r['status']=='gt_qualified' for r in rows),'points':sum(r['point_count'] for r in rows),'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()},indent=2))
if __name__=='__main__':main()
