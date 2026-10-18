#!/usr/bin/env python3
"""Audit proposed real-SAR candidates without inferring registration accuracy."""
import argparse,collections,json
from pathlib import Path

def audit(path):
    root=Path(path)
    summary=json.loads((root/'SUMMARY.json').read_text())
    rows=[json.loads(line) for line in (root/'candidates.jsonl').read_text().splitlines() if line.strip()]
    bad=[]
    for r in rows:
        d=root/r['split']/r['pair_id']/r['chip_id']
        for name in ('reference_float.tif','moving_float.tif','moving_native_float.tif','reference.png','moving.png','moving_native.png','common_valid.png','moving_native_valid.png','metadata.json'):
            if not (d/name).is_file():bad.append(str(d/name))
        if r['GT_STATUS']!='COARSE_GEO_ONLY__INDEPENDENT_GCP_REQUIRED':
            bad.append('False independent GT claim on '+r['pair_id'])
        if r['reference_acquisition_id']==r['moving_acquisition_id']:
            bad.append('Same acquisition '+r['pair_id'])
        if r['size_px']!=[summary['size_px']]*2:bad.append('Size discrepancy '+r['pair_id'])
    report={'candidate_chip_pairs':len(rows), 'sensors':dict(collections.Counter(r['sensor'] for r in rows)),
      'sites':len(set(r['site_id'] for r in rows)),
      'splits':dict(collections.Counter(r['split'] for r in rows)),
      'source_acquisition_pairs':len(set(r['pair_id'] for r in rows)),
      'min_valid_fraction':min((r['common_valid_fraction'] for r in rows),default=None),
      'failed_source_pairs':len(summary['failures']), 'missing_files_and_claim_violations':bad,
      'success':not bad and len(rows)==summary['candidate_chip_pairs'],
      'note':'Candidate QC only. GT and sensor/track look angle require separate verification.'}
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    result=audit(a.data)
    dst=Path(a.out)
    if dst.exists():raise FileExistsError(dst)
    dst.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result['success']:raise SystemExit(2)
if __name__=='__main__':main()
