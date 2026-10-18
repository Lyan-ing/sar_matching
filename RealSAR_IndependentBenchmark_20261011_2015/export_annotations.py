"""Export actual human evidence, including ineligible points, without assigning GT."""
import sys
sys.dont_write_bytecode = True
import argparse, csv, hashlib, json, shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT = Path(__file__).resolve().parent
MANUAL = ROOT.parent / 'RealSAR_GeometryGT_20261011_0220'
sys.path.insert(0, str(MANUAL / 'annotation_app'))
from benchmark_store import Store
from benchmark_quality import POLICY

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def save(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def write_csv(p, fields, rows):
    with p.open('x', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

def export(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out == root or root in out.parents and 'annotations' in out.relative_to(root).parts:
        raise ValueError('Export destination must not be the live annotation store')
    store = Store(root)
    def live_hashes():
        return {str(p.relative_to(root)): sha(p) for p in sorted(store.directory.rglob('*')) if p.is_file()}
    before = live_hashes()
    task_before = {str(p.relative_to(root)): sha(p) for p in store.task_sources}
    docs = {sid: store.load(sid) for sid in store.tasks}
    out.mkdir(parents=True, exist_ok=False)
    rows, points = [], []
    (out / 'documents').mkdir()
    for sid, d in docs.items():
        t, q = store.tasks[sid], d['gt_quality']
        rows.append(dict(sample_id=sid, pair_id=t['pair_id'], component=t['component'], split=t['split'],
                         sensor=t['sensor'], look_pair_category=t['look_pair_category'], revision=d['revision'],
                         content_sha256=d.get('content_sha256'), status=q['status'],
                         formal_errors_allowed=q['formal_errors_allowed'], point_count=len(d['points']),
                         eligible_checks=q['eligible_checks'], eligible_controls=q['eligible_controls'],
                         scope=q['scope'], supported_thresholds_px=json.dumps(q['supported_success_thresholds_px']),
                         disposition=d['disposition'], disposition_note=d['disposition_note']))
        save(out / 'documents' / f'{sid}.json', d)
        evidence = {e['point_id']: e for e in q['point_evidence']}
        for p in d['points']:
            r, e = p.get('review') or {}, evidence[p['point_id']]
            points.append(dict(sample_id=sid, revision=d['revision'], point_id=p['point_id'], role=p['role'],
                x0=p['xy0'][0], y0=p['xy0'][1], x1=p['xy1'][0], y1=p['xy1'][1],
                landmark_type=p['landmark_type'], is_static=p['is_static'], visible_both=p['visible_both'],
                confidence=p['confidence'], sigma0_px=p['sigma0_px'], sigma1_px=p['sigma1_px'],
                annotator=p['annotator'], annotated_at=p['annotated_at'], used_assistance=p['used_assistance'],
                assistance_sources=json.dumps(p.get('assistance_sources', [])), note=p.get('note'),
                reviewer=r.get('reviewer'), reviewed_at=r.get('reviewed_at'), conclusion=r.get('conclusion'),
                review_xy0=json.dumps(r.get('xy0')), review_xy1=json.dumps(r.get('xy1')),
                review_sigma0_px=r.get('sigma0_px'), review_sigma1_px=r.get('sigma1_px'),
                review_static_confirmed=r.get('static_confirmed'), review_visible_confirmed=r.get('visible_confirmed'),
                reviewer_used_assistance=r.get('used_assistance'), review_note=r.get('note'),
                eligible=e['eligible'], ineligible_reasons=json.dumps(e['reasons']),
                combined_uncertainty_px=e['combined_uncertainty_px'], review_disagreement_px=json.dumps(e['review_disagreement_px'])))
    write_csv(out / 'index.csv', list(rows[0]), rows)
    with (out / 'index.jsonl').open('x') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    fields = ['sample_id','revision','point_id','role','x0','y0','x1','y1','landmark_type','is_static','visible_both',
              'confidence','sigma0_px','sigma1_px','annotator','annotated_at','used_assistance','assistance_sources','note',
              'reviewer','reviewed_at','conclusion','review_xy0','review_xy1','review_sigma0_px','review_sigma1_px',
              'review_static_confirmed','review_visible_confirmed','reviewer_used_assistance','review_note','eligible',
              'ineligible_reasons','combined_uncertainty_px','review_disagreement_px']
    write_csv(out / 'points.csv', fields, points)
    save(out / 'policy.json', POLICY)
    with (out / 'tasks.jsonl').open('x') as f:
        for t in store.tasks.values():f.write(json.dumps(t, ensure_ascii=False) + '\n')
    for rel in task_before:
        dst=out/'task_sources'/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(root/rel,dst)
    for rel in before:
        src, dst = root / rel, out / 'history' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    # Fail explicitly if a live server changed any evidence during the snapshot.
    # A failed partial directory is preserved for diagnosis and never called a valid export.
    if (before != live_hashes() or any(store.load(sid) != d for sid, d in docs.items())
        or task_before != {str(p.relative_to(root)):sha(p) for p in Store(root).task_sources}):
        raise RuntimeError('Live annotations changed: stop editing and export to a new directory')
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    summary = dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), samples=len(rows),
                   points=len(points), gt_qualified=sum(r['status']=='gt_qualified' for r in rows),
                   human_GT_created_by_export=False, snapshot_consistent=True,
                   native_coordinates=True, dense_GT=False, task_sha256=sha(out/'tasks.jsonl'),
                   assistance_events_present=store.audit.exists(), sha256=files)
    summary['task_sources_sha256']=task_before
    save(out / 'EXPORT_COMPLETE.json', summary)
    return summary

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', help='New directory; existing directories are refused')
    a = p.parse_args()
    out = Path(a.out) if a.out else ROOT / 'exports' / ('annotations_' + datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M'))
    result = export(ROOT / 'annotation_adapter', out)
    print(json.dumps({k:v for k,v in result.items() if k!='sha256'}, ensure_ascii=False, indent=2))
