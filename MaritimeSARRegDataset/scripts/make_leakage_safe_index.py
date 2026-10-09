"""Explicit derived split protocol: co-locate parent groups linked by identical files.

Keep every image/annotation/mask and the official split. This is a new experiment
protocol, not an official benchmark split, and does not detect near duplicates.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def derive(rows, audit):
    def key(r):
        return ('hrsid' if r['dataset'] == 'background' else r['dataset'], r['source_group'])
    parent = {key(r): key(r) for r in rows}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    by_path = {r['image_path']: r for r in rows}
    for duplicate in audit['byte_identical_sources']:
        keys = [key(by_path[x['path']]) for x in duplicate]
        roots = sorted({find(k) for k in keys})
        for k in roots[1:]:
            parent[k] = roots[0]
    splits = defaultdict(set)
    for r in rows:
        splits[find(key(r))].add(r['split'])
    # Keep held-out membership whenever a component already has it.
    priority = {'test': 0, 'val': 1, 'train': 2}
    targets = {k: min(v, key=priority.get) for k, v in splits.items()}
    result, changes = [], []
    for r in rows:
        new = dict(r)
        root = find(key(r))
        new.update(original_experiment_split=r['split'], source_group_original=r['source_group'],
                   split=targets[root], source_group=root[1],
                   split_protocol='phase1_exact_duplicate_parent_components_v1')
        result.append(new)
        if new['split'] != r['split']:
            changes.append({'image_path': r['image_path'], 'from': r['split'], 'to': new['split']})
    return result, changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True); p.add_argument('--audit', required=True)
    p.add_argument('--out', required=True); p.add_argument('--report', required=True)
    a = p.parse_args()
    index = Path(a.index); rows = [json.loads(x) for x in index.read_text().splitlines()]
    audit = json.loads(Path(a.audit).read_text())
    if Path(audit['index']).resolve() != index.resolve():
        raise ValueError('Audit provenance does not match the input index')
    fatal = [x for x in audit['errors'] if x['type'] != 'identical_file_split_leakage']
    if fatal:
        raise ValueError(f'Cannot derive from invalid image/mask/source data: {fatal[:3]}')
    output, changes = derive(rows, audit)
    # Do not claim data re-reading: reuse the exhaustive audit, check only new splits.
    lookup = {r['image_path']: r for r in output}
    byte_cross = [v for v in audit['byte_identical_sources'] if len({lookup[x['path']]['split'] for x in v}) > 1]
    group_splits = defaultdict(set)
    for r in output:
        ds = 'hrsid' if r['dataset'] == 'background' else r['dataset']
        group_splits[(ds, r['source_group'])].add(r['split'])
    group_cross = [k for k, v in group_splits.items() if len(v) > 1]
    assert not byte_cross and not group_cross
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in output))
    report = {'protocol': 'phase1_exact_duplicate_parent_components_v1', 'index': str(out.resolve()),
              'input_index_sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
              'output_index_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
              'reused_image_mask_audit': str(Path(a.audit).resolve()),
              'image_records': len(output), 'changed_split_rows': len(changes), 'changes': changes,
              'cross_split_byte_duplicate_groups': len(byte_cross), 'cross_split_parent_groups': len(group_cross),
              'passed': True,
              'datasets': {ds: {'splits': dict(Counter(r['split'] for r in output if r['dataset'] == ds)),
                                'official_splits': dict(Counter(r['official_split'] or 'none' for r in output if r['dataset'] == ds))}
                           for ds in sorted({r['dataset'] for r in output})},
              'limitation': 'Only exact-byte duplicates and linked parent IDs; near-duplicate scenes remain unverified. Official benchmark splits are preserved only as metadata.'}
    Path(a.report).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'changes'}, indent=2))


if __name__ == '__main__':
    main()
