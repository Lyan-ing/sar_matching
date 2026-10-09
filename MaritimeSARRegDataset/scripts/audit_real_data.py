"""Read-only exhaustive audit of prepared real-data indexes; no toy imagery."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def decode(path):
    im = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise ValueError(f'Cannot decode {path}')
    return im


def audit(index):
    rows = [json.loads(s) for s in Path(index).read_text().splitlines() if s.strip()]
    summaries = {}
    errors, warnings, details = [], [], []
    documents = {p: json.loads(Path(p).read_text())
                 for p in {r['annotation_source'] for r in rows if r['annotation_source']}}
    # Avoid re-reading source JSON for each image.
    records = {p: {Path(im['file_name']).name: im for im in d['images']}
               for p, d in documents.items()}
    dimensions = defaultdict(Counter)
    groups, paths, hashes = defaultdict(set), Counter(), defaultdict(list)
    for r in rows:
        ds, p = r['dataset'], r['image_path']
        if ds not in summaries:
            summaries[ds] = {'images': 0, 'ships': 0, 'decoded': 0, 'sea_masks': 0,
                             'splits': Counter(), 'official_splits': Counter(),
                             'scene_types': Counter(), 'mask_values': Counter(),
                             'segmentation_types': Counter()}
        s = summaries[ds]
        s['images'] += 1; s['ships'] += len(r['annotations'])
        s['splits'][r['split']] += 1
        s['official_splits'][r['official_split'] or 'none'] += 1
        s['scene_types'][r['scene_type']] += 1
        # HRSID/background share Pxxxx parent scenes, including across sources.
        namespace = 'hrsid' if ds == 'background' else ds
        groups[(namespace, r['source_group'])].add(r['split'])
        paths[p] += 1
        detail = dict(dataset=ds, source_id=r['source_id'], source_group=r['source_group'],
                      split=r['split'], official_split=r['official_split'],
                      scene_type=r['scene_type'], ships=len(r['annotations']), image_path=p,
                      width='', height='', mask_values='', sea_fraction='')
        try:
            im = decode(p); h, w = im.shape[:2]
            s['decoded'] += 1; dimensions[ds][(h, w)] += 1
            detail.update(width=w, height=h)
            hashes[hashlib.sha256(Path(p).read_bytes()).hexdigest()].append(
                {'dataset': ds, 'path': p, 'split': r['split']})
            if r['annotation_source']:
                raw = records[r['annotation_source']][Path(p).name]
                if (raw['height'], raw['width']) != (h, w):
                    errors.append({'type': 'annotation_shape_mismatch', 'path': p,
                                   'annotation_hw': [raw['height'], raw['width']], 'decoded_hw': [h, w]})
            if ds == 'ssdd' and Path(p).parent.name in ('train', 'test'):
                if Path(p).parent.name != r['official_split']:
                    errors.append({'type': 'image_split_mismatch', 'path': p})
            for a in r['annotations']:
                box = np.asarray(a['bbox_xywh'])
                if not np.isfinite(box).all() or np.any(box[2:] <= 0):
                    errors.append({'type': 'invalid_bbox', 'path': p, 'id': a['id']})
                elif box[0] < 0 or box[1] < 0 or box[0]+box[2] > w+1e-6 or box[1]+box[3] > h+1e-6:
                    warnings.append({'type': 'source_bbox_outside_image', 'path': p,
                                     'id': a['id'], 'bbox': box.tolist()})
                seg = a.get('segmentation')
                kind = 'rle' if isinstance(seg, dict) else 'polygon' if seg else 'bbox_proxy'
                s['segmentation_types'][kind] += 1
                if kind == 'polygon':
                    if any(len(x) < 6 or len(x) % 2 or not np.isfinite(x).all() for x in seg):
                        errors.append({'type': 'invalid_polygon', 'path': p, 'id': a['id']})
            mp = r.get('sea_mask_path')
            if ds == 'ssdd' and not mp:
                errors.append({'type': 'missing_sea_mask', 'path': p})
            if mp:
                m = decode(mp); values = np.unique(m).tolist()
                s['sea_masks'] += 1; s['mask_values'][str(values)] += 1
                detail['mask_values'] = str(values)
                detail['sea_fraction'] = float(np.mean(m == 255))
                if m.shape != (h, w) or m.dtype != np.uint8:
                    errors.append({'type': 'sea_mask_shape_dtype', 'path': mp,
                                   'mask_shape': list(m.shape), 'image_hw': [h, w], 'dtype': str(m.dtype)})
                if not set(values).issubset({0, 255}):
                    errors.append({'type': 'sea_mask_values', 'path': mp, 'values': values})
                if Path(mp).stem != Path(p).stem or Path(mp).parent.name != r['official_split']:
                    errors.append({'type': 'sea_mask_wrong_pairing', 'path': mp, 'image': p})
        except (OSError, ValueError, KeyError, cv2.error) as exc:
            errors.append({'type': 'read_failure', 'path': p, 'reason': str(exc)})
        details.append(detail)
    for ds, s in summaries.items():
        dims = dimensions[ds]
        s['dimensions_hw'] = {f'{h}x{w}': count for (h, w), count in sorted(dims.items())}
        s['distinct_groups'] = sum(k[0] == ds for k in groups) if ds != 'background' else len({r['source_group'] for r in rows if r['dataset'] == ds})
    cross = {f'{ds}:{g}': sorted(v) for (ds, g), v in groups.items() if len(v) > 1}
    duplicate_paths = {p: n for p, n in paths.items() if n > 1}
    duplicate_bytes = [v for v in hashes.values() if len(v) > 1]
    if cross: errors.append({'type': 'parent_scene_leakage', 'groups': cross})
    if duplicate_paths: errors.append({'type': 'duplicate_source_paths', 'paths': duplicate_paths})
    cross_byte = [v for v in duplicate_bytes if len({x['split'] for x in v}) > 1]
    if cross_byte: errors.append({'type': 'identical_file_split_leakage', 'files': cross_byte})
    raw_stats = {}
    for p, d in documents.items():
        raw_stats[p] = {'images': len(d['images']), 'annotations': len(d['annotations']),
                        'categories': d.get('categories'),
                        'indexed_annotations': sum(len(r['annotations']) for r in rows if r['annotation_source'] == p)}
    discrepancy = {}
    hrsid_sources = [p for p in documents if Path(p).name in ('train2017.json', 'test2017.json')]
    if hrsid_sources:
        combined = Path(hrsid_sources[0]).with_name('train_test2017.json')
        if combined.exists():
            def per_image(doc):
                names = {im['id']: im['file_name'] for im in doc['images']}
                counts = defaultdict(Counter)
                for a in doc['annotations']:
                    signature = json.dumps({k: v for k, v in a.items() if k not in ('id', 'image_id')}, sort_keys=True)
                    counts[names[a['image_id']]][signature] += 1
                return counts
            joined = {}
            for p in hrsid_sources: joined.update(per_image(documents[p]))
            merged = json.loads(combined.read_text()); full = per_image(merged)
            different = []
            for name in sorted(joined.keys() | full.keys()):
                a, b = joined.get(name, Counter()), full.get(name, Counter())
                if a != b:
                    different.append({'file_name': name, 'official_instances': sum(a.values()),
                                      'merged_instances': sum(b.values()),
                                      'official_only': sum((a-b).values()), 'merged_only': sum((b-a).values())})
            discrepancy = {'merged_path': str(combined), 'official_instances': sum(sum(x.values()) for x in joined.values()),
                           'merged_instances': len(merged['annotations']), 'different_images': different,
                           'policy': 'Retain selected official train/test labels; never mix annotation releases.'}
    return {'index': str(Path(index).resolve()), 'total_images': len(rows), 'datasets': summaries,
            'source_annotation_files': raw_stats, 'hrsid_annotation_discrepancy': discrepancy,
            'cross_split_parent_groups': cross, 'duplicate_paths': duplicate_paths,
            'byte_identical_sources': duplicate_bytes, 'errors': errors, 'warnings': warnings,
            'passed': not errors, 'mask_encoding': {'land': 0, 'sea': 255}}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report, details = audit(args.index)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    with out.with_suffix('.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    print(json.dumps({'images': report['total_images'], 'passed': report['passed'],
                      'errors': len(report['errors']), 'warnings': len(report['warnings']),
                      'datasets': {k: {x: v[x] for x in ('images', 'decoded', 'ships', 'sea_masks', 'mask_values')}
                                   for k, v in report['datasets'].items()}, 'report': str(out)}, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
