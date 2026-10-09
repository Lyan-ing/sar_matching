"""Index extracted HRSID, Official-SSDD-OPEN, and unlabeled SAR background images.

Avoid duplicate SSDD representations, preserve official splits as metadata, and
prefer parent-scene-safe hash splitting for overlapping HRSID image crops.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from .io import image_paths, load_json, jsonl_write


def _hash01(s: str, seed: int) -> float:
    digest = hashlib.sha256((str(seed) + '|' + s).encode()).digest()
    return int.from_bytes(digest[:8], 'big') / 2**64


def _images(root: Path):
    d = defaultdict(list)
    for p in image_paths(root):
        d[p.name.lower()].append(p)
    return d


def _load_coco(p):
    try:
        d = load_json(p)
    except (ValueError, OSError, UnicodeError):
        return None
    if isinstance(d, dict) and isinstance(d.get('images'), list) and isinstance(d.get('annotations'), list):
        return d
    return None


def _coco_sources(root: Path, dataset: str):
    """Select ONE image annotation representation, never mix duplicate SSDD branches."""
    paths = sorted(root.rglob('*.json'))
    if dataset == 'hrsid':
        ann = [p for p in paths if p.parent.name.lower() == 'annotations']
        byname = {p.stem.lower(): p for p in ann}
        if 'train2017' in byname and 'test2017' in byname:
            wanted = [byname['train2017'], byname['test2017']]
        elif 'train_test2017' in byname:
            wanted = [byname['train_test2017']]
        else:
            wanted = [p for p in ann if p.stem.lower() in ('train','test')]
        return [(p, d) for p in wanted if (d := _load_coco(p)) is not None]
    if dataset == 'ssdd':
        # Official-SSDD-OPEN duplicates images/labels across BBox, RBox and PSeg
        # directories. Prefer the richest COCO representation once.
        options = []
        for p in paths:
            if p.parent.name.lower() != 'annotations' or p.stem.lower() not in ('train', 'test'):
                continue
            if 'coco_style' not in [part.lower() for part in p.parts]:
                continue
            parent = p.parent.parent
            norm = str(parent).lower()
            priority = 0 if 'bbox_rbox_pseg_ssdd' in norm else 1 if 'bbox_ssdd' in norm else 2
            options.append((priority, str(parent), p))
        if options:
            byparent = defaultdict(list)
            for priority, dirname, p in options:
                byparent[(priority, dirname)].append(p)
            # Prefer both train/test in the same canonical branch.
            choices = sorted(byparent, key=lambda k: (k[0], -len(byparent[k]), k[1]))
            files = sorted(byparent[choices[0]], key=lambda p: (p.stem.lower() != 'train', str(p)))
            return [(p, d) for p in files if (d := _load_coco(p)) is not None]
    return []


def _ship_ids(d):
    cats = d.get('categories', [])
    ids = {x.get('id') for x in cats if 'ship' in str(x.get('name', '')).lower() or 'vessel' in str(x.get('name', '')).lower()}
    if not ids and len(cats) == 1:
        ids = {cats[0].get('id')}
    return ids


def _pick_image(fname, lookup, annotation_file):
    options = lookup.get(Path(fname).name.lower(), [])
    if not options:
        return None
    # NEVER select a same-name image from another SSDD representation.
    data_root = annotation_file.parent.parent
    if data_root.name.lower() in ('coco_style', 'voc_style'):
        options = [p for p in options if data_root.parent in p.parents]
    if not options:
        return None
    # First select actual image in the SAME image-data variant as its annotations.
    def rank(p):
        same = data_root in p.parents
        is_mask = any('mask' in part.lower() or 'gt_mask' in part.lower() for part in p.parts)
        # In case images live outside coco_style, prefer the same SSDD branch.
        branch = data_root.parent in p.parents
        jpeg = p.parent.name.lower() == 'jpegimages'
        return (0 if same else 1 if branch else 2, bool(is_mask), 0 if jpeg else 1, len(p.parts), str(p))
    return min(options, key=rank).resolve()


def _scene_labels(dataset, root, selected_files):
    labels = {}
    if dataset == 'hrsid':
        candidates = [(root / 'inshore_offshore'), (root / 'HRSID_JPG' / 'inshore_offshore')]
        for dr in candidates:
            for label in ('inshore', 'offshore'):
                p = dr / f'{label}.json'
                if not p.exists():
                    continue
                data = _load_coco(p)
                if data:
                    for im in data['images']:
                        labels[Path(im['file_name']).name.lower()] = label
    elif dataset == 'ssdd' and selected_files:
        base = selected_files[0].parent
        for label in ('inshore', 'offshore'):
            for p in (base / f'test_{label}.json', base / f'train_{label}.json'):
                if p.exists() and (d := _load_coco(p)):
                    for im in d['images']:
                        labels[Path(im['file_name']).name.lower()] = label
    return labels


def _read_coco(root, lookup, dataset):
    sources = _coco_sources(root, dataset)
    if not sources:
        return {}
    labels = _scene_labels(dataset, root, [p for p, _ in sources])
    out = {}
    missing = []
    for source, coco in sources:
        ids = _ship_ids(coco)
        by_id = defaultdict(list)
        for ann in coco['annotations']:
            if ids and ann.get('category_id') not in ids:
                continue
            box = ann.get('bbox')
            if not box or len(box) != 4 or box[2] <= 0 or box[3] <= 0:
                continue
            by_id[ann['image_id']].append({
                'id': str(ann.get('id', len(by_id[ann['image_id']]))),
                'bbox_xywh': list(map(float, box)),
                'segmentation': ann.get('segmentation', []),
                'mask_type': 'segmentation_or_bbox',
            })
        base = source.stem.lower()
        official = 'test' if base in ('test', 'test2017') else 'train' if base in ('train', 'train2017') else None
        for im in coco['images']:
            fname = Path(im['file_name']).name.lower()
            p = _pick_image(fname, lookup, source)
            if p is None:
                missing.append((str(source), fname))
                continue
            if p in out:
                raise RuntimeError(f'Duplicate COCO image in selected annotation sources: {p}')
            out[p] = {
                'annotations': by_id[im['id']], 'official_split': official,
                'scene_type': labels.get(fname, 'unknown'), 'annotation_source': str(source),
            }
    if missing:
        sample = missing[:5]
        raise RuntimeError(f'{len(missing)} COCO images missing in selected dataset representation; examples: {sample}')
    return out


def _read_voc(root, lookup):
    """Fallback for SSDD distributions without COCO annotations.

    Limit to one canonical VOC variant to avoid many duplicate copies.
    """
    variants = sorted({p.parent.parent for p in root.rglob('*.xml') if p.parent.name.lower() == 'annotations'},
                      key=lambda p: (0 if 'BBox_RBox_PSeg_SSDD' in p.parts else 1 if 'BBox_SSDD' in p.parts else 2, str(p)))
    if not variants:
        return {}
    base = variants[0]
    ids = {}
    for sp in ('train', 'test'):
        p = base / 'ImageSets' / 'Main' / f'{sp}.txt'
        if p.exists():
            for line in p.read_text(encoding='utf8', errors='ignore').splitlines():
                if line.strip():
                    ids[line.strip().split()[0]] = sp
    out = {}
    for source in sorted((base / 'Annotations').glob('*.xml')):
        p = _pick_image(source.stem + '.jpg', lookup, source)
        if p is None:
            p = _pick_image(source.stem + '.png', lookup, source)
        if p is None:
            continue
        try:
            tree = ET.parse(source).getroot()
        except ET.ParseError:
            continue
        boxes = []
        for i, obj in enumerate(tree.findall('./object')):
            name = (obj.findtext('name') or '').lower()
            if 'ship' not in name and 'vessel' not in name:
                continue
            bb = obj.find('bndbox')
            if bb is None:
                continue
            try:
                x0, y0, x1, y1 = [float(bb.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            except (TypeError, ValueError):
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append({'id': f'{source.stem}_{i}', 'bbox_xywh': [x0, y0, x1-x0, y1-y0],
                          'segmentation': [], 'mask_type': 'bbox_proxy'})
        out[p] = {'annotations': boxes, 'official_split': ids.get(source.stem),
                  'scene_type': 'unknown', 'annotation_source': str(source)}
    return out


def _mask_lookup(root):
    """Index SL-SSDD sea/land PNG masks by (publisher split, image stem).

    For the official layout ``SL-SSDD/train/000002.png`` and
    ``SL-SSDD/test/000001.png`` are paired only with the respective SSDD
    COCO train/test annotations. Flat mask-only roots also remain supported.
    Never enumerate SSDD ship-instance masks as sea/land masks.
    """
    if root is None:
        return {}
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Expected SL-SSDD sea/land mask directory: {root}')
    paths = image_paths(root)
    if not paths:
        raise ValueError(f'No sea/land mask PNG files found: {root}')
    if any(p.parent.name.lower() in ('jpegimages', 'jpegimages_pseg_gt_mask') for p in paths):
        raise ValueError('sea-mask-root must point to SL-SSDD SEA-LAND mask-only directory, not SSDD imagery/ship masks')
    result = {}
    for p in paths:
        if p.suffix.lower() != '.png':
            raise ValueError(f'SL-SSDD sea/land mask root contains non-PNG image: {p}')
        relparts = [s.lower() for s in p.relative_to(root).parts[:-1]]
        found_splits = set(relparts).intersection({'train', 'test'})
        if len(found_splits) > 1:
            raise ValueError(f'Conflicting train/test folders for SL-SSDD mask: {p}')
        key = (next(iter(found_splits)) if found_splits else None, p.stem)
        if key in result:
            raise ValueError(f'Duplicate SL-SSDD mask for {key}: {result[key]} and {p}')
        result[key] = p.resolve()
    return result


def _find_sea_mask(masks, official_split, stem):
    """Never match a mask in a different publisher-defined split."""
    if not masks:
        return None
    p = masks.get((official_split, stem))
    if p is not None:
        return p
    # Compatibility with a flat directory of uniquely named masks.
    return masks.get((None, stem))


def prepare_index(dataset, root, out, sea_mask_root=None, seed=42, val_fraction=.12,
                  test_fraction=.18, split_policy='auto', allow_partial_sea_masks=False):
    dataset = dataset.lower()
    if dataset not in ('hrsid', 'ssdd', 'background', 'generic'):
        raise ValueError('dataset must be hrsid, ssdd, background, or generic')
    if split_policy not in ('auto', 'official', 'group'):
        raise ValueError('split_policy must be auto, official, or group')
    if not 0 <= val_fraction < 1 or not 0 <= test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError('Invalid validation/test fractions')
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Directory missing: {root}')
    lookup = _images(root)
    if not lookup:
        raise RuntimeError(f'No images in {root}; unzip the dataset first')
    if dataset in ('hrsid', 'ssdd'):
        rows = _read_coco(root, lookup, dataset)
        if not rows and dataset == 'ssdd':
            rows = _read_voc(root, lookup)
        if not rows:
            raise RuntimeError(f'No usable image+COCO/VOC annotation pairs for {dataset} under {root}')
    else:
        # Background and generic are intended to point at EXACT image-only subdirs.
        rows = {p.resolve(): {'annotations': [], 'official_split': None, 'scene_type': 'unknown',
                              'annotation_source': None} for ps in lookup.values() for p in ps}
    if sea_mask_root is not None and dataset != 'ssdd':
        raise ValueError('--sea-mask-root must be provided only for --dataset ssdd')
    masks = _mask_lookup(sea_mask_root)
    policy = ('group' if dataset in ('hrsid', 'background') else 'official') if split_policy == 'auto' else split_policy
    groups = defaultdict(list)
    for p, item in rows.items():
        group = p.stem.split('_')[0] if dataset in ('hrsid','background') else p.stem
        groups[group].append((p, item))
    output = []
    for group, entries in sorted(groups.items()):
        if policy == 'group':
            h = _hash01(group, seed)
            split = 'test' if h < test_fraction else 'val' if h < test_fraction + val_fraction else 'train'
        for p, item in sorted(entries, key=lambda x: str(x[0])):
            if policy == 'official':
                off = item.get('official_split')
                if off == 'test':
                    per_split = 'test'
                elif off == 'train':
                    per_split = 'val' if _hash01(group, seed) < val_fraction else 'train'
                else:
                    h = _hash01(group, seed)
                    per_split = 'test' if h < test_fraction else 'val' if h < test_fraction + val_fraction else 'train'
            else:
                per_split = split
            output.append({
                'dataset': dataset, 'source_id': p.stem, 'source_group': group,
                'image_path': str(p), 'split': per_split,
                'official_split': item.get('official_split'),
                'scene_type': item.get('scene_type', 'unknown'),
                'annotations': item['annotations'], 'annotation_source': item['annotation_source'],
                'sea_mask_path': str(mp) if (mp := _find_sea_mask(masks, item.get('official_split'), p.stem)) else None,
            })
    # Validate the full SSDD+SL-SSDD pairing *before* writing any index.
    missing_masks = [x['source_id'] for x in output if masks and not x['sea_mask_path']]
    if missing_masks and not allow_partial_sea_masks:
        raise ValueError(f'{len(missing_masks)} SSDD images lack same-split SL-SSDD masks; examples: {missing_masks[:8]}. '
                         'Check SL-SSDD/{train,test}/ and SSDD annotation splits, or explicitly pass --allow-partial-sea-masks.')
    jsonl_write(out, output)
    if not output:
        raise RuntimeError('No indexed images')
    group_splits = defaultdict(set)
    for row in output:
        group_splits[row['source_group']].add(row['split'])
    return {
        'total_images': len(output), 'ships': sum(len(x['annotations']) for x in output),
        'splits': dict(Counter(x['split'] for x in output)),
        'official_splits': dict(Counter(x['official_split'] or 'none' for x in output)),
        'scene_types': dict(Counter(x['scene_type'] for x in output)),
        'distinct_groups': len(group_splits),
        'cross_split_groups': sum(len(x) > 1 for x in group_splits.values()),
        'sea_masks': sum(bool(x['sea_mask_path']) for x in output),
        'sea_masks_missing': len(missing_masks),
        'sea_masks_by_official_split': dict(Counter(x['official_split'] for x in output if x['sea_mask_path'])),
        'split_policy': policy, 'index_path': str(Path(out).resolve()),
    }
