"""External filename grouping and optional annotation-only diagnostics."""
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

PATTERN = re.compile(r'^(scene_\d+)_inc(\d+(?:\.\d+)?)_az(\d+(?:\.\d+)?)\.png$', re.I)


@dataclass(frozen=True)
class View:
    scene: str
    inc: float
    az: float
    path: Path
    readable: bool


def scan(root, incs=(20., 30., 45.), azs=(0., 60., 120., 180., 240., 300.)):
    root = Path(root)
    if not root.is_dir(): raise NotADirectoryError(root)
    records = {}; warnings = []; grouped = {}
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.png')
    for p in files:
        m = PATTERN.fullmatch(p.name)
        if not m:
            warnings.append(dict(type='unparseable_filename', file=str(p.relative_to(root)))); continue
        scene = m[1].lower(); inc, az = float(m[2]), float(m[3])
        if not np.isfinite([inc,az]).all():
            warnings.append(dict(type='nonfinite_angle_token',file=str(p.relative_to(root)))); continue
        try:
            image = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            readable = image is not None
        except cv2.error: readable = False
        if not readable: warnings.append(dict(type='unreadable_image', file=str(p.relative_to(root))))
        grouped.setdefault(scene, {}).setdefault((inc, az), []).append(View(scene, inc, az, p, readable))
    expected = set(itertools.product(incs, azs))
    for scene, slots in sorted(grouped.items()):
        ambiguous = {k: v for k, v in slots.items() if len(v) > 1}
        for (inc, az), vv in ambiguous.items():
            warnings.append(dict(type='duplicate_view_excluded', scene=scene, inc=inc, az=az, files=[str(v.path.relative_to(root)) for v in vv]))
        missing = sorted(expected - slots.keys()); unexpected = sorted(slots.keys() - expected)
        for inc, az in missing: warnings.append(dict(type='missing_view', scene=scene, inc=inc, az=az))
        if unexpected: warnings.append(dict(type='unexpected_angle_combinations', scene=scene, combinations=unexpected))
        vv = sorted((v[0] for k, v in slots.items() if k not in ambiguous), key=lambda x: (x.inc, x.az, str(x.path)))
        records[scene] = dict(views=vv, complete=not missing and not unexpected and not ambiguous and all(v.readable for v in vv),
                              present_unique_slots=len(slots), usable_unambiguous_slots=len(vv), missing=missing, duplicate_slots=len(ambiguous))
    return records, warnings


def category(a, b):
    if a.scene != b.scene: raise ValueError('Cross-scene pairing is forbidden.')
    if (a.inc, a.az) == (b.inc, b.az): raise ValueError('Identical view pairing is forbidden.')
    if a.inc == b.inc: return 'same_inc_different_az'
    if a.az == b.az: return 'same_az_different_inc'
    return 'both_different'


def make_pairs(views, pair_type='all', smoke=False):
    if pair_type not in ('all', 'same_inc', 'same_az', 'both_different'): raise ValueError(pair_type)
    allowed = dict(same_inc='same_inc_different_az', same_az='same_az_different_inc', both_different='both_different')
    pairs = [(a, b) for a, b in itertools.combinations(views, 2) if pair_type == 'all' or category(a, b) == allowed[pair_type]]
    if not smoke: return pairs
    # Fixed representatives: one per requested angle category, independent of inference.
    selected = []; seen = set()
    for a, b in pairs:
        c = category(a, b)
        if c not in seen: selected.append((a, b)); seen.add(c)
    return selected


def position_labels(entry, points, shape, annotation_root):
    """-2 unknown/outside, 0 explicitly complete nonship, >0 ships; IDs remain strings."""
    if entry is None: return None, None
    points = np.asarray(points).reshape(-1, 2); h, w = shape
    in_bounds = (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
    labels = np.full(len(points), 0 if entry.get('complete', False) else -2, int)
    ids = np.full(len(points), None, dtype=object)
    if 'mask' in entry:
        path = (Path(annotation_root) / entry['mask']).resolve()
        if not path.is_relative_to(Path(annotation_root).resolve()): raise ValueError('Annotation mask escapes annotation directory.')
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None or mask.ndim != 2 or mask.shape != shape or not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f'Annotation mask must be integer single-channel native size {shape}: {path}')
        q = np.rint(points).astype(int); q[:, 0] = q[:, 0].clip(0, w-1); q[:, 1] = q[:, 1].clip(0, h-1)
        values = mask[q[:, 1], q[:, 0]]
        labels[values > 0] = values[values > 0]
        for i, val in enumerate(values):
            if val > 0: ids[i] = entry.get('instance_ids', {}).get(str(int(val)))
    for index, box in enumerate(entry.get('boxes', []), 1):
        x0, y0, x1, y1 = map(float, box['xyxy'])
        if not np.isfinite([x0,y0,x1,y1]).all() or not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError('Invalid bbox; expected native xyxy, half-open coordinates within image.')
        hit = (points[:,0] >= x0) & (points[:,0] < x1) & (points[:,1] >= y0) & (points[:,1] < y1)
        overlap = hit & (labels > 0)
        labels[hit] = index; ids[hit] = box.get('instance_id')
        labels[overlap] = -2; ids[overlap] = None
    labels[~in_bounds] = -2; ids[~in_bounds] = None
    return labels, ids


def annotation_diagnostics(entries, arrays, shapes, annotation_root):
    if any(e is None for e in entries):
        return dict(annotations_available=False, region_counts=None, inlier_region_counts=None, same_ship_instance=None,
                    annotation_note='Not available for both images. No regions or physical correctness inferred.')
    la, ia = position_labels(entries[0], arrays['matches_A'], shapes[0], annotation_root)
    lb, ib = position_labels(entries[1], arrays['matches_B'], shapes[1], annotation_root)
    categories = dict(background=(la == 0) & (lb == 0), ship=(la > 0) & (lb > 0), cross_region=((la > 0) & (lb == 0)) | ((la == 0) & (lb > 0)), unknown=(la < 0) | (lb < 0))
    selected = (la > 0) & (lb > 0)
    known = selected & np.array([a is not None and b is not None for a,b in zip(ia,ib)], bool)
    same = known & np.array([a == b for a,b in zip(ia,ib)], bool)
    return dict(annotations_available=True, region_counts={k:int(v.sum()) for k,v in categories.items()},
                inlier_region_counts={k:int((v & arrays['inliers']).sum()) for k,v in categories.items()},
                same_ship_instance=dict(both_ids_known=int(known.sum()), same_id=int(same.sum()), different_id=int((known & ~same).sum()), unknown_id=int((selected & ~known).sum())),
                annotation_note='Region membership and instance identity only, not independent pixel-correspondence correctness.')


def pixel_consistency(a, b):
    if a.shape != b.shape: return dict(same_pixel_comparison_available=False)
    d = a.astype(np.float32)-b.astype(np.float32)
    return dict(same_pixel_comparison_available=True, equal_position_pixel_fraction=float(np.mean(d==0)), equal_position_MAE_DN=float(np.mean(np.abs(d))),
                background_reuse_note='Whole-image equal-position measurements, not isolated background. Reuse extent is unconfirmed without exhaustive ship annotations and generation metadata.')
