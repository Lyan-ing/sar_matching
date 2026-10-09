"""Read original Phase 1 meta.json pairs without depending on Phase 1's installed package.
GT and masks are loaded ONLY after method.predict(image0, image1) returns.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import csv
import numpy as np
import cv2

@dataclass(frozen=True)
class PairRef:
    folder: Path
    view_i: int
    view_j: int
    pair_meta: dict
    meta: dict
    profile: str
    scene: str
    source_id: str
    source_group: str
    source_component: str
    dataset: str
    test_source: str

    @property
    def id(self):
        return f'{self.folder.name}:{self.view_i}-{self.view_j}'


def _read_json(path: Path) -> dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Failed to decode required image: {path}')
    return image


def source_component(meta: dict) -> str:
    for key in ('source_component','source_component_id','component_id','merged_source_component',
                'leakage_component_id','pixel_source_component'):
        if meta.get(key) is not None:
            return str(meta[key])
    if meta.get('split_protocol') == 'phase1_pixel_supported_source_components_v2' and meta.get('source_group'):
        namespace='hrsid' if meta.get('dataset')=='background' else meta.get('dataset','unknown')
        return f'{namespace}:{meta["source_group"]}'
    return 'unknown'


def _name_from_meta(meta: dict, folder: Path, keys: tuple, default: str) -> str:
    for k in keys:
        value = meta.get(k)
        if isinstance(value, str) and value:
            return value
    # Formal strata may be in a separate sample metadata JSON.
    for name in ('sample_config.json', 'generation.json', 'sample_meta.json'):
        p = folder/name
        if p.exists():
            extra = _read_json(p)
            for k in keys:
                value = extra.get(k)
                if isinstance(value, str) and value:
                    return value
    return default


def _manifest_lookup(root: Path) -> dict[str, dict]:
    """Optional annotation merge if formal manifest contains profile/stratum fields."""
    candidates = [root/'manifest.jsonl', root/'manifest.json', root/'sample_manifest.jsonl', root/'samples.jsonl',
                  root/'manifest.csv', root/'sample_statistics.csv', root/'statistics.csv',
                  root/'reports'/'sample_statistics.csv']
    mapping: dict[str, dict] = {}
    for path in candidates:
        if not path.is_file():
            continue
        try:
            if path.suffix == '.jsonl':
                rows = [json.loads(s) for s in path.read_text(encoding='utf8').splitlines() if s.strip()]
            elif path.suffix == '.csv':
                with path.open(newline='',encoding='utf8-sig') as file:
                    rows = list(csv.DictReader(file))
            else:
                value = _read_json(path)
                rows = value if isinstance(value, list) else value.get('samples', value.get('records', []))
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key in ('sample_id', 'pair_id', 'id', 'folder_name'):
                    if key in row:
                        mapping[str(row[key])] = row
                        break
        except (OSError, ValueError, TypeError):
            # Metadata is optional; the per-pair meta.json remains mandatory.
            continue
    return mapping


def discover(root: str | Path, *, limit: int | None = None, require_test: bool = True) -> list[PairRef]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Phase 1 frozen dataset directory missing: {root}')
    manifest = _manifest_lookup(root)
    metas = sorted(root.rglob('meta.json'))
    metas = [p for p in metas if p.parent.name.startswith('pair_')]
    if not metas:
        raise ValueError(f'No pair_*/meta.json files under {root}; check --root')
    refs = []
    for path in metas:
        meta = _read_json(path)
        folder = path.parent
        sample_meta = {**manifest.get(folder.name, {}), **meta}
        for nested_key in ('formal', 'experiment', 'sample', 'stratification', 'generation'):
            nested = sample_meta.get(nested_key, {})
            if isinstance(nested,dict):
                for k,v in nested.items():
                    sample_meta.setdefault(k,v)
        source_split = str(sample_meta.get('source_split', sample_meta.get('split', 'unknown')))
        if require_test and source_split.lower() != 'test':
            raise ValueError(f'Non-test sample inside frozen benchmark: {folder} source_split={source_split}')
        profile = _name_from_meta(sample_meta, folder, ('profile', 'difficulty', 'profile_name', 'formal_profile'), 'unknown')
        scene = _name_from_meta(sample_meta, folder, ('scene_stratum', 'scene_type', 'scene', 'stratum_scene'), 'unknown')
        for p in meta.get('pairs', []):
            if 'views' not in p or 'H_i_to_j' not in p:
                raise ValueError(f'Pair metadata missing views or GT matrix: {path}')
            i, j = map(int, p['views'])
            if not (folder/f'view_{i}.png').exists() or not (folder/f'view_{j}.png').exists():
                raise FileNotFoundError(f'Missing view image for {folder}: {i}-{j}')
            refs.append(PairRef(folder, i, j, p, meta, profile, scene,
                                str(sample_meta.get('source_id', 'unknown')),
                                str(sample_meta.get('source_group','unknown')),
                                source_component(sample_meta),
                                str(sample_meta.get('dataset', 'unknown')), source_split))
    if limit is not None:
        if limit < 1:
            raise ValueError('--limit must be >=1')
        refs = refs[:limit]
    return refs


def load_images(ref: PairRef) -> tuple[np.ndarray, np.ndarray]:
    return read_gray(ref.folder/f'view_{ref.view_i}.png'), read_gray(ref.folder/f'view_{ref.view_j}.png')


def load_gt(ref: PairRef) -> dict[str, Any]:
    """Not for inference: call ONLY after prediction is complete."""
    p = ref.pair_meta
    gt_h = np.asarray(p['H_i_to_j'], np.float64)
    if gt_h.shape != (3, 3) or not np.isfinite(gt_h).all():
        raise ValueError(f'Bad GT H: {ref.id}')
    f = ref.folder / p['matches']
    if not f.is_file():
        raise FileNotFoundError(f'Missing GT points: {f}')
    with np.load(f, allow_pickle=False) as a:
        x = np.asarray(a['points_i'], np.float64).reshape(-1, 2)
        y = np.asarray(a['points_j'], np.float64).reshape(-1, 2)
    if len(x) != len(y):
        raise ValueError(f'Bad correspondence shape: {f}')
    overlap = min(float(p.get('overlap_fraction_i', 0.0)), float(p.get('overlap_fraction_j', 0.0)))
    return {'H': gt_h, 'points0': x, 'points1': y, 'overlap': overlap,
            'no_overlap': overlap <= 1e-9,
            'count': len(x)}


def describe(refs: list[PairRef]) -> dict:
    from collections import Counter
    return {'relations':len(refs), 'groups':len({str(x.folder) for x in refs}),
            'datasets':dict(Counter(x.dataset for x in refs)),
            'profiles':dict(Counter(x.profile for x in refs)),
            'scenes':dict(Counter(x.scene for x in refs)),
            'source_splits':dict(Counter(x.test_source for x in refs)),
            'missing_profile':sum(x.profile=='unknown' for x in refs),
            'missing_scene':sum(x.scene=='unknown' for x in refs),
            'missing_component':sum(x.source_component=='unknown' for x in refs)}
