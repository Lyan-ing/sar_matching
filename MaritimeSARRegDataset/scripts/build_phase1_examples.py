"""Fixed source coverage, independent of the default with-replacement sampler."""
import argparse
import json
from pathlib import Path

from sarreg.io import jsonl_read, jsonl_write, read_gray, save_json
from sarreg.synthesis import load_cfg, stable_seed, synthesize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--index', required=True); p.add_argument('--out', required=True)
    p.add_argument('--config', default='configs/easy.yaml'); p.add_argument('--seed', type=int, default=46)
    a = p.parse_args(); rows = jsonl_read(a.index); cfg = load_cfg(a.config)
    out = Path(a.out)
    if out.exists(): raise FileExistsError(out)
    selected = []
    for ds, scene in (('ssdd','inshore'), ('ssdd','offshore'), ('hrsid','inshore'),
                      ('hrsid','offshore'), ('background','unknown')):
        candidates = sorted((r for r in rows if r['dataset'] == ds and r['scene_type'] == scene), key=lambda r: r['image_path'])
        if ds == 'ssdd' and scene == 'inshore':
            candidates = [r for r in candidates if read_gray(r['sea_mask_path']).min() == 0]
        if not candidates: raise ValueError(f'No real source for {ds}/{scene}')
        selected.append(candidates[0])
    # Three extra SSDD views with mixed sea/land, including official train (unknown scene label).
    used = {r['image_path'] for r in selected}
    for r in sorted(rows, key=lambda r: r['image_path']):
        if len(selected) == 8: break
        if r['dataset'] == 'ssdd' and r['image_path'] not in used:
            mask = read_gray(r['sea_mask_path'])
            if mask.min() == 0 and mask.max() == 255:
                selected.append(r); used.add(r['image_path'])
    out.mkdir(parents=True)
    jsonl_write(out/'fixed_source_index.jsonl', selected)
    config = {'index': str(Path(a.index).resolve()), 'config': cfg, 'global_seed': a.seed,
              'source_selection': 'fixed_source_list_sorted_scene_coverage', 'requested_count': len(selected)}
    save_json(out/'generation_config.json', config)
    manifest = []
    for i, row in enumerate(selected):
        folder = out/row['split']/f'pair_{i:06d}'
        manifest.append(synthesize(row, cfg, folder, stable_seed(a.seed, f'{i}:{row["image_path"]}')))
    jsonl_write(out/'manifest_all.jsonl', manifest)
    save_json(out/'generation_report_all.json', {'requested': len(selected), 'generated': len(manifest), 'failure_count': 0, 'failures': []})
    print(json.dumps({'generated': len(manifest), 'out': str(out), 'sources': [(r['dataset'], r['source_id'], r['scene_type']) for r in selected]}, indent=2))


if __name__ == '__main__': main()
