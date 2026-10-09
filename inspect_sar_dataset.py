#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Inspect local SAR datasets (HRSID / SSDD / SL-SSDD / background) without third-party dependencies.

Usage:
    python inspect_sar_dataset.py --root "D:\\SAR\\HRSID" "D:\\SAR\\SSDD" \
        --output dataset_structure_report.txt --json dataset_structure_report.json

ZIP contents can be listed without extraction; RAR should be extracted first.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
ARCHIVE_EXT = {'.zip', '.rar', '.7z', '.tar', '.gz'}
SKIP_DIRS = {'__MACOSX', '.git', '.svn', '__pycache__'}


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def fmt_size(n: int) -> str:
    if n < 1024:
        return f'{n} B'
    for u in ('KiB', 'MiB', 'GiB', 'TiB'):
        n /= 1024
        if n < 1024:
            return f'{n:.2f} {u}'
    return f'{n:.2f} PiB'


def trim(items, n: int = 6) -> str:
    items = [str(x) for x in items]
    shown = ', '.join(items[:n])
    return shown + (f' ... (+{len(items)-n})' if len(items) > n else '')


def scan_files(root: Path):
    """Walk once, without following symlinked directories."""
    import os
    allfiles = []
    all_dirs = []
    errors = []
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith('.'))
        path = Path(base)
        all_dirs.append(path)
        for name in sorted(files):
            if name.startswith('._') or name in {'.DS_Store', 'Thumbs.db'}:
                continue
            p = path / name
            if p.is_symlink():
                continue
            allfiles.append(p)
    return allfiles, all_dirs, errors


def render_tree(root: Path, files: list[Path], dirs: list[Path], max_depth: int, max_items: int):
    file_by_parent = defaultdict(list)
    dir_by_parent = defaultdict(list)
    for f in files:
        file_by_parent[f.parent].append(f)
    for d in dirs:
        if d != root:
            dir_by_parent[d.parent].append(d)
    out = [f'{root.name}/']

    def visit(parent: Path, depth: int):
        if depth >= max_depth:
            extra = len(dir_by_parent[parent])
            if extra:
                out.append('    ' * depth + f'... ({extra} subfolders hidden by depth limit)')
            return
        ds = sorted(dir_by_parent[parent], key=lambda p: p.name.lower())
        fs = sorted(file_by_parent[parent], key=lambda p: p.name.lower())
        for d in ds[:max_items]:
            sub_count = len(dir_by_parent[d])
            file_count = len(file_by_parent[d])
            out.append('    ' * depth + f'+ {d.name}/  [direct: {file_count} files, {sub_count} dirs]')
            visit(d, depth + 1)
        if len(ds) > max_items:
            out.append('    ' * depth + f'... {len(ds)-max_items} more folders')
        for f in fs[:max_items]:
            try:
                size = fmt_size(f.stat().st_size)
            except OSError:
                size = 'size unavailable'
            out.append('    ' * depth + f'- {f.name} ({size})')
        if len(fs) > max_items:
            out.append('    ' * depth + f'... {len(fs)-max_items} more files (not listed)')
    visit(root, 1)
    return out


def sample_image_dimensions(images: list[Path], root: Path, max_sample: int):
    try:
        from PIL import Image
    except ImportError:
        return {'notice': 'Pillow not installed; dimensions skipped (optional: pip install Pillow)', 'samples': []}
    if not images:
        return {'samples': []}
    chosen = images[: max_sample]
    if len(images) > max_sample:
        chosen += [images[-1]]
    output = []
    for path in chosen:
        try:
            with Image.open(path) as im:
                output.append({'file': rel(path, root), 'size_wh': list(im.size), 'mode': im.mode})
        except Exception as exc:
            output.append({'file': rel(path, root), 'error': str(exc)[:150]})
    return {'samples': output}


def coco_info(path: Path, root: Path, image_names: set[str], image_stems: set[str], max_mb: int = 150):
    info = {'path': rel(path, root), 'bytes': path.stat().st_size}
    if info['bytes'] > max_mb * 1024**2:
        info['note'] = f'Skipped JSON parsing (>{max_mb} MiB); increase --max-json-mb if needed'
        return info
    try:
        with path.open('r', encoding='utf-8-sig') as f:
            js = json.load(f)
    except (OSError, ValueError, UnicodeError) as exc:
        info['error'] = str(exc)[:200]
        return info
    if not isinstance(js, dict):
        info['root_type'] = type(js).__name__
        return info
    info['top_keys'] = list(js.keys())[:20]
    ims = js.get('images')
    anns = js.get('annotations')
    if not isinstance(ims, list) or not isinstance(anns, list):
        info['format'] = 'non-COCO JSON (or missing images/annotations)'
        return info
    info['format'] = 'COCO candidate'
    info['images_count'] = len(ims)
    info['annotations_count'] = len(anns)
    cats = js.get('categories', [])
    info['categories'] = [{'id': c.get('id'), 'name': c.get('name')} for c in cats[:15] if isinstance(c, dict)] if isinstance(cats, list) else []
    refs = []
    exact = stem_match = missing = 0
    for image in ims:
        if not isinstance(image, dict):
            continue
        fname = image.get('file_name')
        if not isinstance(fname, str):
            continue
        if len(refs) < 6:
            refs.append(fname)
        base = Path(fname.replace('\\', '/')).name.lower()
        if base in image_names:
            exact += 1
        elif Path(base).stem in image_stems:
            stem_match += 1
        else:
            missing += 1
    info['example_image_filenames'] = refs
    info['reference_match'] = {'exact_basename': exact, 'stem_only': stem_match, 'missing': missing}
    if anns:
        first = next((a for a in anns if isinstance(a, dict)), None)
        if first is not None:
            info['annotation_sample_keys'] = list(first.keys())
            info['has_bbox'] = any(isinstance(a, dict) and 'bbox' in a for a in anns[:100])
            info['has_segmentation'] = any(isinstance(a, dict) and 'segmentation' in a for a in anns[:100])
    return info


def inspect_voc(xmls: list[Path], root: Path, img_stems: set[str], sample_n: int):
    out = {'xml_count': len(xmls), 'matching_image_stem_count': sum(x.stem.lower() in img_stems for x in xmls), 'samples': []}
    for p in xmls[:sample_n]:
        item = {'path': rel(p, root), 'has_matching_image_stem': p.stem.lower() in img_stems}
        try:
            x = ET.parse(p).getroot()
            item['root_tag'] = x.tag
            item['filename_tag'] = x.findtext('filename')
            item['size'] = {k: x.findtext('size/' + k) for k in ('width', 'height', 'depth')}
            objects = x.findall('.//object')
            item['num_objects'] = len(objects)
            item['object_classes'] = dict(Counter((o.findtext('name') or '?') for o in objects))
            bb = next((o.find('bndbox') for o in objects if o.find('bndbox') is not None), None)
            if bb is not None:
                item['bbox_example'] = {k: bb.findtext(k) for k in ('xmin', 'ymin', 'xmax', 'ymax')}
        except (OSError, ET.ParseError) as exc:
            item['parse_error'] = str(exc)[:200]
        out['samples'].append(item)
    return out


def split_files(files: list[Path], root: Path, sample_n: int):
    matched = []
    for path in files:
        if path.suffix.lower() != '.txt':
            continue
        low = path.as_posix().lower()
        name = path.stem.lower()
        if any(x in name for x in ('train', 'test', 'val', 'split')) or 'imagesets/' in low:
            item = {'path': rel(path, root)}
            try:
                with path.open('r', encoding='utf-8-sig', errors='replace') as f:
                    sample = []
                    for _ in range(3):
                        line = f.readline()
                        if not line:
                            break
                        sample.append(line.strip()[:100])
                item['sample_lines'] = sample
            except OSError as exc:
                item['error'] = str(exc)
            matched.append(item)
    return matched[:sample_n]


def check_directory(root: Path, max_depth: int, max_items: int, sample: int, max_json_mb: int):
    files, dirs, scan_errors = scan_files(root)
    by_ext = Counter((p.suffix.lower() or '[no extension]') for p in files)
    images = [p for p in files if p.suffix.lower() in IMAGE_EXT]
    image_names = {p.name.lower() for p in images}
    img_stems = {p.stem.lower() for p in images}
    by_stem = Counter(p.stem.lower() for p in images)
    dup_stems = [(s, c) for s, c in by_stem.items() if c > 1]
    jsons = [p for p in files if p.suffix.lower() == '.json']
    xmls = [p for p in files if p.suffix.lower() == '.xml']
    mask_like = [p for p in images if 'mask' in p.as_posix().lower() or 'segmentation' in p.as_posix().lower()]
    annotation_json = [coco_info(p, root, image_names, img_stems, max_json_mb) for p in jsons[:12]]
    if len(jsons) > 12:
        json_note = f'{len(jsons)-12} additional JSON files not inspected'
    else:
        json_note = None
    return {
        'root': str(root), 'kind': 'directory', 'total_files': len(files), 'total_dirs': len(dirs),
        'extension_counts': dict(by_ext.most_common()),
        'image_count_including_possible_masks': len(images),
        'image_type_counts': dict(Counter(p.suffix.lower() for p in images)),
        'example_images': [rel(p, root) for p in images[:8]],
        'image_dimensions': sample_image_dimensions(images, root, sample),
        'possible_mask_image_count': len(mask_like),
        'example_mask_files': [rel(p, root) for p in mask_like[:5]],
        'duplicate_image_stems': sorted(dup_stems, key=lambda x: (-x[1], x[0]))[:8],
        'json_files': annotation_json, 'json_note': json_note,
        'voc': inspect_voc(xmls, root, img_stems, sample) if xmls else None,
        'split_file_samples': split_files(files, root, sample),
        'archive_files_present': [rel(p, root) for p in files if p.suffix.lower() in ARCHIVE_EXT][:12],
        'tree': render_tree(root, files, dirs, max_depth, max_items), 'scan_errors': scan_errors,
    }


def check_archive(path: Path, max_depth: int, max_items: int):
    result = {'root': str(path), 'kind': 'archive', 'archive_type': path.suffix.lower()}
    if path.suffix.lower() == '.zip':
        try:
            with zipfile.ZipFile(path) as f:
                names = [z.filename for z in f.infolist() if not z.is_dir()]
                result['entries_count'] = len(names)
                result['extension_counts'] = dict(Counter(Path(x).suffix.lower() or '[no extension]' for x in names))
                result['example_entries'] = names[:30]
                result['top_level_entries'] = dict(Counter(x.split('/')[0] for x in names).most_common(15))
                result['note'] = 'ZIP central directory only; extract ZIP and rescan for annotation details'
        except (OSError, zipfile.BadZipFile) as exc:
            result['error'] = str(exc)
    elif path.suffix.lower() == '.rar':
        try:
            import rarfile  # optional
            with rarfile.RarFile(path) as rf:
                names = rf.namelist()
                result['entries_count'] = len(names)
                result['example_entries'] = names[:30]
                result['top_level_entries'] = dict(Counter(x.replace('\\', '/').split('/')[0] for x in names).most_common(15))
                result['note'] = 'RAR list only; please extract and scan directory for label pairing'
        except (ImportError, Exception) as exc:
            result['note'] = 'RAR inspection requires optional rarfile; simplest: extract .rar with 7-Zip and scan the directory'
            result['detail'] = str(exc)[:120]
    else:
        result['note'] = 'Unsupported archive for direct inspection; extract it first and scan directory'
    return result


def create_text(result: dict) -> str:
    lines = ['SAR DATASET STRUCTURE INSPECTION', 'Generated: ' + result['generated'],
             'Python: ' + sys.version.split()[0], 'Dependency: standard library only (Pillow optional)',
             '=' * 78]
    for ds in result['datasets']:
        lines += ['', '#' * 78, 'ROOT: ' + ds['root'], 'TYPE: ' + ds['kind']]
        if ds['kind'] == 'directory':
            lines += [f"Files: {ds['total_files']} | Folders: {ds['total_dirs']} | Image candidates: {ds['image_count_including_possible_masks']}",
                      'Extension counts: ' + json.dumps(ds['extension_counts'], ensure_ascii=False),
                      '', '[DIRECTORY TREE, TRUNCATED]']
            lines += ds['tree']
            lines += ['', '[IMAGE SAMPLES / DIMENSIONS]']
            lines += [json.dumps(ds['image_dimensions'], ensure_ascii=False, indent=2)]
            lines += [f"[POSSIBLE MASKS] count={ds['possible_mask_image_count']} examples={trim(ds['example_mask_files'])}"]
            lines += [f"[DUPLICATE IMAGE STEMS] {trim(ds['duplicate_image_stems'])}"]
            lines += ['', '[COCO / JSON LABEL FILES]']
            for j in ds['json_files']:
                lines.append(json.dumps(j, ensure_ascii=False, indent=2))
            if ds['json_note']:
                lines.append(ds['json_note'])
            lines += ['', '[VOC / XML LABEL FILES]']
            lines.append(json.dumps(ds['voc'], ensure_ascii=False, indent=2) if ds['voc'] else 'none')
            lines += ['', '[TRAIN/VAL/TEST LISTS]']
            lines.append(json.dumps(ds['split_file_samples'], ensure_ascii=False, indent=2))
            lines += ['', '[ARCHIVES STILL UNDER ROOT] ' + trim(ds['archive_files_present'])]
        else:
            lines.append(json.dumps({k: v for k, v in ds.items() if k != 'root'}, ensure_ascii=False, indent=2))
        lines += ['', '[COMPATIBILITY INDICATIONS]']
        if ds['kind'] == 'directory':
            has_coco = any(j.get('format') == 'COCO candidate' for j in ds['json_files'])
            has_voc = ds['voc'] is not None and ds['voc']['xml_count'] > 0
            if has_coco:
                lines.append('COCO-like annotations detected. Check reference_match/missing and categories above.')
            if has_voc:
                lines.append('VOC XML detected. Check matching_image_stem_count vs xml_count above.')
            if not (has_coco or has_voc):
                lines.append('No recognizable COCO/VOC labeling detected: may be backgrounds, unextracted labels, or a different layout.')
            if ds['archive_files_present']:
                lines.append('Archive files remain in this folder; if images/labels are not counted, extract before scanning.')
        else:
            lines.append('Inspect extracted directory before choosing previous pipeline adapter.')
    lines += ['', '=' * 78, 'Send this TXT (and optionally JSON) back for adapter compatibility review.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Inspect SAR dataset file/folder structures without modifying source data.')
    parser.add_argument('--root', nargs='+', required=True, help='One or more dataset directories or ZIP/RAR files')
    parser.add_argument('--output', default='dataset_structure_report.txt', help='Save UTF-8 text report')
    parser.add_argument('--json', default='dataset_structure_report.json', help='Save machine-readable JSON')
    parser.add_argument('--max-depth', type=int, default=4, help='Maximum printed directory depth (default 4)')
    parser.add_argument('--max-items', type=int, default=8, help='Max children shown per folder (default 8)')
    parser.add_argument('--samples', type=int, default=5, help='Number of image/XML samples')
    parser.add_argument('--max-json-mb', type=int, default=150, help='Maximum size of parsed annotation JSON')
    args = parser.parse_args()
    result = {'generated': datetime.now().astimezone().isoformat(timespec='seconds'), 'datasets': []}
    for raw in args.root:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            parser.error(f'Path does not exist: {p}')
        print(f'Scanning: {p}', flush=True)
        if p.is_dir():
            result['datasets'].append(check_directory(p, args.max_depth, args.max_items, args.samples, args.max_json_mb))
        elif p.is_file():
            result['datasets'].append(check_archive(p, args.max_depth, args.max_items))
        else:
            parser.error(f'Not a readable file/folder: {p}')
    txt = Path(args.output)
    jpath = Path(args.json)
    txt.parent.mkdir(parents=True, exist_ok=True)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text(create_text(result), encoding='utf-8')
    jpath.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote TXT:  {txt.resolve()}')
    print(f'Wrote JSON: {jpath.resolve()}')
    print('Summary:')
    for d in result['datasets']:
        if d['kind'] == 'directory':
            c = d['voc']['xml_count'] if d['voc'] else 0
            co = sum(j.get('images_count', 0) for j in d['json_files'])
            print(f"  {Path(d['root']).name}: images={d['image_count_including_possible_masks']}, COCO refs={co}, XML={c}")
        else:
            print(f"  {Path(d['root']).name}: archive entries={d.get('entries_count', 'unknown')}")

if __name__ == '__main__':
    main()
