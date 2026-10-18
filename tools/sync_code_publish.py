#!/usr/bin/env python3
"""Copy an allowlisted code/documentation snapshot to the existing publication checkout.

Never follows data symlinks and never edits source experiments. Does not commit or push.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

SKIP = {'.git','.code-publish','.aws','.agents','.codex','.venv','venv','__pycache__','.pytest_cache',
        'cache','mpl_cache','vendor','_vendor','node_modules','site-packages','outputs','output','reports',
        'logs','runs','predictions','checkpoints','weights','data','dataset','datasets','figures','previews',
        'visualizations','exports','verification','evidence','tables','maintenance','documentation',
        'legacy_review','meta','checks','browser','review','quality','discovery','sources','context_snapshot',
        'backup','code_snapshot','code_snapshot_final'}
PREFIX_SKIP = ('annotation_','browser_','ui_checks','batch_import','gt_intake_','priority_visual_',
               'collection_smoke','previews_','pdf_export','smoke','cli_','synthetic_preflight',
               'preflight_rejected','before_','drafts_','ui_upgrade_')
CODE_EXT = {'.py','.sh','.js','.html','.css','.toml','.yaml','.yml','.ini','.cfg'}
SCHEMAS = {'annotation_schema.json','schema.json'}


def allowed_dir(name):
    return name not in SKIP and not name.endswith('.egg-info') and (name == 'annotation_adapter' or not name.startswith(PREFIX_SKIP))


def sources(root):
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if allowed_dir(d) and not (Path(directory)/d).is_symlink()]
        for name in names:
            path=Path(directory)/name;rel=path.relative_to(root)
            if path.is_symlink() or path.stat().st_size>200000 or name=='codex_lightweight.py':continue
            if path.suffix in {'.html','.js','.css'}:
                if not any(x in rel.parts for x in ['code','annotation_adapter','annotation_ui','sarreg','sarbench','tests','scripts']):continue
                if path.suffix=='.html' and ('report' in name.lower() or 'operations' in name.lower()):continue
            config_json = path.suffix=='.json' and 'configs' in rel.parts and not name.startswith(('provenance','host_','identity'))
            dependency = name.startswith('requirements') and path.suffix=='.txt'
            license_file = name.startswith(('LICENSE','COPYING','NOTICE')) and path.suffix in {'','.txt','.md'}
            # Only module-level usage docs, not histories, experiment reports or handoffs.
            usage = (name=='README.md' and len(rel.parts)==2 and rel.parts[0] in {
                'MaritimeSARRegDataset','MaritimeSARRegistrationPhase2','RealSAR_MultiSource_512_Kit','RealSAR_Sentinel1_DownloadKit'})
            protocol = name.startswith(('DATASET_USAGE','SUPERVISION_PROTOCOL')) and path.suffix=='.md'
            formal = name in {'FORMAL_PROTOCOL_20261017_0908.json','FORMAL_PROTOCOL_20261017_1343.json'}
            if path.suffix in CODE_EXT or config_json or dependency or license_file or usage or protocol or formal or name in SCHEMAS:
                yield rel


def sync(root, destination):
    if not (destination/'.git').is_dir():raise ValueError('Use the existing publication checkout, not the working research tree.')
    entries={str(p):root/p for p in sources(root)}
    # Keep Apache components only. The restricted SuperPoint source is fetched locally after license acceptance.
    for name in ['__init__.py','lightglue.py','utils.py']:
        rel=f'sar_match_portable/sar_match/_vendor/{name}';entries[rel]=root/rel
    # UI schema files remain code; actual annotation task/point data are excluded.
    for p in root.glob('*/annotation_adapter/annotation_schema.json'):entries[str(p.relative_to(root))]=p
    for name,source in sorted(entries.items()):
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink():raise ValueError(f'Unexpected publication symlink: {target}')
        if not target.exists() or source.read_bytes()!=target.read_bytes():shutil.copyfile(source,target)
    return entries


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--destination',type=Path);a=p.parse_args()
    root=a.root.resolve();dst=a.destination.resolve() if a.destination else root/'.code-publish'
    copied=sync(root,dst)
    print(json.dumps({'copied_code_config_files':len(copied),'bytes':sum(x.stat().st_size for x in copied.values()),'destination':str(dst)},ensure_ascii=False))
