"""Plan/copy/verify historical payloads, then replace originals with symlinks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parents[1]
CODE = {'.py', '.pyc', '.pyo', '.sh', '.bash', '.js', '.ts', '.tsx', '.jsx',
        '.ipynb', '.cpp', '.c', '.h', '.cu', '.so', '.whl', '.toml', '.yaml', '.yml', '.html'}
SKIP = {'code', 'scripts', 'tests', 'tools', 'configs', 'sarreg', 'vendor', '__pycache__'}
ACTIVE = {'RealSAR_GeoAssist_20261012_1917', 'RealSAR_SingleExpert_20261015_2332'}


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def category(rel):
    parts = rel.parts
    top = parts[0]
    child = parts[1] if len(parts) > 1 else ''
    if top.startswith('SyntheticSAR_'):
        return 'artifacts' if child.startswith(('features', 'pdf_export')) else 'datasets'
    if top.startswith('DualSAR_DatasetPrep_') and child.startswith('synthetic_'):
        return 'datasets'
    if top == 'real_sar_pilot' or (top == 'MaritimeSARRegDataset' and child in {'data', 'output'}):
        return 'datasets'
    data_children = {
        'RealSAR_Multiview512_': {'acquisitions', 'pairs', 'evidence', 'legacy_pointwise_inputs'},
        'RealSAR_AngleExpansion_': {'assets', 'regions'},
        'RealSAR_MultiViewV2_': {'regions'},
        'RealSAR_ClearCandidates_': {'acquisitions'},
        'RealSAR_PhysicalPrep_': {'blind_release', 'custodian', 'assisted_research'},
        'RealSAR_PairingAudit_': {'pairs'},
    }
    for prefix, children in data_children.items():
        if top.startswith(prefix) and (child in children or child.startswith('annotation_')):
            return 'datasets'
    if top == 'MaritimeSARRegistrationPhase2' and any(x in {'data', 'data_v2', 'confirmation'} for x in parts[2:]):
        return 'datasets'
    return 'artifacts'


def plan():
    selected = []

    def visit(p):
        rel = p.relative_to(BASE)
        if p.is_symlink() or any(x.startswith('.') or x in SKIP for x in rel.parts) or rel.parts[0] in ACTIVE:
            return 0, True, set(), []
        if p.is_file():
            size = p.stat().st_size
            protected = p.suffix.lower() in CODE or p.name.endswith('.egg-info') or 'logs' in rel.parts
            kind = category(rel)
            entry = dict(source=str(p), relative=str(rel), category=kind, bytes=size, directory=False)
            return size, protected, {kind}, [entry] if not protected and size >= 1024**2 else []
        size, protected, kinds, entries = 0, False, set(), []
        for child in sorted(p.iterdir()):
            n, keep, types, found = visit(child)
            size += n
            protected |= keep
            kinds |= types
            entries.extend(found)
        if not protected and len(kinds) == 1 and size >= 8 * 1024**2 and len(rel.parts) >= 2:
            entries = [dict(source=str(p), relative=str(rel), category=next(iter(kinds)), bytes=size, directory=True)]
        return size, protected, kinds, entries

    for p in sorted(BASE.iterdir()):
        if p.name.startswith('StorageMigration_'):
            continue
        selected.extend(visit(p)[3])
    return selected


def inventory(root):
    records = []
    paths = [root] if root.is_file() else sorted(root.rglob('*'))
    for p in paths:
        rel = '.' if p == root else str(p.relative_to(root))
        if p.is_symlink():
            records.append(dict(path=rel, type='link', target=os.readlink(p)))
        elif p.is_file():
            s = p.stat()
            records.append(dict(path=rel, type='file', size=s.st_size, mtime_ns=s.st_mtime_ns, sha256=sha(p)))
        elif p.is_dir():
            records.append(dict(path=rel, type='directory'))
        else:
            raise RuntimeError(f'Unsupported special file: {p}')
    return records


def execute(plan_path, kind, target_root, journal):
    target_root = Path(target_root)
    if not target_root.is_dir() or not os.access(target_root, os.W_OK):
        raise PermissionError(f'Target must already exist and be writable: {target_root}')
    entries = [e for e in json.loads(Path(plan_path).read_text())['entries'] if e['category'] == kind]
    with open(journal, 'x') as log:
        for i, e in enumerate(entries, 1):
            src = Path(e['source']); dst = target_root / e['relative']
            if src.is_symlink() and src.resolve() == dst.resolve():
                print(f'SKIP already relocated {src}', flush=True)
                continue
            assert not src.is_symlink() and src.exists(), src
            assert not os.path.lexists(dst), f'Refusing to overwrite: {dst}'
            dst.parent.mkdir(parents=True, exist_ok=True)
            stage = dst.with_name(dst.name + f'.migration-partial-{os.getpid()}')
            assert not os.path.lexists(stage), stage
            print(f'{now()} [{i}/{len(entries)}] COPY {e["relative"]} {e["bytes"]}', flush=True)
            before = inventory(src)
            if src.is_dir():
                stage.mkdir()
                subprocess.run(['rsync', '-aH', '--', str(src) + '/', str(stage) + '/'], check=True)
            else:
                subprocess.run(['rsync', '-a', '--', str(src), str(stage)], check=True)
            assert before == inventory(stage), f'Copy content/metadata differs: {src}'
            # Preserve external relative links when their parent moves to another volume.
            rebased = []
            for item in before:
                source_item = src / item['path']
                if item['type'] == 'file':
                    st = source_item.stat() if src.is_dir() else src.stat()
                    assert (st.st_size, st.st_mtime_ns) == (item['size'], item['mtime_ns']), f'Source changed: {src}'
                elif item['type'] == 'link' and not os.path.isabs(item['target']):
                    target = Path(os.path.normpath(source_item.parent / item['target']))
                    if not target.is_relative_to(src):
                        copied = stage / item['path']; copied.unlink(); copied.symlink_to(target)
                        rebased.append(dict(path=item['path'], old=item['target'], new=str(target)))
            # Publish only after full SHA256 verification, retaining a rollback source.
            stage.rename(dst)
            backup = src.with_name(src.name + f'.migration-original-{os.getpid()}')
            link = src.with_name(src.name + f'.migration-link-{os.getpid()}')
            assert not os.path.lexists(backup) and not os.path.lexists(link)
            link.symlink_to(dst, target_is_directory=e['directory'])
            src.rename(backup)
            try:
                os.replace(link, src)
                assert src.resolve() == dst.resolve()
            except BaseException:
                if os.path.lexists(src): src.unlink()
                backup.rename(src)
                raise
            record = dict(**e, destination=str(dst), completed=now(), verified=True,
                          files=before, rebased_relative_links=rebased)
            log.write(json.dumps(record, ensure_ascii=False) + '\n'); log.flush(); os.fsync(log.fileno())
            if backup.is_dir(): shutil.rmtree(backup)
            else: backup.unlink()
            print(f'{now()} VERIFIED_AND_LINKED {e["relative"]}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan-output')
    ap.add_argument('--plan')
    ap.add_argument('--category', choices=['datasets', 'artifacts'])
    ap.add_argument('--target')
    ap.add_argument('--journal')
    a = ap.parse_args()
    if a.plan_output:
        entries = plan()
        with open(a.plan_output, 'x') as f:
            json.dump(dict(created=now(), project=str(BASE), entries=entries,
                           skipped_active_roots=sorted(ACTIVE)), f, ensure_ascii=False, indent=2)
        for kind in ['datasets', 'artifacts']:
            group = [e for e in entries if e['category'] == kind]
            print(kind, len(group), 'entries', round(sum(e['bytes'] for e in group) / 1024**3, 3), 'GiB')
    else:
        execute(a.plan, a.category, a.target, a.journal)
