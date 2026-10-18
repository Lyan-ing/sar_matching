"""Seal dev-only choices and protected pre-existing artifacts before validation."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from sarbench.reports import save_json

parser = argparse.ArgumentParser()
parser.add_argument('--run', required=True)
args = parser.parse_args()
run = Path(args.run).resolve()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
selection_path = run/'development/selection.json'
selection = json.loads(selection_path.read_text())
receipt = json.loads((run/'development/RUN_COMPLETE.json').read_text())
assert selection['role'] == 'dev' and not receipt['partial_smoke'] and receipt['exceptions'] == 0
assert sha(run/'development/results.csv') == receipt['results_sha256'] == selection['dev_results_sha256']
assert all(sha(path) == digest for path, digest in selection['inference_code_sha256'].items())
assert not (run/'validation').exists(), 'Validation must not have started'
lock = run/'validation_preregistration.json'
if lock.exists(): raise FileExistsError(lock)
save_json(lock, {'locked_at': datetime.now().astimezone().isoformat(),
                 'selection_sha256': sha(selection_path), 'selection': selection,
                 'validation_unexamined': True})
protected = run/'preserved_artifacts.json'
if not protected.exists():
    phase1 = Path('../MaritimeSARRegDataset/output/phase1_formal_20261009_1846').resolve()
    files = [p for root in Path('output').glob('phase2_*') for p in root.rglob('*') if p.is_file()]
    files += [p for p in Path('reports').glob('phase2_*') if p.is_file()]
    files += [phase1/'file_checksums_20261009_1846.json', phase1/'ACCEPTED_20261009_1846.json']
    save_json(protected, {'files': {str(p.resolve()): sha(p) for p in files}, 'phase1_root': str(phase1)})
print(json.dumps({'sealed': str(lock), 'selection_sha256': sha(selection_path)}, indent=2))
