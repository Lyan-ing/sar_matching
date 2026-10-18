"""Verify a completed build is a no-op, and preserve interrupted partial output."""
from common import *
import build,contextlib,io,tempfile
from unittest.mock import patch

def main():
    cfg=latest('meta/GENERATION_CONFIG_*.json');records=list((ROOT/'pairs').glob('*/*/*_RECORD_*.json'));assert len(records)==6000
    def inventory():return {str(p.relative_to(ROOT)):(p.stat().st_size,p.stat().st_mtime_ns) for p in (ROOT/'pairs').rglob('*') if p.is_file()}
    before=inventory();capture=io.StringIO()
    with contextlib.redirect_stdout(capture):build.build(cfg,4)
    assert before==inventory();assert '"new_files_written": 0' in capture.getvalue()
    with tempfile.TemporaryDirectory(prefix='mixed_resume_test_') as tmp:
        root=Path(tmp);partial=root/'pairs/train/TEST_PARTIAL';partial.mkdir(parents=True);payload=b'interrupted synthetic test fixture, not a SAR image';(partial/'partial.bin').write_bytes(payload)
        plan=root/'EMPTY_PLAN_20261016_2114.jsonl';plan.write_text('')
        with patch.object(build,'ROOT',root),patch.object(build,'latest',lambda pattern:plan),contextlib.redirect_stdout(io.StringIO()):build.build(cfg,1)
        retained=list((root/'interrupted').glob('*/partial.bin'));assert len(retained)==1 and retained[0].read_bytes()==payload and not partial.exists()
    save(ROOT/f'meta/RESUME_CHECK_{stamp()}.json',dict(created=now(),passed=True,completed_pairs=6000,pair_files_unchanged=len(before),
        noop_output=capture.getvalue().strip(),partial_output_preserved_in_isolated_fixture=True,actual_existing_data_overwritten=False))
    print(json.dumps(dict(passed=True,completed_pairs=6000,existing_pair_files_unchanged=len(before))))
if __name__=='__main__':main()
