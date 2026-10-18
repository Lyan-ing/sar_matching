import json
from pathlib import Path
import pytest
from validate_release import qa

def test_legacy_umbra_identity_requires_task_sidecar_and_shape(tmp_path):
    import hashlib,json
    from validate_release import verify_umbra_identity
    p=tmp_path/'sidecar.json';p.write_text(json.dumps({'collects':[{'id':'collect-a','taskId':'task-a'}],'derivedProducts':{'GEC':[{'numRows':512,'numColumns':512}]}}))
    src={'acquisition_id':'collect-a','metadata_path':str(p),'metadata_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'asset_url':'https://vendor.test/task-a/source/source_GEC.tif','metadata_url':'https://vendor.test/task-a/source/source_METADATA.json'}
    info={'tags':{},'height':512,'width':512}
    assert verify_umbra_identity(src,info,src['asset_url']).startswith('legacy_co_located')
    with pytest.raises(AssertionError):verify_umbra_identity(src,{**info,'width':513},src['asset_url'])
    with pytest.raises(AssertionError):verify_umbra_identity(src,{**info,'tags':{'COLLECT_ID':'other-collect'}},src['asset_url'])

def test_empty_download_cannot_pass_engineering_acceptance(tmp_path):
    run=tmp_path/'run';data=run/'data';data.mkdir(parents=True)
    (run/'sources.jsonl').write_text('');(data/'candidates.jsonl').write_text('')
    with pytest.raises(SystemExit) as exc:qa(run,data,run/'quality')
    assert exc.value.code==2
    acceptance=json.loads((run/'quality/ACCEPTANCE.json').read_text())
    assert acceptance['checked_crop_pairs']==0 and not acceptance['engineering_QA_passed']
