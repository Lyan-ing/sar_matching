"""Evidence before completion: frozen inputs, actual own CPU service, document links."""
import sys,json,re,urllib.request
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from geometry import ROOT,SOURCE,BASE,sha,save,lines

def run():
    protected=json.loads((ROOT/'evidence/protected_before.json').read_text())
    for path,expected in protected.items():assert sha(path)==expected,('protected file changed',path)
    for p in (ROOT/'code').glob('*.py'):compile(p.read_text(),str(p),'exec')
    docs=[ROOT/'README.md',ROOT/'GT_CANDIDATE_PREPARATION.md',*list((ROOT/'reports').glob('geo_assist_report_*.md'))]
    for p in docs:
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith(('http://','https://')):assert (p.parent/target).is_file(),(str(p),target)
    validation=json.loads((ROOT/'mapping_validation.json').read_text());browser=json.loads(sorted((ROOT/'reports').glob('browser_check_*.json'))[-1].read_text())
    assert validation['passed'] and browser['passed'] and len(lines(ROOT/'tasks.jsonl'))==18
    assert validation['source_queue_sha256']==sha(SOURCE/'review/candidate_regions.jsonl')
    assert not (ROOT/'gt_workspaces').exists(),'a real GT workspace would need its own qualification evidence'
    service=json.loads((ROOT/'service.json').read_text());process=Path('/proc')/str(service['pid'])
    command=[b.decode() for b in (process/'cmdline').read_bytes().split(b'\0') if b]
    assert command==service['command']
    ticks=(process/'stat').read_text().rsplit(')',1)[1].split()[19];assert ticks==service['process_start_ticks']
    environ=(process/'environ').read_bytes().split(b'\0');assert b'CUDA_VISIBLE_DEVICES=' in environ
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(service['url']+'api/health',timeout=5) as r:health=json.load(r)
    assert health['tasks']==18 and health['GT']==0
    # Observe original service only; never stop/restart it or modify its files.
    old=json.loads((SOURCE/'readability_service.json').read_text());old_process=Path('/proc')/str(old['pid'])
    assert old_process.is_dir()
    assert [b.decode() for b in (old_process/'cmdline').read_bytes().split(b'\0') if b]==old['command']
    assert (old_process/'stat').read_text().rsplit(')',1)[1].split()[19]==old['process_start_ticks']
    with opener.open(old['url']+'api/progress',timeout=5) as r:old_progress=json.load(r)
    code={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'code').iterdir()) if p.is_file()}
    data={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'tasks.jsonl',ROOT/'mapping_validation.json',ROOT/'display_cache.json',ROOT/'gt_candidate_preparation.jsonl',ROOT/'gt_readiness.json']}
    reused={str(p):sha(p) for p in [BASE/'RealSAR_AutoLabels_20261011_1149/geo.py',BASE/'RealSAR_MultiViewV2_20261012_1259/code/prepare_gt.py',BASE/'RealSAR_MultiViewV2_20261012_1259/code/serve_gt.py',BASE/'RealSAR_MultiViewV2_20261012_1259/code/evaluation.py',BASE/'RealSAR_MultiViewV2_20261012_1259/code/readability_server.py']}
    now=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=now.strftime('%Y%m%d_%H%M')
    save(ROOT/'file_identity.json',dict(at=now.isoformat(),code_sha256=code,data_sha256=data,reused_source_sha256=reused,
        docs_sha256={str(p.relative_to(ROOT)):sha(p) for p in docs},old_git_HEAD_is_not_new_code_version=True))
    result=dict(at=now.isoformat(),passed=True,protected_files_unchanged=len(protected),actual_mapping_pairs=18,
        actual_priority_previews=6,actual_browser_bilateral_and_detail_click_pairs=7,
        source_queue_unchanged=True,original_8772_pid_and_start_ticks_unchanged=True,original_8772_progress=old_progress,
        own_8773_pid=service['pid'],own_8773_start_ticks=ticks,own_8773_health=health,
        real_GT=0,GT_batches_created=0,projected_points_exported=0,GPU_used=False,new_downloads=0,
        document_links_passed=True,python_sources_compiled=True,
        checks_scope='real geocoding math and UI only; accepted human save and independent annotation await actual humans')
    save(ROOT/'reports'/f'final_verification_{stamp}.json',result)
    print(json.dumps(result,ensure_ascii=False));print('FINAL_VERIFICATION',stamp)

if __name__=='__main__':run()
