"""Reuse minimal annotation UI ONLY after an actual A/B readability decision."""
import sys
sys.dont_write_bytecode=True
import argparse,json,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from geo_tools import ROOT,BASE,sha,save,jsonl
from readability_server import Reviews
def prepare(root=ROOT):
    root=Path(root).resolve();reviews=Reviews(root);qualified=[]
    for sid,r in reviews.tasks.items():
        d=reviews.load(sid)
        if d['verdict'] in ['A','B']:qualified.append((r,d))
    if not qualified:
        print('No actual A/B human readability decisions: no GT tasks or points created.');return None
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');out=root/'geometry_review'/('batch_'+stamp);out.mkdir(parents=True,exist_ok=False)
    source=BASE/'RealSAR_IndependentBenchmark_20261011_2015/annotation_adapter'
    for name in ['index.html','app.js','coordinates.js']:
        text=(source/name).read_text().replace('原生512坐标；新候选仍待独立确认。','原生坐标（两侧尺寸可不同）；仍待独立确认。').replace('原有5对独立核验','V2母区域独立核验')
        if name=='index.html':text=text.replace('aspect-ratio:1;touch-action','aspect-ratio:auto;touch-action')
        (out/name).write_text(text)
    (out/'annotation_tasks').mkdir();(out/'candidate_matches').mkdir();(out/'annotations').mkdir()
    tasks=[];evidence={}
    for r,d in qualified:
        sid=r['sample_id'];aux=dict(sample_id=sid,geographic=[],sift_candidates=[]);p=out/'candidate_matches'/f'{sid}.json';save(p,aux)
        t=dict(sample_id=sid,pair_id=r['pair_id'],chip_id='mother_000',site_name=r['site_name'],component=r['component'],split='diagnostic_development_seen',sensor=r['sensor'],look_pair_category=r['look_pair_category'],candidate_count=0,candidate_sha256=sha(p),annotation_readiness='human_readability_passed_pending_independent_geometry',
            reference_acquisition_id=r['views'][0]['acquisition_id'],moving_acquisition_id=r['views'][1]['acquisition_id'],reference_utc=r['acquisition_metadata']['views'][0]['utc'],moving_utc=r['acquisition_metadata']['views'][1]['utc'],visual_note='实际A/B快速判读通过，尚非GT。原生坐标；默认隐藏所有自动辅助；必须真实第二人重新点击。')
        for i in [0,1]:
            z=r['views'][i]['native_info'];t['width'+str(i)]=z['width'];t['height'+str(i)]=z['height'];t['image'+str(i)]=str(root/'regions'/sid/f'native_db{i}.png')
        tasks.append(t);evidence[sid]=d
    jsonl(out/'annotation_tasks/tasks.jsonl',tasks);save(out/'annotation_tasks/task_freeze.json',dict(task_sha256=sha(out/'annotation_tasks/tasks.jsonl'),tasks=len(tasks),human_readability_only=True))
    save(out/'readability_provenance.json',evidence)
    exposures=root/'readability/assistance_events.jsonl'
    if exposures.exists():
        events=[json.loads(s) for s in exposures.read_text().splitlines() if json.loads(s)['sample_id'] in evidence]
        if events:jsonl(out/'annotations/assistance_events.jsonl',events)
    save(out/'exposure_source_snapshot.json',dict(source=str(exposures),sha256=sha(exposures) if exposures.exists() else None,note='snapshot carried into annotation eligibility; refresh via serve_gt.py before launch'))
    print(out);return out
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);a=p.parse_args();prepare(a.root)
