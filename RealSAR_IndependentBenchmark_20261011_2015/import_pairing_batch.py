"""Import the eight pairing-audit candidates once; never overwrite prior tasks."""
import sys
sys.dont_write_bytecode=True
import argparse,hashlib,json,os,secrets
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from PIL import Image
import rasterio
from benchmark_store import Store

ROOT=Path(__file__).resolve().parent
PRIORITY=['Capella_0006__chip_001','S1SF_EXTENSION_000__chip_000',
          'Umbra_0056__chip_002','Umbra_0068__chip_001']
BATCH='pairing_audit_20261011_2349'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_new(p,value):
    with p.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def run(root=ROOT/'annotation_adapter',source=ROOT.parent/'RealSAR_PairingAudit_20261011_2349/candidates.json'):
    root,source=Path(root).resolve(),Path(source).resolve()
    store=Store(root);candidates=json.loads(source.read_text())
    if len(candidates)!=8:raise ValueError('Expected exactly eight fixed audit candidates')
    by_id={r['sample_id']:r for r in candidates}
    if len(by_id)!=8 or any(s not in by_id for s in PRIORITY):raise ValueError('Unexpected candidate IDs')
    order=PRIORITY+[s for s in by_id if s not in PRIORITY]
    target=root/'annotation_tasks/batches'/BATCH
    if target.exists():raise FileExistsError('Batch already imported; do not regenerate')
    if any(s in store.tasks for s in order):raise ValueError('Candidate already exists in another batch')
    rows=[];aux={};inputs={str(source):sha(source)}
    for n,sid in enumerate(order):
        c=by_id[sid];r=dict(c['source_row'])
        if c['GT_qualified'] is not False or c['static_physical_correspondence']!='pending':
            raise ValueError('Audit candidates must still await independent confirmation')
        if sid!=r['pair_id']+'__'+r['chip_id']:raise ValueError('Candidate source identity mismatch')
        for side,prefix in enumerate(['reference','moving']):
            png=Path(r[prefix+'_native_png']);tif=Path(r[prefix+'_native_geotiff'])
            with Image.open(png) as im:
                im.load()
                if im.size!=(512,512):raise ValueError('Not a native 512 PNG')
            with rasterio.open(tif) as ds:
                if (ds.width,ds.height)!=(512,512) or not ds.crs:raise ValueError('Invalid native GeoTIFF')
            inputs[str(png)]=sha(png);inputs[str(tif)]=sha(tif)
            r['image'+str(side)]=str(png);r['width'+str(side)]=r['height'+str(side)]=512
        payload=dict(sample_id=sid,geographic=[],sift_candidates=[],
                     note='No auxiliary points imported. New task awaits independent human confirmation.')
        data=(json.dumps(payload,ensure_ascii=False,indent=2)+'\n').encode()
        path=root/'candidate_matches'/f'{sid}.json'
        if path.exists():raise FileExistsError('Existing auxiliary record must not be overwritten')
        aux[path]=data
        r.update(sample_id=sid,batch_id=BATCH,batch_label='配对核查候选 · 8对（待独立确认）',
                 batch_default=True,batch_priority=n,annotation_readiness='pending_independent_confirmation',
                 gt_status='unverified',GT_qualified=False,independent_check_points=0,candidate_count=0,
                 candidate_sha256=hashlib.sha256(data).hexdigest(),
                 visual_note='待独立确认；'+c['selection_reason']+'。自动筛选不证明物理对应；可不加点而结束为无法可靠确认。')
        rows.append(r)
    target.parent.mkdir(exist_ok=True)
    staging=target.parent/('.pending_'+secrets.token_hex(8));staging.mkdir()
    manifest=staging/'tasks.jsonl'
    with manifest.open('x') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    write_new(staging/'task_freeze.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        batch_id=BATCH,task_count=8,task_sha256=sha(manifest),source=str(source),input_sha256=inputs,
        priority=order,coordinate_source='native 512 PNG paired with native GeoTIFF',
        manual_evidence_imported=False,all_tasks_pending=True))
    # Publish batch last. On failure partial files are retained, never replaced.
    for p,data in aux.items():
        with p.open('xb') as f:f.write(data)
    os.rename(staging,target)
    final=Store(root)
    print(json.dumps(dict(batch=BATCH,added=8,total=len(final.tasks),priority=order,
                         new_statuses={s:final.load(s)['gt_quality']['status'] for s in order}),ensure_ascii=False,indent=2))
    return order
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT/'annotation_adapter');p.add_argument('--source',default=ROOT.parent/'RealSAR_PairingAudit_20261011_2349/candidates.json');a=p.parse_args();run(a.root,a.source)
