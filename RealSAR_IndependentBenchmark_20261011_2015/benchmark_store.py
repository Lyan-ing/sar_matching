"""Reuse immutable Store/assistance provenance; only allow unknown candidate sigma."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'RealSAR_AutoLabels_20261011_1149/annotation_adapter'))
import serve as adapter
import store as original
from benchmark_quality import quality,validate_points
original.quality=quality;original.validate_points=validate_points
class Store(adapter.AdapterStore):
    """Append separately frozen batches without rewriting the original manifest."""
    def __init__(self,root):
        super().__init__(root)
        import hashlib,json,re
        base=self.root/'annotation_tasks'
        self.task_sources=[base/'tasks.jsonl']
        if (base/'task_freeze.json').exists():self.task_sources.append(base/'task_freeze.json')
        for folder in sorted((base/'batches').glob('*')):
            if not folder.is_dir() or folder.name.startswith('.'):continue
            manifest=folder/'tasks.jsonl';freeze=folder/'task_freeze.json'
            info=json.loads(freeze.read_text())
            if hashlib.sha256(manifest.read_bytes()).hexdigest()!=info['task_sha256']:
                raise ValueError('frozen batch manifest changed')
            rows=[json.loads(s) for s in manifest.read_text().splitlines()]
            if len(rows)!=info['task_count']:raise ValueError('batch task count mismatch')
            for row in rows:
                sid=row['sample_id']
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,120}',sid):raise ValueError('unsafe sample ID')
                if sid in self.tasks:raise ValueError('duplicate sample ID across batches')
                if row.get('batch_id')!=folder.name:raise ValueError('batch identity mismatch')
                if any(row.get(k)!=512 for k in ['height0','width0','height1','width1']):
                    raise ValueError('batch must use native 512 coordinates')
                self.tasks[sid]=row
            self.task_sources.extend([manifest,freeze])
def next_point_id(store,sid):
    # Deleted IDs remain reserved across the immutable history.
    high=0
    for f in (store.directory/sid).glob('v*.json'):
        for p in original.json.loads(f.read_text()).get('points',[]):
            pid=p['point_id']
            if pid.startswith('P') and pid[1:].isdigit():high=max(high,int(pid[1:]))
    return 'P'+str(high+1).zfill(4)
def response_doc(store,doc,review=False):
    result=original.json.loads(original.json.dumps(doc));result['next_point_id']=next_point_id(store,doc['sample_id'])
    if review:
        result['points']=[{k:p.get(k) for k in ['sample_id','point_id','role','landmark_type','note','annotator']}|
            {'review':{'reviewer':p['review']['reviewer'],'conclusion':p['review']['conclusion']} if p.get('review') else None} for p in doc['points']]
        q=doc['gt_quality']
        result['gt_quality']={k:q[k] for k in ['status','eligible_checks','eligible_controls','eligible_point_ids']}
    return result
