"""Reuse V2 GT tools only after real assisted A/B and two declared independent humans.
No projected points are exported. Exposure is carried and refreshed in a NEW workspace.
"""
import sys,json,os,time,threading,socket,argparse
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from geometry import ROOT,BASE,SOURCE,queue,lines,save,sha
from records import Records
V2=BASE/'RealSAR_MultiViewV2_20261012_1259'
sys.path.insert(0,str(V2/'code'))
import prepare_gt as existing_prepare
import serve_gt as existing_serve
from readability_server import digest as v2_digest

def selections():
    tasks={r['sample_id']:r for r in queue()};store=Records(ROOT/'assisted_triage',tasks)
    selected=[]
    for sid,row in tasks.items():
        doc=store.load(sid)
        if doc.get('verdict') in ['A','B'] and doc.get('recommend_independent_annotation') is True:
            assert doc['stage']=='assisted_triage' and doc['viewed_geographic_hint'] and not doc['GT_qualified']
            selected.append((row,doc))
    return selected

def exposures_for(target_rows):
    source_rows={r['sample_id']:r for r in queue()};source_rows.update({r['sample_id']:r for r in lines(V2/'candidate_regions.jsonl')})
    all_events=[]
    for path in [ROOT/'assisted_triage/exposure_events.jsonl',SOURCE/'review/readability/assistance_events.jsonl',V2/'readability/assistance_events.jsonl']:
        if path.is_file():all_events.extend(dict(event,source_file=str(path)) for event in lines(path))
    output=[]
    for row in target_rows:
        ids={v['acquisition_id'] for v in row['views']}
        for event in all_events:
            source=source_rows.get(event.get('sample_id'))
            if event.get('sample_id')!=row['sample_id'] and (source is None or not ids.intersection(v['acquisition_id'] for v in source['views'])):continue
            output.append(dict(sample_id=row['sample_id'],actor=event['actor'],at=event['at'],
                source='geographic_point_projection_and_coverage',original_sample_id=event['sample_id'],
                original_source_file=event['source_file'],shared_acquisition_exposure=True,independent_GT=False))
    return output

def sync_exposures(workspace):
    rows=lines(workspace/'candidate_regions.jsonl');path=workspace/'readability/assistance_events.jsonl'
    known={json.dumps(e,sort_keys=True) for e in lines(path)} if path.exists() else set()
    new=[]
    for e in exposures_for(rows):
        sig=json.dumps(e,sort_keys=True)
        if sig not in known:known.add(sig);new.append(e)
    if new:
        with path.open('a') as f:
            for e in new:f.write(json.dumps(e,ensure_ascii=False)+'\n')
            f.flush();os.fsync(f.fileno())

def prepare(args):
    selected=selections()
    if not selected:
        print('No actual human assisted A/B recommended: no workspace, GT batch or points created.');return None
    if not args.confirm_real_independent_people or not args.annotator or not args.reviewer:
        raise ValueError('declare actual independent annotator AND actual independent second reviewer; cannot invent roles')
    people=[args.annotator.strip(),args.reviewer.strip()]
    if min(map(len,people))==0 or len(set(people))!=2:raise ValueError('two distinct actual humans required')
    for row,doc in selected:
        exposed={e['actor'] for e in exposures_for([row])};exposed.add(doc['actor'])
        if exposed.intersection(people):raise ValueError('declared independent person already exposed for '+row['sample_id'])
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');out=ROOT/'gt_workspaces'/('batch_'+stamp);out.mkdir(parents=True,exist_ok=False)
    with (out/'candidate_regions.jsonl').open('x') as f:
        for row,doc in selected:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    for row,doc in selected:
        sid=row['sample_id'];folder=out/'regions'/sid;folder.mkdir(parents=True)
        for side in [0,1]:os.link(SOURCE/'review/regions'/sid/f'native_db{side}.png',folder/f'native_db{side}.png')
        # Preserve actual assisted semantics, actor, time, and original hash.
        # Adapt only the hash/one-version chain to the existing Reviews reader.
        bridged=dict(doc,original_assisted_content_sha256=doc['content_sha256'],original_assisted_revision=doc['revision'],
            original_assisted_record=str(ROOT/'assisted_triage'/sid/f"v{doc['revision']:06d}.json"),
            original_assisted_record_sha256=sha(ROOT/'assisted_triage'/sid/f"v{doc['revision']:06d}.json"),revision=1,parent_sha256=None)
        bridged.pop('content_sha256');bridged['content_sha256']=v2_digest(bridged)
        target=out/'readability'/sid;target.mkdir(parents=True);save(target/'v000001.json',bridged)
    save(out/'declared_roles.json',dict(annotator=people[0],reviewer=people[1],declaration='user-declared actual separate unexposed humans; actual clicks/re-clicks still required',
        independent_GT=False,projected_points_exported=0,other_prior_hints_must_be_declared=True))
    sync_exposures(out);batch=existing_prepare.prepare(out)
    save(out/'bridge_identity.json',dict(assisted_selection_only=True,independent_GT=False,projected_points_exported=0,
        geometry_batch=str(batch),source_queue_sha256=sha(SOURCE/'review/candidate_regions.jsonl')))
    print('WORKSPACE',out);print('GT_BATCH',batch);return out

def serve(args):
    workspace=Path(args.workspace).resolve()
    if ROOT/'gt_workspaces' not in workspace.parents:raise ValueError('only our prepared workspace may be served')
    bridge=json.loads((workspace/'bridge_identity.json').read_text());roles=json.loads((workspace/'declared_roles.json').read_text())
    rows=lines(workspace/'candidate_regions.jsonl')
    for row in rows:
        if {roles['annotator'],roles['reviewer']}.intersection(e['actor'] for e in exposures_for([row])):raise ValueError('a declared independent person has become exposed')
    with socket.socket() as s:s.bind(('127.0.0.1',args.port))
    sync_exposures(workspace);stop=threading.Event();lock=threading.Lock()
    def watch():
        while not stop.wait(1):
            with lock:sync_exposures(workspace)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:existing_serve.serve(workspace,Path(bridge['geometry_batch']),args.port)
    finally:stop.set();thread.join(timeout=2)

if __name__=='__main__':
    parser=argparse.ArgumentParser();commands=parser.add_subparsers(dest='command',required=True)
    p=commands.add_parser('prepare');p.add_argument('--annotator');p.add_argument('--reviewer');p.add_argument('--confirm-real-independent-people',action='store_true')
    p=commands.add_parser('serve');p.add_argument('--workspace',required=True);p.add_argument('--port',type=int,default=8774)
    args=parser.parse_args()
    if args.command=='prepare':prepare(args)
    else:serve(args)
