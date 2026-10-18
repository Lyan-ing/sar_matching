"""Reuse existing minimal UI/GT policy; synchronize prior assistance exposure."""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,threading,time
from pathlib import Path
from geo_tools import ROOT,BASE,sha
sys.path.insert(0,str(BASE/'RealSAR_IndependentBenchmark_20261011_2015'))
import serve_review
from evaluation import qualified_quality
def sync_exposure(root,target,store):
    src=Path(root)/'readability/assistance_events.jsonl'
    if not src.exists():return
    with store.lock:
        known=set()
        if store.audit.exists():known={json.dumps(json.loads(s),sort_keys=True) for s in store.audit.read_text().splitlines()}
        new=[json.loads(s) for s in src.read_text().splitlines() if json.loads(s)['sample_id'] in store.tasks and json.dumps(json.loads(s),sort_keys=True) not in known]
        if new:
            with store.audit.open('a') as f:
                for event in new:f.write(json.dumps(event,ensure_ascii=False)+'\n')
                f.flush();os.fsync(f.fileno())
def serve(root,target,port):
    root,target=Path(root),Path(target)
    previous_response=serve_review.response_doc
    def response(store,doc,review=False):
        result=previous_response(store,doc,review);q=qualified_quality(doc,store.task(doc['sample_id']),root)
        result['gt_quality']={k:q[k] for k in ['status','eligible_checks','eligible_controls','eligible_point_ids']} if review else q
        return result
    serve_review.response_doc=response
    s=serve_review.make_server(target,port);sync_exposure(root,target,s.store)
    # New exposure after launch is tracked, not silently dropped. Existing points
    # are checked again by the V2 independent evaluator's exposure check below.
    stop=threading.Event()
    def watch():
        while not stop.wait(1):sync_exposure(root,target,s.store)
    t=threading.Thread(target=watch,daemon=True);t.start();print(f'V2原生几何核验 http://127.0.0.1:{s.server_port}/',flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:stop.set();s.server_close();t.join(timeout=2)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--tasks-root',required=True);p.add_argument('--port',type=int,default=8771);a=p.parse_args();serve(a.root,a.tasks_root,a.port)
