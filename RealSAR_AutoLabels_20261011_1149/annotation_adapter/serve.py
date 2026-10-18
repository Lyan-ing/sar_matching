"""Reuse old annotation app with truthful geographic/pseudo layer provenance.

Only runtime module references change; original app and records stay read-only.
Legacy UI slot name sift_candidates carries pseudo points of the recorded actual
generator, not an assertion that all points were produced by SIFT.
"""
import sys
sys.dont_write_bytecode=True
import argparse,json,os,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import MANUAL
sys.path.insert(0,str(MANUAL/'annotation_app'))
import server as original
from store import Store,now

class AdapterStore(Store):
    def expose(self,sid,actor,source):
        with self.lock:
            t=self.task(sid);actor=self.actor(actor)
            if source not in ['geographic','sift_candidates']:raise ValueError('unknown assistance layer')
            f=self.root/'candidate_matches'/f'{sid}.json'
            if hashlib.sha256(f.read_bytes()).hexdigest()!=t['candidate_sha256']:raise ValueError('auxiliary file hash changed')
            data=json.loads(f.read_text());kind='georeference_correspondences' if source=='geographic' else 'pseudo_correspondences'
            event={'sample_id':sid,'actor':actor,'source':kind,'legacy_UI_slot':source,'generator':data.get('pseudo_generator') if source!='geographic' else 'GeoTIFF CRS/Affine pixel-center mapping','candidate_sha256':t['candidate_sha256'],'at':now(),'independent_GT':False}
            with self.audit.open('a') as out:out.write(json.dumps(event,ensure_ascii=False)+'\n');out.flush();os.fsync(out.fileno())
            return event

def make_server(root,port):
    original.Store=AdapterStore;original.APP=Path(root).resolve()
    return original.make_server(root,port)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(Path(__file__).resolve().parent));p.add_argument('--port',type=int,default=8766);a=p.parse_args();s=make_server(a.root,a.port);print(f'Geographic/pseudo diagnostic annotation: http://127.0.0.1:{s.server_port}',flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
