"""Local V2 fast human readability decisions; append-only, no GT assignment."""
import sys
sys.dont_write_bytecode=True
import argparse,json,hashlib,os,secrets,threading,urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from geo_tools import ROOT,lines
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
class Reviews:
    def __init__(self,root):
        self.root=Path(root).resolve();index=self.root/'candidate_regions.jsonl'
        if not index.exists():index=self.root/'regions.jsonl'
        self.tasks={r['sample_id']:r for r in lines(index)};self.lock=threading.RLock()
    def load(self,sid):
        if sid not in self.tasks:raise ValueError('unknown mother region')
        previous=None;d=dict(sample_id=sid,revision=0,verdict=None,note='',GT_qualified=False)
        for p in sorted((self.root/'readability'/sid).glob('v*.json')):
            d=json.loads(p.read_text());z=dict(d);saved=z.pop('content_sha256')
            if saved!=digest(z) or d['parent_sha256']!=previous:raise ValueError('readability hash chain changed')
            previous=saved
        return d
    def actor(self,a):
        if not isinstance(a,str) or not a.strip() or len(a)>100:raise ValueError('declare actual human identity')
        return a.strip()
    def commit(self,p):
        with self.lock:
            sid=p['sample_id'];old=self.load(sid)
            if p['expected_revision']!=old['revision']:raise ValueError('stale revision: reload task')
            if p.get('verdict') not in ['A','B','C','D']:raise ValueError('explicit readability judgment required')
            if any(k in p for k in ['GT_qualified','gt_status','points']):raise ValueError('readability cannot assign GT or points')
            d=dict(sample_id=sid,revision=old['revision']+1,parent_sha256=old.get('content_sha256'),actor=self.actor(p.get('actor')),verdict=p['verdict'],note=p.get('note',''),at=now(),GT_qualified=False,stage='human_readability_only')
            d['content_sha256']=digest(d);folder=self.root/'readability'/sid;folder.mkdir(parents=True,exist_ok=True);tmp=folder/('.pending_'+secrets.token_hex(8))
            with tmp.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
            try:os.link(tmp,folder/f"v{d['revision']:06d}.json")
            finally:tmp.unlink()
            return d
    def expose(self,p):
        with self.lock:
            self.load(p['sample_id']);actor=self.actor(p.get('actor'));source=p['source']
            if source not in ['common_fov_geographic_grid','geographic_coverage_map']:raise ValueError('unknown assistance')
            event=dict(sample_id=p['sample_id'],actor=actor,source=source,at=now(),independent_GT=False)
            folder=self.root/'readability';folder.mkdir(exist_ok=True)
            with (folder/'assistance_events.jsonl').open('a') as f:f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            return event
def make_server(root,port):
    store=Reviews(root);sessions=set();root=store.root
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,status=200,kind='application/json; charset=utf-8'):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
            self.send_response(status)
            for k,v in {'Content-Type':kind,'Content-Length':str(len(data)),'Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}.items():self.send_header(k,v)
            self.end_headers();self.wfile.write(data)
        def host(self):
            if self.headers.get('Host','').split(':')[0] not in ['localhost','127.0.0.1']:raise ValueError('local Host required')
        def do_GET(self):
            try:
                self.host();u=urllib.parse.urlparse(self.path);q=urllib.parse.parse_qs(u.query)
                if u.path=='/':
                    token=secrets.token_hex(32);sessions.add(token);self.send((root/'quick_review.html').read_text().replace('__TOKEN__',token).encode(),kind='text/html; charset=utf-8')
                elif u.path=='/api/tasks':self.send(list(store.tasks.values()))
                elif u.path=='/api/review':self.send(store.load(q['sample_id'][0]))
                elif u.path=='/api/progress':self.send(dict(reviewed=sum(store.load(sid)['revision']>0 for sid in store.tasks)))
                elif u.path.startswith('/file/'):
                    rel=urllib.parse.unquote(u.path[len('/file/'):]);p=(root/rel).resolve()
                    if root not in p.parents or p.suffix!='.png' or (rel not in ['atlas.png','candidate_atlas.png'] and not rel.startswith('regions/')):raise ValueError('image path not allowed')
                    self.send(p.read_bytes(),kind='image/png')
                else:self.send(dict(error='not found'),404)
            except (ValueError,KeyError,OSError) as e:self.send(dict(error=str(e)),400)
        def do_POST(self):
            try:
                self.host()
                if self.headers.get('X-SAR-Token') not in sessions:raise ValueError('reload page for CSRF token')
                origin=self.headers.get('Origin')
                if origin and urllib.parse.urlparse(origin).netloc!=self.headers.get('Host'):raise ValueError('cross-origin rejected')
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<10000:raise ValueError('invalid request size')
                p=json.loads(self.rfile.read(length))
                if self.path=='/api/review':self.send(store.commit(p))
                elif self.path=='/api/exposure':self.send(store.expose(p))
                else:self.send(dict(error='not found'),404)
            except (ValueError,KeyError,TypeError,OSError) as e:self.send(dict(error=str(e)),400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.reviews=store;return server
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=ROOT);p.add_argument('--port',type=int,default=8770);a=p.parse_args();s=make_server(a.root,a.port);print('V2快速判读 http://127.0.0.1:'+str(s.server_port)+'/',flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
