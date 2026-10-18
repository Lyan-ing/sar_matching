"""Loopback annotation/review pages; legacy immutable store and GT thresholds."""
import sys
sys.dont_write_bytecode=True
import argparse,json,secrets,urllib.parse
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from benchmark_store import Store,response_doc
ROOT=Path(__file__).resolve().parent

def make_server(root,port=8768):
    app=Path(root).resolve();store=Store(app);(app/'gt_quality').mkdir(exist_ok=True);sessions={}
    class Handler(BaseHTTPRequestHandler):
        def send(self,data,status=200,kind='application/json; charset=utf-8'):
            if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            for k,v in {'Content-Type':kind,'Content-Length':str(len(data)),'Cache-Control':'no-store','X-Content-Type-Options':'nosniff','X-Frame-Options':'DENY'}.items():self.send_header(k,v)
            self.end_headers();self.wfile.write(data)
        def guarded(self):
            if self.headers.get('Host','').split(':')[0] not in ['localhost','127.0.0.1']:raise ValueError('localhost Host required')
        def is_review(self):
            token=self.headers.get('X-SAR-Token')
            if token not in sessions:raise ValueError('missing session CSRF token; reload page')
            return sessions[token]
        def do_GET(self):
            try:
                self.guarded();u=urllib.parse.urlparse(self.path);q=urllib.parse.parse_qs(u.query)
                if u.path=='/':
                    review=q.get('review',['0'])[0]=='1';token=secrets.token_hex(32);sessions[token]=review
                    self.send((app/'index.html').read_text().replace('__TOKEN__',token).encode(),kind='text/html; charset=utf-8')
                elif u.path in ['/app.js','/coordinates.js']:self.send((app/u.path[1:]).read_bytes(),kind='text/javascript; charset=utf-8')
                elif u.path=='/api/tasks':self.send(list(store.tasks.values()))
                elif u.path=='/api/annotation':self.send(response_doc(store,store.load(q['sample_id'][0]),self.is_review()))
                elif u.path=='/api/image':
                    t=store.task(q['sample_id'][0]);side=q['side'][0]
                    if side not in ['0','1']:raise ValueError('invalid view')
                    self.send(Path(t['image'+side]).read_bytes(),kind='image/png')
                else:self.send({'error':'not found'},404)
            except (ValueError,KeyError,OSError) as e:self.send({'error':str(e)},400)
        def do_POST(self):
            try:
                self.guarded();review=self.is_review();origin=self.headers.get('Origin')
                if origin and urllib.parse.urlparse(origin).netloc!=self.headers.get('Host'):raise ValueError('cross-origin request rejected')
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=2_000_000:raise ValueError('invalid payload size')
                p=json.loads(self.rfile.read(length))
                if self.path=='/api/save':
                    if review:raise ValueError('review page cannot edit first-person evidence')
                    self.send(response_doc(store,store.save(p)))
                elif self.path=='/api/review':
                    if not review:raise ValueError('open the separate independent review page')
                    self.send(response_doc(store,store.review(p),True))
                elif self.path=='/api/assistance':
                    t=store.task(p['sample_id']);event=store.expose(p['sample_id'],p.get('actor'),p['source'])
                    f=app/'candidate_matches'/f"{p['sample_id']}.json"
                    if __import__('hashlib').sha256(f.read_bytes()).hexdigest()!=t['candidate_sha256']:raise ValueError('candidate file changed')
                    data=json.loads(f.read_text())
                    if data.get('sample_id')!=t['sample_id']:raise ValueError('candidate sample ID mismatch')
                    self.send({'event':event,'points':data[p['source']],'warning':'辅助暴露已永久记录；该人员在此任务上不再具有独立GT资格。'})
                else:self.send({'error':'not found'},404)
            except (ValueError,KeyError,OSError,TypeError,json.JSONDecodeError) as e:self.send({'error':str(e)},400)
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.store=store;return server

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8768);p.add_argument('--root',default=str(ROOT/'annotation_adapter'));a=p.parse_args()
    s=make_server(a.root,a.port);print(f'标注 http://127.0.0.1:{s.server_port}/ | 独立复核 http://127.0.0.1:{s.server_port}/?review=1',flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
