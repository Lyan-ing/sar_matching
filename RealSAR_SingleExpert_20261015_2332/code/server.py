"""Small loopback service: reuse native viewer math, persist actual single-user input."""
from expert import *
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlparse,parse_qs
import argparse,secrets

def make_server(port=8774,readonly_token='',store_root=None):
 data=Dataset();store=Store(data,store_root);renderer=Renderer(data);state={'export':None,'export_error':None};statics={'/':'index.html','/app.js':'app.js','/coordinates.js':'coordinates_20261013_2227.js'}
 if data.config.get('legacy_review_html'):
  statics['/legacy-review/']=data.config['legacy_review_html']
  statics['/legacy-review']=data.config['legacy_review_html']
 class Handler(BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def check_host(self):
   if self.headers.get('Host') not in [f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}']:raise ValueError('只接受本机服务地址')
  def readonly(self):return bool(readonly_token and ('expert_readonly='+readonly_token) in self.headers.get('Cookie',''))
  def send(self,obj,status=200,kind='application/json; charset=utf-8',cookie=None,cache=False):
   b=obj if isinstance(obj,bytes) else json.dumps(obj,ensure_ascii=False,allow_nan=False).encode();self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(b)));self.send_header('Cache-Control','private,max-age=3600' if cache else 'no-store');self.send_header('X-Content-Type-Options','nosniff')
   if cookie:self.send_header('Set-Cookie',cookie)
   self.end_headers();self.wfile.write(b)
  def do_GET(self):
   try:
    self.check_host();u=urlparse(self.path);q=parse_qs(u.query)
    if u.path in statics:
     cookie=None
     if readonly_token and q.get('readonly',[''])[0]==readonly_token:cookie='expert_readonly='+readonly_token+'; Path=/; HttpOnly; SameSite=Strict'
     name=statics[u.path];self.send((ROOT/'code'/name).read_bytes(),kind='text/html; charset=utf-8' if name.endswith('.html') else 'text/javascript; charset=utf-8',cookie=cookie)
    elif u.path=='/api/health':self.send(dict(service='RealSAR Single Expert',tasks=len(data.tasks),high_priority_B=40,pilot30=len(data.pilot),port=self.server.server_port,original_correspondences=sum(1 for _ in store.original_points()),GT_qualified_two_person=0,readonly=self.readonly()))
    elif u.path=='/api/tasks':
     tasks=[]
     for t in data.tasks.values():
      d=store.get(t['sample_id'],'native' if t.get('native_only') else 'patch');r={k:v for k,v in t.items() if k not in ['thumbnail','source_manifest']};r.update(task_status=d['task_status'],saved_revision=d['revision']);tasks.append(r)
     self.send(dict(tasks=tasks,pilot30=data.pilot,readonly=self.readonly(),latest_export=state['export']))
    elif u.path=='/api/scene':
     sid=q['sid'][0];mode=q.get('mode',['patch'])[0];ids=q.get('frames',[None])[0];r=store.scene(sid,mode,json.loads(ids) if ids else None);r['readonly']=self.readonly();self.send(r)
    elif u.path=='/api/thumbnail':self.send(Path(data.tasks[q['sid'][0]]['thumbnail']).read_bytes(),kind='image/jpeg',cache=True)
    elif u.path=='/api/image':
     fid=q['frame'][0];box=json.loads(q['box'][0]) if 'box' in q else None
     if box is not None and (not isinstance(box,list) or len(box)!=4 or any(type(x)!=int for x in box)):raise ValueError('显示窗口必须为整数像素')
     self.send(renderer.png(fid,box),kind='image/png',cache=True)
    elif u.path=='/api/history':
     scope=data.scope(q['sid'][0],q.get('mode',['patch'])[0]);folder=store.root/digest(scope)[:24]
     # Only summaries; repeat mode must not expose original coordinates indirectly.
     self.send([dict(revision=d['revision'],recorded_at=d['recorded_at'],points=len(d['points']),task_status=d['task_status']) for d in (json.loads(p.read_text()) for p in sorted(folder.glob('v*.json')))])
    elif u.path=='/api/export':self.send(dict(latest_export=state['export'],error=state['export_error']))
    elif u.path=='/api/export_file':
     if not state['export']:raise ValueError('尚无导出')
     key=q.get('kind',['manifest'])[0]
     if key not in ['manifest','coverage','original_points']:raise ValueError('未知导出类型')
     path=Path(state['export'][key]);self.send(path.read_bytes(),kind='text/csv; charset=utf-8' if key=='coverage' else 'application/x-ndjson; charset=utf-8')
    else:self.send(dict(error='not found'),404)
   except (ValueError,KeyError,TypeError,OSError,AssertionError) as e:self.send(dict(error=str(e)),400)
  def do_POST(self):
   try:
    self.check_host()
    if self.readonly():self.send(dict(error='自动只读验收禁止保存人工记录和GT'),403);return
    origin=self.headers.get('Origin')
    if origin and urlparse(origin).netloc!=self.headers.get('Host'):raise ValueError('拒绝跨站写入')
    if not self.headers.get('Content-Type','').startswith('application/json'):raise ValueError('需要JSON请求')
    n=int(self.headers.get('Content-Length',0))
    if not 0<n<=2_000_000:raise ValueError('请求大小无效')
    p=json.loads(self.rfile.read(n))
    if self.path=='/api/save':
     previous=store.get(p['sample_id'],p['mode']);doc=store.commit(p)
     changed=[point_hash(q) for q in previous['points']]!=[point_hash(q) for q in doc['points']] or previous['task_status']!=doc['task_status']
     if changed:
      try:state['export']=store.export();state['export_error']=None
      except Exception as e:state['export_error']=str(e)
     # Do not send server-side repeat baselines back to the UI.
     doc.pop('repeat_baselines',None);self.send(dict(document=doc,disk_confirmed=True,revision=doc['revision'],history_count=doc['revision'],export=state['export'],export_error=state['export_error'],quality=store.scene(p['sample_id'],p['mode'],doc['frame_ids'])['quality']))
    elif self.path=='/api/export':
     state['export']=store.export();state['export_error']=None;self.send(state['export'])
    else:self.send(dict(error='not found'),404)
   except (ValueError,KeyError,TypeError,OSError,AssertionError) as e:self.send(dict(error=str(e)),400)
 server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True;server.data=data;server.store=store;server.state=state
 files=sorted(Path(data.config['exports']).glob('export_*/EXPORT_*.json'))
 if files:
  last=files[-1];state['export']=dict(json.loads(last.read_text()),metadata=str(last))
 return server

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--port',type=int,default=8774);a=ap.parse_args();server=make_server(a.port,os.environ.get('SAR_EXPERT_READONLY_TOKEN',''));print(f'Single Expert http://127.0.0.1:{server.server_port}/',flush=True)
 try:server.serve_forever()
 except KeyboardInterrupt:pass
 finally:server.server_close()
