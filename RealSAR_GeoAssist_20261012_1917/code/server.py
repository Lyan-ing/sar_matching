"""Small loopback Geo Assist service; no model, raw-image writes or GT export."""
import sys
sys.dont_write_bytecode=True
import json,os,secrets,time,io,base64,urllib.parse,csv
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
from PIL import Image
from geometry import ROOT,queue,lines,View,point_result
from records import Records

def load_navigation_references(path,tasks):
    references={}
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            sid=row['sample_id'];side=int(row['side']);xy=[float(row['x']),float(row['y'])]
            if sid not in tasks or side not in [0,1] or not np.isfinite(xy).all():raise ValueError('invalid navigation reference')
            v=tasks[sid]['views'][side]
            span=int(row['span'])
            if span not in [128,256,512,1024]:raise ValueError('invalid recovered window span')
            if row['acquisition_id']!=v['acquisition_id'] or row['native_tif']!=v['native_tif']:raise ValueError('navigation reference acquisition mismatch')
            if not (0<=xy[0]<=v['width']-1 and 0<=xy[1]<=v['height']-1):raise ValueError('navigation reference outside Native image')
            if row['GT']!='False' or row['confirmed_human_point']!='False' or row['request_type']!='viewport_center_request':raise ValueError('navigation reference provenance mismatch')
            references.setdefault(sid,[]).append(dict(side=side,native_xy=xy,stage='log_recovered_navigation',
                log_timestamp=row['log_timestamp'],log_file=row['log_file'],log_line=int(row['log_line']),span_native=span,
                acquisition_id=v['acquisition_id'],confirmed_human_point=False,exact_correspondence=False,GT=False,
                may_be_geographic_prediction=True,matched_review_revision=None))
    return references

def make_server(port,verification_secret,records_root=None):
    assert json.loads((ROOT/'mapping_validation.json').read_text())['passed']
    tasks={t['sample_id']:t for t in lines(ROOT/'tasks.jsonl')};rows={r['sample_id']:r for r in queue()}
    views={sid:[View(row,i) for i in [0,1]] for sid,row in rows.items()}
    cache=json.loads((ROOT/'display_cache.json').read_text());arrays={}
    for key,e in cache['assets'].items():
        arrays[key]=np.load(e['array'],mmap_mode='r',allow_pickle=False)
        assert arrays[key].shape==(e['height'],e['width'])
    # Explicit alternate store is only for isolated automated checks, never an HTTP option.
    store=Records(records_root if records_root is not None else ROOT/'assisted_triage',tasks);sessions={}
    navigation_references={};navigation_error=None
    if records_root is None:
        try:navigation_references=load_navigation_references(ROOT/'exports/assisted_review_snapshot_20261013_201035/last_view_positions.csv',tasks)
        except (OSError,ValueError,KeyError) as e:navigation_error=str(e)

    class Handler(BaseHTTPRequestHandler):
        def send(self,value,status=200,kind='application/json; charset=utf-8',cacheable=False):
            data=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status)
            for k,v in {'Content-Type':kind,'Content-Length':str(len(data)),
                'Cache-Control':'private, max-age=86400' if cacheable else 'no-store',
                'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'}.items():self.send_header(k,v)
            self.end_headers();self.wfile.write(data)
        def check_host(self):
            if self.headers.get('Host','').split(':')[0] not in ['127.0.0.1','localhost']:raise ValueError('loopback Host required')
        def session(self):
            token=self.headers.get('X-SAR-Token')
            if token not in sessions:raise ValueError('reload Geo Assist page')
            return sessions[token]
        def authorized_task(self,sid):
            session=self.session()
            if sid not in tasks:raise ValueError('unknown task')
            if not session['readonly'] and sid not in session['actors']:raise ValueError('请先登记真实身份与地理辅助暴露')
            return session,tasks[sid]
        def do_GET(self):
            try:
                self.check_host();url=urllib.parse.urlparse(self.path);q=urllib.parse.parse_qs(url.query)
                if url.path=='/':
                    readonly=q.get('verify',[''])[0]==verification_secret and bool(verification_secret)
                    token=secrets.token_hex(32);sessions[token]=dict(readonly=readonly,actors={})
                    html=(ROOT/'code/index.html').read_text().replace('__TOKEN__',token).replace('__READONLY__','true' if readonly else 'false')
                    self.send(html.encode(),kind='text/html; charset=utf-8')
                elif url.path=='/app.js':self.send((ROOT/'code/app.js').read_bytes(),kind='text/javascript; charset=utf-8')
                elif url.path=='/api/health':
                    self.send(dict(service='SAR Geo Assist',port=self.server.server_port,tasks=len(tasks),
                        assisted_reviewed=sum(store.load(s)['revision']>0 for s in tasks),exposures=len(store.exposures()),GT=0,
                        recovered_navigation_samples=len(navigation_references),recovered_navigation_positions=sum(map(len,navigation_references.values())),
                        raw_tiff_per_click='bounded validity blocks only'))
                elif url.path=='/api/tasks':
                    self.session();self.send([{k:t[k] for k in ['sample_id','pair_id','site_name','priority','azimuth_delta_deg','days']} for t in tasks.values()])
                elif url.path in ['/api/task','/api/overview','/api/map','/api/crop','/api/record','/api/history']:
                    sid=q['sid'][0];session,task=self.authorized_task(sid)
                    if url.path=='/api/task':self.send(task)
                    elif url.path=='/api/record':self.send(store.load(sid))
                    elif url.path=='/api/history':
                        records=store.history(sid)
                        exposure_count=sum(e['sample_id']==sid for e in store.exposures())
                        self.send(dict(records=records,record_count=len(records),
                            latest_revision=records[-1]['revision'] if records else 0,
                            exposure_count=exposure_count,exposure_only=bool(exposure_count and not records),
                            navigation_references=navigation_references.get(sid,[]),navigation_reference_error=navigation_error))
                    else:
                        side=int(q['side'][0])
                        if side not in [0,1]:raise ValueError('invalid side')
                        if url.path=='/api/overview':
                            key=cache['task_view_index'][sid+':'+str(side)]
                            self.send(Path(cache['assets'][key]['overview']).read_bytes(),kind='image/png',cacheable=True)
                        elif url.path=='/api/map':
                            start=time.perf_counter();point=[float(q['x'][0]),float(q['y'][0])]
                            result=point_result(rows[sid],views[sid],side,point)
                            result['mapping_ms']=(time.perf_counter()-start)*1000;self.send(result)
                        else:
                            v=views[sid][side]
                            if 'bbox' in q:
                                bbox=json.loads(q['bbox'][0])
                                if not isinstance(bbox,list) or len(bbox)!=4 or any(isinstance(z,bool) or not isinstance(z,(int,float)) for z in bbox) or not np.isfinite(bbox).all():raise ValueError('invalid Native history window')
                                bx,by,bw,bh=bbox
                                if min(bx,by)<0 or min(bw,bh)<=0 or bx+bw>v.width or by+bh>v.height:raise ValueError('history window outside Native image')
                                x,y=bx+(bw-1)/2,by+(bh-1)/2
                                x0,y0=int(np.floor(bx)),int(np.floor(by));x1,y1=int(np.ceil(bx+bw)),int(np.ceil(by+bh))
                            else:
                                x,y=float(q['x'][0]),float(q['y'][0]);span=int(q.get('span',['256'])[0])
                                if span not in [128,256,512,1024]:raise ValueError('invalid magnifier span')
                                if not np.isfinite([x,y]).all():raise ValueError('finite crop centre required')
                                if not v.inside([[x,y]])[0]:self.send(dict(outside=True,requested_xy=[x,y],bbox_xywh=None));return
                                x0=max(0,int(np.floor(x-span/2)));y0=max(0,int(np.floor(y-span/2)))
                                x1=min(v.width,int(np.ceil(x+span/2)));y1=min(v.height,int(np.ceil(y+span/2)))
                            key=cache['task_view_index'][sid+':'+str(side)];a=arrays[key][y0:y1,x0:x1]
                            stream=io.BytesIO();Image.fromarray(np.asarray(a)).save(stream,format='PNG',compress_level=1)
                            self.send(dict(outside=False,bbox_xywh=[x0,y0,x1-x0,y1-y0],requested_xy=[x,y],
                                png='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode(),
                                geographic_resampling=False,exact_native_display_pixels=True))
                else:self.send(dict(error='not found'),404)
            except (ValueError,KeyError,TypeError,OSError,AssertionError) as e:self.send(dict(error=str(e)),400)
        def do_POST(self):
            try:
                self.check_host();session=self.session()
                if session['readonly']:self.send(dict(error='automated read-only verification cannot save exposure or human triage'),403);return
                origin=self.headers.get('Origin')
                if origin and urllib.parse.urlparse(origin).netloc!=self.headers.get('Host'):raise ValueError('cross-origin write rejected')
                length=int(self.headers.get('Content-Length',0))
                if not 0<length<=100000:raise ValueError('invalid request length')
                p=json.loads(self.rfile.read(length));sid=p['sample_id'];actor=store.actor(p)
                if self.path=='/api/exposure':
                    event=store.expose(sid,actor);session['actors'][sid]=actor;self.send(event)
                elif self.path=='/api/triage':
                    self.authorized_task(sid);doc=store.commit(p,session['actors'][sid]);history=store.history(sid)
                    if not any(d['content_sha256']==doc['content_sha256'] for d in history):raise OSError('saved version could not be read back')
                    self.send(dict(doc,disk_confirmed=True,history_count=len(history)))
                else:self.send(dict(error='not found'),404)
            except (ValueError,KeyError,TypeError,OSError) as e:self.send(dict(error=str(e)),400)

    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.daemon_threads=True
    return server

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8773);args=parser.parse_args()
    secret=os.environ.get('SAR_GEO_VERIFY_SECRET','')
    server=make_server(args.port,secret);print('Geo Assist http://127.0.0.1:'+str(server.server_port)+'/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
