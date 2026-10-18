"""Official fixed weights, resumable validated HTTP ranges; no model substitutes."""
import os,json,hashlib,time,urllib.request,concurrent.futures,signal,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
URLS={'roma_outdoor.pth':'https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth',
      'dinov2_vitl14_pretrain.pth':'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def fetch(name,url):
    p=ROOT/'weights'/name;start=time.monotonic()
    req=urllib.request.Request(url,headers={'Range':'bytes=0-1023'})
    with urllib.request.urlopen(req,timeout=30) as r:
        if r.status!=206:raise ValueError('range access not supported')
        total=int(r.headers['Content-Range'].split('/')[-1]);r.read()
    if p.exists():
        if p.stat().st_size!=total or not zipfile.is_zipfile(p):raise ValueError('existing weight is incomplete')
        return dict(name=name,official_URL=url,status='downloaded',sha256=sha(p),bytes=total,existing_official_file=True)
    tmp=p.with_suffix('.partial');offset=tmp.stat().st_size if tmp.exists() else 0
    if offset>total:raise ValueError('partial larger than official total')
    if offset:
        for lo,hi in [(0,min(offset-1,1023)),(max(0,offset-1024),offset-1)]:
            req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}'})
            with urllib.request.urlopen(req,timeout=60) as r:
                if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{total}':raise ValueError('resume prefix validation failed')
                data=r.read()
            with tmp.open('rb') as f:f.seek(lo);existing=f.read(hi-lo+1)
            if data!=existing:raise ValueError('resume prefix differs from official bytes')
    chunks=ROOT/'weights'/('chunks_'+name);chunks.mkdir(exist_ok=True)
    deadline=start+1200
    def part(spec):
        lo,hi=spec;f=chunks/f'{lo}_{hi}';size=hi-lo+1
        if f.exists() and f.stat().st_size==size:return f
        req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}'})
        with urllib.request.urlopen(req,timeout=60) as r:
            if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {lo}-{hi}/{total}':raise ValueError('range response does not match requested bytes')
            with f.open('wb') as o:
                while True:
                    b=r.read(2**20)
                    if not b:break
                    o.write(b)
                    if time.monotonic()>deadline:raise TimeoutError('remaining download budget20minutes')
        if f.stat().st_size!=size:raise ValueError('truncated chunk')
        print(name,'chunk',lo,hi,'complete',flush=True);return f
    specs=[(lo,min(total-1,lo+16*1024**2-1)) for lo in range(offset,total,16*1024**2)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(part,specs))
    with tmp.open('ab') as out:
        for f in parts:
            with f.open('rb') as src:
                for b in iter(lambda:src.read(2**20),b''):out.write(b)
    if tmp.stat().st_size!=total:raise ValueError('assembled byte count mismatch')
    if not zipfile.is_zipfile(tmp):raise ValueError('assembled weight invalid zip')
    tmp.rename(p)
    return dict(name=name,official_URL=url,status='downloaded',sha256=sha(p),bytes=total,
                range_resume_prefix_bytes=offset,range_parallel_workers=8,elapsed_seconds=time.monotonic()-start)
if __name__=='__main__':
    (ROOT/'weights').mkdir(exist_ok=True);results=[]
    for n,u in URLS.items():
        try:r=fetch(n,u)
        except Exception as e:r=dict(name=n,official_URL=u,status='failed',error=repr(e))
        results.append(r)
    with (ROOT/'weights/identity.json').open('x') as f:json.dump(results,f,indent=2);f.write('\n')
    print(json.dumps(results,indent=2),flush=True)
