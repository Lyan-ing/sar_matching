"""Start only our CPU loopback service after mapping validation; preserve occupied ports."""
import sys,os,json,socket,subprocess,time,secrets,urllib.request
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]

def start():
    assert json.loads((ROOT/'mapping_validation.json').read_text())['passed']
    assert (ROOT/'display_cache.json').is_file()
    if (ROOT/'service.json').exists():raise FileExistsError('launch recorded; inspect own service, never start duplicate')
    # Actual bind check, not a stale snapshot. Server binding also fails if a race occurs.
    with socket.socket() as s:
        try:s.bind(('127.0.0.1',8773))
        except OSError as e:raise RuntimeError('8773 occupied; no listener stopped') from e
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    secret=secrets.token_hex(32);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',SAR_GEO_VERIFY_SECRET=secret)
    cmd=[sys.executable,'-B',str(ROOT/'code/server.py'),'--port','8773'];log=ROOT/'reports'/f'service_{stamp}.log'
    with log.open('x') as f:process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));url='http://127.0.0.1:8773/'
    for _ in range(100):
        if process.poll() is not None:raise RuntimeError('our service exited; inspect '+str(log))
        try:
            with opener.open(url+'api/health',timeout=.5) as r:health=json.load(r)
            break
        except OSError:time.sleep(.1)
    else:raise RuntimeError('health timeout; retain own PID '+str(process.pid))
    doc=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),pid=process.pid,
        process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
        command=cmd,url=url,port=8773,log=str(log),GPU=False,health=health,
        readonly_verification_secret=secret,notes='verification sessions cannot save exposure or triage')
    with (ROOT/'service.json').open('x') as f:json.dump(doc,f,indent=2)
    print(json.dumps({k:v for k,v in doc.items() if k!='readonly_verification_secret'},ensure_ascii=False))

if __name__=='__main__':start()
