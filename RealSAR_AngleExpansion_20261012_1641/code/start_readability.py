"""Launch only this task's loopback CPU service; preserve other listeners."""
import sys
sys.dont_write_bytecode=True
import os,json,socket,subprocess,time,urllib.request
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]

def start():
    assert (ROOT/'review/task_freeze.json').exists()
    if (ROOT/'readability_service.json').exists():raise FileExistsError('Service launch already recorded; check its PID/port, do not duplicate')
    ports=[]
    for port in [8772,8773,8774]:
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1',port))!=0:ports.append(port)
    if not ports:raise RuntimeError('All new-task ports occupied; no existing listener touched')
    port=ports[0];stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    log=ROOT/'reports'/f'readability_service_{stamp}.log';env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    cmd=[sys.executable,'-B',str(ROOT/'code/serve_readability.py'),'--port',str(port)]
    with log.open('x') as f:process=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));url=f'http://127.0.0.1:{port}/'
    for _ in range(50):
        if process.poll() is not None:raise RuntimeError('Service failed; inspect '+str(log))
        try:
            with opener.open(url+'api/progress',timeout=1) as r:progress=json.load(r)
            break
        except OSError:time.sleep(.1)
    else:raise RuntimeError('Service health timeout; PID '+str(process.pid)+' retained for inspection')
    launch=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),pid=process.pid,
        process_start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split(') ',1)[1].split()[19],
        command=cmd,port=port,url=url,log=str(log),GT=0,GPU=False,progress=progress)
    with (ROOT/'readability_service.json').open('x') as f:json.dump(launch,f,indent=2)
    print(json.dumps(launch,indent=2))

if __name__=='__main__':start()
