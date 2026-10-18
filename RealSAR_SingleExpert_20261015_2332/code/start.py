"""Start our own CPU-only loopback service without touching occupied listeners."""
from expert import *
import socket,subprocess,time,secrets,urllib.request

def main():
 config=load_config();port=config['port'];url=f'http://127.0.0.1:{port}/';opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 try:
  with opener.open(url+'api/health',timeout=.5) as r:health=json.load(r)
  if health.get('service')=='RealSAR Single Expert':print('已在运行：'+url,flush=True);return
  raise RuntimeError('端口被其他服务占用，未停止任何进程')
 except OSError:pass
 with socket.socket() as s:
  # Match ThreadingHTTPServer: TIME_WAIT after our restart is not a live listener.
  s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',port))
 t=stamp();token=secrets.token_hex(32);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',SAR_EXPERT_READONLY_TOKEN=token)
 log=ROOT/f'service_{t}.log'
 if log.exists():log=ROOT/f'service_{uuid.uuid4().hex[:8]}_{t}.log'
 command=[sys.executable,'-B',str(ROOT/'code/server.py'),'--port',str(port)]
 with log.open('x') as f:process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
 for _ in range(100):
  if process.poll() is not None:raise RuntimeError('本服务启动失败；请看'+str(log))
  try:
   with opener.open(url+'api/health',timeout=.4) as r:health=json.load(r)
   break
  except OSError:time.sleep(.1)
 else:raise RuntimeError('健康检查超时；仅保留本服务PID '+str(process.pid))
 runtime=ROOT/f'SERVICE_{process.pid}_{t}.json';record=dict(at=now(),url=url,port=port,pid=process.pid,command=command,log=str(log),readonly_verification_token=token,GPU=False)
 with runtime.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
 runtime.chmod(0o600);print(json.dumps({k:v for k,v in record.items() if k!='readonly_verification_token'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
