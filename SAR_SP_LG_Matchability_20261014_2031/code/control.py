from common import *
import signal
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('action',choices=['check','stop']);p.add_argument('variant',nargs='?',choices=['joint','control','frozen']);a=p.parse_args()
 for v in ([a.variant] if a.variant else ['joint','control','frozen']):
  f=output('status',folder=ROOT/'runs'/v)
  if not f.exists():print(v,'not launched');continue
  s=json.loads(f.read_text());pid=s['process']['pid'];print(v,s['state'],'step',s['global_step'],'PID',pid,'GPU',s['process']['physical_gpu'],'heartbeat',s['heartbeat'],'best',s.get('best',{}).get('step') if s.get('best') else None)
  if a.action=='stop' and s['state'] not in ['completed','stopped','failed']:
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ');assert str(ROOT/'code') in cmd and f'--variant {v}' in cmd;os.kill(pid,signal.SIGTERM);print('SIGTERM delivered safely',v)
if __name__=='__main__':main()
