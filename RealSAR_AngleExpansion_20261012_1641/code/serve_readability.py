"""Isolated native-only human review, using existing append-only V2 service."""
import sys
sys.dont_write_bytecode=True
import argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'RealSAR_MultiViewV2_20261012_1259/code'))
from readability_server import make_server

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8772);a=p.parse_args()
    server=make_server(ROOT/'review',a.port)
    print(f'Native human review http://127.0.0.1:{server.server_port}/',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()
