"""Existing audited annotation tool, independent new task/annotation directory."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
root=Path(__file__).resolve().parent
old=root.parent/'RealSAR_AutoLabels_20261011_1149'
sys.path.insert(0,str(old));sys.path.insert(0,str(old/'annotation_adapter'))
import serve
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8767);a=p.parse_args();s=serve.make_server(root/'annotation_adapter',a.port)
    print(f'Independent review (automatic layers hidden): http://127.0.0.1:{s.server_port}',flush=True)
    try:s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close()
