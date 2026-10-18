"""Read-only local gallery. Does not expose or write the annotation store."""
import argparse, functools
from pathlib import Path
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8770);args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    server=ThreadingHTTPServer(('127.0.0.1',args.port),functools.partial(SimpleHTTPRequestHandler,directory=str(root)))
    print(f'Gallery: http://127.0.0.1:{args.port}/ | root={root}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:server.server_close()
