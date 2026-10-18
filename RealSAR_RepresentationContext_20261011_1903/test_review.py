"""Read-only integration check on actual five tasks; no fake human points."""
import sys
sys.dont_write_bytecode=True
import threading,json,urllib.request,urllib.parse,io
from pathlib import Path
from PIL import Image
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root.parent/'RealSAR_AutoLabels_20261011_1149'))
sys.path.insert(0,str(root.parent/'RealSAR_AutoLabels_20261011_1149/annotation_adapter'))
import serve
server=serve.make_server(root/'annotation_adapter',0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}';opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    tasks=json.load(opener.open(base+'/api/tasks'));assert len(tasks)==5
    for t in tasks:
        sid=urllib.parse.quote(t['sample_id']);a=json.load(opener.open(base+'/api/annotation?sample_id='+sid));assert a['points']==[] and a['revision']==0
        for side in [0,1]:
            im=Image.open(io.BytesIO(opener.open(base+f'/api/image?sample_id={sid}&side={side}').read()));assert im.size==(512,512)
    html=opener.open(base+'/').read().decode();assert '__TOKEN__' not in html
    assert not list((root/'annotation_adapter/annotations').rglob('v*.json'))
    assert not (root/'annotation_adapter/annotations/assistance_events.jsonl').exists()
    print('actual five tasks, ten native PNG routes, empty annotations, no exposure writes: PASS')
finally:server.shutdown();server.server_close();thread.join()
