#!/usr/bin/env python3
"""Real-image inference self-test; expected records are engineering outputs, never GT."""
import argparse
import json
import socket
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from sar_match.pipeline import Matcher,read_image,write_json,sha256
from sar_match.visualize import draw_matches
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--device',default='cpu');p.add_argument('--reference',type=Path);p.add_argument('--moving',type=Path)
    p.add_argument('--model',choices=['final','structure','pretrained'],default='final');p.add_argument('--budget',choices=[128,1024],type=int,default=1024)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs');p.add_argument('--offline-check',action='store_true');a=p.parse_args()
    if bool(a.reference)!=bool(a.moving):p.error('reference and moving must both be supplied')
    if a.offline_check:
        def forbidden(*args,**kwargs):raise RuntimeError('Network access forbidden during self-test')
        socket.socket.connect=forbidden;socket.create_connection=forbidden
    paths=[a.reference or ROOT/'examples/reference.png',a.moving or ROOT/'examples/moving.png']
    im=[read_image(x) for x in paths];matcher=Matcher(a.model,a.device,a.budget)
    rec,arrays=matcher.predict(*im)
    rec.update(image_A=paths[0].name,image_B=paths[1].name,input_sha256=[sha256(x) for x in paths],model=a.model,budget=a.budget,device=a.device)
    expected=ROOT/'examples/expected.json'
    if not a.reference and a.model=='final' and a.budget==1024 and expected.exists():
        e=json.loads(expected.read_text())
        e=e.get('backend_references',{}).get(matcher.device.type,e)
        differences={}
        for key in ['n_keypoints_A','n_keypoints_B','n_matches','n_inliers','accepted','reason']:
            if rec[key]!=e[key]: differences[key]={'actual':rec[key],'reference':e[key]}
        rec['reference_diagnostic_match']=not differences
        rec['reference_differences']=differences
    tag=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    import time
    out=a.output_dir.resolve()/f'smoke_{tag}_{time.time_ns()}';out.mkdir(parents=True,exist_ok=False)
    write_json(out/f'selftest_{tag}.json',rec);np.savez_compressed(out/f'prediction_{tag}.npz',**arrays)
    draw_matches(*im,arrays,rec,out/f'matches_{tag}.png',title=f'{a.model}: bundled real SAR, no independent GT')
    print(json.dumps({k:rec[k] for k in ['n_keypoints_A','n_keypoints_B','n_matches','n_inliers','accepted','reason','total_inference_seconds']},ensure_ascii=False));print('Output:',out)
    if rec.get('reference_differences'):
        print('Reference differences retained in output; inspect hardware/numerical compatibility:',rec['reference_differences'],file=sys.stderr)
        return 2
    return 0

if __name__=='__main__':sys.exit(main())
