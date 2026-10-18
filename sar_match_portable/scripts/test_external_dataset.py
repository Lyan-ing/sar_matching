#!/usr/bin/env python3
"""Run the locked SAR matcher on external PNG views; no external GT is assumed."""
import argparse
import csv
import json
import os
import platform
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from sar_match.external import scan,make_pairs,category,annotation_diagnostics,pixel_consistency
from sar_match.pipeline import Matcher,read_image,write_json,clean,load_config,sha256
from sar_match.visualize import draw_matches
import numpy as np
import torch
import cv2


def stamp(): return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')


def write_csv(path, rows):
    columns=list(dict.fromkeys(k for r in rows for k in r)) or ['scene','status']
    with Path(path).open('x',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in clean(row).items()})


def summarize(rows):
    n=len(rows)
    summary=dict(pairs=n,accepted=sum(bool(r.get('accepted')) for r in rows),exceptions=sum(r.get('failure_type')=='program_exception' for r in rows),
                 geometry_acceptance_rate=sum(bool(r.get('accepted')) for r in rows)/n if n else None,
                 failure_types=dict(Counter(r['failure_type'] for r in rows)),rejection_reasons=dict(Counter(r['reason'] for r in rows if not r['accepted'])))
    for key in ['n_keypoints_A','n_keypoints_B','n_matches','n_inliers','inlier_ratio','inlier_fit_rmse_px','corner_mean_displacement_px','total_inference_seconds','feature_seconds','matching_seconds','geometry_seconds']:
        values=[r[key] for r in rows if isinstance(r.get(key),(float,int))]
        summary[key+'_mean']=float(np.mean(values)) if values else None
        summary[key+'_median']=float(np.median(values)) if values else None
        summary[key+'_available_pairs']=len(values)
    for side in ('A','B'):
        for metric in ('grid_coverage','hull_fraction','isotropy'):
            values=[r.get('inlier_support_'+side,{}).get(metric) for r in rows]
            values=[v for v in values if isinstance(v,(float,int))]
            summary[f'inlier_{side}_{metric}_mean']=float(np.mean(values)) if values else None
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--mode',choices=['smoke','all'],default='smoke')
    p.add_argument('--scene',default='all')
    p.add_argument('--pair-type',choices=['all','same_inc','same_az','both_different'],default='all')
    p.add_argument('--expected-inc',default='20,30,45');p.add_argument('--expected-az',default='0,60,120,180,240,300')
    p.add_argument('--device',default='cpu');p.add_argument('--model',choices=['final','structure','pretrained'],default='final')
    p.add_argument('--budget',choices=[128,1024],type=int,default=1024)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    p.add_argument('--visualizations',choices=['representative','all','none'],default='representative')
    p.add_argument('--max-visualizations',type=int,default=12)
    p.add_argument('--annotations',type=Path,help='Optional documented JSON; only post-prediction region diagnostics.')
    p.add_argument('--scan-only',action='store_true',help='Validate names and pair plans only; no model prediction/results.')
    a=p.parse_args()
    if a.max_visualizations<0: p.error('max-visualizations must be nonnegative')
    tag=stamp();out=a.output_dir.resolve()/f'run_{tag}_{time.time_ns()}'
    out.mkdir(parents=True,exist_ok=False)
    scenes,warnings=scan(a.data_root,tuple(float(x) for x in a.expected_inc.split(',')),tuple(float(x) for x in a.expected_az.split(',')))
    if a.scene!='all':
        if a.scene not in scenes: p.error(f'Scene not found: {a.scene}; scan warnings saved below.')
        scenes={a.scene:scenes[a.scene]}
    annotations=json.loads(a.annotations.read_text()) if a.annotations else {'images':{}}
    base=a.annotations.resolve().parent if a.annotations else out
    planned=[(scene,x,y) for scene,s in scenes.items() for x,y in make_pairs(s['views'],a.pair_type,a.mode=='smoke')]
    config=dict(command=sys.argv,created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                torch=torch.__version__,cuda_runtime=torch.version.cuda,opencv=cv2.__version__,numpy=np.__version__,python=platform.python_version(),
                model_protocol=load_config(),scene_scan={k:{key:v for key,v in s.items() if key!='views'} for k,s in scenes.items()},planned_pairs=len(planned),
                data_files=[dict(path=str(v.path.relative_to(a.data_root)),sha256=sha256(v.path),readable=v.readable) for s in scenes.values() for v in s['views']],
                annotation_sha256=sha256(a.annotations) if a.annotations else None,
                metric_scope='No independent GT; acceptance and inlier residual are engineering consistency diagnostics, not registration accuracy.',
                timing_scope='Per-pair cold/no-feature-cache inference, excludes model loading/image I/O/NPZ/visualization. wall_seconds includes these.',
                source_identity={str(f.relative_to(ROOT)):sha256(f) for f in sorted((ROOT/'sar_match').rglob('*.py'))})
    write_json(out/f'run_config_{tag}.json',config)
    write_json(out/f'scan_warnings_{tag}.json',warnings)
    for w in warnings: print('WARNING',json.dumps(w,ensure_ascii=False),flush=True)
    if a.scan_only:
        write_csv(out/f'pair_plan_{tag}.csv',[dict(scene=s,image_A=str(x.path.relative_to(a.data_root)),image_B=str(y.path.relative_to(a.data_root)),category=category(x,y)) for s,x,y in planned])
        print(f'SCAN ONLY: {len(planned)} planned pairs; no predictions. Output: {out}');return 0
    if not planned: raise ValueError(f'No unambiguous pairs found; inspect {out}.')
    matcher=Matcher(a.model,a.device,a.budget)
    if matcher.device.type=='cuda':
        config['device_name']=torch.cuda.get_device_name(matcher.device)
        write_json(out/f'device_identity_{tag}.json',dict(name=config['device_name'],visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),properties=str(torch.cuda.get_device_properties(matcher.device))))
    pred_dir=out/'predictions';pred_dir.mkdir();viz=out/'visualizations';viz.mkdir()
    rows=[];seen_viz=set();saved=0;wall=time.perf_counter()
    logpath=out/f'execution_{tag}.log'
    with logpath.open('x',encoding='utf-8') as log:
        for index,(scene,x,y) in enumerate(planned):
            cat=category(x,y);pid=f'{scene}__inc{x.inc:g}_az{x.az:g}__inc{y.inc:g}_az{y.az:g}'
            row=dict(pair_id=pid,scene=scene,image_A=str(x.path.relative_to(a.data_root)),image_B=str(y.path.relative_to(a.data_root)),inc_A=x.inc,az_A=x.az,inc_B=y.inc,az_B=y.az,angle_category=cat,model=a.model,keypoint_budget=a.budget)
            begin=time.perf_counter()
            try:
                images=[read_image(v.path) for v in (x,y)]
                result,arrays=matcher.predict(*images);row.update(result);row.update(pixel_consistency(*images))
                # Annotation content is never passed into Matcher.predict or used for acceptance.
                entries=[]
                for v in (x,y):
                    rel=str(v.path.relative_to(a.data_root))
                    entry=annotations.get('images',{}).get(rel)
                    if entry is None:
                        # A basename key is permitted only when globally unique across this run.
                        names=[z.path.name for s in scenes.values() for z in s['views']]
                        if names.count(v.path.name)==1: entry=annotations.get('images',{}).get(v.path.name)
                    entries.append(entry)
                try: row.update(annotation_diagnostics(entries,arrays,[im.shape for im in images],base))
                except Exception as exc:
                    row.update(annotations_available=False,annotation_error=str(exc));log.write(f'ANNOTATION_ERROR {pid}: {exc}\n')
                np.savez_compressed(pred_dir/f'{pid}_{tag}.npz',**arrays)
                write_json(pred_dir/f'{pid}_{tag}.json',row)
                key=(scene,cat,'accepted' if row['accepted'] else 'rejected')
                want=a.visualizations=='all' or (a.visualizations=='representative' and saved<a.max_visualizations and key not in seen_viz)
                if want:
                    try:
                        draw_matches(*images,arrays,row,viz/f'{pid}_{tag}.png',title=f'{scene} | inc {x.inc:g}/{y.inc:g}, az {x.az:g}/{y.az:g} | {a.model}')
                        saved+=1;seen_viz.add(key)
                    except Exception as exc: log.write(f'VISUALIZATION_ERROR {pid}: {exc}\n')
            except Exception as exc:
                row.update(accepted=False,reason=type(exc).__name__+': '+str(exc),failure_type='program_exception',total_inference_seconds=None)
                log.write(traceback.format_exc()+'\n')
            row['pair_wall_seconds']=time.perf_counter()-begin;rows.append(clean(row))
            log.write(json.dumps(clean(dict(progress=f'{index+1}/{len(planned)}',**row)),ensure_ascii=False,allow_nan=False)+'\n');log.flush()
            print(f"[{index+1}/{len(planned)}] {pid}: {row['reason']}",flush=True)
    summaries=[]
    for scene in scenes:
        rr=[r for r in rows if r['scene']==scene]
        for cat in ['all','same_inc_different_az','same_az_different_inc','both_different']:
            summaries.append(dict(scene=scene,angle_category=cat,**summarize([r for r in rr if cat=='all' or r['angle_category']==cat])))
    write_csv(out/f'pairs_{tag}.csv',rows);write_csv(out/f'scene_summary_{tag}.csv',summaries)
    write_json(out/f'overall_summary_{tag}.json',dict(**summarize(rows),wall_seconds=time.perf_counter()-wall,visualizations=saved,warning_count=len(warnings),by_angle_category={cat:summarize([r for r in rows if r['angle_category']==cat]) for cat in ['same_inc_different_az','same_az_different_inc','both_different']},independent_GT_available=False,notice=config['metric_scope']))
    # Stable names requested by the user; timestamped files remain authoritative.
    for name in ['pairs.csv','scene_summary.csv','overall_summary.json','run_config.json']:
        pth=Path(name);(out/name).symlink_to(f'{pth.stem}_{tag}{pth.suffix}')
    print('Output:',out,flush=True)
    return 2 if any(r['failure_type']=='program_exception' for r in rows) else 0

if __name__=='__main__': sys.exit(main())
