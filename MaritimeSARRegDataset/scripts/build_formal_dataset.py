"""Generate a fixed, stratified held-out set from an explicitly audited index."""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import yaml
import numpy as np
import cv2

from sarreg.io import jsonl_read, jsonl_write, save_json
from sarreg.synthesis import load_cfg, stable_seed, synthesize
from sarreg.qa import validate_pair
from sarreg.labels import annotation_mask


def resolved_cfg(recipe,stratum):
    cfg=load_cfg(recipe['base_config']);cfg=copy.deepcopy(cfg)
    for k in ('output_size','allow_source_upsampling','output_size_multiple'):
        cfg[k]=recipe[k]
    profile=recipe['profiles'][stratum['profile']]
    for k,v in profile.items():
        if isinstance(v,dict):cfg.setdefault(k,{}).update(v)
        else:cfg[k]=v
    if stratum['dataset']=='ssdd':cfg['output_size']=profile.get('ssdd_output_size',recipe['ssdd_output_size'])
    cfg['center_mode']='ship' if stratum['ships']=='present' else 'uniform'
    if stratum['profile']=='none':cfg['center_mode']='default'
    return cfg


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True)
    p.add_argument('--audit',required=True);p.add_argument('--config',default='configs/phase1_formal_v1.yaml')
    p.add_argument('--out',required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError(f'Preserving existing run: {out}')
    recipe=yaml.safe_load(Path(a.config).read_text());rows=jsonl_read(a.index)
    audit=json.loads(Path(a.audit).read_text())
    if audit['output_sha256']!=hashlib.sha256(Path(a.index).read_bytes()).hexdigest():raise ValueError('Stale audit/index provenance')
    if audit['remaining_accepted_cross_split_edges']:raise ValueError('Unrepaired source leakage')
    canonical={p:min(g) for g in audit['decoded_duplicate_groups'] for p in g}
    eligible=[r for r in rows if r['split']==recipe['source_split'] and canonical.get(r['image_path'],r['image_path'])==r['image_path']]
    out.mkdir(parents=True)
    pools=[]
    for stratum in recipe['strata']:
        pool=sorted([r for r in eligible if r['dataset']==stratum['dataset'] and (stratum['scene'] in ('any','background') or r['scene_type']==stratum['scene'])],key=lambda r:r['source_id'])
        if not pool:raise ValueError(f'Empty real source pool: {stratum}')
        pools.append(pool)
    provenance={'index':str(Path(a.index).resolve()),'index_sha256':audit['output_sha256'],
                'source_inventory_sha256':audit['source_inventory_sha256'],'audit':str(Path(a.audit).resolve()),
                'config_path':str(Path(a.config).resolve()),'recipe_sha256':hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
                'recipe':recipe,'seed':recipe['seed'],'preflight':a.preflight,
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                'code_diff':subprocess.check_output(['git','diff','HEAD','--','sarreg','scripts','configs'],text=True),
                'environment':{'python':__import__('platform').python_version(),'numpy':np.__version__,'cv2':cv2.__version__,
                               'opencv_threads':cv2.getNumThreads(),'workers':1,'rng':'PCG64','checkpoint':None},
                'sampling':'Deterministic shuffled source cycles, explicit rejection for visible-ship strata; source-ID seeds are independent of absolute paths.'}
    save_json(out/'provenance.json',provenance)
    manifest=[];attempts=[];fatal=[];seq=0
    for c,(stratum,pool) in enumerate(zip(recipe['strata'],pools)):
        cfg=resolved_cfg(recipe,stratum);case=f'{c:02d}_{stratum["scene"]}_{stratum["dataset"]}_{stratum["profile"]}_{stratum["ships"]}'
        save_json(out/'configs'/f'{case}.json',cfg)
        rng=np.random.default_rng(stable_seed(recipe['seed'],case));order=rng.permutation(len(pool));cursor=0
        count=1 if a.preflight else stratum['count']
        for i in range(count):
            success=False
            for trial in range(recipe['max_attempts_per_sample']):
                row=pool[int(order[cursor%len(pool)])];cursor+=1
                seed=stable_seed(recipe['seed'],f'{case}:{i}:{trial}:{row["source_id"]}')
                with tempfile.TemporaryDirectory(prefix='sarreg_formal_') as temporary:
                    temp=Path(temporary)/'sample'
                    try:
                        synthesize(row,cfg,temp,seed)
                    except (RuntimeError,ValueError,cv2.error) as exc:
                        attempts.append({'case':case,'sample':i,'trial':trial,'source_id':row['source_id'],'seed':seed,'reason':str(exc)})
                        continue
                    meta=json.loads((temp/'meta.json').read_text());has_ship=any(meta['ship_annotations'])
                    valid_state=(stratum['ships']=='any' or has_ship==(stratum['ships']=='present'))
                    if stratum['ships']=='absent':
                        # No visible annotated support, including instances below label thresholds.
                        shape=tuple(meta['source_shape_hw']);size=meta['output_size']
                        for ann in row['annotations']:
                            support,_=annotation_mask(shape,ann)
                            if any(np.any(cv2.warpPerspective(support,np.asarray(H),(size,size),flags=cv2.INTER_NEAREST)) for H in meta['H_source_to_view']):
                                valid_state=False;break
                    if stratum['profile']=='dynamic':valid_state &= any(meta['dynamic_events'][1:])
                    if not valid_state:
                        attempts.append({'case':case,'sample':i,'trial':trial,'source_id':row['source_id'],'seed':seed,'reason':'visible-ship/dynamic stratum rejection'})
                        continue
                    qa=validate_pair(temp)
                    if not qa['passed']:raise ValueError(f'Invalid generated GT: {qa}')
                    folder=out/'test'/f'pair_{seq:06d}';folder.parent.mkdir(exist_ok=True)
                    shutil.move(str(temp),str(folder))
                    manifest.append({'sample_id':f'pair_{seq:06d}','folder':str(folder.relative_to(out)),
                                     'stratum':case,'profile':stratum['profile'],'scene_stratum':stratum['scene'],
                                     'ship_stratum':stratum['ships'],'dataset':row['dataset'],'source_id':row['source_id'],
                                     'source_group':row['source_group'],'source_path':row['image_path'],'source_split':row['split'],
                                     'seed':seed,'output_size':meta['output_size'],'num_views':meta['num_views'],
                                     'ships':sum(len(v) for v in meta['ship_annotations']),'pairs':len(meta['pairs']),
                                     'min_overlap':min(min(v['overlap_fraction_i'],v['overlap_fraction_j']) for v in meta['pairs']),
                                     'config':f'configs/{case}.json'})
                    success=True;seq+=1;break
            if not success:
                fatal.append({'case':case,'sample':i,'reason':'stratum sampling exhausted'})
                break
            if seq%20==0:print(f'Accepted {seq} groups, rejected {len(attempts)} attempts',flush=True)
        jsonl_write(out/'manifest.jsonl',manifest)
        save_json(out/'generation_report.json',{'requested':len(recipe['strata']) if a.preflight else sum(s['count'] for s in recipe['strata']),
                                                'generated':len(manifest),'fatal':fatal,'rejected_attempts':attempts,
                                                'strata':dict(Counter(m['stratum'] for m in manifest))})
        print(f'{case}: accepted {count if not fatal else i}/{count}, source pool {len(pool)}',flush=True)
        if fatal:raise SystemExit(2)
    print(json.dumps({'groups':len(manifest),'view_pair_relations':sum(m['pairs'] for m in manifest),'rejected_attempts':len(attempts),'out':str(out)},indent=2))


if __name__=='__main__':main()
