"""Full acceptance of a frozen formal run, independent source-mask checks and replay."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import cv2
import numpy as np
from sarreg.io import jsonl_read, save_json
from sarreg.qa import validate_pair
from sarreg.synthesis import synthesize
from scripts.summarize_phase1 import check_sample, source_instance, points, read


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_labels(folder,row,meta,cfg):
    """Check every source instance, including instances absent from output labels."""
    size=meta['output_size'];errors=[];support=0;dropped=0
    matrices=list(map(np.asarray,meta['H_source_to_view']))
    src_shape=tuple(meta['source_shape_hw'])
    for i,H in enumerate(matrices):
        expected={};events={e['instance_id']:e for e in meta['dynamic_events'][i]}
        for a in row['annotations']:
            src=source_instance(src_shape,a)
            mask=cv2.warpPerspective(src,H,(size,size),flags=cv2.INTER_NEAREST)
            area=int(np.count_nonzero(mask));support+=area
            center=np.array([[src_shape[1]/2,src_shape[0]/2],[src_shape[1]/2+1,src_shape[0]/2],[src_shape[1]/2,src_shape[0]/2+1]])
            p=points(H,center);jac=np.column_stack([p[1]-p[0],p[2]-p[0]])
            full=np.count_nonzero(src)*abs(np.linalg.det(jac))
            eligible=area>=cfg['min_ship_pixels'] and min(1.,area/max(1.,full))>=cfg['min_ship_visible_fraction']
            if not eligible:
                dropped+=int(area>0);continue
            key=str(a['id'])
            if key in events:
                dx,dy=events[key]['shift_xy']
                mask=cv2.warpAffine(mask,np.float32([[1,0,dx],[0,1,dy]]),(size,size),flags=cv2.INTER_NEAREST)
                mask[read(folder/f'valid_{i}.png')==0]=0
                area=int(np.count_nonzero(mask))
            if area:expected[key]=area
        got={a['instance_id']:a['area_pixels'] for a in meta['ship_annotations'][i]}
        if got!=expected:errors.append(f'complete source annotation propagation mismatch view {i}')
    return errors,support,dropped


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--stamp',required=True);a=p.parse_args()
    root=Path(a.root);out=Path('reports');stamp=a.stamp;reportpath=out/f'phase1_formal_validation_{stamp}.json'
    if reportpath.exists():raise FileExistsError(reportpath)
    provenance=json.loads((root/'provenance.json').read_text());recipe=provenance['recipe']
    rows=jsonl_read(provenance['index']);lookup={r['image_path']:r for r in rows};manifest=jsonl_read(root/'manifest.jsonl')
    audit=json.loads(Path(provenance['audit']).read_text());protocol=[]
    freeze_path=root/f'version_freeze_{stamp}.json'
    freeze=json.loads(freeze_path.read_text()) if freeze_path.exists() else {'code_commit':provenance['code_commit']}
    for path,digest in freeze.get('code_files',{}).items():
        if path.startswith(('sarreg/','configs/')) and sha(path)!=digest:protocol.append(f'generation code/config changed: {path}')
    if sha(provenance['index'])!=provenance['index_sha256']:protocol.append('index hash changed')
    if sha(provenance['config_path'])!=provenance['recipe_sha256']:protocol.append('recipe hash changed')
    components=defaultdict(set)
    for r in rows:components[('hrsid' if r['dataset']=='background' else r['dataset'],r['source_group'])].add(r['split'])
    if any(len(v)>1 for v in components.values()):protocol.append('source component across splits')
    if any(lookup[e['a']]['split']!=lookup[e['b']]['split'] for e in audit['accepted']):protocol.append('accepted source edge across splits')
    if any(len({lookup[path]['split'] for path in g})>1 for g in audit['decoded_duplicate_groups']):protocol.append('decoded duplicates across splits')
    actual=Counter(m['stratum'] for m in manifest)
    expected={f'{i:02d}_{s["scene"]}_{s["dataset"]}_{s["profile"]}_{s["ships"]}':s['count'] for i,s in enumerate(recipe['strata'])}
    if dict(actual)!=expected:protocol.append('stratum counts differ from recipe')
    if len({m['sample_id'] for m in manifest})!=len(manifest):protocol.append('repeated sample IDs')
    if len(list(root.glob('test/pair_*/meta.json')))!=len(manifest):protocol.append('unmanifested/missing folders')
    checks=[];stats=[];hashes=[];maxkeys=['max_forward_error_px','max_backward_error_px','max_relative_matrix_error','max_inverse_matrix_error','max_cycle_matrix_error']
    for idx,m in enumerate(manifest):
        folder=root/m['folder'];row=lookup[m['source_path']];cfg=json.loads((root/m['config']).read_text())
        c,meta=check_sample(folder,row);qa=validate_pair(folder)
        c['errors']+=qa['issues']
        errors,support,dropped=complete_labels(folder,row,meta,cfg);c['errors']+=errors
        for k in ('source_id','source_group','source_split'):
            if meta[k]!=m[k]:c['errors'].append(f'manifest/meta mismatch {k}')
        if row['split']!='test' or row['source_group']!=meta['source_group']:c['errors'].append('wrong source component/split')
        if meta['source_resize_factors_xy']!=[1.,1.] or meta['scale_policy']!='native_adaptive':c['errors'].append('non-native source scale')
        for pair in meta['pairs']:
            overlap=min(pair['overlap_fraction_i'],pair['overlap_fraction_j'])
            # The existing multi-view sampler constrains each view against reference 0.
            if pair['views'][0]==0 and not cfg['min_overlap']-1e-6<=overlap<=cfg['max_overlap']+1e-6:c['errors'].append('overlap profile mismatch')
            if m['profile']=='none' and (pair['match_count'] or overlap!=0):c['errors'].append('nonempty negative pair')
        if m['ship_stratum']=='absent' and support:c['errors'].append('annotated ship support in absent stratum')
        if m['ship_stratum']=='present' and not m['ships']:c['errors'].append('empty present stratum')
        events=sum(len(e) for e in meta['dynamic_events'])
        if m['profile']=='dynamic' and not events:c['errors'].append('no actual dynamic displacement')
        c['passed']=not c['errors'];checks.append(c)
        stats.append(dict(m,source_height=meta['source_shape_hw'][0],source_width=meta['source_shape_hw'][1],
                          source_resize_x=meta['source_resize_factors_xy'][0],source_resize_y=meta['source_resize_factors_xy'][1],
                          visible_source_support_pixels=support,threshold_dropped_instances=dropped,
                          sea_masks=bool(row.get('sea_mask_path')),dynamic_events=events,match_count=c['match_count'],passed=c['passed']))
        for f in sorted(folder.iterdir()):
            if f.is_file():hashes.append({'path':str(f.relative_to(root)),'sha256':sha(f)})
        if idx%100==0:print(f'Independent acceptance {idx}/{len(manifest)}',flush=True)
    # Fixed representatives from every profile: byte-for-byte replay in disposable directories.
    replay=[]
    for profile in sorted({m['profile'] for m in manifest}):
        options=[m for m in manifest if m['profile']==profile]
        chosen=options[0]
        if profile in ('easy','dynamic','low'):chosen=next((m for m in options if m['dataset']=='ssdd'),chosen)
        cfg=json.loads((root/chosen['config']).read_text());row=lookup[chosen['source_path']]
        with tempfile.TemporaryDirectory(prefix='sarreg_replay_') as d:
            temp=Path(d)/'sample';synthesize(row,cfg,temp,chosen['seed']);orig=root/chosen['folder']
            equal={f.name:sha(f)==sha(orig/f.name) for f in sorted(temp.iterdir())}
            replay.append({'sample_id':chosen['sample_id'],'profile':profile,'files':len(equal),'passed':all(equal.values()),'mismatches':[f for f,v in equal.items() if not v]})
    report={'checked':len(checks),'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),
            'protocol_errors':protocol,'errors':[c for c in checks if not c['passed']],
            'metrics':{k:max(c[k] for c in checks) for k in maxkeys},
            'groups':dict(actual),'datasets':dict(Counter(m['dataset'] for m in manifest)),
            'profiles':dict(Counter(m['profile'] for m in manifest)),
            'scene_strata':dict(Counter(m['scene_stratum'] for m in manifest)),
            'ship_strata':dict(Counter(m['ship_stratum'] for m in manifest)),
            'view_pair_relations':sum(m['pairs'] for m in manifest),
            'matches':sum(c['match_count'] for c in checks),'ship_view_instances':sum(c['bbox_count'] for c in checks),
            'unique_sources':len({m['source_path'] for m in manifest}),
            'unique_source_components':len({('hrsid' if m['dataset']=='background' else m['dataset'],m['source_group']) for m in manifest}),
            'source_reuse_max':max(Counter(m['source_path'] for m in manifest).values()),
            'native_output_sizes':{ds:dict(Counter(m['output_size'] for m in manifest if m['dataset']==ds)) for ds in {m['dataset'] for m in manifest}},
            'manual_valid_mismatch_pixels':sum(c['manual_valid_mismatch_pixels'] for c in checks),
            'manual_sea_mismatch_pixels':sum(c['manual_sea_mismatch_pixels'] for c in checks),
            'sea_mask_groups':sum(s['sea_masks'] for s in stats),'dynamic_events':sum(s['dynamic_events'] for s in stats),
            'threshold_dropped_instances':sum(s['threshold_dropped_instances'] for s in stats),
            'replay':replay,'index_sha256':provenance['index_sha256'],'generation_code_commit':freeze['code_commit'],
            'validation_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'native_source_upsampling_groups':0,'point_matrix_tolerance':1e-3,'source_mask_pixel_tolerance':0}
    report['accepted']=not report['failed'] and not protocol and all(r['passed'] for r in replay)
    with (out/f'phase1_formal_statistics_{stamp}.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    save_json(reportpath,report)
    extra=['manifest.jsonl','provenance.json','generation_report.json']+([freeze_path.name] if freeze_path.exists() else [])
    save_json(root/f'file_checksums_{stamp}.json',hashes+[{'path':str(f.relative_to(root)),'sha256':sha(f)} for f in sorted((root/'configs').glob('*.json'))]+[{'path':name,'sha256':sha(root/name)} for name in extra])
    if report['accepted']:save_json(root/f'ACCEPTED_{stamp}.json',{'validation_report':str(reportpath.resolve()),'sha256':sha(reportpath),'scientific_kind':'synthetic_2d_registration_proxy','accepted':True})
    print(json.dumps({k:v for k,v in report.items() if k not in ('groups','native_output_sizes','errors')},indent=2))
    if not report['accepted']:raise SystemExit(2)


if __name__=='__main__':main()
