"""Audit reserved sources and generate a preregistered confirmation proxy.

Source similarity only retrieves candidates; reuse Phase1 non-ship pixel proof.
Never modifies an earlier index, experiment or split.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import yaml
from sarbench.data import source_component
from sarbench.estimate import perspective
from sarbench.reports import save_json
from sarreg.io import jsonl_read,jsonl_write,read_gray
from sarreg.synthesis import synthesize,stable_seed
from sarreg.qa import validate_pair


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clock():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def source_audit(cfg,out):
    old=Path(cfg['phase25_run']); reserved=jsonl_read(old/'data_v2/confirmation_sources.jsonl'); allrows=jsonl_read(cfg['source_index'])
    assert len(reserved)==863; reserve={source_component(r) for r in reserved};assert len(reserve)==133
    historical=json.loads(Path(cfg['source_audit']).read_text());assert sha(cfg['source_index'])==historical['output_sha256']
    prov=json.loads((old/'data_v2/provenance.json').read_text())
    for role in ['dev','val']:assert not reserve&set(prov['role_components'][role])
    assert not reserve&set(prov['formal_test_components'])
    lookup={r['image_path']:r for r in allrows}; boundary=lambda a,b:(source_component(a) in reserve)!=(source_component(b) in reserve)
    assert all(not boundary(lookup[e['a']],lookup[e['b']]) for e in historical['accepted'])
    root=Path(cfg['phase1_recipe']).resolve().parents[1];sys.path.insert(0,str(root))
    from scripts.audit_source_relationships import verify_pair,ncc
    cv2.setNumThreads(1);cv2.setRNGSeed(20261009)
    hashes=defaultdict(list);phash=[];thumb=[];desc=[];owners=[];coordinates=defaultdict(list)
    orb=cv2.ORB_create(nfeatures=120,fastThreshold=20);texture={};reserved_paths={x['image_path'] for x in reserved}
    for i,r in enumerate(allrows):
        im=read_gray(r['image_path'])
        hashes[hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()].append(i)
        t=cv2.resize(im,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32);thumb.append(t)
        d=cv2.dct(cv2.resize(t,(32,32)))[:8,:8].ravel();phash.append(np.packbits(d>np.median(d[1:])))
        scale=min(1,384/max(im.shape));small=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)))
        _,ds=orb.detectAndCompute(small,None)
        if ds is not None:desc.append(ds);owners.extend([i]*len(ds))
        if r['dataset'] in ['hrsid','background']:coordinates[tuple(r['source_id'].split('_')[1:])].append(i)
        if r['image_path'] in reserved_paths:
            blur=cv2.GaussianBlur(cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA),(5,5),1)
            gradient=cv2.magnitude(cv2.Sobel(blur,cv2.CV_32F,1,0),cv2.Sobel(blur,cv2.CV_32F,0,1));texture[r['image_path']]=float(np.percentile(gradient,75))
        if (i+1)%1000==0:print('Source fingerprints',i+1,len(allrows),flush=True)
    inventory=hashlib.sha256(''.join(sorted(hashes)).encode()).hexdigest();assert inventory==historical['source_inventory_sha256'], 'Source pixels differ from historical audit'
    exact=[];pairs=set();bits=np.array([i.bit_count() for i in range(256)],np.uint8);phash=np.asarray(phash);thumb=np.asarray(thumb)
    for group in hashes.values():
        for i in group:
            for j in group:
                if i<j and boundary(allrows[i],allrows[j]):exact.append((i,j));pairs.add((i,j))
    queries=[i for i,r in enumerate(allrows) if source_component(r) in reserve];other=np.array([i for i,r in enumerate(allrows) if source_component(r) not in reserve]);routes=Counter()
    for i in queries:
        distance=bits[np.bitwise_xor(phash[i],phash[other])].sum(axis=1)
        for j in other[distance<=10]:
            if ncc(thumb[i].ravel(),thumb[j].ravel())>=.85:pairs.add(tuple(sorted((i,int(j)))));routes['phash_thumbnail']+=1
    for group in coordinates.values():
        for i in group:
            for j in group:
                if i<j and boundary(allrows[i],allrows[j]) and ncc(thumb[i].ravel(),thumb[j].ravel())>=.90:pairs.add((i,j));routes['shared_crop_coordinates']+=1
    descriptors=np.vstack(desc);owners=np.asarray(owners);matcher=cv2.FlannBasedMatcher(dict(algorithm=6,table_number=8,key_size=16,multi_probe_level=1),dict(checks=32));matcher.add([descriptors]);matcher.train();votes=Counter()
    query_indices=np.where(np.isin(owners,queries))[0]
    for start in range(0,len(query_indices),5000):
        indices=query_indices[start:start+5000]
        for q,near in enumerate(matcher.knnMatch(descriptors[indices],k=8)):
            i=int(owners[indices[q]]);seen=set()
            for match in near:
                j=int(owners[match.trainIdx])
                if match.distance<=32 and j not in seen and boundary(allrows[i],allrows[j]):seen.add(j);votes[tuple(sorted((i,j)))]+=1
    local={pair for pair,vote in votes.items() if vote>=10};pairs.update(local);routes['local_feature_votes']=len(local)
    old_rejected={tuple(sorted((e['a'],e['b']))) for e in historical['rejected']};accepted=[];rejected=[];reused=0
    for k,(i,j) in enumerate(sorted(pairs)):
        a,b=allrows[i],allrows[j];edge={'a':a['image_path'],'b':b['image_path'],'components':[source_component(a),source_component(b)]}
        if (i,j) in exact:evidence={'method':'decoded_pixel_equality'}
        elif tuple(sorted((a['image_path'],b['image_path']))) in old_rejected:evidence=None;reused+=1
        else:evidence=verify_pair(a,b)
        (accepted if evidence else rejected).append({**edge,**(evidence or {})})
        if (k+1)%50==0:print('Cross-reserve candidates checked',k+1,len(pairs),'new accepted',len(accepted),flush=True)
    quarantine={c for e in accepted for c in e['components'] if c in reserve};eligible=[r for r in reserved if source_component(r) not in quarantine]
    jsonl_write(out/'eligible_sources.jsonl',eligible);save_json(out/'source_texture.json',texture)
    audit={'accepted':bool(eligible),'original_reserved_records':863,'original_reserved_components':133,'eligible_records':len(eligible),'eligible_components':len({source_component(r) for r in eligible}),
           'quarantined_components':sorted(quarantine),'known_accepted_boundary_edges':0,'fresh_accepted_boundary_edges':accepted,'fresh_rejected_boundary_candidates':rejected,
           'cross_boundary_exact_duplicates':len(exact),'candidate_routes':dict(routes),'candidates':len(pairs),'historical_rejections_reused':reused,'source_inventory_sha256':inventory,
           'source_index_sha256':sha(cfg['source_index']),'reserved_index_sha256':sha(old/'data_v2/confirmation_sources.jsonl'),'historical_audit_sha256':sha(cfg['source_audit']),
           'audit_code_sha256':{str(p.resolve()):sha(p) for p in [Path(__file__),root/'scripts/audit_source_relationships.py']},
           'thresholds':historical['thresholds'],'known_component_intersections':{'dev':[],'val':[],'test':[]},
           'limitations':historical['limitations']+['No acquisition timestamps, sensor/orbit/look-angle/geolocation metadata in source index; filename/pixel evidence is not proof of independent radar acquisitions.','Background pool has only one parent P0001; SSDD source_scene remains unknown.','Fresh retrieval is targeted at reserve boundaries; previous pixel-supported rejected candidates are reused after inventory hash verification.']}
    save_json(out/'source_audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ['fresh_rejected_boundary_candidates','fresh_accepted_boundary_edges']},indent=2))


def generate(cfg,out):
    assert json.loads((out/'source_audit.json').read_text())['accepted'];root=out/'confirmation'
    if root.exists():raise FileExistsError(root)
    rows=jsonl_read(out/'eligible_sources.jsonl');texture=json.loads((out/'source_texture.json').read_text());threshold=float(np.quantile([texture[r['image_path']] for r in rows if r['dataset']=='hrsid' and r['scene_type']=='offshore'],cfg['low_texture_quantile']))
    recipe_path=Path(cfg['phase1_recipe']).resolve();recipe=yaml.safe_load(recipe_path.read_text());recipe['base_config']=str(recipe_path.parent/'easy.yaml')
    spec=importlib.util.spec_from_file_location('frozen_formal_builder',recipe_path.parents[1]/'scripts/build_formal_dataset.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    plan=[]
    for ci,s in enumerate(cfg['strata']):
        pool=[r for r in rows if r['dataset']==s['dataset'] and (s['scene'] in ['any','background'] or r['scene_type']==s['scene']) and (s['ships']!='present' or r['annotations'])]
        if s.get('texture')=='low':pool=[r for r in pool if texture[r['image_path']]<=threshold]
        plan.append({**s,'stratum':ci,'effective_count':s['count'] if pool else 0,'pool_records':len(pool),'pool_components':len({source_component(r) for r in pool})})
    selection=Path(cfg['phase25_run'])/'development/selection.json';locked=json.loads(selection.read_text())
    assert all(sha(path)==digest for path,digest in locked['inference_code_sha256'].items());assert sha('configs/phase25.yaml')==locked['config_sha256']
    save_json(out/'preregistration.json',{'sealed_at':clock(),'config':cfg,'config_sha256':sha('configs/phase26.yaml'),'selection_sha256':sha(selection),'source_audit_sha256':sha(out/'source_audit.json'),
                                       'eligible_sources_sha256':sha(out/'eligible_sources.jsonl'),'strata':plan,'texture_threshold':threshold,'inference_code_sha256':locked['inference_code_sha256'],
                                       'primary_analysis':'Paired success/wrong acceptance and equal-component summaries for seven frozen methods. Strong-stress reported separately. Attribution intermediates never used for choice.',
                                       'acceptance':'All qa checks, native-scale factors [1,1], profile overlap range, 512 static GT matches for positives/0 negatives, independent H reprojection<1e-3px, dynamic profiles require an actual move. Max source reuse 2; exhausted construction slots are explicitly omitted before inference.'})
    generation_files=[Path(__file__),recipe_path,recipe_path.parent/'easy.yaml',recipe_path.parents[1]/'scripts/build_formal_dataset.py']+sorted((recipe_path.parents[1]/'sarreg').glob('*.py'))
    save_json(out/'generation_code_sha256.json',{str(p.resolve()):sha(p) for p in generation_files})
    root.mkdir();cv2.setNumThreads(1);manifest=[];checks=[];rejects=[];source_usage=Counter();component_usage=Counter();seq=0
    for s in plan:
        ci=s['stratum'];pool=[r for r in rows if r['dataset']==s['dataset'] and (s['scene'] in ['any','background'] or r['scene_type']==s['scene']) and (s['ships']!='present' or r['annotations'])]
        if s.get('texture')=='low':pool=[r for r in pool if texture[r['image_path']]<=threshold]
        surrogate={**s,'profile':'medium' if s['profile']=='strong' else s['profile']};effective=helper.resolved_cfg(recipe,surrogate);effective['views']=2
        if s['profile']=='strong':effective['perspective_fraction']=s['perspective_fraction']
        save_json(root/'configs'/f'{ci:02d}.json',effective)
        rng=np.random.default_rng(stable_seed(cfg['data_seed'],str(ci)));ordered=[pool[i] for i in rng.permutation(len(pool))];tie={r['source_id']:i for i,r in enumerate(ordered)}
        for sample in range(s['effective_count']):
            succeeded=False;attempted=Counter()
            for trial in range(cfg['max_attempts_per_sample']):
                available=[r for r in ordered if source_usage[r['image_path']]<cfg['max_source_reuse']]
                if not available:break
                row=min(available,key=lambda r:(component_usage[source_component(r)]+attempted[source_component(r)],source_usage[r['image_path']],tie[r['source_id']]))
                attempted[source_component(row)]+=1
                # Rotate unsuccessful source attempts without changing its accepted-use budget.
                tie[row['source_id']]+=len(ordered)
                seed=stable_seed(cfg['data_seed'],f'confirmation:{ci}:{sample}:{trial}:{row["source_id"]}')
                with tempfile.TemporaryDirectory(prefix='phase26_generate_') as tmp:
                    temp=Path(tmp)/'sample'
                    try:synthesize(row,effective,temp,seed)
                    except (RuntimeError,ValueError,cv2.error) as exc:rejects.append({'stratum':ci,'sample':sample,'trial':trial,'source_id':row['source_id'],'reason':str(exc)});continue
                    meta=json.loads((temp/'meta.json').read_text());ships=any(meta['ship_annotations']);moving=any(meta['dynamic_events'][1:])
                    if (s['ships']=='present' and not ships) or (s['ships']=='absent' and ships) or (s['profile']=='dynamic' and not moving):rejects.append({'stratum':ci,'sample':sample,'trial':trial,'source_id':row['source_id'],'reason':'ship/dynamic acceptance'});continue
                    qa=validate_pair(temp);assert qa['passed'],qa;assert meta['source_resize_factors_xy']==[1.,1.]
                    pair=meta['pairs'][0];overlap=min(pair['overlap_fraction_i'],pair['overlap_fraction_j']);assert effective['min_overlap']-1e-6<=overlap<=effective['max_overlap']+1e-6
                    with np.load(temp/pair['matches']) as gt:x=gt['points_i'];y=gt['points_j']
                    assert len(x)==(0 if s['profile']=='none' else 512);error=float(np.linalg.norm(perspective(pair['H_i_to_j'],x)-y,axis=1).max()) if len(x) else 0;assert error<1e-3
                    folder=root/'confirm'/f'pair_{seq:06d}';folder.parent.mkdir(exist_ok=True);shutil.move(str(temp),folder)
                    manifest.append({'sample_id':folder.name,'folder':str(folder.relative_to(root)),'profile':s['profile'],'scene_stratum':s['scene'],'texture_stratum':s.get('texture','unrestricted'),'evaluation_subset':s.get('evaluation_subset','primary'),
                                     'dataset':row['dataset'],'source_id':row['source_id'],'source_component':source_component(row),'source_split':row['split'],'source_path':row['image_path'],'seed':seed,'config':f'configs/{ci:02d}.json','output_size':meta['output_size'],'num_views':2})
                    checks.append({'id':folder.name,'qa_passed':True,'max_gt_error':error,'dynamic_move':moving});source_usage[row['image_path']]+=1;component_usage[source_component(row)]+=1;seq+=1;succeeded=True;break
            if not succeeded:rejects.append({'stratum':ci,'sample':sample,'reason':'omitted: source-only construction budget exhausted; no replacement from earlier roles'})
        jsonl_write(root/'manifest.jsonl',manifest);save_json(root/'generation.json',{'groups':seq,'rejections':rejects,'accepted_source_usage':dict(source_usage),'component_usage':dict(component_usage)})
        print('Generated stratum',ci,s['profile'],'total',seq,flush=True)
    assert manifest;assert max(source_usage.values())<=cfg['max_source_reuse'];save_json(root/'acceptance.json',{'accepted':True,'groups':seq,'profiles':dict(Counter(m['profile'] for m in manifest)),
                       'source_components':len(component_usage),'unique_sources':len(source_usage),'source_usage_max':max(source_usage.values()),'qa_passed':len(checks),'max_gt_error':max(c['max_gt_error'] for c in checks),'checks':checks})
    save_json(root/'file_checksums.json',[{'path':str(p.relative_to(root)),'sha256':sha(p)} for p in sorted(root.rglob('*')) if p.is_file()]);print('DATA ACCEPTED',seq,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','generate']);p.add_argument('--run',required=True);p.add_argument('--config',default='configs/phase26.yaml');a=p.parse_args();out=Path(a.run).resolve();cfg=yaml.safe_load(Path(a.config).read_text())
    if a.stage=='audit':
        if (out/'source_audit.json').exists():raise FileExistsError(out/'source_audit.json')
        source_audit(cfg,out)
    else:generate(cfg,out)
