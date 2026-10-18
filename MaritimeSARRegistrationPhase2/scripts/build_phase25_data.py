"""Build component-disjoint development/validation proxies; reserve unseen confirmation sources."""
import argparse, collections, copy, hashlib, importlib.util, json, math, shutil, subprocess, tempfile
from pathlib import Path
import cv2
import numpy as np
import yaml
from sarreg.synthesis import load_cfg, stable_seed, synthesize
from sarreg.qa import validate_pair
from sarreg.io import jsonl_read, jsonl_write
from sarbench.data import source_component
from sarbench.reports import save_json
from sarbench.estimate import perspective

p=argparse.ArgumentParser();p.add_argument('--config',default='configs/phase25.yaml');p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out).resolve()
if out.exists():raise FileExistsError(out)
config_path=Path(a.config).resolve();cfg=yaml.safe_load(config_path.read_text());index=Path(cfg['source_index']).resolve();rows=jsonl_read(index);audit_path=Path(cfg['source_audit']).resolve();audit=json.loads(audit_path.read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(index)==audit['output_sha256'] and audit['remaining_accepted_cross_split_edges']==0
assert all(r['split_protocol']=='phase1_pixel_supported_source_components_v2' for r in rows)
components={s:{source_component(r) for r in rows if r['split']==s} for s in ['train','val','test']}
assert all('unknown' not in v for v in components.values())
assert not (components['train']&components['val'] or components['train']&components['test'] or components['val']&components['test'])
canonical={p:min(g) for g in audit['decoded_duplicate_groups'] for p in g}
rows=[r for r in rows if canonical.get(r['image_path'],r['image_path'])==r['image_path']]
train_components=sorted(components['train']);reserve=set()
for namespace in ['hrsid','ssdd']:
 pool=[k for k in train_components if k.startswith(namespace+':')];rng=np.random.default_rng(stable_seed(cfg['seed'],namespace));order=rng.permutation(len(pool));reserve.update(pool[i] for i in order[:math.ceil(len(pool)*cfg['confirmation_fraction'])])
# Ensure at least one background parent is untouched and at least one remains in dev.
bg=sorted({source_component(r) for r in rows if r['split']=='train' and r['dataset']=='background'})
assert len(bg)>=2
reserve.difference_update(bg);reserve.add(bg[int(stable_seed(cfg['seed'],'background_reserve')%len(bg))])
roles={'dev':[r for r in rows if r['split']=='train' and source_component(r) not in reserve], 'val':[r for r in rows if r['split']=='val'], 'confirmation':[r for r in rows if r['split']=='train' and source_component(r) in reserve]}
role_components={k:{source_component(r) for r in v} for k,v in roles.items()}
assert all(not role_components[x]&role_components[y] for x,y in [('dev','val'),('dev','confirmation'),('val','confirmation')])
assert all(not c&components['test'] for c in role_components.values())
for role in ['dev','val']:
 for s in cfg['strata']:
  pool=[r for r in roles[role] if r['dataset']==s['dataset'] and (s['scene'] in ['any','background'] or r['scene_type']==s['scene'])]
  assert pool, f'Empty source-only preflight pool: {role}/{s}'
out.mkdir();cv2.setNumThreads(1)
recipe_path=Path(cfg['phase1_recipe']).resolve();recipe=yaml.safe_load(recipe_path.read_text());recipe['base_config']=str(recipe_path.parent/'easy.yaml');recipe['max_attempts_per_sample']=cfg['max_attempts_per_sample']
helper=recipe_path.parent.parent/'scripts/build_formal_dataset.py';spec=importlib.util.spec_from_file_location('phase1_formal_builder',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
for role,rs in roles.items():jsonl_write(out/f'{role}_sources.jsonl',rs)
provenance={'config':cfg,'config_sha256':sha(config_path),'source_index_sha256':sha(index),'source_audit_sha256':sha(audit_path),'phase1_recipe_sha256':sha(recipe_path),'phase1_code_commit':subprocess.check_output(['git','-C',str(recipe_path.parent.parent),'rev-parse','HEAD'],text=True).strip(),'phase25_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'code_diff':subprocess.check_output(['git','diff','HEAD','--','sarbench','scripts','configs'],text=True),'role_components':{k:sorted(v) for k,v in role_components.items()},'formal_test_components':sorted(components['test']),'source_protocol':'dev=train minus reserved confirmation components; validation=original val; test never sampled; decoded exact duplicates canonicalized','confirmation':'Index reservation only; no confirmation images generated or inspected by a matcher. Future confirmation claims remain conditional on unknown source relationships and synthetic-proxy validity.'}
save_json(out/'provenance.json',provenance)
# Texture selection is image-only dataset construction before algorithm evaluation.
# Relative low-texture quartile is computed separately within each source-role pool.
for role in ['dev','val']:
 rs=roles[role];texture={}
 for r in rs:
  im=cv2.imread(r['image_path'],0)
  if im is None:raise FileNotFoundError(r['image_path'])
  small=cv2.resize(im,(128,128),interpolation=cv2.INTER_AREA);blur=cv2.GaussianBlur(small,(5,5),1.)
  grad=cv2.magnitude(cv2.Sobel(blur,cv2.CV_32F,1,0),cv2.Sobel(blur,cv2.CV_32F,0,1))
  texture[r['image_path']]=float(np.percentile(grad,75))
 thresholds={}
 for ds,scene in {('hrsid','offshore')}:
  vals=[texture[r['image_path']] for r in rs if r['dataset']==ds and r['scene_type']==scene];assert vals
  thresholds[f'{ds}|{scene}']=float(np.quantile(vals,cfg['low_texture_quantile']))
 root=out/role;root.mkdir();manifest=[];rejections=[];source_hashes={};checks=[];seq=0
 for ci,stratum in enumerate(cfg['strata']):
  case=f'{ci:02d}_{stratum["scene"]}_{stratum["dataset"]}_{stratum["profile"]}_{stratum.get("texture","any")}'
  pool=[r for r in rs if r['dataset']==stratum['dataset'] and (stratum['scene'] in ['any','background'] or r['scene_type']==stratum['scene'])]
  if stratum.get('texture')=='low':pool=[r for r in pool if texture[r['image_path']]<=thresholds[f'{stratum["dataset"]}|{stratum["scene"]}']]
  if stratum['ships']=='present':pool=[r for r in pool if r['annotations']]
  if not pool:raise ValueError(f'Empty construction pool: {role}/{case}')
  bycomponent=collections.defaultdict(list)
  for r in sorted(pool,key=lambda r:r['source_id']):bycomponent[source_component(r)].append(r)
  rng=np.random.default_rng(stable_seed(cfg['seed'],role+case));groups=sorted(bycomponent);groups=[groups[i] for i in rng.permutation(len(groups))];orders={g:[bycomponent[g][i] for i in rng.permutation(len(bycomponent[g]))] for g in groups};cursor=collections.Counter();draw=0
  effective=mod.resolved_cfg(recipe,stratum);effective['views']=2
  save_json(root/'configs'/f'{case}.json',effective)
  for sample in range(stratum['count']):
   for trial in range(cfg['max_attempts_per_sample']):
    comp=groups[draw%len(groups)];draw+=1;row=orders[comp][cursor[comp]%len(orders[comp])];cursor[comp]+=1;seed=stable_seed(cfg['seed'],f'{role}:{case}:{sample}:{trial}:{row["source_id"]}')
    with tempfile.TemporaryDirectory(prefix='phase25_generate_') as temporary:
     temp=Path(temporary)/'sample'
     try:synthesize(row,effective,temp,seed)
     except (RuntimeError,ValueError,cv2.error) as exc:
      rejections.append({'case':case,'sample':sample,'trial':trial,'source_id':row['source_id'],'reason':str(exc)});continue
     meta=json.loads((temp/'meta.json').read_text());ships=any(meta['ship_annotations']);dynamic=any(meta['dynamic_events'][1:])
     if (stratum['ships']=='present' and not ships) or (stratum['ships']=='absent' and ships) or (stratum['profile']=='dynamic' and not dynamic):
      rejections.append({'case':case,'sample':sample,'trial':trial,'source_id':row['source_id'],'reason':'stratum ships/dynamics not satisfied'});continue
     qa=validate_pair(temp);assert qa['passed'],qa
     assert meta['source_resize_factors_xy']==[1.,1.]
     # Independent point direction and static support checks; no matching algorithm is involved.
     worst=0.
     for pair in meta['pairs']:
      with np.load(temp/pair['matches']) as points:x=points['points_i'];y=points['points_j']
      overlap=min(pair['overlap_fraction_i'],pair['overlap_fraction_j']);assert effective['min_overlap']-1e-6<=overlap<=effective['max_overlap']+1e-6
      assert len(x)==(0 if stratum['profile']=='none' else 512)
      if len(x):worst=max(worst,float(np.max(np.linalg.norm(perspective(pair['H_i_to_j'],x)-y,axis=1))))
     assert worst<1e-3
     # Blank annotated support cannot be created by changing the source annotations.
     if stratum['ships']=='absent':assert not row['annotations']
     folder=root/role/f'pair_{seq:06d}';folder.parent.mkdir(exist_ok=True);shutil.move(str(temp),folder)
     manifest.append({'sample_id':folder.name,'folder':str(folder.relative_to(root)),'profile':stratum['profile'],'scene_stratum':stratum['scene'],'texture_stratum':stratum.get('texture','unrestricted'),'source_texture_grad75':texture[row['image_path']],'dataset':row['dataset'],'source_id':row['source_id'],'source_group':row['source_group'],'source_component':source_component(row),'source_split':row['split'],'source_path':row['image_path'],'seed':seed,'config':f'configs/{case}.json','stratum':case,'output_size':meta['output_size'],'num_views':2})
     checks.append({'id':folder.name,'qa':qa,'max_gt_error':worst,'has_ship':ships,'dynamic':dynamic})
     if row['image_path'] not in source_hashes:
      im=cv2.imread(row['image_path'],0);source_hashes[row['image_path']]=hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()
     seq+=1;break
   else:raise RuntimeError(f'Sampling exhausted: {role}/{case}/{sample}; rejected={rejections[-8:]}')
  jsonl_write(root/'manifest.jsonl',manifest);save_json(root/'generation.json',{'generated':len(manifest),'rejections':rejections,'texture_thresholds':thresholds,'sampling':'component-balanced shuffled source cycles; explicit ship/dynamic rejection; relative source texture quartile before any model evaluation'})
  print(f'{role} {case}: {stratum["count"]} groups; {len(groups)} available source components; total {seq}',flush=True)
 assert seq==384
 save_json(root/'source_decoded_hashes.json',source_hashes)
 report={'accepted':True,'groups':seq,'relations':seq,'positive':sum(m['profile']!='none' for m in manifest),'negative':sum(m['profile']=='none' for m in manifest),'profiles':dict(collections.Counter(m['profile'] for m in manifest)),'unique_sources':len(source_hashes),'source_components':len({m['source_component'] for m in manifest}),'background_components':sorted({m['source_component'] for m in manifest if m['dataset']=='background'}),'max_gt_error':max(c['max_gt_error'] for c in checks),'checks':checks,'split':role,'source_index_sha256':sha(index)}
 save_json(root/'acceptance.json',report)
 files=[{'path':str(f.relative_to(root)),'sha256':sha(f)} for f in sorted(root.rglob('*')) if f.is_file()];save_json(root/'file_checksums.json',files);save_json(root/'ACCEPTED_phase25.json',{'accepted':True,'validation_report':str(root/'acceptance.json'),'sha256':sha(root/'acceptance.json')})
# Independent decoded hashes of selected source images must not overlap between roles.
devhash=json.loads((out/'dev/source_decoded_hashes.json').read_text());valhash=json.loads((out/'val/source_decoded_hashes.json').read_text());assert not set(devhash.values())&set(valhash.values())
save_json(out/'isolation.json',{'accepted':True,'dev_val_component_intersection':[],'dev_confirmation_component_intersection':[],'val_confirmation_component_intersection':[],'any_test_component_intersection':[],'selected_dev_val_decoded_duplicates':0,'available_source_components':{k:len(v) for k,v in role_components.items()},'source_index_sha256':sha(index),'confirmation_source_records':len(roles['confirmation']),'confirmation_images_generated':0,'known_near_duplicate_audit':str(audit_path),'known_accepted_cross_split_edges_remaining':0,'limitation':'Source audit is not exhaustive; source component independence is based on available identifiers and pixel-supported evidence.'})
print(json.dumps({'out':str(out),'dev':384,'validation':384,'confirmation_components':len(reserve),'confirmation_images_generated':0},indent=2))
