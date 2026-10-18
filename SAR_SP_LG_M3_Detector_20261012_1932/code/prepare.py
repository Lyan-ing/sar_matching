"""Pre-register new component-isolated development validation; no formal-test reads."""
from m3_common import *
from collections import defaultdict,Counter
def main():
 assert not (ROOT/'config.json').exists(),'Never overwrite a registered experiment'
 train=rows(OLD/'train_sources.jsonl');vals=rows(OLD/'val_sources.jsonl');oldplan=json.loads((OLD/'validation_plan.json').read_text())
 old={component(x['source']) for x in oldplan};tc={component(r) for r in train};vc={component(r) for r in vals};assert not tc&vc
 available=[r for r in vals if component(r) not in old];groups=defaultdict(list)
 for r in available:groups[(component(r),r['scene_type'] if r['dataset']=='hrsid' else r['dataset'])].append(r)
 chosen=[]
 for key,pool in sorted(groups.items()):
  pool=sorted(pool,key=lambda r:sha_text(r['source_id']+'26101203'))
  chosen.extend(pool if key[1]=='inshore' else pool[:1])
 plan=[]
 for i,r in enumerate(chosen):
  for j,regime in enumerate(['easy','medium','low']):
   plan.append(dict(id=f'm3val_{i:03d}_{regime}',source=r,seed=83000+i*10+j,regime=regime,dynamic_occlusion=bool(r['annotations']) and (i+j)%4==0))
 large=defaultdict(list)
 for r in available:
  if r['native_width']>=768 and r['native_height']>=768:large[component(r)].append(r)
 for c,pool in sorted(large.items()):
  for i,r in enumerate(sorted(pool,key=lambda r:r['source_id'])[:8]):plan.append(dict(id=f'm3val_none_{c.split(":")[-1]}_{i}',source=r,seed=84000+len(plan),regime='none',dynamic_occlusion=False))
 assert not {component(p['source']) for p in plan}&(tc|old)
 atomic_json(ROOT/'validation_plan.json',plan)
 for spec in plan:
  z=pair(spec['source'],spec['seed'],spec['regime'],dynamic_occlusion=spec['dynamic_occlusion']);p=ROOT/'validation_data'/spec['id'];p.mkdir(parents=True)
  for i,im in enumerate(z['images']):assert cv2.imwrite(str(p/f'image{i}.png'),im)
  np.savez_compressed(p/'gt.npz',H_A=z['H_A'],H_B=z['H_B'],H_AB=z['H_AB'],static0=z['static'][0],static1=z['static'][1],valid0=z['valid'][0],valid1=z['valid'][1],points0=z['points0'],points1=z['points1'])
  texture=np.std(z['images'][0][z['static'][0]]) if z['static'][0].any() else 0
  atomic_json(p/'metadata.json',dict(id=spec['id'],source_id=spec['source']['source_id'],source_component=component(spec['source']),dataset=spec['source']['dataset'],scene=spec['source']['scene_type'],regime=spec['regime'],overlap=z['overlap'],dynamic=spec['dynamic_occlusion'],low_texture=texture<12,static_texture_std=float(texture),no_overlap=spec['regime']=='none',input_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in z['images']]))
  if spec['regime']=='none':assert z['overlap']==0,'Negative pair actually overlaps'
 config=dict(run_id=ROOT.name,variant='m3',seed=26101203,physical_gpu=5,input_size=256,trainable=['sp.convPa.weight','sp.convPa.bias','sp.convPb.weight','sp.convPb.bias'],microbatch=1,batch_accumulation=2,learning_rate=3e-5,weight_decay=1e-4,precision='FP32',background_weight=.1,equivariance_weight=.2,preserve_weight=.1,initial_step_budget=3000,allow_evidence_based_extension=True,extension_steps=1000,extension_rule='at budget: current selected category-balanced component S5 > M2; last 3 validations improve S5 by >=0.5pp AND mean correct static LG matches; no increase in no-overlap false accepts; 4h cap remains',max_training_seconds=14400,validate_every=250,checkpoint_every=100,early_stop_min_steps=1000,early_stop_patience=4,selection_rule=['maximize available-category-balanced component macro Success@5','minimize no-overlap false accept rate','minimize positive incorrect accept rate','maximize correct static LG matches with hull>=.01 and count>=8','maximize mean detector repeatability@3'],NMS_radius=4,detector_threshold=.0005,max_keypoints=1024,geometry='unchanged historical Homography/RANSAC3/full gate',curriculum='steps<500 easy;500:1500 medium;1500+ easy20/medium60/low20',sampling='55%nearshore35%offshore10%background, component-uniform then source',anchors='canonical Gaussian sigma1.2 ShiTomasi(quality.01,minDistance8,block5,eigen>=.00002); known H projects; common static; frozen M2 corresponding descriptor cosine>=.60; quota<=4 per 4x4 cell; at most one per 8x8 cell in each view',no_ground_truth_in_prediction=True)
 atomic_json(ROOT/'config.json',config)
 from m3_model import M2,M2_SHA
 identity=dict(created=now(),index_sha256=sha(BASE/'MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl'),parent_checkpoint=str(M2),parent_sha256=M2_SHA,train_sources_file=str(OLD/'train_sources.jsonl'),train_sources_sha256=sha(OLD/'train_sources.jsonl'),val_sources_sha256=sha(OLD/'val_sources.jsonl'),old_validation_plan_sha256=sha(OLD/'validation_plan.json'),new_validation_plan_sha256=sha(ROOT/'validation_plan.json'),train_components=len(tc),old_val_components=len(old),new_available_sources=len(available),new_available_components=len({component(r) for r in available}),new_val_components=len({component(s['source']) for s in plan}),train_newval_intersection=[],oldval_newval_intersection=[],strata=dict(Counter(s['source']['dataset']+':'+s['source']['scene_type']+':'+s['regime'] for s in plan)),new_background_components=0,new_inshore_components=len({component(s['source']) for s in plan if s['source']['scene_type']=='inshore'}),limitations=['new validation is development/model selection, not untouched confirmation','SSDD raw acquisition independence unproven','no new background components; only one inshore component','no frozen900 or realV2 used in selection'],official_weights={f:sha(BASE/'MaritimeSARRegistrationPhase2/cache/torch/hub/checkpoints'/f) for f in ['superpoint_v1.pth','superpoint_lightglue_v0-1_arxiv.pth']})
 atomic_json(ROOT/'data_identity.json',identity)
 real=[]
 for r in rows(BASE/'RealSAR_AngleExpansion_20261012_1641/image_only_pairs.jsonl'):
  folder=BASE/'RealSAR_AngleExpansion_20261012_1641/regions'/r['sample_id']
  if not (folder/'native_db0.png').exists():folder=BASE/'RealSAR_MultiViewV2_20261012_1259/regions'/r['sample_id']
  paths=[folder/f'native_db{i}.png' for i in [0,1]];assert all(p.exists() for p in paths)
  real.append(dict(id=r['sample_id'],paths=[str(p) for p in paths],input_sha256=[sha(p) for p in paths],independent_GT=False))
 atomic_json(ROOT/'real_inputs.json',real);print(json.dumps(dict(validation_pairs=len(plan),positive=sum(s['regime']!='none' for s in plan),components=identity['new_val_components'],strata=identity['strata']),ensure_ascii=False))
def sha_text(x):return hashlib.sha256(x.encode()).hexdigest()
if __name__=='__main__':main()
