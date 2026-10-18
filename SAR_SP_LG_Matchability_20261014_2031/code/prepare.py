from common import *
from collections import defaultdict,Counter
from pairdata import pair
def category(r):return r['scene_type'] if r['dataset']=='hrsid' else r['dataset']
def main():
 assert not output('config').exists()
 train=readrows(OLD/'train_sources.jsonl');val=readrows(OLD/'val_sources.jsonl');tc={component(r) for r in train};vc={component(r) for r in val};assert len(train)==5111 and len(val)==865 and not tc&vc
 groups=defaultdict(list)
 for r in val:groups[(component(r),category(r))].append(r)
 # Component-level stratification; historic Val is development data, not new independent acquisition.
 strata=defaultdict(list)
 for c in sorted(vc):
  cats={category(r) for r in val if component(r)==c};cat=next(x for x in ['inshore','background','offshore','ssdd'] if x in cats);strata[cat].append(c)
 select=set();rng=np.random.default_rng(26101431)
 for cat,cs in sorted(strata.items()):select.update(np.array(cs)[rng.permutation(len(cs))[:max(1,round(.7*len(cs)))]] )
 plan=[]
 for ci,((c,cat),rs) in enumerate(sorted(groups.items())):
  rs=sorted(rs,key=lambda r:r['source_id']);order=rng.permutation(len(rs))
  for j,regime in enumerate(['easy','medium','low']):
   r=rs[order[j%len(order)]];size=128 if min(r['native_width'],r['native_height'])<256 else (384 if j==1 else 256)
   plan.append(dict(id=f'v_{ci:03d}_{regime}',source=r,source_component=c,category=cat,regime=regime,size=size,seed=941000+ci*10+j,split='select' if c in select else 'holdout',dynamic_occlusion=bool(r['annotations']) and j==2))
 large=[r for r in val if min(r['native_width'],r['native_height'])>=768]
 assert len(large)>=24
 for i,r in enumerate(large[:32]):plan.append(dict(id=f'v_none_{i:03d}',source=r,source_component=component(r),category=category(r),regime='none',size=256,seed=942000+i,split='select' if component(r) in select else 'holdout',dynamic_occlusion=False))
 for s in plan:
  z=pair(s['source'],s['seed'],s['regime'],size=s['size'],dynamic_occlusion=s['dynamic_occlusion']);folder=ROOT/'validation_data'/s['id'];folder.mkdir(parents=True)
  for i,im in enumerate(z['images']):cv2.imwrite(str(folder/f'image{i}.png'),im)
  with output('gt',ext='npz',folder=folder).open('xb') as f:np.savez_compressed(f,H_AB=z['H_AB'],H_A=z['H_A'],H_B=z['H_B'],static0=z['static'][0],static1=z['static'][1],valid0=z['valid'][0],valid1=z['valid'][1])
  texture=float(np.std(z['images'][0][z['static'][0]])) if z['static'][0].any() else 0
  meta=dict(id=s['id'],source_id=s['source']['source_id'],source_component=s['source_component'],dataset=s['source']['dataset'],scene=s['source']['scene_type'],category=s['category'],regime=s['regime'],split=s['split'],overlap=z['overlap'],low_texture=texture<12,dynamic=s['dynamic_occlusion'],no_overlap=s['regime']=='none',input_size=s['size'],relative_2D_rotation_deg=z['relative_rotation'],source_origin=z['source_origin_xy'])
  write_new(output('metadata',folder=folder),meta)
  assert np.allclose(z['H_AB'],z['H_B']@np.linalg.inv(z['H_A']))
  if s['regime']=='none':assert z['overlap']==0
 write_new(output('validation_plan'),plan)
 # A complete first pass covers every source, followed by maritime-balanced repeated sources.
 schedule=list(map(int,np.random.default_rng(26101432).permutation(len(train))));pools=defaultdict(list)
 for i,r in enumerate(train):pools[category(r)].append(i)
 rng=np.random.default_rng(26101433)
 while len(schedule)<10000:
  cat=str(rng.choice(['inshore','offshore','ssdd','background'],p=[.4,.4,.15,.05]));schedule.append(int(rng.choice(pools[cat])))
 write_new(output('source_schedule'),schedule)
 c=dict(seed=26101434,steps=5000,accumulation=2,detector_lr=3e-6,descriptor_lr=1e-5,shared_lr=1e-6,weight_decay=1e-4,temperature=.1,preserve_weight=.1,quality_weight=.2,validate_every=1000,checkpoint_every=250,precision='FP32, TF32 disabled',variants=['control','joint','frozen'],input_sizes=[128,256,384],max_input=512,NMS=4,threshold=.0005,max_keypoints=1024,equal_cap=128,geometry='historical Homography RANSAC3/5000/.995 + full gate',selection='select components only: hard blank0/noise<=M2/density<=1.5M2/negativeFPR<=.05; category-component macro S5 then correct static matches then repeatability',no900=True,real_for_selection=False,initialization=str(FIX/'runs/m2/best.ckpt'),train_pool_fixed=True,wall_seconds=14400)
 write_new(output('config'),c)
 write_new(output('data_identity'),dict(train_records=len(train),val_records=len(val),train_components=len(tc),val_components=len(vc),select_components=len(select),holdout_components=len(vc-select),split_intersection=[],val_selection_holdout_intersection=[],validation_pairs=len(plan),categories=dict(Counter(s['category'] for s in plan)),train_source_sha=sha(OLD/'train_sources.jsonl'),val_source_sha=sha(OLD/'val_sources.jsonl'),index_sha=sha(BASE/'MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl'),schedule_sha=sha(output('source_schedule')),planned_pair_calls=10000,planned_views=20000,all_training_sources_in_first_5111_calls=True,caveats=['all available known val components previously used in development','SSDD raw acquisition identity incomplete','no harbor-specific synthetic label available; unknown preserved']))
 print(json.dumps(dict(validation_pairs=len(plan),components=len(vc),select=len(select),holdout=len(vc-select),categories=dict(Counter(s['category'] for s in plan)))))
if __name__=='__main__':main()
