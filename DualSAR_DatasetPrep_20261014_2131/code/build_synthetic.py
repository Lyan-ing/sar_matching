"""Materialize source-isolated <=512 synthetic pairs using existing sarreg engine/QA."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json,sys,copy,hashlib,os,argparse,time
import numpy as np,cv2
BASE=Path('/home/yl/matching/sar_matching');ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'MaritimeSARRegDataset'))
from sarreg.synthesis import synthesize,load_cfg,stable_seed
from sarreg.qa import validate_pair
from sarreg.io import read_gray,write_png
from sarreg.geometry import transform_points

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def jsonl(p,rows):
 with p.open('x',encoding='utf-8') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def readrows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def profiles():
 easy=load_cfg(BASE/'MaritimeSARRegDataset/configs/easy.yaml');easy.update(output_size=512,views=2,allow_source_upsampling=False,output_size_multiple=32,center_mode='ship',correspondences_per_pair=512)
 medium=load_cfg(BASE/'MaritimeSARRegDataset/configs/default.yaml');medium.update(easy|medium)
 medium.update(output_size=512,views=2,allow_source_upsampling=False,output_size_multiple=32,center_mode='ship',correspondences_per_pair=512)
 rotation=copy.deepcopy(medium);rotation.update(rotation_degrees=60,scale_min=.9,scale_max=1.1,shear_degrees=0,perspective_fraction=0,min_source_side_factor=1.6,min_overlap=.40,max_overlap=.9)
 low=copy.deepcopy(medium);low.update(rotation_degrees=12,shear_degrees=2,perspective_fraction=.003,min_source_side_factor=2,min_overlap=.08,max_overlap=.25,min_valid_fraction=.8,max_sampling_attempts=360)
 none=copy.deepcopy(easy);none.update(rotation_degrees=3,min_source_side_factor=3,min_overlap=0.,max_overlap=0.,max_sampling_attempts=360)
 dynamic=copy.deepcopy(medium);dynamic.update(rotation_degrees=10,shear_degrees=1,perspective_fraction=.001,min_source_side_factor=1.6,min_overlap=.6,max_overlap=.9,dynamic_ships=dict(enabled=True,move_probability=1.,max_shift_pixels=12,min_area=12))
 return dict(easy=easy,medium=medium,rotation=rotation,low=low,none=none,dynamic=dynamic)

def jacobian(H,p):
 H=np.asarray(H,float);x,y=p;n=H@np.array([x,y,1.]);z=n[2]
 return (H[:2,:2]*z-n[:2,None]*H[2:3,:2])/(z*z)

def worker(item):
 cv2.setNumThreads(1);idx,r,profile,cfg,out,global_seed=item;start=time.monotonic();attempts=[]
 if not r['annotations']:cfg=copy.deepcopy(cfg);cfg['center_mode']='uniform'
 sid=f"syn_{idx:06d}";parent=Path(out)/r['split']/sid;parent.mkdir(parents=True,exist_ok=False)
 for trial in range(4):
  folder=parent/f'attempt_{trial:02d}';folder.mkdir()
  seed=stable_seed(global_seed,f"{r['dataset']}:{r['source_id']}:{profile}:{trial}")
  try:
   synthesize(r,cfg,folder,seed)
   qa=validate_pair(folder)
   if not qa['passed']:raise ValueError('existing QA failed: '+repr(qa['issues']))
   meta=json.loads((folder/'meta.json').read_text());size=meta['output_size']
   assert size<=512 and meta['source_resize_factors_xy']==[1.,1.] and meta['registration_type']=='synthetic_2d_registration_proxy'
   static=[]
   for i in [0,1]:
    v=read_gray(folder/f'valid_{i}.png')>0;s=read_gray(folder/f'ships_{i}.png');d=read_gray(folder/f'dynamic_{i}.png')>0
    # Reuse existing ship masks; static supervision excludes all annotated vessels, not only moved ones.
    mask=v&~(cv2.dilate(s,np.ones((11,11),np.uint8))>0)&~d
    static.append(mask);write_png(folder/f'static_{i}.png',mask.astype('uint8')*255)
   p=meta['pairs'][0];H=np.asarray(p['H_i_to_j']);old=np.load(folder/p['matches']);a=old['points_i'];b=old['points_j']
   keep=np.ones(len(a),bool)
   for xy,m in zip([a,b],static):
    ij=np.rint(xy).astype(int);keep&=m[ij[:,1],ij[:,0]]
   a,b=a[keep],b[keep]
   assert not len(a) or np.linalg.norm(transform_points(H,a)-b,axis=1).max()<1e-3
   np.savez_compressed(folder/'matches_static_0_1.npz',points_i=a,points_j=b)
   J=jacobian(H,[(size-1)/2,(size-1)/2]);U,_,VT=np.linalg.svd(J);rotation=U@VT
   angle=abs(float(np.degrees(np.arctan2(rotation[1,0],rotation[0,0]))))
   imgs=[read_gray(folder/f'view_{i}.png') for i in [0,1]]
   sf=[]
   for i in [0,1]:
    if (folder/f'sea_{i}.png').exists():
     valid=read_gray(folder/f'valid_{i}.png')>0;sf.append(float((read_gray(folder/f'sea_{i}.png')[valid]>0).mean()))
   row=dict(sample_id=sid,split=r['split'],construction_pool=r['construction_pool'],source_dataset=r['dataset'],source_id=r['source_id'],
    source_path=r['image_path'],known_component=r['known_component'],canonical_source_path=r['canonical_image_path'],
    source_isolation_strength=r['isolation_strength'],source_scene_category=r['scene_category'],source_scene_status=r['scene_category_status'],
    pixel_sea_fraction=r['pixel_sea_fraction'],view_pixel_sea_fraction=sf,port_dense_candidate=r['port_dense_candidate'],
    width=size,height=size,num_views=2,pair_relations=1,profile=profile,seed=seed,
    view0=str(folder/'view_0.png'),view1=str(folder/'view_1.png'),metadata=str(folder/'meta.json'),
    valid_masks=[str(folder/f'valid_{i}.png') for i in [0,1]],static_supervision_masks=[str(folder/f'static_{i}.png') for i in [0,1]],
    static_correspondences=str(folder/'matches_static_0_1.npz'),static_check_count=len(a),
    H_0_to_1=p['H_i_to_j'],overlap_fractions=[p['overlap_fraction_i'],p['overlap_fraction_j']],
    geometric_label='known_synthetic_no_overlap' if profile=='none' else 'known_synthetic_overlap',
    image_rotation_2D_deg=angle,real_radar_azimuth_delta_deg=None,
    synthetic_dynamic_events=sum(map(len,meta['dynamic_events'])),visible_ship_counts=list(map(len,meta['ship_annotations'])),
    valid_fractions=[float((read_gray(folder/f'valid_{i}.png')>0).mean()) for i in [0,1]],
    static_fractions=[float(m.mean()) for m in static],view_std=[float(im.std()) for im in imgs],
    low_texture_or_static_structure_unverified=True,QA_passed=True,source_upsampling=False,
    physical_multiview_GT=False,supervision='exact 2D proxy only; use ship-excluded static masks; sea-only geometric GT need not have matchable texture',
    weak_component_extension_not_physical_acquisition_independence=r['dataset']=='ssdd')
   save(folder/f'static_supervision_record_{stamp()}.json',dict(static_pixels=[int(m.sum()) for m in static],static_matches=len(a),all_annotated_vessels_excluded=True,guard_pixels=5,metadata_unmodified=True))
   return dict(ok=True,row=row,attempts=attempts,elapsed_seconds=time.monotonic()-start)
  except Exception as exc:
   attempts.append(dict(trial=trial,seed=seed,reason=repr(exc),preserved_folder=str(folder)))
   save(folder/f'generation_failure_{stamp()}.json',attempts[-1])
 return dict(ok=False,source_id=r['source_id'],dataset=r['dataset'],split=r['split'],profile=profile,attempts=attempts,elapsed_seconds=time.monotonic()-start)

def main():
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args();T=stamp()
 pool=readrows(next(ROOT.glob('SYNTHETIC_SOURCE_POOL_*.jsonl')))
 screen=readrows(next(ROOT.glob('SCENE_SCREEN_ENTRIES_*.jsonl')))
 ports={r['image_path'] for r in screen[:24]};urban={screen[i]['image_path'] for i in [24,25,26,27,28,29,32,34,35,36,37,38,39,40,41,42,43,46]}
 for r in pool:
  if r['image_path'] in ports:r.update(scene_category='port_quay_dense_vessels',scene_category_status='AI_visual_provisional_source_context_not_manual_GT')
  elif r['image_path'] in urban:r.update(scene_category='other_land_structures',scene_category_status='AI_visual_provisional_urban_source_context_not_manual_GT')
 ps=profiles();selected=pool
 if a.smoke:
  selected=[next(r for r in pool if r['dataset']==d and r['split']==s) for d in ['hrsid','ssdd','background'] for s in ['train','val']]*2
 out=ROOT/(('synthetic_preflight_' if a.smoke else 'synthetic_materialized_')+T);out.mkdir(exist_ok=False)
 items=[]
 for i,r in enumerate(selected):
  k=stable_seed(20261014,r['dataset']+':'+r['source_id'])%20
  profile=['easy']*7+['medium']*6+['rotation']*3+['low']*2+['none','dynamic']
  name=list(ps)[i%len(ps)] if a.smoke else profile[k]
  items.append((i,r,name,ps[name],str(out),20261014))
 save(out/f'GENERATION_CONFIG_{T}.json',dict(at=now(),profiles=ps,global_seed=20261014,workers=a.workers,opencv_threads_per_worker=1,
  source_manifest=str(next(ROOT.glob('SYNTHETIC_SOURCE_POOL_*.jsonl'))),source_scene_review='SCENE_SCREEN 48 sources; ports 0..23, urban explicit indices; AI provisional',
  source_selection_without_replacement=True,attempts_max_per_source=4,existing_engine=str(BASE/'MaritimeSARRegDataset/sarreg/synthesis.py'),
  engine_sha256=hashlib.sha256((BASE/'MaritimeSARRegDataset/sarreg/synthesis.py').read_bytes()).hexdigest(),
  no_source_test_used=True,static_masks_exclude_vessels=True,no_network_training=True))
 jsonl(out/f'SOURCES_SELECTED_{T}.jsonl',selected)
 results=[];start=time.monotonic()
 with ProcessPoolExecutor(max_workers=a.workers) as executor:
  for j,res in enumerate(executor.map(worker,items,chunksize=1),1):
   results.append(res)
   if j%100==0 or j==len(items):print(json.dumps(dict(at=now(),completed=j,total=len(items),passed=sum(x['ok'] for x in results),elapsed_seconds=time.monotonic()-start)),flush=True)
 success=[r['row'] for r in results if r['ok']];finished_stamp=stamp()
 for split in ['train','val']:
  for name,rr in [('all',[r for r in success if r['split']==split]),('parent_core',[r for r in success if r['split']==split and r['construction_pool']=='parent_component_core']),('ssdd_extension',[r for r in success if r['split']==split and r['construction_pool']!='parent_component_core'])]:
   jsonl(out/f'{split}_{name}_manifest_{finished_stamp}.jsonl',rr)
 jsonl(out/f'GENERATION_RESULTS_{finished_stamp}.jsonl',results)
 summary=dict(at=now(),requested=len(items),generated_pairs=len(success),generated_view_instances=2*len(success),
  distinct_known_deduplicated_source_images=len({r['canonical_source_path'] for r in success}),
  counts=dict(Counter(r['split']+'/'+r['construction_pool'] for r in success)),
  scenes=dict(Counter(r['split']+'/'+r['source_scene_category'] for r in success)),
  sizes=dict(Counter(r['width'] for r in success)),profiles=dict(Counter(r['profile'] for r in success)),
  actual_dynamic_pairs=sum(r['synthetic_dynamic_events']>0 for r in success),
  known_synthetic_negative_pairs=sum(r['profile']=='none' for r in success),
  failed_sources=[r for r in results if not r['ok']],rejected_attempt_count=sum(len(r['attempts']) for r in results),
  QA_passed_count=sum(r['QA_passed'] for r in success),elapsed_seconds=time.monotonic()-start,
  independent_real_GT_created=0,no_model_training_started=True,no_downloads=True)
 save(out/f'GENERATION_SUMMARY_{finished_stamp}.json',summary)
 print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
