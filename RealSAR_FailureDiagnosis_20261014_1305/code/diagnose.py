"""Read frozen caches; CPU-only geometry counterfactuals, never real GT accuracy."""
import sys, json, hashlib, time, platform, csv as csvlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parents[1]
TS=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
BASE=ROOT/'SAR_SP_LG_StructureStable_20261013_2210'
sys.path.insert(0,str(ROOT/'MaritimeSARRegistrationPhase2'))
from sarbench.matchers import Prediction
from sarbench.estimate import estimate,perspective
from sarbench.reliability import geometry_diagnostics,gate,spatial
SEED=42
IDS=['ANGLE_e76cc299fa7b__native_000','ANGLE_e3bfffe1139e__native_000',
     'ANGLE_75c261e08eae__native_000','ANGLE_5da40401a853__native_000',
     'S1SF_EXTENSION_000__mother_000','Capella_0006__mother_000']
used={}
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def tracked(path):
 path=Path(path);used[str(path)]=sha(path);return path
def load(path):return json.loads(tracked(path).read_text())
def write(name,data):
 with (OUT/f'{name}_{TS}.json').open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def csv(name,rows):
 with (OUT/'tables'/f'{name}_{TS}.csv').open('x',newline='') as f:
  writer=csvlib.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def table(name):
 with tracked(BASE/name).open() as f:rows=list(csvlib.DictReader(f))
 for row in rows:
  for k,v in row.items():
   if v in ['True','False']:row[k]=(v=='True')
   else:
    try:row[k]=float(v)
    except ValueError:pass
 return rows
def avg(rows,key):
 values=[r[key] for r in rows if isinstance(r[key],(int,float,bool)) and np.isfinite(r[key])]
 return float(np.mean(values)) if values else None
def fit(p0,p1,sh0,sh1,model):
 pred=Prediction('cached',p0,p1)
 e=estimate(pred,sh0,model=model,seed=SEED)
 d=geometry_diagnostics(e,pred,sh0,sh1,seed=SEED)
 g=gate(e,d,'full')
 return e,g,d
def test_geometry():
 rng=np.random.default_rng(SEED);p=rng.uniform(20,450,(60,2)).astype('float32')
 for model,H in [('affine',np.array([[1,.05,12],[-.03,1,8],[0,0,1.]])),
                 ('homography',np.array([[1,.05,12],[-.03,1,8],[.0001,.0002,1.]]))]:
  q=perspective(H,p).astype('float32');e,g,d=fit(p,q,(512,512),(512,512),model)
  assert e.n_inliers==60 and g.accepted and np.max(np.linalg.norm(perspective(e.candidate_H,p)-q,axis=1))<.01
  origin=np.array([100,90]);local=estimate(Prediction('local',p-origin,q),(128,128),model=model)
  assert np.max(np.linalg.norm(perspective(local.candidate_H,p-origin)-q,axis=1))<.01
 e,g,d=fit(p[:3],p[:3],(512,512),(512,512),'affine');assert e.reason=='too_few_matches'
 return dict(analytical_affine=True,analytical_homography=True,local_origin_conversion=True,insufficient_match_rejection=True)
def main():
 start=time.perf_counter();cv2.setNumThreads(1);np.random.seed(SEED)
 write('preregister',dict(seed=SEED,selected_ids=IDS,methods=['m0','m2','b'],selection_basis='Four existing physical preparation tasks plus two same-look controls; no algorithm scores used',
  input='Frozen strict common-count Native match pools',models=['affine','homography'],ransac_px=3,min_matches=8,min_inliers=8,min_ratio=.3,full_gate='existing sarbench.reliability full',
  local='Fixed reference 4x4 cells; >=8 matches; affine basic gate, reference coordinates translated to cell origin; not global correctness',
  split_fit='one seeded random 60/40 partition, >=8 fit points; evaluate ALL held matches, never choose by residual; not independent GT',training=False,new_feature_inference=False,real_GT_access=False))
 tests=test_geometry();real=load(BASE/'real_equal_results_20261014_0028.json')['rows'];natural=load(BASE/'real_results_20261013_2346.json')['rows'];refs={r['id']:r for r in load(BASE/'real_inputs_20261013_2210.json')}
 summaries=[];localrows=[];pointrows=[];maps={};shape_cache={}
 for row in real:
  data=np.load(tracked(row['output_npz']),allow_pickle=False);p0=data['points0'];p1=data['points1'];s0,s1=map(tuple,row['input_shapes']);shape_cache[row['id']]=(s0,s1)
  record={k:row.get(k) for k in ['id','method','site','sensor','look_category','n_keypoints0','n_keypoints1','n_matches','n_inliers','accepted','reason','decision','elapsed_ms']}
  record['independent_GT']=False
  for side,shape in enumerate([s0,s1]):
   for typ,key in [('keypoints',f'keypoints{side}'),('matches',f'points{side}')]:
    for k,v in spatial(data[key],shape).items():record[f'{typ}{side}_{k}']=v
  nat=next(r for r in natural if r['id']==row['id'] and r['method']==row['method'])
  record.update(natural_keypoints0=nat['n_keypoints0'],natural_keypoints1=nat['n_keypoints1'],natural_matches=nat['n_matches'],natural_inliers=nat['n_inliers'],natural_accepted=nat['accepted'])
  pointrows.append(record)
  if row['id'] not in IDS or row['method'] not in ['m0','m2','b']:continue
  for model in ['affine','homography']:
   e,g,d=fit(p0,p1,s0,s1,model);maps[(row['id'],row['method'],model)]=(e,g,d,p0,p1)
   n=len(p0);rng=np.random.default_rng(SEED);order=rng.permutation(n);nf=max(8,int(.6*n));trainids=order[:nf];heldids=order[nf:]
   err=[]
   if len(trainids)>=8 and len(heldids):
    train_e=estimate(Prediction('cached_split',p0[trainids],p1[trainids]),s0,model=model,seed=SEED)
    if train_e.candidate_H is not None:err=np.linalg.norm(perspective(train_e.candidate_H,p0[heldids])-p1[heldids],axis=1)
   finite=np.asarray(err)[np.isfinite(err)]
   summaries.append(dict(id=row['id'],method=row['method'],model=model,n_matches=n,n_inliers=e.n_inliers,inlier_ratio=e.inlier_ratio,fit_rmse_NOT_GT=e.inlier_rmse,
    basic_accepted=e.accepted,full_accepted=g.accepted,reason=g.reason,hull0=d.get('inlier0_hull_fraction'),grid0=d.get('inlier0_grid_coverage'),
    cycle_mean_NOT_GT=d.get('bidirectional_cycle_mean_px'),projection_pole=d.get('forward_denominator_same_sign')==False,
    split_fit_n=len(trainids),held_n=len(heldids),held_finite_n=len(finite),held_median_NOT_GT=float(np.median(finite)) if len(finite) else None,
    held_consensus_3px_NOT_GT=float(np.mean(np.isfinite(err)&(np.asarray(err)<=3))) if len(err) else None,independent_GT=False))
  for gy in range(4):
   for gx in range(4):
    h,w=s0;origin=np.array([gx*w/4,gy*h/4]);mask=(p0[:,0]>=origin[0])&(p0[:,0]<(gx+1)*w/4)&(p0[:,1]>=origin[1])&(p0[:,1]<(gy+1)*h/4)
    e=estimate(Prediction('cached_local',p0[mask]-origin,p1[mask]),(int(np.ceil(h/4)),int(np.ceil(w/4))),model='affine',seed=SEED)
    localrows.append(dict(id=row['id'],method=row['method'],cell_x=gx,cell_y=gy,n_matches=int(mask.sum()),n_inliers=e.n_inliers,basic_local_support=e.accepted,reason=e.reason,inlier_ratio=e.inlier_ratio,fit_rmse_NOT_GT=e.inlier_rmse,independent_GT=False))
 csv('real_point_support',pointrows);csv('geometry_counterfactual',summaries);csv('local_affine',localrows)
 val=table('tables/validation_per_pair_20261013_2210.csv');native=table('tables/native_validation_20261013_2210.csv')
 strata=[];budget=[]
 for field in ['split','scene','regime','actual_overlap_bin','low_texture','dynamic']:
  for value in sorted({str(r[field]) for r in val if not r['no_overlap']}):
   group=[r for r in val if not r['no_overlap'] and str(r[field])==value]
   for m in ['m0','m2','a','b','control','orb']:
    a=[r for r in group if r['method']==m];base={r['id']:r for r in group if r['method']=='m2'}
    strata.append(dict(field=field,value=str(value),method=m,n=len(a),components=len({r['source_component'] for r in a}),success5=avg(a,'success_5px'),
     repeatability=avg(a,'detector_repeatability_3px'),correct_LG_mean=avg(a,'LG_correct_3px'),descriptor_actual_end_to_end_R1=avg(a,'descriptor_end_to_end_recall1'),
     enough_correct_support=avg(a,'has_enough_correct_support'),gained=sum(r['success_5px'] and not base[r['id']]['success_5px'] for r in a),lost=sum(not r['success_5px'] and base[r['id']]['success_5px'] for r in a)))
 for n in ['strict','natural']:
  df=val if n=='strict' else native
  for m in sorted({r['method'] for r in df}):
   a=[r for r in df if not r['no_overlap'] and r['method']==m]
   budget.append(dict(protocol=n,method=m,n=len(a),success5=avg(a,'success_5px'),mean_points=(avg(a,'n_keypoints0')+avg(a,'n_keypoints1'))/2,correct_LG_mean=avg(a,'LG_correct_3px'),repeatability=avg(a,'detector_repeatability_3px')))
 csv('synthetic_strata',strata);csv('budget_comparison',budget)
 # Synthetic diagnostic cases selected deterministically, preserve all rows in CSV.
 m2=[r for r in val if r['method']=='m2' and not r['no_overlap']];b={r['id']:r for r in val if r['method']=='b' and not r['no_overlap']};cases=[]
 for r in m2:
  case={k:r[k] for k in ['id','scene','regime','low_texture','source_component','success_5px','repeatable_detected0','LG_correct_3px','descriptor_conditional_recall1','descriptor_end_to_end_recall1','reason']}
  case.update({k+'_b':b[r['id']][k] for k in ['success_5px','repeatable_detected0','LG_correct_3px','descriptor_conditional_recall1','reason']});cases.append(case)
 csv('synthetic_paired_cases',cases)
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(10,3.5))
 for ax,proto in zip(axes,['strict','natural']):
  records=[r for r in budget if r['protocol']==proto and r['method']!='orb'];x=np.arange(len(records));ax.bar(x,[100*r['success5'] for r in records],color=['#777777','#0072B2','#E69F00','#009E73','#CC79A7']);ax.set_xticks(x,[r['method'].upper() for r in records]);ax.set_ylim(0,100);ax.set_ylabel('Synthetic Success@5px (%)');ax.set_title(proto+' effective point protocol')
 fig.suptitle('Observed development validation: common-count vs natural-count (not real SAR accuracy)');fig.tight_layout();fig.savefig(OUT/'figures'/f'budget_{TS}.png',dpi=220);plt.close(fig)
 # Native keypoints and matches spatial plots: no GT or geographic windows.
 for id in IDS:
  images=[]
  for path,expected in zip(refs[id]['paths'],refs[id]['input_sha256']):
   assert sha(tracked(path))==expected
   with Image.open(path) as im:im.thumbnail((800,800));images.append(np.asarray(im).copy())
  fig,axes=plt.subplots(3,2,figsize=(9,11));s0,s1=shape_cache[id]
  for ri,m in enumerate(['m0','m2','b']):
   e,g,d,p0,p1=maps[(id,m,'homography')];row=next(r for r in real if r['id']==id and r['method']==m);data=np.load(row['output_npz']);ins=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(p0),bool)
   for side,shape in enumerate([s0,s1]):
    ax=axes[ri,side];h,w=shape;ax.imshow(images[side],cmap='gray',extent=[0,w,h,0]);kp=data[f'keypoints{side}'];p=[p0,p1][side];ax.scatter(kp[:,0],kp[:,1],s=3,c='#0072B2',alpha=.4);ax.scatter(p[~ins,0],p[~ins,1],s=12,c='#D55E00',marker='x');ax.scatter(p[ins,0],p[ins,1],s=16,c='#009E73');ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_title(f'{m.upper()} view{side}: KP={len(kp)}, matches={len(p)}, H support={e.n_inliers}\n{g.reason}; green=self-fit support, NOT verified');ax.set_aspect('equal')
  fig.suptitle(id+'\nNative coordinates; blue=detected, orange=rejected, green=self-fit; independent GT absent',fontsize=10);fig.tight_layout();fig.savefig(OUT/'figures'/f'spatial_{id}_{TS}.png',dpi=180);plt.close(fig)
 write('summary',dict(geometry=summaries,local_supported=[dict(id=id,method=m,n=int(sum(r['basic_local_support'] for r in localrows if r['id']==id and r['method']==m))) for id in IDS for m in ['m0','m2','b']],budget=budget,tests=tests,cpu_seconds=time.perf_counter()-start,new_feature_inference=0,new_training=0,new_GPU_jobs=0))
 write('provenance',dict(seed=SEED,inputs=used,source_sha=sha(__file__),python=sys.version,platform=platform.platform(),numpy=np.__version__,opencv=cv2.__version__,verification='Every directly read cached source hashed before and after; old files not written',inputs_unchanged=all(sha(p)==h for p,h in used.items())))
 print(json.dumps(dict(exit_status=0,seconds=time.perf_counter()-start,real_rows=len(pointrows),global_fits=len(summaries),local_cells=len(localrows),tests=tests)))
if __name__=='__main__':main()
