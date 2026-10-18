import inspect,json
from core import *
from core import _mutual_ratio
from data import pair,valid_at,component,warp
from run_diagnostics import retrieval
def main():
 gpu=setup_gpu(2);checks={};engine=Engine()
 index=BASE/'MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl'
 records=[json.loads(s) for s in index.read_text().splitlines()]
 sets={split:{component(r) for r in records if r['split']==split} for split in ['train','val','test']}
 assert not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test']
 checks['known_component_isolation']={s:len(v) for s,v in sets.items()}
 checks['source_index_sha256']=sha(index)
 plan=json.loads((OLD/'validation_plan.json').read_text());spec=plan[0];z=pair(spec['source'],spec['seed'],spec['regime'],dynamic_occlusion=spec['dynamic_occlusion'])
 assert np.max(np.abs(warp(z['H_AB'],z['points0'])-z['points1']))<1e-3
 assert np.max(np.abs(warp(np.linalg.inv(z['H_AB']),warp(z['H_AB'],z['points0']))-z['points0']))<1e-6
 assert valid_at(z['static'][0],z['points0']).all() and valid_at(z['static'][1],z['points1']).all()
 checks['synthetic_coordinate_mask_checks']=True
 yy,xx=np.mgrid[32:225:32,32:225:32];p0=np.c_[xx.ravel(),yy.ravel()].astype(np.float32)
 H=np.array([[1,0,4],[0,1,2],[0,0,1]],float);p1=perspective(H,p0).astype(np.float32)
 pred=Prediction('unit',p0,p1);est,diag=geometry(pred,(256,256),(256,256));assert est.accepted and np.max(np.abs(perspective(est.H,p0)-p1))<1e-4
 from sarbench.reliability import projection_diagnostics
 pole=projection_diagnostics(np.array([[1,0,0],[0,1,0],[.01,0,-1]]),(256,256));assert not pole['denominator_same_sign']
 empty=Prediction('empty');e,_=geometry(empty,(256,256),(256,256));assert not e.accepted and e.candidate_H is None
 checks['geometry_direction_pole_empty']=True
 assert list(inspect.signature(Engine.predict).parameters)==['self','method','image0','image1']
 source=inspect.getsource(Engine)
 assert all(token not in source for token in ['GeoTransform','H_AB','static','manual','load_gt','read_text','rasterio'])
 checks['image_only_prediction_boundary']=True
 kp=None;smoke=[]
 for method in ['m0','m1','m2']:
  p,e,d,s,ms=engine.predict(method,*z['images']);assert p.n_keypoints0<=1024 and p.n_keypoints1<=1024
  if kp is None:kp=(p.keypoints0,p.keypoints1)
  else:assert np.array_equal(kp[0],p.keypoints0) and np.array_equal(kp[1],p.keypoints1)
  smoke.append(dict(method=method,**summary_prediction(p,e,d,ms)))
 checks['real_model_smoke']=smoke;checks['same_detector_points']=True
 # Retrieval separates detector feasibility from descriptor rank.
 zz=dict(H_AB=H,static=[np.ones((256,256),bool)]*2)
 d=np.eye(len(p0),dtype=np.float32);rank,truth,vis=retrieval(p0,p1,d,d,zz)
 assert rank['detector_repeatability_3px']==1 and rank['detected_descriptor_conditional_recall1']==1
 rank2,_,_=retrieval(p0,p1,np.roll(d,1,axis=0),d,zz)
 assert rank2['detector_repeatability_3px']==1 and rank2['detected_descriptor_conditional_recall1']==0
 checks['oracle_vs_detected_retrieval_definition']=True
 assert rank['visible_detected0']==49
 i,j=_mutual_ratio(d,d,cv2.BFMatcher(cv2.NORM_L2),.8);assert np.array_equal(i,j) and len(i)==len(d)
 checks['mutual_nearest_neighbor']=True
 for method in ['sift','orb']:
  p,*_=engine.predict(method,*z['images']);assert p.n_keypoints0<=1024 and p.n_keypoints1<=1024
 checks['classic_real_input_smoke']=True
 atomic_json(ROOT/'preflight.json',dict(passed=True,created=now(),gpu=gpu,checks=checks))
 print('PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':main()
