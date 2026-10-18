"""Validation GT is post-hoc diagnostic only; real inference receives pixels only."""
import argparse,fcntl,traceback
from core import *
from core import _mutual_ratio
from data import pair,valid_at,warp

def retrieval(k0,k1,d0,d1,z):
 target=warp(z['H_AB'],k0)
 visible=valid_at(z['static'][0],k0)&valid_at(z['static'][1],target)
 visible1=valid_at(z['static'][1],k1)&valid_at(z['static'][0],warp(np.linalg.inv(z['H_AB']),k1))
 distances=np.linalg.norm(target[:,None,:]-k1[None,:,:],axis=2)
 true=(distances<=3)&visible[:,None]&valid_at(z['static'][1],k1)[None,:]
 repeat=true.any(1);reverse=(np.linalg.norm(warp(np.linalg.inv(z['H_AB']),k1)[:,None,:]-k0[None,:,:],axis=2)<=3)&visible1[:,None]&valid_at(z['static'][0],k0)[None,:]
 sim=d0@d1.T;order=np.argsort(-sim,axis=1,kind='stable')
 result=dict(visible_detected0=int(visible.sum()),visible_detected1=int(visible1.sum()),repeatable_detected0=int(repeat.sum()),detector_repeatability_3px=float(repeat.sum()/visible.sum()) if visible.any() else None,detector_repeatability_reverse_3px=float(reverse.any(1).sum()/visible1.sum()) if visible1.any() else None)
 for K in [1,5,10]:
  hits=np.take_along_axis(true,order[:,:K],axis=1).any(1) if len(k1) else np.zeros(len(k0),bool)
  result[f'detected_descriptor_conditional_recall{K}']=float(hits[repeat].mean()) if repeat.any() else None
  result[f'detected_descriptor_end_to_end_recall{K}']=float(hits[visible].mean()) if visible.any() else None
 result['positive_cosine_detected']=float(sim[true].mean()) if true.any() else None
 negative=visible[:,None]&valid_at(z['static'][1],k1)[None,:]&(distances>=8)
 result['negative_cosine_detected']=float(sim[negative].mean()) if negative.any() else None
 return result,true,visible

@torch.no_grad()
def oracle(model,z):
 if len(z['points0'])<16:return {}
 ims=[torch.from_numpy(x.copy()).cuda().float()[None,None]/255 for x in z['images']]
 pts=[torch.from_numpy(x.copy()).cuda()[None] for x in [z['points0'],z['points1']]]
 ds=[model.descriptors(x,p)[0][0].float() for x,p in zip(ims,pts)]
 sim=ds[0]@ds[1].T;near=(torch.cdist(pts[0][0],pts[0][0])<8)|(torch.cdist(pts[1][0],pts[1][0])<8);near.fill_diagonal_(False);sim=sim.masked_fill(near,-1e4);gt=torch.arange(len(sim),device='cuda')
 return {f'oracle_GT_location_recall{k}':float((sim.topk(min(k,len(sim)),1).indices==gt[:,None]).any(1).float().mean()) for k in [1,5]}

def validation(engine):
 plan=json.loads((OLD/'validation_plan.json').read_text());rows=[];start=time.perf_counter()
 for spec in plan:
  folder=ROOT/'validation_predictions'/spec['id'];result=folder/'diagnosis.json'
  if result.exists():rows.extend(json.loads(result.read_text()));continue
  z=pair(spec['source'],spec['seed'],spec['regime'],dynamic_occlusion=spec['dynamic_occlusion']);a,b=z['images'];folder.mkdir(parents=True,exist_ok=True)
  cv2.imwrite(str(folder/'image0.png'),a);cv2.imwrite(str(folder/'image1.png'),b)
  basekp=None;local=[]
  for method in ['m0','m1','m2']:
   torch.cuda.synchronize();t=time.perf_counter();f0,f1=engine.features(method,a,b);pred,scores=engine.from_features(method,f0,f1);est,diag=geometry(pred,a.shape,b.shape);torch.cuda.synchronize();ms=1000*(time.perf_counter()-t)
   save_prediction(folder,method,pred,est,diag,scores,ms)
   k0,k1=pred.keypoints0,pred.keypoints1;d0,d1=[f['descriptors'][0].cpu().numpy() for f in [f0,f1]]
   if basekp is None:basekp=[k0.copy(),k1.copy()]
   else:assert np.array_equal(k0,basekp[0]) and np.array_equal(k1,basekp[1]),'Frozen detector changed'
   np.savez_compressed(folder/(method+'_features.npz'),keypoints0=k0,keypoints1=k1,descriptors0=d0,descriptors1=d1)
   # All the following functions are evaluators, never prediction inputs.
   rank,truth,visible=retrieval(k0,k1,d0,d1,z)
   i,j=_mutual_ratio(d0,d1,cv2.BFMatcher(cv2.NORM_L2),.8);nn=Prediction(method+'_NN',k0[i],k1[j]);ne,nd=geometry(nn,a.shape,b.shape)
   np.savez_compressed(folder/(method+'_NN.npz'),**arrays(nn,ne))
   nncorrect=truth[i,j] if len(i) else np.zeros(0,bool)
   lgdist=np.linalg.norm(warp(z['H_AB'],pred.points0)-pred.points1,axis=1)
   lgcorrect=(lgdist<=3)&valid_at(z['static'][0],pred.points0)&valid_at(z['static'][1],pred.points1)
   def fit_error(e):
    yy,xx=np.mgrid[12:a.shape[0]-12:12,12:a.shape[1]-12:12];q=np.c_[xx.ravel(),yy.ravel()].astype(float);v=warp(z['H_AB'],q);keep=valid_at(z['static'][0],q)&valid_at(z['static'][1],v)
    error=None
    if e.candidate_H is not None and keep.any():
     r=np.linalg.norm(perspective(e.candidate_H,q[keep])-v[keep],axis=1);error=float(r.mean()) if np.isfinite(r).all() else None
    return error,int(keep.sum())
   ep,count=fit_error(est);en,_=fit_error(ne)
   sp=spatial(pred.points0[lgcorrect],a.shape)
   row=clean(dict(id=spec['id'],method=method,regime=spec['regime'],scene=spec['source']['scene_type'],dataset=spec['source']['dataset'],source_component=spec['source']['training_component'],dynamic=spec['dynamic_occlusion'],low_texture=float(a.std())<12,overlap=z['overlap'],no_overlap=spec['regime']=='none',**summary_prediction(pred,est,diag,ms),**rank,**oracle(engine.models[method],z),NN_matches=len(i),NN_correct_3px=int(nncorrect.sum()),NN_precision_3px=float(nncorrect.mean()) if len(i) else None,NN_end_to_end_recall_3px=float(nncorrect.sum()/visible.sum()) if visible.any() else None,NN_accepted=ne.accepted,NN_reason=ne.reason,NN_candidate_epe_mean=en,LG_correct_3px=int(lgcorrect.sum()),LG_precision_3px=float(lgcorrect.mean()) if len(lgcorrect) else None,LG_end_to_end_recall_3px=float(lgcorrect.sum()/visible.sum()) if visible.any() else None,LG_correct_inliers_3px=int((lgcorrect&est.inlier_mask).sum()) if est.inlier_mask is not None else 0,LG_correct_hull=sp['hull_fraction'],LG_correct_grid=sp['grid_coverage'],LG_correct_isotropy=sp['isotropy'],candidate_epe_mean=ep,n_uniform_static_geometry_checks=count,detector_identical=True,input_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in [a,b]]))
   row.update({f'success_{t}px':False if spec['regime']!='none' else None for t in [3,5,10]})
   if spec['regime']!='none':row.update({f'success_{t}px':bool(est.accepted and ep is not None and ep<=t) for t in [3,5,10]})
   row['false_accept']=bool(spec['regime']=='none' and est.accepted)
   row['decision_status']='success' if row['success_5px'] else ('incorrect_accept' if est.accepted else row['decision'])
   np.savez_compressed(folder/(method+'_GT_diagnostics.npz'),LG_correct_3px=lgcorrect,NN_correct_3px=nncorrect,GT_H=z['H_AB'],static0=z['static'][0],static1=z['static'][1],GT_points0=z['points0'],GT_points1=z['points1'])
   local.append(row)
  atomic_json(result,local);rows.extend(local)
  atomic_json(ROOT/'validation_status.json',dict(state='running',completed=len(rows)//3,total=56,last_id=spec['id'],updated=now()))
  print(f'{now()} validation {spec["id"]} done',flush=True)
 atomic_json(ROOT/'validation_results.json',rows)
 atomic_json(ROOT/'validation_status.json',dict(state='completed',rows=len(rows),elapsed_seconds=time.perf_counter()-start,updated=now()))

def real(engine):
 rows=[];start=time.perf_counter()
 for ref in json.loads((ROOT/'real_inputs.json').read_text()):
  folder=ROOT/'real_predictions'/ref['id'];result=folder/'results.json'
  if result.exists():rows.extend(json.loads(result.read_text()));continue
  ims=[cv2.imread(p,0) for p in ref['paths']];assert all(x is not None for x in ims);local=[]
  for method in ['m0','m1','m2']:
   try:pred,est,diag,scores,ms=engine.predict(method,*ims)
   except Exception as ex:
    print(traceback.format_exc(),flush=True)
    if isinstance(ex,torch.cuda.OutOfMemoryError):raise
    pred=Prediction(method,reason='exception:'+type(ex).__name__);est,diag=geometry(pred,*[x.shape for x in ims]);scores=[];ms=0
   save_prediction(folder,method,pred,est,diag,scores,ms)
   local.append(dict(id=ref['id'],method=method,site=ref['site'],sensor=ref['sensor'],source_component=ref['component'],look=ref['look'],shapes=ref['shapes'],input='Native_db original pixel grid; no geographic prior',independent_GT=False,**summary_prediction(pred,est,diag,ms)))
  atomic_json(result,local);rows.extend(local);print(f'{now()} real {ref["id"]} done',flush=True)
 atomic_json(ROOT/'real_results.json',rows)
 atomic_json(ROOT/'real_status.json',dict(state='completed',rows=len(rows),elapsed_seconds=time.perf_counter()-start,max_gpu_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,max_gpu_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,updated=now()))

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
 lock=(ROOT/'diagnostics.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 atomic_json(ROOT/'diagnostics_process.json',setup_gpu(args.gpu));engine=Engine();validation(engine);real(engine)
if __name__=='__main__':main()
