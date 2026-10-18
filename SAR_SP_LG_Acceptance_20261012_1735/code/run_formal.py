import argparse,fcntl,traceback
from core import *
from sarbench.data import discover,load_images,load_gt
from sarbench.evaluate import evaluate
DATA=BASE/'MaritimeSARRegDataset/output/phase1_formal_20261009_1846'
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);args=p.parse_args()
 lock=(ROOT/'formal.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 identity=setup_gpu(args.gpu);identity['protocol_sha256']=sha(ROOT/'protocol.json');atomic_json(ROOT/'formal_process.json',identity)
 refs=discover(DATA);assert len(refs)==900
 engine=Engine();start=time.perf_counter()
 # Warm-up on validation only, never use formal results for adjustments.
 from data import pair
 spec=json.loads((OLD/'validation_plan.json').read_text())[0];z=pair(spec['source'],spec['seed'],spec['regime'],dynamic_occlusion=spec['dynamic_occlusion'])
 for method in METHODS:engine.predict(method,*z['images'])
 for index,ref in enumerate(refs):
  folder=ROOT/'formal_predictions'/ref.id.replace(':','__');rowpath=folder/'evaluation.json'
  if rowpath.exists():continue
  ims=load_images(ref);exceptions={}
  for method in METHODS[index%5:]+METHODS[:index%5]:
   if (folder/(method+'.json')).exists():continue
   try:p,e,d,s,ms=engine.predict(method,*ims)
   except Exception as ex:
    if isinstance(ex,torch.cuda.OutOfMemoryError):raise
    exceptions[method]=traceback.format_exc();print(exceptions[method],flush=True)
    p=Prediction(method,reason='exception:'+type(ex).__name__);e,d=geometry(p,*[v.shape for v in ims]);s=[];ms=0
   save_prediction(folder,method,p,e,d,s,ms)
  # Prediction files are immutable and already written before GT access.
  gt=load_gt(ref);rows=[]
  for method in METHODS:
   pred,est,raw=load_prediction(folder,method);ev=evaluate(est,gt,ims[0].shape)
   err=None
   if est.candidate_H is not None and gt['count']:
    errors=np.linalg.norm(perspective(est.candidate_H,gt['points0'])-gt['points1'],axis=1)
    if np.isfinite(errors).all():err=float(errors.mean())
   errors=np.linalg.norm(perspective(gt['H'],pred.points0)-pred.points1,axis=1)
   # Geometric match correctness; does not certify static physical ships.
   from data import valid_at
   masks=[(cv2.imread(str(ref.folder/f'valid_{v}.png'),0)>0)&(cv2.imread(str(ref.folder/f'dynamic_{v}.png'),0)==0) for v in [ref.view_i,ref.view_j]]
   keep=valid_at(masks[0],pred.points0)&valid_at(masks[1],pred.points1)&valid_at(masks[1],perspective(gt['H'],pred.points0))
   correct=int(((errors<=3)&keep).sum()) if not gt['no_overlap'] else 0
   rows.append(clean(dict(id=ref.id,method=method,dataset=ref.dataset,scene=ref.scene,profile=ref.profile,source_component=ref.source_component,source_group=ref.source_group,source_id=ref.source_id,group=ref.folder.name,dynamic=bool(any(ref.meta.get('dynamic_events',[]))),low_texture=bool(float(ims[0].std())<12),height0=ims[0].shape[0],width0=ims[0].shape[1],height1=ims[1].shape[0],width1=ims[1].shape[1],**raw,**ev,candidate_epe_mean=err,synthetic_static_geometric_matches_3px=correct,exception=exceptions.get(method))))
  atomic_json(rowpath,rows)
  atomic_json(ROOT/'formal_status.json',dict(state='running',completed_pairs=index+1,total=900,elapsed_seconds=time.perf_counter()-start,last_id=ref.id,updated=now()))
  if index%25==0:print(f'{now()} formal {index+1}/900, {time.perf_counter()-start:.1f}s',flush=True)
 allrows=[r for ref in refs for r in json.loads((ROOT/'formal_predictions'/ref.id.replace(':','__')/'evaluation.json').read_text())]
 atomic_json(ROOT/'formal_results.json',allrows)
 atomic_json(ROOT/'formal_status.json',dict(state='completed',pairs=len(refs),rows=len(allrows),elapsed_seconds=time.perf_counter()-start,max_gpu_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,max_gpu_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,updated=now()))
 print('FORMAL_COMPLETED',flush=True)
if __name__=='__main__':main()
