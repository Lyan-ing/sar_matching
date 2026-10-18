"""Standalone pure-image CLI; shared total budget uses all frozen SP variants."""
import argparse,json,time,hashlib,sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
def main():
 p=argparse.ArgumentParser();p.add_argument('left');p.add_argument('right');p.add_argument('--gpu',type=int,required=True);p.add_argument('--model',choices=['M2','B','M2_V22'],default='M2');p.add_argument('--split',choices=['64_64','96_32'],default='64_64');p.add_argument('--method',choices=['F2','F3','Concat'],default='F3');p.add_argument('--output',required=True);a=p.parse_args()
 allowed={str(Path(x).resolve()) for x in [a.left,a.right]};reads=[]
 def audit(event,args):
  if event!='open' or not isinstance(args[0],(str,bytes)):return
  path=str(Path(args[0]).resolve())
  if path in allowed:return
  if any(x in path for x in ['SyntheticSAR_','DualSAR_DatasetPrep_','RealSAR_','MaritimeSARRegDataset/data']):raise RuntimeError('Forbidden label/metadata read: '+path)
  reads.append(path)
 sys.addaudithook(audit)
 from feature_runtime import initialize,infer_images,WEIGHTS,cv2,np,torch
 models,lg=initialize(a.gpu);ims=[cv2.imread(s,0) for s in [a.left,a.right]];torch.cuda.synchronize();start=time.perf_counter();outputs,_=infer_images(ims,models,lg);torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
 x=next(x for x in outputs if x['method']==a.model and x['split']==a.split);pred,e,d,meta=x['outputs'][a.method];r=Path(a.output);r.mkdir(parents=True,exist_ok=False);tag=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
 np.savez_compressed(r/f'prediction_{tag}.npz',points0=pred.points0,points1=pred.points1,keypoints0=pred.keypoints0,keypoints1=pred.keypoints1,sources=meta['sources'],inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(pred.points0),bool),H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
 state=dict(accepted=e.accepted,reason=e.reason,route=meta['route'],model=a.model,method=a.method,split=a.split,budgets=x['budgets'],matches=len(pred.points0),inliers=e.n_inliers,physical_gpu=a.gpu,python_open_audit=True,audit_scope='Python open hooks, not system-call tracing',allowed_images=sorted(allowed),geography_or_GT_read=False,weight_sha={m:hashlib.sha256(path.read_bytes()).hexdigest() for m,path in WEIGHTS.items()},selected_component_ms=x['base_ms']+meta['geometry_ms'],post_load_all_variants_ms=elapsed,all_variants_evaluated=True,peak_torch_MiB=torch.cuda.max_memory_allocated()/2**20)
 (r/f'status_{tag}.json').write_text(json.dumps(state,ensure_ascii=False,indent=2));print(json.dumps(state,ensure_ascii=False))
if __name__=='__main__':main()
