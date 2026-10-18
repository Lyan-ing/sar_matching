"""Post-hoc fixed-validation illustrations; no selection or benchmark rerun."""
import sys,json,time
from pathlib import Path
import numpy as np,torch,cv2
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
RUNTIME=BASE/'SAR_SP_LG_M2_Numerics_20261012_1627';OLD=BASE/'SAR_SP_LG_Train_20261012_1544'
sys.path.insert(0,str(RUNTIME/'code'))
from common import ensure_gpu,atomic_json,sha,now
from data import pair
from model import load_model
from evaluate import image_matches
from lightglue import LightGlue
IDS=['val_000','val_025','val_037','val_043','val_none_000']
def run():
 gpu=ensure_gpu(1);torch.set_num_threads(2);cv2.setNumThreads(1);torch.manual_seed(42);np.random.seed(42)
 plan={p['id']:p for p in json.loads((OLD/'validation_plan.json').read_text())}
 samples={i:pair(plan[i]['source'],plan[i]['seed'],plan[i]['regime'],dynamic_occlusion=plan[i]['dynamic_occlusion']) for i in IDS}
 cache=ROOT/'illustration_predictions';cache.mkdir(exist_ok=False)
 lg=LightGlue(features='superpoint').eval().cuda().requires_grad_(False)
 weights={'m0':None,'m1':OLD/'runs/m1/best.ckpt','m2':RUNTIME/'runs/m2/best.ckpt'}
 records=[]
 for variant,checkpoint in weights.items():
  model=load_model(variant,checkpoint).eval()
  for sid,z in samples.items():
   # Predictor gets only the two image arrays, never z or its GT/masks.
   torch.cuda.synchronize();start=time.perf_counter();f0,f1,m,e,d=image_matches(model,lg,*z['images']);torch.cuda.synchronize()
   elapsed=1000*(time.perf_counter()-start);ids=m['matches'][0].cpu().numpy();kp=[f['keypoints'][0].cpu().numpy() for f in [f0,f1]];p0,p1=kp[0][ids[:,0]],kp[1][ids[:,1]]
   mask=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(ids),bool)
   np.savez_compressed(cache/f'{sid}_{variant}.npz',points0=p0,points1=p1,keypoints0=kp[0],keypoints1=kp[1],inlier_mask=mask,candidate_H=e.candidate_H if e.candidate_H is not None else np.empty((0,0)))
   r=dict(id=sid,variant=variant,checkpoint_sha256=sha(checkpoint) if checkpoint else None,n_matches=len(ids),n_inliers=e.n_inliers,accepted=e.accepted,decision='accepted' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),reason=e.reason,elapsed_ms=elapsed,geometry=d,predictor_GT_read=False,predictor_geography_read=False,scope='posthoc validation illustration only; frozen selections unchanged',gpu=gpu)
   atomic_json(cache/f'{sid}_{variant}.json',r);records.append(r);print(sid,variant,r['decision'],len(ids),e.n_inliers,flush=True)
  del model;torch.cuda.empty_cache()
 atomic_json(cache/'manifest.json',dict(at=now(),ids=IDS,calls=len(records),records=records,checkpoint_selection_changed=False,formal900_calls=0,realSAR_calls=0))
if __name__=='__main__':run()
