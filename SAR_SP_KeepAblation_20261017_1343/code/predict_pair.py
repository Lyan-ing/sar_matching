"""Standalone image-only registration. Coordinates are native pixels, reference -> moving."""
import argparse,sys,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('reference',type=Path);p.add_argument('moving',type=Path);p.add_argument('--kind',choices=['m0','m2','b','m2v22'],default='b');p.add_argument('--checkpoint',type=Path);p.add_argument('--gpu',choices=list(range(8)),type=int,default=1);p.add_argument('--budget',type=int,choices=[128,1024],default=1024);p.add_argument('--output',type=Path,required=True);p.add_argument('--audit',action='store_true');a=p.parse_args();os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
# Guard forbidden read sources before importing the runtime. Image inputs and weights remain allowed.
readpaths=set()
def hook(event,args):
 if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
 path=str(args[0]);mode=args[1]
 if isinstance(mode,str) and any(x in mode for x in 'wax+'):return
 readpaths.add(path)
 if any(s in path.lower() for s in ['/manifests/','/gt/','/ground_truth/','dataset_entrypoints','geotransform','geoassist','manual_check','annotations.json']) or path.lower().endswith('.npz'):raise RuntimeError('Forbidden inference read: '+path)
if a.audit:sys.addaudithook(hook)
from runtime import *
state=setup();a.output.mkdir(parents=True,exist_ok=True);ims=[cv2.imread(str(x),cv2.IMREAD_GRAYSCALE) for x in [a.reference,a.moving]];assert all(x is not None for x in ims)
assert all(max(x.shape)<=640 for x in ims);m=Model(a.kind,a.checkpoint).cuda();lg=LightGlue(features='superpoint').cuda().eval().requires_grad_(False)
torch.cuda.synchronize();start=time.perf_counter()
fs=[trim(m.extract(torch.tensor(im,device='cuda',dtype=torch.float32)[None,None]/255),a.budget) for im in ims]
assert ims[0].shape==ims[1].shape,'Current common geometry protocol requires equal image dimensions'
pred,e,diag,confidence=predict(lg,fs,ims[0].shape);torch.cuda.synchronize();elapsed=time.perf_counter()-start;t=stamp()
np.savez_compressed(a.output/f'prediction_{t}.npz',**{k:v for k,v in pred.items() if k!='H_AB'},H_AB=pred['H_AB'] if pred['H_AB'] is not None else np.empty((0,3)),confidence=confidence,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(e.n_matches,bool))
record=dict(state=state,kind=a.kind,checkpoint=str(a.checkpoint or WEIGHTS[a.kind]),checkpoint_sha=sha(a.checkpoint or WEIGHTS[a.kind]),images=[str(a.reference),str(a.moving)],image_sha=[sha(a.reference),sha(a.moving)],native_shapes=[im.shape for im in ims],budget=a.budget,accepted=e.accepted,reason=e.reason,inliers=e.n_inliers,matches=e.n_matches,elapsed_seconds=elapsed,geometry=diag,H_AB=pred['H_AB'],GT_used=False,geographic_inputs_used=False,peak_memory_MiB=torch.cuda.max_memory_allocated()/2**20,read_paths=sorted(readpaths) if a.audit else None)
write(a.output/f'prediction_metadata_{t}.json',record);print(json.dumps(clean(record),ensure_ascii=False))
