"""Pure-image SAR SuperPoint/LightGlue entry, no GT or geographic input."""
import argparse,time,json
from pathlib import Path
import numpy as np,torch,cv2
from PIL import Image,ImageDraw
from common import ROOT,atomic_json,sha,ensure_gpu
from model import load_model
from evaluate import image_matches
from lightglue import LightGlue
def run(image0,image1,variant,checkpoint,out,device='cuda',keypoints=1024):
 if device.startswith('cuda'):
  import os
  if device not in ['cuda','cuda:0']:raise ValueError('Only the sole visible cuda:0 is allowed')
  ensure_gpu(int(os.environ.get('CUDA_VISIBLE_DEVICES','-1')))
 out=Path(out);out.mkdir(parents=True,exist_ok=False);model=load_model(variant,checkpoint,device);model.sp.conf.max_num_keypoints=keypoints;lg=LightGlue(features='superpoint').eval().to(device).requires_grad_(False)
 images=[np.array(Image.open(p).convert('L')) for p in [image0,image1]]
 if device=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
 start=time.perf_counter();f0,f1,m,e,d=image_matches(model,lg,*images)
 if device=='cuda':torch.cuda.synchronize()
 ms=(time.perf_counter()-start)*1000;ids=m['matches'][0].cpu().numpy();kp=[f['keypoints'][0].cpu().numpy() for f in [f0,f1]];p0,p1=kp[0][ids[:,0]],kp[1][ids[:,1]];inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(ids),bool)
 np.savez_compressed(out/'matches.npz',points0=p0,points1=p1,keypoints0=kp[0],keypoints1=kp[1],inlier_mask=inliers,matching_scores=m['matching_scores0'][0].cpu().numpy())
 width=images[0].shape[1]+images[1].shape[1];height=max(x.shape[0] for x in images)+100;canvas=Image.new('RGB',(width,height),(20,20,20));draw=ImageDraw.Draw(canvas)
 canvas.paste(Image.fromarray(images[0]).convert('RGB'),(0,100));canvas.paste(Image.fromarray(images[1]).convert('RGB'),(images[0].shape[1],100));w0=images[0].shape[1]
 # Deterministic cap solely for readability, all predictions retained in NPZ.
 selected=np.linspace(0,len(ids)-1,min(120,len(ids)),dtype=int) if len(ids) else []
 for j in selected:
  a,b=p0[j],p1[j];color=(40,210,210) if inliers[j] else (235,110,65);draw.line((float(a[0]),float(a[1]+100),float(b[0]+w0),float(b[1]+100)),fill=color,width=1)
 for i,pts in enumerate(kp):
  for x,y in pts:draw.ellipse((float(x+(w0 if i else 0)-1),float(y+99),float(x+(w0 if i else 0)+1),float(y+101)),outline=(255,215,60))
 draw.text((10,12),f'{variant} | pure-image | matches {len(ids)} | inliers {e.n_inliers} | accepted {e.accepted} | {e.reason}',fill='white')
 draw.text((10,34),'Cyan: RANSAC inliers; orange: rejected matches; yellow: keypoints. None is independent correctness.',fill='white');draw.text((10,55),'Native pixel coordinates; no geographical or manual prior; real-SAR accuracy unevaluable without GT.',fill='white');canvas.save(out/'matches.png')
 overlay=Image.new('RGB',(images[0].shape[1],images[0].shape[0]));warped=None
 if e.candidate_H is not None:
  warped=cv2.warpPerspective(images[1],np.linalg.inv(e.candidate_H),images[0].shape[::-1]);overlay=Image.fromarray(np.dstack([images[0],warped,images[0]]));overlay.save(out/'candidate_overlay_NOT_GT.png')
 hull=cv2.convexHull(p0[inliers].astype(np.float32)).reshape(-1,2).tolist() if inliers.sum()>=3 else []
 record=dict(variant=variant,input_protocol='image_only',input_sha256=[sha(p) for p in [image0,image1]],checkpoint_sha256=sha(checkpoint) if checkpoint else None,input_shapes=[list(a.shape) for a in images],coordinate_frame='reference_native_to_moving_native',n_keypoints=[len(p) for p in kp],n_matches=len(ids),n_inliers=e.n_inliers,accepted=e.accepted,decision='accepted_unassessable' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),reason=e.reason,candidate_H=e.candidate_H.tolist() if e.candidate_H is not None else None,global_H=e.H.tolist() if e.H is not None else None,geometry=d,sampled_support_polygon=hull,support_not_dense_correct_region=True,elapsed_ms=ms,GPU_peak_GiB=torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else 0,GT_read=False,geography_read=False,real_accuracy_available=False)
 atomic_json(out/'prediction.json',record);print(json.dumps(record),flush=True);return record
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('image0');p.add_argument('image1');p.add_argument('--variant',choices=['m0','m1','m2'],required=True);p.add_argument('--checkpoint');p.add_argument('--out',required=True);p.add_argument('--device',default='cuda');p.add_argument('--keypoints',type=int,default=1024);a=p.parse_args()
 if a.variant!='m0' and not a.checkpoint:p.error('Trained checkpoint required for adapted models')
 run(a.image0,a.image1,a.variant,a.checkpoint,a.out,a.device,a.keypoints)
