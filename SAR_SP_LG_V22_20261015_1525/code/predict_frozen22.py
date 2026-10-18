"""Standalone inference. Opens only two images and local frozen model weights."""
import os,sys,json,argparse,time,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
BASE=Path(__file__).resolve().parents[2]
os.environ['TORCH_HOME']=str(BASE/'MaritimeSARRegistrationPhase2/cache/torch')
sys.path.extend([str(BASE/'SAR_SP_LG_StructureStable_20261013_2210/code'),str(BASE/'MaritimeSARRegistrationPhase2')])
import numpy as np,cv2,torch
from network_corrected import SARNet
from lightglue import LightGlue
from lightglue.superpoint import sample_descriptors,simple_nms
from sarbench.matchers import Prediction
from sarbench.estimate import estimate
from sarbench.reliability import geometry_diagnostics,gate
def main():
 p=argparse.ArgumentParser();p.add_argument('image0');p.add_argument('image1');p.add_argument('--kind',choices=['m2','b'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(a.gpu) and torch.cuda.device_count()==1
 torch.set_num_threads(2);cv2.setNumThreads(1);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
 ims=[cv2.imread(s,0) for s in [a.image0,a.image1]];assert all(v is not None and max(v.shape)<=512 for v in ims)
 m=SARNet('b' if a.kind=='b' else 'control');m.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=False)['model_state'],strict=True);m=m.cuda().eval().requires_grad_(False);lg=LightGlue(features='superpoint').eval().cuda().requires_grad_(False)
 with torch.no_grad():
  torch.cuda.synchronize();start=time.perf_counter();fs=[]
  for image in ims:
   image=torch.as_tensor(image,device='cuda').float()[None,None]/255;logits,dense=m.maps(image);s=logits.softmax(1)[:,:64].permute(0,2,3,1);bs,h,w,_=s.shape;s=s.reshape(bs,h,w,8,8).permute(0,1,3,2,4).reshape(bs,h*8,w*8);s=simple_nms(s,4);s[:,:4]=-1;s[:,-4:]=-1;s[:,:,:4]=-1;s[:,:,-4:]=-1;_,y,x=torch.where(s>.0005);values=s[0,y,x];ids=torch.argsort(values,descending=True,stable=True)[:1024];k=torch.stack([x[ids],y[ids]],1).float();fs.append(dict(keypoints=k[None],keypoint_scores=values[ids][None],descriptors=sample_descriptors(k[None],dense,8).transpose(1,2),image_size=torch.tensor([[image.shape[-1],image.shape[-2]]],device='cuda')))
  r=lg(dict(image0=fs[0],image1=fs[1]));ij=r['matches'][0].cpu().numpy();ks=[f['keypoints'][0].cpu().numpy() for f in fs];pred=Prediction('SP+LG',ks[0][ij[:,0]],ks[1][ij[:,1]],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1]);e=estimate(pred,ims[0].shape,model='homography',seed=42);d=geometry_diagnostics(e,pred,*[v.shape for v in ims],42) if e.candidate_H is not None else {};e=gate(e,d,'full');torch.cuda.synchronize();elapsed_ms=1000*(time.perf_counter()-start)
 output=Path(a.output);output.mkdir(parents=True,exist_ok=False);tag=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');np.savez_compressed(output/f'prediction_{tag}.npz',keypoints0=ks[0],keypoints1=ks[1],points0=pred.points0,points1=pred.points1,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(e.n_matches,bool),H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
 (output/f'status_{tag}.json').write_text(json.dumps(dict(accepted=e.accepted,reason=e.reason,n_matches=e.n_matches,n_inliers=e.n_inliers,physical_gpu=a.gpu,only_image_input=True,elapsed_ms=elapsed_ms,peak_MiB=torch.cuda.max_memory_allocated()/2**20,checkpoint_sha=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest())));print(e.accepted,e.reason,e.n_matches,e.n_inliers)
if __name__=='__main__':main()
