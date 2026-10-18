"""Frozen pure-image <=512 inference; no annotation, CRS, angle, or mask inputs."""
from common import *
from net import load_model
from evaluate import Evaluator
def predict_pair(image0,image1,checkpoint,variant):
 ims=[cv2.imread(str(p),cv2.IMREAD_GRAYSCALE) for p in [image0,image1]];assert all(im is not None and max(im.shape)<=512 for im in ims)
 model=load_model(variant,checkpoint);ev=Evaluator();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 fs=[ev.features(variant,model,im) for im in ims];p,e,d,s=ev.predict(variant,fs,[im.shape for im in ims]);torch.cuda.synchronize();ms=1000*(time.perf_counter()-start)
 return ims,p,e,d,s,dict(variant=variant,image_sha=[sha(image0),sha(image1)],checkpoint=str(checkpoint),checkpoint_sha=sha(checkpoint),n_keypoints=[p.n_keypoints0,p.n_keypoints1],n_matches=e.n_matches,n_inliers=e.n_inliers,accepted=e.accepted,reason=e.reason,decision='accepted' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),geometry=d,elapsed_ms=ms,GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20,independent_GT=False,real_accuracy=None,pure_image=True)
def save_result(folder,name,result):
 from visualize import matches
 ims,p,e,d,s,row=result;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);ts=stamp();target=folder/f'{name}_{ts}.npz';assert not target.exists()
 np.savez_compressed(target,keypoints0=p.keypoints0,keypoints1=p.keypoints1,points0=p.points0,points1=p.points1,scores=s,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(s),bool),candidate_H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
 write_new(folder/f'{name}_{ts}.json',row);matches(ims,p.points0,p.points1,e.inlier_mask,row,folder/f'{name}_{ts}.png');return row
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--image0',required=True);p.add_argument('--image1',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--variant',choices=['joint','frozen','control'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();gpu(a.gpu);row=save_result(a.output,a.variant,predict_pair(a.image0,a.image1,a.checkpoint,a.variant));print(json.dumps(row))
if __name__=='__main__':main()
