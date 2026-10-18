"""Frozen CLI: two pixels-only images, native <=512. No data manifests or GT read."""
from support22 import Model,gpu,write,stamp,Path,cv2,torch,sha,now
from evaluate22 import Evaluation,features,predict
import argparse,numpy as np
def main():
 p=argparse.ArgumentParser();p.add_argument('image0');p.add_argument('image1');p.add_argument('--kind',choices=['m2','b'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args();proc=gpu(a.gpu);ims=[cv2.imread(s,0) for s in [a.image0,a.image1]];assert all(v is not None and max(v.shape)<=512 for v in ims)
 m=Model(a.kind,a.checkpoint).cuda();ev=Evaluation()
 with torch.no_grad():fs=[features(m,v) for v in ims];pred,e,d,c=predict(ev.lg,fs,[v.shape for v in ims])
 target=Path(a.output);target.mkdir(parents=True,exist_ok=False);t=stamp();np.savez_compressed(target/f'prediction_{t}.npz',keypoints0=pred.keypoints0,keypoints1=pred.keypoints1,points0=pred.points0,points1=pred.points1,confidence=c,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(c),bool),H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
 write(target/f'status_{t}.json',dict(at=now(),process=proc,checkpoint_sha=sha(a.checkpoint),accepted=e.accepted,reason=e.reason,n_matches=e.n_matches,n_inliers=e.n_inliers,geometry=d,GT_used=False,geographic_prior_used=False));print(e.accepted,e.reason,e.n_matches,e.n_inliers)
if __name__=='__main__':main()
