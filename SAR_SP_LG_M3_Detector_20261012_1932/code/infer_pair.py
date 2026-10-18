"""Locked M3 + frozen LightGlue. Only two uint8 grayscale pixel images are prediction inputs."""
import argparse
from m3_evaluate import *
def save_prediction(folder,method,p,e,d,scores,ms):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);js=folder/(method+'.json');npz=folder/(method+'.npz');assert not js.exists() and not npz.exists(),'Prediction overwrite prohibited'
 with npz.open('xb') as f:np.savez_compressed(f,keypoints0=p.keypoints0,keypoints1=p.keypoints1,points0=p.points0,points1=p.points1,scores=scores,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(p.points0),bool),candidate_H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
 atomic_json(js,pred_summary(p,e,d,ms))
def load_pixels(path):
 im=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
 if im is None or im.ndim!=2 or im.dtype!=np.uint8:raise ValueError('Use fixed Native_db grayscale uint8 PNG protocol; no implicit per-image normalization or GeoTIFF mapping')
 return im
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--image0',required=True);p.add_argument('--image1',required=True);p.add_argument('--output',required=True);a=p.parse_args();identity=gpu(a.gpu)
 model=load_m3(a.checkpoint);ev=Evaluator();ims=[load_pixels(x) for x in [a.image0,a.image1]];tiled=max(max(x.shape) for x in ims)>2048
 pp,est,diag,ss,ms,_=ev.predict(model,*ims,tiled=tiled);save_prediction(a.output,'m3',pp,est,diag,ss,ms)
 atomic_json(Path(a.output)/'identity.json',dict(process=identity,checkpoint_sha256=sha(a.checkpoint),input_sha256=[sha(x) for x in [a.image0,a.image1]],input_shapes=[x.shape for x in ims],tiled_exact_native=tiled,no_GT_or_geography_used=True));print(json.dumps(pred_summary(pp,est,diag,ms),indent=2))
if __name__=='__main__':main()
