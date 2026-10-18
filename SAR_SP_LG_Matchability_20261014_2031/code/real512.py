from common import *
from net import load_model
from evaluate import Evaluator
from visualize import matches
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();ident=gpu(a.gpu);models={m:load_model(m) for m in ['m0','m2','b']}
 for v in ['joint','control','frozen']:
  locked=json.loads(output('selection_locked',folder=ROOT/'runs'/v).read_text());assert sha(ROOT/'runs'/v/'best.ckpt')==locked['best_sha'];models[v]=load_model(v,ROOT/'runs'/v/'best.ckpt')
 models['orb']=None;ev=Evaluator();base=BASE/'RealSAR_Fixed512_20261014_1743/release_20261014_1756/image_only_inputs';rr=[];folder=ROOT/'real512';folder.mkdir(exist_ok=True)
 for id in ['C001','C002','C003']:
  paths=[base/f'{id}_side{i}_DN_20261014_1754.png' for i in [0,1]];ims=[cv2.imread(str(p),0) for p in paths];assert all(im.shape==(512,512) for im in ims)
  for method,model in models.items():
   torch.cuda.synchronize();t=time.perf_counter();fs=[ev.features(method,model,im) for im in ims];p,e,d,s=ev.predict(method,fs,[im.shape for im in ims]);torch.cuda.synchronize();row=dict(id=id,method=method,n_keypoints=[p.n_keypoints0,p.n_keypoints1],n_matches=e.n_matches,n_inliers=e.n_inliers,accepted=e.accepted,reason=e.reason,decision='accepted' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),geometry=d,elapsed_ms=1000*(time.perf_counter()-t),independent_GT=False,accuracy=None,input_sha=[sha(path) for path in paths]);rr.append(row)
   np.savez_compressed(output(f'{id}_{method}',ext='npz',folder=folder),keypoints0=p.keypoints0,keypoints1=p.keypoints1,points0=p.points0,points1=p.points1,scores=s,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(s),bool),candidate_H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)));matches(ims,p.points0,p.points1,e.inlier_mask,row,output(f'{id}_{method}',ext='png',folder=folder))
 write_new(output('real512'),dict(process=ident,rows=rr,only_images_used=True,all_tasks_observed_development=True,physical_GT=0));export_csv(output('real512',ext='csv'),rr);print('REAL512 COMPLETE',len(rr))
if __name__=='__main__':main()
