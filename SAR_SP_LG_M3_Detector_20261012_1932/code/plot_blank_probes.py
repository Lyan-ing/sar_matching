"""Visualize the documented detector failure; no training or model selection."""
from m3_evaluate import *
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();gpu(a.gpu);ev=Evaluator();ev.models['m3']=load_m3(ROOT/'runs/m3/best.ckpt')
 fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained');record=[]
 for row,(name,value) in enumerate([('All black',0),('Constant gray',128)]):
  im=np.full((256,256),value,np.uint8)
  for col,method in enumerate(['m0','m2','m3']):
   with torch.no_grad():f=ev.models[method].extract(torch.from_numpy(im).cuda().float()[None,None]/255)
   k=f['keypoints'][0].cpu().numpy();ax=axs[row,col];ax.imshow(im,cmap='gray',vmin=0,vmax=255);ax.scatter(k[:,0],k[:,1],s=4,c='#D55E00');ax.set_title(f'{method.upper()} | {name} | {len(k)} points');ax.set_xlim(0,255);ax.set_ylim(255,0)
   record.append(dict(probe=name,method=method,n_points=len(k)));out=ROOT/'probe_predictions';out.mkdir(exist_ok=True);np.savez_compressed(out/(name.replace(' ','_')+'_'+method+'.npz'),keypoints=k,scores=f['keypoint_scores'][0].cpu().numpy())
 fig.suptitle('No image structure: dense M3 detections are degeneration, not useful landmarks')
 out=ROOT/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'blank_detector_failure.png',dpi=160);plt.close(fig);atomic_json(ROOT/'blank_figure_counts.json',record)
if __name__=='__main__':main()
