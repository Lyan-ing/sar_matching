"""Two uint8 grayscale PNGs only; no geo metadata, GT, angle, masks, or initialization."""
import argparse
from evaluation import *
from finalize import save_native

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--variant',choices=['a','b','control'],required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--image0',required=True);p.add_argument('--image1',required=True);p.add_argument('--output',required=True);a=p.parse_args();identity=gpu(a.gpu);m=load(a.variant,a.checkpoint);ev=FairEvaluator.__new__(FairEvaluator);ev.plan=[];ev.matcher=LightGlue(features='superpoint').eval().cuda().requires_grad_(False);ev.budgets={};ev.common_budgets=None;ev.equal=False
 ims=[cv2.imread(s,cv2.IMREAD_UNCHANGED) for s in [a.image0,a.image1]];assert all(im is not None and im.ndim==2 and im.dtype==np.uint8 for im in ims),'Use native grayscale uint8 PNG, no implicit scaling'
 out=Path(a.output);pp,e,d,ss,ms,_=ev.predict(m,*ims);row=save_native(out,a.variant,pp,e,d,ss,ms);event('prediction',dict(**row,checkpoint_sha=sha(a.checkpoint),input_sha=[sha(x) for x in [a.image0,a.image1]],process=identity,image_only=True),out);print(json.dumps(row,indent=2))
if __name__=='__main__':main()
