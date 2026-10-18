"""Pure-image registration entry, native PNG coordinates and fixed acceptance."""
import argparse
from core import *
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,2],required=True);p.add_argument('--method',choices=METHODS,default='m2');p.add_argument('--image0',required=True);p.add_argument('--image1',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 setup_gpu(a.gpu);ims=[cv2.imread(x,0) for x in [a.image0,a.image1]];assert all(x is not None for x in ims)
 engine=Engine();pred,est,diag,scores,ms=engine.predict(a.method,*ims);save_prediction(Path(a.output),a.method,pred,est,diag,scores,ms)
 print(json.dumps(summary_prediction(pred,est,diag,ms),ensure_ascii=False))
if __name__=='__main__':main()
