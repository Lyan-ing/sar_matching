"""Small isolated speed check after all training/evaluation; no accuracy recalculation."""
from support22 import *
from evaluate22 import Evaluation,features,predict
import argparse,gc
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();process=gpu(a.gpu);records=rows(E['recommended_val_all']);chosen=[]
 for size in [256,384,512]:chosen.extend([r for r in records if r['width']==size][:2])
 ims=[[cv2.imread(r[f'image{i}'],0) for i in [0,1]] for r in chosen];ev=Evaluation();results=[]
 methods=[('M0','m0',None),('M2','m2',None),('B','b',None),('M2_V22','m2',ROOT/'runs/m2_core/best.ckpt'),('B_V22','b',ROOT/'runs/b_core_evidence/best.ckpt'),('ORB',None,None)]
 for method,kind,path in methods:
  m=Model(kind,path).cuda() if kind else None
  with torch.no_grad():
   for _ in range(2):fs=[features(m,x) for x in ims[0]];predict(ev.lg,fs,[x.shape for x in ims[0]],kind is None)
   torch.cuda.reset_peak_memory_stats()
   for r,images in zip(chosen,ims):
    for repeat in range(2):
     torch.cuda.synchronize();start=time.perf_counter();fs=[features(m,x) for x in images];pred,e,d,c=predict(ev.lg,fs,[x.shape for x in images],kind is None);torch.cuda.synchronize();ms=1000*(time.perf_counter()-start);results.append(dict(method=method,sample_id=r['sample_id'],size=r['width'],repeat=repeat,elapsed_ms=ms,n_keypoints0=pred.n_keypoints0,n_keypoints1=pred.n_keypoints1,n_matches=e.n_matches,accepted=e.accepted,torch_peak_MiB=torch.cuda.max_memory_allocated()/2**20))
  del m,fs;gc.collect();torch.cuda.empty_cache()
 t=stamp();csvwrite(ROOT/f'isolated_timing_{t}.csv',results);write(ROOT/f'isolated_timing_{t}.json',dict(at=now(),process=process,protocol='Six fixed pairs: first2 per 256/384/512 native size, two repeats, two warmup calls, natural1024. One SP model plus LG resident. Manifest only locates paths/dimensions; predictor receives only image features/shapes; accuracy is not recalculated. Not a full-corpus latency estimate.',rows=results,summaries={m:dict(mean_ms=float(np.mean([r['elapsed_ms'] for r in results if r['method']==m])),median_ms=float(np.median([r['elapsed_ms'] for r in results if r['method']==m])),peak_MiB=max(r['torch_peak_MiB'] for r in results if r['method']==m)) for m,_,_ in methods}));print('TIMING_COMPLETE',t)
if __name__=='__main__':main()
