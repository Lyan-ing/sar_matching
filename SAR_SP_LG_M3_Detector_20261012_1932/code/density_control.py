"""Post-selection diagnostic only: M3 top-score features truncated to cached M2 per-image counts."""
from m3_evaluate import *
import argparse
class EqualCountEvaluator(Evaluator):
 def __init__(self):
  super().__init__();self.budgets={}
  for r in json.loads((ROOT/'m2_validation.json').read_text()):self.budgets[tuple(r['input_sha256'])]=(r['n_keypoints0'],r['n_keypoints1'])
 @torch.no_grad()
 def predict(self,model,a,b,tiled=False):
  # Budget uses prior frozen-model feature counts, never GT, masks or geometric accuracy.
  hashes=tuple(hashlib.sha256(im.tobytes()).hexdigest() for im in [a,b]);ns=self.budgets[hashes]
  torch.cuda.synchronize();t=time.perf_counter();fs=[]
  for im,n in zip([a,b],ns):
   f=model.extract(torch.from_numpy(im.copy()).cuda().float()[None,None]/255);order=torch.argsort(f['keypoint_scores'][0],descending=True,stable=True)[:n]
   f={k:(v[:,order] if k in ['keypoints','descriptors','keypoint_scores'] else v) for k,v in f.items()};fs.append(f)
  out=self.matcher(dict(image0=fs[0],image1=fs[1]));ids=out['matches'][0].cpu().numpy();ks=[f['keypoints'][0].cpu().numpy() for f in fs]
  p=Prediction('M3_equal_count',ks[0][ids[:,0]],ks[1][ids[:,1]],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1]);e,d=geometry(p,a.shape,b.shape);torch.cuda.synchronize()
  return p,e,d,out['scores'][0].cpu().numpy(),1000*(time.perf_counter()-t),fs
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();lock=json.loads((ROOT/'runs/m3/selection_locked.json').read_text());assert sha(ROOT/'runs/m3/best.ckpt')==lock['best_sha256'];gpu(a.gpu)
 ev=EqualCountEvaluator();m=load_m3(ROOT/'runs/m3/best.ckpt');rr=ev.evaluate(m,'m3_equal_M2_count','density_control',save=True)
 from finish_m3 import export_csv
 atomic_json(ROOT/'density_control_results.json',rr);atomic_json(ROOT/'density_control_summary.json',dict(post_selection_only=True,not_used_for_selection=True,rule='M3 original scores topN; N=cached M2 detector count per image; no threshold or loss tuning',summary=summarize(rr)));export_csv(ROOT/'tables/density_control_per_pair.csv',rr);print(json.dumps(summarize(rr),indent=2))
if __name__=='__main__':main()
