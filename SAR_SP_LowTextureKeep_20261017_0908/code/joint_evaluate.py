"""Frozen native evaluation; predictions completed before dataset GT assessment."""
from mixed_io import *
import argparse

def feature_save(path,fs):
 data={f'{k}{i}':v.cpu().numpy() for i,f in enumerate(fs) for k,v in trim(f,128).items()}
 np.savez_compressed(path,**data)
def feature_load(path):
 with np.load(path) as z:return [{k:torch.as_tensor(z[k+str(i)],device='cuda') for k in ['keypoints','keypoint_scores','descriptors','image_size']} for i in [0,1]]
def save_prediction(path,pred,e,conf):
 np.savez_compressed(path,**{k:v for k,v in pred.items() if k not in ['H_AB']},H_AB=np.empty((0,3)) if pred['H_AB'] is None else pred['H_AB'],inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(e.n_matches,bool),confidence=conf)
def run(models,tag,rows=None,natural=True,equal=True,save=True):
 rows=records('val') if rows is None else rows;folder=ROOT/'predictions'/tag;folder.mkdir(parents=True,exist_ok=True)
 log=folder/f'scores_{TAG}.jsonl';done=[]
 if log.exists():done=[json.loads(s) for s in log.read_text().splitlines() if s]
 completed={r['pair_id'] for r in done};lg=LightGlue(features='superpoint').cuda().eval().requires_grad_(False);cachemodel=models.get('b') or Model('b').cuda()
 for n,row in enumerate(rows):
  if row['pair_id'] in completed:continue
  ims=images(row);shape=ims[0].shape;fs={};times={};x,rs,sharedms=shared_cache(row,cachemodel,ims)
  with torch.no_grad():
   for name,m in models.items():
    torch.cuda.synchronize();t=time.perf_counter()
    if m is None:fs[name]=[orb_features(im) for im in ims]
    else:
     l,d=m.cached_maps(x,rs if hasattr(m,'structure') else None);fs[name]=[m.extract_maps(l[i:i+1],d[i:i+1],shape) for i in [0,1]]
    torch.cuda.synchronize();times[name]=(time.perf_counter()-t)*1000
  budget=[min(128,*[len(f[i]['keypoints'][0]) for f in fs.values()]) for i in [0,1]]
  rr=[]
  for name,m in models.items():
   if save:feature_save(folder/f"{row['pair_id']}_{name}_top128_{TAG}.npz",fs[name])
   for mode in (['natural'] if natural else [])+(['equal'] if equal else []):
    selected=[trim(f,budget[i] if mode=='equal' else 1024) for i,f in enumerate(fs[name])]
    torch.cuda.synchronize();t=time.perf_counter();pred,e,diag,conf=predict(lg,selected,shape,name=='orb');torch.cuda.synchronize();matchms=(time.perf_counter()-t)*1000
    # Dataset is consulted only after every prediction/gate decision is complete.
    r=scorer.assess(row,pred);r.update(method=name,protocol=mode,common_budget=budget,keypoints0=len(selected[0]['keypoints'][0]),keypoints1=len(selected[1]['keypoints'][0]),inliers=e.n_inliers,reason=e.reason,state='accepted' if e.accepted else 'rejected' if e.candidate_H is not None else 'no_output',feature_heads_ms=times[name],shared_feature_ms=sharedms,matcher_geometry_ms=matchms,cache_used=True,peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20,role=row['loss_role'],component=row['source_component_id'])
    rr.append(clean(r))
    if save:save_prediction(folder/f"{row['pair_id']}_{name}_{mode}_{TAG}.npz",pred,e,conf)
  with log.open('a') as f:
   for r in rr:f.write(json.dumps(r,allow_nan=False)+'\n')
   f.flush();os.fsync(f.fileno())
  done+=rr
  if n%20==0 or n==len(rows)-1:
   write(ROOT/f'status_{tag}_{TAG}.json',dict(stage='evaluation',tag=tag,completed=n+1,total=len(rows),updated=now(),pid=os.getpid()));print(now(),tag,n+1,len(rows),flush=True)
 del lg
 torch.cuda.empty_cache()
 output=ROOT/f'results_{tag}_{stamp()}.csv'
 if not output.exists():csvwrite(output,done)
 summary={f'{k}/{mode}':scorer.summaries([r for r in done if r['method']==k and r['protocol']==mode]) for k in models for mode in ['natural','equal'] if any(r['method']==k and r['protocol']==mode for r in done)}
 write(ROOT/f'summary_{tag}_{stamp()}.json',summary);return done,summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tag',default='baseline');p.add_argument('--limit',type=int);p.add_argument('--gpu',type=int,default=1);a=p.parse_args();setup();models={k:None if k=='orb' else Model(k).cuda() for k in ['m0','m2','b','m2v22','orb']};rr=records('val');run(models,a.tag,rr[:a.limit] if a.limit else rr)
