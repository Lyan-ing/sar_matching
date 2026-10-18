"""Selected frozen predictions reused; only final common-actual-budget matching is new."""
from mixed_io import *
from joint_evaluate import feature_load,save_prediction
import argparse
METHODS=['m0','m2','b','m2v22','orb','t1','t2','t3','repeat']

def selection():
 selected={m:dict(tag='baseline',checkpoint=str(WEIGHTS[m]) if m!='orb' else None,sha=sha(WEIGHTS[m]) if m!='orb' else None,step=None,qualified=None) for m in METHODS[:5]}
 for m in METHODS[5:]:
  s=json.loads((ROOT/'runs'/m/f'status_{TAG}.json').read_text());assert s['state']=='completed' and s['step']==6000
  qualified=bool(s['qualified']);path=ROOT/'runs'/m/f"{'best_qualified' if qualified else 'best_trained'}_{TAG}.ckpt";ck=torch.load(path,map_location='cpu',weights_only=False);step=ck['global_step'];assert step>0
  selected[m]=dict(tag=f'{m}_step{step}',checkpoint=str(path),sha=sha(path),step=step,qualified=qualified,kind=ck['config']['kind'],code_sha=ck['config']['code_sha'],seed=ck['config']['seed'])
 return selected

def run():
 setup();selected=selection();folder=ROOT/'predictions/final';folder.mkdir(exist_ok=True);write(ROOT/f'final_model_selection_{stamp()}.json',selected)
 scores={}
 for m,meta in selected.items():
  source=ROOT/'predictions'/meta['tag']/f'scores_{TAG}.jsonl';scores[m]={r['pair_id']:r for r in [json.loads(s) for s in source.read_text().splitlines() if s] if r['method']==m and r['protocol']=='natural'};assert len(scores[m])==1200
  for path in (ROOT/'predictions'/meta['tag']).glob(f'*_{m}_natural_{TAG}.npz'):
   dst=folder/path.name
   if not dst.exists():dst.symlink_to(os.path.relpath(path,folder))
  for path in (ROOT/'predictions'/meta['tag']).glob(f'*_{m}_top128_{TAG}.npz'):
   dst=folder/path.name
   if not dst.exists():dst.symlink_to(os.path.relpath(path,folder))
 log=folder/f'scores_{TAG}.jsonl';done=[json.loads(s) for s in log.read_text().splitlines() if s] if log.exists() else [];completed={r['pair_id'] for r in done};lg=LightGlue(features='superpoint').cuda().eval().requires_grad_(False)
 for n,row in enumerate(records('val')):
  pid=row['pair_id']
  if pid in completed:continue
  budgets=[min(128,*[scores[m][pid][f'keypoints{i}'] for m in METHODS]) for i in [0,1]];rr=[]
  for m in METHODS:
   r=dict(scores[m][pid],common_budget=budgets,final_comparison=True,weights_sha=selected[m]['sha'],prediction_source_tag=selected[m]['tag'],selection_qualified=selected[m]['qualified']);rr.append(r)
   fs=feature_load(folder/f'{pid}_{m}_top128_{TAG}.npz');fs=[trim(f,budgets[i]) for i,f in enumerate(fs)]
   torch.cuda.synchronize();start=time.perf_counter();pred,e,diag,conf=predict(lg,fs,(row['height'],row['width']),m=='orb');torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
   r=scorer.assess(row,pred);r.update(method=m,protocol='equal',common_budget=budgets,keypoints0=len(fs[0]['keypoints'][0]),keypoints1=len(fs[1]['keypoints'][0]),inliers=e.n_inliers,reason=e.reason,state='accepted' if e.accepted else 'rejected' if e.candidate_H is not None else 'no_output',feature_heads_ms=scores[m][pid]['feature_heads_ms'],shared_feature_ms=None,matcher_geometry_ms=ms,cache_used=True,role=row['loss_role'],component=row['source_component_id'],final_comparison=True,weights_sha=selected[m]['sha'],prediction_source_tag=selected[m]['tag'],selection_qualified=selected[m]['qualified']);rr.append(clean(r));save_prediction(folder/f'{pid}_{m}_equal_{TAG}.npz',pred,e,conf)
  with log.open('a') as f:f.write(''.join(json.dumps(clean(r),allow_nan=False)+'\n' for r in rr));f.flush();os.fsync(f.fileno())
  done+=rr
  if n%25==0 or n==1199:write(ROOT/f'final_evaluation_status_{TAG}.json',dict(state='running',completed=n+1,total=1200,updated=now(),pid=os.getpid()));print(now(),'FINAL_COMMON_BUDGET',n+1,1200,flush=True)
 assert len(done)==21600 and all(sum(r['pair_id']==pid for r in done)==18 for pid in {r['pair_id'] for r in done})
 t=stamp();csvwrite(ROOT/f'FINAL_PER_PAIR_{t}.csv',done);summary={f'{m}/{mode}':scorer.summaries([r for r in done if r['method']==m and r['protocol']==mode]) for m in METHODS for mode in ['natural','equal']};write(ROOT/f'FINAL_SUMMARIES_{t}.json',summary)
 flat=[]
 for method,groups in summary.items():
  for strata,stats in groups.items():flat.append(dict(method=method,stratum=strata,**stats))
 csvwrite(ROOT/f'FINAL_STRATIFIED_{t}.csv',flat);write(ROOT/f'final_evaluation_status_{TAG}.json',dict(state='completed',rows=len(done),pairs=1200,updated=now(),pid=os.getpid()))
if __name__=='__main__':run()
