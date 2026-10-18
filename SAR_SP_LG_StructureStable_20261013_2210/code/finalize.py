"""Locked weights only; equal-feature-count final comparisons, real image-only diagnostics."""
import argparse,time,traceback
from evaluation import *
from sarbench.matchers import _mutual_ratio
from network import load
class ORBEvaluator(FairEvaluator):
 @torch.no_grad()
 def predict(self,model,a,b,tiled=False):
  start=time.perf_counter();hashes=tuple(hashlib.sha256(im.tobytes()).hexdigest() for im in [a,b]);budget=(self.common_budgets or self.budgets).get(hashes,(1024,1024));fs=[];ks=[];ds=[]
  for im,n in zip([a,b],budget):
   detector=cv2.ORB_create(nfeatures=1024,fastThreshold=12);k,d=detector.detectAndCompute(im,None);idx=sorted(range(len(k)),key=lambda i:(-k[i].response,i))[:n];q=np.array([k[i].pt for i in idx],np.float32).reshape(-1,2);d=d[idx] if d is not None else np.empty((0,32),np.uint8);ks.append(q);ds.append(d)
   # Binary vector diagnostics are not reported as SuperPoint descriptor or LG metrics.
   bits=np.unpackbits(d,axis=1).astype(np.float32);norm=np.linalg.norm(bits,axis=1,keepdims=True);bits/=np.maximum(norm,1e-8)
   fs.append(dict(keypoints=torch.tensor(q)[None],descriptors=torch.tensor(bits)[None],keypoint_scores=torch.tensor([k[i].response for i in idx])[None]))
  i,j=_mutual_ratio(ds[0],ds[1],cv2.BFMatcher(cv2.NORM_HAMMING),.8);pp=Prediction('orb',ks[0][i],ks[1][j],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1]);e,d=geometry(pp,a.shape,b.shape)
  return pp,e,d,np.ones(len(i),np.float32),1000*(time.perf_counter()-start),fs

def save_native(folder,method,p,e,d,scores,ms):
 folder.mkdir(parents=True,exist_ok=True);name=f'{method}_{RUNSTAMP}';npz=folder/(name+'.npz');assert not npz.exists()
 with npz.open('xb') as f:np.savez_compressed(f,keypoints0=p.keypoints0,keypoints1=p.keypoints1,points0=p.points0,points1=p.points1,scores=scores,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(p.points0),bool),candidate_H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
 return pred_summary(p,e,d,ms)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True);args=parser.parse_args();ident=gpu(args.gpu);event('final_process',dict(process=ident))
 variants=['m0','m2','a','b','control'];models={v:load(v,ROOT/'runs'/v/'best.ckpt' if v in ['a','b','control'] else None) for v in variants}
 for v in ['a','b','control']:
  lock=json.loads(output('selection_locked',folder=ROOT/'runs'/v).read_text());assert sha(ROOT/'runs'/v/'best.ckpt')==lock['best_sha256']
 plan=json.loads(output('plan_select').read_text())+json.loads(output('plan_holdout').read_text());ev=FairEvaluator(plan)
 budgets={};density=[];lowtexture=[]
 with torch.no_grad():
  for spec in plan:
   folder=ROOT/'validation_data'/spec['id'];meta=json.loads((folder/'metadata.json').read_text());ims=[cv2.imread(str(folder/f'image{i}.png'),0) for i in [0,1]];hashes=tuple(hashlib.sha256(im.tobytes()).hexdigest() for im in ims);counts={}
   for v,m in models.items():
    ns=[len(m.extract(torch.tensor(im.copy(),device='cuda').float()[None,None]/255)['keypoints'][0]) for im in ims];counts[v]=ns;density.append(dict(id=spec['id'],method=v,keypoints0=ns[0],keypoints1=ns[1],low_texture=meta['low_texture'],no_overlap=meta['no_overlap']))
   budgets[hashes]=tuple(min(counts[v][i] for v in variants) for i in [0,1])
 ev.common_budgets=budgets;event('density_budget',dict(rule='per-image minimum actual count across all five neural models, original scores topN; never GT; same effective counts',rows=density))
 allrr=[];split_ids={k:{s['id'] for s in json.loads(output('plan_'+k).read_text())} for k in ['select','holdout']}
 for v,m in models.items():
  rr=ev.evaluate(m,v,'equal_final',save=True)
  for r in rr:r['split']='select' if r['id'] in split_ids['select'] else 'holdout';r['protocol']='strict_common_count';r['actual_overlap_bin']='none' if r['no_overlap'] else ('<.3' if r['overlap']<.3 else ('.3-.6' if r['overlap']<.6 else ('.6-.8' if r['overlap']<.8 else '>=.8')))
  allrr.extend(rr)
 orb=ORBEvaluator(plan);orb.common_budgets=budgets
 rr=orb.evaluate(None,'orb','equal_final',save=True)
 for r in rr:
  r['split']='select' if r['id'] in split_ids['select'] else 'holdout';r['protocol']='same common cap, ORB may have fewer actual points';r['actual_overlap_bin']='none' if r['no_overlap'] else ('<.3' if r['overlap']<.3 else ('.3-.6' if r['overlap']<.6 else ('.6-.8' if r['overlap']<.8 else '>=.8')))
  for k in list(r):
   if k.startswith(('descriptor_','NN_','positive_cosine','negative_cosine')):r[k]=None
 allrr+=rr
 event('validation_results',dict(rows=allrr));export_csv(output('validation_per_pair',ext='csv',folder=ROOT/'tables'),allrr)
 summary={split:{v:summarize([r for r in allrr if r['method']==v and (split=='all' or r['split']==split)]) for v in variants+['orb']} for split in ['select','holdout','all']};event('comparison',summary)
 strata=[]
 for field in ['dataset','scene','regime','actual_overlap_bin','low_texture','dynamic','source_component']:
  for value in sorted({str(r[field]) for r in allrr}):
   for v in variants+['orb']:
    sub=[r for r in allrr if r['method']==v and str(r[field])==value]
    s=summarize(sub) if any(not r['no_overlap'] for r in sub) else dict(n_positive=0,n_negative=len(sub),no_overlap_false_accept=float(np.mean([r['false_accept'] for r in sub])))
    strata.append(dict(field=field,value=value,method=v,**s))
 export_csv(output('strata',ext='csv',folder=ROOT/'tables'),strata);export_csv(output('density',ext='csv',folder=ROOT/'tables'),density)
 sanity=[dict(method=v,**r) for v,m in models.items() for r in probes(m)];event('sanity',dict(rows=sanity));export_csv(output('sanity',ext='csv',folder=ROOT/'tables'),sanity)
 # Detector point counts are independently checked without the common truncation.
 native=[];ev.equal=False
 for v,m in models.items():
  rr=ev.evaluate(m,v,'native_budget',save=True)
  for r in rr:r['split']='select' if r['id'] in split_ids['select'] else 'holdout'
  native+=rr
 event('native_validation',dict(rows=native));export_csv(output('native_validation',ext='csv',folder=ROOT/'tables'),native)
 # Check newly trained branch tile equivalence, not just its zero-initial version.
 tilechecks={}
 with torch.no_grad():
  torch.manual_seed(8123);odd=torch.rand(1,1,517,531,device='cuda')
  for v in ['a','b']:
   f=models[v].extract(odd);g=models[v].tiled(odd,tile=256);equal=torch.equal(f['keypoints'],g['keypoints']);delta=float((f['descriptors']-g['descriptors']).abs().max()) if equal and f['descriptors'].numel() else None
   tilechecks[v]=dict(keypoints_identical=equal,descriptor_max_delta=delta,passed=equal and (delta is None or delta<2e-6));assert tilechecks[v]['passed'],tilechecks[v]
 event('trained_tile_check',tilechecks)
 real=[]
 for idx,ref in enumerate(json.loads(output('real_inputs').read_text())):
  ims=[cv2.imread(path,cv2.IMREAD_UNCHANGED) for path in ref['paths']];assert all(im is not None and im.ndim==2 and im.dtype==np.uint8 for im in ims)
  for path,h in zip(ref['paths'],ref['input_sha256']):assert sha(path)==h
  features={};extract_times={};pair_start=time.perf_counter()
  with torch.no_grad():
   for v in ['m0','m2','a','b']:
    torch.cuda.synchronize();t0=time.perf_counter();features[v]=[]
    for im in ims:
     tensor=torch.tensor(im.copy()).float()[None,None]/255
     features[v].append(models[v].extract(tensor if max(im.shape)>2048 else tensor.cuda()))
    torch.cuda.synchronize();extract_times[v]=1000*(time.perf_counter()-t0)
  ns=[min(len(features[v][i]['keypoints'][0]) for v in features) for i in [0,1]]
  for v in ['m0','m2','a','b']:
   torch.cuda.reset_peak_memory_stats()
   try:
    torch.cuda.synchronize();mt=time.perf_counter();fs=[]
    for i,f in enumerate(features[v]):
     order=torch.argsort(f['keypoint_scores'][0],descending=True,stable=True)[:ns[i]]
     fs.append({k:(val[:,order] if k in ['keypoints','descriptors','keypoint_scores'] else val) for k,val in f.items()})
    with torch.no_grad():matched=ev.matcher(dict(image0=fs[0],image1=fs[1]))
    ids=matched['matches'][0].cpu().numpy();ss=matched['scores'][0].cpu().numpy();ks=[f['keypoints'][0].cpu().numpy() for f in fs];pp=Prediction(v,ks[0][ids[:,0]],ks[1][ids[:,1]],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1]);e,d=geometry(pp,ims[0].shape,ims[1].shape);torch.cuda.synchronize();ms=extract_times[v]+1000*(time.perf_counter()-mt)
    row=save_native(ROOT/'real_predictions'/ref['id'],v,pp,e,d,ss,ms);row.update(id=ref['id'],method=v,independent_GT=False,real_accuracy=None,input_shapes=[im.shape for im in ims],max_gpu_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,protocol='exact common native-coordinate point count, four neural methods',common_counts=ns,native_counts=[len(f['keypoints'][0]) for f in features[v]],extraction_ms=extract_times[v])
   except Exception as ex:
    row=dict(id=ref['id'],method=v,decision='exception',accepted=False,reason=repr(ex),traceback=traceback.format_exc(),independent_GT=False);print(traceback.format_exc(),flush=True)
   meta=json.loads((Path(ref['paths'][0]).parent/'metadata.json').read_text());row.update(site=meta['site_name'],sensor=meta['sensor'],look_category=meta['look_pair_category']);real.append(row);print(f'{now()} REAL {idx+1}/18 {v} {row["decision"]}',flush=True)
  del features;torch.cuda.empty_cache()
 event('real_results',dict(rows=real));export_csv(output('real_per_pair',ext='csv',folder=ROOT/'tables'),real)
 frozen_checks={k:sha(ROOT/k)==h for k,h in json.loads(output('code_identity').read_text())['files'].items()};assert all(frozen_checks.values())
 event('final_acceptance',dict(state='completed',validation_rows=len(allrr),native_rows=len(native),real_rows=len(real),real_exceptions=sum(r['decision']=='exception' for r in real),training_code_unchanged=frozen_checks,parent_m2_unchanged=sha(FIX/'runs/m2/best.ckpt')==json.loads(output('code_identity').read_text())['parent_m2_sha256'],no900=True,no_realGT=True))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
