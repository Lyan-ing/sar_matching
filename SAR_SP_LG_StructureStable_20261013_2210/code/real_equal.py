"""Protocol correction only: native-coordinate real inference at exact common counts."""
import argparse,time
from evaluation import *

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);args=p.parse_args();ident=gpu(args.gpu);ts=stamp();folder=ROOT/f'real_equal_predictions_{ts}';event('real_equal_protocol',dict(process=ident,output_directory=str(folder),selection_locked=True,rule='pure-image per-view minimum detected count across M0/M2/A/B; raw pixels, fixed weights/LG/full gate',reason='earlier native diagnostics had common upper limit but unequal effective counts; preserve them'))
 models={v:load(v,ROOT/'runs'/v/'best.ckpt' if v in ['a','b'] else None) for v in ['m0','m2','a','b']};matcher=LightGlue(features='superpoint').eval().cuda().requires_grad_(False);rows=[]
 for number,ref in enumerate(json.loads(output('real_inputs').read_text())):
  ims=[cv2.imread(path,cv2.IMREAD_UNCHANGED) for path in ref['paths']];assert all(im is not None and im.ndim==2 and im.dtype==np.uint8 for im in ims)
  for path,h in zip(ref['paths'],ref['input_sha256']):assert sha(path)==h
  features={};times={};peaks={}
  with torch.no_grad():
   for v,m in models.items():
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();features[v]=[]
    for im in ims:
     tensor=torch.tensor(im.copy()).float()[None,None]/255;features[v].append(m.extract(tensor if max(im.shape)>2048 else tensor.cuda()))
    torch.cuda.synchronize();times[v]=1000*(time.perf_counter()-start);peaks[v]=torch.cuda.max_memory_allocated()/2**20
  ns=[min(len(features[v][i]['keypoints'][0]) for v in models) for i in [0,1]]
  for v,m in models.items():
   torch.cuda.synchronize();start=time.perf_counter();fs=[]
   for i,f in enumerate(features[v]):
    ids=torch.argsort(f['keypoint_scores'][0],descending=True,stable=True)[:ns[i]];fs.append({k:(val[:,ids] if k in ['keypoints','descriptors','keypoint_scores'] else val) for k,val in f.items()})
   with torch.no_grad():out=matcher(dict(image0=fs[0],image1=fs[1]))
   ids=out['matches'][0].cpu().numpy();ss=out['scores'][0].cpu().numpy();ks=[f['keypoints'][0].cpu().numpy() for f in fs];pp=Prediction(v,ks[0][ids[:,0]],ks[1][ids[:,1]],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1]);e,d=geometry(pp,ims[0].shape,ims[1].shape);torch.cuda.synchronize();ms=times[v]+1000*(time.perf_counter()-start)
   target=folder/ref['id'];target.mkdir(parents=True,exist_ok=True)
   with (target/f'{v}_{ts}.npz').open('xb') as f:np.savez_compressed(f,keypoints0=ks[0],keypoints1=ks[1],points0=pp.points0,points1=pp.points1,scores=ss,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(pp.points0),bool),candidate_H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
   row=dict(id=ref['id'],method=v,independent_GT=False,real_accuracy=None,protocol='exact common effective point count, original Native coordinates',common_counts=ns,native_counts=[len(f['keypoints'][0]) for f in features[v]],input_shapes=[im.shape for im in ims],max_gpu_allocated_MiB=peaks[v],output_npz=str(target/f'{v}_{ts}.npz'),**pred_summary(pp,e,d,ms))
   # Metadata attached after all prediction; never an input to model or acceptance.
   meta=json.loads((Path(ref['paths'][0]).parent/'metadata.json').read_text());row.update(site=meta['site_name'],sensor=meta['sensor'],look_category=meta['look_pair_category']);rows.append(row);write_new(target/f'{v}_{ts}.json',row);print(f'{now()} real_equal {number+1}/18 {v} points={ns} {row["decision"]}',flush=True)
  del features;torch.cuda.empty_cache()
 assert all(len({r[f'n_keypoints{i}'] for r in rows if r['id']==id})==1 for id in {r['id'] for r in rows} for i in [0,1])
 event('real_equal_results',dict(rows=rows,process=ident,output_directory=str(folder),exact_counts_verified=True,old_native_results_unchanged=True,models_selected_before_this_run=True));export_csv(ROOT/f'real_equal_per_pair_{stamp()}.csv',rows);print('REAL EQUAL COMPLETE',len(rows))
if __name__=='__main__':main()
