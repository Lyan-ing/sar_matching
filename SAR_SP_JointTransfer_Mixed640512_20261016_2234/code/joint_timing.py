"""Isolated file-to-registration latency after all training; no cached image embeddings."""
from mixed_io import *
from joint_finalize import selection,METHODS

def main():
 identity=setup();selected=selection();rows=records('val');chosen=[]
 for size in [512,640]:
  for scene in ['offshore_vessels_low_texture_sea','port_quay_dense_vessels']:
   chosen.append(next(r for r in rows if r['width']==size and r['source_scene']==scene and r['overlap_bin']!='zero'))
 lg=LightGlue(features='superpoint').cuda().eval().requires_grad_(False);result=[];probes=[];orig=np.random.default_rng(42)
 for method in METHODS:
  m=None if method=='orb' else Model(method if method in ['m0','m2','b','m2v22'] else 'b',selected[method]['checkpoint']).cuda()
  # Warm each size once. This is timing only; no scoring, no hyperparameter changes.
  for row in [chosen[0],chosen[2]]:
   ims=images(row)
   with torch.no_grad():fs=[orb_features(im) if m is None else m.extract(torch.tensor(im,device='cuda',dtype=torch.float32)[None,None]/255) for im in ims];predict(lg,fs,ims[0].shape,m is None)
  torch.cuda.reset_peak_memory_stats()
  for row in chosen:
   for repeat in range(3):
    torch.cuda.synchronize();start=time.perf_counter();ims=images(row)
    with torch.no_grad():fs=[orb_features(im) if m is None else m.extract(torch.tensor(im,device='cuda',dtype=torch.float32)[None,None]/255) for im in ims];pred,e,di,c=predict(lg,fs,ims[0].shape,m is None)
    torch.cuda.synchronize();result.append(dict(method=method,pair_id=row['pair_id'],size=row['width'],source_scene=row['source_scene'],repeat=repeat,file_to_result_ms=(time.perf_counter()-start)*1000,keypoints0=len(fs[0]['keypoints'][0]),keypoints1=len(fs[1]['keypoints'][0]),matches=e.n_matches,accepted=e.accepted,peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20,model_load_excluded=True,image_decoding_included=True,embedding_cache_used=False))
  for size in [256,512,640]:
   noise=np.random.default_rng(42).integers(0,256,(size,size),dtype=np.uint8)
   for label,im in [('black',np.zeros((size,size),np.uint8)),('gray',np.full((size,size),128,np.uint8)),('noise',noise)]:
    with torch.no_grad():f=orb_features(im) if m is None else m.extract(torch.tensor(im,device='cuda',dtype=torch.float32)[None,None]/255)
    constant_accept=None;constant_matches=None;constant_reason=None
    if size==512 and label in ['black','gray']:
     _,est,_,_=predict(lg,[f,f],im.shape,m is None);constant_accept=est.accepted;constant_matches=est.n_matches;constant_reason=est.reason
    probes.append(dict(method=method,size=size,probe=label,keypoints=len(f['keypoints'][0]),threshold_unchanged=True,constant_pair_engineering_accepted=constant_accept,constant_pair_matches=constant_matches,constant_pair_reason=constant_reason,constant_probe_not_in_no_overlap_FPR=True))
  print(now(),'ISOLATED_TIMING',method,flush=True);del m;torch.cuda.empty_cache()
 t=stamp();csvwrite(ROOT/f'ISOLATED_TIMING_{t}.csv',result);csvwrite(ROOT/f'BLANK_GRAY_NOISE_PROBES_{t}.csv',probes);write(ROOT/f'ISOLATED_TIMING_IDENTITY_{t}.json',dict(state=identity,selected_pair_ids=[r['pair_id'] for r in chosen],selection_rule='first frozen Val relationship per width512/640 and offshore/port source context; independent of success',repeats=3,methods=selected,concurrent_own_training=False,no_GT_read_for_predictions=True))
if __name__=='__main__':main()
