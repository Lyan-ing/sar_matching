"""Only after new-Val selection is locked: final diagnostic evaluation and 18 real Native pairs."""
import argparse,csv,time,traceback
from m3_evaluate import *
from infer_pair import save_prediction,load_pixels
def export_csv(path,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r));Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rr])
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();run=ROOT/'runs/m3';locked=json.loads((run/'selection_locked.json').read_text());assert sha(run/'best.ckpt')==locked['best_sha256'];assert json.loads((run/'status.json').read_text())['state']=='completed'
 identity=gpu(a.gpu);atomic_json(ROOT/'final_evaluation_process.json',identity);m3=load_m3(run/'best.ckpt');ev=Evaluator();ev.models['m3']=m3
 t=time.perf_counter();rr=[r for method in ['m0','m2'] for r in json.loads((ROOT/f'{method}_validation.json').read_text())]
 rr+=ev.evaluate(m3,'m3','selected',save=True)
 for r in rr:r['actual_overlap_bin']='none' if r['no_overlap'] else ('<.3' if r['overlap']<.3 else ('.3-.6' if r['overlap']<.6 else ('.6-.8' if r['overlap']<.8 else '>=.8')))
 summ={m:summarize([r for r in rr if r['method']==m]) for m in ['m0','m2','m3']};assert selection(summ['m3'])==selection(locked['validation']),'Locked validation not reproduced'
 atomic_json(ROOT/'validation_results.json',rr);atomic_json(ROOT/'comparison.json',summ);export_csv(ROOT/'tables/validation_per_pair.csv',rr)
 strata=[]
 for field in ['dataset','scene','regime','actual_overlap_bin','low_texture','dynamic','source_component']:
  for val in sorted({str(r[field]) for r in rr}):
   for method in ['m0','m2','m3']:
    subset=[r for r in rr if str(r[field])==val and r['method']==method];s=summarize(subset) if any(not r['no_overlap'] for r in subset) else dict(n_positive=0,n_negative=len(subset),no_overlap_false_accept=float(np.mean([r['false_accept'] for r in subset])))
    strata.append(dict(stratum=field,value=val,method=method,**s))
 export_csv(ROOT/'tables/validation_strata.csv',strata)
 real=[]
 for idx,ref in enumerate(json.loads((ROOT/'real_inputs.json').read_text())):
  folder=ROOT/'real_predictions'/ref['id'];folder.mkdir(parents=True,exist_ok=True);ims=[load_pixels(x) for x in ref['paths']];tiled=max(max(im.shape) for im in ims)>2048
  for method in ['m0','m2','m3']:
   if (folder/(method+'_record.json')).exists():real.append(json.loads((folder/(method+'_record.json')).read_text()));continue
   for path,h in zip(ref['paths'],ref['input_sha256']):assert sha(path)==h,'Real image identity changed'
   torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
   try:
    pp,e,d,scores,ms,_=ev.predict(ev.models[method],*ims,tiled=tiled);save_prediction(folder,method,pp,e,d,scores,ms)
    row=dict(id=ref['id'],method=method,shapes=[im.shape for im in ims],input_protocol='fixed Native_db, original pixel coordinates',tiled_exact_native=tiled,independent_GT=False,real_accuracy=None,**pred_summary(pp,e,d,ms),max_gpu_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,max_gpu_reserved_MiB=torch.cuda.max_memory_reserved()/2**20)
   except Exception as ex:
    row=dict(id=ref['id'],method=method,shapes=[im.shape for im in ims],tiled_exact_native=tiled,independent_GT=False,accepted=False,decision='exception',reason=repr(ex),elapsed_ms=1000*(time.perf_counter()-start));atomic_json(folder/(method+'_failure.json'),dict(**row,traceback=traceback.format_exc()));print(traceback.format_exc(),flush=True);torch.cuda.empty_cache()
   # Source/look labels are attached AFTER image-only prediction; never matcher inputs.
   meta=json.loads((Path(ref['paths'][0]).parent/'metadata.json').read_text())
   row.update(site=meta['site_name'],sensor=meta['sensor'],look_category=meta['look_pair_category'],angle_bin=meta.get('angle_bin'),source_component=meta['component'])
   atomic_json(folder/(method+'_record.json'),row);real.append(row);atomic_json(ROOT/'real_status.json',dict(state='running',completed=len(real),total=54,last_id=ref['id'],method=method,updated=now()));print(f'{now()} real {idx+1}/18 {method}: {row["decision"]} {row.get("n_matches",0)} matches {row["elapsed_ms"]/1000:.1f}s',flush=True)
  # Return memory to allocator between Native pairs, not during training.
  torch.cuda.empty_cache()
 atomic_json(ROOT/'real_results.json',real);export_csv(ROOT/'tables/real_native_per_pair.csv',real);atomic_json(ROOT/'real_status.json',dict(state='completed',rows=len(real),exceptions=sum(r['decision']=='exception' for r in real),updated=now()))
 # Frozen historical identities checked only for direct references, not a full old-file audit.
 checks={p:sha(p)==h for p,h in [(FIX/'runs/m2/best.ckpt',json.loads((ROOT/'data_identity.json').read_text())['parent_sha256']),(OLD/'train_sources.jsonl',json.loads((ROOT/'data_identity.json').read_text())['train_sources_sha256']),(OLD/'validation_plan.json',json.loads((ROOT/'data_identity.json').read_text())['old_validation_plan_sha256'])]}
 atomic_json(ROOT/'final_acceptance.json',dict(completed=now(),elapsed_seconds=time.perf_counter()-t,validation_rows=len(rr),real_rows=len(real),new_validation_selection_locked=True,selected_best_sha256=sha(run/'best.ckpt'),frozen_references_unchanged={str(p):v for p,v in checks.items()},passed=all(checks.values()),no_frozen900_inference=True,no_training_beyond_detector=True,no_human_GT_access_in_prediction=True))
 print(json.dumps(summ,indent=2))
if __name__=='__main__':main()
