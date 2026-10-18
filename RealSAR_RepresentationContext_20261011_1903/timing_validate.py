"""Actual complete context pipelines without interleaved QA/cache serialization.

Targeted timing correction only. Assert predictions agree with fixed first-run
conditions; do not use timing or confirmation to select/modify a parameter.
"""
from experiment import *
def pipeline(model,method,folder,row,condition):
    # Predictor accepts image paths only. No geotransforms/masks/map/labels.
    a,b=[np.array(Image.open(folder/f'view{s}.png')) for s in [0,1]]
    coarsepred=model.predict(a,b);coarse,diag=model_estimate(coarsepred,(1024,1024))
    if condition=='context_direct':
        pp=restore_prediction(coarsepred,[256,256],[256,256]);m=(pp.points0.min(1)>=0)&(pp.points0.max(1)<=511)&(pp.points1.min(1)>=0)&(pp.points1.max(1)<=511) if len(pp.points0) else np.zeros(0,bool)
        native=Prediction(method,pp.points0[m],pp.points1[m]);result,d=model_estimate(native)
    else:
        if not coarse.accepted:return coarsepred,coarse,None,None,'coarse_frozen_gate_rejected:'+coarse.reason
        T=np.array([[1,0,256],[0,1,256],[0,0,1.] ]);H=np.linalg.inv(T)@coarse.H@T
        reference=np.array(Image.open(row['reference_native_png']));moving=np.array(Image.open(row['moving_native_png']));warped,support=nativewarp(moving,H)
        fine=model.predict(reference,warped);native=Prediction(method,fine.points0,perspective(H,fine.points1));result,d=model_estimate(native)
    return coarsepred,coarse,native,result,result.reason
def main():
    models,f,torch=study.initialize();records=[];reference={(r['sample_id'],r['method'],r['condition']):r for r in lines(ROOT/'runs/context/records.jsonl')};folder=ROOT/'timing_validation';folder.mkdir(exist_ok=False)
    for row in selected():
        for method in METHODS:
            for condition in ['context_direct','context_fine']:
                sync(torch)
                if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
                start=time.perf_counter();p,e,lp,le,reason=pipeline(models[method],method,context_path(row['sample_id']),row,condition);sync(torch);ms=1000*(time.perf_counter()-start)
                old=reference[row['sample_id'],method,condition]
                assert reason==old['reason'] and bool(le and le.accepted)==old['accepted'],(row['sample_id'],method,condition,reason,old['reason'])
                if condition=='context_direct':
                    z=np.load(old['coarse_cache']);assert np.array_equal(z['points0'],p.points0) and np.array_equal(z['points1'],p.points1),'coarse matches changed'
                if lp is not None:
                    z=np.load(old['cache_path']);assert np.allclose(z['points0'],lp.points0) and np.allclose(z['points1_native'],lp.points1),'native matches changed'
                    assert le.n_inliers==old['n_inliers']
                records.append(dict(sample_id=row['sample_id'],pair_id=row['pair_id'],site_name=row['site_name'],component=row['component'],method=method,condition=condition,accepted=old['accepted'],reason=reason,actual_end_to_end_ms=ms,peak_cuda_allocated_MB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0.,predictions_equal_first_run=True,timing_scope='fresh independent whole pipeline:1024 PNG read+coarse features/Affine/full; direct512 filtering/Affine/full OR accepted coarse->native512 PNG read->warp->fresh fine features/native-restored Affine/full; excludes model initialization, product fetch, output writes and postprediction geographic/H diagnostics',selection_uses_GT=False))
        print('actual pipeline timing',row['sample_id'],'PASS',flush=True)
    jsonlines(folder/'records.jsonl',records);csvwrite(folder/'records.csv',records);save(folder/'SUMMARY.json',{'rows':len(records),'all_predictions_equal':True,'frozen':f,'no_new_parameter_or_selection':True})
if __name__=='__main__':main()
