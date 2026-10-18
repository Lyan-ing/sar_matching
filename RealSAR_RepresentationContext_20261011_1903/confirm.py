"""Conditional, single locked factor replay on remaining current-stage components."""
from experiment import *
def main():
    progress=read(ROOT/'analysis/progression.json');eligible=[r['condition'] for r in progress if r['expansion_eligible']]
    # Preregistered fixed order; no joint input/model/threshold optimization.
    order=['linear','asinh','clahe','context_direct','context_fine'];factor=next((c for c in order if c in eligible),None)
    if factor is None:
        save(ROOT/'extension_decision.json',{'at':now(),'run':False,'reason':'no factor added a new ANY-family general/high B sample; no expansion or combination'});return
    if factor not in REPRESENTATIONS:
        save(ROOT/'extension_decision.json',{'at':now(),'run':False,'reason':'only context benefit; no independent verified-different-look components remain; context extension would not answer target hypothesis','eligible':eligible});return
    candidates=[r for r in study.rows() if r['component'] in config()['holdout_components']];chosen=[]
    for component in config()['holdout_components']:
        group=[r for r in candidates if r['component']==component]
        row=min(group,key=lambda r:(-min(v['fraction_of_all_pixels'] for v in r['effective_geographic_overlap']),hashlib.sha256((str(config()['seed'])+r['sample_id']).encode()).hexdigest()))
        chosen.append(row)
    jsonlines(ROOT/'confirmation_samples.jsonl',chosen)
    save(ROOT/'extension_decision.json',{'at':now(),'run':True,'fixed_factor':factor,'eligible_factors':eligible,'choice_rule':'single first improving factor in fixed simple order linear/asinh/clahe/context; no combination','selection_rule':'one maximum overlap row per remaining source component; same seeded identity tie as primary','samples_sha256':sha(ROOT/'confirmation_samples.jsonl'),'config_sha256':sha(ROOT/'config.json'),'code_sha256':sha(__file__),'frozen':study.frozen(),'components':config()['holdout_components'],'independent_from_current_factor_selection_components':True,'independent_unbiased_real_accuracy':False,'no_unseen_verified_different_look_components_available':True,'full191_expansion':False})
    folder=ROOT/'runs/confirmation';folder.mkdir(exist_ok=False);(folder/'matches').mkdir();models,frozen,torch=study.initialize();records=[]
    lookup={(r['sample_id'],r['method']):r for r in lines(OLD/'quality/model_grades.jsonl') if r['protocol']=='native'}
    for row in chosen:
        raw,valid,old,_,mapper=load_pair(row)
        for method in METHODS:
            records.append(dict(lookup[row['sample_id'],method],condition='original',baseline_reused=True))
            sync(torch);start=time.perf_counter()
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
            a,b=[representation(raw[i],valid[i],old[i],factor,row[p+'_input_units'])[1] for i,p in enumerate(['reference','moving'])]
            pred=models[method].predict(a,b);e,d=model_estimate(pred);sync(torch);elapsed=1000*(time.perf_counter()-start);cache=folder/'matches'/f'{row["sample_id"]}_{factor}_{method}.npz'
            r=study.diagnose(row,method,0,pred,e,d,elapsed,valid,mapper,cache);r.update(condition=factor,confirmation=True,peak_cuda_allocated_MB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0.,runtime_scope='representation from preloaded raw +features+Affine/full, excludes disk I/O/diagnostic/QA; primary includes raw I/O');records.append(r)
        print('confirmation',row['sample_id'],'done',flush=True)
    from geo import pixel_grid
    for row in chosen:
        _,valid,_,_,mapper=load_pair(row);rs=[r for r in records if r['sample_id']==row['sample_id'] and r['condition']==factor]
        grid=regular_grid((512,512));grid=grid[sample_valid(valid[0],grid)&sample_valid(valid[1],mapper(grid))]
        for r in rs:
            if grade(r,True,config()['B_frozen'],0)[0]=='unreliable_candidate':continue
            z=np.load(r['cache_path']);m=z['eligible_inlier_mask'];a,b=z['points0'][m],z['points1_native'][m]
            geographic=np.array(Image.open(OLD/'georeference_correspondences'/row['sample_id']/'geographic_valid_overlap0.png'))>0
            support=hull_mask(a,(512,512))&valid[0]&geographic&sample_valid(valid[1]&hull_mask(b,(512,512)),perspective(r['candidate_H'],pixel_grid((512,512)))).reshape(512,512)
            r['pseudo_support_pixels']=int(support.sum())
            if support.any():Image.fromarray(support.astype(np.uint8)*255).save(folder/'matches'/f'{row["sample_id"]}_{r["method"]}_support.png')
        for r in rs:
            peers={r['method']}
            if r.get('pseudo_support_pixels',0):
                for s in rs:
                    if s['method']==r['method'] or not s.get('pseudo_support_pixels',0):continue
                    a,b=perspective(r['candidate_H'],grid),perspective(s['candidate_H'],grid);ok=sample_valid(valid[1],a)&sample_valid(valid[1],b);e=np.linalg.norm(a[ok]-b[ok],axis=1)
                    if len(e)>=16 and np.quantile(e,.9)<=config()['B_frozen']['B_high']['cross_family_prediction_p90_max_px']:peers.add(s['method'])
            r['pseudo_quality'],r['pseudo_rejection_reasons']=grade(r,True,config()['B_frozen'],len(peers))
            if r['pseudo_quality']!='unreliable_candidate' and not r.get('pseudo_support_pixels',0):r['pseudo_quality']='unreliable_candidate';r['pseudo_rejection_reasons'].append('empty_pseudo_support_region')
    jsonlines(folder/'records.jsonl',records);csvwrite(folder/'records.csv',records)
    finish_summary()
def finish_summary():
    # Previous reused modules insert old directory into sys.path; use explicit local path.
    import importlib.util
    spec=importlib.util.spec_from_file_location('representation_analysis',ROOT/'analyze.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=ROOT/'runs/confirmation';records=lines(folder/'records.jsonl');chosen=lines(ROOT/'confirmation_samples.jsonl');factor=read(ROOT/'extension_decision.json')['fixed_factor']
    save(folder/'SUMMARY.json',{'fixed_factor':factor,'frozen':study.frozen(),'rows':len(records),'primary_selection_components':sorted({r['component'] for r in selected()}),'confirmation_components':sorted({r['component'] for r in chosen}),'component_overlap':0,'C':0,'Success_metrics_computed':False,'summary':[dict(condition=c,method=m,**module.summarize([r for r in records if r['condition']==c and r['method']==m])) for c in ['original',factor] for m in METHODS]})
if __name__=='__main__':main()
