"""Only the new model runs; baseline predictions and exact Equal budgets stay frozen."""
from mixed_io import *
from keep_data import OLD,OLDTAG
from joint_evaluate import feature_save,save_prediction


def frozen_rows():
    return [json.loads(s) for s in (OLD/'predictions/final'/f'scores_{OLDTAG}.jsonl').read_text().splitlines()]


def run(model,tag,rows=None):
    rows=records('val') if rows is None else rows
    old={(r['pair_id'],r['protocol']):r for r in frozen_rows() if r['method']=='b'}
    folder=ROOT/'predictions'/tag;folder.mkdir(parents=True,exist_ok=True);log=folder/f'scores_{TAG}.jsonl'
    done=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else [];completed={r['pair_id'] for r in done}
    lg=LightGlue(features='superpoint').cuda().eval().requires_grad_(False)
    for n,row in enumerate(rows):
        pid=row['pair_id']
        if pid in completed:continue
        ims=images(row);shape=ims[0].shape;x,rs,ms=shared_cache(row,model,ims)
        with torch.no_grad():
            start=time.perf_counter();logits,dense=model.cached_maps(x,rs);fs=[model.extract_maps(logits[i:i+1],dense[i:i+1],shape) for i in [0,1]];torch.cuda.synchronize();feature_ms=(time.perf_counter()-start)*1000
        # This comparison detects frozen-detector drift; it never alters candidate points.
        with np.load(OLD/'predictions/final'/f'{pid}_b_natural_{OLDTAG}.npz') as z:
            assert all(np.array_equal(fs[i]['keypoints'][0].cpu().numpy(),z['keypoints_'+L]) for i,L in enumerate('AB')), 'Detector changed'
        budgets=old[(pid,'equal')]['common_budget']
        feature_save(folder/f'{pid}_keep_top128_{TAG}.npz',fs);rr=[]
        for mode in ['natural','equal']:
            selected=[trim(f,1024 if mode=='natural' else budgets[i]) for i,f in enumerate(fs)]
            start=time.perf_counter();pred,e,diag,conf=predict(lg,selected,shape);torch.cuda.synchronize();match_ms=(time.perf_counter()-start)*1000
            r=scorer.assess(row,pred);r.update(method='keep',protocol=mode,common_budget=budgets,keypoints0=len(selected[0]['keypoints'][0]),keypoints1=len(selected[1]['keypoints'][0]),inliers=e.n_inliers,reason=e.reason,state='accepted' if e.accepted else 'rejected' if e.candidate_H is not None else 'no_output',feature_heads_ms=feature_ms,shared_feature_ms=ms,matcher_geometry_ms=match_ms,cache_used=True,role=row['loss_role'],component=row['source_component_id'],detector_identical_to_initialization=True)
            rr.append(clean(r));save_prediction(folder/f'{pid}_keep_{mode}_{TAG}.npz',pred,e,conf)
        with log.open('a') as f:f.write(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr));f.flush();os.fsync(f.fileno())
        done.extend(rr)
        if n%50==0 or n==len(rows)-1:print(now(),'VAL',tag,n+1,len(rows),flush=True)
    summary={mode:scorer.summaries([r for r in done if r['protocol']==mode]) for mode in ['natural','equal']}
    write(ROOT/f'summary_{tag}_{stamp()}.json',summary);csvwrite(ROOT/f'RESULTS_{tag}_{stamp()}.csv',done);del lg;torch.cuda.empty_cache()
    return done,summary
