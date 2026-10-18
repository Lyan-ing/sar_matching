"""All-condition tables, unchanged pseudo qualification and auditable review tasks."""
import sys
sys.dont_write_bytecode=True
import shutil,itertools
from experiment import *
from geo import pixel_grid
from study import projection_diagnostics
def units(recs):
    return {'chips':len({r['sample_id'] for r in recs}),'acquisition_pairs':len({r['pair_id'] for r in recs}),'sites':len({r['site_name'] for r in recs}),'components':len({r['component'] for r in recs})}
def metric(recs,key):
    a=[r[key] for r in recs if r.get(key) is not None and np.isfinite(r[key])]
    return {'n':len(a),'median':float(np.median(a)) if a else None,'p90':float(np.quantile(a,.9)) if a else None}
def qualified(r):return r.get('pseudo_quality','unreliable_candidate')!='unreliable_candidate'
def failure(r):
    if r.get('reason')=='source_context_unavailable':return 'source_missing'
    if r.get('reason','').startswith('coarse_frozen'):return 'coarse_rejected'
    if r.get('reason','')=='no_coarse_projection_support':return 'coarse_outside_native512'
    if min(r.get('n_keypoints0') or 0,r.get('n_keypoints1') or 0)<8:return 'detection_insufficient'
    if r.get('n_matches',0)<8:return 'descriptor_matches_insufficient'
    if not r.get('accepted'):return 'geometry_or_reliability_rejected'
    if not qualified(r):return 'B_QA_rejected'
    return 'B_candidate_NOT_GT'
def summarize(recs):
    return dict(units(recs),prediction_rows=len(recs),accepted=sum(r.get('accepted',False) for r in recs),accepted_units=units([r for r in recs if r.get('accepted')]),B_eligible=sum(qualified(r) for r in recs),B_units=units([r for r in recs if qualified(r)]),high=sum(r.get('pseudo_quality')=='high_candidate' for r in recs),matches=metric(recs,'n_matches'),inliers=metric(recs,'n_inliers'),time_ms=metric(recs,'end_to_end_ms'),peak_cuda_MB=metric(recs,'peak_cuda_allocated_MB'),failures=dict(Counter(failure(r) for r in recs)))
def main():
    out=ROOT/'analysis';out.mkdir(exist_ok=False);(out/'candidate_support').mkdir();source={r['sample_id']:r for r in selected()};records=lines(ROOT/'runs/radiometry/records.jsonl')+lines(ROOT/'runs/context/records.jsonl');cfg=config()['B_frozen'];byid=defaultdict(list)
    for r in records:byid[r['sample_id']].append(r)
    for sid,recs in byid.items():
        _,valid,_,_,mapper=load_pair(source[sid]);grid=regular_grid((512,512));grid=grid[sample_valid(valid[0],grid)&sample_valid(valid[1],mapper(grid))]
        for r in recs:
            # Both frozen controls retain their previously cross-protocol graded version.
            if r['condition'] in ['original','geo_assisted512_control']:continue
            if grade(r,True,cfg,0)[0]=='unreliable_candidate':continue
            z=np.load(r['cache_path']);mask=z['eligible_inlier_mask'];a=z['points0'][mask];b=z['points1_native'][mask]
            geo=np.array(Image.open(OLD/'georeference_correspondences'/sid/'geographic_valid_overlap0.png'))>0
            region=hull_mask(a,(512,512))&geo&valid[0]&sample_valid(valid[1]&hull_mask(b,(512,512)),perspective(r['candidate_H'],pixel_grid((512,512)))).reshape(512,512)
            r['pseudo_support_pixels']=int(region.sum())
            if region.any():
                p=out/'candidate_support'/f'{sid}_{r["condition"]}_{r["method"]}.png';Image.fromarray(region.astype(np.uint8)*255).save(p);r['pseudo_support_path']=str(p)
        for r in recs:
            if r['condition'] in ['original','geo_assisted512_control']:r['baseline_quality_scope']='old fixed candidate quality, with original cross-protocol consensus';continue
            peers={r['method']}
            if r.get('pseudo_support_pixels',0):
                for s in recs:
                    if s['method']==r['method'] or s['condition']!=r['condition'] or not s.get('pseudo_support_pixels',0):continue
                    a=perspective(r['candidate_H'],grid);b=perspective(s['candidate_H'],grid);m=sample_valid(valid[1],a)&sample_valid(valid[1],b);e=np.linalg.norm(a[m]-b[m],axis=1)
                    if len(e)>=16 and np.quantile(e,.9)<=cfg['B_high']['cross_family_prediction_p90_max_px']:peers.add(s['method'])
            r['pseudo_quality'],r['pseudo_rejection_reasons']=grade(r,True,cfg,len(peers));r['agreeing_algorithm_families']=len(peers)
            if qualified(r) and not r.get('pseudo_support_pixels',0):r['pseudo_quality']='unreliable_candidate';r['pseudo_rejection_reasons'].append('empty_pseudo_support_region')
            r['quality_consensus_scope']='same representation/context condition, different algorithm families only; no factor cherry picking'
        for r in recs:r['failure_stage']=failure(r)
    jsonlines(out/'all_records.jsonl',records);csvwrite(out/'all_records.csv',records)
    summary=[];strata=[];changes=[]
    for c in ['original',*REPRESENTATIONS,'geo_assisted512_control','context_direct','context_fine']:
        for m in METHODS:
            group=[r for r in records if r['condition']==c and r['method']==m];summary.append(dict(condition=c,method=m,**summarize(group)))
            for axis in ['sensor','look_pair_category','scene_class','component','overlap_bin']:
                bins=defaultdict(list)
                for r in group:
                    overlap=min(v['fraction_of_all_pixels'] for v in source[r['sample_id']]['effective_geographic_overlap']);value=('low_<.5' if overlap<.5 else 'medium_.5_.8' if overlap<.8 else 'high_>=.8') if axis=='overlap_bin' else r[axis];bins[value].append(r)
                for value,rs in bins.items():strata.append(dict(condition=c,method=m,axis=axis,value=value,**summarize(rs)))
            if c=='original':continue
            base={r['sample_id']:r for r in records if r['condition']=='original' and r['method']==m}
            for r in group:
                b=base[r['sample_id']];changes.append(dict({k:r[k] for k in ['sample_id','pair_id','sensor','site_name','component','look_pair_category','condition','method']},delta_keypoints_min=min(r.get('n_keypoints0') or 0,r.get('n_keypoints1') or 0)-min(b.get('n_keypoints0') or 0,b.get('n_keypoints1') or 0),delta_matches=r['n_matches']-b['n_matches'],delta_inliers=r['n_inliers']-b['n_inliers'],baseline_accepted=b['accepted'],candidate_accepted=r.get('accepted',False),new_gate=bool(r.get('accepted') and not b['accepted']),lost_gate=bool(b['accepted'] and not r.get('accepted')),new_B=qualified(r) and not qualified(b),lost_B=qualified(b) and not qualified(r),geographic_difference_median_px=r.get('geo_difference_median_px'),time_ms=r['end_to_end_ms'],baseline_time_ms=b['end_to_end_ms'],baseline_time_scope='old cached run different process/date, not paired runtime benchmark'))
    save(out/'summary.json',summary);csvwrite(out/'summary.csv',summary);save(out/'strata.json',strata);csvwrite(out/'strata.csv',strata);csvwrite(out/'changes.csv',changes)
    baseline_union={r['sample_id'] for r in records if r['condition']=='original' and qualified(r)}
    newcandidates=[];progress=[]
    for c in REPRESENTATIONS+['context_direct','context_fine']:
        rs=[r for r in records if r['condition']==c];new=[r for r in rs if qualified(r) and r['sample_id'] not in baseline_union];delta=[r for r in changes if r['condition']==c];progress.append(dict(condition=c,new_vs_baseline_ANY_family_B_units=units(new),new_vs_same_method_gate_units=units([r for r in delta if r['new_gate']]),new_vs_same_method_B_units=units([r for r in delta if r['new_B']]),lost_vs_same_method_B_units=units([r for r in delta if r['lost_B']]),expansion_eligible=bool(new)))
        for r in rs:
            if not qualified(r):continue
            z=np.load(r['cache_path']);m=z['eligible_inlier_mask'];pts=[{'point_id':f'PSEUDO_{i:04d}','xy0':a.tolist(),'xy1':b.tolist(),'label_type':'pseudo_correspondences','generator':c+'/'+r['method'],'human_verified':False,'static_verified':False} for i,(a,b) in enumerate(zip(z['points0'][m],z['points1_native'][m]))]
            obj=dict(sample_id=r['sample_id'],condition=c,method=r['method'],pseudo_quality=r['pseudo_quality'],label_type='new_version_candidate',points=pts,transform=r['candidate_H'],support_path=r.get('pseudo_support_path'),not_old_B_replacement=True,independent_GT=False,training_eligible_now=False,new_sample_vs_baseline_ANY_family=r['sample_id'] not in baseline_union)
            folder=ROOT/'new_candidates';folder.mkdir(exist_ok=True);save(folder/f'{r["sample_id"]}_{c}_{r["method"]}.json',obj);newcandidates.append(obj)
    save(out/'progression.json',progress);save(out/'candidate_summary.json',{'candidate_predictions':len(newcandidates),'units':units([r for r in records if r['condition'] in REPRESENTATIONS+['context_direct','context_fine'] and qualified(r)]),'new_sample_ids':sorted({r['sample_id'] for r in records if r['condition'] in REPRESENTATIONS+['context_direct','context_fine'] and qualified(r)}-baseline_union),'C':0})
    build_review(source,records,changes)
    print('summary',[(r['condition'],r['method'],r['accepted'],r['B_eligible']) for r in summary],flush=True);print('progression',progress,flush=True)

def build_review(source,records,changes):
    root=ROOT/'annotation_adapter'
    for d in ['annotation_tasks','candidate_matches','annotations','gt_quality']:(root/d).mkdir(parents=True,exist_ok=False)
    for name in ['index.html','app.js','coordinates.js']:shutil.copyfile(OLD/'annotation_adapter'/name,root/name)
    ids=['S1SF_EXTENSION_000__chip_002','Capella_0003__chip_001','Umbra_0061__chip_001','Umbra_0071__chip_000']
    diagnostic=[r for r in changes if r['condition'] in REPRESENTATIONS+['context_fine','context_direct'] and r['sample_id'] not in ids and r['new_gate']]
    if diagnostic:
        d=min(diagnostic,key=lambda r:(not r['new_B'],-r['delta_inliers'],r['sample_id']));ids.append(d['sample_id']);reason='post-result diagnostic: new gate candidate, prioritize new B then inlier change; not unbiased'
    else:ids.append(next(r['sample_id'] for r in source.values() if r['site_name']=='Bingham Copper Mine'));reason='pre-fixed mine sample for difficult high relief diagnosis, no newly plausible candidate exists'
    tasks=[]
    for i,sid in enumerate(ids):
        row=source[sid];rs=[r for r in records if r['sample_id']==sid and r['condition']!='geo_assisted512_control' and r.get('cache_path')];r=min(rs,key=lambda r:(not qualified(r),not r.get('accepted'),-r['n_inliers'],r['condition'],r['method']))
        cache=np.load(r['cache_path']);p0=cache['points0'];p1=cache['points1_native'];m=cache.get('eligible_inlier_mask',np.zeros(len(p0),bool));points=[{'xy0':a.tolist(),'xy1':b.tolist(),'point_id':f'AUTO_{j:04d}'} for j,(a,b) in enumerate(zip(p0[m],p1[m]))]
        geo=read(row['geographic_correspondences_path'])['points'];payload={'sample_id':sid,'geographic':geo,'sift_candidates':points,'pseudo_generator':r['condition']+'/'+r['method'],'pseudo_quality':r['pseudo_quality'],'independent_GT':False,'legacy_slot_mapping':'sift_candidates contains named actual generator, NOT necessarily SIFT','exposure_rule':'hidden by default, permanently audited, assistance is not independent'}
        p=root/'candidate_matches'/f'{sid}.json';save(p,payload)
        tasks.append(dict(row,image0=row['reference_native_png'],image1=row['moving_native_png'],width0=512,height0=512,width1=512,height1=512,visual_scene='site_context:'+row['scene_class'],annotation_readiness='independent_static_structure_review_pending',visual_note='先独立查找建筑角点、固定码头/道路交叉、固定矿区边界等；不能假定亮点是同一散射中心。候选默认隐藏。',selection_reason=reason if i==4 else 'pre-fixed representative: same-look control, different-look city and two fixed port sites',selection_scope='post_result_diagnostic' if i==4 and diagnostic else 'pre-fixed metadata representative',demo_priority=i+1,candidate_sha256=sha(p),candidate_count=len(geo)+len(points)))
    path=root/'annotation_tasks/tasks.jsonl';jsonlines(path,tasks);save(root/'annotation_tasks/task_freeze.json',{'at':now(),'task_sha256':sha(path),'tasks':len(tasks),'C':0,'diagnostic_not_unbiased':True});save(root/'review_selection.json',[{k:r[k] for k in ['sample_id','site_name','look_pair_category','selection_reason','selection_scope']} for r in tasks])

if __name__=='__main__':main()
