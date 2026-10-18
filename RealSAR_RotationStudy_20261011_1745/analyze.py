"""Exploratory evidence, not Success@px. All chips/acquisitions/sites distinguished."""
import sys
sys.dont_write_bytecode=True
import json,csv,itertools
from collections import defaultdict,Counter
import numpy as np
from study import ROOT,OLD,MANUAL,rows,context,grade,cfg,save,clean,perspective,regular_grid,sample_valid
from geo import pixel_grid
from labels import hull_mask
from PIL import Image
from sarbench.matchers import Prediction
from sarbench.estimate import estimate

def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def export(name,records):
    if not records:return
    fields=list(dict.fromkeys(k for r in records for k in r))
    with (ROOT/'analysis'/name).open('x',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in records)
def stats(values):
    x=[float(v) for v in values if v is not None and np.isfinite(v)]
    return {'n':len(x),'median':float(np.median(x)) if x else None,'p90':float(np.quantile(x,.9)) if x else None,'sum':float(sum(x))}
def units(ids,source):
    r=[source[s] for s in set(ids)];return {'chips':len(r),'acquisition_pairs':len({x['pair_id'] for x in r}),'sites':len({x['site_name'] for x in r}),'components':len({x['component'] for x in r})}
def failure(r):
    if r.get('reason')=='runtime_error':return 'F_evidence_missing_runtime'
    nk=min(r.get('n_keypoints0') or 0,r.get('n_keypoints1') or 0)
    d=r.get('distribution',{});keyd=[d.get('native_keypoint'+str(i),d.get('raw_keypoint'+str(i),{})) for i in [0,1]]
    if nk<8 or any(q.get('hull_fraction',1)<.01 for q in keyd):return 'A_detection_insufficient_or_clustered'
    if r.get('n_matches',0)<8:return 'B_descriptor_matching_insufficient'
    if r['reason'] in ['few_inliers','low_inlier_ratio','estimation_failed','spatially_degenerate','large_fit_residual']:return 'C_geometry_estimation_or_support'
    if not r.get('accepted') or r.get('pseudo_quality')=='unreliable_candidate':return 'D_reliability_or_B_filter'
    return 'passed_B_candidate_NOT_GT'

def main():
    (ROOT/'analysis').mkdir(exist_ok=False);source={r['sample_id']:r for r in rows()};records=read(ROOT/'runs/full191/records.jsonl');assert len(records)==2292
    geo={g['sample_id']:g for g in json.loads((OLD/'georeference_correspondences/SUMMARY.json').read_text())['records']};byid=defaultdict(list)
    for r in records:byid[r['sample_id']].append(r)
    comparisons=[]
    for sid,recs in byid.items():
        valid,mapper=context(source[sid]);grid=regular_grid((512,512),16);grid=grid[sample_valid(valid[0],grid)&sample_valid(valid[1],mapper(grid))]
        for r in recs:
            if grade(r,True,cfg()['B_frozen'],0)[0]=='unreliable_candidate':continue
            z=np.load(r['cache_path']);mask=z['eligible_inlier_mask'];a=z['points0'][mask];b=z['points1_native'][mask];dense=pixel_grid((512,512));pred=perspective(r['candidate_H'],dense)
            geographic=np.array(Image.open(OLD/'georeference_correspondences'/sid/'geographic_valid_overlap0.png'))>0
            support=hull_mask(a,(512,512))&geographic&valid[0]&sample_valid(valid[1]&hull_mask(b,(512,512)),pred).reshape(512,512);r['pseudo_support_pixel_count']=int(support.sum())
            if support.any():
                folder=ROOT/'analysis/candidate_support';folder.mkdir(exist_ok=True);path=folder/f'{sid}_{r["method"]}_{r["angle"]}.png';Image.fromarray(support.astype(np.uint8)*255).save(path);r['pseudo_support_path']=str(path)
        for r in recs:
            peers={r['method']};eligible=grade(r,True,cfg()['B_frozen'],0)[0]!='unreliable_candidate' and r.get('pseudo_support_pixel_count',0)>0
            if eligible:
                for s in recs:
                    if s['method']==r['method'] or grade(s,True,cfg()['B_frozen'],0)[0]=='unreliable_candidate' or s.get('pseudo_support_pixel_count',0)==0:continue
                    a=perspective(r['candidate_H'],grid);b=perspective(s['candidate_H'],grid);keep=sample_valid(valid[1],a)&sample_valid(valid[1],b);e=np.linalg.norm(a[keep]-b[keep],axis=1)
                    if len(e)>=16 and np.quantile(e,.9)<=cfg()['B_frozen']['B_high']['cross_family_prediction_p90_max_px']:peers.add(s['method'])
            r['pseudo_quality'],r['pseudo_rejection_reasons']=grade(r,True,cfg()['B_frozen'],len(peers) if eligible else 0);r['agreeing_algorithm_families']=len(peers) if eligible else 0;r['failure_stage']=failure(r)
            if r['pseudo_quality']!='unreliable_candidate' and r.get('pseudo_support_pixel_count',0)==0:
                r['pseudo_quality']='unreliable_candidate';r['pseudo_rejection_reasons'].append('empty_pseudo_support_region');r['failure_stage']=failure(r)
        for method in ['sift','orb','lightglue']:
            accepted=[r for r in recs if r['method']==method and r.get('accepted')]
            for a,b in itertools.combinations(accepted,2):
                q0=perspective(a['candidate_H'],grid);q1=perspective(b['candidate_H'],grid);ok=sample_valid(valid[1],q0)&sample_valid(valid[1],q1);e=np.linalg.norm(q0[ok]-q1[ok],axis=1);p90=float(np.quantile(e,.9)) if len(e)>=16 else None
                comparisons.append({'sample_id':sid,'method':method,'angle0':a['angle'],'angle1':b['angle'],'valid_map_grid_points':int(ok.sum()),'native_prediction_difference_p90_px':p90,'contradictory_proxy':p90 is not None and p90>cfg()['contradictory_prediction_grid_p90_px'],'no_independent_correctness_claim':True})
    # Native 0deg general qualification exactly matches the previously frozen B union.
    baselineB={r['sample_id'] for r in records if r['angle']==0 and r['pseudo_quality']!='unreliable_candidate'};oldB={s for s,r in source.items() if r['has_B_pseudo']};assert baselineB==oldB,(baselineB^oldB)
    summaries=[];recoveries=[];strata=[];pairrows=[]
    source_bins={s:('low_<0.5' if min(q['fraction_of_all_pixels'] for q in geo[s]['overlap'])<.5 else 'medium_0.5_0.8' if min(q['fraction_of_all_pixels'] for q in geo[s]['overlap'])<.8 else 'high_>=0.8') for s in source}
    for method in ['sift','orb','lightglue']:
        lookup={(r['sample_id'],r['angle']):r for r in records if r['method']==method}
        for angle in [0,90,180,270]:
            group=[r for r in records if r['method']==method and r['angle']==angle]
            summaries.append({'method':method,'angle':angle,'attempts':len(group),'matching_ge_8':sum(r.get('n_matches',0)>=8 for r in group),'model_available':sum(r.get('output_available',False) for r in group),'gate_accepted':sum(r.get('accepted',False) for r in group),'B_general_eligible':sum(r['pseudo_quality']!='unreliable_candidate' for r in group),'B_high':sum(r['pseudo_quality']=='high_candidate' for r in group),'keypoints0':stats(r.get('n_keypoints0') for r in group),'keypoints1':stats(r.get('n_keypoints1') for r in group),'matches':stats(r.get('n_matches') for r in group),'inliers':stats(r.get('n_inliers') for r in group),'fit_inlier_p90_NOT_GT':stats(r.get('native_fit_residual_p90_px') for r in group),'geo_difference_median_NOT_GT':stats(r.get('geo_difference_median_px') for r in group if r.get('accepted')),'time_ms':stats(r.get('end_to_end_ms') for r in group),'runtime_errors':sum(r.get('reason')=='runtime_error' for r in group),'H_gate_accepted':sum(r.get('homography_diagnostic',{}).get('accepted_full_gate',False) for r in group)})
        for mode,angles in [('180',[180]),('four',[0,180,90,270])]:
            for outcome in ['matching_ge_8','model','gate','B']:
                def status(r):return r.get('n_matches',0)>=8 if outcome=='matching_ge_8' else bool(r.get('output_available')) if outcome=='model' else bool(r.get('accepted')) if outcome=='gate' else r['pseudo_quality']!='unreliable_candidate'
                restored_ids=[s for s in source if not status(lookup[s,0]) and any(status(lookup[s,a]) for a in angles)]
                lost_ids=[s for s in source if status(lookup[s,0]) and not any(status(lookup[s,a]) for a in angles)]
                recoveries.append({'method':method,'comparison':mode,'outcome':outcome,'baseline':sum(status(lookup[s,0]) for s in source),'candidate_union':sum(any(status(lookup[s,a]) for a in angles) for s in source),'recovered':units(restored_ids,source),'lost':units(lost_ids,source),'sample_ids_recovered':restored_ids})
            for s,row in source.items():
                b=lookup[s,0];qs=[lookup[s,a] for a in angles];matched=max(r.get('n_matches',0) for r in qs);hhelp=any(r.get('homography_diagnostic',{}).get('accepted_full_gate') and not r.get('accepted') for r in qs);improved=matched>=max(b.get('n_matches',0)*2,b.get('n_matches',0)+8)
                passgate=any(r.get('accepted') for r in qs);passB=any(r['pseudo_quality']!='unreliable_candidate' for r in qs);allfail=not any(r.get('accepted') or r.get('homography_diagnostic',{}).get('accepted_full_gate') for r in qs);contradict=any(c['sample_id']==s and c['method']==method and c['contradictory_proxy'] for c in comparisons)
                cases=[]
                if improved and passgate:cases.append('A_matches_grow_with_geometry_support')
                if improved and not passgate:cases.append('B_matches_grow_but_geometry_rejected')
                if hhelp:cases.append('C_H_gate_without_Affine_gate_NOT_GT')
                if passgate and not passB:cases.append('D_geometry_candidate_B_rejected')
                if allfail:cases.append('E_all_models_fail_physical_cause_unknown')
                if contradict:cases.append('F_multiple_angles_contradictory')
                pairrows.append({'sample_id':s,'pair_id':row['pair_id'],'method':method,'comparison':mode,'sensor':row['sensor'],'look':row['look_pair_category'],'baseline_matches':b.get('n_matches'), 'max_rotation_matches':matched,'match_growth_proxy':improved,'rotation_gate_restored':not b.get('accepted') and passgate,'rotation_B_restored':b['pseudo_quality']=='unreliable_candidate' and passB,'H_gate_without_Affine_gate':hhelp,'all_models_all_rotations_rejected':allfail,'mechanism_cases_nonexclusive':cases,'data_structure_status':'unknown_requires_independent_static_correspondence'})
    for key in ['sensor','look_pair_category','component','overlap_bin','scene_class']:
        values=sorted({source_bins[s] if key=='overlap_bin' else r[key] for s,r in source.items()})
        for value in values:
            ids=[s for s,r in source.items() if (source_bins[s] if key=='overlap_bin' else r[key])==value]
            for method in ['sift','orb','lightglue','ANY_ALGORITHM_EXPLORATORY_UNION']:
                group=[r for r in records if r['sample_id'] in ids and (method=='ANY_ALGORITHM_EXPLORATORY_UNION' or r['method']==method)]
                for mode,angles in [('0',[0]),('180',[180]),('four',[0,90,180,270])]:
                    selected=[r for r in group if r['angle'] in angles];gateids={r['sample_id'] for r in selected if r.get('accepted')};Bids={r['sample_id'] for r in selected if r['pseudo_quality']!='unreliable_candidate'}
                    strata.append({'stratifier':key,'value':value,'method':method,'mode':mode,**units(ids,source),'gate_chips':len(gateids),'gate_acquisition_pairs':units(gateids,source)['acquisition_pairs'],'B_chips':len(Bids),'B_acquisition_pairs':units(Bids,source)['acquisition_pairs']})
    stages=[]
    geometry=[]
    for sid,r in source.items():
        g=geo[sid]
        geometry.append({k:r.get(k) for k in ['sample_id','pair_id','sensor','site_name','component','scene_class','look_pair_category','reference_acquisition_id','moving_acquisition_id','reference_utc','moving_utc','reference_observation_direction','moving_observation_direction','reference_orbit_direction','moving_orbit_direction','reference_azimuth_degrees','moving_azimuth_degrees','reference_incidence_degrees','moving_incidence_degrees','reference_physical_resolution_metadata','moving_physical_resolution_metadata']})
        geometry[-1].update(metric_geometry=g['metric_geometry'],scale_ratio=g['scale_ratio_0_over_1'],effective_overlap=g['overlap'],image_quality=[{k:v for k,v in q.items() if k!='tiles'} for q in g['image_quality']])
        axes=[float(np.degrees(np.arctan2(q['grid_jacobian_m_per_px'][1][0],q['grid_jacobian_m_per_px'][0][0]))) for q in g['metric_geometry']]
        geometry[-1].update(local_projected_raster_x_axis_angle_from_east_deg=axes,raster_axis_delta_deg=(axes[1]-axes[0]+180)%360-180,raster_axis_note='local map grid basis, NOT SAR observation azimuth; cardinal rotations do not cover arbitrary angles')
    export('source_geometry.csv',geometry)
    negative_metadata=[]
    for r in json.loads((ROOT/'negative_index.json').read_text()):
        a=source[r['reference_source_sample']];b=source[r['moving_source_sample']]
        negative_metadata.append({'negative_id':r['sample_id'],'reference_source_sample':a['sample_id'],'moving_source_sample':b['sample_id'],'reference_metadata':{k:v for k,v in a.items() if k.startswith('reference_') or k.startswith('native_reference_')},'moving_metadata':{k:v for k,v in b.items() if k.startswith('moving_') or k.startswith('native_moving_')},'note':'Use this joined metadata for ancillary angles/licensing. Initial frozen negative manifest had correct paths/IDs/UTC/hash/masks but some unused moving ancillary fields inherited from reference record. Pair selection, geography and inference unchanged.'})
    save(ROOT/'analysis/negative_source_metadata.json',negative_metadata)
    for protocol,group in [('native_0',[r for r in records if r['angle']==0]),('native_all_rotations',records),('geo_assisted_0_reused',[r for r in read(OLD/'quality/model_grades.jsonl') if r['protocol']=='geo_assisted'])]:
        for method in ['sift','orb','lightglue']:
            g=[r for r in group if r['method']==method]
            for stage in ['A_detection_insufficient_or_clustered','B_descriptor_matching_insufficient','C_geometry_estimation_or_support','D_reliability_or_B_filter','F_evidence_missing_runtime','passed_B_candidate_NOT_GT']:
                chosen=[r for r in g if failure(r)==stage];stages.append({'protocol':protocol,'method':method,'stage':stage,'prediction_rows':len(chosen),**units([r['sample_id'] for r in chosen],source),'median_keypoints_min':stats(min(r.get('n_keypoints0') or 0,r.get('n_keypoints1') or 0) for r in chosen),'median_matches':stats(r.get('n_matches') for r in chosen),'median_inliers':stats(r.get('n_inliers') for r in chosen),'example_ids':list(dict.fromkeys(r['sample_id'] for r in chosen))[:3]})
    strategies=read(ROOT/'runs/strategies/strategies.jsonl');negative=read(ROOT/'runs/negative/records.jsonl');negstrategy=read(ROOT/'runs/negative/strategies.jsonl');deploy=[];negsummary=[]
    zero_checks=[];original={(r['sample_id'],r['method']):r for r in records if r['angle']==0}
    for r in read(ROOT/'runs/strategies/records.jsonl'):
        if r['angle']!=0:continue
        b=original[r['sample_id'],r['method']];both=r.get('candidate_H') is not None and b.get('candidate_H') is not None
        zero_checks.append({'sample_id':r['sample_id'],'method':r['method'],'accepted_agrees':r['accepted']==b['accepted'],'reason_agrees':r['reason']==b['reason'],'matches_agree':r.get('n_matches')==b.get('n_matches'),'inliers_agree':r.get('n_inliers')==b.get('n_inliers'),'matrix_max_abs_delta':float(np.max(np.abs(np.array(r['candidate_H'])-b['candidate_H']))) if both else None,'matrix_presence_agrees':(r.get('candidate_H') is None)==(b.get('candidate_H') is None),'purpose':'actual cascade 0deg rerun for timing; frozen cached baseline remains untouched'})
    export('zero_rerun_consistency.csv',zero_checks);save(ROOT/'analysis/zero_rerun_consistency.json',{'checks':len(zero_checks),'accepted_disagreements':sum(not r['accepted_agrees'] for r in zero_checks),'reason_disagreements':sum(not r['reason_agrees'] for r in zero_checks),'match_count_disagreements':sum(not r['matches_agree'] for r in zero_checks),'inlier_count_disagreements':sum(not r['inliers_agree'] for r in zero_checks),'matrix_delta':stats(r['matrix_max_abs_delta'] for r in zero_checks)})
    local=[];representatives=set(cfg()['diagnostic8']+cfg()['representative24'])
    for r in records:
        if r['sample_id'] not in representatives or r.get('n_matches',0)<8:continue
        z=np.load(r['cache_path']);a=z['points0'];b=z['points1_native']
        for gy in range(4):
            for gx in range(4):
                keep=(a[:,0]>=gx*128)&(a[:,0]<(gx+1)*128)&(a[:,1]>=gy*128)&(a[:,1]<(gy+1)*128)
                if keep.sum()<8:continue
                est=estimate(Prediction(r['method'],a[keep].astype(np.float32),b[keep].astype(np.float32)),(512,512),model='affine',seed=42)
                local.append({'sample_id':r['sample_id'],'method':r['method'],'angle':r['angle'],'tile_x':gx,'tile_y':gy,'matches_in_tile':int(keep.sum()),'basic_local_estimator_accepted':est.accepted,'reason':est.reason,'native_candidate_H':clean(est.candidate_H),'n_inliers':est.n_inliers,'inlier_ratio':est.inlier_ratio,'fit_rmse_NOT_GT':est.inlier_rmse,'scope':'128x128 reference tile only; no global/full-gate or pseudo-label upgrade','independent_GT':False})
    for method in ['sift','orb','lightglue']:
        for name in ['A','B']:
            for population,data in [('positive',strategies),('geographic_negative',negstrategy)]:
                g=[r for r in data if r['method']==method and r['strategy']==name]
                deploy.append({'population':population,'method':method,'strategy':name,'pairs':len(g),'accepted':sum(r['accepted'] for r in g),'B_eligible_after_selection':sum(r.get('pseudo_general_eligible_post_selection',False) for r in g),'time_ms':stats(r['end_to_end_ms'] for r in g),'rotations':stats(r['rotations_executed'] for r in g),'actual_execution_shared_prefix_not_sum_estimate':True})
        for angle in [0,90,180,270,'ANY_FOUR']:
            g=[r for r in negative if r['method']==method and (angle=='ANY_FOUR' or r['angle']==angle)];accepted={r['sample_id'] for r in g if r['accepted']};negsummary.append({'method':method,'angle':angle,'negative_pairs':30,'wrong_geographic_accept_pairs':len(accepted),'empirical_rate':len(accepted)/30,'accepted_ids':sorted(accepted),'note':'geographic nonoverlap known; shared acquisition/site sources, no independent-binomial claim'})
    fullB={r['sample_id'] for r in records if r['pseudo_quality']!='unreliable_candidate'};fullgate={r['sample_id'] for r in records if r.get('accepted')};newB=fullB-baselineB;different={s for s,r in source.items() if r['look_pair_category']=='different_look_verified'}
    baseline_gate={r['sample_id'] for r in records if r['angle']==0 and r.get('accepted')};summary={'source_units':units(source,source),'record_rows':len(records),'C':0,'baseline_B':units(baselineB,source),'four_B':units(fullB,source),'new_B':units(newB,source),'new_B_ids':sorted(newB),'baseline_gate':units(baseline_gate,source),'four_gate':units(fullgate,source),'new_gate':units(fullgate-baseline_gate,source),'different_look_B':units(fullB&different,source),'different_look_new_gate':units((fullgate-baseline_gate)&different,source),'contradictions':len([r for r in comparisons if r['contradictory_proxy']]),'contradictory_chips':units([r['sample_id'] for r in comparisons if r['contradictory_proxy']],source),'physical_static_structure_confirmed_E_failures':0,'physical_static_structure_evidence_insufficient':191,'no_accuracy_metric':True,'new_pseudo_grade_does_not_modify_old_labels':True}
    for name,data in [('rotation_results.csv',records),('rotation_summary.csv',summaries),('recoveries.csv',recoveries),('strata.csv',strata),('mechanisms.csv',pairrows),('failure_stages.csv',stages),('strategy_results.csv',strategies+negstrategy),('strategy_summary.csv',deploy),('negative_results.csv',negative),('negative_summary.csv',negsummary),('contradictions.csv',comparisons)]:export(name,data)
    mechanism_summary=[]
    for case in ['A_matches_grow_with_geometry_support','B_matches_grow_but_geometry_rejected','C_H_gate_without_Affine_gate_NOT_GT','D_geometry_candidate_B_rejected','E_all_models_fail_physical_cause_unknown','F_multiple_angles_contradictory']:
        g=[r for r in pairrows if r['comparison']=='four' and case in r['mechanism_cases_nonexclusive']];mechanism_summary.append(dict(case=case,method_pair_rows=len(g),**units([r['sample_id'] for r in g],source),example_ids=list(dict.fromkeys(r['sample_id'] for r in g))[:3]))
    export('mechanism_summary.csv',mechanism_summary);save(ROOT/'analysis/mechanism_summary.json',mechanism_summary)
    export('local_affine_diagnostics.csv',local)
    for name,data in [('rotation_summary.json',summaries),('recoveries.json',recoveries),('strata.json',strata),('failure_stages.json',stages),('strategy_summary.json',deploy),('negative_summary.json',negsummary),('summary.json',summary),('graded_records.json',records),('contradictions.json',comparisons)]:save(ROOT/'analysis'/name,data)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
