"""Independent correspondence evidence; never qualifies by fit residual alone."""
import math
import numpy as np
import cv2

POLICY={'schema_version':1,'coordinate_convention':'pixel centers; 0..W-1, 0..H-1',
        'local_min_checks':3,'local_min_hull_fraction':.001,
        'global_min_checks':6,'global_min_hull_fraction':.1,'global_min_span_fraction':.5,
        'success_min_checks':4,'uncertainty_max_fraction_of_threshold':1/3,
        'independent_review_required':True,'assisted_points_formal_eligible':False,
        'notes':'Predeclared evidence screening. Hull is sampled support, not proof of dense accuracy. Uncertainty is declared by humans + observed reviewer disagreement, not calibrated 95% confidence.'}
DISPOSITIONS=['pending','reliable_correspondences','local_only','insufficient_points','excessive_view_change','no_static_common_structure','other_not_evaluable']

def coord(p,shape):
    if not isinstance(p,list) or len(p)!=2:raise ValueError('missing coordinate in one view')
    h,w=shape
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in p):raise ValueError('nonfinite coordinate')
    if not 0<=p[0]<=w-1 or not 0<=p[1]<=h-1:raise ValueError('out-of-bounds pixel center')
    return np.asarray(p,float)

def spatial(points,shape):
    p=np.asarray(points,float).reshape(-1,2);h,w=shape
    if not len(p):return {'hull_fraction':0.,'span':[0.,0.],'grid_cells':0}
    hull=float(cv2.contourArea(cv2.convexHull(np.float32(p))))/(h*w) if len(p)>=3 else 0.
    return {'hull_fraction':hull,'span':(np.ptp(p,axis=0)/[w,h]).tolist(),'grid_cells':len(set(map(tuple,np.minimum(3,np.floor(p/[w,h]*4)).astype(int))))}

def validate_points(points,task):
    ids=[];a=[];b=[]
    for p in points:
        pid=p.get('point_id')
        if not isinstance(pid,str) or not pid or len(pid)>80:raise ValueError('invalid point ID')
        if p.get('sample_id')!=task['sample_id']:raise ValueError('point/sample ID mismatch')
        if p.get('role') not in ['control','check']:raise ValueError('role must be control or check')
        ids.append(pid);a.append(coord(p.get('xy0'),(task['height0'],task['width0'])));b.append(coord(p.get('xy1'),(task['height1'],task['width1'])))
        for k in ['sigma0_px','sigma1_px']:
            v=p.get(k)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError('explicit nonnegative localization uncertainty required')
    if len(set(ids))!=len(ids):raise ValueError('duplicate point IDs')
    for pts in [a,b]:
        for i in range(len(pts)):
            if any(np.linalg.norm(pts[i]-pts[j])<.5 for j in range(i)):raise ValueError('duplicate coordinate within 0.5px in either view')
    return np.asarray(a).reshape(-1,2),np.asarray(b).reshape(-1,2)

def quality(doc,task):
    pts=doc.get('points',[]);a,b=validate_points(pts,task);eligible=[];warnings=[];evidence=[]
    for i in range(len(pts)):
        for j in range(i):
            d0=float(np.linalg.norm(a[i]-a[j]));d1=float(np.linalg.norm(b[i]-b[j]))
            if min(d0,d1)<32 and max(d0,d1)>128:warnings.append(f"{pts[i]['point_id']}/{pts[j]['point_id']}: local neighborhood conflict or strong nonuniform geometry; independent recheck needed")
    for p in pts:
        r=p.get('review');reasons=[]
        if not p.get('annotator') or not p.get('annotated_at'):reasons.append('missing_annotation_provenance')
        if p.get('is_static') is not True or p.get('visible_both') is not True or not p.get('landmark_type'):reasons.append('static_or_visibility_not_explicit')
        if p.get('confidence') not in ['high','medium']:reasons.append('low_or_unknown_confidence')
        if p.get('used_assistance'):reasons.append('annotator_assisted')
        u=None;disagreement=None
        if not r:reasons.append('unreviewed')
        else:
            if r.get('reviewer')==p.get('annotator') or not r.get('reviewer') or not r.get('reviewed_at'):reasons.append('nonindependent_review')
            if r.get('conclusion')!='confirmed' or r.get('static_confirmed') is not True or r.get('visible_confirmed') is not True:reasons.append('review_not_confirmed')
            if r.get('used_assistance'):reasons.append('reviewer_assisted')
            try:
                q0=coord(r.get('xy0'),(task['height0'],task['width0']));q1=coord(r.get('xy1'),(task['height1'],task['width1']))
                d0=float(np.linalg.norm(q0-np.asarray(p['xy0'])));d1=float(np.linalg.norm(q1-np.asarray(p['xy1'])))
                sigma=[r['sigma0_px'],r['sigma1_px']]
                if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in sigma):raise ValueError('invalid review uncertainty')
                u=float(math.hypot(max(p['sigma0_px'],sigma[0],d0),max(p['sigma1_px'],sigma[1],d1)));disagreement=[d0,d1]
                if max(d0,d1)>3:warnings.append(p['point_id']+': reviewer localization difference >3px; recheck landmark')
                if max(d0,d1)>max(3,3*max(p['sigma0_px'],p['sigma1_px'],*sigma)):reasons.append('unresolved_large_annotator_difference')
            except (KeyError,ValueError,TypeError):reasons.append('missing_independent_reclick_or_uncertainty')
        evidence.append({'point_id':p['point_id'],'eligible':not reasons,'reasons':reasons,'combined_uncertainty_px':u,'review_disagreement_px':disagreement,'role':p['role']})
        if not reasons:eligible.append(p)
    checks=[p for p in eligible if p['role']=='check'];controls=[p for p in eligible if p['role']=='control']
    s0=spatial([p['xy0'] for p in checks],(task['height0'],task['width0']));s1=spatial([p['xy1'] for p in checks],(task['height1'],task['width1']))
    scope=doc.get('evaluation_scope','local');disposition=doc.get('disposition','pending')
    if disposition not in DISPOSITIONS:raise ValueError('unknown disposition')
    if scope not in ['local','global_sparse']:raise ValueError('unknown evaluation scope')
    support=len(checks)>=POLICY['local_min_checks'] and min(s0['hull_fraction'],s1['hull_fraction'])>=POLICY['local_min_hull_fraction']
    if scope=='global_sparse':support=support and len(checks)>=6 and min(s0['hull_fraction'],s1['hull_fraction'])>=.1 and min(*s0['span'],*s1['span'])>=.5
    if disposition in ['excessive_view_change','no_static_common_structure','other_not_evaluable']:
        if not doc.get('disposition_note','').strip():raise ValueError('not_evaluable requires explicit reason')
        state='not_evaluable'
    elif support and disposition in ['reliable_correspondences','local_only']:state='gt_qualified'
    elif any(p.get('review') for p in pts):state='reviewed_correspondences'
    elif pts:state='reviewed_correspondences' # preliminary human annotation, never qualified
    else:state='candidate_matches' if task.get('candidate_count',0) else 'unverified'
    bounds=[e['combined_uncertainty_px'] for e in evidence if e['eligible'] and e['role']=='check'];maximum=max(bounds) if bounds else None
    thresholds=[t for t in [3,5,10] if state=='gt_qualified' and len(checks)>=4 and maximum<=t/3]
    fit=None
    # Review signals only: never override evidence eligibility or force a global H.
    if len(controls)>=3 and len(checks):
        X=np.c_[[p['xy0'] for p in controls],np.ones(len(controls))]
        if np.linalg.matrix_rank(X)==3:
            T=np.linalg.lstsq(X,np.array([p['xy1'] for p in controls]),rcond=None)[0]
            q=np.c_[[p['xy0'] for p in checks],np.ones(len(checks))]@T;e=np.linalg.norm(q-np.array([p['xy1'] for p in checks]),axis=1)
            fit={'controls_fit_only':len(controls),'held_out_checks_only':len(checks),'affine_2x3':T.T.tolist(),'check_mean_px':float(e.mean()),'check_p90_px':float(np.quantile(e,.9)),'role':'model limitation signal, not qualification proof'}
            if np.quantile(e,.9)>10:warnings.append('global affine cannot explain local points or possible point conflict; inspect without deleting difficult GT')
    if pts and min(spatial(a,(task['height0'],task['width0']))['hull_fraction'],spatial(b,(task['height1'],task['width1']))['hull_fraction'])<.01:warnings.append('annotations spatially concentrated')
    return {'status':state,'scope':scope,'eligible_checks':len(checks),'eligible_controls':len(controls),'eligible_point_ids':[p['point_id'] for p in eligible],'check_point_ids':[p['point_id'] for p in checks],'point_evidence':evidence,'check_support0':s0,'check_support1':s1,'max_combined_uncertainty_px':maximum,'supported_success_thresholds_px':thresholds,'fit_check_diagnostic':fit,'warnings':warnings,'formal_errors_allowed':state=='gt_qualified','full_image_dense_GT':False}

def quantitative(H,accepted,doc,task,ground_metric=None,prediction_mapping=None):
    q=quality(doc,task)
    if q['status']!='gt_qualified':raise ValueError('Independent gt_qualified evidence required for formal accuracy')
    ids=set(q['check_point_ids']);p=[p for p in doc['points'] if p['point_id'] in ids]
    # Only checks. Control points never enter prediction evaluation.
    result={'n_checks':len(p),'scope':q['scope'],'gt_uncertainty_px':q['max_combined_uncertainty_px'],'supported_thresholds_px':q['supported_success_thresholds_px'],'error_mean_px':None,'error_median_px':None,'error_p90_px':None,'error_p95_px':None,'ground_error_mean_m':None,'decision':'rejected' if not accepted else 'accepted_unassessable'}
    if H is None:
        for t in q['supported_success_thresholds_px']:result[f'sparse_Success@{t}px']=False
        result['decision']='no_output' if accepted is False else 'invalid_output';return result
    H=np.asarray(H,float)
    if H.shape!=(3,3) or not np.isfinite(H).all():raise ValueError('invalid prediction H')
    a=np.array([p['xy0'] for p in p]);b=np.array([p['xy1'] for p in p]);h=np.c_[a,np.ones(len(a))]@H.T
    with np.errstate(divide='ignore',invalid='ignore'):pred=h[:,:2]/h[:,2:]
    if prediction_mapping is not None:pred=prediction_mapping(pred)
    err=np.linalg.norm(pred-b,axis=1)
    if not np.isfinite(err).all():result['decision']='invalid_output';return result
    result.update(error_mean_px=float(err.mean()),error_median_px=float(np.median(err)),error_p90_px=float(np.quantile(err,.9)),error_p95_px=float(np.quantile(err,.95)))
    result['independent_check_errors']=[{'point_id':point['point_id'],'moving_native_error_px':float(e)} for point,e in zip(p,err)]
    if ground_metric is not None:
        # Caller may supply a map-grid Jacobian only under explicit geocoding assumptions.
        M=np.asarray(ground_metric,float)
        if M.shape!=(2,2) or not np.isfinite(M).all():raise ValueError('ground Jacobian must be finite 2x2')
        result['ground_error_mean_m']=float(np.mean(np.linalg.norm((pred-b)@M.T,axis=1)));result['ground_metric_note']='Map-grid residual distance conditional on provider geocoding; not independent absolute ground accuracy'
    for t in q['supported_success_thresholds_px']:
        result[f'sparse_Success@{t}px']=bool(accepted and np.mean(err)<=t)
        result[f'decision@{t}px']=('sparse_success' if np.mean(err)<=t else 'incorrect_acceptance') if accepted else 'rejected'
    if accepted and q['supported_success_thresholds_px']:
        threshold=max(q['supported_success_thresholds_px']);result['decision']='accepted_sparse_success' if np.mean(err)<=threshold else 'incorrect_acceptance';result['decision_threshold_px']=threshold
    elif accepted:result['decision']='accepted_continuous_error_only'
    else:result['decision']='rejected'
    return result
