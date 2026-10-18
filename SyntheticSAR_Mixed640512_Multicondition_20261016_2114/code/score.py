"""GT consumed only after a prediction. Missing predictions remain in denominators."""
from common import *
import argparse

def symmetric_error(H,p,q):
    if H is None or not len(p):return None
    try:
        H=np.asarray(H,np.float64).reshape(3,3)
        if not np.isfinite(H).all() or np.linalg.cond(H)>1e12:return None
        d=np.r_[np.linalg.norm(transform_points(H,p)-q,axis=1),np.linalg.norm(transform_points(np.linalg.inv(H),q)-p,axis=1)]
        return float(d.mean()) if np.isfinite(d).all() else None
    except (ValueError,np.linalg.LinAlgError):return None

def assess(row,pred):
    with np.load(row['gt_path'],allow_pickle=False) as z:gt={k:z[k] for k in z.files}
    p,q=gt['check_points_A'],gt['check_points_B'];no=bool(gt['no_overlap']);eligible=not no and len(p)>=4
    declared=bool(pred.get('accepted',False));H=pred.get('H_AB');err=symmetric_error(H,p,q)
    a=np.asarray(pred.get('matches_A',[]),np.float64).reshape(-1,2);b=np.asarray(pred.get('matches_B',[]),np.float64).reshape(-1,2)
    if len(a)!=len(b):raise ValueError('Prediction match-array lengths differ')
    static=[gt['static_'+L]&~gt['ignore_'+L]&gt['valid_'+L] for L in 'AB']
    good=np.zeros(len(a),bool);validmatch=np.zeros(len(a),bool)
    if len(a) and not no:
        target=transform_points(gt['H_AB'],a)
        validmatch=bilinear_at(static[0],a)&bilinear_at(static[1],b)&bilinear_at(static[1],target)
        forward=np.linalg.norm(target-b,axis=1);reverse=np.linalg.norm(transform_points(gt['H_BA'],b)-a,axis=1)
        good=validmatch&(forward<=3)&(reverse<=3)
    # Anchor recall: fraction of eligible saved STRUCTURE anchors recovered in both
    # images within 3px by at least one geometrically correct predicted match.
    sp,sq=gt['structure_points_A'],gt['structure_points_B'];anchor_recalled=0
    if len(sp) and good.any():
        covered=(np.linalg.norm(sp[:,None]-a[None,good],axis=2)<=3)&(np.linalg.norm(sq[:,None]-b[None,good],axis=2)<=3)
        anchor_recalled=int(covered.any(1).sum())
    k0=np.asarray(pred.get('keypoints_A',[]),np.float64).reshape(-1,2);k1=np.asarray(pred.get('keypoints_B',[]),np.float64).reshape(-1,2)
    visible=0;repeatable=0
    if len(k0) and not no:
        mapped=transform_points(gt['H_AB'],k0);vk=bilinear_at(static[0],k0)&bilinear_at(static[1],mapped);visible=int(vk.sum())
        if len(k1) and visible:
            qvalid=k1[bilinear_at(static[1],k1)]
            if len(qvalid):repeatable=int((np.linalg.norm(mapped[vk,None]-qvalid[None],axis=2)<=3).any(1).sum())
    result=dict(pair_id=row['pair_id'],size=row['width'],split=row['split'],scene=row['source_scene'],overlap_bin=row['overlap_bin'],angle_bin=row['angle_bin'],profile=row['enhancement_profile'],
        prediction_present=bool(pred),accepted=declared,valid_predicted_H=H is not None and (err is not None or no),
        no_overlap=no,geometry_eligible=eligible,structure_global_eligible=row['global_eligible'],
        bidirectional_error_px=err,normalized_error_diagonal=err/(np.sqrt(2)*row['width']) if err is not None else None,
        match_count=len(a),supervisable_match_count=int(validmatch.sum()),correct_structure_matches=int(good.sum()),
        structure_precision=float(good.sum()/validmatch.sum()) if validmatch.any() else None,
        all_match_correct_structure_fraction=float(good.mean()) if len(good) else 0.,
        anchor_recalled=anchor_recalled,structure_anchor_count=len(sp),structure_anchor_recall=anchor_recalled/len(sp) if len(sp) else None,
        visible_detected_structure_points=visible,repeatable_structure_points=repeatable,structure_repeatability=repeatable/visible if visible else None,
        false_accept=declared if no else None,rejected=not declared)
    for t in [3,5,10]:
        value=bool(declared and err is not None and err<=t)
        result[f'Success@{t}px']=value if eligible else None
        result[f'StructureSuccess@{t}px']=value if row['global_eligible'] else None
    return result

def aggregate(rr):
    def avg(key,items=rr):
        a=[x[key] for x in items if x.get(key) is not None];return float(np.mean(a)) if a else None
    return dict(count=len(rr),geometry_positive_count=sum(r['geometry_eligible'] for r in rr),structure_global_count=sum(r['structure_global_eligible'] for r in rr),
        no_overlap_count=sum(r['no_overlap'] for r in rr),missing_predictions=sum(not r['prediction_present'] for r in rr),
        **{k:avg(k) for k in ['Success@3px','Success@5px','Success@10px','StructureSuccess@3px','StructureSuccess@5px','StructureSuccess@10px','bidirectional_error_px','normalized_error_diagonal','structure_precision','structure_anchor_recall','structure_repeatability']},
        no_overlap_FPR=avg('false_accept'),rejection_rate=avg('rejected'),
        normalized_error_has_no_unstated_meter_interpretation=True)

def summaries(rr):
    out={'all':aggregate(rr)}
    for k in ['size','split','scene','overlap_bin','angle_bin','profile']:
        for v in sorted(set(r[k] for r in rr),key=str):out[k+'/'+str(v)]=aggregate([r for r in rr if r[k]==v])
    for size in [512,640]:
        for k in ['scene','overlap_bin','angle_bin','profile']:
            for v in sorted(set(r[k] for r in rr),key=str):
                subset=[r for r in rr if r['size']==size and r[k]==v]
                if subset:out[f'size/{size}/{k}/{v}']=aggregate(subset)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('predictions');p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    preds={r['pair_id']:r for r in rows(a.predictions)};rr=[assess(r,preds.get(r['pair_id'],{})) for r in rows(a.manifest)];t=stamp()
    jsonl(a.output_dir/f'SCORES_{t}.jsonl',rr);save(a.output_dir/f'METRICS_{t}.json',summaries(rr));print(json.dumps(summaries(rr)['all']))
