from data22 import *
from concurrent.futures import ProcessPoolExecutor
from collections import Counter,defaultdict
import argparse,time
import build_v2 as oldbuilder
from plan22 import iou
_CFG=None;_SOURCES=None;_LABELS=None;_NEW=None

def audit_old(r,row,f,newlab):
    Hs=[np.asarray(h) for h in row['H_source_to_view']];p=f['old'];xy=[transform_points(h,p) for h in Hs]
    valid=[read_gray(p)>0 for p in row['valid_masks']];safe=[read_gray(p)>0 for p in row['safe_static_masks']];ig=[read_gray(p)>0 for p in row['ignore_masks']]
    sm=[(read_gray(p)>0)&~i for p,i in zip(row['structure_masks'],ig)]
    kv=bilinear_at(valid[0],xy[0])&bilinear_at(valid[1],xy[1]);ks=bilinear_at(safe[0],xy[0])&bilinear_at(safe[1],xy[1]);kk=bilinear_at(sm[0],xy[0])&bilinear_at(sm[1],xy[1])
    pp,qq=thin(xy[0][kk].astype('float32'),xy[1][kk].astype('float32'));n=len(pp)
    assert n==row['spatial_support'][0]['count'],(row['sample_id'],n,row['spatial_support'])
    support=[spatial_support(a,row['width']) for a in [pp,qq]]
    flags=[]
    if n<16:flags.append('below16')
    if any(a['cells4x4']<4 for a in support):flags.append('grid_below4')
    if any(a['hull_fraction']<.05 for a in support):flags.append('hull_below5pct')
    if any(a['eigen_ratio']<.03 for a in support):flags.append('eigen_ratio_below_.03')
    no=row['profile']=='none';expected='no_overlap' if no else 'structural_positive' if not flags else 'limited_structure' if n>=2 else 'no_structural_positive'
    assert expected==row['loss_role'],(row['sample_id'],expected,row['loss_role'])
    if no:reason='analytic_no_overlap'
    elif not len(p):
        if r['scene_category'] in ['offshore_vessels_low_texture_sea','unclassified_background']:
            reason='semantic_veto_with_proposals' if len(f['proposal']) else 'semantic_veto_no_accepted_source_corner'
        else:reason='source_selector_no_corner_after_ship_land_texture_persistence'
    elif not kv.any():reason='source_corners_outside_common_crop'
    elif not ks.any():reason='view_ship_dynamic_or_validity_filter'
    elif not kk.any():reason='appearance_ignore_or_bilinear_support'
    elif flags:reason=flags[0]
    else:reason='strict_global_pass'
    return dict(sample_id=row['sample_id'],split=row['split'],pool=row['construction_pool'],source_path=r['image_path'],component=r['known_component'],scene=row['source_scene_category'],
        size=row['width'],angle_bin=row['angle_bin_2D'],profile=row['profile'],overlap=min(row['overlap_fractions']),old_role=row['loss_role'],
        source_old_corners=len(p),source_proposal_corners=len(f['proposal']),common_valid_before_view_ship=int(kv.sum()),common_after_view_ship_dynamic=int(ks.sum()),
        accepted_before_global=n,global_fail_flags=flags,primary_reason=reason,new_same_image_count=len(newlab['points0']),new_same_image_role=newlab['loss_role'])

def reuse_worker(row):
    try:
        r=_SOURCES[row['source_path']];f=features(r);a=labels(r,row,_CFG,f=f)
        out=materialize_labels(r,row,_CFG,'reused_V21',_LABELS/row['sample_id']/row['arm'],a=a)
        audit=audit_old(r,row,f,a) if row['arm']=='V2-R' else None
        return dict(ok=True,row=out,audit=audit)
    except Exception as ex:return dict(ok=False,sample_id=row['sample_id'],arm=row['arm'],error=repr(ex))

def new_worker(item):
    r=_SOURCES[item['source_path']];spec=item['spec'];saved=v21.sample_geometry;v21.sample_geometry=sample_geometry22
    try:result=oldbuilder.worker((r,spec,_CFG,str(_NEW)))
    finally:v21.sample_geometry=saved
    if not result['ok']:return result
    out=[]
    for row in result['rows']:
        row.update(roi_center=item['center'],roi_plan_id=spec['roi_id'])
        labelled=materialize_labels(r,row,_CFG,'new_ROI',_LABELS/row['sample_id']/row['arm'])
        out.append(labelled)
    return dict(ok=True,rows=out)

def main():
    global _CFG,_SOURCES,_LABELS,_NEW
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    _CFG=json.loads(a.config.read_text());_SOURCES={r['image_path']:r for r in sources()};t=stamp();run=ROOT/(('smoke_' if a.smoke else 'release_build_')+t);run.mkdir(exist_ok=False);_LABELS=run/'labels';_NEW=run/'new_views';_LABELS.mkdir();_NEW.mkdir()
    previous=old_rows('V2-R')+old_rows('V2-S');plan=rows(a.plan)
    if a.smoke:
        selected=[];seen=set()
        for r in previous:
            key=(r['arm'],r['split'],r['loss_role'],r['source_scene_category'])
            if key not in seen:selected.append(r);seen.add(key)
        previous=selected;plan=plan[:16]
    start=time.monotonic();results=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,res in enumerate(ex.map(reuse_worker,previous,chunksize=2),1):
            results.append(res)
            if i%1000==0 or i==len(previous):print(json.dumps(dict(stage='relabel',done=i,total=len(previous),failed=sum(not r['ok'] for r in results),seconds=round(time.monotonic()-start))),flush=True)
    t1=stamp();jsonl(run/f'REUSE_RESULTS_{t1}.jsonl',results);jsonl(run/f'V21_ATTRIBUTION_{t1}.jsonl',[r['audit'] for r in results if r.get('audit')]);assert all(r['ok'] for r in results),[r for r in results if not r['ok']][:3]
    rr=[r['row'] for r in results];newresults=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,res in enumerate(ex.map(new_worker,plan,chunksize=1),1):
            newresults.append(res)
            if i%200==0 or i==len(plan):print(json.dumps(dict(stage='new_ROI',done=i,total=len(plan),failed=sum(not r['ok'] for r in newresults),seconds=round(time.monotonic()-start))),flush=True)
    windows=defaultdict(list)
    for r in old_rows():windows[r['source_path']].extend(r['source_window_corners'])
    newrows=[]
    for result in newresults:
        if not result['ok']:continue
        row=result['rows'][0];poly=np.asarray(row['source_window_corners'][0]);center=poly.mean(0);side=np.sqrt(cv2.contourArea(poly.astype('float32')));dup=[]
        for oldpoly in windows[row['source_path']]:
            oldpoly=np.asarray(oldpoly);oldside=np.sqrt(cv2.contourArea(oldpoly.astype('float32')))
            if iou(poly,oldpoly)>.7 or np.linalg.norm(center-oldpoly.mean(0))<.25*min(side,oldside):dup.append(True);break
        if dup:
            for row in result['rows']:row.update(qa_state='quarantined_duplicate_native_ROI',duplicate_reason='actual native FOV too close to existing/frozen earlier new FOV; no recrop')
        elif all(row['qa_state']=='usable' for row in result['rows']):windows[row['source_path']].extend(row['source_window_corners'])
        newrows.extend(result['rows'])
    rr+=newrows;t2=stamp();jsonl(run/f'ALL_V22_{t2}.jsonl',rr);jsonl(run/f'NEW_FAILURES_{t2}.jsonl',[r for r in newresults if not r['ok']]);jsonl(run/f'QUARANTINE_{t2}.jsonl',[r for r in rr if r['qa_state']!='usable'])
    summary=dict(at=now(),config=str(a.config.resolve()),plan=str(a.plan.resolve()),all_rows=str(run/f'ALL_V22_{t2}.jsonl'),attribution=str(run/f'V21_ATTRIBUTION_{t1}.jsonl'),
        reused_pairs_per_arm=len(previous)//2,attempted_new_relations=len(plan),new_generated_relations=len(newrows)//2,new_geometry_failed=sum(not r['ok'] for r in newresults),
        states=dict(Counter(r['qa_state'] for r in rr)),roles=dict(Counter(r['arm']+'/'+r['split']+'/'+r['loss_role'] for r in rr if r['qa_state']=='usable')),
        elapsed_seconds=time.monotonic()-start,no_new_downloads=True,network_training_started=False)
    save(run/f'BUILD_SUMMARY_{t2}.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
