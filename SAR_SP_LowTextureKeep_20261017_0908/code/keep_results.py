"""After selection: reuse old predictions, audit coverage, diagnose descriptor drift."""
from keep_train import *
from old_statistics import detailed
from collections import Counter,defaultdict

METHODS=['m0','b','t2','keep']
NAMES_LOCAL={'m0':'原始SuperPoint','b':'历史SAR结构增强','t2':'前轮描述子迁移','keep':'低纹理特征保持迁移'}


def finalize():
    status=json.loads((ROOT/'runs/formal'/f'status_{TAG}.json').read_text());assert status['state']=='completed' and status['step']==6000
    qualified=bool(status['qualified']);path=ROOT/'runs/formal'/f"{'best_qualified' if qualified else 'best_trained'}_{TAG}.ckpt";ck=torch.load(path,map_location='cpu',weights_only=False);step=ck['global_step'];assert step>0
    selected=dict(checkpoint=str(path),sha=sha(path),step=step,qualified=qualified,config=ck['config'],frozen_parameters_unchanged=ck['frozen_parameters_unchanged'],old_initialization_sha=sha(WEIGHTS['b']),old_descriptor_checkpoint=str(OLD/'runs/t2/best_trained_20261016_2234.ckpt'),old_descriptor_sha=sha(OLD/'runs/t2/best_trained_20261016_2234.ckpt'))
    initial=Model('b');initial_state=initial.state_dict();changes=[];changed=[]
    for key,value in ck['model_state'].items():
        before=initial_state[key];difference=(value-before).float();is_changed=not torch.equal(value,before)
        if is_changed:changed.append(key)
        changes.append(dict(name=key,elements=value.numel(),changed=is_changed,max_absolute_delta=float(difference.abs().max()),L2_delta=float(torch.linalg.vector_norm(difference))))
    assert set(changed)==set(ck['config']['trainable_names'])
    csvwrite(ROOT/f'PARAMETER_CHANGES_{stamp()}.csv',changes);selected['actual_changed_parameters']=changed
    write(ROOT/f'SELECTED_MODEL_{stamp()}.json',selected)
    old=[r for r in frozen_rows() if r['method'] in METHODS[:-1]];new=[json.loads(s) for s in (ROOT/f'predictions/formal_step{step}/scores_{TAG}.jsonl').read_text().splitlines()];assert len(old)==7200 and len(new)==2400
    allrows=old+new;lookup={(r['pair_id'],r['method'],r['protocol']):r for r in allrows};assert len(lookup)==9600
    for r in allrows:
        pid=r['pair_id'];mode=r['protocol'];ref=lookup[(pid,'b',mode)];assert r['common_budget']==ref['common_budget']
        if mode=='equal' or r['method']=='keep':assert [r['keypoints0'],r['keypoints1']]==[ref['keypoints0'],ref['keypoints1']]
        source=ROOT/f'predictions/formal_step{step}/{pid}_keep_{mode}_{TAG}.npz' if r['method']=='keep' else OLD/f'predictions/final/{pid}_{r["method"]}_{mode}_{OLDTAG}.npz'
        assert source.exists();r['frozen_prediction_path']=str(source)
    t=stamp();csvwrite(ROOT/f'FINAL_PER_PAIR_{t}.csv',allrows);write(ROOT/f'final_scores_{TAG}.json',allrows)
    table=[];strata=[];components=[];paired=[];roles=[]
    for m in METHODS:
        for mode in ['natural','equal']:
            rr=[r for r in allrows if r['method']==m and r['protocol']==mode];assert len(rr)==1200
            table.append(dict(method=m,protocol=mode,**detailed(rr)))
            for axis in ['size','scene','overlap_bin','angle_bin','profile']:
                for value in sorted({r[axis] for r in rr},key=str):
                    label=str(value)
                    if axis=='angle_bin':label=f'{dc.ANGLES[int(value)]:g}–{min(dc.ANGLES[int(value)+1],180):g} degrees synthetic2D'
                    elif axis=='overlap_bin':label='no_overlap' if value=='zero' else str(value).replace('_','–')+' percent'
                    strata.append(dict(method=m,protocol=mode,axis=axis,stratum=value,stratum_label=label,**detailed([r for r in rr if r[axis]==value])))
            for role in sorted({r['role'] for r in rr}):roles.append(dict(method=m,protocol=mode,role=role,**detailed([r for r in rr if r['role']==role])))
            for c in sorted({r['component'] for r in rr}):components.append(dict(method=m,protocol=mode,component=c,**detailed([r for r in rr if r['component']==c])))
    for mode in ['natural','equal']:
        for pid in sorted({r['pair_id'] for r in new}):
            a,b,c=[lookup[(pid,m,mode)] for m in ['b','t2','keep']]
            paired.append(dict(pair_id=pid,protocol=mode,scene=a['scene'],size=a['size'],component=a['component'],geometry_eligible=a['geometry_eligible'],structure_eligible=a['structure_global_eligible'],baseline_success=a['Success@5px'],old_descriptor_success=b['Success@5px'],keep_success=c['Success@5px'],old_lost=bool(a['Success@5px'] and not b['Success@5px']),old_loss_restored=bool(a['Success@5px'] and not b['Success@5px'] and c['Success@5px']),old_gain=bool(b['Success@5px'] and not a['Success@5px']),old_gain_retained=bool(b['Success@5px'] and not a['Success@5px'] and c['Success@5px']),new_loss_vs_initial=bool(a['Success@5px'] and not c['Success@5px']),new_gain_vs_initial=bool(c['Success@5px'] and not a['Success@5px']),static_matches_initial=a['correct_structure_matches'],static_matches_previous=b['correct_structure_matches'],static_matches_keep=c['correct_structure_matches'],keep_accepted=c['accepted'],keep_error=c['bidirectional_error_px'],keep_reason=c['reason']))
    csvwrite(ROOT/f'UNIFIED_COMPARISON_{t}.csv',table);write(ROOT/f'UNIFIED_COMPARISON_{t}.json',table);csvwrite(ROOT/f'STRATIFIED_{t}.csv',strata);csvwrite(ROOT/f'ROLE_SPECIFIC_{t}.csv',roles);csvwrite(ROOT/f'COMPONENT_RESULTS_{t}.csv',components);csvwrite(ROOT/f'PAIRED_RECOVERY_{t}.csv',paired)
    # Exploratory component intervals, not independent acquisition confidence guarantees.
    cis=[]
    for mode in ['natural','equal']:
        subset=[p for p in paired if p['protocol']==mode and p['geometry_eligible']];groups=defaultdict(list)
        for p in subset:groups[p['component']].append(int(p['keep_success'])-int(p['baseline_success']))
        keys=sorted(groups);sums=np.array([sum(groups[k]) for k in keys]);counts=np.array([len(groups[k]) for k in keys]);idx=np.random.default_rng(42).integers(0,len(keys),(10000,len(keys)));bs=sums[idx].sum(1)/counts[idx].sum(1)
        cis.append(dict(protocol=mode,components=len(keys),delta=float(sums.sum()/counts.sum()),lower=float(np.quantile(bs,.025)),upper=float(np.quantile(bs,.975)),caveat='Observed development Val and known source components, not proven independent original acquisitions'))
    write(ROOT/f'PAIRED_COMPONENT_INTERVALS_{t}.json',cis)
    coverage()
    return selected


def coverage():
    rows=records('train');folder=ROOT/'runs/formal';actual={};metrics={}
    for p in sorted(folder.glob('sampling_actual_*.jsonl')):
        for s in p.read_text().splitlines():q=json.loads(s);actual[q['step']]=q
    for p in sorted(folder.glob('metrics_*.jsonl')):
        for s in p.read_text().splitlines():q=json.loads(s);metrics[q['step']]=q
    assert set(actual)==set(metrics)==set(range(1,6001));plan=[json.loads(s) for s in (ROOT/f'plans/sampling_seed42_{TAG}.jsonl').read_text().splitlines()];calls=Counter();conditions=Counter();points=Counter();scenes=Counter()
    for p in plan:
        a=actual[p['step']];assert a['ids']==p['ids'] and a['roles']==p['roles'];m=metrics[p['step']];assert m['structure_pairs']==4 and m['keep_pairs']==4 and m['keep_positions']>0 and m['points']>0 and np.isfinite(m['preclip_gradient'])
        for i,role,info in zip(a['ids'],a['roles'],a['pairs']):
            calls[(i,role)]+=1;points[(i,role)]+=info['points_used'] if role=='desc' else info['positions'];r=rows[i];scenes[(r['source_scene'],role)]+=1
            for axis in ['width','overlap_bin','angle_bin','enhancement_profile','loss_role']:conditions[(role,axis,str(r[axis]))]+=1
    roles={}
    for role in ['desc','keep']:
        used={i for i,t in calls if t==role};roles[role]=dict(calls=sum(v for (i,t),v in calls.items() if t==role),relations=len(used),center_source_files=len({rows[i]['source_path'] for i in used}),visible_member_source_files=len({v for i in used for v in rows[i]['visible_source_members']}),known_components=len({rows[i]['source_component_id'] for i in used}),positions=sum(v for (i,t),v in points.items() if t==role),scene_calls={s:v for (s,t),v in scenes.items() if t==role})
    assert roles['desc']['calls']==roles['keep']['calls']==24000
    used={i for i,t in calls};trainvis={v for r in rows for v in r['visible_source_members']};valvis={v for r in records('val') for v in r['visible_source_members']};assert not trainvis&valvis
    mm=list(metrics.values());t=stamp();write(ROOT/f'ACTUAL_TRAINING_COVERAGE_{t}.json',dict(steps=6000,effective_pairs=48000,roles=roles,union_relations=len(used),source_member_train_val_overlap=0,zero_supervision_update_steps=0,zero_data_gradient_steps=sum(m['preclip_gradient']<=1e-12 for m in mm),nonzero_keep_gradient_steps=sum(m['gradient_keep']>1e-8 for m in mm),per_source_count_definition='Included in backward objective with legal supervision; per-sample nonzero derivatives not separately measured; aggregate descriptor/keep norms recorded each step',nonfinite_steps=0,mean_gradient_desc=float(np.mean([m['gradient_desc'] for m in mm])),mean_gradient_keep=float(np.mean([m['gradient_keep'] for m in mm])),mean_weighted_keep_gradient=float(np.mean([m['weighted_keep_gradient'] for m in mm])),mean_gradient_cosine=float(np.mean([m['gradient_cosine'] for m in mm])),negative_gradient_cosine_fraction=float(np.mean([m['gradient_cosine']<0 for m in mm])),gradient_clip_fraction=float(np.mean([m['preclip_gradient']>5 for m in mm])),peak_gpu_MiB=max(m['peak_memory_MiB'] for m in mm),first100_desc=float(np.mean([metrics[s]['desc'] for s in range(1,101)])),last100_desc=float(np.mean([metrics[s]['desc'] for s in range(5901,6001)])),sampler_change_is_part_of_method=True,previous_descriptor_calls=48000,current_descriptor_calls=24000))
    csvwrite(ROOT/f'SOURCE_COVERAGE_{t}.csv',[dict(pair_id=r['pair_id'],source=r['source_path'],component=r['source_component_id'],scene=r['source_scene'],size=r['width'],role=r['loss_role'],structure_calls=calls[(i,'desc')],keep_calls=calls[(i,'keep')],structure_positions=points[(i,'desc')],keep_positions=points[(i,'keep')]) for i,r in enumerate(rows)])
    csvwrite(ROOT/f'CONDITION_COVERAGE_{t}.csv',[dict(role=role,axis=axis,value=value,calls=n) for (role,axis,value),n in sorted(conditions.items())])


def descriptor_drift(selected):
    setup(42);teacher=Model('b').cuda();old=Model('b',OLD/'runs/t2/best_trained_20261016_2234.ckpt').cuda();student=Model('b',selected['checkpoint']).cuda();out=[]
    for n,row in enumerate(records('val')):
        positions=[];regions=[];target=row['source_scene']=='offshore_vessels_low_texture_sea' or not row['global_eligible']
        with np.load(row['gt_path'],allow_pickle=False) as z:
            for L in 'AB':
                mask=z['valid_'+L]&~z['ignore_'+L]&~z['supervision_ignore_'+L]&~z['ship_'+L];positions.append(uniform_positions(mask,z['structure_points_'+L]) if target else np.empty((0,2),np.float32))
                region=cv2.erode(mask.astype(np.uint8),np.ones((9,9),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0) if target else np.zeros_like(mask,np.uint8)
                for px,py in z['structure_points_'+L]:cv2.circle(region,(int(round(px)),int(round(py))),12,0,-1)
                regions.append(region.astype(bool))
        # Same actually detected top128 coordinates, evaluated without new network predictions.
        pid=row['pair_id'];paths=[OLD/f'predictions/final/{pid}_b_top128_{OLDTAG}.npz',OLD/f'predictions/final/{pid}_t2_top128_{OLDTAG}.npz',ROOT/f'predictions/formal_step{selected["step"]}/{pid}_keep_top128_{TAG}.npz']
        features=[]
        for path in paths:
            with np.load(path) as z:features.append({k:z[k].copy() for k in z.files})
        diffs=[[],[]];inside=[]
        for i in [0,1]:
            kp=features[0]['keypoints'+str(i)][0];assert all(np.array_equal(kp,f['keypoints'+str(i)][0]) for f in features[1:]);inside.extend(dc.bilinear_at(regions[i],kp).tolist())
            td=features[0]['descriptors'+str(i)][0]
            for j in [0,1]:diffs[j].extend(np.maximum(0,1-(td*features[j+1]['descriptors'+str(i)][0]).sum(1)).tolist())
        inside=np.array(inside,bool);kv=dict(detected_points=len(inside),detected_keep_region_points=int(inside.sum()),old_detected_drift=float(np.mean(diffs[0])) if len(inside) else None,keep_detected_drift=float(np.mean(diffs[1])) if len(inside) else None)
        for label,mask in [('inside',inside),('outside',~inside)]:
            for j,name in enumerate(['old','keep']):kv[f'{name}_detected_{label}_drift']=float(np.mean(np.array(diffs[j])[mask])) if mask.any() else None
        if not any(len(p) for p in positions):
            out.append(dict(pair_id=row['pair_id'],scene=row['source_scene'],size=row['width'],positions=0,old_drift=None,keep_drift=None,**kv));continue
        x,rs,_=shared_cache(row,teacher)
        with torch.no_grad():
            ds=[m.cached_maps(x,rs)[1] for m in [teacher,old,student]];vals=[[],[]]
            for i,p in enumerate(positions):
                if not len(p):continue
                xy=torch.tensor(p,device='cuda')[None];tt=sample_descriptors(xy,ds[0][i:i+1],8)[0].T
                for j in [0,1]:
                    dd=sample_descriptors(xy,ds[j+1][i:i+1],8)[0].T;vals[j].extend((1-(tt*dd).sum(1)).clamp_min(0).cpu().tolist())
        out.append(dict(pair_id=row['pair_id'],scene=row['source_scene'],size=row['width'],positions=sum(map(len,positions)),old_drift=float(np.mean(vals[0])),keep_drift=float(np.mean(vals[1])),**kv))
        if n%100==0:print(now(),'DRIFT_DIAGNOSTIC',n,1200,flush=True)
    t=stamp();csvwrite(ROOT/f'DESCRIPTOR_DRIFT_{t}.csv',out);write(ROOT/f'DESCRIPTOR_DRIFT_{t}.json',out)
    return out


if __name__=='__main__':s=finalize();descriptor_drift(s)
