"""Freeze natural, role and optional ship views; report attribution without model results."""
from data22 import *
from plan22 import iou
from collections import Counter,defaultdict
import argparse,csv
def csvwrite(path,rr):
    with Path(path).open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]) if rr else []);w.writeheader();w.writerows(rr)
def main():
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);a=p.parse_args();build=json.loads(a.summary.read_text());cfg=json.loads(Path(build['config']).read_text());allr=rows(build['all_rows']);rr=[r for r in allr if r['qa_state']=='usable'];t=stamp()
    byid=defaultdict(dict)
    for r in rr:byid[r['relation_id']][r['arm']]=r
    families=defaultdict(list)
    for relation,group in byid.items():
        r=group['V2-R'];poly=np.asarray(r['source_window_corners'][0]);family=None
        for fid,oldpoly in families[r['source_path']]:
            if iou(poly,oldpoly)>.7:family=fid;break
        if family is None:
            family='roiFamily_'+str(stable_seed(22,r['source_path']+str(len(families[r['source_path']]))));families[r['source_path']].append((family,poly))
        for r in group.values():r['roi_family_id']=family
    manifests={};stats=[]
    for arm in cfg['arms']:
        for split in ['train','val']:
            for pool in ['core','ssdd','all']:
                selected=[r for r in rr if r['arm']==arm and r['split']==split and (pool=='all' or (r['source_dataset']=='ssdd')==(pool=='ssdd'))]
                source_first={}
                for r in sorted(selected,key=lambda r:(r['origin']!='reused_V21',r['sample_id'])):source_first.setdefault(r['source_path'],r)
                for name,data in [('all',selected),('natural',list(source_first.values()))]+[(role,[r for r in selected if r['loss_role']==role]) for role in ['structural_positive','limited_structure','no_structural_positive','no_overlap']]+[
                    ('local_descriptor',[r for r in selected if r['local_descriptor_eligible']]),
                    ('local_detection_only',[r for r in selected if 0<r['structural_anchor_count']<4]),
                    ('calibration_evidence',[r for r in selected if r['loss_role']=='no_structural_positive' and sum(r['calibration_pixels'])>=64]),
                    ('optional_ship_aux',[r for r in selected if r['ship_descriptor_eligible']]),
                    ('new_ROIs',[r for r in selected if r['origin']=='new_ROI'])]:
                    path=ROOT/f'MANIFEST_{arm}_{split}_{pool}_{name}_{t}.jsonl';jsonl(path,data);manifests[f'{arm}/{split}/{pool}/{name}']=str(path)
                    stats.append(dict(arm=arm,split=split,pool=pool,role=name,pairs=len(data),views=2*len(data),sources=len({r['source_path'] for r in data}),components=len({r['known_component'] for r in data}),
                        roi_families=len({(r['source_path'],r['roi_family_id']) for r in data}),ground_points=sum(r['structural_anchor_count'] for r in data),ship_aux_points=sum(r['ship_aux_count'] for r in data)))
    csvwrite(ROOT/f'ROLE_SOURCE_COMPONENTS_{t}.csv',stats)
    rrows=[r for r in rr if r['arm']=='V2-R'];groups=Counter((r['split'],r['source_scene_category'],r['angle_bin_2D'],'negative' if r['profile']=='none' else 'low' if r['profile']=='low' else 'normal',r['width'],r['loss_role'],r['local_descriptor_eligible']) for r in rrows)
    csvwrite(ROOT/f'SCENE_ANGLE_OVERLAP_SIZE_SUPPORT_{t}.csv',[dict(split=k[0],scene=k[1],angle_2D=k[2],overlap=k[3],size=k[4],role=k[5],descriptor_eligible=k[6],pairs=n) for k,n in sorted(groups.items())])
    attrib=rows(build['attribution']);reason=Counter(r['primary_reason'] for r in attrib);csvwrite(ROOT/f'V21_PRIMARY_CAUSES_{t}.csv',[dict(reason=k,pairs=n) for k,n in reason.items()])
    flag=Counter(f for r in attrib if r['profile']!='none' for f in r['global_fail_flags']);csvwrite(ROOT/f'V21_NONEXCLUSIVE_GLOBAL_GATE_FAILURES_{t}.csv',[dict(gate=k,pairs=n) for k,n in flag.items()])
    sensitivity=[]
    for threshold in [1,4,8,12,16]:
        for split in ['train','val']:
            for pool in ['core','ssdd','all']:
                old=[r for r in attrib if r['split']==split and (pool=='all' or ('ssdd' in r['pool'])==(pool=='ssdd')) and r['profile']!='none' and r['accepted_before_global']>=threshold]
                same=[r for r in attrib if r['split']==split and (pool=='all' or ('ssdd' in r['pool'])==(pool=='ssdd')) and r['profile']!='none' and r['new_same_image_count']>=threshold]
                final=[r for r in rrows if r['split']==split and (pool=='all' or (r['source_dataset']=='ssdd')==(pool=='ssdd')) and r['structural_anchor_count']>=threshold]
                for stage,data in [('V21_same_selector_count_only',old),('V22_same_images_and_area_review',same),('V22_with_new_ROIs',final)]:
                    sensitivity.append(dict(stage=stage,threshold=threshold,split=split,pool=pool,pairs=len(data),sources=len({r.get('source_path') for r in data}),components=len({r.get('component',r.get('known_component')) for r in data})))
    csvwrite(ROOT/f'LOCAL_THRESHOLD_SENSITIVITY_{t}.csv',sensitivity)
    contribution=[]
    for path in sorted({r['source_path'] for r in rrows}):
        data=[r for r in rrows if r['source_path']==path];r=data[0]
        contribution.append(dict(source_path=path,source_id=r['source_id'],component=r['known_component'],split=r['split'],scene=r['source_scene_category'],pairs=len(data),new_ROIs=sum(r['origin']=='new_ROI' for r in data),
            full=sum(r['global_eligible'] for r in data),limited=sum(r['loss_role']=='limited_structure' for r in data),local_descriptor=sum(r['local_descriptor_eligible'] for r in data),ship_aux=sum(r['ship_descriptor_eligible'] for r in data),
            roi_families=len(families[path])))
    csvwrite(ROOT/f'PER_SOURCE_CONTRIBUTION_{t}.csv',contribution)
    # Explicit natural distribution includes unknown and hard samples, with no success-based selection.
    new_roles=Counter(r['loss_role'] for r in rrows);new_local=sum(r['local_descriptor_eligible'] for r in rrows)
    old_ge4=[r for r in attrib if r['profile']!='none' and r['accepted_before_global']>=4]
    same_ge4=[r for r in attrib if r['profile']!='none' and r['new_same_image_count']>=4]
    old_ground_sources={r['source_path'] for r in old_rows() if r['split']=='train' and r['source_dataset']!='ssdd' and r['loss_role']=='structural_positive'}
    new_ground_sources={r['source_path'] for r in rrows if r['split']=='train' and r['source_dataset']!='ssdd' and r['local_descriptor_eligible']}
    summary=dict(at=now(),build_summary=str(a.summary.resolve()),config=build['config'],all_usable_pairs_records=len(rr),R_pairs=len(rrows),R_views=2*len(rrows),R_roles=dict(new_roles),
        R_local_descriptor_pairs=new_local,R_optional_ship_descriptor_pairs=sum(r['ship_descriptor_eligible'] for r in rrows),R_sources=len({r['source_path'] for r in rrows}),
        old_core_training_full_sources=len(old_ground_sources),new_core_training_local_sources=len(new_ground_sources),new_source_coverage_increase=len(new_ground_sources-old_ground_sources),
        attribution=dict(V21_strict=1264,V21_same_pixels_count_ge4=len(old_ge4),V22_same_pixels_ge4=len(same_ge4),V22_final_ge4=new_local,
            gain_training_gate=len(old_ge4)-1264,gain_area_evidence_same_pixels=len(same_ge4)-len(old_ge4),gain_new_native_ROI_local_relations=new_local-len(same_ge4)),
        primary_causes=dict(reason),manifests=manifests,role_stats=stats,all_rows_with_roi_family=str(ROOT/f'ALL_WITH_ROI_FAMILY_{t}.jsonl'),
        V21_frozen_entrypoint=str(OLD/'DATASET_ENTRYPOINTS_V2_20261015_1326.json'))
    jsonl(summary['all_rows_with_roi_family'],rr);save(ROOT/f'MANIFEST_SUMMARY_{t}.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['manifests','role_stats']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
