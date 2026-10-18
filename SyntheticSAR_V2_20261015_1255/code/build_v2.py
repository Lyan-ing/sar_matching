"""Materialize three matched-geometry arms, with all attempted QA states retained."""
from dataset_v2 import *
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import argparse,time

def worker(job):
    r,spec,cfg,out=job;folder=Path(out)/r['split']/f'p{spec["index"]:06d}'
    folder.mkdir(parents=True,exist_ok=False)
    try:
        geom,values,meta=generate_arrays(r,spec,cfg)
        # Deduplicate identical masks within this relation without copying originals.
        mask_cache={}
        def png(name,a):
            p=folder/(name+'.png');assert not p.exists()
            ok=cv2.imwrite(str(p),a,[cv2.IMWRITE_PNG_COMPRESSION,2]);assert ok
            return str(p)
        def masks(name,arr):
            result=[]
            for i,a in enumerate(arr):
                a=a.astype('uint8')*255;key=(a.shape,digest_array(a))
                if key not in mask_cache:mask_cache[key]=png(name+f'_{i}',a)
                result.append(mask_cache[key])
            return result
        all_rows=[];common_fail=[f'{arm}:{why}' for arm,a in values.items() for why in a['qa_failures']]
        for arm,a in values.items():
            p=folder/f'{arm}_matches.npz';np.savez_compressed(p,points0=a['points0'],points1=a['points1'])
            error=float(np.linalg.norm(transform_points(geom['H'],a['points0'])-a['points1'],axis=1).max()) if len(a['points0']) else 0.
            assert error<.001
            assert not any(np.any(m&bad) for m,bad in zip(a['static'],a['ignore']))
            H=geom['H'];Hs=geom['Hs'];size=geom['size']
            row=dict(sample_id=f'p{spec["index"]:06d}',arm=arm,split=r['split'],source_dataset=r['dataset'],source_id=r['source_id'],source_path=r['image_path'],
                known_component=r['known_component'],construction_pool=r['construction_pool'],source_isolation_strength=r['isolation_strength'],
                source_scene_category=r['scene_category'],scene_label_status=r['scene_category_status'],source_units=r['data_units'],source_shape=meta['source_shape'],
                source_canonical_path=r['canonical_image_path'],scene_review_index=r.get('scene_review_index'),
                width=size,height=size,source_preupsampling=False,profile=spec['profile'],geometry_seed=spec['geometry_seed'],sampling_spec=spec,
                H_source_to_view=[h.tolist() for h in Hs],H_AB=H.tolist(),H_BA=np.linalg.inv(H).tolist(),
                source_window_corners=[transform_points(np.linalg.inv(h),[[0,0],[size-1,0],[size-1,size-1],[0,size-1]]).tolist() for h in Hs],
                image_rotation_2D_deg=geom['angle'],angle_bin_2D='negative_excluded' if spec['profile']=='none' else angle_bin(geom['angle']),real_radar_azimuth_delta_deg=None,
                overlap_fractions=geom['overlap'],valid_fractions=[float(x.mean()) for x in a['valid']],
                safe_static_fractions=[float(x.mean()) for x in a['safe_static']],structural_supervision_fractions=[float(x.mean()) for x in a['static']],
                ignore_fractions=[float(x.mean()) for x in a['ignore']],sea_fractions=meta['sea_fraction'],visible_ship_counts=meta['visible_ship_counts'],visible_ship_instances=meta['visible_ship_instances'],
                source_structural_candidates=meta['source_structural_candidates'],structural_anchor_count=len(a['points0']),spatial_support=a['support'],
                loss_role=a['role'],is_low_texture_or_no_structure=a['role'] in ['limited_structure','no_structural_positive'],
                geometric_label='synthetic_no_overlap' if spec['profile']=='none' else 'synthetic_overlap_not_necessarily_matchable',
                image0=png(arm+'_image0',a['images'][0]),image1=png(arm+'_image1',a['images'][1]),
                valid_masks=masks('valid',a['valid']),ship_masks=masks('ships_all',a['ship_masks']),
                static_masks=masks(arm+'_static',a['static']),ignore_masks=masks(arm+'_ignore',a['ignore']),visibility_masks=masks(arm+'_visibility',a['visibility']),
                safe_static_masks=masks('safe_static',a['safe_static']),structure_masks=masks('structure',a['structure_support']),calibration_masks=masks(arm+'_calibration',a['calibration']),
                correspondences=str(p),appearance=a['appearance'],dynamic_events=meta['dynamic_events'],local_scattering_events=a['local_events'],
                intensity=a['intensity'],qa_state='quarantined_common_arm_gate' if common_fail else 'usable',qa_failures=common_fail,
                qa_geometry_max_error=error,geometry_sampling_rejected=geom['rejected'],geometry_attempt=geom['trial'],
                all_annotated_vessels_including_partial_excluded=True,physical_multiview_GT=False,shared_original_speckle_fully_removed=False)
            all_rows.append(row)
        save(folder/f'RECORD_{stamp()}.json',dict(spec=spec,source_id=r['source_id'],meta=meta,rows=all_rows))
        return dict(ok=True,rows=all_rows)
    except Exception as ex:
        result=dict(ok=False,sample_id=f'p{spec["index"]:06d}',source_path=r['image_path'],split=r['split'],spec=spec,error=repr(ex))
        save(folder/f'FAILURE_{stamp()}.json',result)
        return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    cfg=json.loads(args.config.read_text());src=rows(cfg['source_pool']);seed=cfg['seed'];t=stamp()
    out=ROOT/(('smoke_' if args.smoke else 'fixed_')+t);out.mkdir(exist_ok=False)
    chosen=list(src)
    for split in ['train','val']:
        for scene,short in [('port_quay_dense_vessels','port'),('other_land_structures','land'),('nearshore_vessels','nearshore')]:
            pool=[r for r in src if r['split']==split and r['scene_category']==scene]
            pool=sorted(pool,key=lambda r:stable_seed(seed,r['image_path']))
            n=cfg['scene_supplement'][f'additional_{split}_{short}']
            chosen.extend(pool[i%len(pool)] for i in range(n))
    if args.smoke:
        chosen=[]
        for split in ['train','val']:
            for d in ['hrsid','background','ssdd']:
                for scene in sorted(set(r['scene_category'] for r in src if r['dataset']==d)):
                    selected=[r for r in src if r['split']==split and r['dataset']==d and r['scene_category']==scene]
                    chosen.extend(selected[:2])
    jobs=[]
    for i,r in enumerate(chosen):
        spec=make_spec(i,stable_seed(seed,f'fixed:{i}:{r["image_path"]}'),cfg)
        if args.smoke:
            spec.update(profile=['affine','projective','low','none'][i%4],angle_bin=i%5,requested_size=[256,384,512][i%3])
        jobs.append((r,spec,cfg,str(out)))
    jsonl(out/f'PLAN_{t}.jsonl',[dict(source_path=r['image_path'],split=r['split'],spec=spec) for r,spec,_,_ in jobs]);save(out/f'CONFIG_SNAPSHOT_{t}.json',cfg)
    results=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i,res in enumerate(ex.map(worker,jobs,chunksize=1),1):
            results.append(res)
            if i%100==0 or i==len(jobs):print(json.dumps(dict(at=now(),complete=i,total=len(jobs),seconds=round(time.monotonic()-start),geometry_failed=sum(not r['ok'] for r in results))),flush=True)
    t=stamp();rr=[row for res in results if res['ok'] for row in res['rows']]
    jsonl(out/f'ALL_GENERATED_{t}.jsonl',rr);jsonl(out/f'GEOMETRY_FAILURES_{t}.jsonl',[r for r in results if not r['ok']]);jsonl(out/f'INPUT_QUARANTINE_{t}.jsonl',[r for r in rr if r['qa_state']!='usable'])
    manifests={}
    for arm in cfg['arms']:
        for split in ['train','val']:
            for pool in ['core','ssdd','all']:
                selected=[r for r in rr if r['qa_state']=='usable' and r['arm']==arm and r['split']==split and (pool=='all' or (r['source_dataset']=='ssdd')==(pool=='ssdd'))]
                path=out/f'{arm}_{split}_{pool}_{t}.jsonl';jsonl(path,selected);manifests[f'{arm}/{split}/{pool}']=str(path)
                path=out/f'{arm}_{split}_{pool}_MATCH_{t}.jsonl';jsonl(path,[r for r in selected if r['loss_role']=='structural_positive']);manifests[f'{arm}/{split}/{pool}/match']=str(path)
    summary=dict(at=now(),config=str(args.config.resolve()),attempted_relations=len(jobs),generated_relations=len(rr)//3,geometry_failures=sum(not r['ok'] for r in results),
        qa_states=dict(Counter(r['qa_state'] for r in rr)),roles=dict(Counter(r['arm']+'/'+r['split']+'/'+r['loss_role'] for r in rr if r['qa_state']=='usable')),
        manifests=manifests,all_generated=str(out/f'ALL_GENERATED_{t}.jsonl'),input_quarantine=str(out/f'INPUT_QUARANTINE_{t}.jsonl'),
        elapsed_seconds=time.monotonic()-start,workers=args.workers,network_training_started=False)
    save(out/f'BUILD_SUMMARY_{t}.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='manifests'},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
