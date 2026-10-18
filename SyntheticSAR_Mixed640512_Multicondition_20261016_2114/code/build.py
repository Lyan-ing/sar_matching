from generator import *
from concurrent.futures import ProcessPoolExecutor
import argparse, traceback

def freeze():
    t=stamp();audit=rows(latest('meta/SOURCE_AUDIT_*.jsonl'));domains=rows(latest('meta/SOURCE_DOMAINS_*.jsonl'))
    cfg=dict(created=now(),version='Mixed640512-1',master_seed=202610162114,total=6000,train=4800,val=1200,
        source_audit=str(latest('meta/SOURCE_AUDIT_*.jsonl')),source_domains=str(latest('meta/SOURCE_DOMAINS_*.jsonl')),
        source_pool=CFG21['source_pool'],old_entry=str(OLD/'DATASET_ENTRYPOINTS_V22_20261015_1434.json'),
        geometry=dict(sizes=[640,512],attempts_per_size=80,min_valid_fraction=.98,perspective_fraction=.0006,
            scale_range=[.96,1.],shear_degrees=[-1,1],kinds=dict(rigid=.4,similarity=.4,projective=.2),
            overlap_counts=[300,900,1200,1200,1200,1200],angle_percent=[15,15,20,20,20,10],
            relative_angle_definition='polar rotation of H_AB Jacobian at view-A center; absolute [0,180]',
            pixel_convention='OpenCV integer pixel centers, x right/y down; global published-parent pixel coordinate',
            source_preupsampling=False,minimum_source_domain_gap_pixels=4),
        appearance=dict(profile_percent=[20,20,35,15,10],radiometry_light=dict(gamma=[.96,1.04],contrast=[.97,1.03],offset=.004),
            radiometry_moderate=dict(gamma=[.92,1.08],contrast=[.95,1.05],offset=.007),
            local_scatter_max_area=.025,original_speckle_remaining=True,domain='display_DN_not_calibrated_SAR_intensity'),
        scene_percent=[35,30,25,10],max_anchor_pairs_train=16,max_anchor_pairs_val=24,max_raw_member_pair_uses=96,
        max_source_candidates_per_job=120,acceptance=dict(min_std=.5,max_saturated=.30,max_zero=.50),
        structure=dict(min_local_descriptor_points=4,global_min_points=16,global_min_4x4_cells=4,global_min_hull=.05,global_min_eigen_ratio=.03,
            selector='unchanged V2.2 persistent original-source sigma2/4 corners, ships/sea/unknown excluded',ship_aux_default=False),
        scoring=dict(thresholds_px=[3,5,10],match_tolerance_px=3,normalized_denominator='sqrt(width^2+height^2)',
            geometry_success_population='all positive overlaps with >=4 valid grid checks; not a structural or real physical success claim',
            structure_success_population='unchanged strict V2.2 global_eligible subset',minimum_checks=4),
        no_training=True,no_model_result_selection=True,old_protocol_unchanged=True,
        split_policy='retain original known components; exclude any byte/pixel/component conflict; original Test excluded',
        source_assembly='only existing same-dataset same-parent native tiles joined by >=99.5% exact overlap pixels; missing or inconsistent pixels invalid; seams ignored')
    jobs=[]
    for split,total in [('train',4800),('val',1200)]:
        rng=np.random.default_rng(stable_seed(cfg['master_seed'],split))
        overlaps=np.concatenate([np.repeat(o,int(c*total/6000)) for o,c in zip(OVERLAPS,cfg['geometry']['overlap_counts'])]).tolist();rng.shuffle(overlaps)
        angles=np.concatenate([np.repeat(i,total*p//100) for i,p in enumerate(cfg['geometry']['angle_percent'])]).tolist();rng.shuffle(angles)
        # A rotated equal square cannot realize >90% mutual field at 30-60 degrees.
        # Preserve exact angle and overlap marginals, changing only impossible joint cells.
        for i,(o,a) in enumerate(zip(overlaps,angles)):
            if o=='90_100' and a in [1,2]:
                j=next(j for j in rng.permutation(total) if overlaps[j]!='90_100' and angles[j] not in [1,2])
                angles[i],angles[j]=angles[j],angles[i]
        scenes=np.concatenate([np.repeat(s,total*p//100) for s,p in zip(SCENES,cfg['scene_percent'])]).tolist();rng.shuffle(scenes)
        # Background land contexts are mostly isolated 800-pixel tiles. Do not allocate
        # impossible strictly disjoint 512 fields to them merely for a full factorial table.
        for i,(o,s) in enumerate(zip(overlaps,scenes)):
            if o=='zero' and s==SCENES[3]:
                j=next(j for j in rng.permutation(total) if overlaps[j]!='zero' and scenes[j]!=SCENES[3])
                scenes[i],scenes[j]=scenes[j],scenes[i]
        profiles=np.concatenate([np.repeat('P'+str(i),total*p//100) for i,p in enumerate(cfg['appearance']['profile_percent'])]).tolist();rng.shuffle(profiles)
        kinds=rng.choice(['rigid','similarity','projective'],total,p=[.4,.4,.2])
        for i in range(total):
            seed=stable_seed(cfg['master_seed'],f'{split}:{i}')
            jobs.append(dict(pair_id=f'MIX_{split}_{i:05d}',split=split,source_scene=scenes[i],overlap_bin=overlaps[i],angle_bin=int(angles[i]),
                enhancement_profile=profiles[i],geometry_kind=str(kinds[i]),seed=seed,geometry_seed=stable_seed(seed,'geometry')))
    # Test-component isolation from the existing complete historical inventory.
    inv=rows(BASE/'DualSAR_DatasetPrep_20261014_2131/SOURCE_INVENTORY_20261014_2142.jsonl')
    test={r['known_component'] for r in inv if r['split']=='test'}
    assert not test&{d['source_component_id'] for d in domains}
    path=ROOT/f'meta/GENERATION_CONFIG_{t}.json';save(path,cfg);jsonl(ROOT/f'meta/FROZEN_JOB_PLAN_{t}.jsonl',jobs)
    protected=[OLD/'DATASET_ENTRYPOINTS_V22_20261015_1434.json',OLD/'CONFIG_V22_20261015_1404.json',Path(CFG21['source_pool']),
        BASE/'MaritimeSARRegDataset/sarreg/geometry.py',BASE/'SyntheticSAR_V2_20261015_1255/code/dataset_v2.py',
        BASE/'SAR_SP_LG_V22_20261015_1525/code/graded_loss.py',BASE/'SAR_SP_LG_V22_20261015_1525/code/stream22.py']
    protected += [Path(p) for p in json.loads((OLD/'DATASET_ENTRYPOINTS_V22_20261015_1434.json').read_text())['manifests'].values() if isinstance(p,str) and Path(p).is_file()][:8]
    save(ROOT/f'meta/PREREGISTRATION_{t}.json',dict(created=now(),config=str(path),config_hash=sha(path),plan=str(ROOT/f'meta/FROZEN_JOB_PLAN_{t}.jsonl'),
        plan_hash=sha(ROOT/f'meta/FROZEN_JOB_PLAN_{t}.jsonl'),protected_files={str(p):sha(p) for p in protected},
        scene_quotas_are_source_contexts_not_verified_crop_semantics=True,
        impossible_cells=['90-100% overlap × 15-30/30-60deg rotation omitted by square footprint feasibility','zero overlap × isolated land context not requested'],
        before_any_formal_generation=True))
    print(json.dumps(dict(config=str(path),jobs=len(jobs),created=now())),flush=True)

def store(d,spec,cfg,config_hash,folder=None):
    a=generate(d,spec,cfg);g=a['g'];n=g['size'];pair=spec['pair_id'];split=spec['split'];base=Path(folder) if folder else ROOT
    dest=base/'pairs'/split/pair
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    imagepaths=[]
    for i,letter in enumerate('AB'):
        p=dest/f'{pair}_{letter}.png';png(p,a['images'][i]);imagepaths.append(str(p))
    arrays=dict(T_A=g['T'][0],T_B=g['T'][1],H_AB=g['H'],H_BA=np.linalg.inv(g['H']),no_overlap=np.bool_(spec['overlap_bin']=='zero'),
        image_size=np.array([n,n]),source_origin_xy=np.array(d['bounds_xyxy'][:2]),
        check_points_A=a['checks'][0],check_points_B=a['checks'][1],structure_points_A=a['points'][0],structure_points_B=a['points'][1],
        points0=a['points'][0].astype('float32'),points1=a['points'][1].astype('float32'))
    for i,L in enumerate('AB'):
        for name,value in [('valid',g['valid']),('overlap',g['overlap']),('ignore',a['ignore']),('appearance_changed',a['appearance_masks']),('ship',a['ship']),('safe_static',a['safe']),('static',a['static']),('calibration',a['calibration']),('supervision_ignore',a['unknown'])]:arrays[name+'_'+L]=value[i]
    gt=dest/f'{pair}_GT.npz';np.savez_compressed(gt,**arrays)
    role={'sufficient_structure':'structural_positive','limited_structure':'limited_structure','weak_or_unknown_structure':'no_structural_positive','no_overlap':'no_overlap'}[a['level']]
    used=[];seen_owners=set()
    for A,v in zip(g['local'],g['valid']):
        own=cv2.warpPerspective(a['source']['owner'].astype('float32'),A,(n,n),flags=cv2.INTER_NEAREST,borderValue=-1)
        seen_owners.update(map(int,np.unique(own[v])))
    for i,p in enumerate(d['members']):
        if i in seen_owners:used.append(p)
    support=[spatial_support(x,n) for x in a['points']]
    row=dict(pair_id=pair,sample_id=pair,split=split,image_A=imagepaths[0],image_B=imagepaths[1],image0=imagepaths[0],image1=imagepaths[1],gt_path=str(gt),
        width=n,height=n,size_A=[n,n],size_B=[n,n],source_id=d['source_id'],source_path=d['anchor'],source_component_id=d['source_component_id'],known_component=d['source_component_id'],
        source_hash=d['source_hash'],anchor_raw_hash=sha(d['anchor']),source_scene=d['source_scene'],source_scene_category=d['source_scene'],
        source_context_status=d['scene_status'],verified_visible_scene='unknown',visible_scene_reason='source context label retained; no per-crop human sea/land/port segmentation confirmation',
        source_kind=d['source_kind'],source_members=d['members'],visible_source_members=used,source_domain_bounds_xyxy=d['bounds_xyxy'],source_shape=[d['height'],d['width']],
        source_raw_acquisition_identity_known=d['raw_acquisition_identity_known'],
        rotation_actual_deg=g['angle'],image_rotation_2D_deg=g['angle'],angle_bin=spec['angle_bin'],rotation_is_synthetic_2D_not_radar_azimuth=True,
        overlap_ratio=min(g['fractions']),overlap_A=g['fractions'][0],overlap_B=g['fractions'][1],overlap_fractions=g['fractions'],overlap_bin=spec['overlap_bin'],
        valid_fraction_A=float(g['valid'][0].mean()),valid_fraction_B=float(g['valid'][1].mean()),enhancement_profile=spec['enhancement_profile'],
        enhancement_ops_A=a['appearance'][0]['ordered_ops'],enhancement_ops_B=a['appearance'][1]['ordered_ops'],enhancement_params_A=a['appearance'][0],enhancement_params_B=a['appearance'][1],
        seed=spec['seed'],geometry_seed=spec['geometry_seed'],structure_supervision_level=a['level'],loss_role=role,
        structural_anchor_count=len(a['points'][0]),check_point_count=len(a['checks'][0]),spatial_support=support,geometry_gt_available=True,
        global_eligible=a['level']=='sufficient_structure',local_descriptor_eligible=len(a['points'][0])>=4,local_detector_eligible=len(a['points'][0])>0,
        local_weight=1. if a['level']=='sufficient_structure' else min(1.,len(a['points'][0])/16),reason_if_512=g['reason_if_512'],
        source_preupsampling=False,generation_config_hash=config_hash,sampling_spec=spec,geometry_parameters={k:g[k] for k in ['geometry_kind','nominal_theta','scales','shear','polygon_common_source_area','rejected','trial']},
        H_source_to_view=[x.tolist() for x in g['T']],H_AB=g['H'].tolist(),H_BA=np.linalg.inv(g['H']).tolist(),
        intensity=a['intensity'],source_conflict_pixels=a['source']['conflict_pixels'],output_pixel_hashes=[ahash(x) for x in a['images']],
        image_file_hashes=[sha(p) for p in imagepaths],gt_file_hash=sha(gt),qa_state='usable',physical_multiview_GT=False,ship_aux_enabled_by_default=False)
    save(dest/f'{pair}_RECORD_{TAG}.json',row)
    return row

def attempt(job):
    d,spec,cfg,h,folder=job
    try:return dict(ok=True,row=store(d,spec,cfg,h,folder))
    except Exception as e:return dict(ok=False,pair_id=spec['pair_id'],source=d['anchor'],error=repr(e),trace=traceback.format_exc() if not isinstance(e,ValueError) else None)

def smoke(cfgpath):
    cfg=json.loads(Path(cfgpath).read_text());h=sha(cfgpath);ds=rows(cfg['source_domains']);jobs=rows(latest('meta/FROZEN_JOB_PLAN_*.jsonl'))
    folder=ROOT/f'smoke_{stamp()}';folder.mkdir(exist_ok=False);out=[];fail=[];seen=set()
    # Cover all profiles/overlap bins/scenes/splits, then explicitly test 512 via
    # native isolated 800 sources which cannot support their 640 requested fields.
    selections=[]
    for j in jobs:
        keys=[('profile',j['enhancement_profile']),('overlap',j['overlap_bin']),('scene',j['source_scene']),('split',j['split']),('angle',j['angle_bin'])]
        if any(k not in seen for k in keys):selections.append(j);seen.update(keys)
    for i,j in enumerate(selections):
        candidates=[d for d in ds if d['split']==j['split'] and d['source_scene']==j['source_scene']]
        candidates.sort(key=lambda d:(-(d['width']*d['height']),stable_seed(j['seed'],d['anchor'])))
        for d in candidates[:30]:
            r=attempt((d,j,cfg,h,str(folder)))
            if r['ok']:out.append(r['row']);break
            if r['trace']:raise RuntimeError(r['trace'])
            fail.append(r)
        else:raise RuntimeError(('smoke no source',j,fail[-1]))
        print(json.dumps(dict(stage='smoke',done=i+1,total=len(selections),size=out[-1]['width'])),flush=True)
    # Same native view/geometry compared under every P0..P4, outside release count.
    d=next(d for d in ds if d['split']=='train' and d['source_scene']==SCENES[2] and d['width']>=1400 and d['height']>=1400)
    base=dict(jobs[0],overlap_bin='50_70',angle_bin=1,geometry_kind='rigid',seed=901,geometry_seed=902)
    for p in range(5):
        j=dict(base,pair_id='PROFILE_CONTROL_P'+str(p),enhancement_profile='P'+str(p));out.append(store(d,j,cfg,h,folder))
    # Geometrical primitives are literally the old imported functions; also exercise
    # old composed generator as a compatibility reference, independent of new sampler.
    rng=np.random.default_rng(812);errors=[]
    for _ in range(64):
        n=int(rng.choice([512,640]));A=view_matrix((700,800),n,rng.uniform(-180,180),rng.uniform(.96,1.),rng.uniform(-1,1),.0006,rng)
        p=rng.uniform(10,400,(50,2));q=transform_points(A,p);errors.append(float(np.linalg.norm(transform_points(np.linalg.inv(A),q)-p,axis=1).max()))
    legacy=v21._apply_augment
    im=domain_arrays(d)['image'][:640,:640];params=dict(enabled=True,gamma_min=.96,gamma_max=1.04,contrast_min=.97,contrast_max=1.03,brightness_max=.004,speckle_sigma=0,blur_probability=0)
    ref,rp=legacy(im,params,np.random.default_rng(99));actual,_,_,ap=appearance(im,np.ones(im.shape,bool),'P1',99,cfg)
    assert np.array_equal(ref,actual) and rp==ap['radiometry']
    jsonl(folder/f'SMOKE_MANIFEST_{stamp()}.jsonl',out);jsonl(folder/f'SMOKE_REJECTIONS_{stamp()}.jsonl',fail)
    save(folder/f'CONSISTENCY_{stamp()}.json',dict(passed=True,old_geometry_primitives_reused=True,max_roundtrip_error=max(errors),
        unmodified_legacy_radiometry_pixel_equality=True,profile_control_same_source_and_geometry=True,smoke_pairs=len(out)))
    print(json.dumps(dict(smoke_dir=str(folder),pairs=len(out),sizes=dict(Counter(r['width'] for r in out)))),flush=True)

def build(cfgpath,workers):
    cfg=json.loads(Path(cfgpath).read_text());h=sha(cfgpath);jobs=rows(latest('meta/FROZEN_JOB_PLAN_*.jsonl'));ds=rows(cfg['source_domains']);lookup={d['anchor']:d for d in ds}
    anchor=Counter();raw=Counter();results=[];rejections=[];finished=set();start=time.monotonic()
    for directory in sorted((ROOT/'pairs').glob('*/*')):
        records=list(directory.glob('*_RECORD_*.json'))
        complete=False
        if len(records)==1:
            try:r=json.loads(records[0].read_text());complete=True
            except json.JSONDecodeError:pass
        if not complete:
            quarantine=ROOT/'interrupted';quarantine.mkdir(exist_ok=True);target=quarantine/(directory.name+'_'+stamp())
            if target.exists():raise FileExistsError(target)
            directory.rename(target);continue
        assert r['generation_config_hash']==h
        assert all(sha(p)==v for p,v in zip([r['image_A'],r['image_B']],r['image_file_hashes'])) and sha(r['gt_path'])==r['gt_file_hash'],'Previously completed data changed; refusing overwrite'
        results.append(r);finished.add(r['pair_id']);anchor[r['source_path']]+=1;raw.update(r['visible_source_members'])
    jobs=[j for j in jobs if j['pair_id'] not in finished]
    if not jobs:
        print(json.dumps(dict(already_complete=len(results),previous_outputs_verified=True,new_files_written=0)),flush=True);return
    failed=[];batchsize=workers
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for offset in range(0,len(jobs),batchsize):
            pending=jobs[offset:offset+batchsize];tried=defaultdict(set)
            while pending:
                tasks=[];submitted=[]
                for spec in pending:
                    cap=cfg['max_anchor_pairs_'+spec['split']]
                    candidates=[d for d in ds if d['split']==spec['split'] and d['source_scene']==spec['source_scene'] and d['anchor'] not in tried[spec['pair_id']] and anchor[d['anchor']]<cap and all(raw[p]<cfg['max_raw_member_pair_uses'] for p in d['members'])]
                    # Same predeclared source/geometry/pixel quality rules for all roles.
                    candidates.sort(key=lambda d:(anchor[d['anchor']],stable_seed(spec['seed'],d['anchor'])))
                    if not candidates or len(tried[spec['pair_id']])>=cfg['max_source_candidates_per_job']:
                        failed.append(dict(spec=spec,reason='bounded source/geometry capacity exhausted',candidates_tried=len(tried[spec['pair_id']])));continue
                    d=candidates[0];anchor[d['anchor']]+=1;raw.update(d['members']);tried[spec['pair_id']].add(d['anchor']);tasks.append((d,spec,cfg,h,None));submitted.append(spec)
                pending=[]
                for task,spec,res in zip(tasks,submitted,ex.map(attempt,tasks,chunksize=1)):
                    d=task[0]
                    raw.subtract(d['members'])
                    if res['ok']:
                        row=res['row'];results.append(row);raw.update(row['visible_source_members'])
                    else:
                        anchor[d['anchor']]-=1;rejections.append(res)
                        # A programming error must stop the build, never be hidden by resampling.
                        if res['trace']:raise RuntimeError(res['trace'])
                        pending.append(spec)
            if (offset//batchsize)%25==0:
                print(json.dumps(dict(stage='formal',done=len(results),failed_jobs=len(failed),rejections=len(rejections),seconds=round(time.monotonic()-start))),flush=True)
    t=stamp();results.sort(key=lambda r:r['pair_id'])
    for split in ['train','val']:jsonl(ROOT/f'manifests/{split}_{t}.jsonl',[r for r in results if r['split']==split])
    jsonl(ROOT/f'meta/BUILD_REJECTIONS_{t}.jsonl',rejections);jsonl(ROOT/f'meta/BUILD_FAILED_JOBS_{t}.jsonl',failed)
    save(ROOT/f'meta/BUILD_SUMMARY_{t}.json',dict(created=now(),config=str(cfgpath),pairs=len(results),views=len(results)*2,split=dict(Counter(r['split'] for r in results)),
        sizes=dict(Counter(r['width'] for r in results)),failed_jobs=len(failed),source_rejections=len(rejections),elapsed_seconds=time.monotonic()-start))
    print(json.dumps(dict(done=len(results),failed=len(failed),sizes=dict(Counter(r['width'] for r in results)))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','smoke','build']);p.add_argument('--config');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.mode=='freeze':freeze()
    elif a.mode=='smoke':smoke(a.config)
    else:build(a.config,a.workers)
