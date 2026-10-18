from data22 import *
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import argparse,time,subprocess

def verify(r):
    try:
        z=load_pair(r,use_ship_aux=True);n=r['width'];assert n<=512 and n%32==0
        H=z['H_AB'];assert np.linalg.norm(H@np.asarray(r['H_BA'])-np.eye(3))<1e-8
        assert np.allclose(H,(z['H_B']@np.linalg.inv(z['H_A']))/(z['H_B']@np.linalg.inv(z['H_A']))[2,2],atol=1e-10)
        for i in range(2):
            v=z['valid'][i];s=z['static'][i];ig=z['ignore'][i];c=z['calibration_mask'][i];unknown=z['supervision_ignore'][i];aux=z['ship_aux_masks'][i]
            assert z['images'][i].shape==(n,n) and z['images'][i].dtype==np.uint8
            assert not np.any(z['images'][i][~v])
            assert z['images'][i][v].std()>=.5 and (z['images'][i][v]==255).mean()<=.3
            assert not (s&(~v|ig|dilate(z['ship_masks'][i],5))).any()
            assert not (c&(~v|ig|dilate(z['ship_masks'][i],5)|s)).any()
            assert np.array_equal(unknown,v&~s&~c)
            assert not (aux&(~v|ig)).any()
            assert bilinear_at(s,z['points0' if i==0 else 'points1']).all()
            assert bilinear_at(aux,z['ship_points0' if i==0 else 'ship_points1']).all()
            valid_rebuilt=erode(cv2.warpPerspective(np.ones(r['source_shape'],np.float32),np.asarray(r['H_source_to_view'][i]),(n,n),flags=cv2.INTER_LINEAR)>1-1e-6,1)
            assert np.array_equal(valid_rebuilt,v)
        for p,q in [(z['points0'],z['points1']),(z['ship_points0'],z['ship_points1'])]:
            if len(p):
                assert np.max(np.linalg.norm(transform_points(H,p)-q,axis=1))<.001
                assert np.max(np.linalg.norm(transform_points(np.linalg.inv(H),q)-p,axis=1))<.001
        assert len(z['points0'])==r['structural_anchor_count']
        assert len(z['ship_points0'])==r['ship_aux_count']
        assert r['local_descriptor_eligible']==(len(z['points0'])>=4)
        global_eligible=global_ok(z['points0'],z['points1'],n)
        assert global_eligible==r['global_eligible']
        o=overlap_mask(*(a.astype('uint8')*255 for a in z['valid']),H);fr=[ratio(a,b) for a,b in zip(o,z['valid'])];assert np.max(np.abs(np.asarray(fr)-r['overlap_fractions']))<1e-9
        if r['loss_role']=='no_overlap':
            assert max(fr)==0 and not len(z['points0']) and not len(z['ship_points0'])
            for i,xy in enumerate(z['negative_probe_points']):assert bilinear_at(z['safe_static'][i]&~z['ignore'][i],xy).all()
        else:assert min(fr)>=.12-1e-8
        if len(z['ship_points0'])>1:
            for xy in [z['ship_points0'],z['ship_points1']]:
                d=np.linalg.norm(xy[:,None]-xy[None],axis=2);np.fill_diagonal(d,999);assert d.min()>=8-1e-5
        if r['dynamic_events']:assert not len(z['ship_points0']),'dynamic ship responses must have no auxiliary pair labels'
        off=load_pair(r,use_ship_aux=False);assert not len(off['ship_points0']) and np.array_equal(off['points0'],z['points0'])
        return dict(sample_id=r['sample_id'],arm=r['arm'],passed=True,role=r['loss_role'],ground=len(z['points0']),ship=len(z['ship_points0']),
            input_pixels_verified=True,ground_pixels=[int(a.sum()) for a in z['static']],calibration_pixels=[int(a.sum()) for a in z['calibration_mask']],
            new_image_sha=[digest_array(a) for a in z['images']] if r['origin']=='new_ROI' else None,
            label_sha=[digest_array(z[k]) for k in ['points0','points1','ship_points0','ship_points1']])
    except Exception as ex:return dict(sample_id=r['sample_id'],arm=r['arm'],passed=False,error=repr(ex))

def interface(summary,rr):
    import torch
    torch.set_num_threads(2)
    chosen=[];seen=set()
    for r in rr:
        keys=[('base',r['arm'],r['split'],r['loss_role']),('shape',r['width'],r['split']),('profile',r['arm'],r['profile']),('scene',r['source_scene_category'],r['split']),('n',r['structural_anchor_count']) if r['structural_anchor_count'] in [1,4,8,12,16] else None,('ship',r['ship_descriptor_eligible'],r['arm'])]
        keys=[k for k in keys if k is not None]
        if any(k not in seen for k in keys):chosen.append(r);seen.update(keys)
    tensor_tests=[]
    for r in chosen:
        z=load_pair(r,use_ship_aux=True);x=tensors22(z)
        assert x['images'].shape==(2,1,r['height'],r['width'])
        assert all(x[k].shape==x['images'].shape for k in ['valid','static','ignore','supervision_ignore','calibration_mask','ship_aux_masks'])
        assert x['images'].min()>=0 and x['images'].max()<=1
        tensor_tests.append(dict(sample_id=r['sample_id'],arm=r['arm'],split=r['split'],role=r['loss_role'],N=r['structural_anchor_count'],shape=list(x['images'].shape)))
    cpu_cases=[]
    for N in [1,4,8,12,16]:cpu_cases.append(next(r for r in rr if r['arm']=='V2-R' and r['structural_anchor_count']==N))
    for size in [256,384,512]:cpu_cases.append(next(r for r in rr if r['arm']=='V2-R' and r['global_eligible'] and r['width']==size))
    cpu_cases += [next(r for r in rr if r['loss_role']=='no_overlap'),next(r for r in rr if r['loss_role']=='no_structural_positive' and sum(r['calibration_pixels'])>100),
                  next(r for r in rr if r['source_dataset']=='ssdd' and r['local_descriptor_eligible']),next(r for r in rr if r['arm']=='V2-S' and r['local_scattering_events'] and r['local_descriptor_eligible']),
                  next(r for r in rr if r['source_scene_category']=='offshore_vessels_low_texture_sea' and r['ship_descriptor_eligible'])]
    t=stamp();cases=ROOT/f'CPU_CASES_{t}.jsonl';jsonl(cases,cpu_cases);cpu=[]
    for model in ['M2','B']:
        p=subprocess.run([sys.executable,'-B',str(ROOT/'code/cpu22.py'),model,str(cases)],capture_output=True,text=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),timeout=240)
        if p.returncode:raise RuntimeError(model+': '+p.stdout+'\n'+p.stderr)
        cpu.append(json.loads(p.stdout.strip().splitlines()[-1]))
    from sampling22 import Online22,replay22,BalancedSampler
    online_logs=[];online_ok=0;online_rejected=[]
    reference=summary['manifests']['V2-R/train/core/all'];cfg=json.loads(Path(summary['config']).read_text());sourcemap={r['image_path']:r for r in sources()}
    for arm in ['V2-R','V2-S']:
        for mode in ['M2','B']:
            log=ROOT/f'ONLINE_CHECK_{arm}_{mode}_{t}.jsonl';ds=Online22(summary['config'],reference,log,mode=mode,arm=arm,use_ship_aux=True)
            for step in range(8):
                try:z=ds.get(step,epoch=0)
                except ValueError as ex:online_rejected.append(dict(arm=arm,mode=mode,step=step,reason=str(ex)));continue
                event=z['sampling_record'];replay22(event,sourcemap[event['source_path']],cfg);online_ok+=1
            online_logs.append(str(log))
    # Exact quotas + hierarchy simulation is NOT model training or new imagery.
    simulations=[]
    for mode in ['B','M2']:
        refs=rows(reference);sampler=BalancedSampler(refs,mode=mode)
        # Direct nested keys avoid counting a source more than once across roles.
        eligible={src for role,comps in sampler.groups.items() if sampler.weights[role] for group in comps.values() for src in group}
        calls=Counter();roles=Counter();comps=set();rois=set();sequence=[]
        for step in range(50000):
            r,event=sampler.get(step);calls[r['source_path']]+=1;roles[event['requested_role']]+=1;comps.add(r['known_component']);rois.add((r['source_path'],r['roi_family_id']))
            if step<10000:sequence.append(dict(**event,mode=mode))
            if step+1 in [10000,50000]:simulations.append(dict(mode=mode,calls=step+1,roles=dict(roles),seen_source_files=len(calls),eligible_source_files=len(eligible),coverage_fraction=len(calls)/len(eligible),components=len(comps),seen_roi_families=len(rois),min_seen_source_calls=min(calls.values()),max_source_calls=max(calls.values()),median_seen_source_calls=float(np.median(list(calls.values()))),sampling_simulation_only=True))
        jsonl(ROOT/f'SAMPLING_SEQUENCE_{mode}_{t}.jsonl',sequence)
    save(ROOT/f'SAMPLING_COVERAGE_{t}.json',dict(at=now(),simulations=simulations,no_images_generated=True,no_model_training=True))
    return dict(at=now(),passed=True,tensor_cases=tensor_tests,CPU_models=cpu,online_replayed_successes=online_ok,online_rejections=online_rejected,online_logs=online_logs,
        sampling_simulation=simulations,model_optimizer_steps=0,GPU_training=False,ship_aux_default_off_verified=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest_summary',type=Path);a=p.parse_args();s=json.loads(a.manifest_summary.read_text());rr=rows(s['all_rows_with_roi_family']);res=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,r in enumerate(ex.map(verify,rr,chunksize=5),1):
            res.append(r)
            if i%1500==0 or i==len(rr):print(json.dumps(dict(checked=i,total=len(rr),failed=sum(not x['passed'] for x in res),seconds=round(time.monotonic()-start))),flush=True)
    t=stamp();jsonl(ROOT/f'FULL_LABEL_INPUT_QA_{t}.jsonl',res);bad=[r for r in res if not r['passed']]
    save(ROOT/f'GT_MASK_QA_{t}.json',dict(at=now(),passed=not bad,checked_pair_records=len(res),checked_view_instances=2*len(res),failures=bad))
    assert not bad,bad[:5]
    # Source, parent-component and original Test isolation use cached existing index evidence.
    sets={split:{r['known_component'] for r in rr if r['split']==split} for split in ['train','val']};assert not sets['train']&sets['val']
    originals=rows(BASE/'DualSAR_DatasetPrep_20261014_2131/SOURCE_INVENTORY_20261014_2142.jsonl');tests={r['known_component'] for r in originals if r['split']=='test'};assert not tests&(sets['train']|sets['val'])
    output=interface(s,rr);save(ROOT/f'INTERFACE_QA_V22_{stamp()}.json',output);print(json.dumps(dict(passed=True,disk_pairs=len(res),tensor_cases=len(output['tensor_cases']),online_ok=output['online_replayed_successes'])),flush=True)
if __name__=='__main__':main()
