"""Actual disk-read geometry/mask QA, stratified CPU interface test and replay."""
from dataset_v2 import *
from concurrent.futures import ProcessPoolExecutor
from collections import Counter,defaultdict
import argparse, csv, time
_SOURCE_LOOKUP=None

def verify_row(r):
    try:
        z=load_training_pair(r,max_points=None);n=r['width'];assert n<=512 and n==r['height'] and n%32==0
        H=z['H_AB'];assert np.linalg.norm(H@np.asarray(r['H_BA'])-np.eye(3))<1e-8
        recompute=z['H_B']@np.linalg.inv(z['H_A']);recompute/=recompute[2,2];assert np.allclose(H,recompute,atol=1e-10)
        for i in range(2):
            im=z['images'][i];v=z['valid'][i];s=z['static'][i];ign=z['ignore'][i];ship=z['ship_masks'][i]
            assert im.shape==(n,n) and im.dtype==np.uint8
            for key in ['valid','static','ignore','visibility','safe_static','structure_support','calibration_mask']:
                assert z[key][i].shape==(n,n) and z[key][i].dtype==bool
            assert np.all(im[~v]==0)
            assert not (s&~v).any() and not (s&ign).any() and not (s&dilate(ship,5)).any()
            assert not (s&~z['structure_support'][i]).any()
            assert np.array_equal(z['visibility'][i],v&~ign)
            assert not (z['calibration_mask'][i]&(ign|~v|dilate(ship,5))).any()
            assert float(im[v].std())>=.5 and float((im[v]==255).mean())<=.30
            assert bilinear_at(s,z['points0'] if i==0 else z['points1']).all()
        p,q=z['points0'],z['points1'];assert len(p)==r['structural_anchor_count']
        error=float(np.linalg.norm(transform_points(H,p)-q,axis=1).max()) if len(p) else 0.
        assert error<1e-3
        if len(q):assert np.max(np.linalg.norm(transform_points(np.linalg.inv(H),q)-p,axis=1))<1e-3
        o=overlap_mask(*(v.astype('uint8')*255 for v in z['valid']),H)
        f=[ratio(a,b) for a,b in zip(o,z['valid'])];assert np.max(np.abs(np.array(f)-r['overlap_fractions']))<1e-10
        if r['profile']=='none':assert max(f)==0 and not len(p)
        else:assert min(f)>.1
        if r['loss_role']=='structural_positive':
            for xy in [p,q]:
                a=spatial_support(xy,n);assert a['count']>=16 and a['cells4x4']>=4 and a['hull_fraction']>=.05 and a['eigen_ratio']>=.03
        else:assert not len(p) and not any(m.any() for m in z['static'])
        if r['source_scene_category']=='offshore_vessels_low_texture_sea':assert not len(p)
        # Audit transformed source corners; preserve source coordinates, not an implicit local crop H.
        raw_shape=r['source_shape'];source_ones=np.ones(raw_shape,np.float32)
        for i,h in enumerate(r['H_source_to_view']):
            v=erode(cv2.warpPerspective(source_ones,np.asarray(h),(n,n),flags=cv2.INTER_LINEAR)>1-1e-6,1)
            assert np.array_equal(v,z['valid'][i])
        visible=[list(x) for x in r['visible_ship_instances']]
        for i in range(2):
            if (z['ship_masks'][i]&~z['valid'][i]).any():
                global _SOURCE_LOOKUP
                if _SOURCE_LOOKUP is None:_SOURCE_LOOKUP={s['image_path']:s for s in rows(ROOT/'SOURCES_V2_20261015_1258.jsonl')}
                source=_SOURCE_LOOKUP[r['source_path']];visible[i]=[]
                for ann in source['annotations']:
                    m,method=annotation_mask(r['source_shape'],ann)
                    area=int(((warp(m,np.asarray(r['H_source_to_view'][i]),n,mask=True)>0)&z['valid'][i]).sum())
                    if area:visible[i].append(dict(id=str(ann['id']),visible_pixels=area,method=method))
        return dict(sample_id=r['sample_id'],arm=r['arm'],passed=True,max_error=error,read_images=2,
                    valid_visible_ship_counts=list(map(len,visible)),valid_visible_ship_instances=visible,
                    decoded_image_sha=[digest_array(x) for x in z['images']],
                    decoded_mask_sha={k:[digest_array(x) for x in z[k]] for k in ['valid','static','ignore','ship_masks','calibration_mask']},
                    calibration_pixels=[int(x.sum()) for x in z['calibration_mask']],ignore_pixels=[int(x.sum()) for x in z['ignore']],
                    decoded_points_sha=[digest_array(p),digest_array(q)],H_AB_sha=digest_array(H))
    except Exception as ex:return dict(sample_id=r['sample_id'],arm=r['arm'],passed=False,error=repr(ex),read_images=0)

def interface_checks(rr,cfg,configpath):
    import torch
    torch.set_num_threads(2)
    chosen=[];seen=set()
    for r in rr:
        keys=[('combo',r['split'],r['arm'],r['profile'],r['loss_role']),('scene',r['split'],r['source_scene_category'],r['arm']),('size',r['width'],r['split']),('angle',r['angle_bin_2D'],r['arm'])]
        if any(k not in seen for k in keys):chosen.append(r);seen.update(keys)
    tested=[]
    for r in chosen:
        z=load_training_pair(r);x=tensors(z)
        assert x['images'].shape==(2,1,r['height'],r['width']) and x['images'].dtype==torch.float32
        assert x['images'].min()>=0 and x['images'].max()<=1
        assert all(x[k].shape==x['images'].shape for k in ['valid','static','ignore','visibility','calibration_mask'])
        assert x['points0'].shape==(r['structural_anchor_count'],2)
        # Existing candidate visibility function really consumes new `static`, including ignore.
        oldcode=BASE/'SAR_SP_LG_Train_20261012_1544/code'
        sys.path.insert(0,str(oldcode));from data import valid_at
        assert np.array_equal(valid_at(z['static'][0],z['points0']),bilinear_at(z['static'][0],z['points0']))
        from loss_bridge import legacy_view
        legacy=legacy_view(z)
        assert all(not np.any((~m)&ign) for m,ign in zip(legacy['ship_masks'],z['ignore']))
        tested.append(dict(sample_id=r['sample_id'],arm=r['arm'],split=r['split'],shape=list(x['images'].shape),profile=r['profile'],role=r['loss_role'],ignore_pixels=sum(int(m.sum()) for m in z['ignore'])))
    # Real frozen architecture, CPU-only forward of the existing M2 loss. No optimizer.
    import model as old_model
    torch.manual_seed(20261015);m=old_model.SARModel('m2').cpu().eval()
    positive=next(r for r in chosen if r['loss_role']=='structural_positive' and r['arm']=='V2-R')
    z=load_training_pair(positive)
    with torch.no_grad():loss,info=old_model.loss(m,z,'cpu',amp=False)
    assert torch.isfinite(loss) and info['n_points']>=16
    # B's actual architecture loaded in a subprocess to avoid `common` module-name collision.
    import subprocess
    bc=[]
    for n in [256,384,512]:
        r=next((r for r in rr if r['width']==n and r['loss_role']=='structural_positive' and r['arm']=='V2-R'),None)
        if r:bc.append(r)
    bc += [next(r for r in rr if r['profile']=='none'),next(r for r in rr if r['loss_role']=='no_structural_positive'),next(r for r in rr if r['arm']=='V2-S' and r['local_scattering_events'] and r['loss_role']=='structural_positive')]
    sub=subprocess.run([sys.executable,'-B',str(ROOT/'code/check_B_cpu.py'),json.dumps(bc)],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=180)
    if sub.returncode:raise RuntimeError('B CPU compatibility: '+sub.stdout+'\n'+sub.stderr)
    bresult=json.loads(sub.stdout.strip().splitlines()[-1]);assert bresult['passed']
    # Online get changes per visit; replay uses logged source SHA, explicit seeds and masks.
    allsources=rows(cfg['source_pool']);paths={r['source_path'] for r in rr if r['split']=='train' and r['loss_role']=='structural_positive' and r['source_dataset']!='ssdd'}
    source_map={r['image_path']:r for r in allsources};fixed_replayed=[]
    for arm in cfg['arms']:
        for profile in ['affine','projective','low','none']:
            r=next(r for r in rr if r['arm']==arm and r['profile']==profile)
            generated=array_pair(source_map[r['source_path']],r['sampling_spec'],cfg,arm);fixed=load_training_pair(r)
            for key in ['images','valid','static','ignore','calibration_mask']:
                assert all(np.array_equal(a,b) for a,b in zip(generated[key],fixed[key])),(r['sample_id'],arm,key)
            assert np.array_equal(generated['H_AB'],fixed['H_AB'])
            fixed_replayed.append(dict(sample_id=r['sample_id'],arm=arm,profile=profile))
    sources=[r for r in allsources if r['image_path'] in paths]
    t=stamp();sp=ROOT/f'ONLINE_CORE_STRUCTURAL_SOURCES_{t}.jsonl';jsonl(sp,sources)
    logs=[];online_success=0;rejections=[]
    for arm in cfg['arms']:
        log=ROOT/f'ONLINE_SMOKE_{arm}_{t}.jsonl';ds=OnlinePairs(configpath,sp,log,arm=arm)
        outputs=[]
        for visit in range(4):
            try:outputs.append(ds.get(0,epoch=0,visit=visit));online_success+=1
            except ValueError as exc:rejections.append(str(exc))
        records=rows(log)[1:];ok=[r for r in records if r['state']=='usable']
        assert ok
        for record in ok: replay(record,sources[record['index']],cfg)
        assert len(set(tuple(r['image_sha']) for r in ok))>1
        logs.append(str(log))
    # Forbidden val source fails immediately, no cross-split online default.
    bad=ROOT/f'VAL_REJECTION_SOURCE_{t}.jsonl';jsonl(bad,[next(r for r in allsources if r['split']=='val')])
    try:OnlinePairs(configpath,bad,ROOT/f'FORBIDDEN_VAL_LOG_{t}.jsonl');raise RuntimeError('val not rejected')
    except AssertionError:pass
    return dict(passed=True,tensor_samples=tested,M2_CPU_existing_loss=dict(loss=float(loss),**info),B_CPU_existing_loss_bridge=bresult,
        online_source_pool=str(sp),online_source_count=len(sources),online_success=online_success,online_rejections=rejections,online_logs=logs,replay_exact=True,
        fixed_generator_replay=fixed_replayed,no_optimizer=True,no_gpu=True,no_model_training=True,validation_online_forbidden=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);args=p.parse_args()
    summary=json.loads(args.summary.read_text());cfg=json.loads(Path(summary['config']).read_text());allrows=rows(summary['all_generated']);usable=[r for r in allrows if r['qa_state']=='usable'];t=stamp()
    results=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,res in enumerate(ex.map(verify_row,usable,chunksize=10),1):
            results.append(res)
            if i%1500==0 or i==len(usable):print(json.dumps(dict(checked=i,total=len(usable),failed=sum(not q['passed'] for q in results),seconds=round(time.monotonic()-start))),flush=True)
    t=stamp();jsonl(ROOT/f'DISK_QA_{t}.jsonl',results)
    failures=[r for r in results if not r['passed']]
    # Freeze actual relation scopes before any future algorithm run.
    byid=defaultdict(list)
    for r in usable:byid[r['sample_id']].append(r)
    assert all(len(v)==3 and len({json.dumps(r['H_AB']) for r in v})==1 for v in byid.values())
    sets={s:{r['known_component'] for r in usable if r['split']==s} for s in ['train','val']};assert not sets['train']&sets['val']
    inventory=rows(V1/'SOURCE_INVENTORY_20261014_2142.jsonl')
    test_components={r['known_component'] for r in inventory if r['split']=='test'}
    assert not test_components&(sets['train']|sets['val'])
    sourcepaths={s:{r['source_path'] for r in usable if r['split']==s} for s in ['train','val']};assert not sourcepaths['train']&sourcepaths['val']
    common={sid for sid,group in byid.items() if all(r['loss_role']=='structural_positive' for r in group)}
    common_manifest={}
    for arm in cfg['arms']:
        path=ROOT/f'COMMON_VAL_{arm}_{t}.jsonl';jsonl(path,[r for r in usable if r['split']=='val' and r['sample_id'] in common and r['arm']==arm]);common_manifest[arm]=str(path)
    save(ROOT/f'QA_GEOMETRY_{t}.json',dict(at=now(),passed=not failures,read_pair_records=len(results),read_view_instances=2*len(results),max_point_error=max(r.get('max_error',0) for r in results),failures=failures,
        component_counts={s:len(v) for s,v in sets.items()},component_intersection=[],test_component_intersection=[],source_intersection=[],matched_geometry_verified=True,common_validation_manifests=common_manifest))
    assert not failures,failures[:3]
    interface=interface_checks(usable,cfg,summary['config']);save(ROOT/f'INTERFACE_QA_{stamp()}.json',interface)
    print(json.dumps(dict(passed=True,paired_disk_checks=len(results),tensor_samples=len(interface['tensor_samples']),online_source_count=interface['online_source_count'],common_val=common_manifest)),flush=True)
if __name__=='__main__':main()
