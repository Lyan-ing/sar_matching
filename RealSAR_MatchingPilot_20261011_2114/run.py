"""Bounded four-pair SAR/RoMa execution; human annotations never enter predictors."""
from core import *
import argparse, os, signal, subprocess, random
from PIL import Image
from datetime import datetime
from zoneinfo import ZoneInfo

def prepare():
    prep=BASE/'RealSAR_IndependentBenchmark_20261011_2015';plan=json.loads((prep/'pilot_plan.json').read_text())
    ids=plan['priority_pairs'];source=lines(BASE/'RealSAR_RepresentationContext_20261011_1903/samples.jsonl')
    chosen=[next(r for r in source if r['sample_id']==sid) for sid in ids]
    if len(chosen)!=4 or 'Umbra_0071__chip_000' in ids:raise ValueError('wrong fixed samples')
    with (ROOT/'samples.jsonl').open('x') as f:
        for r in chosen:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    cfg=dict(seed=42,grid=16,gradient_half_window=3,direction_bins=8,descriptor_window=48,descriptor_spatial_cells=4,
        histogram_signed_directions=True,orientation_bin_linear_interpolation=True,L2_Hys_clip=.2,
        epsilon_relative_median_positive=1e-6,ratio=.8,cycle_px=3,geographic_window_px=64,
        geographic_window_norm='L_infinity; BOTH axes within +/-64 moving native pixels',
        A_validity='complete descriptor48 and gradient7 support; true valid zero retained',
        A_input='native linear_amplitude; linear_power sqrt; no contrast processing',
        B_input='frozen native grayscalePNG converted to RGB by official matcher',
        RoMa_commit=plan['actual_repositories']['Parskatt/RoMa'],B_resolution=[560,864],batch=1,max_matches=2048,
        B_backend='official native torch correlation fallback use_custom_corr=False; no compiled optional extension',
        B_sampling='official threshold_balanced; sample_thresh.05; sample num2048; same exact pool forB1/B2',
        B_coordinates='official corner coordinates minus0.5 to native pixel centers',
        global_geometry='unchanged estimate affine+full gate, native512',local_geometry='4x4 source tiles; unchanged full gate in native512 frame; no global promotion',
        CPU_hours_max=2,GPU_hours_max=2,pair_timeout_seconds=120,GPU_allocated_GiB_max=20,
        methods=['A1','A2','B1','B2'],new_pair_calls_max=16,extra_timing_max=3,
        C=0,GT_access_by_predictors=False,training=False,preprocessing_search=False)
    save(ROOT/'config.json',cfg)
    old=BASE/'RealSAR_AutoLabels_20261011_1149';baseline=[r for r in lines(old/'quality/model_grades.jsonl') if r['sample_id'] in ids]
    assert len(baseline)==24
    protected={prep/'pilot_plan.json',prep/'PREPARATION_ACCEPTANCE.json',old/'quality/model_grades.jsonl',
        BASE/'MaritimeSARRegistrationPhase2/output/phase25_20261010_0000/development/selection.json',
        BASE/'MaritimeSARRegistrationPhase2/sarbench/estimate.py',BASE/'MaritimeSARRegistrationPhase2/sarbench/reliability.py',
        BASE/'RealSAR_AutoLabels_20261011_1149/geo.py'}
    for r in baseline:
        p=Path(r['cache_path']);assert sha(p)==r['cache_sha256'];protected.add(p)
    for r in chosen:
        protected.update(Path(r[k]) for k in ['reference_native_png','moving_native_png','reference_native_geotiff','moving_native_geotiff'])
        protected.update(Path(r['folder'])/n for n in ['reference_valid.png','moving_native_valid.png'])
    save(ROOT/'baseline_selection.json',baseline)
    save(ROOT/'protected_inputs.json',{str(p):sha(p) for p in sorted(protected)})
    save(ROOT/'preregistration.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),plan_sha256=sha(prep/'pilot_plan.json'),
        config_sha256=sha(ROOT/'config.json'),samples_sha256=sha(ROOT/'samples.jsonl'),core_sha256=sha(ROOT/'core.py'),run_sha256=sha(__file__),
        operational_descriptor_choices_fixed_before_inference=True,original_samples_unchanged=True))
    (ROOT/'logs').mkdir();print('prepared4pairs,24cachedbaseline, descriptor implementation fixed before outcomes',flush=True)

def persist(folder,row,method,result):
    r,local,cache=result
    p=folder/(method+'_'+row['sample_id']);p.mkdir(exist_ok=False)
    np.savez_compressed(p/'matches.npz',**cache)
    r.update(cache_path=str(p/'matches.npz'),cache_sha256=sha(p/'matches.npz'))
    save(p/'record.json',r);save(p/'local.json',local)
    print(row['sample_id'],method,'matches',r['n_matches'],'inliers',r['n_inliers'],'accepted',r['accepted'],r['reason'],'local',r['local_accepted_tiles'],flush=True)

def timeout(signum,frame):raise TimeoutError('pair exceeded120seconds')

def run_A():
    folder=ROOT/'route_A';folder.mkdir(exist_ok=False);budget=time.perf_counter();cv2.setNumThreads(1)
    for row in selected():
        if time.perf_counter()-budget>7200:raise TimeoutError('CPU budget exceeded')
        t=time.perf_counter();signal.signal(signal.SIGALRM,timeout);signal.alarm(120)
        try:
            raw,valid,images,infos,mapper=load_pair(row);pts=[];des=[];meta=[]
            for i,prefix in enumerate(['reference','moving']):
                p,d,m=descriptors(raw[i],valid[i],row[prefix+'_input_units']);pts.append(p);des.append(d);meta.append(m)
            feature_ms=1000*(time.perf_counter()-t)
            artifact=folder/(row['sample_id']+'_features');artifact.mkdir()
            np.savez_compressed(artifact/'features.npz',points0=pts[0],points1=pts[1],descriptors0=des[0],descriptors1=des[1],valid0=valid[0],valid1=valid[1])
            save(artifact/'metadata.json',dict(views=meta,feature_ms=feature_ms,input_files=[row['reference_native_geotiff'],row['moving_native_geotiff']]))
            for method,geowindow in [('A1',False),('A2',True)]:
                s=time.perf_counter();allowed=geographical_window(pts[0],pts[1],mapper) if geowindow else None
                p0,p1,confidence,stats=mutual_match(pts[0],des[0],pts[1],des[1],allowed)
                search_ms=1000*(time.perf_counter()-s)
                stats.update(descriptor_metadata=meta,input_protocol='native_amplitude',n_candidates0=len(pts[0]),n_candidates1=len(pts[1]),
                    failure='no_descriptors' if not(len(pts[0]) and len(pts[1])) else ('no_ratio_mutual_matches' if not len(p0) else None),
                    features_shared_between_A1_A2=True,geographic_window=geowindow)
                r=diagnostics(row,method,'native_amplitude_geo64' if geowindow else 'native_amplitude',p0,p1,confidence,valid,mapper,
                    dict(feature_and_read_ms=feature_ms,matching_ms=search_ms,shared_feature_cost=True,GPU_allocated_MiB=0),stats)
                r[0]['end_to_end_ms']=feature_ms+search_ms+r[0]['diagnostic_ms']
                persist(folder,row,method,r)
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0);save(folder/(row['sample_id']+'_FAILURE.json'),dict(error=repr(e),elapsed_ms=1000*(time.perf_counter()-t)))
            print('A failed',row['sample_id'],repr(e),flush=True)

def run_B():
    # Dependencies are local to this run; protect the base match package set.
    sys.path.insert(0,str(ROOT/'vendor/deps'));sys.path.insert(0,str(ROOT/'vendor/RoMa'))
    import torch
    from romatch import roma_outdoor
    folder=ROOT/'route_B';folder.mkdir(exist_ok=False)
    if not torch.cuda.is_available():raise RuntimeError('RoMa requires actualGPU for this bounded run')
    torch.cuda.set_per_process_memory_fraction(20*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.set_float32_matmul_precision('highest');torch.set_num_threads(4)
    weights=json.loads((ROOT/'weights/identity.json').read_text())
    for w in weights:
        if w['status']!='downloaded' or sha(ROOT/'weights'/w['name'])!=w['sha256']:raise ValueError('official weight identity invalid')
    startup=time.perf_counter();torch.manual_seed(42)
    # Pass explicit local states: no automatic hub download or random fallback.
    model=roma_outdoor(device='cuda',weights=torch.load(ROOT/'weights/roma_outdoor.pth',map_location='cpu',weights_only=True),
        dinov2_weights=torch.load(ROOT/'weights/dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True),use_custom_corr=False).eval()
    for param in model.parameters():param.requires_grad_(False)
    for param in model.encoder.dinov2_vitl14[0].parameters():param.requires_grad_(False)
    torch.cuda.synchronize();startup_ms=1000*(time.perf_counter()-startup)
    save(folder/'model_identity.json',dict(startup_ms=startup_ms,commit='77f8d68803526dcddfd9b7a46bc76125bdc25f15',weights=weights,
        licenses=['RoMaMIT','DINOv2Apache2.0'],backend='official native torch local correlation',
        coarse_res=[model.h_resized,model.w_resized],upsample_res=model.upsample_res,sample_mode=model.sample_mode,sample_thresh=model.sample_thresh,
        torch=torch.__version__,cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(0),training=False))
    budget=time.perf_counter()
    for index,row in enumerate(selected()):
        if time.perf_counter()-budget>7200:raise TimeoutError('GPU budget exceeded')
        signal.signal(signal.SIGALRM,timeout);signal.alarm(120);start=time.perf_counter()
        try:
            raw,valid,images,infos,mapper=load_pair(row);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            torch.manual_seed(42);np.random.seed(42);random.seed(42)
            with torch.inference_mode():
                warp,certainty=model.match(row['reference_native_png'],row['moving_native_png'],device='cuda')
                matches,sc=model.sample(warp,certainty,num=2048)
                c0,c1=model.to_pixel_coordinates(matches,512,512,512,512)
            torch.cuda.synchronize();gpu_ms=1000*(time.perf_counter()-start);peak=torch.cuda.max_memory_allocated()/1024**2
            if peak>20*1024:raise MemoryError('allocated GPU exceeds20GiB')
            p0,p1=c0.cpu().numpy().astype(float)-.5,c1.cpu().numpy().astype(float)-.5
            normalized=matches.cpu().numpy();conf=sc.cpu().numpy();densewarp=warp.cpu().numpy();densecert=certainty.cpu().numpy()
            # Retain raw certainty for each sampled match; official sample()'s returned
            # confidence may be thresholded to1. Locate exact normalized tuple.
            lookup={tuple(x):i for i,x in enumerate(densewarp.reshape(-1,4))}
            sample_ids=np.array([lookup[tuple(x)] for x in normalized]);raw_conf=densecert.ravel()[sample_ids]
            asset=folder/(row['sample_id']+'_dense');asset.mkdir()
            np.savez_compressed(asset/'dense.npz',warp=densewarp,certainty=densecert,sample_flat_indices=sample_ids,
                                normalized_samples=normalized,sampled_thresholded_certainty=conf,raw_sample_certainty=raw_conf,valid0=valid[0],valid1=valid[1])
            save(asset/'timing.json',dict(read_match_sample_gpu_ms=gpu_ms,peak_allocated_MiB=peak,peak_reserved_MiB=torch.cuda.max_memory_reserved()/1024**2,
                actual_batch=1,cold_first=index==0,startup_ms_excluded=startup_ms))
            for method,use_window in [('B1',False),('B2',True)]:
                s=time.perf_counter();keep=np.ones(len(p0),bool)
                if use_window:keep=np.isfinite(mapper(p0)).all(1)&(np.abs(p1-mapper(p0))<=64).all(1)
                filter_ms=1000*(time.perf_counter()-s)
                result=diagnostics(row,method,'native_PNG_geo64_postfilter' if use_window else 'native_PNG',p0[keep],p1[keep],raw_conf[keep],valid,mapper,
                    dict(read_match_sample_gpu_ms=gpu_ms,geo_filter_ms=filter_ms,GPU_allocated_MiB=peak,shared_dense_inference=True,cold_first=index==0),
                    dict(input_protocol='native_PNG',same_exact_dense_output=True,raw_pool=2048,geographic_filtered_out=int((~keep).sum()),
                         sampling_certainty_is_thresholded=True,n_keypoints0=None,n_keypoints1=None))
                result[0]['end_to_end_ms']=gpu_ms+filter_ms+result[0]['diagnostic_ms'];persist(folder,row,method,result)
            del warp,certainty,matches,densewarp,densecert;signal.alarm(0)
        except Exception as e:
            signal.alarm(0);save(folder/(row['sample_id']+'_FAILURE.json'),dict(error=repr(e),elapsed_ms=1000*(time.perf_counter()-start)))
            print('B failed',row['sample_id'],repr(e),flush=True)
            if isinstance(e,(MemoryError,torch.cuda.OutOfMemoryError)):break
    # Exactly three predeclared warm timing repetitions on the first same-look
    # control in the fixed order. They never replace the primary predictions.
    control=next(r for r in selected() if r['look_pair_category']=='repeat_pass_same_look')
    original=folder/('B1_'+control['sample_id'])/'matches.npz'
    if original.exists():
        with np.load(original) as z:expected0=z['points0'];expected1=z['points1_native']
        repeats=[]
        for repetition in range(3):
            if time.perf_counter()-budget>7200:break
            signal.signal(signal.SIGALRM,timeout);signal.alarm(120);t=time.perf_counter()
            try:
                raw,valid,images,infos,mapper=load_pair(control);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(42);np.random.seed(42);random.seed(42)
                with torch.inference_mode():
                    warp,cert=model.match(control['reference_native_png'],control['moving_native_png'],device='cuda')
                    nm,sc=model.sample(warp,cert,num=2048);c0,c1=model.to_pixel_coordinates(nm,512,512,512,512)
                torch.cuda.synchronize();inference_ms=1000*(time.perf_counter()-t)
                p0,p1=c0.cpu().numpy().astype(float)-.5,c1.cpu().numpy().astype(float)-.5
                diag=diagnostics(control,'B1_timing_only','native_PNG',p0,p1,sc.cpu().numpy(),valid,mapper,{})
                repeats.append(dict(repetition=repetition+1,sample_id=control['sample_id'],read_match_sample_ms=inference_ms,
                    end_to_end_ms=1000*(time.perf_counter()-t),peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                    exact_primary_coordinates=np.array_equal(p0,expected0) and np.array_equal(p1,expected1),
                    primary_coordinate_max_difference_px=float(max(np.max(np.abs(p0-expected0)),np.max(np.abs(p1-expected1)))),
                    full_gate=diag[0]['accepted'],reason=diag[0]['reason'],primary_predictions_replaced=False))
                del warp,cert,nm;signal.alarm(0)
            except Exception as e:
                signal.alarm(0);repeats.append(dict(repetition=repetition+1,error=repr(e)));break
        save(folder/'timing_repeats.json',repeats)

def baseline():
    folder=ROOT/'baselines';folder.mkdir(exist_ok=False);rows={r['sample_id']:r for r in selected()}
    for old in json.loads((ROOT/'baseline_selection.json').read_text()):
        row=rows[old['sample_id']];raw,valid,images,infos,mapper=load_pair(row)
        with np.load(old['cache_path']) as c:p0=c['points0'];p1=c['points1_native']
        result=diagnostics(row,old['method'],old['protocol'],p0,p1,np.full(len(p0),np.nan),valid,mapper,
            dict(old_cached_end_to_end_ms=old['end_to_end_ms'],historical_process_NOT_new_timing=True,GPU_allocated_MiB=None),
            dict(baseline_reused=True,original_frozen_prediction_frame_accept=old['accepted'],original_frozen_reason=old['reason'],
                old_cache_path=old['cache_path'],old_cache_sha256=old['cache_sha256'],original_n_keypoints0=old['n_keypoints0'],original_n_keypoints1=old['n_keypoints1'],
                original_native_candidate_H=old['native_candidate_H']))
        result[0]['end_to_end_ms']=old['end_to_end_ms'];persist(folder,row,old['protocol']+'_'+old['method'],result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','A','B','baseline']);a=p.parse_args()
    if a.stage!='prepare':
        pr=json.loads((ROOT/'preregistration.json').read_text())
        assert sha(ROOT/'config.json')==pr['config_sha256'] and sha(ROOT/'samples.jsonl')==pr['samples_sha256']
    {'prepare':prepare,'A':run_A,'B':run_B,'baseline':baseline}[a.stage]()
