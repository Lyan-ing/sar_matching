"""Image-only A / geography-assisted B. Predictors never import human evidence."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time,os,importlib.util
from pathlib import Path
import numpy as np,cv2
from PIL import Image
from geo_tools import ROOT,BASE,PixelMap,Affine,read_tif,sha,save,clean,lines,jsonl,sample_valid,center_metric
PROJECT=BASE/'MaritimeSARRegistrationPhase2';sys.path.insert(0,str(PROJECT))
from sarbench.matchers import Prediction,LocalFeatureMatcher,SuperPointLightGlue
from sarbench.estimate import estimate,perspective
from sarbench.reliability import geometry_diagnostics,gate,spatial
SELECTION=PROJECT/'output/phase25_20261010_0000/development/selection.json'
spec=importlib.util.spec_from_file_location('v2_frozen_structure',BASE/'RealSAR_MatchingPilot_20261011_2114/core.py')
structure=importlib.util.module_from_spec(spec);spec.loader.exec_module(structure)
def detector(method):
    if method in ['sift','orb']:return LocalFeatureMatcher(method,max_features=2048,ratio=.8,detector_options={'contrastThreshold':.01} if method=='sift' else {'edgeThreshold':16})
    if method=='lightglue':
        os.environ['TORCH_HOME']=str(PROJECT/'cache/torch')
        paths=list((PROJECT/'cache/torch/hub/checkpoints').glob('*'))
        if not all(any(p.name==n for p in paths) for n in ['superpoint_v1.pth','superpoint_lightglue_v0-1_arxiv.pth']):raise RuntimeError('Frozen official weights not locally cached; automatic download forbidden')
        return SuperPointLightGlue(device='auto',max_features=2048,resize=None,detector_threshold=.0001)
    raise ValueError('unknown matcher')
def fit(p0,p1,shape0,shape1):
    pred=Prediction('shared',np.asarray(p0,np.float32),np.asarray(p1,np.float32));e=estimate(pred,shape0,model='affine',seed=42)
    d=geometry_diagnostics(e,pred,shape0,shape1,42) if e.candidate_H is not None else {};return gate(e,d,'full'),d
def register(image0,image1,method,protocol='A',geography=None,matcher=None,raw=None,technical_validity=None,units=None,geographic_window_px=64):
    """Return pixel-center input-frame prediction, not independent correctness.

    Geography is forbidden for A. B only exposes provider geometry; human points,
    overlap GT, annotations and pseudo labels have no API argument or code path.
    Technical validity comes from native file masks, not semantic/GT masks.
    """
    if protocol not in ['A','B']:raise ValueError('unknown protocol')
    if protocol=='A' and geography is not None:raise ValueError('A cannot access geography')
    if protocol=='B' and geography is None:raise ValueError('B requires provider geography')
    start=time.perf_counter()
    mapper=None if geography is None else PixelMap(Affine(*geography[0]['transform']),geography[0]['crs'],Affine(*geography[1]['transform']),geography[1]['crs'])
    if method=='sar_structure':
        if raw is None or units is None or technical_validity is None:raise ValueError('Structure requires native radiometry and technical validity')
        d=[structure.descriptors(a,v,u) for a,v,u in zip(raw,technical_validity,units)];p0,d0,_=d[0];p1,d1,_=d[1]
        allowed=structure.geographical_window(p0,p1,mapper,64) if mapper is not None else None
        p0,p1,confidence,matchstats=structure.mutual_match(p0,d0,p1,d1,allowed)
        pred=Prediction(method,p0.astype(np.float32),p1.astype(np.float32));extra=dict(descriptor_stats=[a[2] for a in d],matching=matchstats,confidence=confidence)
    else:
        model=matcher if matcher is not None else detector(method);pred=model.predict(image0,image1);extra={}
        if mapper is not None and geographic_window_px is not None:
            good=np.isfinite(mapper(pred.points0)).all(1)&(np.abs(pred.points1-mapper(pred.points0))<=geographic_window_px).all(1)
            extra['geographic_window_rejected']=int((~good).sum());pred.points0=pred.points0[good];pred.points1=pred.points1[good]
    if technical_validity is not None:
        good=sample_valid(technical_validity[0],pred.points0)&sample_valid(technical_validity[1],pred.points1)
        extra['technical_invalid_candidates']=int((~good).sum());pred.points0=pred.points0[good];pred.points1=pred.points1[good]
    shape0,shape1=image0.shape,image1.shape;e,d=fit(pred.points0,pred.points1,shape0,shape1);iv=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(pred.points0),bool)
    local=[]
    for gy in range(4):
        for gx in range(4):
            x0,x1=gx*shape0[1]/4,(gx+1)*shape0[1]/4;y0,y1=gy*shape0[0]/4,(gy+1)*shape0[0]/4
            ids=np.flatnonzero((pred.points0[:,0]>=x0)&(pred.points0[:,0]<x1)&(pred.points0[:,1]>=y0)&(pred.points0[:,1]<y1))
            le,ld=fit(pred.points0[ids],pred.points1[ids],shape0,shape1)
            local.append(dict(tile_xyxy=[x0,y0,x1,y1],n_matches=len(ids),n_inliers=le.n_inliers,accepted=le.accepted,candidate_H=le.candidate_H,reason=le.reason,geometry=ld,scope='reference tile only; full-frame frozen gate; no whole-image extrapolation'))
    if iv.sum()>=3:support=cv2.convexHull(pred.points0[iv].astype(np.float32)).reshape(-1,2)
    else:support=np.empty((0,2))
    return clean(dict(method=method,protocol=protocol,points0=pred.points0,points1=pred.points1,inlier_mask=iv,global_H=e.H,candidate_H=e.candidate_H,accepted=e.accepted,
        decision='accepted_unassessable' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),reason=e.reason,
        n_matches=len(pred.points0),n_inliers=e.n_inliers,inlier_ratio=e.inlier_ratio,n_keypoints=[pred.n_keypoints0,pred.n_keypoints1],geometry=d,
        sampled_support_polygon=support,support_is_NOT_dense_correct_region=True,local=local,scope='global_candidate_with_sampled_support' if e.accepted else 'diagnostic_only',
        confidence=dict(calibrated_probability=None,signals=dict(inlier_ratio=e.inlier_ratio,spatial0=spatial(pred.points0[iv],shape0),spatial1=spatial(pred.points1[iv],shape1))),
        independent_accuracy_available=False,elapsed_ms=1000*(time.perf_counter()-start),extra=extra))
def raw_geography(z0,z1,points):return PixelMap(Affine(*z0['transform']),z0['crs'],Affine(*z1['transform']),z1['crs'])(points)
def infer_region(region,method,protocol,matcher=None):
    start=time.perf_counter();folder=ROOT/'regions'/region['sample_id'];views=region['views']
    # A: no metadata goes into matching or gate; geometry only restores published
    # coordinates. B: shared FOV is geographically prepared input, not truth.
    scope='native' if protocol=='A' or method=='sar_structure' else 'common'
    pack=[read_tif(folder/f'{scope}{i}.tif') for i in [0,1]];raw=[p[0] for p in pack];valid=[p[1] for p in pack];zs=[p[2] for p in pack]
    images=[np.array(Image.open(folder/f'{scope}_db{i}.png')) for i in [0,1]];units=[region['acquisition_metadata']['views'][i]['input_units'] for i in [0,1]]
    # B common grids get a native +-64 filter after restoration; an input-space
    # window would alter physical extent across sampling scales.
    result=register(*images,method,protocol,geography=zs if protocol=='B' else None,matcher=matcher,raw=raw,technical_validity=valid,units=units,geographic_window_px=64 if scope=='native' else None)
    maps=[PixelMap(Affine(*zs[i]['transform']),zs[i]['crs'],Affine(*views[i]['native_info']['transform']),views[i]['native_info']['crs']) for i in [0,1]]
    p0=maps[0](result['points0']);p1=maps[1](result['points1']);native_z=[v['native_info'] for v in views]
    if protocol=='B':
        delta=np.abs(raw_geography(*native_z,p0)-p1);good=np.isfinite(delta).all(1)&(delta<=64).all(1)
        result['extra']['native_geographic_window_rejected']=int((~good).sum());p0,p1=p0[good],p1[good]
    # Fit and gate in SAME native pixel coordinates for both protocols; B cannot
    # inherit more permissive resized-grid residuals or space support.
    shape0=(native_z[0]['height'],native_z[0]['width']);shape1=(native_z[1]['height'],native_z[1]['width'])
    e,d=fit(p0,p1,shape0,shape1);iv=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(p0),bool)
    native_masks=[read_tif(folder/f'native{i}.tif')[1] for i in [0,1]];ok=sample_valid(native_masks[0],p0)&sample_valid(native_masks[1],p1)
    # Conservative restoration: remove any fractional candidate invalid in native
    # data, then refit once, independent of GT.
    if not ok.all():p0,p1=p0[ok],p1[ok];e,d=fit(p0,p1,shape0,shape1);iv=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(p0),bool)
    local=[]
    for gy in range(4):
        for gx in range(4):
            bounds=[gx*shape0[1]/4,gy*shape0[0]/4,(gx+1)*shape0[1]/4,(gy+1)*shape0[0]/4]
            ids=np.flatnonzero((p0[:,0]>=bounds[0])&(p0[:,0]<bounds[2])&(p0[:,1]>=bounds[1])&(p0[:,1]<bounds[3]));le,ld=fit(p0[ids],p1[ids],shape0,shape1)
            local.append(dict(tile_xyxy=bounds,n_matches=len(ids),n_inliers=le.n_inliers,accepted=le.accepted,candidate_H=le.candidate_H,reason=le.reason,geometry=ld))
    support=cv2.convexHull(p0[iv].astype(np.float32)).reshape(-1,2) if iv.sum()>=3 else np.empty((0,2))
    result.update(sample_id=region['sample_id'],input_scope=scope,input_shapes=[a.shape for a in images],points0=p0,points1=p1,inlier_mask=iv,candidate_H=e.candidate_H,global_H=e.H,accepted=e.accepted,reason=e.reason,
        decision='accepted_unassessable' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),n_matches=len(p0),n_inliers=e.n_inliers,inlier_ratio=e.inlier_ratio,geometry=d,local=local,sampled_support_polygon=support,coordinate_frame='reference_native_to_moving_native_pixel_centers',
        geo_candidate_deviation_median_px=float(np.median(np.linalg.norm(raw_geography(*native_z,p0)-p1,axis=1))) if len(p0) else None,
        geographic_overlap=region['native_geographic_overlap'],end_to_end_ms=1000*(time.perf_counter()-start),map_is_NOT_GT=True,
        confidence=dict(calibrated_probability=None,signals=dict(inlier_ratio=e.inlier_ratio,spatial0=spatial(p0[iv],shape0),spatial1=spatial(p1[iv],shape1))))
    return clean(result)
def smoke():
    import torch
    torch.set_num_threads(1);cv2.setNumThreads(1);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    selection=json.loads(SELECTION.read_text());expected={'sift':'sift_contrast01','orb':'orb_edge16','lightglue':'lightglue_detect1e4'}
    assert all(selection['selected'][m]['variant']==v for m,v in expected.items())
    for name in ['sarbench/matchers.py','sarbench/estimate.py','sarbench/reliability.py']:assert sha(PROJECT/name)==selection['inference_code_sha256'][name]
    regions=lines(ROOT/'regions.jsonl');chosen=[next((r for r in regions if r['look_pair_category']==look),None) for look in ['different_look_verified','repeat_pass_same_look']];chosen=[r for r in chosen if r]
    out=ROOT/'smoke';out.mkdir(exist_ok=False);records=[];initialization=[]
    for method in ['sift','orb','lightglue','sar_structure']:
        model=None;start=time.perf_counter()
        try:
            if method!='sar_structure':model=detector(method)
        except Exception as e:initialization.append(dict(method=method,error=repr(e)));continue
        initialization.append(dict(method=method,ms=1000*(time.perf_counter()-start)))
        for region in chosen:
            for protocol in ['A','B']:
                try:
                    if torch.cuda.is_available():torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter();r=infer_region(region,method,protocol,model)
                    if torch.cuda.is_available():torch.cuda.synchronize()
                    r['end_to_end_ms_synchronized']=1000*(time.perf_counter()-start);r['GPU_peak_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else 0.
                except Exception as e:r=dict(sample_id=region['sample_id'],method=method,protocol=protocol,decision='runtime_failure',error=repr(e))
                save(out/f"{region['sample_id']}_{protocol}_{method}.json",r);records.append(r);print('SMOKE',r['sample_id'],method,protocol,r.get('decision'),r.get('n_matches'),flush=True)
    jsonl(out/'results.jsonl',records)
    weight_files=list((PROJECT/'cache/torch/hub/checkpoints').glob('*'))
    save(out/'identity.json',dict(selection_sha256=sha(SELECTION),weights={str(p):sha(p) for p in weight_files},initialized=initialization,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',GT_read_by_predictors=False,raw_geography_baseline='exact PixelMap exposed by raw_geography; independent precision unavailable'))
def finalize_existing_smoke():
    import torch
    out=ROOT/'smoke';records=[json.loads(p.read_text()) for p in sorted(out.glob('*_A_*.json'))+sorted(out.glob('*_B_*.json'))]
    if len(records)!=16:raise ValueError('Expected all 16 already-completed smoke calls; no missing calls may be invented')
    jsonl(out/'results.jsonl',records);weight_files=list((PROJECT/'cache/torch/hub/checkpoints').glob('*'))
    save(out/'identity.json',dict(selection_sha256=sha(SELECTION),weights={str(p):sha(p) for p in weight_files},torch=torch.__version__,cuda_build=torch.version.cuda,
        measured_GPU_peak_GiB=max(r.get('GPU_peak_allocated_GiB',0) for r in records),execution_device='CPU' if not any(r.get('GPU_peak_allocated_GiB',0)>0 for r in records) else 'CUDA',
        initialization_ms=None,initialization_timing_not_retained='final summary previously failed due missing jsonl import; completed per-call timing/cache retained',GT_read_by_predictors=False,finalized_without_new_inference=True))
    print('FINALIZED',len(records),'existing smoke calls; no repeat inference',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--finalize-existing-smoke',action='store_true');a=p.parse_args()
    finalize_existing_smoke() if a.finalize_existing_smoke else smoke()
