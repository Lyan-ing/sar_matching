"""Frozen native rotation study. Prediction, geographic QA and evidence remain separate."""
import sys
sys.dont_write_bytecode=True
import os,json,time,random,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np,cv2,rasterio
from PIL import Image
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;OLD=BASE/'RealSAR_AutoLabels_20261011_1149';MANUAL=BASE/'RealSAR_GeometryGT_20261011_0220';PROJECT=BASE/'MaritimeSARRegistrationPhase2'
sys.path.insert(0,str(OLD));sys.path.insert(0,str(PROJECT))
from common import save,sha,clean
from geo import load_pair,PixelMap,sample_valid,overlap,regular_grid
from labels import grade,hull_mask
from infer import frozen
from sarbench.matchers import Prediction,LocalFeatureMatcher,SuperPointLightGlue
from sarbench.estimate import estimate,perspective
from sarbench.reliability import geometry_diagnostics,gate,spatial,projection_diagnostics

def rows():return [json.loads(s) for s in (OLD/'dataset_index.jsonl').read_text().splitlines()]
def cfg():return json.loads((ROOT/'config.json').read_text())
def rotation(degrees,shape=(512,512)):
    """np.rot90 CCW. Coordinates denote pixel centers, integers 0..size-1."""
    h,w=shape
    return {0:[[1,0,0],[0,1,0],[0,0,1]],90:[[0,1,0],[-1,0,w-1],[0,0,1]],180:[[-1,0,w-1],[0,-1,h-1],[0,0,1]],270:[[0,-1,h-1],[1,0,0],[0,0,1]]}[degrees]
def rotate(image,degrees):return np.ascontiguousarray(np.rot90(image,degrees//90))
def restored(points,degrees,shape=(512,512)):return perspective(np.linalg.inv(rotation(degrees,shape)),points)
def restore_H(H,degrees,shape=(512,512)):return None if H is None else np.linalg.inv(rotation(degrees,shape))@np.asarray(H)
def selection(records):
    """Image-only gate, first accepted in preregistered order. No map/B/GT."""
    lookup={r['angle']:r for r in records}
    return next((lookup[a] for a in [0,180,90,270] if a in lookup and lookup[a].get('accepted')),None)
def context(row):
    if not row.get('negative'):
        data=load_pair(row);return data[1],data[4]
    valid=[];transforms=[];crs=[]
    for side in [0,1]:
        with rasterio.open(row['reference_native_geotiff'] if side==0 else row['moving_native_geotiff']) as ds:
            valid.append((np.array(Image.open(row['valid_path'+str(side)]))>0)&(ds.read_masks(1)>0)&np.isfinite(ds.read(1)));transforms.append(ds.transform);crs.append(ds.crs)
    return valid,PixelMap(transforms[0],crs[0],transforms[1],crs[1])

def initialize():
    os.environ['TORCH_HOME']=str(PROJECT/'cache/torch');f=frozen();import torch
    torch.set_num_threads(1);cv2.setNumThreads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    models={'sift':LocalFeatureMatcher('sift',2048,detector_options={'contrastThreshold':.01}),'orb':LocalFeatureMatcher('orb',2048,detector_options={'edgeThreshold':16}),'lightglue':SuperPointLightGlue(device='auto',max_features=2048,resize=None,detector_threshold=.0001)}
    return models,f,torch

def image_predict(model,row,angle,torch):
    # No CRS, masks, B labels or GT enter this function, its gate or selection.
    sync=lambda:torch.cuda.synchronize() if torch.cuda.is_available() else None
    sync();start=time.perf_counter();a=np.array(Image.open(row['reference_native_png']));native=np.array(Image.open(row['moving_native_png']));b=rotate(native,angle);pred=model.predict(a,b)
    initial=estimate(pred,a.shape,model='affine',seed=42);diagnostics=geometry_diagnostics(initial,pred,a.shape,b.shape,42) if initial.accepted else {};est=gate(initial,diagnostics,'full');sync()
    elapsed=1000*(time.perf_counter()-start)
    return pred,est,diagnostics,elapsed

def diagnose(row,method,angle,pred,est,diag,elapsed,valid,mapper,cache_path):
    p0=pred.points0.astype(float);p1=restored(pred.points1,angle);k1=restored(pred.keypoints1,angle);H=restore_H(est.candidate_H,angle);inliers=est.inlier_mask if est.inlier_mask is not None else np.zeros(len(p0),bool);native_valid=sample_valid(valid[0],p0)&sample_valid(valid[1],p1);eligible=inliers&native_valid
    map_difference=np.linalg.norm(p1[eligible]-mapper(p0[eligible]),axis=1);fit=np.linalg.norm(perspective(H,p0[eligible])-p1[eligible],axis=1) if H is not None and eligible.any() else np.array([])
    start=time.perf_counter();alt=estimate(pred,(512,512),model='homography',seed=42);altd=geometry_diagnostics(alt,pred,(512,512),(512,512),42) if alt.candidate_H is not None else {};altgate=gate(alt,altd,'full');altH=restore_H(alt.candidate_H,angle)
    r={k:row.get(k) for k in ['sample_id','pair_id','sensor','site_name','component','scene_class','look_pair_category','split']};r.update(method=method,angle=angle,protocol='native',negative=row.get('negative',False),output_available=est.candidate_H is not None,accepted=est.accepted,reason=est.reason,n_keypoints0=pred.n_keypoints0,n_keypoints1=pred.n_keypoints1,n_matches=est.n_matches,n_inliers=est.n_inliers,inlier_ratio=est.inlier_ratio,n_native_valid_inliers=int(eligible.sum()),native_valid_inlier_fraction=float(eligible.sum()/max(1,inliers.sum())),candidate_H=H,native_candidate_H=H,prediction_frame_H=est.candidate_H,distribution={**{'native_inlier'+str(i):spatial(p,(512,512)) for i,p in enumerate([p0[eligible],p1[eligible]])},**{'native_keypoint'+str(i):spatial(p,(512,512)) for i,p in enumerate([pred.keypoints0,k1])},**{'native_matches'+str(i):spatial(p,(512,512)) for i,p in enumerate([p0,p1])}},frozen_geometry=diag,native_projection_diagnostics=projection_diagnostics(H,(512,512)) if H is not None else None,geo_difference_median_px=float(np.median(map_difference)) if len(map_difference) else None,geo_difference_p90_px=float(np.quantile(map_difference,.9)) if len(map_difference) else None,native_fit_residual_median_px=float(np.median(fit)) if len(fit) else None,native_fit_residual_p90_px=float(np.quantile(fit,.9)) if len(fit) else None,end_to_end_ms=elapsed,homography_diagnostic={'output_available':altH is not None,'accepted_full_gate':altgate.accepted,'reason':altgate.reason,'n_inliers':alt.n_inliers,'candidate_H':altH,'prediction_frame_H':alt.candidate_H,'geometry':altd,'extra_ms':1000*(time.perf_counter()-start)},independent_GT=False)
    tiles=[]
    for gy in range(4):
        for gx in range(4):
            t=(p0[eligible,0]>=gx*128)&(p0[eligible,0]<(gx+1)*128)&(p0[eligible,1]>=gy*128)&(p0[eligible,1]<(gy+1)*128)
            if t.any() and len(fit):tiles.append({'gx':gx,'gy':gy,'count':int(t.sum()),'fit_resubstitution_p90_NOT_GT':float(np.quantile(fit[t],.9))})
    r['local_geometry_diagnostic_NOT_GT']=tiles
    r['pseudo_quality'],r['pseudo_rejection_reasons']=grade(r,not r['negative'],cfg()['B_frozen'],0)
    np.savez_compressed(cache_path,points0=p0,points1_native=p1,points1_rotated=pred.points1,keypoints0=pred.keypoints0,keypoints1_native=k1,inlier_mask=inliers,eligible_inlier_mask=eligible,homography_inlier_mask=alt.inlier_mask if alt.inlier_mask is not None else np.zeros(len(p0),bool))
    r.update(cache_path=str(cache_path),cache_sha256=sha(cache_path),baseline_reused=False)
    return clean(r)

def baseline(row,method,lookup):
    previous=lookup[row['sample_id'],method];assert sha(previous['cache_path'])==previous['cache_sha256']
    r=dict(previous,angle=0,negative=False,baseline_reused=True,baseline_record_sha256=sha(OLD/'predictions/all_frozen/records'/('native_'+method+'_'+row['sample_id']+'.json')))
    r['pseudo_quality'],r['pseudo_rejection_reasons']=grade(r,True,cfg()['B_frozen'],0)
    return r

def run(stage):
    settings=cfg();prereg=json.loads((ROOT/'preregistration.json').read_text());assert prereg['config_sha256']==sha(ROOT/'config.json');assert prereg['old_index_sha256']==sha(OLD/'dataset_index.jsonl')
    models,f,torch=initialize();folder=ROOT/'runs'/stage;folder.mkdir(parents=True,exist_ok=False);(folder/'matches').mkdir();(folder/'records').mkdir();save(folder/'provenance.json',{'frozen':f,'config_sha256':sha(ROOT/'config.json'),'torch':torch.__version__,'torch_cuda':torch.version.cuda,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','seed':42,'GT_read_by_predictor':False,'runner_sha256':sha(__file__)})
    source=rows();lookup={(r['sample_id'],r['method']):r for r in [json.loads(s) for s in (OLD/'quality/model_grades.jsonl').read_text().splitlines()] if r['protocol']=='native'}
    records=[];strategy=[];selected_predictions=[]
    if stage in ['diagnostic8','representative24','full191']:
        ids=settings['diagnostic8'] if stage=='diagnostic8' else settings['representative24'] if stage=='representative24' else [r['sample_id'] for r in source]
        selected=[r for sid in ids for r in source if r['sample_id']==sid]
        prior={}
        if stage!='diagnostic8':
            for previous in ['diagnostic8','representative24']:
                path=ROOT/'runs'/previous/'records.jsonl'
                if path.exists():
                    for r in [json.loads(s) for s in path.read_text().splitlines()]:prior[r['sample_id'],r['method'],r['angle']]=r
        for i,row in enumerate(selected):
            valid,mapper=context(row)
            for angle in [0,90,180,270]:
                for method in ['sift','orb','lightglue']:
                    key=row['sample_id'],method,angle
                    try:
                        if angle==0:r=baseline(row,method,lookup)
                        elif key in prior:r=dict(prior[key],reused_current_study_stage=True)
                        else:
                            pred,est,diag,elapsed=image_predict(models[method],row,angle,torch);cache=folder/'matches'/f'{method}_{angle}_{row["sample_id"]}.npz';r=diagnose(row,method,angle,pred,est,diag,elapsed,valid,mapper,cache)
                    except Exception as e:r=dict(sample_id=row['sample_id'],method=method,angle=angle,protocol='native',accepted=False,output_available=False,reason='runtime_error',error=repr(e),negative=False,pseudo_quality='unreliable_candidate')
                    records.append(r);save(folder/'records'/f'{method}_{angle}_{row["sample_id"]}.json',r)
            if (i+1)%8==0:print(stage,'pairs',i+1,'/',len(selected),flush=True)
    else:
        selected=json.loads((ROOT/'negative_index.json').read_text()) if stage=='negative' else source
        for i,row in enumerate(selected):
            valid,mapper=context(row) if stage=='negative' else (None,None)
            for method in ['sift','orb','lightglue']:
                # Actual single pass B cascade; A is timed at its actual stopping
                # prefix. No summed four-angle-cache estimate or GT/map selection.
                start=time.perf_counter();prefix=[];timeA=None;selectedA=None;selected_raw=None
                for angle in [0,180,90,270]:
                    try:
                        pred,est,diag,elapsed=image_predict(models[method],row,angle,torch)
                        rec={'sample_id':row['sample_id'],'method':method,'angle':angle,'accepted':est.accepted,'output_available':est.candidate_H is not None,'candidate_H':restore_H(est.candidate_H,angle),'reason':est.reason,'n_matches':est.n_matches,'n_inliers':est.n_inliers,'inlier_ratio':est.inlier_ratio,'end_to_end_ms':elapsed}
                    except Exception as e:rec={'sample_id':row['sample_id'],'method':method,'angle':angle,'accepted':False,'output_available':False,'reason':'runtime_error','error':repr(e),'end_to_end_ms':0}
                    prefix.append(rec)
                    now=1000*(time.perf_counter()-start)
                    if rec['accepted'] or angle==180:
                        if timeA is None:timeA=now;selectedA=selection(prefix)
                    if rec['accepted']:
                        selected_raw=(pred,est,diag,elapsed,angle)
                        break
                total=1000*(time.perf_counter()-start);choice=selection(prefix)
                selected_grade=None
                if selected_raw is not None:
                    if valid is None:valid,mapper=context(row)
                    pp,ee,dd,tt,aa=selected_raw;cache=folder/'matches'/f'selected_{method}_{aa}_{row["sample_id"]}.npz';selected_grade=diagnose(row,method,aa,pp,ee,dd,tt,valid,mapper,cache);selected_predictions.append(selected_grade)
                for name,n,time_measured,chosen in [('A',min(2,len(prefix)),timeA,selectedA),('B',len(prefix),total,choice)]:
                    strategy.append(dict(sample_id=row['sample_id'],pair_id=row['pair_id'],sensor=row['sensor'],site_name=row['site_name'],component=row['component'],look_pair_category=row['look_pair_category'],method=method,strategy=name,negative=row.get('negative',False),accepted=chosen is not None,selected_angle=chosen['angle'] if chosen else None,candidate_H=chosen.get('candidate_H') if chosen else None,rotations_executed=n,end_to_end_ms=time_measured,pseudo_general_eligible_post_selection=bool(chosen is not None and selected_grade is not None and selected_grade['pseudo_quality']!='unreliable_candidate'),execution='actual single pass; A time measured at stopping prefix of B, shared computation',stop_rule='first frozen image-only gate accepted',B_or_GT_used_for_selection=False))
                # Negative ablation must retain ALL four angles even if cascade
                # stops; remaining directions are separately timed fresh calls.
                if stage=='negative':
                    for angle in [0,90,180,270]:
                        pred,est,diag,elapsed=image_predict(models[method],row,angle,torch);cache=folder/'matches'/f'{method}_{angle}_{row["sample_id"]}.npz';r=diagnose(row,method,angle,pred,est,diag,elapsed,valid,mapper,cache);records.append(r)
                else:records.extend(dict(r,execution='actual_cascade') for r in prefix)
            if (i+1)%8==0:print(stage,'pairs',i+1,'/',len(selected),flush=True)
    (folder/'records.jsonl').write_text(''.join(json.dumps(clean(r),allow_nan=False)+'\n' for r in records));(folder/'strategies.jsonl').write_text(''.join(json.dumps(clean(r),allow_nan=False)+'\n' for r in strategy));(folder/'selected_predictions.jsonl').write_text(''.join(json.dumps(clean(r),allow_nan=False)+'\n' for r in selected_predictions));save(folder/'SUMMARY.json',{'stage':stage,'pairs':len(selected),'records':len(records),'strategy_rows':len(strategy),'runtime_errors':sum(r.get('reason')=='runtime_error' for r in records),'C':0});print(stage,'complete',len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['diagnostic8','representative24','full191','strategies','negative']);run(p.parse_args().stage)
