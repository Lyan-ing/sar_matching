"""All-pair frozen native + geographic-assisted feature diagnostics. NO GT."""
import sys
sys.dont_write_bytecode=True
import argparse,os,json,time,subprocess,importlib.util,random
from pathlib import Path
import numpy as np,cv2
from PIL import Image
from common import ROOT,MANUAL,PROJECT,source_rows,configure,save,sha,clean
from geo import load_pair,sample_valid

def frozen():
    sys.path.insert(0,str(MANUAL/'annotation_app'));sys.path.insert(0,str(PROJECT))
    spec=importlib.util.spec_from_file_location('previous_frozen_checks',MANUAL/'evaluation/run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.frozen()

def main(out):
    out=Path(out).resolve();cfg=configure(out);assert (out/'georeference_correspondences/SUMMARY.json').exists();target=out/'predictions/all_frozen';target.mkdir(exist_ok=False);(target/'matches').mkdir();(target/'records').mkdir()
    os.environ['TORCH_HOME']=str(PROJECT/'cache/torch');freeze=frozen()
    from sarbench.matchers import LocalFeatureMatcher,SuperPointLightGlue
    from sarbench.estimate import estimate,perspective
    from sarbench.reliability import geometry_diagnostics,gate,spatial,projection_diagnostics
    import torch
    torch.set_num_threads(1);cv2.setNumThreads(1);random.seed(cfg['inference_seed']);np.random.seed(cfg['inference_seed']);torch.manual_seed(cfg['inference_seed'])
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg['inference_seed'])
    sync=lambda:torch.cuda.synchronize() if torch.cuda.is_available() else None
    models={};init=[]
    for m in ['sift','orb','lightglue']:
        start=time.perf_counter()
        try:
            models[m]=LocalFeatureMatcher(m,max_features=2048,detector_options={'contrastThreshold':.01} if m=='sift' else {'edgeThreshold':16}) if m!='lightglue' else SuperPointLightGlue(device='auto',max_features=2048,resize=None,detector_threshold=.0001)
            sync();init.append({'method':m,'init_ms':1000*(time.perf_counter()-start)})
        except Exception as e:init.append({'method':m,'error':repr(e)})
    provenance={'frozen':freeze,'config_sha256':sha(out/'config.json'),'runner_sha256':sha(__file__),'python':sys.executable,'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','initialization':init,'seed':cfg['inference_seed'],'methods':['sift','orb','lightglue'],'protocols':['native','geo_assisted'],'primary_model':'affine/full','homography_role':'diagnostic-only same feature matches, no selection or tuning','no_independent_GT_read':True,'timing':'PNG decode to frozen affine gate, GPU synchronized; separate homography diagnostic/caching/map metrics excluded; first call flagged; geo warp cost in geographic stage','dependencies':json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True))}
    save(target/'provenance.json',provenance);rows=source_rows();records=[];calls={m:0 for m in models}
    for i,row in enumerate(rows):
        _,valid,_,_,mapper=load_pair(row);geo_folder=out/'georeference_correspondences'/row['sample_id']
        for protocol in ['native','geo_assisted']:
            order=['sift','orb','lightglue'];order=order[i%3:]+order[:i%3]
            for method in order:
                sid=row['sample_id'];tag=protocol+'_'+method+'_'+sid;start=time.perf_counter();rec={k:row[k] for k in ['sample_id','pair_id','chip_id','sensor','site_name','component','split','scene_class','look_pair_category']};rec.update(protocol=protocol,method=method,label_type='pseudo_correspondences',independent_GT=False)
                try:
                    if method not in models:raise RuntimeError('frozen matcher initialization unavailable')
                    sync();start=time.perf_counter();a=np.array(Image.open(row['reference_native_png']));b=np.array(Image.open(row['moving_native_png'] if protocol=='native' else geo_folder/'moving_native_geo_assisted.png'));cold=calls[method]==0;calls[method]+=1
                    pred=models[method].predict(a,b);base=estimate(pred,a.shape,model='affine',seed=cfg['inference_seed']);diag=geometry_diagnostics(base,pred,a.shape,b.shape,cfg['inference_seed']) if base.accepted else {};est=gate(base,diag,'full');sync();elapsed=1000*(time.perf_counter()-start)
                    diagnostic_start=time.perf_counter();alt=estimate(pred,a.shape,model='homography',seed=cfg['inference_seed']);alt_diag=geometry_diagnostics(alt,pred,a.shape,b.shape,cfg['inference_seed']) if alt.candidate_H is not None else {};alt_filtered=gate(alt,alt_diag,'full');alt_ms=1000*(time.perf_counter()-diagnostic_start)
                    p0=pred.points0.astype(float);p1=pred.points1.astype(float) if protocol=='native' else mapper(pred.points1)
                    inliers=est.inlier_mask if est.inlier_mask is not None else np.zeros(len(p0),bool);native_valid=sample_valid(valid[0],p0)&sample_valid(valid[1],p1);eligible=inliers&native_valid
                    M=mapper.affine_matrix();H=est.candidate_H if protocol=='native' else (M@est.candidate_H if M is not None and est.candidate_H is not None else None)
                    q=mapper(p0);offset=p1-q;geo_error=np.linalg.norm(offset[eligible],axis=1)
                    fit_native=None
                    if est.candidate_H is not None and eligible.any():
                        z=perspective(est.candidate_H,pred.points0[eligible]);z=z if protocol=='native' else mapper(z);fit_native=np.linalg.norm(z-p1[eligible],axis=1)
                    distribution={}
                    for side in [0,1]:
                        distribution['native_inlier'+str(side)]=spatial((p0 if side==0 else p1)[eligible],valid[side].shape)
                        distribution['raw_keypoint'+str(side)]=spatial(getattr(pred,'keypoints'+str(side)),a.shape if side==0 else b.shape)
                    rec.update(output_available=est.candidate_H is not None,accepted=est.accepted,reason=est.reason,n_keypoints0=pred.n_keypoints0,n_keypoints1=pred.n_keypoints1,n_matches=est.n_matches,n_inliers=est.n_inliers,inlier_ratio=est.inlier_ratio,n_native_valid_inliers=int(eligible.sum()),native_valid_inlier_fraction=float(eligible.sum()/max(1,inliers.sum())),candidate_H=est.candidate_H,native_candidate_H=H,coordinate_frame='reference_native_to_moving_native' if protocol=='native' else 'reference_native_to_reference_coarse_grid',native_conversion='exact same-CRS affine composition' if H is not None and protocol=='geo_assisted' else 'CRS composition required' if protocol=='geo_assisted' else 'identity',distribution=distribution,frozen_geometry=diag,fit_inlier_rmse_not_GT=est.inlier_rmse,end_to_end_ms=elapsed,cold_first_call=cold,geo_difference_median_px=float(np.median(geo_error)) if len(geo_error) else None,geo_difference_p90_px=float(np.quantile(geo_error,.9)) if len(geo_error) else None,geo_signed_median_displacement_px=np.median(offset[eligible],axis=0) if eligible.any() else None,native_fit_residual_median_px=float(np.median(fit_native)) if fit_native is not None else None,native_fit_residual_p90_px=float(np.quantile(fit_native,.9)) if fit_native is not None else None,homography_diagnostic={'output_available':alt.candidate_H is not None,'accepted_full_gate':alt_filtered.accepted,'reason':alt_filtered.reason,'inliers':alt.n_inliers,'candidate_H':alt.candidate_H,'geometry':alt_diag,'extra_ms':alt_ms},native_projection_diagnostics=projection_diagnostics(H,a.shape) if H is not None else None)
                    file=target/'matches'/f'{tag}.npz';np.savez_compressed(file,points0=p0,points1_native=p1,points1_prediction_frame=pred.points1,keypoints0=pred.keypoints0,keypoints1_prediction_frame=pred.keypoints1,inlier_mask=inliers,native_valid_mask=native_valid,eligible_inlier_mask=eligible);rec.update(cache_path=str(file),cache_sha256=sha(file))
                    tiles=[]
                    if fit_native is not None:
                        xy=p0[eligible];res=fit_native
                        for gy in range(4):
                            for gx in range(4):
                                t=(xy[:,0]>=gx*128)&(xy[:,0]<(gx+1)*128)&(xy[:,1]>=gy*128)&(xy[:,1]<(gy+1)*128)
                                if t.any():tiles.append({'gx':gx,'gy':gy,'count':int(t.sum()),'resubstitution_median_px':float(np.median(res[t])),'resubstitution_p90_px':float(np.quantile(res[t],.9))})
                    rec['local_fit_residual_distribution_not_GT']=tiles
                except Exception as e:rec.update(accepted=False,output_available=False,reason='runtime_error',error=repr(e),end_to_end_ms=1000*(time.perf_counter()-start))
                save(target/'records'/f'{tag}.json',rec);records.append(clean(rec))
        if (i+1)%12==0:print('Frozen pair inference',i+1,'/',191,flush=True)
    (target/'records.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records));save(target/'SUMMARY.json',{'crop_pairs':191,'rows':len(records),'expected_rows':1146,'initialization':init,'runtime_errors':sum(r['reason']=='runtime_error' for r in records),'no_independent_GT_accuracy':True});print('Inference rows',len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default=str(ROOT));a=p.parse_args();main(a.out)
