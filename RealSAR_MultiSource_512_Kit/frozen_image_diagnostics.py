"""Frozen Phase2.5 image-only diagnostics on one native crop per real source pair.

NO GT, masks, map coordinates or true transforms enter prediction. Gate acceptance
and inlier fit error are NOT accuracy. This script cannot compute Success@px.
"""
import argparse,hashlib,json,csv,sys,time,os
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,cv2
from PIL import Image,ImageDraw

def main(run,data,project):
    sys.path.insert(0,str(project));os.environ['TORCH_HOME']=str(project/'cache/torch');os.environ.setdefault('OMP_NUM_THREADS','1')
    from sarbench.matchers import LocalFeatureMatcher,SuperPointLightGlue
    from sarbench.estimate import estimate
    from sarbench.reliability import geometry_diagnostics,gate
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(1)
    selected=project/'output/phase25_20261010_0000/development/selection.json';selection=json.loads(selected.read_text());out=run/'image_only_diagnostics';out.mkdir(exist_ok=False);(out/'matches').mkdir();(out/'previews').mkdir()
    weights={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (project/'cache/torch/hub/checkpoints').glob('*.pth')}
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    sources=[json.loads(x) for x in (data/'candidates.jsonl').read_text().splitlines()];samples={}
    for r in sources:samples.setdefault(r['pair_id'],r)
    protocol={'selection_path':str(selected),'selection_sha256':sha(selected),'weights_sha256':weights,'seed':42,'input':'reference.png and moving_native.png ONLY','evaluation_GT':None,'formal_accuracy_denominator':0,'pairs':len(samples),'init_excluded_from_pair_times':True,'cuda_available':torch.cuda.is_available(),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','core_sha256':{f:sha(project/'sarbench'/f) for f in ['matchers.py','estimate.py','reliability.py']}}
    for name,digest in protocol['core_sha256'].items():
        assert digest==selection['inference_code_sha256']['sarbench/'+name],('Frozen inference code changed',name)
    expected_variants={'sift':'sift_contrast01','orb':'orb_edge16','lightglue':'lightglue_detect1e4'}
    for method,variant in expected_variants.items():
        assert selection['selected'][method]['variant']==variant,('Unsupported frozen feature settings',method)
    protocol['frozen_core_and_variants_verified']=True
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[];init=[];seen_sites=set()
    for method in ['sift','orb','lightglue']:
        assert selection['selected'][method]['model']=='affine' and selection['selected'][method]['policy']=='full'
        start=time.perf_counter()
        try:
            matcher=LocalFeatureMatcher(method,max_features=2048,detector_options={'contrastThreshold':.01} if method=='sift' else {'edgeThreshold':16}) if method!='lightglue' else SuperPointLightGlue(device='auto',max_features=2048,resize=None,detector_threshold=.0001)
            init.append({'method':method,'initialization_ms':1000*(time.perf_counter()-start)})
        except Exception as e:init.append({'method':method,'failure':repr(e)});continue
        for i,r in enumerate(samples.values()):
            attrs=r['source_attributes'];folder=data/r['split']/r['pair_id']/r['chip_id'];start=time.perf_counter()
            record={'method':method,'pair_id':r['pair_id'],'chip_id':r['chip_id'],'sensor':r['sensor'],'site_name':attrs['site_name'],'look_pair_category':attrs['look_pair_category'],'GT_verified':False,'cold_first_call':i==0}
            try:
                if method=='lightglue' and torch.cuda.is_available():torch.cuda.synchronize()
                a=np.array(Image.open(folder/'reference.png'));b=np.array(Image.open(folder/'moving_native.png'))
                pred=matcher.predict(a,b);base=estimate(pred,a.shape,model='affine',seed=42);geo=geometry_diagnostics(base,pred,a.shape,b.shape,42) if base.accepted else {};est=gate(base,geo,'full')
                if method=='lightglue' and torch.cuda.is_available():torch.cuda.synchronize()
                record.update(image_only_accepted=est.accepted,reason=est.reason,keypoints0=pred.n_keypoints0,keypoints1=pred.n_keypoints1,matches=est.n_matches,inliers=est.n_inliers,fit_inlier_rmse_px=est.inlier_rmse if est.inlier_rmse is not None and np.isfinite(est.inlier_rmse) else None,end_to_end_ms=1000*(time.perf_counter()-start),H=est.H.tolist() if est.H is not None else None,geometry=geo)
                np.savez_compressed(out/'matches'/f"{method}_{r['pair_id']}.npz",points0=pred.points0,points1=pred.points1,keypoints0=pred.keypoints0,keypoints1=pred.keypoints1)
                key=(method,attrs['site_name'])
                if key not in seen_sites:
                    seen_sites.add(key);warped=cv2.warpPerspective(b,np.linalg.inv(est.H),(512,512)) if est.H is not None else b
                    canvas=Image.new('RGB',(1536,590),'white');draw=ImageDraw.Draw(canvas);canvas.paste(Image.fromarray(a).convert('RGB'),(0,65));canvas.paste(Image.fromarray(b).convert('RGB'),(512,65));overlay=np.stack([a,warped,warped],-1);canvas.paste(Image.fromarray(overlay),(1024,65));draw.text((5,5),f"{r['sensor']} / {attrs['site_name']} / {method}: image-only gate accepted={est.accepted}, {est.reason}",fill='black');draw.text((5,25),'Reference native / Moving native / Prediction overlay (or unwarped when rejected). NO INDEPENDENT GT; not accuracy.',fill='black');draw.text((5,45),'SAR: Umbra CC BY 4.0; Copyright Capella Space Corp CC BY 4.0; Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery, 2026',fill='black');canvas.save(out/'previews'/f"{method}_{r['pair_id']}.png")
            except Exception as e:record.update(image_only_accepted=False,reason='runtime_error',error=repr(e),end_to_end_ms=1000*(time.perf_counter()-start))
            results.append(record)
            if (i+1)%10==0:print('Diagnostic',method,i+1,'/',len(samples),flush=True)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in results))
    fields=list(dict.fromkeys(k for r in results for k in r if k not in ['H','geometry']))
    with (out/'per_pair_diagnostics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(results)
    summary={'scope':'Qualitative native-image diagnostics; no GT accuracy, training, or tuning','initialization':init,'methods':{},'formal_accuracy_denominator':0}
    for method in ['sift','orb','lightglue']:
        rows=[r for r in results if r['method']==method];times=[r['end_to_end_ms'] for r in rows]
        summary['methods'][method]={'evaluated_source_pairs':len(rows),'image_only_gate_accepted':sum(r['image_only_accepted'] for r in rows),'reasons':dict(Counter(r['reason'] for r in rows)),'mean_ms':float(np.mean(times)) if times else None,'median_ms':float(np.median(times)) if times else None,'note':'Acceptance does not prove correctness; inlier residual is not independent reprojection error.'}
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--data',required=True);p.add_argument('--project',required=True);a=p.parse_args();main(Path(a.run).resolve(),Path(a.data).resolve(),Path(a.project).resolve())
