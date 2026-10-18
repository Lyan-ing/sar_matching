"""Frozen image inference, separated from independent GT evaluation.

native: only native PNGs enter predictor.
coarse_geo: reference PNG + geographically reprojected moving PNG;
reported as a separate protocol. Errors return to moving native pixel frame.
"""
import argparse,sys,json,hashlib,os,time,subprocess,random,csv
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,cv2
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];PROJECT=ROOT.parent/'MaritimeSARRegistrationPhase2'
sys.path.insert(0,str(ROOT/'annotation_app'));sys.path.insert(0,str(PROJECT))
from store import Store
from quality import quantitative,spatial

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def frozen():
    selection_path=PROJECT/'output/phase25_20261010_0000/development/selection.json';s=json.loads(selection_path.read_text())
    for p,h in s['inference_code_sha256'].items():
        f=Path(p) if Path(p).is_absolute() else PROJECT/p
        if sha(f)!=h:raise ValueError('Frozen code mismatch '+str(f))
    if sha(PROJECT/'configs/phase25.yaml')!=s['config_sha256']:raise ValueError('Frozen configuration changed')
    expected={'sift':'sift_contrast01','orb':'orb_edge16','lightglue':'lightglue_detect1e4'}
    for method,variant in expected.items():
        if any(s['selected'][method][k]!=v for k,v in {'variant':variant,'model':'affine','policy':'full'}.items()):raise ValueError('Frozen parameters changed')
    model=json.loads((PROJECT/'output/phase25_20261010_0000/model_freeze.json').read_text());cache=PROJECT/'cache/LightGlue'
    if subprocess.check_output(['git','-C',str(cache),'rev-parse','HEAD'],text=True).strip()!=model['lightglue_commit']:raise ValueError('LightGlue source commit changed')
    if subprocess.check_output(['git','-C',str(cache),'diff','HEAD'],text=True)!=model['lightglue_diff']:raise ValueError('LightGlue source diff changed')
    for name,h in model['weights_sha256'].items():
        if sha(PROJECT/'cache/torch/hub/checkpoints'/name)!=h:raise ValueError('Weight mismatch')
    return {'selection':s,'selection_sha256':sha(selection_path),'model':model,'runner_sha256':sha(__file__),'git_version':subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip()}

def native_mapping(task):
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(task['reference_native_geotiff']) as a,rasterio.open(task['moving_native_geotiff']) as b:
        A,B,crsa,crsb=a.transform,b.transform,a.crs,b.crs
    def convert(points):
        if not np.isfinite(points).all():return points
        m=np.array([A*(x+.5,y+.5) for x,y in points]);xx,yy=transform(crsa,crsb,m[:,0].tolist(),m[:,1].tolist());return np.array([(~B)*(x,y) for x,y in zip(xx,yy)])-.5
    return convert

def ground_jacobian(task):
    # Only projected meter grids; rotated affine terms retained. Geographic degrees
    # require additional geodetic assumptions and are intentionally omitted.
    import rasterio
    with rasterio.open(task['moving_native_geotiff']) as s:
        if s.crs.is_projected and s.crs.linear_units=='metre':return np.array([[s.transform.a,s.transform.b],[s.transform.d,s.transform.e]])
    return None

def evaluate_saved(root,predictions,out):
    store=Store(root);rows=[];qualified=0
    for r in [json.loads(s) for s in Path(predictions).read_text().splitlines()]:
        task=store.task(r['sample_id']);doc=store.load(r['sample_id']);result=dict(r)
        if doc['gt_quality']['status']=='gt_qualified':
            mapping=native_mapping(task) if r['protocol']=='coarse_geo' else None
            # Candidate H may be retained for diagnostics even if algorithm rejected.
            result.update(quantitative(r['candidate_H'],r['accepted'],doc,task,ground_metric=ground_jacobian(task),prediction_mapping=mapping));qualified+=1
            result['formal_accuracy_eligible']=True
            result['gt_revision']=doc['revision'];result['gt_content_sha256']=doc['content_sha256']
            if not r['output_available']:result['decision']='no_output'
            elif not r['accepted']:result['decision']='rejected'
        else:result.update(formal_accuracy_eligible=False,gt_status=doc['gt_quality']['status'],decision='accepted_unverified' if r['accepted'] else 'rejected' if r['output_available'] else 'no_output')
        rows.append(result)
    out=Path(out);out.mkdir(exist_ok=False)
    save(out/'evaluation.json',{'records':rows,'gt_qualified_prediction_rows':qualified,'formal_accuracy_computed':qualified>0,'sparse_point_accuracy_is_not_dense_synthetic_accuracy':True,'success_definition':'accepted and mean independent held-out moving-native check error <= threshold; threshold requires predeclared GT uncertainty/support','unqualified_pairs_never_in_accuracy_denominator':True})
    flat=[{k:v for k,v in r.items() if k not in ['H','candidate_H','branches','distribution','independent_check_errors']} for r in rows]
    fields=list(dict.fromkeys(k for r in flat for k in r))
    with (out/'per_sample.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields);w.writeheader();w.writerows(flat)
    summary=[]
    for keys in [('method',),('method','sensor'),('method','visual_scene'),('method','look_pair_category'),('method','component')]:
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in keys)].append(r)
        for vals,rs in groups.items():
            item={'group':dict(zip(keys,vals)),'crop_pairs':len(rs),'distinct_acquisition_pairs':len({r['pair_id'] for r in rs}),'output_rate':sum(r['output_available'] for r in rs)/len(rs),'gate_acceptance_rate':sum(r['accepted'] for r in rs)/len(rs),'mean_e2e_ms':float(np.mean([r['end_to_end_ms'] for r in rs])),'reasons':dict(Counter(r['reason'] for r in rs)),'accuracy_rows':sum('error_mean_px' in r for r in rs),'decisions':dict(Counter(r['decision'] for r in rs))}
            errors=[r['error_mean_px'] for r in rs if r.get('error_mean_px') is not None];item['mean_of_pair_check_means_px']=float(np.mean(errors)) if errors else None
            for t in [3,5,10]:
                supported=[r for r in rs if t in r.get('supported_thresholds_px',[])];accepted=[r for r in supported if r['accepted']]
                item[f'sparse_Success@{t}px_denominator']=len(supported);item[f'sparse_Success@{t}px_rate']=sum(r.get(f'sparse_Success@{t}px',False) for r in supported)/len(supported) if supported else None
                item[f'incorrect_acceptance@{t}px_count']=sum(not r.get(f'sparse_Success@{t}px',False) for r in accepted) if supported else None
                item[f'rejected_or_no_output@{t}px_count']=sum(not r['accepted'] for r in supported) if supported else None
            summary.append(item)
    save(out/'stratified.json',summary);return rows

def run(protocol,ids,out):
    import torch
    from sarbench.matchers import LocalFeatureMatcher,SuperPointLightGlue
    from sarbench.estimate import estimate
    from sarbench.reliability import geometry_diagnostics,gate
    provenance=frozen();out=Path(out);out.mkdir(exist_ok=False);(out/'matches').mkdir();(out/'previews').mkdir()
    os.environ['TORCH_HOME']=str(PROJECT/'cache/torch');cv2.setNumThreads(1);torch.set_num_threads(1);random.seed(42);np.random.seed(42);torch.manual_seed(42)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
    sync=lambda:torch.cuda.synchronize() if torch.cuda.is_available() else None
    store=Store(ROOT);tasks=[store.task(sid) for sid in ids];models={};init=[]
    for m in ['sift','orb','lightglue']:
        start=time.perf_counter()
        try:
            models[m]=LocalFeatureMatcher(m,max_features=2048,detector_options={'contrastThreshold':.01} if m=='sift' else {'edgeThreshold':16}) if m!='lightglue' else SuperPointLightGlue(device='auto',max_features=2048,resize=None,detector_threshold=.0001)
            sync();init.append({'method':m,'initialization_ms':(time.perf_counter()-start)*1000})
        except Exception as e:init.append({'method':m,'failure':repr(e)})
    provenance.update(protocol=protocol,seed=42,task_manifest_sha256=sha(ROOT/'annotation_tasks/tasks.jsonl'),selected_ids=ids,initialization=init,timing='measured PNG decode through prediction + robust estimate + frozen full gate; GPU synchronized; initialization/save/GT loading excluded; first_call flagged, no benchmark warmup',dependencies=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True)),python=sys.executable,cuda_available=torch.cuda.is_available(),torch_version=torch.__version__,torch_cuda=torch.version.cuda)
    save(out/'protocol.json',provenance);results=[];calls=Counter()
    for ti,t in enumerate(tasks):
        methods=['sift','orb','lightglue','orb_then_sift'];methods=methods[ti%4:]+methods[:ti%4]
        for method in methods:
            sync();start=time.perf_counter();branches=[];chosen=None
            record={k:t[k] for k in ['sample_id','pair_id','sensor','site_name','visual_scene','component','look_pair_category','geographic_overlap_stratum']};record.update(method=method,protocol=protocol)
            try:
                a=np.array(Image.open(t['image0']));b=np.array(Image.open(t['image1'] if protocol=='native' else t['moving_coarse_png']))
                def execute(m):
                    if m not in models:raise RuntimeError('matcher initialization failed: '+m)
                    cold=calls[m]==0;calls[m]+=1;pred=models[m].predict(a,b);base=estimate(pred,a.shape,model='affine',seed=42);geo=geometry_diagnostics(base,pred,a.shape,b.shape,42) if base.accepted else {};est=gate(base,geo,'full');return {'method':m,'pred':pred,'est':est,'geo':geo,'cold':cold}
                if method=='orb_then_sift':
                    first=execute('orb');branches.append(first);chosen=first
                    if not first['est'].accepted:chosen=execute('sift');branches.append(chosen)
                else:chosen=execute(method);branches.append(chosen)
                sync();elapsed=(time.perf_counter()-start)*1000;est=chosen['est'];pred=chosen['pred']
                dist={}
                for side in [0,1]:
                    for kind,pts in [('keypoints',getattr(pred,'keypoints'+str(side))),('matches',getattr(pred,'points'+str(side)))]:dist[kind+str(side)]=spatial(pts,a.shape if side==0 else b.shape)
                    mask=est.inlier_mask if est.inlier_mask is not None else np.zeros(len(pred.points0),bool);dist['inliers'+str(side)]=spatial(getattr(pred,'points'+str(side))[mask],a.shape if side==0 else b.shape)
                record.update(output_available=est.candidate_H is not None,accepted=est.accepted,reason=est.reason,H=est.H.tolist() if est.H is not None else None,candidate_H=est.candidate_H.tolist() if est.candidate_H is not None else None,n_keypoints0=pred.n_keypoints0,n_keypoints1=pred.n_keypoints1,n_matches=est.n_matches,n_inliers=est.n_inliers,end_to_end_ms=elapsed,cold_first_call=any(x['cold'] for x in branches),fallback_used=len(branches)>1,executed_branch_count=len(branches),distribution=dist,coordinate_frame='reference_native_to_moving_native' if protocol=='native' else 'reference_native_to_moving_coarse_reference_grid',independent_GT_used_for_prediction=False)
                record['branches']=[]
                for bi,branch in enumerate(branches):
                    p=branch['pred'];e=branch['est'];file=out/'matches'/f"{method}_{t['sample_id']}_b{bi}.npz";np.savez_compressed(file,points0=p.points0,points1=p.points1,keypoints0=p.keypoints0,keypoints1=p.keypoints1,inlier_mask=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(p.points0),bool));record['branches'].append({'method':branch['method'],'cache':str(file),'sha256':sha(file),'reason':e.reason,'matches':e.n_matches,'inliers':e.n_inliers})
                canvas=Image.new('RGB',(1024,565),'white');d=ImageDraw.Draw(canvas);canvas.paste(Image.fromarray(a).convert('RGB'),(0,50));canvas.paste(Image.fromarray(b).convert('RGB'),(512,50));d.text((4,4),f"{t['sample_id']} {method} {protocol} gate={est.accepted} {est.reason}",fill='black');d.text((4,22),'NO INDEPENDENT GT. Lines are algorithm suggestions; not verified correspondences.',fill='black')
                for i,(p0,p1) in enumerate(zip(pred.points0[:60],pred.points1[:60])):d.line((p0[0],p0[1]+50,p1[0]+512,p1[1]+50),fill=(255,190,0) if est.inlier_mask is not None and est.inlier_mask[i] else (80,140,240),width=1)
                canvas.save(out/'previews'/f"{method}_{t['sample_id']}.png")
            except Exception as e:
                sync();record.update(output_available=False,accepted=False,reason='runtime_error',error=repr(e),H=None,candidate_H=None,end_to_end_ms=(time.perf_counter()-start)*1000)
            results.append(record);print(protocol,method,t['sample_id'],record['reason'],flush=True)
    (out/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in results));evaluate_saved(ROOT,out/'predictions.jsonl',out/'evaluation')
    save(out/'SUMMARY.json',{'samples':len(tasks),'methods':4,'rows':len(results),'protocol':protocol,'gate_accepted':dict(Counter(r['method'] for r in results if r['accepted'])),'accuracy_not_available_until_independent_review':True,'initialization':init})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('predict');a.add_argument('--protocol',choices=['native','coarse_geo'],required=True);a.add_argument('--out',required=True);a.add_argument('--ids',help='comma separated selected sample IDs; defaults frozen demo5')
    a=sub.add_parser('evaluate');a.add_argument('--predictions',required=True);a.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='predict':run(a.protocol,a.ids.split(',') if a.ids else json.loads((ROOT/'annotation_tasks/demo5.json').read_text()),a.out)
    else:evaluate_saved(ROOT,a.predictions,a.out)
