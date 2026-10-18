"""One-shot frozen confirmation, with actual decode-to-decision hybrid timing."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import time
from zoneinfo import ZoneInfo
import cv2
import numpy as np
import torch
import yaml
from sarbench.data import discover,load_images,load_gt
from sarbench.estimate import estimate
from sarbench.evaluate import evaluate
from sarbench.reliability import gate,geometry_diagnostics
from sarbench.reports import save_json,write_csv,summarize

spec=importlib.util.spec_from_file_location('frozen_phase25',Path(__file__).with_name('phase25_benchmark.py'))
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
sha=old.checksum


def single_pipeline(variant,matcher,image0,image1,model,policy,seed):
    """This decision path accepts images only; GT/mask loaders are outside it."""
    start=time.perf_counter();pred=old.predict_variant(variant,matcher,image0,image1);torch.cuda.synchronize();match_ms=1000*(time.perf_counter()-start)
    start=time.perf_counter();base=estimate(pred,image0.shape,model=model,seed=seed);fit_ms=1000*(time.perf_counter()-start)
    start=time.perf_counter();geo=geometry_diagnostics(base,pred,image0.shape,image1.shape,seed) if policy!='baseline' and base.accepted else {}
    filtered=gate(base,geo,policy);gate_ms=1000*(time.perf_counter()-start)
    return {'variant':variant,'pred':pred,'base':base,'estimate':filtered,'geo':geo,'matching_ms':match_ms,'fitting_ms':fit_ms,'checking_ms':gate_ms}


def fallback_pipeline(run_orb,run_sift):
    """Execute SIFT exactly when ORB rejects/has no output; never consult GT."""
    first=run_orb()
    if first['estimate'].accepted:return first,[first],False
    second=run_sift();return second,[first,second],True


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--config',default='configs/phase26.yaml');a=p.parse_args()
    run=Path(a.run).resolve();cfg=yaml.safe_load(Path(a.config).read_text());root=run/'confirmation';out=run/'evaluation'
    if out.exists():raise FileExistsError(out)
    registration=json.loads((run/'preregistration.json').read_text());assert registration['config_sha256']==sha(a.config)
    phase25=Path(cfg['phase25_run']);sp=phase25/'development/selection.json';selected=json.loads(sp.read_text());assert sha(sp)==registration['selection_sha256']
    assert selected['immutable_before_validation'] and selected['role']=='dev';assert sha('configs/phase25.yaml')==selected['config_sha256']
    assert all(sha(path)==digest for path,digest in selected['inference_code_sha256'].items())
    model=json.loads((phase25/'model_freeze.json').read_text())
    assert subprocess.check_output(['git','-C','cache/LightGlue','rev-parse','HEAD'],text=True).strip()==model['lightglue_commit']
    assert subprocess.check_output(['git','-C','cache/LightGlue','diff','HEAD'],text=True)==model['lightglue_diff']
    assert all(sha(Path('cache/torch/hub/checkpoints')/name)==digest for name,digest in model['weights_sha256'].items())
    assert json.loads((root/'acceptance.json').read_text())['accepted']
    for record in json.loads((root/'file_checksums.json').read_text()):assert sha(root/record['path'])==record['sha256']
    assert all(sha(path)==digest for path,digest in json.loads((run/'generation_code_sha256.json').read_text()).items())
    sources={json.loads(line)['source_component'] for line in (root/'manifest.jsonl').read_text().splitlines()}
    assert sources<=set(json.loads((phase25/'data_v2/provenance.json').read_text())['role_components']['confirmation'])
    refs=discover(root,require_test=False);assert all(r.test_source=='train' and r.source_component in sources for r in refs)
    metadata={r['sample_id']:r for r in [json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]}
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.manual_seed(cfg['inference_seed']);torch.cuda.manual_seed_all(cfg['inference_seed']);random.seed(cfg['inference_seed']);np.random.seed(cfg['inference_seed']);assert torch.cuda.is_available()
    methods={}
    for algorithm in ['sift','orb','lightglue']:
        s=selected['selected'][algorithm]
        methods[algorithm+'_original']={'variant':algorithm+'_base','model':'homography','policy':'baseline'}
        methods[algorithm+'_improved']={k:s[k] for k in ['variant','model','policy']}
    models={v:old.make_variant(v,2048) for v in sorted({s['variant'] for s in methods.values()})}
    warm=discover(phase25/'data_v2/dev',require_test=False)[0];im0,im1=load_images(warm)
    for variant,matcher in models.items():old.predict_variant(variant,matcher,im0,im1);torch.cuda.synchronize()
    out.mkdir();(out/'matches').mkdir()
    provenance={'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'selection_sha256':sha(sp),'selection':selected,'config':cfg,'config_sha256':sha(a.config),
                'manifest_sha256':sha(root/'manifest.jsonl'),'preregistration_sha256':sha(run/'preregistration.json'),'model':model,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],text=True),
                'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'runner_sha256':sha(__file__),'frozen_inference_sha256':selected['inference_code_sha256'],
                'actual_dependencies':json.loads(subprocess.check_output([__import__('sys').executable,'-m','pip','list','--format=json'],text=True)),
                'timing':cfg['timing'],'execution_order':'Seven primary methods rotate by pair index; each invocation rereads raw images, hybrid decodes once and executes its own matching/estimation/gates. Attribution-only models reuse selected-feature points and are untimed.'}
    save_json(out/'provenance.json',provenance);rows=[];features=[]
    order=list(methods)+['orb_then_sift']
    for idx,ref in enumerate(refs):
        seed=cfg['inference_seed']+idx;m=metadata[ref.folder.name]
        for method in order[idx%len(order):]+order[:idx%len(order)]:
            torch.cuda.synchronize();start=time.perf_counter();image0,image1=load_images(ref);decode_ms=1000*(time.perf_counter()-start)
            def execute(name):
                s=methods[name];return single_pipeline(s['variant'],models[s['variant']],image0,image1,s['model'],s['policy'],seed)
            if method=='orb_then_sift':chosen,branches,fallback=fallback_pipeline(lambda:execute('orb_improved'),lambda:execute('sift_improved'))
            else:chosen=execute(method);branches=[chosen];fallback=False
            torch.cuda.synchronize();elapsed=1000*(time.perf_counter()-start)
            # All GT reading, persistence and attribution follow the completed decision/timer.
            cachepaths=[]
            for bi,branch in enumerate(branches):
                pred=branch['pred'];cache=out/'matches'/f'{method}_{ref.folder.name}_branch{bi}.npz';np.savez_compressed(cache,points0=pred.points0,points1=pred.points1,keypoints0=pred.keypoints0,keypoints1=pred.keypoints1);cachepaths.append({'path':str(cache),'sha256':sha(cache),'variant':branch['variant']})
            pred=chosen['pred'];est=chosen['estimate'];gt=load_gt(ref);diagnostic=old.offline_features(ref,pred,gt)
            geo=chosen['geo'] or geometry_diagnostics(chosen['base'],pred,image0.shape,image1.shape,seed)
            feature={'id':ref.id,'method':method,'algorithm':'hybrid' if method=='orb_then_sift' else method.split('_')[0],'variant':chosen['variant'],'profile':ref.profile,'scene':ref.scene,'source_scene':'background' if ref.dataset=='background' else ref.meta.get('scene_type','unknown'),
                     'dataset':ref.dataset,'source_component':ref.source_component,'source_id':ref.source_id,'source_split':ref.test_source,'texture_stratum':m['texture_stratum'],'evaluation_subset':m['evaluation_subset'],
                     'image_size':image0.shape[0],'folder':str(ref.folder),'has_ship':any(ref.meta['ship_annotations']),'view_i':ref.view_i,'view_j':ref.view_j,'num_views':2,'n_keypoints0':pred.n_keypoints0,'n_keypoints1':pred.n_keypoints1,
                     'cache_file':cachepaths[-1]['path'],'cache_sha256':cachepaths[-1]['sha256'],'executed_branch_caches':json.dumps(cachepaths),'raw_n_matches':len(pred.points0),
                     'overlap_bin':'none' if gt['no_overlap'] else '0-0.3' if gt['overlap']<.3 else '0.3-0.5' if gt['overlap']<.5 else '0.5-0.7' if gt['overlap']<.7 else '0.7-1.0',**diagnostic}
            features.append(feature)
            def record(name,estimate_object,geometry,timing,model_name,policy):
                return {**feature,'method':name,'model':model_name,'policy':policy,'timing_mode':'measured_decode_to_decision' if timing is not None else 'cached_point_attribution_untimed',
                        'end_to_end_ms':timing,'runtime_ms':timing,'decode_ms':decode_ms if timing is not None else None,
                        'detection_matching_ms':sum(b['matching_ms'] for b in branches) if timing is not None else None,'geometry_estimation_ms':sum(b['fitting_ms'] for b in branches) if timing is not None else None,
                        'reliability_check_ms':sum(b['checking_ms'] for b in branches) if timing is not None else None,'fallback_used':fallback,'executed_branches':len(branches),
                        'executed_n_keypoints0':sum(b['pred'].n_keypoints0 for b in branches),'executed_n_keypoints1':sum(b['pred'].n_keypoints1 for b in branches),
                        'selected_from':chosen['variant'],'accepted':estimate_object.accepted,'reason':estimate_object.reason,'error':None,'n_matches':estimate_object.n_matches,'n_inliers':estimate_object.n_inliers,
                        'inlier_ratio':estimate_object.inlier_ratio,'inlier_rmse':estimate_object.inlier_rmse,'coverage':estimate_object.coverage,'candidate_H':json.dumps(estimate_object.candidate_H.tolist()) if estimate_object.candidate_H is not None else '',
                        'pred_H':json.dumps(estimate_object.H.tolist()) if estimate_object.H is not None else '',**geometry,**evaluate(estimate_object,gt,image0.shape)}
            rows.append(record(method,est,geo,elapsed,est.model_type,'full' if method.endswith('improved') or method=='orb_then_sift' else 'baseline'))
            if method.endswith('_improved'):
                algorithm=method.split('_')[0];H=estimate(pred,image0.shape,model='homography',seed=seed)
                rows.append(record(algorithm+'_feature_H',H,geometry_diagnostics(H,pred,image0.shape,image1.shape,seed),None,'homography','baseline'))
                rows.append(record(algorithm+'_feature_A',chosen['base'],geo,None,'affine','baseline'))
        if (idx+1)%32==0:print('Confirmed',idx+1,'/',len(refs),'pairs; frozen methods only',flush=True)
    write_csv(out/'results.csv',rows);write_csv(out/'features.csv',features);save_json(out/'summary.json',summarize(rows))
    save_json(out/'RUN_COMPLETE.json',{'accepted':True,'relations':len(refs),'primary_methods':order,'rows':len(rows),'features':len(features),'exceptions':0,'results_sha256':sha(out/'results.csv'),'features_sha256':sha(out/'features.csv')})
    print(json.dumps({'out':str(out),'relations':len(refs),'rows':len(rows),'features':len(features),'exceptions':0},indent=2))


if __name__=='__main__':main()
