"""Command line: inspect, benchmark, summarize. No tuning on frozen test."""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, platform, sys, time, traceback, subprocess, importlib.metadata, random
from pathlib import Path
import cv2
import numpy as np
from .data import discover, describe, load_images, load_gt, source_component
from .matchers import make_matcher
from .estimate import estimate
from .evaluate import evaluate
from .reports import save_json, write_csv, read_csv, summarize, bootstrap_group_success, cycle_diagnostics, figures, preview

DEFAULT_ROOT='output/phase1_formal_20261009_1846'
SHARED_PROTOCOL=('model','seed','threads','max_features','ratio','reproj_px','min_matches',
                 'min_inliers','min_inlier_ratio','min_coverage','max_inlier_rmse','min_phase_response')

def _protocol(settings):
    return {k:settings[k] for k in SHARED_PROTOCOL}

def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1<<20),b''):
            h.update(x)
    return h.hexdigest()


def _source_index(path):
    if not path:return {}
    index={}
    for line in Path(path).open(encoding='utf8'):
        if not line.strip():continue
        r=json.loads(line)
        key=(str(r.get('dataset','')),str(r.get('source_id','')))
        comp=source_component(r)
        if comp!='unknown':index[key]=comp
    return index


def _metadata_profile(ref):
    meta=ref.meta
    nested=meta.get('formal',meta.get('simulation',{}))
    if not isinstance(nested,dict):nested={}
    stratum=str(meta.get('stratum',nested.get('stratum',meta.get('stratum_name','')))).lower()
    profile=str(ref.profile if ref.profile!='unknown' else nested.get('profile','unknown')).lower()
    if profile=='unknown':
        profile=next((x for x in ('easy','medium','low','none','dynamic','three')
                      if f'_{x}_' in '_'+stratum+'_'), 'unknown')
    scene=str(ref.scene if ref.scene!='unknown' else nested.get('scene','unknown')).lower()
    if scene=='unknown':
        scene=next((x for x in ('inshore','offshore','background')
                    if f'_{x}_' in '_'+stratum+'_'), 'unknown')
    return profile,scene


def _group_key(r):
    return r.folder.name


def _is_accepted(root):
    return any(root.glob('ACCEPTED_*.json'))


def _overlap_bin(v):
    if v<=1e-9:return 'none'
    if v<.3:return '0-0.3'
    if v<.5:return '0.3-0.5'
    if v<.7:return '0.5-0.7'
    return '0.7-1.0'


def _has_ships(meta,views=None):
    # Annotation identity and original masks are not input to any matcher; this
    # information is stored solely for post-hoc stratification.
    x=meta.get('ship_annotations',[])
    if isinstance(x,list):
        return any(bool(x[i]) for i in (views if views is not None else range(len(x))) if i<len(x))
    return None


def _cycle_report(cycles):
    return {'triangles':len(cycles),'accepted_complete_triangles':sum(c['complete_accepted'] for c in cycles),'entries':cycles}


def _run(args):
    root=Path(args.root).expanduser().resolve()
    if not args.allow_unfrozen and not _is_accepted(root):
        raise RuntimeError(f'Frozen Phase 1 ACCEPTED_*.json receipt not found under {root}. '
                           'Use --allow-unfrozen only for toy/dev experiments.')
    refs=discover(root,limit=args.limit,require_test=not args.allow_non_test)
    if root.name == 'phase1_formal_20261009_1846' and args.limit is None and len(refs) != 900:
        raise ValueError(f'Frozen formal dataset expected 900 pair relations, found {len(refs)}; refusing incomplete benchmark.')
    dataset_summary=describe(refs)
    if dataset_summary['missing_profile'] or dataset_summary['missing_scene']:
        print('WARNING: Phase 1 metadata incomplete: %d/%d profiles and %d/%d scenes unknown. '
              'Run inspect and adapt metadata fields before scientific stratification.' % (
                  dataset_summary['missing_profile'], len(refs),
                  dataset_summary['missing_scene'], len(refs)),file=sys.stderr)
    methods=args.methods
    cv2.setNumThreads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed)
    if 'lightglue' in methods:
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(args.threads)
    # Resolve all dependencies before creating output so missing lightglue does
    # not leave a deceptively partial or nominally "all methods" run.
    matchers={name:make_matcher(name,max_features=args.max_features,ratio=args.ratio,device=args.device)
              for name in methods}
    out=Path(args.out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite prior benchmark: {out}')
    out.mkdir(parents=True)
    (out/'predictions').mkdir()
    config={k:v for k,v in vars(args).items() if k!='func'}
    save_json(out/'settings.json', config)
    source_components=_source_index(args.source_index)
    receipt=next(iter(sorted(root.glob('ACCEPTED_*.json'))),None)
    provenance={'phase1_root':str(root),'receipt_path':str(receipt) if receipt else None,
                'receipt_sha256':_sha(receipt) if receipt else None,
                'source_index_sha256':_sha(args.source_index) if args.source_index else None,
                'partial_run':args.limit is not None,
                'scientific_kind':'2D synthetic SAR proxy, not real multi-angle observations',
                'platform':platform.platform(),'python':sys.version,'opencv':cv2.__version__,
                'numpy':np.__version__,'methods':methods,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    provenance.update(python_executable=sys.executable,conda_env=os.environ.get('CONDA_DEFAULT_ENV'),
                      code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                      code_diff=subprocess.check_output(['git','diff','HEAD','--','sarbench'],text=True),
                      dependencies={d['name']:d['version'] for d in json.loads(subprocess.check_output(
                          [sys.executable,'-m','pip','list','--format=json'],text=True))},
                      manifest_sha256=_sha(root/'manifest.jsonl') if (root/'manifest.jsonl').exists() else None,
                      file_checksums_sha256={p.name:_sha(p) for p in root.glob('file_checksums_*.json')},
                      timing='image decode excluded; feature extraction + matching + RANSAC/gate included; one image-only warmup per method excluded',
                      max_features=args.max_features,opencv_threads=cv2.getNumThreads())
    try:
        import torch
        provenance['torch']=torch.__version__
        provenance['cuda_available']=bool(torch.cuda.is_available())
        provenance['torch_cuda_build']=torch.version.cuda
        if 'lightglue' in methods:
            import lightglue
            lgroot=Path(lightglue.__file__).resolve().parent.parent
            matcher=matchers['lightglue']
            provenance['lightglue']={
                'source_path':str(lgroot),
                'source_commit':subprocess.check_output(['git','-C',str(lgroot),'rev-parse','HEAD'],text=True).strip(),
                'source_diff':subprocess.check_output(['git','-C',str(lgroot),'diff','HEAD'],text=True),
                'device':str(matcher.device),'resize':matcher.resize,
                'extractor_config':vars(matcher.extractor.conf),
                'matcher_config':vars(matcher.matcher.conf),
                'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                'cudnn_benchmark':torch.backends.cudnn.benchmark,
                'cudnn_deterministic':torch.backends.cudnn.deterministic,
                'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                'weights_sha256':{p.name:_sha(p) for p in Path(torch.hub.get_dir()).joinpath('checkpoints').glob('*.pth')},
            }
            if matcher.device.type=='cuda':
                provenance['lightglue']['gpu_name']=torch.cuda.get_device_name(matcher.device)
                provenance['lightglue']['gpu_inventory']=subprocess.check_output(
                    ['nvidia-smi','--query-gpu=index,uuid,name,driver_version,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
    except ImportError:
        provenance['torch']=None
    save_json(out/'provenance.json',provenance)
    # One deterministic image-only warmup. No GT or scores enter it.
    warm0,warm1=load_images(refs[0])
    for name,matcher in matchers.items():
        start=time.perf_counter();matcher.predict(warm0,warm1)
        if name=='lightglue' and matcher.device.type=='cuda':matcher.torch.cuda.synchronize(matcher.device)
        provenance.setdefault('warmup_ms',{})[name]=1000*(time.perf_counter()-start)
    rows=[]; preview_buckets=collections.Counter()
    for idx,ref in enumerate(refs):
        image0,image1=load_images(ref)
        profile,scene=_metadata_profile(ref)
        for name in methods:
            model=matchers[name]
            if name=='lightglue' and model.device.type=='cuda':model.torch.cuda.synchronize(model.device)
            start=time.perf_counter()
            error=None
            pred=None
            try:
                # CRITICAL: prediction first, GT is only loaded afterwards.
                pred=model.predict(image0,image1)
                est=estimate(pred,image0.shape,model=args.model,reproj_px=args.reproj_px,
                             min_matches=args.min_matches,min_inliers=args.min_inliers,
                             min_inlier_ratio=args.min_inlier_ratio,
                             min_coverage=args.min_coverage,max_inlier_rmse=args.max_inlier_rmse,
                             min_phase_response=args.min_phase_response,seed=args.seed+idx)
            except Exception as e:
                error=f'{type(e).__name__}: {e}'
                from .estimate import Estimated
                est=Estimated(None,False,'exception',0,0,0.,None,None,None,
                              'translation' if name=='phase' else args.model)
            if name=='lightglue' and model.device.type=='cuda':model.torch.cuda.synchronize(model.device)
            duration=time.perf_counter()-start
            gt=load_gt(ref)
            ev=evaluate(est,gt,image0.shape)
            component=source_components.get((ref.dataset,ref.source_id),ref.source_component)
            row={
                'id':ref.id,'folder':str(ref.folder),'dataset':ref.dataset,
                'source_id':ref.source_id,'source_group':ref.source_group,
                'source_component':component,'source_split':ref.test_source,
                'profile':profile,'scene':scene,'has_ship':_has_ships(ref.meta,(ref.view_i,ref.view_j)),
                'source_scene':'background' if ref.dataset=='background' else ref.meta.get('scene_type','unknown'),
                'view_i':ref.view_i,'view_j':ref.view_j,
                'num_views':ref.meta['num_views'],'image_height':image0.shape[0],'image_width':image0.shape[1],
                'method':name,'model':est.model_type,
                'accepted':est.accepted,'reason':est.reason,'error':error,
                'n_matches':est.n_matches,'n_inliers':est.n_inliers,
                'n_keypoints0':pred.n_keypoints0 if pred is not None else None,
                'n_keypoints1':pred.n_keypoints1 if pred is not None else None,
                'prediction_reason':pred.reason if pred is not None else 'exception',
                'inlier_ratio':est.inlier_ratio,'inlier_rmse':est.inlier_rmse,
                'coverage':est.coverage,'confidence':est.confidence,
                'runtime_ms':round(1000*duration,3),
                'pred_H':json.dumps(est.H.tolist()) if est.H is not None else '',
                'overlap_bin':_overlap_bin(gt['overlap']),
                **ev,
            }
            rows.append(row)
            if ev['no_overlap']:
                bucket='false_accept' if est.accepted else 'correct_reject'
            elif not est.accepted:
                bucket='positive_rejected'
            else:
                bucket='success' if ev['success_5px'] else 'inaccurate_accept'
            if args.preview_count>0 and preview_buckets[(name,bucket)]<max(1,args.preview_count//5):
                try:
                    preview(ref,est,out/'previews'/f'{name}_{bucket}_{idx:04d}_{ref.view_i}_{ref.view_j}.png',prediction=pred)
                    preview_buckets[(name,bucket)]+=1
                except Exception as e:
                    row['preview_error']=str(e)
        if (idx+1)%args.log_every==0 or idx==len(refs)-1:
            print(f'Progress {idx+1}/{len(refs)} view pair relations',flush=True)
    write_csv(out/'results.csv',rows)
    report=summarize(rows)
    for method in methods:
        part=[r for r in rows if r['method']==method]
        report.setdefault('cluster_bootstrap',{})[method]=bootstrap_group_success(part,seed=args.seed)
    cycles=cycle_diagnostics(rows)
    report['three_view_cycles']=_cycle_report(cycles)
    report['dataset']=describe(refs)
    report['partial_run']=args.limit is not None
    save_json(out/'summary.json',report)
    if cycles:write_csv(out/'cycle_consistency.csv',cycles)
    if not args.no_figures:figures(report,out/'figures')
    provenance['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    provenance['sample_rows']=len(rows)
    provenance['exceptions']=sum(bool(r.get('error')) for r in rows)
    save_json(out/'provenance.json',provenance)
    save_json(out/'RUN_COMPLETE.json', {'relations':len(refs),'methods':methods,'rows':len(rows),
                                        'method_exceptions':provenance['exceptions'],
                                        'results_sha256':_sha(out/'results.csv')})
    print(json.dumps({'out':str(out),'relations':len(refs),'methods':methods,
                      'exceptions':provenance['exceptions'],'overview':report['by_method']},
                     ensure_ascii=False,indent=2))
    if provenance['exceptions']:
        print('WARNING: Some method evaluations raised exceptions. See results.csv; none was silently dropped.',file=sys.stderr)
    return 0


def _inspect(args):
    refs=discover(args.root,limit=args.limit,require_test=not args.allow_non_test)
    report=describe(refs)
    report['first_meta_keys']=sorted(refs[0].meta)
    report['first_pair_keys']=sorted(refs[0].pair_meta)
    report['root_accepted']=_is_accepted(Path(args.root))
    print(json.dumps(report,ensure_ascii=False,indent=2))


def _summarize(args):
    rows=read_csv(args.results)
    summary=summarize(rows)
    for method in sorted(summary.get('by_method',{})):
        summary.setdefault('cluster_bootstrap',{})[method]=bootstrap_group_success(
            [r for r in rows if r['method']==method],seed=args.seed)
    cycles=cycle_diagnostics(rows)
    summary['three_view_cycles']=_cycle_report(cycles)
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    save_json(out/'summary.json',summary)
    if cycles:write_csv(out/'cycle_consistency.csv',cycles)
    if not args.no_figures:figures(summary,out/'figures')
    print(f'Wrote {out}/summary.json')



def _compare(args):
    """Combine independent CPU/GPU runs only when sample IDs and dataset receipts match."""
    paths=[]
    for x in args.runs:
        p=Path(x)
        paths.append(p/'results.csv' if p.is_dir() else p)
    if len(paths)<2:
        raise ValueError('Provide at least two run directories or results.csv paths')
    combined=[]; known=set(); expected_ids=None; receipts=[]; method_models={};protocols=[]
    for p in paths:
        if not p.is_file():raise FileNotFoundError(p)
        source=p.parent
        provenance=source/'provenance.json'
        if not provenance.exists():
            raise FileNotFoundError(f'Run provenance required for safe merging: {provenance}')
        if not (source/'RUN_COMPLETE.json').exists():
            raise RuntimeError(f'Incomplete run cannot be merged: {source}')
        entry=json.loads(provenance.read_text(encoding='utf8'))
        protocols.append(_protocol(json.loads((source/'settings.json').read_text())))
        receipts.append((entry.get('phase1_root'),entry.get('receipt_sha256'),entry.get('partial_run'),entry.get('source_index_sha256'),entry.get('manifest_sha256')))
        rows=read_csv(p)
        receipt_data=json.loads((source/'RUN_COMPLETE.json').read_text())
        if receipt_data.get('results_sha256')!=_sha(p):raise ValueError(f'Result checksum mismatch: {p}')
        ids={r['id'] for r in rows}
        for method in {r['method'] for r in rows}:
            if {r['id'] for r in rows if r['method']==method}!=ids:raise ValueError(f'Incomplete method rows: {method}')
        if expected_ids is None:expected_ids=ids
        elif ids != expected_ids:
            raise ValueError(f'Different sample ID sets across runs; missing={len(expected_ids-ids)}, extra={len(ids-expected_ids)} in {p}')
        for row in rows:
            key=(row['id'],row['method'])
            if key in known:raise ValueError(f'Duplicate prediction for same sample/method: {key}')
            known.add(key)
            if row['method'] != 'phase':
                model=row.get('model','')
                old=method_models.setdefault(row['method'],model)
                if old!=model:raise ValueError(f'Inconsistent geometry model for method {row["method"]}')
            combined.append(row)
    if len(set(receipts))>1:
        raise ValueError(f'Phase1 receipt/root/partial-run mismatch across runs: {receipts}')
    if any(p!=protocols[0] for p in protocols[1:]):
        raise ValueError(f'Shared experimental protocol mismatch: {protocols}')
    out=Path(args.out).expanduser().resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    write_csv(out/'results.csv',combined)
    report=summarize(combined)
    for m in sorted(report['by_method']):
        report.setdefault('cluster_bootstrap',{})[m]=bootstrap_group_success(
            [r for r in combined if r['method']==m],seed=args.seed)
    cycles=cycle_diagnostics(combined)
    report['three_view_cycles']=_cycle_report(cycles)
    report['comparison_sources']=[str(p) for p in paths]
    report['shared_protocol']=protocols[0]
    save_json(out/'summary.json',report)
    if cycles:write_csv(out/'cycle_consistency.csv',cycles)
    if not args.no_figures:figures(report,out/'figures')
    save_json(out/'RUN_COMPLETE.json',{'relations':len(expected_ids),'methods':sorted(report['by_method']),'rows':len(combined),'results_sha256':_sha(out/'results.csv')})
    print(json.dumps({'out':str(out),'methods':sorted(report['by_method']),
                      'relations_per_method':len(expected_ids),'receipt_sha256':receipts[0][1]},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(prog='sarbench',description='Image-only SAR registration baselines on frozen Phase 1 dataset')
    sub=parser.add_subparsers(dest='cmd',required=True)
    ip=sub.add_parser('inspect',help='Inspect the actual Phase 1 format and metadata')
    ip.add_argument('--root',default=DEFAULT_ROOT)
    ip.add_argument('--limit',type=int)
    ip.add_argument('--allow-non-test',action='store_true')
    ip.set_defaults(func=_inspect)
    rp=sub.add_parser('run',help='Run matchers and evaluate predictions after inference')
    rp.add_argument('--root',default=DEFAULT_ROOT)
    rp.add_argument('--out',required=True)
    rp.add_argument('--methods',nargs='+',choices=['sift','orb','phase','lightglue'],default=['sift','orb','phase'])
    rp.add_argument('--model',choices=['homography','affine'],default='homography')
    rp.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    rp.add_argument('--limit',type=int,help='Debug subset only; never present as full benchmark')
    rp.add_argument('--source-index',help='Optional frozen JSONL for authoritative source_component IDs')
    rp.add_argument('--allow-unfrozen',action='store_true',help='Only toy or independent validation experiments')
    rp.add_argument('--allow-non-test',action='store_true',help='Development sets only; never use to tune frozen test')
    rp.add_argument('--max-features',type=int,default=2048)
    rp.add_argument('--ratio',type=float,default=.8)
    rp.add_argument('--reproj-px',type=float,default=3.)
    rp.add_argument('--min-matches',type=int,default=8)
    rp.add_argument('--min-inliers',type=int,default=8)
    rp.add_argument('--min-inlier-ratio',type=float,default=.3)
    rp.add_argument('--min-coverage',type=float,default=1/16)
    rp.add_argument('--max-inlier-rmse',type=float,default=3.)
    rp.add_argument('--min-phase-response',type=float,default=.20)
    rp.add_argument('--seed',type=int,default=42)
    rp.add_argument('--threads',type=int,default=1,help='OpenCV threads (default one for repeatability)')
    rp.add_argument('--preview-count',type=int,default=16)
    rp.add_argument('--log-every',type=int,default=50)
    rp.add_argument('--no-figures',action='store_true')
    rp.set_defaults(func=_run)
    sp=sub.add_parser('summarize',help='Recalculate summaries and plots from recorded prediction CSV')
    sp.add_argument('--results',required=True)
    sp.add_argument('--out',required=True)
    sp.add_argument('--seed',type=int,default=42)
    sp.add_argument('--no-figures',action='store_true')
    sp.set_defaults(func=_summarize)
    cp=sub.add_parser('compare',help='Merge CPU/GPU method runs after validating identical frozen sample IDs')
    cp.add_argument('--runs',nargs='+',required=True,help='At least 2 run directories or result CSV files')
    cp.add_argument('--out',required=True)
    cp.add_argument('--seed',type=int,default=42)
    cp.add_argument('--no-figures',action='store_true')
    cp.set_defaults(func=_compare)
    args=parser.parse_args(argv)
    return args.func(args)

if __name__=='__main__':
    sys.exit(main())
