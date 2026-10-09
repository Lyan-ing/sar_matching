"""Per-pair CSV, stratified summaries, group-aware descriptive uncertainty, and preview figures."""
from __future__ import annotations
from collections import defaultdict,Counter
from pathlib import Path
import csv, json
import numpy as np
import cv2


def save_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        raise ValueError('No rows')
    keys=list(dict.fromkeys(k for r in rows for k in r.keys()))
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline='',encoding='utf8') as f:
        return list(csv.DictReader(f))


def _float(r,k):
    try:
        x=float(r[k]);return x if np.isfinite(x) else None
    except (KeyError,ValueError,TypeError):
        return None


def _truth(v):
    if isinstance(v,bool): return v
    return str(v).lower() in ('true','1','yes')


def _stats(rows):
    positive=[r for r in rows if not _truth(r['no_overlap'])]
    negative=[r for r in rows if _truth(r['no_overlap'])]
    valid=[_float(r,'epe_mean') for r in positive if _float(r,'epe_mean') is not None]
    normalized=[_float(r,'epe_normalized') for r in positive if _float(r,'epe_normalized') is not None]
    runtime=[_float(r,'runtime_ms') for r in rows if _float(r,'runtime_ms') is not None]
    reasons=dict(Counter(str(r.get('reason','unknown')) for r in rows))
    return {
        'relations':len(rows), 'positive':len(positive), 'negative':len(negative),
        'accepted_positive':sum(_truth(r['accepted']) for r in positive),
        'acceptance_positive_rate':sum(_truth(r['accepted']) for r in positive)/len(positive) if positive else None,
        'success_3px_rate':sum(_truth(r.get('success_3px')) for r in positive)/len(positive) if positive else None,
        'success_5px_rate':sum(_truth(r.get('success_5px')) for r in positive)/len(positive) if positive else None,
        'success_10px_rate':sum(_truth(r.get('success_10px')) for r in positive)/len(positive) if positive else None,
        'mean_epe_on_accepted_estimates':float(np.mean(valid)) if valid else None,
        'median_epe_on_accepted_estimates':float(np.median(valid)) if valid else None,
        'mean_normalized_epe_on_accepted_estimates':float(np.mean(normalized)) if normalized else None,
        'epe_evaluable_accepted_count':len(valid),
        'acceptance_all_rate':sum(_truth(r['accepted']) for r in rows)/len(rows) if rows else None,
        'failure_5px_positive_rate':1-sum(_truth(r.get('success_5px')) for r in positive)/len(positive) if positive else None,
        'decision_counts':dict(Counter(r.get('decision_status','unknown') for r in rows)),
        'negative_no_output_count':sum(r.get('decision_status')=='no_output' for r in negative),
        'negative_rejected_estimate_count':sum(r.get('decision_status')=='rejected' for r in negative),
        'runtime_ms_mean':float(np.mean(runtime)) if runtime else None,
        'runtime_ms_median':float(np.median(runtime)) if runtime else None,
        'runtime_ms_p95':float(np.percentile(runtime,95)) if runtime else None,
        'reject_reasons':reasons,
        'false_accept_count':sum(_truth(r['accepted']) for r in negative),
        'false_accept_rate':sum(_truth(r['accepted']) for r in negative)/len(negative) if negative else None,
        'avg_matches':float(np.mean([_float(r,'n_matches') or 0 for r in rows])) if rows else None,
        'avg_inliers':float(np.mean([_float(r,'n_inliers') or 0 for r in rows])) if rows else None,
        'avg_keypoints0':float(np.mean([_float(r,'n_keypoints0') for r in rows if _float(r,'n_keypoints0') is not None])) if any(_float(r,'n_keypoints0') is not None for r in rows) else None,
        'avg_keypoints1':float(np.mean([_float(r,'n_keypoints1') for r in rows if _float(r,'n_keypoints1') is not None])) if any(_float(r,'n_keypoints1') is not None for r in rows) else None,
        'exceptions':sum(bool(r.get('error')) for r in rows),
    }


def summarize(rows):
    if not rows:return {}
    report={'overall':_stats(rows),'by_method':{},'by_method_and_profile':{},'by_method_and_scene':{},
            'by_method_and_dataset':{},'by_method_and_overlap_bin':{},'by_method_and_ship':{},'by_method_and_source_scene':{},
            'dependency_note':'Rows from shared source groups are correlated; do not treat pair-level CI as independent acquisitions.'}
    methods=sorted({r['method'] for r in rows})
    for method in methods:
        mr=[r for r in rows if r['method']==method]
        report['by_method'][method]=_stats(mr)
        for field,outname in [('profile','by_method_and_profile'),('scene','by_method_and_scene'),
                               ('dataset','by_method_and_dataset'),('overlap_bin','by_method_and_overlap_bin'),
                               ('has_ship','by_method_and_ship'),('source_scene','by_method_and_source_scene')]:
            for val in sorted({str(r.get(field,'unknown')) for r in mr}):
                sub=[r for r in mr if str(r.get(field,'unknown'))==val]
                report[outname][f'{method}|{val}']=_stats(sub)
    return report


def bootstrap_group_success(rows, key='source_component', n_boot=1000, seed=42):
    """Descriptive cluster bootstrap only when genuine component IDs are available.
    Pair-level bootstrap deliberately disallowed to prevent pseudo-replication.
    """
    groups=defaultdict(list)
    for r in rows:
        if _truth(r['no_overlap']):continue
        val=r.get(key)
        if val is None or str(val).lower() in ('unknown','none',''):
            return {'available':False,'reason':'Missing source component identifiers; no pair-independent CI.'}
        namespace='hrsid' if r.get('dataset')=='background' else str(r.get('dataset',''))
        groups[(namespace,str(val))].append(float(_truth(r.get('success_5px'))))
    if len(groups)<5:
        return {'available':False,'reason':'Fewer than 5 identifiable source components.'}
    rng=np.random.default_rng(seed)
    cluster_means=np.array([np.mean(v) for v in groups.values()])
    cluster_sizes=np.array([len(v) for v in groups.values()])
    ks=len(cluster_means)
    vals=[]
    for _ in range(n_boot):
        ix=rng.integers(0,ks,size=ks)
        vals.append(float(np.average(cluster_means[ix],weights=cluster_sizes[ix])))
    return {'available':True,'source_components':ks,
            'point_estimate_component_weighted':float(np.mean(cluster_means)),
            'point_estimate_pair_weighted':float(np.average(cluster_means,weights=cluster_sizes)),
            'ci95_cluster_bootstrap':list(map(float,np.percentile(vals,[2.5,97.5]))),
            'note':'Exploratory cluster bootstrap; unidentified source relationships may remain.'}


def cycle_diagnostics(rows):
    """Independent 0->1, 1->2, 0->2 estimates; compose on view 0 coordinates."""
    groups=defaultdict(dict)
    for r in rows:
        groups[(r['method'],r['folder'])][f"{r['view_i']}-{r['view_j']}"]=r
    out=[]
    for (method,folder),d in groups.items():
        keys=('0-1','0-2','1-2')
        if not all(k in d for k in keys) and not any(str(r.get('num_views'))=='3' for r in d.values()):continue
        complete=all(k in d and _truth(d[k].get('accepted')) and d[k].get('pred_H') for k in keys)
        record={'method':method,'folder':folder,'complete_accepted':bool(complete),'cycle_mean_px':None,'cycle_max_px':None,
                'all_edges_gt_success_5px':all(_truth(d[k].get('success_5px')) for k in keys) if all(k in d for k in keys) else False,
                'reason':'edge_rejected_or_missing'}
        if not complete:
            out.append(record);continue
        try:
            H01=np.asarray(json.loads(d['0-1']['pred_H']),float)
            H02=np.asarray(json.loads(d['0-2']['pred_H']),float)
            H12=np.asarray(json.loads(d['1-2']['pred_H']),float)
            C=np.linalg.inv(H02)@H12@H01
            C/=C[2,2]
            # Observe spatially meaningful cycle error, not matrix coefficient norm.
            from .estimate import perspective
            w=float(d['0-1'].get('image_width',129))-1;h=float(d['0-1'].get('image_height',129))-1
            x,y=np.meshgrid(np.linspace(0,w,5),np.linspace(0,h,5));p=np.column_stack([x.ravel(),y.ravel()])
            error=np.linalg.norm(perspective(C,p)-p,axis=1)
            if np.isfinite(error).all():record.update(cycle_mean_px=float(np.mean(error)),cycle_max_px=float(np.max(error)),reason='ok')
            else:record['reason']='nonfinite_cycle_projection'
        except (ValueError,KeyError,np.linalg.LinAlgError):
            record['reason']='invalid_cycle_matrix'
        out.append(record)
    return out


def preview(ref, est, dest, prediction=None):
    from .data import load_images
    img0,img1=load_images(ref)
    if est.H is None:
        H=np.eye(3)
    else:
        H=est.H
    h0,w0=img0.shape
    warped=cv2.warpPerspective(img1,np.linalg.inv(H),(w0,h0)) if est.H is not None else np.zeros_like(img0)
    panel=np.stack([warped,warped,img0],axis=2)  # BGR: cyan=warped view1; red=view0
    col0=cv2.cvtColor(img0,cv2.COLOR_GRAY2BGR)
    col1=cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    def fit(img):
        return cv2.resize(img,(w0,h0),interpolation=cv2.INTER_NEAREST) if img.shape[:2]!=(h0,w0) else img
    panel=np.concatenate([col0,fit(col1),panel],axis=1)
    if prediction is not None and len(prediction.points0) and len(prediction.points1):
        # Show a deterministic subset of image-only matches. If image dimensions
        # differ, draw view 1 endpoints using the displayed resizing ratio only.
        sx=w0/max(img1.shape[1],1); sy=h0/max(img1.shape[0],1)
        for p0,p1 in zip(prediction.points0[:60],prediction.points1[:60]):
            q0=(int(round(p0[0])),int(round(p0[1])))
            q1=(int(round(w0+p1[0]*sx)),int(round(p1[1]*sy)))
            if all(-2*v <= x <= 3*v for x,v in zip(q0,(w0,h0))) and all(-2*v<=x<=4*v for x,v in zip(q1,(w0,h0))):
                cv2.line(panel,q0,q1,(80,200,100),1,cv2.LINE_AA)
    label=f'{ref.id} {est.model_type} {"ACCEPT" if est.accepted else "REJECT"} {est.reason}'
    cv2.putText(panel,label[:90],(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(60,255,60),1,cv2.LINE_AA)
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(dest),panel):
        raise IOError(dest)


def figures(report,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    methods=list(report['by_method'])
    if not methods:return
    fig,ax=plt.subplots(figsize=(7,4))
    rates=[report['by_method'][m]['success_5px_rate'] for m in methods]
    ax.bar(methods,[100*v if v is not None else 0. for v in rates]);ax.set_ylim(0,100)
    ax.set_ylabel('Success @ 5 px (% of positive pairs)');ax.set_title('Frozen SAR proxy benchmark')
    fig.tight_layout();fig.savefig(out/'success5.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    fpr=[report['by_method'][m]['false_accept_rate'] for m in methods]
    ax.bar(methods,[100*v if v is not None else 0. for v in fpr]);ax.set_ylim(0,100)
    ax.set_ylabel('False acceptance (% of no-overlap pairs)');ax.set_title('No-overlap rejection')
    fig.tight_layout();fig.savefig(out/'no_overlap_fpr.png',dpi=160);plt.close(fig)
