"""Independent-check evaluation is separate from prediction; never fits check GT."""
import sys
sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np,cv2
from geo_tools import ROOT,BASE,clean,save,PixelMap,Affine,center_metric
sys.path.insert(0,str(BASE/'RealSAR_IndependentBenchmark_20261011_2015'))
from benchmark_quality import quality,POLICY
sys.path.insert(0,str(BASE/'MaritimeSARRegistrationPhase2'))
from sarbench.estimate import perspective

def qualified_quality(doc,task,root=ROOT):
    """Apply explicit prior geographic-preview exposure without editing evidence."""
    effective=json.loads(json.dumps(doc));events=Path(root)/'readability/assistance_events.jsonl'
    actors=set()
    if events.exists():actors={r['actor'] for r in [json.loads(s) for s in events.read_text().splitlines()] if r['sample_id']==task['sample_id']}
    for p in effective.get('points',[]):
        if p.get('annotator') in actors:p['used_assistance']=True
        if (p.get('review') or {}).get('reviewer') in actors:p['review']['used_assistance']=True
    return quality(effective,task)

def predict_points(pred,xy,views):
    if pred['method']=='georeference':return PixelMap(Affine(*views[0]['native_info']['transform']),views[0]['native_info']['crs'],Affine(*views[1]['native_info']['transform']),views[1]['native_info']['crs'])(xy),np.ones(len(xy),bool)
    result=np.full((len(xy),2),np.nan);support=np.zeros(len(xy),bool)
    if pred.get('local_only'):
        for tile in pred.get('local',[]):
            if not tile.get('accepted') or tile.get('candidate_H') is None:continue
            x0,y0,x1,y1=tile['tile_xyxy'];mask=(xy[:,0]>=x0)&(xy[:,0]<x1)&(xy[:,1]>=y0)&(xy[:,1]<y1)&(~support)
            result[mask]=perspective(tile['candidate_H'],xy[mask]);support[mask]=True
    elif pred.get('accepted') and pred.get('global_H') is not None:
        polygon=np.asarray(pred.get('sampled_support_polygon',[]),np.float32).reshape(-1,2)
        if len(polygon)>=3:support=np.array([cv2.pointPolygonTest(polygon,tuple(map(float,p)),False)>=0 for p in xy])
        result[support]=perspective(pred['global_H'],xy[support])
    return result,support
def evaluate(region,pred,doc,task):
    q=qualified_quality(doc,task)
    if q['status']!='gt_qualified':return dict(sample_id=region['sample_id'],method=pred['method'],independent_accuracy_available=False,reason='independent gt_qualified check evidence required',GT_status=q['status'],Success_computed=False)
    if task['sample_id']!=region['sample_id'] or doc['sample_id']!=region['sample_id']:raise ValueError('GT task is not this native mother region')
    for i in [0,1]:
        z=region['views'][i]['native_info']
        if task['width'+str(i)]!=z['width'] or task['height'+str(i)]!=z['height']:raise ValueError('GT coordinates must use exact V2 native dimensions, not old 512 points')
    eligible={p['point_id'] for p in q['point_evidence'] if p['eligible'] and p['role']=='check'};points=[p for p in doc['points'] if p['point_id'] in eligible]
    xy=np.array([p['xy0'] for p in points]);target=np.array([p['xy1'] for p in points]);output,supported=predict_points(pred,xy,region['views'])
    baseline=PixelMap(Affine(*region['views'][0]['native_info']['transform']),region['views'][0]['native_info']['crs'],Affine(*region['views'][1]['native_info']['transform']),region['views'][1]['native_info']['crs'])(xy)
    errors=np.linalg.norm(output-target,axis=1);geo_errors=np.linalg.norm(baseline-target,axis=1);zs=region['views'][1]['native_info']
    ground=np.full(len(xy),np.nan);good=supported&np.isfinite(output).all(1)
    if good.any():ground[good]=np.linalg.norm(center_metric(zs,output[good],region['metric_crs'])-center_metric(zs,target[good],region['metric_crs']),axis=1)
    geo_ground=np.linalg.norm(center_metric(zs,baseline,region['metric_crs'])-center_metric(zs,target,region['metric_crs']),axis=1)
    thresholds=q['supported_success_thresholds_px'];accepted=bool(pred.get('accepted',False) or pred['method']=='georeference')
    result=dict(sample_id=region['sample_id'],method=pred['method'],GT_status=q['status'],independent_accuracy_available=True,scope=q['scope'],n_checks=len(points),n_supported_checks=int(good.sum()),coverage=float(good.mean()),raw_geography_mean_px=float(geo_errors.mean()),raw_geography_mean_grid_ground_m=float(geo_ground.mean()),
        prediction_mean_px=float(errors[good].mean()) if good.any() else None,prediction_mean_grid_ground_m=float(ground[good].mean()) if good.any() else None,
        georeference_and_algorithm_same_check_ids=[p['point_id'] for p in points],same_supported_subset_geography_mean_px=float(geo_errors[good].mean()) if good.any() else None,
        uncertainty_px=q['max_combined_uncertainty_px'],supported_thresholds_px=thresholds,checks=[dict(point_id=p['point_id'],within_declared_prediction_support=bool(good[i]),error_px=float(errors[i]) if good[i] else None,grid_ground_error_m=float(ground[i]) if good[i] else None,raw_geography_error_px=float(geo_errors[i]),raw_geography_grid_ground_m=float(geo_ground[i])) for i,p in enumerate(points)],
        decision='accepted_partial_unassessable' if accepted and not good.all() else 'accepted_on_checks' if accepted else 'rejected',ground_accuracy_caveat='moving-product metric grid distance, not absolute ground-truth geolocation')
    for threshold in thresholds:
        success=bool(accepted and len(points)>=POLICY['success_min_checks'] and good.all() and float(errors.mean())<=threshold)
        result['sparse_Success@'+str(threshold)+'px']=success
        result['incorrect_accept@'+str(threshold)+'px']=bool(accepted and good.all() and float(errors.mean())>threshold)
    return clean(result)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--region',required=True);p.add_argument('--prediction',required=True);p.add_argument('--annotation',required=True);p.add_argument('--task',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    read=lambda f:json.loads(Path(f).read_text());save(a.out,evaluate(read(a.region),read(a.prediction),read(a.annotation),read(a.task)))
