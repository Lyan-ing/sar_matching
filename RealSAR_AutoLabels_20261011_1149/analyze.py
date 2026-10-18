"""Evidence summaries: geographic differences and acceptance are NOT accuracy."""
import sys
sys.dont_write_bytecode=True
import argparse,json,csv
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from PIL import Image,ImageDraw
from common import ROOT,save,clean

def readlines(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def stats(values):
    a=np.array([v for v in values if v is not None and np.isfinite(v)],float)
    return {'n':len(a),'min':float(a.min()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,'p90':float(np.quantile(a,.9)) if len(a) else None,'max':float(a.max()) if len(a) else None}
def count_units(rows):
    return {'crop_pairs':len(rows),'real_acquisition_pairs':len({r['pair_id'] for r in rows}),'acquisitions':len({r[k] for r in rows for k in ['reference_acquisition_id','moving_acquisition_id']}),'sites':len({r['site_name'] for r in rows}),'components':len({r['component'] for r in rows})}
def csvwrite(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def main(out):
    out=Path(out).resolve();rows=readlines(out/'dataset_index.jsonl');models=readlines(out/'quality/model_grades.jsonl');geo=json.loads((out/'georeference_correspondences/SUMMARY.json').read_text())['records'];bygeo={g['sample_id']:g for g in geo};lookup={r['sample_id']:r for r in rows}
    for r in rows:
        overlap=min(v['fraction_of_all_pixels'] for v in bygeo[r['sample_id']]['overlap'])
        r['overlap_bin']='low_<0.5' if overlap<.5 else 'medium_0.5_to_0.8' if overlap<.8 else 'high_>=0.8'
        r['low_structure_proxy']=max(g['low_structure_tile_fraction'] for g in bygeo[r['sample_id']]['image_quality'])>.75
    strata=[]
    for key in ['sensor','site_name','component','scene_class','look_pair_category','overlap_bin','split','low_structure_proxy']:
        buckets=defaultdict(list)
        for r in rows:buckets[str(r[key])].append(r)
        for value,group in sorted(buckets.items()):
            strata.append(dict(stratifier=key,value=value,**count_units(group),A=sum(r['has_A_georeference'] for r in group),B=sum(r['has_B_pseudo'] for r in group),B_high=sum(r['allow_high_confidence_pseudo_candidate'] for r in group),B_general=sum(r['has_B_pseudo'] and not r['allow_high_confidence_pseudo_candidate'] for r in group),A_only=sum(not r['has_B_pseudo'] for r in group),weak_candidates=sum(r['allow_weak_supervision_candidate'] for r in group),C=0,overlap=stats(min(v['fraction_of_all_pixels'] for v in bygeo[r['sample_id']]['overlap']) for r in group)))
    csvwrite(out/'quality/strata.csv',strata);save(out/'quality/strata.json',strata)
    algorithm=[];algorithm_strata=[]
    def model_summary(group):
        return dict(attempts=len(group),outputs=sum(r.get('output_available',False) for r in group),gate_accepted=sum(r.get('accepted',False) for r in group),B_eligible=sum(r['pseudo_quality']!='unreliable_candidate' for r in group),B_high=sum(r['pseudo_quality']=='high_candidate' for r in group),no_output=sum(not r.get('output_available',False) for r in group),output_rejected=sum(r.get('output_available',False) and not r.get('accepted',False) for r in group),runtime_errors=sum(r['reason']=='runtime_error' for r in group),gate_reasons=dict(Counter(r['reason'] for r in group)),pseudo_reasons=dict(Counter(x for r in group for x in r['pseudo_rejection_reasons'])),keypoints0=stats(r.get('n_keypoints0') for r in group),keypoints1=stats(r.get('n_keypoints1') for r in group),matches=stats(r.get('n_matches') for r in group),native_valid_inliers=stats(r.get('n_native_valid_inliers') for r in group),time_ms=stats(r.get('end_to_end_ms') for r in group),total_primary_s=sum(r.get('end_to_end_ms',0) for r in group)/1000,homography_extra_s=sum(r.get('homography_diagnostic',{}).get('extra_ms',0) for r in group)/1000,homography_gate_accepted=sum(r.get('homography_diagnostic',{}).get('accepted_full_gate',False) for r in group),accepted_geo_difference_median_px=stats(r.get('geo_difference_median_px') for r in group if r.get('accepted')),accepted_geo_difference_p90_px=stats(r.get('geo_difference_p90_px') for r in group if r.get('accepted')),accepted_native_inlier_hull0=stats(r.get('distribution',{}).get('native_inlier0',{}).get('hull_fraction') for r in group if r.get('accepted')),accepted_fit_resubstitution_p90_px_NOT_GT=stats(r.get('native_fit_residual_p90_px') for r in group if r.get('accepted')))
    for protocol in ['native','geo_assisted']:
        for method in ['sift','orb','lightglue']:
            group=[r for r in models if r['protocol']==protocol and r['method']==method];algorithm.append(dict(protocol=protocol,method=method,**model_summary(group)))
            for key in ['sensor','look_pair_category','overlap_bin','scene_class']:
                buckets=defaultdict(list)
                for r in group:buckets[str(lookup[r['sample_id']][key])].append(r)
                for value,b in sorted(buckets.items()):algorithm_strata.append(dict(protocol=protocol,method=method,stratifier=key,value=value,**model_summary(b)))
    save(out/'quality/algorithm_summary.json',algorithm);csvwrite(out/'quality/algorithm_summary.csv',algorithm);save(out/'quality/algorithm_strata.json',algorithm_strata);csvwrite(out/'quality/algorithm_strata.csv',algorithm_strata)
    # One row per true acquisition pair, never pretend chips are independent.
    pair_groups=defaultdict(list)
    for r in rows:pair_groups[r['pair_id']].append(r)
    pairs=[dict(pair_id=k,sensor=g[0]['sensor'],site=g[0]['site_name'],component=g[0]['component'],look=g[0]['look_pair_category'],**count_units(g),B_chips=sum(r['has_B_pseudo'] for r in g),B_high_chips=sum(r['allow_high_confidence_pseudo_candidate'] for r in g),C=0) for k,g in sorted(pair_groups.items())]
    csvwrite(out/'quality/acquisition_pairs.csv',pairs)
    features=[]
    for r in models:
        features.append({k:r.get(k) for k in ['sample_id','pair_id','sensor','site_name','component','protocol','method','output_available','accepted','reason','n_keypoints0','n_keypoints1','n_matches','n_inliers','inlier_ratio','n_native_valid_inliers','native_valid_inlier_fraction','geo_difference_median_px','geo_difference_p90_px','native_fit_residual_p90_px','end_to_end_ms','pseudo_quality','pseudo_rejection_reasons','agreeing_algorithm_families','distribution','native_projection_diagnostics','homography_diagnostic']})
    csvwrite(out/'quality/feature_diagnostics.csv',features)
    scale=[]
    for sensor in sorted({r['sensor'] for r in rows}):
        g=[bygeo[r['sample_id']] for r in rows if r['sensor']==sensor]
        scale.append(dict(sensor=sensor,spacing_x_m=stats(i['spacing_m_x'] for x in g for i in x['metric_geometry']),spacing_y_m=stats(i['spacing_m_y'] for x in g for i in x['metric_geometry']),coverage_edge_m=stats(e for x in g for i in x['metric_geometry'] for e in i['coverage_edge_lengths_m']),within_pair_scale_ratio=stats(e for x in g for e in x['scale_ratio_0_over_1']),different_crs=sum(not x['crs_equal'] for x in g),geographic_generation_ms=stats(x['runtime_ms'] for x in g),note='Grid spacing derived from geocoding; not physical sensor resolution'))
    save(out/'quality/scale.json',scale);csvwrite(out/'quality/scale.csv',scale)
    conditions=[]
    for r in rows:
        for side,q in enumerate(bygeo[r['sample_id']]['image_quality']):
            conditions.append(dict(sample_id=r['sample_id'],sensor=r['sensor'],site=r['site_name'],side=side,**{k:v for k,v in q.items() if k not in ['tiles','interpretation']}))
    csvwrite(out/'quality/image_conditions.csv',conditions)
    candidates={name:count_units([r for r in rows if r[key]]) for name,key in [('weak','allow_weak_supervision_candidate'),('pseudo','has_B_pseudo'),('high_pseudo','allow_high_confidence_pseudo_candidate')]}
    summary=dict(**count_units(rows),A=sum(r['has_A_georeference'] for r in rows),B=sum(r['has_B_pseudo'] for r in rows),B_high=sum(r['allow_high_confidence_pseudo_candidate'] for r in rows),C=0,candidates=candidates,overlap_min_both=stats(min(v['fraction_of_all_pixels'] for v in g['overlap']) for g in geo),geographic_point_count=sum(sum(g['grid_points_by_view']) for g in geo),grid_count_per_view=stats(n for g in geo for n in g['grid_points_by_view']),roundtrip_max_px=max(s['max_px'] for g in geo for s in g['roundtrip']),overlap_ge_0_5_both=sum(min(v['fraction_of_all_pixels'] for v in g['overlap'])>=.5 for g in geo),selected_generators=dict(Counter(r['pseudo_generator'] for r in rows if r['has_B_pseudo'])),no_accuracy_computed=True,independent_GT_points=0,training_started=False)
    save(out/'quality/analysis_summary.json',summary)
    # Reuse all per-pair previews, plus a compact deliberate diagnostic selection.
    diagnostic=json.loads((out/'annotation_adapter/diagnostic_tasks.json').read_text());selected=[lookup[t['sample_id']] for t in diagnostic]
    for ordered in [sorted(rows,key=lambda r:min(v['fraction_of_all_pixels'] for v in bygeo[r['sample_id']]['overlap'])),sorted(rows,key=lambda r:max(q['low_structure_tile_fraction'] for q in bygeo[r['sample_id']]['image_quality']),reverse=True)]:
        r=ordered[0]
        if r not in selected:selected.append(r)
    manifest=[];sheet=Image.new('RGB',(1024,420*len(selected)),'white');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(selected):
        sid=r['sample_id'];image=Image.open(out/'pseudo_labels'/sid/'preview.png').convert('RGB');image.thumbnail((1024,400));sheet.paste(image,(0,i*420+20));draw.text((4,i*420),sid+' '+r['pseudo_quality']+' (diagnostic, NOT GT)',fill='black');manifest.append({'sample_id':sid,'geographic_preview':str(out/'georeference_correspondences'/sid/'preview.png'),'pseudo_preview':str(out/'pseudo_labels'/sid/'preview.png'),'B':r['has_B_pseudo'],'purpose':'deliberate diagnostic, not unbiased accuracy evaluation'})
    sheet.save(out/'visualization/diagnostic_contact_sheet.png');save(out/'visualization/manifest.json',manifest)
    from visualize_bidirectional import main as bidirectional
    bidirectional(out)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default=str(ROOT));a=p.parse_args();main(a.out)
