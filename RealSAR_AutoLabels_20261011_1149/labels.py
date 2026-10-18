"""Multi-dimensional pseudo evidence, no independent reference promotion."""
import sys
sys.dont_write_bytecode=True
import argparse,json,csv
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np,cv2
from PIL import Image,ImageDraw
from common import ROOT,source_rows,save,configure,sha,clean,PROJECT
from geo import load_pair,sample_valid,pixel_grid,regular_grid
sys.path.insert(0,str(PROJECT))
from sarbench.estimate import perspective

def predicted_native(rec,p,mapper):
    if rec.get('candidate_H') is None:return None
    q=perspective(rec['candidate_H'],p)
    return q if rec['protocol']=='native' else mapper(q)

def grade(rec,A,cfg,consensus_families=0):
    reasons=[];general=cfg['B_general'];high=cfg['B_high'];dist=rec.get('distribution',{})
    hull=[dist.get('native_inlier'+str(i),{}).get('hull_fraction',0) for i in [0,1]];cells=[dist.get('native_inlier'+str(i),{}).get('grid_coverage',0)*16 for i in [0,1]]
    med=rec.get('geo_difference_median_px');p90=rec.get('geo_difference_p90_px')
    if not A:reasons.append('not_A_qualified')
    if not rec.get('accepted'):reasons.append('frozen_gate:'+rec.get('reason','missing_prediction'))
    if rec.get('n_native_valid_inliers',0)<general['minimum_native_valid_inliers']:reasons.append('insufficient_native_valid_inliers')
    if rec.get('native_valid_inlier_fraction',0)<general['min_native_valid_fraction']:reasons.append('matches_outside_native_support')
    if min(hull)<general['minimum_hull_both']:reasons.append('limited_native_spatial_support')
    if med is None or p90 is None or med>general['geo_difference_median_max_px'] or p90>general['geo_difference_p90_max_px']:reasons.append('large_or_unknown_geographic_disagreement')
    if reasons:return 'unreliable_candidate',reasons
    strong=rec.get('n_native_valid_inliers',0)>=high['minimum_native_valid_inliers'] and rec.get('inlier_ratio',0)>=high['minimum_inlier_ratio'] and min(hull)>=high['minimum_hull_both'] and min(cells)>=high['minimum_grid_cells_both'] and med<=high['geo_difference_median_max_px'] and p90<=high['geo_difference_p90_max_px'] and consensus_families>=high['minimum_distinct_algorithm_families']
    return ('high_candidate' if strong else 'general_candidate'),([] if strong else ['high_grade_evidence_not_all_met'])

def hull_mask(points,shape):
    m=np.zeros(shape,np.uint8)
    if len(points)>=3:
        hull=cv2.convexHull(np.asarray(points,np.float32));cv2.fillConvexPoly(m,np.rint(hull[:,0]).astype(np.int32),1)
    return m>0

def main(out):
    out=Path(out).resolve();cfg=configure(out);assert not (out/'dataset_index.jsonl').exists();rows=source_rows();records=[json.loads(s) for s in (out/'predictions/all_frozen/records.jsonl').read_text().splitlines()];assert len(records)==1146
    byid=defaultdict(list)
    for r in records:byid[r['sample_id']].append(r)
    geo={r['sample_id']:r for r in json.loads((out/'georeference_correspondences/SUMMARY.json').read_text())['records']};index=[];modelgrades=[]
    for row in rows:
        sid=row['sample_id'];g=geo[sid];folder=out/'pseudo_labels'/sid;folder.mkdir(exist_ok=False);_,valid,images,infos,mapper=load_pair(row);recs=byid[sid];grid=regular_grid(valid[0].shape,16);q=mapper(grid);support=sample_valid(valid[0],grid)&sample_valid(valid[1],q);grid=grid[support]
        consensus={};comparisons=[]
        for i,r in enumerate(recs):
            if grade(r,g['A_qualified'],cfg)[0]=='unreliable_candidate':continue
            peers={r['method']}
            for s in recs:
                if s['method']==r['method'] or grade(s,g['A_qualified'],cfg)[0]=='unreliable_candidate':continue
                a=predicted_native(r,grid,mapper);b=predicted_native(s,grid,mapper);mask=sample_valid(valid[1],a)&sample_valid(valid[1],b);e=np.linalg.norm(a[mask]-b[mask],axis=1)
                p90=float(np.quantile(e,.9)) if len(e)>=16 else None
                comparisons.append({'first':r['protocol']+'/'+r['method'],'second':s['protocol']+'/'+s['method'],'grid_common_count':len(e),'difference_p90_px':p90,'independent_evidence':False})
                if p90 is not None and p90<=cfg['B_high']['cross_family_prediction_p90_max_px']:peers.add(s['method'])
            consensus[r['protocol']+'/'+r['method']]=len(peers)
        for r in recs:
            key=r['protocol']+'/'+r['method'];quality,reasons=grade(r,g['A_qualified'],cfg,consensus.get(key,0));r.update(pseudo_quality=quality,pseudo_rejection_reasons=reasons,agreeing_algorithm_families=consensus.get(key,0));modelgrades.append(r)
        order=cfg['label_priority_within_grade'];ranking={'high_candidate':0,'general_candidate':1,'unreliable_candidate':2};chosen=min(recs,key=lambda r:(ranking[r['pseudo_quality']],order.index(r['protocol']+'/'+r['method'])))
        B=chosen['pseudo_quality']!='unreliable_candidate';points=[];region=np.zeros(valid[0].shape,bool);pseudo=None
        if B:
            cache=np.load(chosen['cache_path']);mask=cache['eligible_inlier_mask'];a,b=cache['points0'][mask],cache['points1_native'][mask]
            points=[{'point_id':f'PSEUDO_{i:04d}','xy0':p.tolist(),'xy1':q.tolist(),'label_type':'pseudo_correspondences','generator':chosen['protocol']+'/'+chosen['method'],'human_verified':False,'static_verified':False,'dynamic_exclusion_verified':False} for i,(p,q) in enumerate(zip(a,b))]
            hull0=hull_mask(a,valid[0].shape);hull1=hull_mask(b,valid[1].shape);dense=pixel_grid(valid[0].shape);pred=predicted_native(chosen,dense,mapper)
            geographic=np.array(Image.open(out/'georeference_correspondences'/sid/'geographic_valid_overlap0.png'))>0
            region=hull0&geographic&valid[0]&sample_valid(valid[1]&hull1,pred).reshape(valid[0].shape)
            if not region.any():B=False;chosen['pseudo_quality']='unreliable_candidate';chosen['pseudo_rejection_reasons'].append('empty_pseudo_support_region');points=[]
        if B:
            transform={'label_type':'pseudo_transform','model':'affine','generator':chosen['protocol']+'/'+chosen['method'],'prediction_frame_H':chosen['candidate_H'],'native_H':chosen['native_candidate_H'],'native_conversion':chosen['native_conversion'],'composition_if_nonlinear':'geographic pixel-map AFTER affine to reference-coarse grid; no native global H claimed','scope':'only pseudo_support0.png region, image correspondences are not verified static/GT','applicability':'2D apparent geometry over matched support; terrain, layover and changed scatter centers can violate a global transform'}
            confidence={k:chosen.get(k) for k in ['n_matches','n_inliers','inlier_ratio','n_native_valid_inliers','native_valid_inlier_fraction','distribution','frozen_geometry','geo_difference_median_px','geo_difference_p90_px','geo_signed_median_displacement_px','native_fit_residual_median_px','native_fit_residual_p90_px','agreeing_algorithm_families','local_fit_residual_distribution_not_GT']}
            pseudo={'sample_id':sid,'label_type':'pseudo_correspondences','independent_GT':False,'quality':chosen['pseudo_quality'],'points':points,'transform':transform,'confidence':confidence,'confidence_is_probability':False,'consensus':comparisons,'generator_record':chosen,'dynamic_exclusion_verified':False}
            save(folder/'pseudo_correspondences.json',pseudo);save(folder/'pseudo_transform.json',transform);Image.fromarray(region.astype(np.uint8)*255).save(folder/'pseudo_support0.png')
            with (folder/'pseudo_correspondences.csv').open('x',newline='') as f:w=csv.writer(f);w.writerow(['point_id','x0','y0','x1','y1','label_type','human_verified']);w.writerows([p['point_id'],*p['xy0'],*p['xy1'],p['label_type'],False] for p in points)
        quality={'sample_id':sid,'A':g['A_qualified'],'B':B,'C':False,'pseudo_quality':chosen['pseudo_quality'] if B else 'unreliable_candidate','candidates':[{k:r.get(k) for k in ['protocol','method','accepted','reason','pseudo_quality','pseudo_rejection_reasons','agreeing_algorithm_families','geo_difference_median_px','geo_difference_p90_px','n_native_valid_inliers']} for r in recs],'algorithm_consensus_not_independent_GT':comparisons}
        save(folder/'quality.json',quality)
        overlap_min=min(s['fraction_of_all_pixels'] for s in g.get('overlap',[{'fraction_of_all_pixels':0}]))
        texture_ok=all(x['low_structure_tile_fraction']<=cfg['weak_candidate_max_low_structure_tile_fraction'] for x in g.get('image_quality',[])) and bool(g.get('image_quality'))
        weak=bool(g['A_qualified'] and overlap_min>=cfg['weak_candidate_min_overlap_both'] and min(g['grid_points_by_view'])>=cfg['weak_candidate_min_grid_points'] and texture_ok)
        result=dict(row,geographic_correspondences_path=str(out/'georeference_correspondences'/sid/'correspondences.json'),geographic_quality_path=str(out/'georeference_correspondences'/sid/'geographic.json'),effective_geographic_overlap=g.get('overlap'),native_scale_geometry=g.get('metric_geometry'),pseudo_correspondences_path=str(folder/'pseudo_correspondences.json') if B else None,pseudo_transform_path=str(folder/'pseudo_transform.json') if B else None,pseudo_support_path=str(folder/'pseudo_support0.png') if B else None,pseudo_generator=chosen['protocol']+'/'+chosen['method'] if B else None,pseudo_quality=quality['pseudo_quality'],quality_report=str(folder/'quality.json'),evidence_level='B' if B else 'A' if g['A_qualified'] else 'failed_geographic_QA',has_A_georeference=g['A_qualified'],has_B_pseudo=B,has_C_independent_reference=False,GT_evidence_source='automatic geographic reference and/or algorithm pseudo only',gt_status='unverified',allow_weak_supervision_candidate=weak,allow_pseudo_supervision_candidate=B,allow_high_confidence_pseudo_candidate=B and chosen['pseudo_quality']=='high_candidate',independent_test_GT=False,training_eligible_now=False,training_use_status='candidate_only_requires_new_site_component_split_and_license_review',confirmation_is_sealed_unbiased_test=False,dynamic_exclusion_verified=False,license_attribution='Umbra CC BY4.0 / Copyright Capella Space CC BY4.0 / Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026 (Indigo research/nonprofit conditions)')
        index.append(clean(result))
        # Pseudo lines use original native coordinates; background always native.
        canvas=Image.new('RGB',(1536,600),'white');d=ImageDraw.Draw(canvas);canvas.paste(Image.fromarray(images[0]).convert('RGB'),(0,70));canvas.paste(Image.fromarray(images[1]).convert('RGB'),(512,70));overlay=np.stack([images[0]]*3,-1);overlay[region,1]=255;canvas.paste(Image.fromarray(overlay),(1024,70));d.text((4,3),f"{sid} {row['sensor']} {row['site_name']} A={g['A_qualified']} B={B} C=False",fill='black');d.text((4,22),f"PSEUDO ONLY, NOT GT: {quality['pseudo_quality']} generator={result['pseudo_generator']} | right=inlier support proxy",fill='black');d.text((4,42),'Moving ships, shadow and layover NOT independently excluded. Consensus/fit residual do not prove accuracy.',fill='black')
        for p in points[::max(1,len(points)//50)]:d.line((p['xy0'][0],p['xy0'][1]+70,p['xy1'][0]+512,p['xy1'][1]+70),fill=(255,180,20),width=1)
        canvas.save(folder/'preview.png')
    (out/'dataset_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in index));fields=list(index[0]);
    with (out/'dataset_index.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in index)
    (out/'quality/model_grades.jsonl').write_text(''.join(json.dumps(clean(r),allow_nan=False)+'\n' for r in modelgrades));save(out/'quality/SUMMARY.json',{'crop_pairs':191,'A':sum(r['has_A_georeference'] for r in index),'B':sum(r['has_B_pseudo'] for r in index),'B_quality':dict(Counter(r['pseudo_quality'] for r in index if r['has_B_pseudo'])),'C':0,'weak_supervision_candidates':sum(r['allow_weak_supervision_candidate'] for r in index),'independent_GT_accuracy_computed':False,'grade_code_sha256':sha(__file__)});print(json.dumps(json.loads((out/'quality/SUMMARY.json').read_text()),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default=str(ROOT));a=p.parse_args();main(a.out)
