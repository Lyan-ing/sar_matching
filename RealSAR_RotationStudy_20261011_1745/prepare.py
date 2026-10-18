"""Fix metadata-only negative samples and protocol BEFORE rotation inference."""
import sys
sys.dont_write_bytecode=True
import json,itertools
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict
from study import ROOT,OLD,MANUAL,rows,context,overlap,save,sha,frozen

def main():
    source=rows();diagnostic=[r['sample_id'] for r in json.loads((OLD/'annotation_adapter/diagnostic_tasks.json').read_text())];representative=[json.loads(s)['sample_id'] for s in (MANUAL/'annotation_tasks/tasks.jsonl').read_text().splitlines()]
    configs={'seed':261011,'inference_seed':42,'angles_ccw':[0,90,180,270],'strategy_order':[0,180,90,270],'strategy_A_order':[0,180],'selection_rule':'first original image-only Affine/full gate accepted; no geographic/B/GT selection','strategy_execution':'A is measured actual stopping prefix of a single B cascade; do not sum cached times','diagnostic8':diagnostic,'representative24':representative,'negative_count_per_provider':10,'negative_rule':'different geographic sites and acquisitions; zero dense valid geographic overlap both ways; no imagery/algorithm result used for selection','B_frozen':json.loads((OLD/'config.json').read_text()),'optional_geo_rotations':False,'geo_analysis':'reuse all 573 original geo_assisted 0deg records; native new rotation study primary','failure_detection_threshold':8,'failure_local_cluster_hull':.01,'feature_change_ratio':2,'feature_change_absolute_matches':8,'contradictory_prediction_grid_p90_px':10,'local_affine_diagnostic':'4x4 tiles, >=8 original matches; frozen estimator; self-fit only, never GT or pseudo-label selector','no_independent_GT':True,'exploratory_previously_analyzed_data':True}
    save(ROOT/'config.json',configs);negative=[]
    for provider in ['Capella','Sentinel-1','Umbra']:
        group=[r for r in source if r['sensor']==provider];candidates=[];seen=set()
        for a,b in itertools.product(group,group):
            if a['site_name']==b['site_name'] or a['reference_acquisition_id']==b['moving_acquisition_id']:continue
            key=(a['reference_acquisition_id'],b['moving_acquisition_id'])
            if key in seen:continue
            seen.add(key);candidates.append((a,b))
        # Interleave geographic site combinations rather than ten chips at one place.
        buckets=defaultdict(list)
        for a,b in candidates:buckets[a['site_name']+'->'+b['site_name']].append((a,b))
        ordered=[]
        while any(buckets.values()):
            for key in sorted(buckets):
                if buckets[key]:ordered.append(buckets[key].pop(0))
        count=0
        for a,b in ordered:
            sid=f'NEG_{provider.replace("-","")}_{count:03d}';r=dict(a,sample_id=sid,pair_id=sid,negative=True,moving_native_png=b['moving_native_png'],moving_native_geotiff=b['moving_native_geotiff'],moving_acquisition_id=b['moving_acquisition_id'],moving_utc=b['moving_utc'],valid_path0=str(__import__('pathlib').Path(a['folder'])/'reference_valid.png'),valid_path1=str(__import__('pathlib').Path(b['folder'])/'moving_native_valid.png'),reference_source_sample=a['sample_id'],moving_source_sample=b['sample_id'],site_name=a['site_name']+' -> '+b['site_name'],component=a['component']+' -> '+b['component'],look_pair_category='geographic_nonoverlap_negative')
            for key,value in b.items():
                if key.startswith('moving_') or key.startswith('native_moving_') or key=='provider_metadata_moving':r[key]=value
            valid,mapper=context(r);_,_,support=overlap(valid,mapper)
            if any(s['joint_pixels'] for s in support):continue
            r['negative_evidence']={'joint_valid_pixels':[s['joint_pixels'] for s in support],'CRS_A':str(mapper.crsA),'CRS_B':str(mapper.crsB),'reference_tif_sha256':sha(r['reference_native_geotiff']),'moving_tif_sha256':sha(r['moving_native_geotiff']),'claim':'zero common valid geographic support, not a low-feature positive'};negative.append(r);count+=1
            if count==10:break
        assert count==10,(provider,count)
    save(ROOT/'negative_index.json',negative)
    freeze=frozen();save(ROOT/'preregistration.json',{'fixed_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'config_sha256':sha(ROOT/'config.json'),'negative_index_sha256':sha(ROOT/'negative_index.json'),'old_index_sha256':sha(OLD/'dataset_index.jsonl'),'old_config_sha256':sha(OLD/'config.json'),'old_receipt_sha256':sha(OLD/'ENGINEERING_ACCEPTANCE.json'),'baseline_records_sha256':sha(OLD/'predictions/all_frozen/records.jsonl'),'frozen':freeze,'rotation_results_seen':False,'historical_0deg_results_already_analyzed':True})
    print('Fixed',len(diagnostic),'diagnostic,',len(representative),'representative, union',len(set(diagnostic+representative)),'and',len(negative),'negative pairs',flush=True)

if __name__=='__main__':main()
