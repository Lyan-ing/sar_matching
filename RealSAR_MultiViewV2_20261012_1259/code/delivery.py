"""Small read-only comparator index, raw geography baseline and delivery evidence."""
import sys
sys.dont_write_bytecode=True
import json,csv,time,subprocess,platform
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import rasterio,cv2,torch
from geo_tools import *
from registration import raw_geography

def main():
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');candidates=lines(ROOT/'candidate_regions.jsonl')
    pairs={r['pair_id'] for r in candidates};cache=BASE/'RealSAR_AutoLabels_20261011_1149/predictions/all_frozen/records.jsonl'
    old=[r for r in lines(cache) if r['pair_id'] in pairs and r['method'] in ['sift','orb','lightglue']]
    refs=[]
    for r in old:
        p=Path(r['cache_path']);expected=r.get('cache_sha256');actual=sha(p)
        if expected is not None and actual!=expected:raise ValueError('Referenced old cache changed: '+str(p))
        refs.append(dict(kind='existing_frozen_baseline',sample_id=r['sample_id'],pair_id=r['pair_id'],method=r['method'],protocol=r['protocol'],record=r,cache_sha256_verified=True,comparison_scope='old 512 crop context only; NOT same-FOV V2 quantitative comparison'))
    roma=BASE/'RealSAR_MatchingPilot_20261011_2114/route_B'
    for p in sorted(roma.glob('B*/record.json')):
        r=json.loads(p.read_text());sid=r['sample_id'];pair=sid.split('__')[0]
        refs.append(dict(kind='existing_frozen_RoMa',sample_id=sid,pair_id=pair,has_V2_candidate=pair in pairs,record=r,record_path=str(p),record_sha256=sha(p),matches_path=str(p.parent/'matches.npz'),matches_sha256=sha(p.parent/'matches.npz'),comparison_scope='old native512 exploratory pilot; NOT V2 inference'))
    jsonl(ROOT/'existing_comparators.jsonl',refs)
    save(ROOT/'existing_comparators_identity.json',dict(old_records_sha256=sha(cache),RoMa_model_identity_path=str(roma/'model_identity.json'),RoMa_model_identity_sha256=sha(roma/'model_identity.json'),RoMa_model_identity=json.loads((roma/'model_identity.json').read_text()),new_RoMa_inference=0,matching_cache_reexecuted=0))
    out=ROOT/'raw_geography';out.mkdir(exist_ok=False);rows=[]
    for r in candidates:
        t=time.perf_counter();z0,z1=[v['native_info'] for v in r['views']];p=geo.regular_grid((z0['height'],z0['width']),16);q=raw_geography(z0,z1,p)
        m0=read_tif(ROOT/'regions'/r['sample_id']/'native0.tif')[1];m1=read_tif(ROOT/'regions'/r['sample_id']/'native1.tif')[1]
        valid=sample_valid(m0,p)&sample_valid(m1,q);p,q=p[valid],q[valid]
        np.savez_compressed(out/(r['sample_id']+'.npz'),points0=p,points1=q)
        row=dict(sample_id=r['sample_id'],method='georeference',protocol='B',coordinate_frame='reference_native_to_moving_native_pixel_centers',n_geographic_samples=len(p),decision='coarse_geographic_relation_only',accepted=False,scatterer_GT=False,independent_accuracy_available=False,elapsed_ms=1000*(time.perf_counter()-t),mapping=[z0,z1],semantic_visibility_unknown=True)
        save(out/(r['sample_id']+'.json'),row);rows.append(row)
    jsonl(out/'results.jsonl',rows)
    smoke=lines(ROOT/'smoke/results.jsonl');fields=['sample_id','method','protocol','decision','reason','n_matches','n_inliers','inlier_ratio','geo_candidate_deviation_median_px','end_to_end_ms_synchronized','GPU_peak_allocated_GiB']
    with (ROOT/'smoke/results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in smoke)
    with (ROOT/'candidate_summary.csv').open('x',newline='') as f:
        fields=['sample_id','sensor','site_name','pair_id','component','look_pair_category','ground_edge_m','native_shape0','native_shape1','common_joint_geographic_valid_fraction','human_readability','gt_qualified']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in candidates:w.writerow({k:(str([r['views'][int(k[-1])]['native_info'][a] for a in ['height','width']]) if k.startswith('native_shape') else r.get(k)) for k in fields})
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=BASE,capture_output=True,text=True)
    modules=['numpy','opencv-python-headless','rasterio','torch','torchvision','lightglue','Pillow','matplotlib','playwright']
    from importlib.metadata import version,PackageNotFoundError
    versions={}
    for m in modules:
        try:versions[m]=version(m)
        except PackageNotFoundError:versions[m]=None
    save(ROOT/'environment_identity.json',dict(timestamp=stamp,python=sys.version,executable=sys.executable,conda_environment='match',platform=platform.platform(),dependencies=versions,torch_cuda_build=torch.version.cuda,cuda_available_in_this_process=torch.cuda.is_available(),git_head=git.stdout.strip() if git.returncode==0 else None,new_code_uncommitted=True,packages_installed_this_round=0))
    protected=json.loads((ROOT/'protected_history.json').read_text());changes=[p for p,h in protected.items() if not Path(p).exists() or sha(p)!=h]
    save(ROOT/('history_protection_'+stamp+'.json'),dict(checked_direct_historical_records=len(protected),changed=changes,unchanged=len(protected)-len(changes),no_full_history_audit=True))
    if changes:raise ValueError('Directly protected historic records changed: '+repr(changes))
    print('DELIVERY',len(candidates),'recommended regions,',len(refs),'existing comparator references,',len(rows),'raw geography baselines; old records unchanged')
if __name__=='__main__':main()
