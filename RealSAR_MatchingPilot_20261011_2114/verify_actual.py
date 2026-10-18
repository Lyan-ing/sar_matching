"""Acceptance of actual frozen artifacts, not an independent correctness test."""
from core import *
import importlib.metadata as md,zipfile
from datetime import datetime
from zoneinfo import ZoneInfo
def main():
    old=json.loads((ROOT/'protected_inputs.json').read_text())
    changed=[p for p,h in old.items() if sha(p)!=h]
    assert not changed,changed
    pr=json.loads((ROOT/'preregistration.json').read_text());bfreeze=json.loads((ROOT/'B_run_freeze.json').read_text())
    for name in ['core','config','samples']:
        filename=name+('.jsonl' if name=='samples' else '.json' if name=='config' else '.py')
        assert sha(ROOT/filename)==pr[name+'_sha256']
    assert sha(ROOT/'run.py')==bfreeze['run_sha256']
    assert sha(ROOT/'guarded_run.py')==bfreeze['guard_sha256']
    for w in json.loads((ROOT/'weights/identity.json').read_text()):
        p=ROOT/'weights'/w['name'];assert w['status']=='downloaded' and p.stat().st_size==w['bytes'] and sha(p)==w['sha256'] and zipfile.is_zipfile(p)
    records=[];input_metadata=[]
    for row in selected():
        sid=row['sample_id'];raw,valid,images,infos,mapper=load_pair(row)
        input_metadata.append(dict(sample_id=sid,views=[dict(info,valid_fraction=float(v.mean()),
            valid_zero_fraction=float(np.mean(a[v]==0)),raw_min=float(a[v].min()),raw_max=float(a[v].max()),
            amplitude_min=float(amplitude(a,v,info['input_units'])[0][v].min()),
            display_min=int(im.min()),display_max=int(im.max())) for a,v,im,info in zip(raw,valid,images,infos)],
            map_numerical_roundtrip_NOT_GT=geo.roundtrip(mapper,regular_grid((512,512),16))))
        with np.load(ROOT/'route_A'/(sid+'_features')/'features.npz') as f:
            assert len(f['points0'])<=256 and len(f['points1'])<=256
            for method in ['A1','A2']:
                p=ROOT/'route_A'/(method+'_'+sid)
                with np.load(p/'matches.npz') as c:
                    assert all(any(np.array_equal(x,y) for y in f['points0']) for x in c['points0'])
                    assert all(any(np.array_equal(x,y) for y in f['points1']) for x in c['points1_native'])
                    if method=='A2':assert np.all(np.abs(c['points1_native']-mapper(c['points0']))<=64+1e-7)
        with np.load(ROOT/'route_B'/(sid+'_dense')/'dense.npz') as d,np.load(ROOT/'route_B'/('B1_'+sid)/'matches.npz') as b1,np.load(ROOT/'route_B'/('B2_'+sid)/'matches.npz') as b2:
            assert len(b1['points0'])==2048
            norm=d['normalized_samples'];assert np.allclose(roma_native_coordinates(norm[:,:2]),b1['points0'],atol=5e-5,rtol=0)
            assert np.allclose(roma_native_coordinates(norm[:,2:]),b1['points1_native'],atol=5e-5,rtol=0)
            keep=np.isfinite(mapper(b1['points0'])).all(1)&(np.abs(b1['points1_native']-mapper(b1['points0']))<=64).all(1)
            assert np.array_equal(b2['points0'],b1['points0'][keep]) and np.array_equal(b2['points1_native'],b1['points1_native'][keep])
            assert np.array_equal(b2['confidence'],b1['confidence'][keep])
            assert np.array_equal(d['certainty'].ravel()[d['sample_flat_indices']],b1['confidence'])
    for group,count in [('route_A',8),('route_B',8),('baselines',24)]:
        fs=list((ROOT/group).glob('*/record.json'));assert len(fs)==count
        for p in fs:
            r=json.loads(p.read_text());assert sha(r['cache_path'])==r['cache_sha256']
            assert r['GT_evidence'].startswith('C=0')
            with np.load(r['cache_path']) as c:assert len(c['points0'])==r['n_matches'] and int(c['inlier_mask'].sum())==r['n_inliers']
            assert len(json.loads(p.with_name('local.json').read_text()))==16
            records.append(r)
    assert len({r['sample_id'] for r in records})==4
    repeats=json.loads((ROOT/'route_B/timing_repeats.json').read_text());assert len(repeats)==3 and all(r['exact_primary_coordinates'] for r in repeats)
    guard=json.loads((ROOT/'logs/file_access_B.json').read_text());assert not guard['blocked']
    before=json.loads((ROOT/'environment_before_B.json').read_text())['packages']
    now={d.metadata['Name']:d.version for d in md.distributions()};assert before==now,'match package set changed'
    for name in ['torch','torchvision','opencv-python-headless']:assert md.version(name)==before[name]
    for name in ['opencv-python','opencv-contrib-python','opencv-contrib-python-headless']:assert name not in now
    assert max(r['timing']['GPU_allocated_MiB'] for r in records if r['method'] in ['B1','B2'])<20*1024
    assert all(r['end_to_end_ms']<120000 for r in records if r['method'] in ['A1','A2','B1','B2'])
    for p in (ROOT/'visualizations').glob('*/local_zoom.png'):assert p.is_file()
    manifest=json.loads((ROOT/'visualizations/manifest.json').read_text());assert len(manifest)==104
    for f in manifest:assert not f['independent_GT'] and sha(ROOT/f['file'])==f['sha256']
    result=dict(status='passed_engineering_acceptance_NOT_accuracy',protected_old_files_unchanged=len(old),
        new_primary_version_outputs=16,new_dense_RoMa_calls=4,extra_timing_calls=3,baseline_caches_reused=24,
        method_records=len(records),local_tile_records=640,figures=len(manifest),unit_tests=9,
        environment_package_set_unchanged=True,B_Python_human_GT_access_blocked_events=0,
        B_coordinate_conversion_verified=True,B2_exact_subset_verified=True,GT_C=0,
        actual_inference_failed_pairs=0,download_failures_preserved=True,old_results_overwritten=False,
        max_GPU_allocated_MiB=max(r['timing']['GPU_allocated_MiB'] for r in records if r['method'] in ['B1','B2']))
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    save(ROOT/('ACCEPTANCE_'+stamp+'.json'),result);save(ROOT/'actual_input_metadata.json',input_metadata)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
