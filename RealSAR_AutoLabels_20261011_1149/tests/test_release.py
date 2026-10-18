"""Release checks on real files; simulated annotation actors live only in tmp_path."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from common import ROOT,source_rows,configure
from geo import load_pair,sample_valid

def test_all_real_geographic_points_and_masks():
    configure(ROOT)
    summary=json.loads((ROOT/'georeference_correspondences/SUMMARY.json').read_text())
    assert summary['attempted']==191 and summary['A_qualified']==191
    for row in source_rows():
        folder=ROOT/'georeference_correspondences'/row['sample_id']
        _,valid,_,_,mapper=load_pair(row)
        g=json.loads((folder/'geographic.json').read_text())
        points=json.loads((folder/'correspondences.json').read_text())
        assert points['independent_GT'] is False
        for key in ['points','reverse_sampled_points']:
            a=np.array([p['xy0'] for p in points[key]]).reshape(-1,2)
            b=np.array([p['xy1'] for p in points[key]]).reshape(-1,2)
            assert sample_valid(valid[0],a).all() and sample_valid(valid[1],b).all()
            assert np.allclose(mapper(a),b,atol=1e-4)
        for side in [0,1]:
            mask=np.array(Image.open(folder/f'geographic_valid_overlap{side}.png'))>0
            assert mask.sum()==g['overlap'][side]['joint_pixels']
            assert not (mask&~valid[side]).any()
        warp=np.array(Image.open(folder/'moving_native_geo_assisted.png'))
        support=np.array(Image.open(folder/'geographic_valid_overlap0.png'))>0
        assert (warp[~support]==0).all()

def test_existing_source_components_and_acquisitions_not_split():
    from collections import defaultdict
    buckets=[defaultdict(set) for _ in range(3)]
    for r in source_rows():
        for bucket,key in zip(buckets,['component','site_name','reference_acquisition_id']):bucket[r[key]].add(r['split'])
        buckets[2][r['moving_acquisition_id']].add(r['split'])
    assert all(len(s)==1 for b in buckets for s in b.values())

def test_adapter_exposure_is_truthful_and_never_autofills(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('new_adapter',ROOT/'annotation_adapter/serve.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for d in ['annotation_tasks','candidate_matches','annotations','gt_quality']:(tmp_path/d).mkdir()
    sid='SIMULATED_ADAPTER_TEST'
    payload={'sample_id':sid,'geographic':[],'sift_candidates':[],'pseudo_generator':'geo_assisted/orb'}
    f=tmp_path/'candidate_matches'/f'{sid}.json';f.write_text(json.dumps(payload))
    task={'sample_id':sid,'width0':512,'height0':512,'width1':512,'height1':512,'candidate_sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
    (tmp_path/'annotation_tasks/tasks.jsonl').write_text(json.dumps(task)+'\n')
    store=module.AdapterStore(tmp_path)
    for slot,actual in [('geographic','georeference_correspondences'),('sift_candidates','pseudo_correspondences')]:
        event=store.expose(sid,'SIMULATED_ACTOR',slot)
        assert event['source']==actual and event['independent_GT'] is False
        assert event['generator']==('geo_assisted/orb' if slot=='sift_candidates' else 'GeoTIFF CRS/Affine pixel-center mapping')
    assert len(store.assisted(sid,'SIMULATED_ACTOR'))==2
    assert store.load(sid)['points']==[]

def test_final_pseudo_labels_valid_and_never_GT():
    import pytest
    if not (ROOT/'dataset_index.jsonl').exists():pytest.skip('frozen inference not finished yet')
    rows=[json.loads(s) for s in (ROOT/'dataset_index.jsonl').read_text().splitlines()]
    assert len(rows)==191 and len({r['sample_id'] for r in rows})==191
    for row in rows:
        assert row['has_C_independent_reference'] is False
        assert row['gt_status']=='unverified' and row['independent_test_GT'] is False
        assert row['training_eligible_now'] is False
        if row['has_B_pseudo']:
            _,valid,_,_,_=load_pair(row)
            data=json.loads(Path(row['pseudo_correspondences_path']).read_text())
            assert data['independent_GT'] is False and data['confidence_is_probability'] is False
            assert len(data['points'])>=8
            for side in [0,1]:
                points=np.array([p['xy'+str(side)] for p in data['points']])
                assert sample_valid(valid[side],points).all()
            assert all(p['human_verified'] is False and p['static_verified'] is False for p in data['points'])
            assert (np.array(Image.open(row['pseudo_support_path']))>0).any()
