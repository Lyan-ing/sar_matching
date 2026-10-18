import numpy as np,pytest,json
from study import rotation,rotate,restored,restore_H,perspective,selection,ROOT,OLD,rows,context,overlap,sha,grade

@pytest.mark.parametrize('angle',[0,90,180,270])
def test_integer_pixels_coordinates_and_inverse(angle):
    image=np.arange(15).reshape(3,5);R=rotation(angle,image.shape);out=rotate(image,angle)
    xy=np.array([[x,y] for y in range(3) for x in range(5)],float);q=perspective(R,xy)
    assert np.array_equal(out[q[:,1].astype(int),q[:,0].astype(int)],image[xy[:,1].astype(int),xy[:,0].astype(int)])
    assert np.allclose(restored(q,angle,image.shape),xy)
    H=np.array([[1.1,.2,5],[-.1,.9,3],[.0001,-.00005,1.]])
    assert np.allclose(restore_H(R@H,angle,image.shape),H)

def test_512_centers_and_geo_composition():
    from geo import PixelMap
    from affine import Affine
    m=PixelMap(Affine(2,0,100,0,-2,200),'EPSG:32610',Affine(1,0,100,0,-1,200),'EPSG:32610');p=np.array([[0.,0.],[511.,511.],[42.125,68.5]])
    assert np.array_equal(perspective(rotation(180),p[:2]),[[511,511],[0,0]])
    H=np.array([[1,0,3],[0,1,4],[0,0,1.]])
    for angle in [0,90,180,270]:
        predicted=perspective(np.asarray(rotation(angle))@H,p)
        assert np.allclose(m(restored(predicted,angle)),m(perspective(H,p)))
    nonlinear=PixelMap(Affine(.01,0,-123,0,-.01,40),'EPSG:4326',Affine(1000,0,-13700000,0,-1000,5000000),'EPSG:3857')
    for angle in [0,90,180,270]:
        rotated=perspective(np.asarray(rotation(angle))@H,p)
        assert np.allclose(nonlinear(restored(rotated,angle)),nonlinear(perspective(H,p)))

def test_selection_ignores_poisoned_GT_B_and_geography():
    class NoGT(dict):
        def __getitem__(self,key):
            if key in ['GT','B','mask','geo_difference']:raise AssertionError('GT read')
            return super().__getitem__(key)
    records=[NoGT(angle=a,accepted=a!=0,GT='poison',B=False,geo_difference=1e9) for a in [0,90,180,270]]
    assert selection(records)['angle']==180
    records[0]['accepted']=True;assert selection(records)['angle']==0
    assert selection([dict(angle=0,accepted=False)]) is None

def test_fixed_protocol_and_negative_geography():
    c=json.loads((ROOT/'config.json').read_text());p=json.loads((ROOT/'preregistration.json').read_text())
    assert p['config_sha256']==sha(ROOT/'config.json')
    assert c['B_frozen']==json.loads((OLD/'config.json').read_text())
    assert p['negative_index_sha256']==sha(ROOT/'negative_index.json')
    neg=json.loads((ROOT/'negative_index.json').read_text());assert len(neg)==30
    for r in neg:
        valid,m=context(r);assert all(s['joint_pixels']==0 for s in overlap(valid,m)[2]);assert r['reference_acquisition_id']!=r['moving_acquisition_id']
    assert len(rows())==191 and len({r['pair_id'] for r in rows()})==65

def test_zero_degree_reuse_is_exact():
    file=ROOT/'runs/diagnostic8/records.jsonl'
    if not file.exists():pytest.skip('small actual inference not executed yet')
    old={(r['sample_id'],r['method']):r for r in [json.loads(s) for s in (OLD/'quality/model_grades.jsonl').read_text().splitlines()] if r['protocol']=='native'}
    for r in [json.loads(s) for s in file.read_text().splitlines()]:
        if r['angle']!=0:continue
        before=old[r['sample_id'],r['method']]
        for key in ['accepted','reason','n_matches','n_inliers','candidate_H','cache_sha256']:assert r[key]==before[key]
        assert r['baseline_reused']

def test_actual_rotated_features_and_matrices_restored():
    file=ROOT/'runs/diagnostic8/records.jsonl'
    if not file.exists():pytest.skip('small actual inference not executed yet')
    recs=[json.loads(s) for s in file.read_text().splitlines()];assert len(recs)==96
    assert not any(r.get('reason')=='runtime_error' for r in recs)
    for r in recs:
        if r['angle']==0:continue
        z=np.load(r['cache_path']);assert np.allclose(restored(z['points1_rotated'],r['angle']),z['points1_native'])
        if r['candidate_H'] is not None:assert np.allclose(restore_H(r['prediction_frame_H'],r['angle']),r['candidate_H'])
        assert np.all(z['points1_native']>=-1e-6) and np.all(z['points1_native']<=511+1e-6)

def test_complete_experiment_and_real_strategy_prefixes():
    analysis=ROOT/'analysis/summary.json'
    if not analysis.exists():pytest.skip('full experiment not complete yet')
    allr=[json.loads(s) for s in (ROOT/'runs/full191/records.jsonl').read_text().splitlines()]
    assert len(allr)==len({(r['sample_id'],r['method'],r['angle']) for r in allr})==2292
    assert sum(r['baseline_reused'] for r in allr)==573
    summary=json.loads(analysis.read_text());assert summary['source_units']['chips']==191 and summary['source_units']['acquisition_pairs']==65 and summary['C']==0
    strategies=[json.loads(s) for s in (ROOT/'runs/strategies/strategies.jsonl').read_text().splitlines()]
    assert len(strategies)==1146
    for s in rows():
        for method in ['sift','orb','lightglue']:
            pair=[r for r in strategies if r['sample_id']==s['sample_id'] and r['method']==method];assert {r['strategy'] for r in pair}=={'A','B'}
            a=next(r for r in pair if r['strategy']=='A');b=next(r for r in pair if r['strategy']=='B');assert 1<=a['rotations_executed']<=2 and a['rotations_executed']<=b['rotations_executed']<=4
            assert 0<a['end_to_end_ms']<=b['end_to_end_ms']
            assert all(r['B_or_GT_used_for_selection'] is False for r in pair)
    neg=[json.loads(s) for s in (ROOT/'runs/negative/records.jsonl').read_text().splitlines()]
    assert len(neg)==360 and all(r['negative'] and r['pseudo_quality']=='unreliable_candidate' for r in neg)
