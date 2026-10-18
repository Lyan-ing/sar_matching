import sys
sys.dont_write_bytecode=True
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from experiment import ROOT,OLD,normalize,representation,fill_invalid,window_read,restore_prediction,Prediction,perspective,model_estimate,read,sha,selected

def test_quantization_monotone_zero_and_nodata():
    raw=np.arange(10000,dtype=float).reshape(100,100);raw[0,1]=np.nan;raw[0,2]=-9999;v=np.isfinite(raw)&(raw!=-9999)
    for kind in ['linear','asinh']:
        im,meta=normalize(raw,v,kind,'linear_amplitude');assert im.dtype==np.uint8 and im.shape==raw.shape and (im[~v]==0).all()
        assert im[0,0]==0 and meta['valid_zero_preserved'];z=im[v][np.argsort(raw[v])];assert np.all(np.diff(z.astype(int))>=0)

def test_units_and_degenerate():
    for units,a in [('linear_amplitude',np.ones((32,32))),('linear_power',np.zeros((32,32))),('db',np.linspace(-50,5,1024).reshape(32,32))]:
        for kind in ['linear','asinh']:
            im,m=normalize(a,np.ones_like(a,bool),kind,units);assert np.isfinite(im).all()
    import pytest
    with pytest.raises(ValueError):normalize(np.full((8,8),-1.),np.ones((8,8),bool),'linear','linear_power')
    with pytest.raises(ValueError):normalize(np.ones((8,8)),np.zeros((8,8),bool),'linear','db')

def test_invalid_extension_no_false_black_edge():
    im=np.full((64,64),90,np.uint8);v=np.ones((64,64),bool);v[20:40,20:40]=False;im[~v]=0
    assert (fill_invalid(im,v)==90).all()
    a,b,m=representation(np.ones((64,64)),v,im,'clahe','linear_power');assert (a[~v]==0).all() and np.unique(b).size==1 and not v[30,30]

def test_real_window_and_center_coordinate(tmp_path):
    a=np.arange(1400*1400,dtype=np.float32).reshape(1400,1400);p=tmp_path/'source.tif';t=from_origin(500000,5000000,2,2)
    with rasterio.open(p,'w',driver='GTiff',height=1400,width=1400,count=1,dtype='float32',crs='EPSG:32613',transform=t) as d:d.write(a,1)
    with rasterio.open(p) as ds:
        b,v,T,xy=window_read(ds,[450,450]);assert np.array_equal(b[256:768,256:768],a[450:962,450:962]);assert T*(256.5,256.5)==t*(450.5,450.5)
        import pytest
        with pytest.raises(ValueError):window_read(ds,[0,0])

def test_prediction_coordinates_restore_and_model():
    rng=np.random.default_rng(7);p=rng.uniform(50,400,(80,2)).astype(np.float32);H=np.array([[1,.03,3],[-.02,1,8],[0,0,1.]])
    T=np.array([[1,0,256],[0,1,256],[0,0,1.] ]);C=T@H@np.linalg.inv(T)
    pred=Prediction('sift',p+256,perspective(C,p+256),keypoints0=p+256,keypoints1=perspective(C,p+256))
    native=restore_prediction(pred,[256,256],[256,256]);e,d=model_estimate(native);assert e.accepted and np.allclose(perspective(e.H,p),perspective(H,p),atol=1e-3)

def test_selection_and_protocol_frozen():
    rows=selected();assert len(rows)==12 and len({r['pair_id'] for r in rows})==12;cfg=read(ROOT/'config.json');assert cfg['B_frozen']==read(OLD/'config.json')
    assert not ({r['component'] for r in rows}&set(cfg['holdout_components']))
    assert all('no algorithm outcomes' in r['selection_reason'] for r in rows)
    assert cfg['no_independent_GT'] and cfg['Success_metrics_computed'] is False
    from collections import Counter
    assert Counter(r['look_pair_category'] for r in rows)=={'different_look_verified':9,'repeat_pass_same_look':3}
    prereg=read(ROOT/'preregistration.json');assert prereg['samples_sha256']==sha(ROOT/'samples.jsonl') and prereg['config_sha256']==sha(ROOT/'config.json')

def test_old_png_replay_and_preservation():
    if not (ROOT/'data/quality.json').exists():return
    q=read(ROOT/'data/quality.json');assert len(q)==12 and all(v['PNG_byte_replay_equal'] for r in q for v in r['views'])
    for path,digest in read(ROOT/'protected_inputs.json').items():assert sha(path)==digest,path

def test_actual_context_centers():
    if not (ROOT/'contexts/SUMMARY.json').exists():return
    summary=read(ROOT/'contexts/SUMMARY.json');assert len(summary['records'])==12
    for r in summary['records']:
        if r['available']:assert len(r['views'])==2 and all(v.get('valid_native_pixels_equal',v.get('pixels_equal')) and v['display_center_equal'] and v['source_grid_unresampled'] for v in r['views'])
    if (ROOT/'contexts_resolved.json').exists():
        resolved=read(ROOT/'contexts_resolved.json');assert resolved['available']==12 and resolved['failed']==0
        from PIL import Image
        for r in resolved['records']:
            for side in [0,1]:
                p=__import__('pathlib').Path(r['context_folder']);row=next(row for row in selected() if row['sample_id']==r['sample_id']);native=np.array(Image.open(row[('reference' if side==0 else 'moving')+'_native_png']))
                assert np.array_equal(np.array(Image.open(p/f'view{side}.png'))[256:768,256:768],native)

def test_results_no_accuracy_and_one_factor():
    if not (ROOT/'runs/context/records.jsonl').exists():return
    from experiment import lines
    radiometry=lines(ROOT/'runs/radiometry/records.jsonl');context=lines(ROOT/'runs/context/records.jsonl')
    assert len(radiometry)==180 and len(context)==72
    assert all(not r['independent_GT'] for r in radiometry+context)
    assert not any('Success@' in k for r in radiometry+context for k in r)
    assert len({(r['sample_id'],r['method'],r['condition']) for r in radiometry+context})==252
    assert sum(r.get('baseline_reused',False) for r in radiometry)==72
    for r in context:
        if r['condition']=='context_fine' and r.get('accepted'):assert r['coarse_accepted'] and r['original_inference_protocol']=='image_only_context_coarse_to_fine'

def test_confirmation_and_independent_review():
    if not (ROOT/'runs/confirmation/SUMMARY.json').exists():return
    from experiment import lines
    primary=selected();confirmation=lines(ROOT/'confirmation_samples.jsonl');assert len(confirmation)==4 and not ({r['component'] for r in primary}&{r['component'] for r in confirmation})
    records=lines(ROOT/'runs/confirmation/records.jsonl');assert len(records)==24 and {r['condition'] for r in records}=={'original','linear'}
    tasks=lines(ROOT/'annotation_adapter/annotation_tasks/tasks.jsonl');assert len(tasks)==5 and sum(r['look_pair_category']=='different_look_verified' for r in tasks)>=2
    assert any(r['sample_id']=='Umbra_0061__chip_001' for r in tasks)
    assert not list((ROOT/'annotation_adapter/annotations').rglob('v*.json'))

def test_same_match_geometric_diagnostics():
    if not (ROOT/'geometry/models.jsonl').exists():return
    from experiment import lines
    records=lines(ROOT/'geometry/models.jsonl');assert len(records)==366
    lookup={(r['sample_id'],r['condition'],r['method'],r['model']):r for r in records}
    for r in records:
        mate=lookup[r['sample_id'],r['condition'],r['method'],'homography' if r['model']=='affine' else 'affine'];assert r['n_matches']==mate['n_matches'] and not r['independent_GT']

def test_predictor_isolated_from_geography_and_labels(tmp_path):
    import importlib.util
    spec=importlib.util.spec_from_file_location('actual_context_timing',ROOT/'timing_validate.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    from PIL import Image
    im=np.tile(np.arange(1024,dtype=np.uint8),(1024,1))
    for side in [0,1]:Image.fromarray(im).save(tmp_path/f'view{side}.png');Image.fromarray(im[:512,:512]).save(tmp_path/f'native{side}.png')
    class ImageOnlyRow(dict):
        def __getitem__(self,key):
            assert key in ['reference_native_png','moving_native_png'],'GT/Mask/map read by predictor'
            return super().__getitem__(key)
    class Matcher:
        def predict(self,a,b):
            x,y=np.meshgrid(np.linspace(70,a.shape[1]-70,8),np.linspace(70,a.shape[0]-70,8));p=np.c_[x.ravel(),y.ravel()].astype(np.float32)
            return Prediction('orb',p,p,n_keypoints0=len(p),n_keypoints1=len(p),keypoints0=p,keypoints1=p)
    row=ImageOnlyRow(reference_native_png=str(tmp_path/'native0.png'),moving_native_png=str(tmp_path/'native1.png'),GT='POISON',Mask='POISON',transform='POISON')
    for condition in ['context_direct','context_fine']:
        p,e,lp,le,reason=m.pipeline(Matcher(),'orb',tmp_path,row,condition);assert le.accepted

def test_complete_timing_predictions_identical():
    if not (ROOT/'timing_validation/SUMMARY.json').exists():return
    from experiment import lines
    s=read(ROOT/'timing_validation/SUMMARY.json');assert s['rows']==72 and s['all_predictions_equal']
    assert all(r['predictions_equal_first_run'] and not r['selection_uses_GT'] and r['actual_end_to_end_ms']>0 for r in lines(ROOT/'timing_validation/records.jsonl'))
