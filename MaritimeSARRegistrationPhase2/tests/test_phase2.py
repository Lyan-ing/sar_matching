import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from sarbench.estimate import perspective, estimate
from sarbench.matchers import LocalFeatureMatcher, PhaseCorrelation, Prediction
from sarbench.evaluate import evaluate
from sarbench.data import discover,load_gt,describe
from sarbench.reports import summarize,bootstrap_group_success,cycle_diagnostics
from sarbench.cli import main


def make_fixture(root, sample='pair_000000', dx=8, dy=-5, no_overlap=False):
    folder=root/'test'/sample;folder.mkdir(parents=True)
    rng=np.random.default_rng(18)
    im=rng.integers(0,90,size=(192,192),dtype=np.uint8)
    for i in range(45):
        x,y=int(rng.integers(14,179)),int(rng.integers(14,179))
        cv2.circle(im,(x,y),int(rng.integers(2,5)),int(rng.integers(150,255)),-1)
    if no_overlap:
        moving=np.full_like(im,0)
        dx=300
        pts0=np.empty((0,2),np.float32)
        pts1=np.empty((0,2),np.float32)
        overlap=0.
    else:
        M=np.float32([[1,0,dx],[0,1,dy]])
        moving=cv2.warpAffine(im,M,(192,192))
        xx,yy=np.meshgrid(np.arange(30,165,9),np.arange(30,165,9))
        pts0=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32)
        pts1=pts0+np.array([dx,dy],np.float32)
        overlap=.80
    cv2.imwrite(str(folder/'view_0.png'),im)
    cv2.imwrite(str(folder/'view_1.png'),moving)
    np.savez_compressed(folder/'matches_0_1.npz',points_i=pts0,points_j=pts1)
    H=[[1,0,dx],[0,1,dy],[0,0,1]]
    meta={'dataset':'hrsid','source_id':sample,'source_group':'P0001','source_component':'comp1',
          'source_split':'test','profile':'none' if no_overlap else 'easy',
          'scene_type':'offshore','ship_annotations':[[],[]],
          'num_views':2,'pairs':[{'views':[0,1],'H_i_to_j':H,'overlap_fraction_i':overlap,
                               'overlap_fraction_j':overlap,'matches':'matches_0_1.npz'}]}
    (folder/'meta.json').write_text(json.dumps(meta))
    return folder


def test_phase_translation_sign():
    rng=np.random.default_rng(11)
    im=rng.random((128,128)).astype('float32')
    moved=cv2.warpAffine(im,np.float32([[1,0,8],[0,1,-5]]),(128,128))
    pred=PhaseCorrelation().predict((im*255).astype('uint8'),(moved*255).astype('uint8'))
    assert np.allclose(pred.precomputed_H[:2,2],[8,-5],atol=.75)
    assert pred.confidence>0.2


def test_matching_and_projection():
    p0=np.array([[0,0],[14,22]],float)
    H=np.array([[1,0,8],[0,1,-5],[0,0,1]],float)
    assert np.allclose(perspective(H,p0),p0+[8,-5])
    rng=np.random.default_rng(8)
    image=np.zeros((256,256),np.uint8)
    for i in range(100):
        cv2.circle(image,tuple(map(int,rng.integers(8,245,size=2))),3,int(rng.integers(120,255)),-1)
    shifted=cv2.warpAffine(image,np.float32([[1,0,8],[0,1,-5]]),(256,256))
    pred=LocalFeatureMatcher('sift').predict(image,shifted)
    est=estimate(pred,image.shape,min_coverage=0)
    assert est.accepted,est.reason
    err=np.linalg.norm(perspective(est.H,np.array([[120.,120.]])) - [[128.,115.]],axis=1)
    assert err[0]<2


def test_gt_is_only_post_prediction(tmp_path):
    root=tmp_path/'frozen'
    make_fixture(root)
    refs=discover(root)
    assert len(refs)==1
    gt=load_gt(refs[0])
    H=gt['H']
    est=estimate(Prediction('sift',points0=np.empty((0,2)),points1=np.empty((0,2))), (192,192))
    outcome=evaluate(est,gt,(192,192))
    assert outcome['success_5px'] is False
    assert outcome['failure_to_register']
    assert np.allclose(H,[[1,0,8],[0,1,-5],[0,0,1]])


def test_negative_excluded_from_success_denominator(tmp_path):
    root=tmp_path/'d'; make_fixture(root, sample='pair_000000');make_fixture(root,sample='pair_000001',no_overlap=True)
    refs=discover(root)
    neg=load_gt(refs[1])
    accept=estimate(Prediction('phase',confidence=.9,model_type='translation',precomputed_H=np.eye(3)),(192,192))
    n=evaluate(accept,neg,(192,192))
    assert n['false_accept'] and n['success_5px'] is None
    reject=estimate(Prediction('orb'),(192,192))
    n2=evaluate(reject,neg,(192,192))
    assert n2['correct_reject']


def test_split_guard(tmp_path):
    root=tmp_path/'f';folder=make_fixture(root)
    p=folder/'meta.json'; m=json.loads(p.read_text());m['source_split']='train';p.write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Non-test'):
        discover(root)
    assert len(discover(root,require_test=False))==1


def test_cycle():
    def row(name,H,i,j):
        return {'method':'sift','folder':'test/pair_1','accepted':True,'pred_H':json.dumps(H),
                'view_i':i,'view_j':j}
    I=np.eye(3)
    H01=np.array([[1,0,5],[0,1,0],[0,0,1.]])
    H12=np.array([[1,0,4],[0,1,2],[0,0,1.]])
    H02=H12@H01
    out=cycle_diagnostics([row('0-1',H01.tolist(),0,1),row('0-2',H02.tolist(),0,2),row('1-2',H12.tolist(),1,2)])
    assert len(out)==1 and out[0]['cycle_max_px']<1e-6


def test_cluster_bootstrap_requires_real_component():
    r={'dataset':'hrsid','source_component':'unknown','no_overlap':False,'success_5px':True}
    assert not bootstrap_group_success([r])['available']


def test_cli_smoke_two_methods(tmp_path):
    root=tmp_path/'frozen';make_fixture(root)
    make_fixture(root,sample='pair_000001',no_overlap=True)
    (root/'ACCEPTED_demo.json').write_text('{}')
    dest=tmp_path/'run'
    main(['run','--root',str(root),'--out',str(dest),'--methods','sift','phase',
          '--no-figures','--preview-count','1','--log-every','1'])
    result=json.loads((dest/'summary.json').read_text())
    assert result['dataset']['relations']==2
    assert result['by_method']['sift']['negative']==1
    assert result['by_method']['phase']['negative']==1
    assert (dest/'results.csv').exists()
    with pytest.raises(FileExistsError):
        main(['run','--root',str(root),'--out',str(dest),'--methods','sift','--no-figures'])


def test_inspect_warn_missing_fields(tmp_path,capsys):
    root=tmp_path/'f'; folder=make_fixture(root)
    meta=folder/'meta.json';d=json.loads(meta.read_text());d.pop('profile');d.pop('scene_type');meta.write_text(json.dumps(d))
    main(['inspect','--root',str(root)])
    info=json.loads(capsys.readouterr().out)
    assert info['missing_profile']==1
    assert info['missing_scene']==1

def test_compare_separate_runs(tmp_path):
    root=tmp_path/'data';make_fixture(root);(root/'ACCEPTED_test.json').write_text('{}')
    a,b,c=[tmp_path/p for p in ('a','b','c')]
    main(['run','--root',str(root),'--out',str(a),'--methods','sift','--no-figures','--preview-count','0'])
    main(['run','--root',str(root),'--out',str(b),'--methods','phase','--no-figures','--preview-count','0'])
    main(['compare','--runs',str(a),str(b),'--out',str(c),'--no-figures'])
    result=json.loads((c/'summary.json').read_text())
    assert set(result['by_method'])=={'sift','phase'}
    with pytest.raises(FileExistsError):
        main(['compare','--runs',str(a),str(b),'--out',str(c),'--no-figures'])
    settings=json.loads((b/'settings.json').read_text())
    settings['reproj_px']=9.
    (b/'settings.json').write_text(json.dumps(settings))
    with pytest.raises(ValueError,match='protocol mismatch'):
        main(['compare','--runs',str(a),str(b),'--out',str(tmp_path/'unsafe'),'--no-figures'])

def test_discover_three_view_edges(tmp_path):
    root=tmp_path/'f'; folder=make_fixture(root)
    image=cv2.imread(str(folder/'view_1.png'),cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(str(folder/'view_2.png'),image)
    mfile=folder/'meta.json'
    m=json.loads(mfile.read_text())
    m['num_views']=3
    m['pairs'].extend([
        {'views':[0,2],'H_i_to_j':[[1,0,8],[0,1,-5],[0,0,1]],
         'overlap_fraction_i':.8,'overlap_fraction_j':.8,'matches':'matches_0_1.npz'},
        {'views':[1,2],'H_i_to_j':np.eye(3).tolist(),
         'overlap_fraction_i':1.,'overlap_fraction_j':1.,'matches':'matches_0_1.npz'}])
    mfile.write_text(json.dumps(m))
    refs=discover(root)
    assert len(refs)==3
    assert {x.id for x in refs} == {f'pair_000000:{i}-{j}' for i,j in [(0,1),(0,2),(1,2)]}


def test_formal_source_components_share_background_parent():
    from sarbench.data import source_component
    common={'source_group':'P0012','split_protocol':'phase1_pixel_supported_source_components_v2'}
    assert source_component(dict(common,dataset='hrsid'))==source_component(dict(common,dataset='background'))=='hrsid:P0012'
    assert source_component(dict(common,dataset='ssdd'))=='ssdd:P0012'
    assert source_component({'dataset':'hrsid','source_group':'P0012'})=='unknown'
    rows=[{'dataset':ds,'source_component':comp,'no_overlap':False,'success_5px':True} for ds,comp in [('hrsid','hrsid:P0012'),('background','hrsid:P0012')]+[('ssdd',f'ssdd:{i}') for i in range(4)]]
    assert bootstrap_group_success(rows,n_boot=10)['source_components']==5


def test_four_decision_states(tmp_path):
    root=tmp_path/'f';make_fixture(root);gt=load_gt(discover(root)[0])
    no=estimate(Prediction('sift'),(192,192))
    assert evaluate(no,gt,(192,192))['decision_status']=='no_output'
    low=estimate(Prediction('phase',confidence=.1,model_type='translation',precomputed_H=gt['H']),(192,192))
    assert low.candidate_H is not None and evaluate(low,gt,(192,192))['decision_status']=='rejected'
    good=estimate(Prediction('phase',confidence=.9,model_type='translation',precomputed_H=gt['H']),(192,192))
    assert evaluate(good,gt,(192,192))['decision_status']=='success'
    wrong=estimate(Prediction('phase',confidence=.9,model_type='translation',precomputed_H=np.eye(3)),(192,192))
    assert evaluate(wrong,gt,(192,192))['decision_status']=='incorrect_accept'


def test_cycle_covers_full_image_and_reports_rejection():
    rows=[]
    for i,j,H in [(0,1,np.eye(3)),(1,2,np.eye(3)),(0,2,np.diag([2.,2.,1.]))]:
        rows.append({'method':'sift','folder':'g','view_i':i,'view_j':j,'num_views':3,'image_width':384,'image_height':384,'pred_H':json.dumps(H.tolist()),'accepted':True})
    assert cycle_diagnostics(rows)[0]['cycle_max_px']>200
    rows[0]['accepted']=False;rows[0]['pred_H']=''
    record=cycle_diagnostics(rows)[0]
    assert not record['complete_accepted'] and record['cycle_mean_px'] is None


def test_ship_stratification_uses_pair_views():
    from sarbench.cli import _has_ships
    meta={'ship_annotations':[[],[],[{'instance_id':'1'}]]}
    assert not _has_ships(meta,(0,1)) and _has_ships(meta,(0,2))
