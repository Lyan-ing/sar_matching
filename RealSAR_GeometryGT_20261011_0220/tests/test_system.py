"""SIMULATED point fixtures only. Never saved in real annotations/GT indexes."""
import sys,json,copy,hashlib,threading,urllib.request,urllib.error
from pathlib import Path
import numpy as np,pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'annotation_app'));sys.path.insert(0,str(ROOT/'evaluation'))
from store import Store
from quality import quality,quantitative,validate_points
from server import make_server
from run import native_mapping,ground_jacobian,evaluate_saved

def task(size=512):return dict(sample_id='SIMULATED_ONLY',width0=size,height0=size,width1=size,height1=size,candidate_count=0)
def points():
    locations=[[30,30],[450,30],[450,450],[30,450],[240,100],[140,400],[120,150],[390,360]]
    return [dict(point_id='P'+str(i),sample_id='SIMULATED_ONLY',xy0=p,xy1=[p[0]+2,p[1]+3],role='control' if i>=4 else 'check',landmark_type='SIMULATED',is_static=True,visible_both=True,confidence='high',sigma0_px=.2,sigma1_px=.2,annotator='alice',annotated_at='test_timestamp',used_assistance=False,review=dict(reviewer='bob',reviewed_at='test_timestamp',conclusion='confirmed',static_confirmed=True,visible_confirmed=True,used_assistance=False,xy0=p,xy1=[p[0]+2,p[1]+3],sigma0_px=.2,sigma1_px=.2)) for i,p in enumerate(locations)]
def document():return dict(points=points(),evaluation_scope='local',disposition='reliable_correspondences')
def new_store(tmp_path):
    for d in ['annotation_tasks','gt_quality','candidate_matches']:(tmp_path/d).mkdir()
    (tmp_path/'annotation_tasks/tasks.jsonl').write_text(json.dumps(task())+'\n');return Store(tmp_path)
def payload(ps,rev=0,actor='alice'):return dict(sample_id='SIMULATED_ONLY',actor=actor,expected_revision=rev,points=ps,evaluation_scope='local',disposition='reliable_correspondences',disposition_note='SIMULATED TEST')

def test_input_selection_and_all_sensor_loads():
    sel=[json.loads(s) for s in (ROOT/'selection/samples.jsonl').read_text().splitlines()];tasks=Store(ROOT).tasks
    master={r['pair_id']+'/'+r['chip_id']:r for r in [json.loads(s) for s in (ROOT.parent/'real_sar_pilot/multisource512_20261010_2352/dataset_index.jsonl').read_text().splitlines()]}
    assert len(sel)==24==len(tasks);assert len({r['pair_id'] for r in sel})==24;assert {r['sensor'] for r in sel}=={'Umbra','Capella','Sentinel-1'}
    from PIL import Image
    for r in sel:
        assert master[r['original_index_id']]['reference_acquisition_id']==r['reference_acquisition_id'];assert r['reference_acquisition_id']!=r['moving_acquisition_id']
        t=tasks[r['sample_id']];assert t['image0']==r['reference_native_png'];assert t['image1']==r['moving_native_png']
        for k in ['image0','image1']:assert Image.open(t[k]).size==(512,512)

@pytest.mark.parametrize('change',[lambda p:p.update(xy0=[512,1]),lambda p:p.update(xy1=None),lambda p:p.update(sample_id='WRONG'),lambda p:p.update(xy1=[float('nan'),1]),lambda p:p.update(sigma0_px=-1)])
def test_coordinate_and_identity_rejection(change):
    p=points()[0];change(p)
    with pytest.raises(ValueError):validate_points([p],task())

def test_duplicates_roles_and_scale():
    ps=points();ps[1]['point_id']=ps[0]['point_id']
    with pytest.raises(ValueError,match='duplicate'):validate_points(ps,task())
    ps=points();ps[1]['xy1']=ps[0]['xy1']
    with pytest.raises(ValueError,match='duplicate'):validate_points(ps,task())
    p=points()[0];p['xy0']=[255,0];p['xy1']=[255,255];validate_points([p],task(256))
    p['xy0']=[256,0]
    with pytest.raises(ValueError):validate_points([p],task(256))

def test_gt_evidence_and_illegal_upgrade():
    d=document();q=quality(d,task());assert q['status']=='gt_qualified';assert q['supported_success_thresholds_px']==[3,5,10]
    for p in d['points']:p.pop('review')
    d['status']='gt_qualified';assert quality(d,task())['status']!='gt_qualified'
    d=document()
    for p in d['points']:p['review']['reviewer']='alice'
    assert quality(d,task())['status']!='gt_qualified'
    d=document()
    for p in d['points']:p['used_assistance']=True
    assert quality(d,task())['eligible_checks']==0
    d=document()
    for p in d['points']:p['review']['used_assistance']=True
    assert quality(d,task())['status']!='gt_qualified'

def test_uncertainty_review_difference_and_support():
    d=document()
    for p in d['points']:p['sigma0_px']=p['sigma1_px']=3.
    q=quality(d,task());assert q['status']=='gt_qualified' and not q['supported_success_thresholds_px']
    d=document()
    for p in d['points']:p['review']['xy0']=[p['xy0'][0]+20,p['xy0'][1]]
    assert quality(d,task())['eligible_checks']==0
    d=document();d['points']=d['points'][:2];assert quality(d,task())['status']!='gt_qualified'
    d=document();d['evaluation_scope']='global_sparse';assert quality(d,task())['status']!='gt_qualified' # four checks < six

def test_no_gt_and_fit_check_isolation():
    with pytest.raises(ValueError,match='gt_qualified'):quantitative(np.eye(3),True,{'points':[]},task())
    H=np.array([[1,0,2],[0,1,3],[0,0,1]]);d=document();q=quantitative(H,True,d,task());assert q['n_checks']==4 and q['error_mean_px']<1e-10 and q['sparse_Success@3px']
    for p in d['points'][4:]:p['xy1'][0]+=10;p['review']['xy1'][0]+=10
    assert quantitative(H,True,d,task())['error_mean_px']<1e-10 # deliberately wrong controls cannot leak into checks
    bad=quantitative(np.eye(3),True,document(),task());assert bad['error_mean_px']==pytest.approx(np.sqrt(13)) and not bad['sparse_Success@3px']

def test_store_versions_edit_review_and_stale_revision(tmp_path):
    s=new_store(tmp_path);ps=points();saved=s.save(payload(ps));assert saved['revision']==1 and saved['gt_quality']['status']!='gt_qualified';assert not any(p.get('review') for p in saved['points']) # client cannot forge review
    assert s.load('SIMULATED_ONLY')==saved
    with pytest.raises(ValueError,match='stale'):s.save(payload(ps))
    p=saved['points'][0];r=dict(sample_id='SIMULATED_ONLY',actor='bob',expected_revision=1,point_id=p['point_id'],xy0=p['xy0'],xy1=p['xy1'],sigma0_px=.2,sigma1_px=.2,conclusion='confirmed',static_confirmed=True,visible_confirmed=True)
    second=s.review(r);assert second['points'][0]['review']['reviewer']=='bob';assert s.load('SIMULATED_ONLY')==second
    changed=copy.deepcopy(second['points']);changed[0]['xy0'][0]+=.2;third=s.save(payload(changed,2));assert 'review' not in third['points'][0];assert len(list((tmp_path/'annotations/SIMULATED_ONLY').glob('v*.json')))==3
    with pytest.raises(ValueError,match='status'):s.save(dict(payload(changed,3),gt_status='gt_qualified'))
    r.update(actor='alice',expected_revision=3)
    with pytest.raises(ValueError,match='reviewer'):s.review(r)

def test_assistance_is_permanent_and_not_imported(tmp_path):
    s=new_store(tmp_path);s.expose('SIMULATED_ONLY','alice','geographic');assert s.load('SIMULATED_ONLY')['points']==[]
    saved=s.save(payload(points()));assert all(p['used_assistance'] for p in saved['points']);assert saved['gt_quality']['eligible_checks']==0

def test_history_tampering_and_source_id(tmp_path):
    s=new_store(tmp_path);s.save(payload(points()));f=tmp_path/'annotations/SIMULATED_ONLY/v000001.json';data=json.loads(f.read_text());data['points'][0]['xy0'][0]+=1;f.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='hash chain'):s.load('SIMULATED_ONLY')
    with pytest.raises(ValueError):s.task('../escape')

def test_actual_geographic_center_mapping_and_distance():
    tasks=Store(ROOT).tasks;cap=next(t for t in tasks.values() if t['sensor']=='Capella');M=ground_jacobian(cap);assert M is not None and M.shape==(2,2)
    geo=json.loads((ROOT/'candidate_matches'/f"{cap['sample_id']}.json").read_text())['geographic'];assert geo
    p=geo[0];assert np.max(np.abs(native_mapping(cap)(np.array([p['xy0']]))[0]-p['xy1']))<1e-8
    H=np.array([[1,0,2],[0,1,3],[0,0,1]]);q=quantitative(H,True,document(),task(),ground_metric=np.diag([2,3]));assert q['ground_error_mean_m']==0

def test_local_geometry_conflict_does_not_define_qualification():
    d=document()
    for p in d['points'][4:]:p['xy1'][0]+=25;p['review']['xy1'][0]+=25
    q=quality(d,task());assert q['status']=='gt_qualified';assert q['fit_check_diagnostic']['check_mean_px']>10;assert q['warnings']

def test_not_evaluable_requires_reason():
    d={'points':[],'disposition':'no_static_common_structure'}
    with pytest.raises(ValueError,match='reason'):quality(d,task())
    d['disposition_note']='SIMULATED no static scene';assert quality(d,task())['status']=='not_evaluable'

def test_end_to_end_evaluation_denominator_and_check_only(tmp_path,monkeypatch):
    # This explicitly simulated fixture has no GeoTIFF; real sensor geocoding
    # is exercised separately above, without using map references as GT.
    import run
    monkeypatch.setattr(run,'ground_jacobian',lambda _:None)
    s=new_store(tmp_path)
    row=dict(sample_id='SIMULATED_ONLY',pair_id='SIMULATED_PAIR',method='SIMULATED_METHOD',protocol='native',sensor='SIMULATED',visual_scene='SIMULATED',look_pair_category='SIMULATED',component='SIMULATED',candidate_H=[[1,0,2],[0,1,3],[0,0,1]],accepted=True,output_available=True,end_to_end_ms=1.,reason='simulated')
    inp=tmp_path/'simulated_predictions.jsonl';inp.write_text(json.dumps(row)+'\n');evaluate_saved(tmp_path,inp,tmp_path/'no_gt_evaluation');result=json.loads((tmp_path/'no_gt_evaluation/evaluation.json').read_text());assert not result['formal_accuracy_computed'] and 'error_mean_px' not in result['records'][0]
    saved=s.save(payload(points()))
    for p in saved['points'][:4]:
        saved=s.review(dict(sample_id='SIMULATED_ONLY',point_id=p['point_id'],expected_revision=saved['revision'],actor='bob',xy0=p['xy0'],xy1=p['xy1'],sigma0_px=.2,sigma1_px=.2,conclusion='confirmed',static_confirmed=True,visible_confirmed=True,note='SIMULATED TEST ONLY'))
    evaluate_saved(tmp_path,inp,tmp_path/'with_gt_evaluation');result=json.loads((tmp_path/'with_gt_evaluation/evaluation.json').read_text());assert result['formal_accuracy_computed'] and result['records'][0]['n_checks']==4 and result['records'][0]['error_mean_px']==0
    summary=json.loads((tmp_path/'with_gt_evaluation/stratified.json').read_text())[0];assert summary['sparse_Success@3px_denominator']==1 and summary['sparse_Success@3px_rate']==1

def test_http_loopback_guards_and_no_candidate_autofill(tmp_path):
    s=new_store(tmp_path);server=make_server(tmp_path,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}'
    # localhost must not be routed through server-wide outbound HTTP proxies.
    urlopen=urllib.request.build_opener(urllib.request.ProxyHandler({})).open
    try:
        page=urlopen(url).read().decode();assert 'SAR_TOKEN' in page and '独立标注' in page
        current=json.load(urlopen(url+'/api/annotation?sample_id=SIMULATED_ONLY'));assert current['points']==[]
        request=urllib.request.Request(url+'/api/save',data=json.dumps(payload(points())).encode(),headers={'Content-Type':'application/json'})
        with pytest.raises(urllib.error.HTTPError) as e:urlopen(request)
        assert e.value.code==400
        request=urllib.request.Request(url+'/api/tasks',headers={'Host':'attacker.invalid'})
        with pytest.raises(urllib.error.HTTPError):urlopen(request)
    finally:server.shutdown();server.server_close();thread.join()
