import sys
sys.dont_write_bytecode=True
from core import *
import pytest
from affine import Affine

def test_amplitude_nodata_and_valid_zero():
    a=np.array([[0.,4.],[-9999.,np.nan]]);v=np.array([[1,1],[0,0]],bool)
    b,m=amplitude(a,v,'linear_power');assert m[0,0] and b[0,0]==0 and b[0,1]==2
    assert b[1,0]==0 and not m[1,1]
    with pytest.raises(ValueError):amplitude(a,np.ones_like(v),'linear_amplitude')

def test_descriptor_determinism_flat_and_masks():
    a=np.ones((512,512));v=np.ones_like(a,bool)
    p,d,m=descriptors(a,v,'linear_amplitude');assert len(p)==0 and m['zero_structure']>0
    y,x=np.indices(a.shape);a=1+x*.01+y*.02+np.sin(x/12)**2
    p,d,m=descriptors(a,v,'linear_amplitude');p2,d2,m2=descriptors(a,v,'linear_amplitude')
    assert len(p)<=256 and np.array_equal(p,p2) and np.array_equal(d,d2)
    assert np.allclose(np.linalg.norm(d,axis=1),1)
    v[250:300,250:300]=False;p3,d3,m3=descriptors(a,v,'linear_amplitude')
    assert len(p3)<len(p) and m3['invalid_support']>0

def test_geography_cross_crs_and_centers():
    A=Affine(1e-5,0,10,0,-1e-5,45);B=Affine(1,0,1113194,0,-1,5621522)
    m=PixelMap(A,'EPSG:4326',B,'EPSG:3857');p=np.array([[0,0],[511,511],[143.25,222.75]])
    assert np.max(np.abs(m.reverse()(m(p))-p))<1e-6
    xx,yy=geo.transform('EPSG:4326','EPSG:3857',[(A*(.5,.5))[0]],[(A*(.5,.5))[1]])
    expected=(~B)*(xx[0],yy[0]);assert np.allclose(m([[0,0]])[0],np.array(expected)-.5)

def test_window_square_boundary_in_moving_pixels():
    p0=np.array([[0.,0.]]);p1=np.array([[64,64],[64.001,0],[-64,-64],[0,65.]])
    m=lambda p:p
    assert geographical_window(p0,p1,m).tolist()==[[True,False,True,False]]

def test_mutual_cycle_and_no_candidates():
    p=np.c_[np.arange(12)*32+30,np.arange(12)*20+30];d=np.eye(12,dtype=np.float32)
    a,b,c,s=mutual_match(p,d,p+2,d);assert len(a)==12 and s['cycle_max_px']==0
    a,b,c,s=mutual_match(p,d,p+2,d,np.zeros((12,12),bool));assert len(a)==0 and s['forward_no_candidates']==12
    a,b,c,s=mutual_match(p,d,p+2,np.ones_like(d));assert len(a)==0 # ambiguous identical descriptors fail ratio

def test_roma_align_corners_false_conversion():
    normalized=np.array([[-1+1/512,-1+1/512],[1-1/512,1-1/512],[0,0]])
    assert np.allclose(roma_native_coordinates(normalized),[[0,0],[511,511],[255.5,255.5]])
    sys.path.insert(0,str(ROOT/'vendor/deps'));sys.path.insert(0,str(ROOT/'vendor/RoMa'))
    import torch
    from romatch.models.matcher import RegressionMatcher
    official=RegressionMatcher._to_pixel_coordinates(None,torch.tensor(normalized),512,512)
    assert np.allclose(official.numpy()-.5,roma_native_coordinates(normalized))

def test_shared_global_and_local_geometry():
    p=regular_grid((512,512),16);H=np.array([[1,.02,8],[-.01,1,-3],[0,0,1.]])
    q=perspective(H,p);e,d=fit(p,q)
    assert e.accepted and d['bidirectional_cycle_mean_px']<1e-3 and np.allclose(e.candidate_H,H,atol=1e-4)
    row={'sample_id':'SIMULATED_ONLY','sensor':'test','component':'tmp','look_pair_category':'test','site_name':'tmp'}
    r,t,c=diagnostics(row,'test','test',p,q,np.ones(len(p)),[np.ones((512,512),bool)]*2,lambda p:p,{})
    assert r['local_accepted_tiles']>0 and r['GT_evidence'].startswith('C=0')
    e,d=fit(p[:7],q[:7]);assert e.reason=='too_few_matches'

def test_predictor_interfaces_no_human_GT():
    import inspect
    assert list(inspect.signature(descriptors).parameters)==['raw','valid','units']
    assert 'check' not in inspect.signature(mutual_match).parameters
    assert 'control' not in inspect.signature(fit).parameters
    class Poison:
        def __array__(self,*args):raise AssertionError('human GT accessed')
    # Adding inaccessible external evidence does not affect the numeric functions.
    unused={'check_points':Poison(),'control_points':Poison()}
    p=np.array([[0,0],[32,0]]);d=np.eye(2,dtype=np.float32)
    a,b,c,s=mutual_match(p,d,p,d);assert len(a)==2

def test_future_evaluator_requires_independent_qualified_GT():
    import evaluate_sparse
    t={'sample_id':'SIMULATED_ONLY','width0':512,'height0':512,'width1':512,'height1':512}
    with pytest.raises(ValueError,match='gt_qualified'):
        evaluate_sparse.quantitative(np.eye(3),True,{'points':[]},t)
    import inspect
    source=inspect.getsource(evaluate_sparse)
    assert 'estimate(' not in source and 'fit(' not in source
