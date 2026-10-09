import numpy as np
from sarreg.geometry import transform_points,view_matrix,relative_matrix,overlap_mask,shared_points,sample_views,warp
from sarreg.synthesis import load_cfg

def test_inverse_and_relative():
    rng=np.random.default_rng(42)
    H0=view_matrix((200,200),256,12,1.1,2,0.003,rng)
    H1=view_matrix((220,220),256,-15,.93,1,0.003,rng)
    H=relative_matrix(H0,H1)
    x=rng.uniform(20,150,size=(500,2))
    err=np.linalg.norm(transform_points(H,x)-transform_points(H1,transform_points(np.linalg.inv(H0),x)),axis=1)
    assert err.max()<1e-6
    composed=relative_matrix(H1,H0)@H
    composed/=composed[2,2]
    assert np.allclose(composed,np.eye(3),atol=1e-7)

def test_overlap_masks_and_points():
    size=128
    a=np.ones((size,size),dtype='uint8')*255
    H=np.array([[1,0,23],[0,1,-12],[0,0,1]],dtype=float)
    o0,o1=overlap_mask(a,a,H)
    assert np.count_nonzero(o0)>0
    assert np.count_nonzero(o0)==np.count_nonzero(o1)
    pi,pj=shared_points(H,o0,o1,100,np.random.default_rng(7))
    assert len(pi)>40
    assert np.max(np.linalg.norm(transform_points(H,pi)-pj,axis=1))<1e-4

def test_sample_views():
    c=load_cfg('configs/easy.yaml')
    c['output_size']=128
    c['min_source_side_factor']=1.35
    for seed in (0,4,10,23):
        matrices,valid=sample_views((350,350),c,np.random.default_rng(seed))
        assert len(matrices)==2
        a,b=overlap_mask(valid[0],valid[1],relative_matrix(matrices[0],matrices[1]))
        score=min(np.count_nonzero(a)/np.count_nonzero(valid[0]),np.count_nonzero(b)/np.count_nonzero(valid[1]))
        assert c['min_overlap']<=score<=c['max_overlap']
