"""Simulated mathematical fixtures never enter real index or label folders."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pytest
from affine import Affine
from geo import PixelMap,sample_valid,overlap,regular_grid,roundtrip

def test_half_pixel_centers_and_unequal_scale():
    m=PixelMap(Affine(2,0,100,0,-2,200),'EPSG:32610',Affine(1,0,100,0,-1,200),'EPSG:32610')
    assert np.allclose(m([[0,0],[10,20]]),[[.5,.5],[20.5,40.5]])
    assert np.allclose(m.reverse()(m([[1.2,3.4],[11,21]])),[[1.2,3.4],[11,21]])

def test_rotation_shear_affine_composition():
    a=Affine.rotation(31)*Affine.shear(8)*Affine.scale(2,-3);b=Affine.translation(10,-5)*Affine.rotation(-19)*Affine.scale(.8,-1.7);m=PixelMap(a,'EPSG:32610',b,'EPSG:32610');p=np.array([[0,0],[511,511],[122.32,444.4]]);q=m(p);H=m.affine_matrix();assert np.allclose(np.c_[p,np.ones(len(p))]@H.T,np.c_[q,np.ones(len(q))],atol=1e-8);assert roundtrip(m,p)['max_px']<1e-8

def test_different_crs_and_nonlinearity():
    a=Affine(.02,0,-123,0,-.02,40);b=Affine(1000,0,-13700000,0,-1000,5000000);m=PixelMap(a,'EPSG:4326',b,'EPSG:3857');p=np.array([[0,0],[100,100],[200,200]]);assert m.affine_matrix() is None;assert roundtrip(m,p)['max_px']<1e-8
    q=m(p);assert abs((q[2,1]-q[1,1])-(q[1,1]-q[0,1]))>1 # map projections cannot be assumed affine

def test_invalid_and_fractional_support():
    mask=np.ones((8,8),bool);mask[2,2]=False
    assert sample_valid(mask,[[0,0],[7,7],[8,1],[-1,2],[float('nan'),0]]).tolist()==[True,True,False,False,False]
    assert not sample_valid(mask,[[1.8,1.8]])[0];assert sample_valid(mask,[[1,1]])[0]
    assert sample_valid(mask,[[-1e-9,0]])[0]

def test_overlap_and_no_overlap():
    masks=[np.ones((16,16),bool),np.ones((16,16),bool)];m=PixelMap(Affine.identity(),'EPSG:32610',Affine.translation(8,0),'EPSG:32610');joint,_,stats=overlap(masks,m);assert all(s['joint_pixels']==128 for s in stats)
    m=PixelMap(Affine.identity(),'EPSG:32610',Affine.translation(100,0),'EPSG:32610');assert all(s['joint_pixels']==0 for s in overlap(masks,m)[2])

@pytest.mark.parametrize('A,B',[(Affine(0,0,0,0,0,0),Affine.identity()),(Affine.identity(),Affine(0,0,0,0,0,0))])
def test_singular_geotransforms_rejected(A,B):
    with pytest.raises(ValueError):PixelMap(A,'EPSG:32610',B,'EPSG:32610')

def test_grid_spacing_and_no_forced_padding():
    p=regular_grid((512,512),16);assert len(p)==256;assert np.allclose(p[0],[15.5,15.5]);assert np.allclose(p[-1],[495.5,495.5]);assert len(regular_grid((128,256),8))==64
