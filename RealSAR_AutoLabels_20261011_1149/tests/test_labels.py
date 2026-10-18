"""Simulated pseudo-quality cases; no simulated labels written to real data."""
import sys,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from affine import Affine
from labels import grade,predicted_native,hull_mask
from geo import PixelMap
from common import ROOT

def record():return {'accepted':True,'reason':'ok','n_native_valid_inliers':30,'native_valid_inlier_fraction':1.,'inlier_ratio':.8,'geo_difference_median_px':2,'geo_difference_p90_px':4,'distribution':{'native_inlier0':{'hull_fraction':.3,'grid_coverage':.5},'native_inlier1':{'hull_fraction':.3,'grid_coverage':.5}}}
def config():return json.loads((ROOT/'config.json').read_text())

def test_consensus_and_grade_are_not_GT():
    r=record();assert grade(r,True,config(),1)[0]=='general_candidate';assert grade(r,True,config(),2)[0]=='high_candidate';assert grade(r,False,config(),2)[0]=='unreliable_candidate'
    assert 'independent_reference' not in grade(r,True,config(),3)

def test_large_map_disagreement_and_invalid_pixels_reject():
    r=record();r['geo_difference_p90_px']=65;assert grade(r,True,config(),2)[0]=='unreliable_candidate'
    r=record();r['native_valid_inlier_fraction']=.94;assert grade(r,True,config(),2)[0]=='unreliable_candidate'
    r=record();r['distribution']['native_inlier0']['hull_fraction']=.009;assert grade(r,True,config(),2)[0]=='unreliable_candidate'
    r=record();r.update(accepted=False,reason='weak_spatial_support');assert grade(r,True,config(),2)[0]=='unreliable_candidate'

def test_prediction_frames_compose_correctly():
    m=PixelMap(Affine(2,0,10,0,-2,20),'EPSG:32610',Affine(1,0,10,0,-1,20),'EPSG:32610');p=np.array([[10,20],[20.5,31.2]]);H=[[1,0,3],[0,1,4],[0,0,1]]
    r={'candidate_H':H,'protocol':'geo_assisted'};assert np.allclose(predicted_native(r,p,m),m(p+[3,4]));r['protocol']='native';assert np.allclose(predicted_native(r,p,m),p+[3,4])

def test_support_not_whole_image_or_dense_GT():
    m=hull_mask([[20,20],[60,20],[60,60],[20,60]],(128,128));assert m[40,40] and not m[0,0] and m.mean()<.15
