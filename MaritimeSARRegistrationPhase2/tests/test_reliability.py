import numpy as np
from sarbench.matchers import Prediction,LocalFeatureMatcher
from sarbench.estimate import estimate,Estimated
from sarbench.reliability import projection_diagnostics,geometry_diagnostics,gate

def test_scale_invariant_pole_and_benign_translation():
 H=np.array([[1.,0,20],[0,1,-10],[0,-.01,1]])
 for scale in [1.,-7.,.001]:
  d=projection_diagnostics(H*scale,(384,384));assert not d['denominator_same_sign']
 T=np.array([[1.,0,50],[0,1,-30],[0,0,1]])
 d=projection_diagnostics(T,(384,384));assert d['denominator_same_sign'] and d['denominator_margin']==1 and abs(d['corner_area_ratio']-1)<1e-6

def test_independent_reverse_and_no_gt_gate():
 rng=np.random.default_rng(42);p=rng.uniform(25,300,(100,2)).astype(np.float32);q=p+np.array([9,-7],np.float32)
 pred=Prediction('sift',p,q);est=estimate(pred,(384,384));d=geometry_diagnostics(est,pred,(384,384),(384,384))
 assert est.accepted and d['bidirectional_cycle_mean_px']<1e-3
 assert gate(est,d,'full').accepted
 H=np.array([[1.,0,0],[0,1,0],[0,-.01,1]])
 bad=Estimated(H,True,'ok',10,10,1.,.1,.5,None,'homography',H)
 d={'forward_denominator_same_sign':False}
 assert not gate(bad,d,'pole').accepted and bad.accepted

def test_original_detectors_defaults_and_keypoint_metadata():
 image=np.random.default_rng(7).integers(0,256,(128,128),dtype=np.uint8)
 model=LocalFeatureMatcher('orb');assert model.detector.getEdgeThreshold()==31 and model.detector.getFastThreshold()==12
 pred=model.predict(image,image);assert pred.keypoints0.shape==(pred.n_keypoints0,2) and np.array_equal(pred.points0,pred.points1)
 changed=LocalFeatureMatcher('orb',detector_options={'edgeThreshold':16});assert changed.detector.getEdgeThreshold()==16
