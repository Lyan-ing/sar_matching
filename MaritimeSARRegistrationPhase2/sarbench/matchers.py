"""Image-only matching methods. No GT/masks are accepted by this interface."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import cv2

@dataclass
class Prediction:
    method: str
    points0: np.ndarray = field(default_factory=lambda: np.empty((0,2), dtype=np.float32))
    points1: np.ndarray = field(default_factory=lambda: np.empty((0,2), dtype=np.float32))
    confidence: float | None = None
    reason: str = ''
    model_type: str = 'homography'
    # phase only estimates translation; it should not be silently compared as unconstrained homography
    precomputed_H: np.ndarray | None = None
    n_keypoints0: int | None = None
    n_keypoints1: int | None = None


def _mutual_ratio(d0, d1, matcher, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    if d0 is None or d1 is None or len(d0)<2 or len(d1)<2:
        return np.empty((0,),dtype=np.int32),np.empty((0,),dtype=np.int32)
    forward = matcher.knnMatch(d0, d1, k=2)
    backward = matcher.knnMatch(d1, d0, k=2)
    def good_matches(mset):
        result = {}
        for row in mset:
            if len(row) >= 2 and row[0].distance < ratio*row[1].distance:
                result[row[0].queryIdx] = row[0].trainIdx
        return result
    f, b = good_matches(forward), good_matches(backward)
    matches = sorted((a, c) for a,c in f.items() if b.get(c) == a)
    if not matches:
        return np.empty((0,),dtype=np.int32),np.empty((0,),dtype=np.int32)
    return np.asarray(matches,dtype=np.int32).T

class LocalFeatureMatcher:
    def __init__(self, name: str, max_features: int = 2048, ratio: float = .8):
        if name not in ('sift','orb'):
            raise ValueError(name)
        self.name=name; self.ratio=ratio
        if name=='sift':
            if not hasattr(cv2,'SIFT_create'):
                raise RuntimeError('OpenCV build lacks SIFT; install a modern opencv-python-headless.')
            self.detector=cv2.SIFT_create(nfeatures=max_features)
            self.norm=cv2.NORM_L2
        else:
            self.detector=cv2.ORB_create(nfeatures=max_features, fastThreshold=12)
            self.norm=cv2.NORM_HAMMING

    def predict(self, img0: np.ndarray, img1: np.ndarray) -> Prediction:
        k0, d0 = self.detector.detectAndCompute(img0,None)
        k1, d1 = self.detector.detectAndCompute(img1,None)
        counts={'n_keypoints0':len(k0),'n_keypoints1':len(k1)}
        if not k0 or not k1:
            return Prediction(self.name, reason='no_descriptors',**counts)
        bf=cv2.BFMatcher(self.norm,crossCheck=False)
        ids0,ids1=_mutual_ratio(d0,d1,bf,self.ratio)
        if not len(ids0):
            return Prediction(self.name, reason='no_ratio_mutual_matches',**counts)
        p0=np.asarray([k0[int(i)].pt for i in ids0],dtype=np.float32)
        p1=np.asarray([k1[int(j)].pt for j in ids1],dtype=np.float32)
        return Prediction(self.name,p0,p1,**counts)

class PhaseCorrelation:
    name='phase'
    def predict(self, img0: np.ndarray, img1: np.ndarray) -> Prediction:
        # Translation-only diagnostic. Different input sizes may be used across
        # pairs, but both views of a pair must have equal dimensions.
        if img0.shape != img1.shape:
            raise ValueError('Phase correlation needs same-size views')
        h,w=img0.shape
        a=img0.astype(np.float32)
        b=img1.astype(np.float32)
        win=cv2.createHanningWindow((w,h),cv2.CV_32F)
        shift,response=cv2.phaseCorrelate(a,b,win)
        dx,dy=shift
        H=np.array([[1.,0.,dx],[0.,1.,dy],[0.,0.,1.]])
        return Prediction('phase',confidence=float(response),model_type='translation',precomputed_H=H)

class SuperPointLightGlue:
    name='lightglue'
    def __init__(self, device='auto', max_features=2048, resize=None):
        try:
            import torch
            from lightglue import SuperPoint, LightGlue
            from lightglue.utils import rbd
        except ImportError as e:
            raise RuntimeError('LightGlue/torch not installed; see README installation section.') from e
        if device=='auto':
            device='cuda' if torch.cuda.is_available() else 'cpu'
        if device=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA but torch.cuda.is_available() is False')
        self.torch=torch; self.rbd=rbd;self.device=torch.device(device)
        self.resize=resize
        try:
            self.extractor=SuperPoint(max_num_keypoints=max_features).eval().to(self.device)
            self.matcher=LightGlue(features='superpoint').eval().to(self.device)
        except Exception as e:
            raise RuntimeError('Could not initialize pretrained SuperPoint/LightGlue. Ensure official weights are cached or network download works.') from e

    def predict(self, img0: np.ndarray, img1: np.ndarray) -> Prediction:
        torch=self.torch
        a=torch.from_numpy(np.ascontiguousarray(img0).copy()).float().to(self.device)[None,None]/255.0
        b=torch.from_numpy(np.ascontiguousarray(img1).copy()).float().to(self.device)[None,None]/255.0
        # LightGlue extractor accepts a batched 1-channel tensor (B,1,H,W).
        # extract(resize=None) ensures keypoints stay in original pixel coordinates.
        with torch.inference_mode():
            f0=self.extractor.extract(a,resize=self.resize)
            f1=self.extractor.extract(b,resize=self.resize)
            m=self.matcher({'image0':f0,'image1':f1})
            f0,f1,m=[self.rbd(q) for q in (f0,f1,m)]
            counts={'n_keypoints0':len(f0['keypoints']),'n_keypoints1':len(f1['keypoints'])}
            match=m['matches']
            if match.numel()==0:
                return Prediction(self.name,reason='no_learned_matches',**counts)
            p0=f0['keypoints'][match[:,0]].detach().cpu().numpy().astype(np.float32)
            p1=f1['keypoints'][match[:,1]].detach().cpu().numpy().astype(np.float32)
        return Prediction(self.name,p0,p1,**counts)


def make_matcher(name: str, *, max_features=2048, ratio=.8, device='auto'):
    if name in ('sift','orb'):
        return LocalFeatureMatcher(name,max_features,ratio)
    if name=='phase':
        return PhaseCorrelation()
    if name=='lightglue':
        return SuperPointLightGlue(device,max_features,resize=None)
    raise ValueError(f'Unknown method: {name}')
