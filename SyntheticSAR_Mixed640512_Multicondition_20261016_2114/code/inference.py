"""Pure-image prediction entry point: two explicit image paths, no GT/manifest input."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
import argparse,json
from pathlib import Path
import cv2
import numpy as np

def predict(image_A,image_B):
    images=[cv2.imread(str(p),cv2.IMREAD_GRAYSCALE) for p in [image_A,image_B]]
    if any(a is None for a in images):raise ValueError('Cannot decode image')
    orb=cv2.ORB_create(nfeatures=1024,fastThreshold=12);features=[orb.detectAndCompute(a,None) for a in images];pairs=[]
    (ka,da),(kb,db)=features
    if da is not None and db is not None and len(da)>1 and len(db)>1:
        matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        forward=[m for m,n in matcher.knnMatch(da,db,k=2) if m.distance<.8*n.distance]
        back={m.queryIdx:m.trainIdx for m,n in matcher.knnMatch(db,da,k=2) if m.distance<.8*n.distance}
        pairs=[m for m in forward if back.get(m.trainIdx)==m.queryIdx]
    p=np.array([ka[m.queryIdx].pt for m in pairs],np.float64).reshape(-1,2);q=np.array([kb[m.trainIdx].pt for m in pairs],np.float64).reshape(-1,2)
    cv2.setRNGSeed(42);H=None;inlier=None
    if len(p)>=4:H,inlier=cv2.findHomography(p,q,cv2.RANSAC,3.)
    accepted=H is not None and bool(np.isfinite(H).all()) and int(inlier.sum())>=8
    return dict(accepted=accepted,H_AB=H.tolist() if H is not None else None,matches_A=p.tolist(),matches_B=q.tolist(),
        keypoints_A=[list(k.pt) for k in ka],keypoints_B=[list(k.pt) for k in kb],method='fixed_ORB_1024_mutual_ratio_.8_RANSAC3_min8',
        acceptance_is_image_only_proxy_not_physical_GT=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('image_A');p.add_argument('image_B');a=p.parse_args();print(json.dumps(predict(a.image_A,a.image_B)))
