"""Reuse known geometry; native-sized source canvas replaces old small-image padding."""
from common import *
def pair(r,seed,regime='easy',size=256,dynamic_occlusion=False):
 rng=np.random.default_rng(seed);img=cv2.imread(r['image_path'],0)
 if img is None:raise ValueError('source unreadable')
 h,w=img.shape;side=768 if regime=='none' else min(size+64,h,w);x=int(rng.integers(max(w-side+1,1)));y=int(rng.integers(max(h-side+1,1)))
 raw=np.zeros((side,side),np.float32);valid=np.zeros((side,side),np.uint8);cw,ch=min(side,w-x),min(side,h-y);raw[:ch,:cw]=img[y:y+ch,x:x+cw]/255.;valid[:ch,:cw]=1
 static=valid.copy();ship=np.zeros_like(valid)
 for a in r['annotations']:
  bx,by,bw,bh=a['bbox_xywh'];x0,y0=max(0,int(bx-x)-5),max(0,int(by-y)-5);x1,y1=min(side,int(bx+bw-x)+6),min(side,int(by+bh-y)+6)
  if x1>x0 and y1>y0:static[y0:y1,x0:x1]=0;ship[y0:y1,x0:x1]=1
 def H(center,angle,scale,shift,projective):
  M=np.eye(3);M[:2]=cv2.getRotationMatrix2D(tuple(map(float,center)),angle,scale);M[:2,2]+=np.array([size/2,size/2])-center+shift;M[2,:2]=projective;return M
 center=np.array([side/2,side/2]);a=H(center,float(rng.uniform(-8,8)),float(rng.uniform(.95,1.05)),rng.uniform(-12,12,2),[0,0])
 if regime=='none':a=H(np.array([128,128]),0,1,[0,0],[0,0]);b=H(np.array([640,640]),0,1,[0,0],[0,0])
 else:
  limits={'easy':(15,22,0),'medium':(45,45,.00012),'low':(65,120,.00018)};ang,translation,proj=limits[regime]
  b=H(center,float(rng.uniform(-ang,ang)),float(rng.uniform(.85,1.15)),rng.uniform(-translation,translation,2),rng.uniform(-proj,proj,2))
 hab=b@np.linalg.inv(a);views=[];masks=[];stable=[];ships=[]
 for H0 in [a,b]:
  v=cv2.warpPerspective(raw,H0,(size,size),flags=cv2.INTER_LINEAR);mv=cv2.warpPerspective(valid,H0,(size,size),flags=cv2.INTER_NEAREST)>0;sv=cv2.warpPerspective(static,H0,(size,size),flags=cv2.INTER_NEAREST)>0;sm=cv2.warpPerspective(ship,H0,(size,size),flags=cv2.INTER_NEAREST)>0
  # Independent view radiometry/speckle; display data units already uint8-derived.
  v=np.clip(v,0,1)**rng.uniform(.75,1.3);v*=rng.uniform(.8,1.2)
  speckle=rng.gamma(30,1/30,v.shape);v=np.clip(v*np.sqrt(speckle),0,1)
  if dynamic_occlusion and sm.any():v[sm]=np.clip(rng.normal(float(v[mv].mean()),.04,int(sm.sum())),0,1)
  v[~mv]=0;views.append(np.round(v*255).astype(np.uint8));masks.append(mv);stable.append(sv);ships.append(sm)
 p=np.empty((0,2),np.float32);q=p.copy()
 yy,xx=np.mgrid[0:size,0:size];grid=np.c_[xx.ravel(),yy.ravel()];joint=valid_at(masks[0],grid)&valid_at(masks[1],warp(hab,grid))
 return dict(images=views,valid=masks,static=stable,ship_masks=ships,H_A=a,H_B=b,H_AB=hab,points0=p.astype(np.float32),points1=q.astype(np.float32),overlap=float(joint.mean()),source=r,source_origin_xy=[x,y],regime=regime,dynamic_occlusion=dynamic_occlusion,seed=seed,relative_rotation=float(np.degrees(np.arctan2(hab[1,0],hab[0,0]))))
