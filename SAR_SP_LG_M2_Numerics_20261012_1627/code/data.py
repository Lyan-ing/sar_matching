"""Online known geometry, native crop/no upsampling; ship-excluded supervision."""
import json,cv2,numpy as np
from common import ROOT,BASE,sha,readrows,write_new
INDEX=BASE/'MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl'
def component(r):return ('hrsid_scene:' if r['dataset'] in ['hrsid','background'] else 'ssdd_known_component:')+str(r['source_group'])
def warp(H,p):
 q=np.c_[p,np.ones(len(p))]@np.asarray(H).T
 return q[:,:2]/q[:,2:]
def valid_at(mask,p):
 p=np.asarray(p);x,y=np.floor(p).astype(int).T;good=np.isfinite(p).all(1)&(x>=0)&(y>=0)&(x+1<mask.shape[1])&(y+1<mask.shape[0]);ids=np.flatnonzero(good)
 good[ids]&=mask[y[ids],x[ids]]&mask[y[ids]+1,x[ids]]&mask[y[ids],x[ids]+1]&mask[y[ids]+1,x[ids]+1];return good
def setup():
 rows=readrows(INDEX);selected=[];bad=[]
 from PIL import Image
 for r in rows:
  if r['split'] not in ['train','val']:continue
  try:
   with Image.open(r['image_path']) as im:im.verify()
   im=cv2.imread(r['image_path'],cv2.IMREAD_GRAYSCALE)
   if im is None or min(im.shape)<48:raise ValueError('unreadable/too small')
   r=dict(r,training_component=component(r),native_width=int(im.shape[1]),native_height=int(im.shape[0]));selected.append(r)
  except Exception as e:bad.append(dict(source_id=r['source_id'],reason=repr(e)))
 sets={s:{component(r) for r in selected if r['split']==s} for s in ['train','val']};assert not sets['train']&sets['val']
 # The official repaired index controls source grouping; no reassignment from test.
 for s in ['train','val']:
  with (ROOT/(s+'_sources.jsonl')).open('x') as f:
   for r in selected:
    if r['split']==s:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 from collections import Counter
 report=dict(index=str(INDEX),index_sha256=sha(INDEX),readable_counts=dict(Counter(r['split']+':'+r['dataset'] for r in selected)),components={s:len(v) for s,v in sets.items()},overlap_components=[],failures=bad,
  caveat='SSDD component isolation follows existing pixel-supported repair; independent raw acquisitions not fully proven. HRSID/background same original scene joined.',formal900_read_for_selection=False)
 write_new(ROOT/'data_identity.json',report)
 # Fixed 56 validation pairs, source component diverse, before training.
 vals=[r for r in selected if r['split']=='val'];rng=np.random.default_rng(261012);used=set();chosen=[]
 pools=[[r for r in vals if r['scene_type']=='inshore'],[r for r in vals if r['dataset']=='hrsid' and r['scene_type']=='offshore'],[r for r in vals if r['dataset']=='background'],[r for r in vals if r['dataset']=='ssdd']]
 for pool in pools:
  order=rng.permutation(len(pool));n=0
  for j in order:
   r=pool[j];key=r['dataset']+':'+r['source_id']
   if key in used:continue
   used.add(key);chosen.append(r);n+=1
   if n==12:break
 assert len(chosen)==48
 plan=[]
 for i,r in enumerate(chosen):
  regime=['easy','medium','low'][i%3];plan.append(dict(id=f'val_{i:03d}',source=r,seed=71000+i,regime=regime,dynamic_occlusion=bool(r['annotations']) and i%4==0))
 large=[r for r in vals if r['native_width']>=700 and r['native_height']>=700]
 for i,r in enumerate(large[:8]):plan.append(dict(id=f'val_none_{i:03d}',source=r,seed=72000+i,regime='none',dynamic_occlusion=False))
 assert len(plan)==56;write_new(ROOT/'validation_plan.json',plan);print(json.dumps(report,ensure_ascii=False),flush=True)
def pair(r,seed,regime='easy',size=256,dynamic_occlusion=False):
 rng=np.random.default_rng(seed);img=cv2.imread(r['image_path'],0)
 if img is None:raise ValueError('source unreadable')
 h,w=img.shape;side=768 if regime=='none' else 384;x=int(rng.integers(max(w-side+1,1)));y=int(rng.integers(max(h-side+1,1)))
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
 ys,xs=np.mgrid[12:size-12:4,12:size-12:4];p=np.c_[xs.ravel(),ys.ravel()].astype(float);q=warp(hab,p);keep=valid_at(stable[0],p)&valid_at(stable[1],q);p,q=p[keep],q[keep]
 if len(p):
  grad=cv2.magnitude(cv2.Sobel(views[0],cv2.CV_32F,1,0),cv2.Sobel(views[0],cv2.CV_32F,0,1));scores=grad[p[:,1].astype(int),p[:,0].astype(int)];order=np.argsort(-(scores+rng.uniform(0,25,len(p))));chosen=[]
  for j in order:
   if not chosen or np.min(np.linalg.norm(p[j]-p[chosen],axis=1))>=12:chosen.append(int(j))
   if len(chosen)==128:break
  p,q=p[chosen],q[chosen]
 yy,xx=np.mgrid[0:size,0:size];grid=np.c_[xx.ravel(),yy.ravel()];joint=valid_at(masks[0],grid)&valid_at(masks[1],warp(hab,grid))
 return dict(images=views,valid=masks,static=stable,ship_masks=ships,H_A=a,H_B=b,H_AB=hab,points0=p.astype(np.float32),points1=q.astype(np.float32),overlap=float(joint.mean()),source=r,source_origin_xy=[x,y],regime=regime,dynamic_occlusion=dynamic_occlusion,seed=seed)
if __name__=='__main__':setup()
