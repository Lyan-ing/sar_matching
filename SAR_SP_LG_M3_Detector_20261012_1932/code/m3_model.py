"""M2 frozen encoder/descriptors, trainable original SuperPoint detector heads only."""
import copy
from m3_common import *
from torch import nn
from torch.nn import functional as F
from lightglue import SuperPoint
from lightglue.superpoint import sample_descriptors,simple_nms
M2=FIX/'runs/m2/best.ckpt'
M2_SHA='69e671f743adaf134983a3c01c3f9f1a152c387f8a999218b089b422799ac1ef'
class M3(nn.Module):
 def __init__(self):
  super().__init__();assert sha(M2)==M2_SHA
  self.variant='m3';self.sp=SuperPoint(max_num_keypoints=1024,detection_threshold=.0005).eval()
  c=torch.load(M2,map_location='cpu',weights_only=False)
  self.sp.load_state_dict({k.removeprefix('sp.'):v for k,v in c['model_state'].items() if k.startswith('sp.')},strict=True)
  self.sp.requires_grad_(False)
  self.teacherPa=copy.deepcopy(self.sp.convPa).requires_grad_(False);self.teacherPb=copy.deepcopy(self.sp.convPb).requires_grad_(False)
  self.sp.convPa.requires_grad_(True);self.sp.convPb.requires_grad_(True)
 def train(self,mode=True):super().train(mode);self.sp.eval();self.teacherPa.eval();self.teacherPb.eval();return self
 @torch.no_grad()
 def shared(self,image):
  s=self.sp;x=s.relu(s.conv1a(image));x=s.relu(s.conv1b(x));x=s.pool(x)
  x=s.relu(s.conv2a(x));x=s.relu(s.conv2b(x));x=s.pool(x)
  x=s.relu(s.conv3a(x));x=s.relu(s.conv3b(x));x=s.pool(x)
  return s.relu(s.conv4b(s.relu(s.conv4a(x))))
 def responses(self,image):
  x=self.shared(image);s=self.sp;logits=s.convPb(s.relu(s.convPa(x)))
  with torch.no_grad():teacher=self.teacherPb(s.relu(self.teacherPa(x)))
  return logits,teacher
 @torch.no_grad()
 def dense(self,image):
  x=self.shared(image);d=F.normalize(self.sp.convDb(self.sp.relu(self.sp.convDa(x))),dim=1);return d,d
 @torch.no_grad()
 def descriptors(self,image,points):
  d,_=self.dense(image);s=sample_descriptors(points,d,8).transpose(1,2);return s,s
 @torch.no_grad()
 def extract(self,image):return self.sp.extract(image,resize=None)
def score_map(logits):return F.pixel_shuffle(logits.softmax(1)[:,:64],8)[:,0]
def load_m3(path=None):
 m=M3().cuda()
 if path:
  c=torch.load(path,map_location='cuda',weights_only=False);assert c['variant']=='m3';m.load_state_dict(c['model_state'],strict=True)
 return m.eval()

def structure_targets(z,model):
 """Canonical image corners + known geometry; never student detections as labels."""
 r=z['source'];x,y=z['source_origin_xy'];side=384;raw=cv2.imread(r['image_path'],0)[y:y+side,x:x+side]
 im=np.zeros((side,side),np.float32);im[:raw.shape[0],:raw.shape[1]]=raw/255.
 smooth=cv2.GaussianBlur(im,(0,0),1.2);eig=cv2.cornerMinEigenVal(smooth,5,3)
 mask=(eig>=.00002).astype(np.uint8)*255
 p=cv2.goodFeaturesToTrack(smooth,256,.01,8,mask=mask,blockSize=5)
 p=np.empty((0,2),np.float32) if p is None else p[:,0]
 p0,p1=warp(z['H_A'],p),warp(z['H_B'],p)
 keep=valid_at(z['static'][0],p0)&valid_at(z['static'][1],p1)
 for q in [p0,p1]:keep&=(q[:,0]>=8)&(q[:,0]<248)&(q[:,1]>=8)&(q[:,1]<248)
 p0,p1=p0[keep].astype(np.float32),p1[keep].astype(np.float32)
 if len(p0):
  ims=[torch.tensor(a.copy(),device='cuda',dtype=torch.float32)[None,None]/255 for a in z['images']]
  with torch.no_grad():d=[model.descriptors(a,torch.tensor(q,device='cuda')[None])[0][0] for a,q in zip(ims,[p0,p1])]
  ok=((d[0]*d[1]).sum(1)>=.60).cpu().numpy();p0,p1=p0[ok],p1[ok]
 # At most four fixed geometry anchors per reference 4x4 cell; do not force empty cells.
 chosen=[];counts={};cells=[set(),set()]
 for i,(a,b) in enumerate(zip(p0,p1)):
  grid=tuple((a//64).astype(int));ca=tuple(np.round(a).astype(int)//8);cb=tuple(np.round(b).astype(int)//8)
  if counts.get(grid,0)>=4 or ca in cells[0] or cb in cells[1]:continue
  chosen.append(i);counts[grid]=counts.get(grid,0)+1;cells[0].add(ca);cells[1].add(cb)
 return p0[chosen],p1[chosen]

def detector_loss(model,z,config,anchors=None):
 p0,p1=structure_targets(z,model) if anchors is None else anchors
 ims=torch.tensor(np.stack(z['images']).copy(),device='cuda',dtype=torch.float32)[:,None]/255
 logits,teacher=model.responses(ims);lp=logits.log_softmax(1);tp=teacher.softmax(1)
 mask=torch.tensor(np.stack(z['static']),device='cuda',dtype=torch.float32)[:,None]
 cells=(F.avg_pool2d(mask,8,8)[:,0]>=.999)
 target=torch.full((2,32,32),64,device='cuda',dtype=torch.long);pos=torch.zeros_like(cells)
 for i,p in enumerate([p0,p1]):
  q=torch.tensor(np.round(p),device='cuda',dtype=torch.long)
  if len(q):target[i,q[:,1]//8,q[:,0]//8]=(q[:,1]%8)*8+(q[:,0]%8);pos[i,q[:,1]//8,q[:,0]//8]=True
 positive=pos&cells;negative=(~pos)&cells
 ce=F.nll_loss(lp,target,reduction='none')
 zero=logits.sum()*0
 peak=ce[positive].mean() if positive.any() else zero
 background=ce[negative].mean() if negative.any() else zero
 preserve=(tp*(teacher.log_softmax(1)-lp)).sum(1)[cells].mean() if cells.any() else zero
 # Corresponding 5x5 local response neighborhoods, accurately projected pixel centers.
 equiv=zero
 if len(p0):
  offs=np.array([(x,y) for y in range(-2,3) for x in range(-2,3)],np.float32)
  q0=(p0[:,None]+offs).reshape(-1,2);q1=warp(z['H_AB'],q0).astype(np.float32)
  keep=valid_at(z['static'][0],q0)&valid_at(z['static'][1],q1)
  scores=score_map(logits)
  vals=[]
  for i,q in enumerate([q0[keep],q1[keep]]):
   grid=torch.tensor(q,device='cuda')[None,None]/255*2-1
   vals.append(F.grid_sample(scores[i:i+1,None],grid,align_corners=True,padding_mode='zeros')[0,0,0].clamp_min(1e-6).log())
  if len(vals[0]):equiv=(vals[0]-vals[1]).square().mean()
 total=peak+config['background_weight']*background+config['equivariance_weight']*equiv+config['preserve_weight']*preserve
 return total,dict(loss=float(total.detach()),peak=float(peak.detach()),background=float(background.detach()),equivariance=float(equiv.detach()),preserve=float(preserve.detach()),anchors=len(p0),positive_cells=int(positive.sum()),static_cells=int(cells.sum()))

@torch.no_grad()
def tiled_extract(sp,image,tile=1024,halo=64):
 """Exact convolution tiles, global unchanged NMS/topK; sparse global-coordinate descriptor sampling."""
 # Inputs remain native pixel grids. Padding occurs only at true image boundaries.
 H,W=image.shape[-2:];fh,fw=H//8,W//8
 scores=torch.zeros((1,fh*8,fw*8),dtype=torch.float32)
 def shared(a):
  x=sp.relu(sp.conv1a(a));x=sp.relu(sp.conv1b(x));x=sp.pool(x);x=sp.relu(sp.conv2a(x));x=sp.relu(sp.conv2b(x));x=sp.pool(x);x=sp.relu(sp.conv3a(x));x=sp.relu(sp.conv3b(x));x=sp.pool(x);return sp.relu(sp.conv4b(sp.relu(sp.conv4a(x))))
 for y in range(0,fh*8,tile):
  for x in range(0,fw*8,tile):
   ey,ex=min(y+tile,fh*8),min(x+tile,fw*8);sy,sx=max(y-halo,0),max(x-halo,0)
   ty,tx=min(ey+halo,H),min(ex+halo,W)
   a=image[...,sy:ty,sx:tx].cuda();f=shared(a);s=score_map(sp.convPb(sp.relu(sp.convPa(f))))
   scores[:,y:ey,x:ex]=s[:,y-sy:ey-sy,x-sx:ex-sx].cpu()
 scores=simple_nms(scores.cuda(),sp.conf.nms_radius);scores[:,:4]=-1;scores[:,-4:]=-1;scores[:,:,:4]=-1;scores[:,:,-4:]=-1
 _,ys,xs=torch.where(scores>sp.conf.detection_threshold);v=scores[0,ys,xs]
 if len(v)>1024:v,idx=torch.topk(v,1024);xs,ys=xs[idx],ys[idx]
 kp=torch.stack([xs,ys],1).float().cpu();d=torch.empty((len(kp),256),dtype=torch.float32,device='cuda');del scores
 # Official sample_descriptors uses this image-wide align_corners coordinate convention.
 coord=(kp-3.5)/torch.tensor([fw*8-4.5,fh*8-4.5])*torch.tensor([fw-1,fh-1])
 for y in range(0,fh*8,tile):
  for x in range(0,fw*8,tile):
   keep=(kp[:,0]>=x)&(kp[:,0]<x+tile)&(kp[:,1]>=y)&(kp[:,1]<y+tile)
   if not keep.any():continue
   ey,ex=min(y+tile,fh*8),min(x+tile,fw*8);sy,sx=max(y-halo,0),max(x-halo,0);ty,tx=min(ey+halo,H),min(ex+halo,W)
   f=shared(image[...,sy:ty,sx:tx].cuda());dense=F.normalize(sp.convDb(sp.relu(sp.convDa(f))),dim=1)
   c=coord[keep]-torch.tensor([sx//8,sy//8]);grid=c/torch.tensor([dense.shape[-1]-1,dense.shape[-2]-1])*2-1
   dd=F.grid_sample(dense,grid.cuda()[None,None],align_corners=True)[0,:,0].T
   d[keep.cuda()]=F.normalize(dd,dim=1)
 return dict(keypoints=kp.cuda()[None],keypoint_scores=v.cuda()[None],descriptors=d[None],image_size=torch.tensor([[W,H]],device='cuda',dtype=torch.float32))
