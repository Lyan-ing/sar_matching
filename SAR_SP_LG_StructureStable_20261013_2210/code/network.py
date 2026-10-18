"""Three fixed log-gradient scales; small zero-initial residual descriptor branch."""
import copy,math
from ss_common import *
from torch import nn
from torch.nn import functional as F
from lightglue import SuperPoint
from lightglue.superpoint import sample_descriptors,simple_nms
from m3_model import score_map
M2=FIX/'runs/m2/best.ckpt'
class Structure(nn.Module):
 def __init__(self):
  super().__init__()
  for i,sigma in enumerate([1.,2.,4.]):
   r=int(3*sigma);x=torch.arange(-r,r+1,dtype=torch.float32);g=torch.exp(-x*x/(2*sigma*sigma));g/=g.sum();self.register_buffer(f'g{i}',(g[:,None]*g[None])[None,None])
  self.register_buffer('sx',torch.tensor([[-3.,0,3],[-10,0,10],[-3,0,3]])[None,None]/32)
  self.register_buffer('sy',self.sx.transpose(-1,-2).clone())
  self.encode=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,256,1))
  nn.init.zeros_(self.encode[-1].weight);nn.init.zeros_(self.encode[-1].bias)
 def responses(self,image):
  x=torch.log1p(9*image)/math.log(10);rs=[]
  for i in range(3):
   g=getattr(self,f'g{i}');r=g.shape[-1]//2;y=F.conv2d(F.pad(x,(r,r,r,r),mode='replicate'),g)
   gx=F.conv2d(F.pad(y,(1,1,1,1),mode='replicate'),self.sx);gy=F.conv2d(F.pad(y,(1,1,1,1),mode='replicate'),self.sy)
   rs.append((torch.sqrt(gx.square()+gy.square()+1e-12)-1e-6).clamp_min(0))
  return torch.cat(rs,1)
 def forward(self,image):return self.encode(F.avg_pool2d(self.responses(image),8,8))
class SARNet(nn.Module):
 def __init__(self,variant):
  super().__init__();assert variant in ['m0','m2','control','a','b'];self.variant=variant
  self.sp=SuperPoint(max_num_keypoints=1024,detection_threshold=.0005).eval()
  if variant!='m0':
   c=torch.load(M2,map_location='cpu',weights_only=False);self.sp.load_state_dict({k.removeprefix('sp.'):v for k,v in c['model_state'].items() if k.startswith('sp.')},strict=True)
  self.sp.requires_grad_(False)
  if variant in ['a','b']:self.structure=Structure()
  if variant=='b':self.sp.convPa.requires_grad_(True);self.sp.convPb.requires_grad_(True)
  if variant=='control':self.sp.convDa.requires_grad_(True);self.sp.convDb.requires_grad_(True)
 def train(self,mode=True):super().train(mode);self.sp.eval();return self
 @torch.no_grad()
 def shared(self,image):
  s=self.sp;x=s.relu(s.conv1a(image));x=s.relu(s.conv1b(x));x=s.pool(x);x=s.relu(s.conv2a(x));x=s.relu(s.conv2b(x));x=s.pool(x);x=s.relu(s.conv3a(x));x=s.relu(s.conv3b(x));x=s.pool(x);return s.relu(s.conv4b(s.relu(s.conv4a(x))))
 def maps(self,image):
  x=self.shared(image);s=self.sp;logits=s.convPb(s.relu(s.convPa(x)));h=s.convDa(x)
  if self.variant in ['a','b']:h=h+self.structure(image)
  return logits,F.normalize(s.convDb(s.relu(h)),dim=1)
 def dense(self,image):
  d=self.maps(image)[1];return d,d
 def descriptors(self,image,points):
  d=self.maps(image)[1];d=sample_descriptors(points,d,8).transpose(1,2);return d,d
 @torch.no_grad()
 def extract(self,image):
  if max(image.shape[-2:])>2048:return self.tiled(image)
  logits,d=self.maps(image);s=score_map(logits);k,v=self.keypoints(s);desc=sample_descriptors(k[None],d,8).transpose(1,2)
  return dict(keypoints=k[None],keypoint_scores=v[None],descriptors=desc,image_size=torch.tensor([[image.shape[-1],image.shape[-2]]],device=image.device))
 def keypoints(self,s):
  s=simple_nms(s,4);s[:,:4]=-1;s[:,-4:]=-1;s[:,:,:4]=-1;s[:,:,-4:]=-1
  _,y,x=torch.where(s>.0005);v=s[0,y,x]
  if len(v)>1024:v,i=torch.topk(v,1024);x,y=x[i],y[i]
  return torch.stack([x,y],1).float(),v
 @torch.no_grad()
 def tiled(self,image,tile=1024,halo=64):
  # Fully native convolutions, global NMS, no geographic inputs. CPU holds only global scores.
  H,W=image.shape[-2:];fh,fw=H//8,W//8;scores=torch.zeros((1,fh*8,fw*8))
  def tiles():
   for y in range(0,fh*8,tile):
    for x in range(0,fw*8,tile):
     ey,ex=min(y+tile,fh*8),min(x+tile,fw*8);sy,sx=max(0,y-halo),max(0,x-halo);ty,tx=min(H,ey+halo),min(W,ex+halo)
     yield y,x,ey,ex,sy,sx,self.maps(image[...,sy:ty,sx:tx].cuda())
  for y,x,ey,ex,sy,sx,(logits,d) in tiles():scores[:,y:ey,x:ex]=score_map(logits)[:,y-sy:ey-sy,x-sx:ex-sx].cpu()
  k,v=self.keypoints(scores.cuda());kc=k.cpu();dd=torch.empty((len(k),256),device='cuda');coord=(kc-3.5)/torch.tensor([fw*8-4.5,fh*8-4.5])*torch.tensor([fw-1,fh-1])
  for y,x,ey,ex,sy,sx,(_,d) in tiles():
   keep=(kc[:,0]>=x)&(kc[:,0]<ex)&(kc[:,1]>=y)&(kc[:,1]<ey)
   if not keep.any():continue
   q=coord[keep]-torch.tensor([sx//8,sy//8]);grid=q/torch.tensor([d.shape[-1]-1,d.shape[-2]-1])*2-1
   dd[keep.cuda()]=F.normalize(F.grid_sample(d,grid.cuda()[None,None],align_corners=True)[0,:,0].T,dim=1)
  return dict(keypoints=k[None],keypoint_scores=v[None],descriptors=dd[None],image_size=torch.tensor([[W,H]],device='cuda'))
def load(variant,checkpoint=None):
 m=SARNet(variant).cuda()
 if checkpoint:m.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model_state'],strict=True)
 return m.eval()

def joint_loss(m,teacher,z):
 ims=torch.tensor(np.stack(z['images']).copy(),device='cuda').float()[:,None]/255
 logs,dense=m.maps(ims)
 with torch.no_grad():tl,td=teacher.maps(ims);tf=[teacher.extract(ims[i:i+1]) for i in [0,1]]
 # Actual detector locations, selected without response differentiability; geometry defines positives.
 with torch.no_grad():
  f=m.extract(ims[:1]) if m.variant=='b' else tf[0];p0=f['keypoints'][0].cpu().numpy();p1=warp(z['H_AB'],p0).astype(np.float32)
  keep=valid_at(z['static'][0],p0)&valid_at(z['static'][1],p1);p0,p1=p0[keep],p1[keep]
  if len(p0)>128:idx=np.linspace(0,len(p0)-1,128,dtype=int);p0,p1=p0[idx],p1[idx]
 ps=[torch.tensor(q,device='cuda')[None] for q in [p0,p1]]
 zero=dense.sum()*0;match=zero;preserve=zero
 if len(p0)>=2:
  ds=[sample_descriptors(q,dense[i:i+1],8)[0].T for i,q in enumerate(ps)];ts=[sample_descriptors(q,td[i:i+1],8)[0].T for i,q in enumerate(ps)]
  sim=ds[0]@ds[1].T/.1;near=(torch.cdist(ps[0][0],ps[0][0])<8)|(torch.cdist(ps[1][0],ps[1][0])<8);near.fill_diagonal_(False);sim=sim.masked_fill(near,-1e4);lab=torch.arange(len(p0),device='cuda')
  match=(F.cross_entropy(sim,lab)+F.cross_entropy(sim.T,lab))/2;preserve=sum((d-t).square().sum(1).mean() for d,t in zip(ds,ts))/2
 det=zero;equiv=zero;blank=zero;calib=zero
 if m.variant=='b':
  # Fixed teacher distribution retains class balance; weak structure anchors cannot overpower dustbin calibration.
  from m3_model import structure_targets
  anchors=structure_targets(z,teacher);lp=logs.log_softmax(1);tp=tl.softmax(1)
  masks=torch.tensor(np.stack(z['static']),device='cuda').float()[:,None];cells=F.avg_pool2d(masks,8,8)[:,0]>=.999
  # Includes padding suppression, excludes dynamic bbox. Invalid padding is explicitly negative, never positive.
  dynamic=torch.tensor(np.stack(z['ship_masks']),device='cuda').float()[:,None];nondynamic=F.max_pool2d(dynamic,8,8)[:,0]==0
  calib=(tp*(tl.log_softmax(1)-lp)).sum(1)[nondynamic].mean()
  struct=m.structure.responses(ims).detach().amax(1);structured=F.max_pool2d(struct[:,None],8,8)[:,0]>.012
  positive=torch.zeros_like(cells);targets=torch.full(cells.shape,64,device='cuda',dtype=torch.long)
  for i,q in enumerate(anchors):
   q=torch.tensor(np.round(q),device='cuda').long()
   if len(q):positive[i,q[:,1]//8,q[:,0]//8]=True;targets[i,q[:,1]//8,q[:,0]//8]=(q[:,1]%8)*8+q[:,0]%8
  positive &= cells & structured
  weak=F.nll_loss(lp,targets,reduction='none');det=weak[positive].mean() if positive.any() else logs.sum()*0
  negative=(~structured | ~cells)&nondynamic
  suppress=(-lp[:,64][negative]).mean() if negative.any() else logs.sum()*0
  # Consistency against teacher-guided peaks only. Student cannot fabricate self-supervised positives.
  if len(anchors[0]):
   vals=[];score=score_map(logs)
   for i,q in enumerate(anchors):
    grid=torch.tensor(q,device='cuda')[None,None]/255*2-1;vals.append(F.grid_sample(score[i:i+1,None],grid,align_corners=True)[0,0,0])
   equiv=(vals[0]-vals[1]).square().mean()
  blankims=torch.cat([torch.zeros_like(ims[:1]),torch.full_like(ims[:1],.5)],0)
  bl=m.maps(blankims)[0].log_softmax(1);blank=-bl[:,64].mean()
  det=.05*det+calib+suppress+blank+.2*equiv
 total=match+.1*preserve+det
 return total,dict(loss=float(total.detach()),match=float(match.detach()),preserve=float(preserve.detach()),detector=float(det.detach()),calibration=float(calib.detach()),blank=float(blank.detach()),equivariance=float(equiv.detach()),n_points=len(p0))
