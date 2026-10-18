"""M1 near-zero descriptor adapter; M2 descriptor heads only; fixed SP detector."""
import copy,torch
from torch import nn
from torch.nn import functional as F
from lightglue import SuperPoint,LightGlue
from lightglue.superpoint import sample_descriptors
from common import ROOT,BASE
class SARModel(nn.Module):
 def __init__(self,variant):
  super().__init__();assert variant in ['m0','m1','m2'];self.variant=variant
  for name in ['superpoint_v1.pth','superpoint_lightglue_v0-1_arxiv.pth']:
   assert (BASE/'MaritimeSARRegistrationPhase2/cache/torch/hub/checkpoints'/name).exists(),'Official frozen weights must already exist'
  self.sp=SuperPoint(max_num_keypoints=1024,detection_threshold=.0005).eval();self.sp.requires_grad_(False)
  if variant=='m1':
   self.adapter=nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Linear(64,256));nn.init.zeros_(self.adapter[-1].weight);nn.init.zeros_(self.adapter[-1].bias)
  if variant=='m2':
   self.teacherDa=copy.deepcopy(self.sp.convDa).requires_grad_(False);self.teacherDb=copy.deepcopy(self.sp.convDb).requires_grad_(False)
   self.sp.convDa.requires_grad_(True);self.sp.convDb.requires_grad_(True)
 def dense(self,image):
  s=self.sp
  with torch.no_grad():
   x=s.relu(s.conv1a(image));x=s.relu(s.conv1b(x));x=s.pool(x);x=s.relu(s.conv2a(x));x=s.relu(s.conv2b(x));x=s.pool(x);x=s.relu(s.conv3a(x));x=s.relu(s.conv3b(x));x=s.pool(x);x=s.relu(s.conv4a(x));x=s.relu(s.conv4b(x))
   if self.variant!='m2':base=F.normalize(s.convDb(s.relu(s.convDa(x))),dim=1)
   else:base=F.normalize(self.teacherDb(s.relu(self.teacherDa(x))),dim=1)
  if self.variant=='m2':
   # FP32 trainable head prevents reproduced FP16 scaled-backward overflow.
   with torch.autocast('cuda',enabled=False):learned=F.normalize(s.convDb(s.relu(s.convDa(x.float()))),dim=1)
  else:learned=base
  return learned,base
 def descriptors(self,image,points):
  learned,base=self.dense(image);d=sample_descriptors(points,learned.float(),8).transpose(1,2);teacher=sample_descriptors(points,base.float(),8).transpose(1,2)
  if self.variant=='m1':d=F.normalize(d+.1*self.adapter(d),dim=-1)
  return d,teacher
 @torch.no_grad()
 def extract(self,image):
  f=self.sp.extract(image,resize=None)
  if self.variant=='m1':f['descriptors']=F.normalize(f['descriptors']+.1*self.adapter(f['descriptors']),dim=-1)
  return f
 def train(self,mode=True):
  super().train(mode);self.sp.eval();return self
def loss(model,pair,device,temperature=.1,preserve=.1,amp=True):
 from data import valid_at
 import numpy as np
 p0,p1=pair['points0'],pair['points1']
 if len(p0)<16:return None
 ims=[torch.from_numpy(v.copy()).to(device).float()[None,None]/255 for v in pair['images']];ps=[torch.from_numpy(p.copy()).to(device)[None] for p in [p0,p1]]
 with torch.autocast('cuda',enabled=amp,dtype=torch.float16):out=[model.descriptors(im,p) for im,p in zip(ims,ps)]
 d0,d1=[x[0].float()[0] for x in out];b0,b1=[x[1].float()[0] for x in out];logits=d0@d1.T/temperature;labels=torch.arange(len(p0),device=device)
 # Exclude geometrically nearby alternative positives from negative labels.
 near=(torch.cdist(ps[1][0],ps[1][0])<8)|(torch.cdist(ps[0][0],ps[0][0])<8);near.fill_diagonal_(False);logits=logits.masked_fill(near,-1e4)
 match=(F.cross_entropy(logits,labels)+F.cross_entropy(logits.T,labels))/2;regular=((d0-b0).square().sum(-1).mean()+(d1-b1).square().sum(-1).mean())/2
 return match+preserve*regular,dict(match=float(match.detach()),preserve=float(regular.detach()),recall1=float((logits.argmax(1)==labels).float().mean()),n_points=len(p0))
def load_model(variant,checkpoint=None,device='cuda'):
 m=SARModel(variant).to(device).eval()
 if checkpoint is not None:
  c=torch.load(checkpoint,map_location=device,weights_only=False);assert c['variant']==variant;m.load_state_dict(c['model_state'],strict=True)
 return m
