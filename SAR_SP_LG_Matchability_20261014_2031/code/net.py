"""Geometry-confirmed candidate-quality ranking; unchanged compact SuperPoint architecture."""
from common import *
from torch import nn
from torch.nn import functional as F
from lightglue import SuperPoint
from lightglue.superpoint import sample_descriptors,simple_nms
def score_map(logits):
 s=logits.softmax(1)[:,:64].permute(0,2,3,1);b,h,w,_=s.shape
 return s.reshape(b,h,w,8,8).permute(0,1,3,2,4).reshape(b,h*8,w*8)
class Net(nn.Module):
 def __init__(self,variant='joint'):
  super().__init__();self.variant=variant;self.sp=SuperPoint(max_num_keypoints=1024,detection_threshold=.0005).eval()
  if variant!='m0':
   ck=torch.load(FIX/'runs/m2/best.ckpt',map_location='cpu',weights_only=False);self.sp.load_state_dict({k[3:]:v for k,v in ck['model_state'].items() if k.startswith('sp.')},strict=True)
  self.sp.requires_grad_(False)
  if variant in ['joint','control','frozen']:
   for name in ['convPa','convPb','convDa','convDb']+(['conv4a','conv4b'] if variant!='frozen' else []):getattr(self.sp,name).requires_grad_(True)
 def train(self,mode=True):super().train(mode);self.sp.eval();return self
 def maps(self,image):
  s=self.sp
  with torch.no_grad():
   x=s.relu(s.conv1b(s.relu(s.conv1a(image))));x=s.pool(x);x=s.relu(s.conv2b(s.relu(s.conv2a(x))));x=s.pool(x);x=s.relu(s.conv3b(s.relu(s.conv3a(x))));x=s.pool(x)
  x=s.relu(s.conv4b(s.relu(s.conv4a(x))));logits=s.convPb(s.relu(s.convPa(x)));d=F.normalize(s.convDb(s.relu(s.convDa(x))),dim=1)
  return logits,d
 @torch.no_grad()
 def extract(self,image):
  assert max(image.shape[-2:])<=512,'Native <=512 only; no implicit downsampling'
  logits,d=self.maps(image);s=simple_nms(score_map(logits),4);s[:,:4]=-1;s[:,-4:]=-1;s[:,:,:4]=-1;s[:,:,-4:]=-1
  _,y,x=torch.where(s>.0005);v=s[0,y,x];ids=torch.argsort(v,descending=True,stable=True)[:1024];k=torch.stack([x[ids],y[ids]],1).float();v=v[ids]
  return dict(keypoints=k[None],keypoint_scores=v[None],descriptors=sample_descriptors(k[None],d,8).transpose(1,2),image_size=torch.tensor([[image.shape[-1],image.shape[-2]]],device=image.device))
def load_model(variant,path=None,device='cuda'):
 if variant=='b':
  from network_corrected import SARNet
  m=SARNet('b');m.load_state_dict(torch.load(STRUCT/'runs/b/best.ckpt',map_location='cpu',weights_only=False)['model_state'],strict=True);m.requires_grad_(False)
 else:
  m=Net(variant)
  if path:m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state'],strict=True)
 return m.to(device).eval()

def joint_loss(model,teacher,z,c):
 ims=torch.tensor(np.stack(z['images']).copy(),device='cuda').float()[:,None]/255
 logits,d=model.maps(ims);scores=score_map(logits)
 with torch.no_grad():
  tl,td=teacher.maps(ims);fs=[model.extract(ims[i:i+1]) for i in [0,1]];ks=[f['keypoints'][0] for f in fs]
  xy=[k.cpu().numpy() for k in ks];p=warp(z['H_AB'],xy[0]);dxy=np.linalg.norm(p[:,None]-xy[1][None],axis=2)
  visible=[valid_at(z['static'][0],xy[0])&valid_at(z['static'][1],p),valid_at(z['static'][1],xy[1])&valid_at(z['static'][0],warp(np.linalg.inv(z['H_AB']),xy[1]))]
  truth=(dxy<=3)&visible[0][:,None]&visible[1][None,:]
  i=np.flatnonzero(truth.any(1));j=np.argmin(dxy[i],axis=1) if len(i) else np.empty(0,int)
  if len(i)>128:a=np.linspace(0,len(i)-1,128,dtype=int);i,j=i[a],j[a]
 descriptors=[sample_descriptors(k[None],d[n:n+1],8)[0].T for n,k in enumerate(ks)]
 zero=logits.sum()*0;match=zero;distill=zero;ranking=zero;equiv=zero;calibration=zero
 if len(i)>=2:
  a,b=descriptors[0][i],descriptors[1][j];sim=a@b.T/c['temperature'];near=(torch.cdist(ks[0][i],ks[0][i])<8)|(torch.cdist(ks[1][j],ks[1][j])<8);near.fill_diagonal_(False);sim=sim.masked_fill(near,-1e4);target=torch.arange(len(i),device='cuda');match=(F.cross_entropy(sim,target)+F.cross_entropy(sim.T,target))/2
 # Whole-cell detector calibration retains pretraining; all ship-overlapping cells excluded.
 static=torch.tensor(np.stack(z['static']),device='cuda').float()[:,None];cells=F.avg_pool2d(static,8,8)[:,0]>.999
 kl=(tl.softmax(1)*(tl.log_softmax(1)-logits.log_softmax(1))).sum(1)
 cell_mass=tl.softmax(1)[:,:64].sum(1).detach().clamp_min(.01)
 if cells.any():distill=10*(kl/cell_mass)[cells].mean() # Class-balanced calibration; background mass cannot dilute the constraint.
 if len(i):
  teach=[sample_descriptors(ks[n][None],td[n:n+1],8)[0].T for n in [0,1]]
  distill=distill+sum((descriptors[n][idx]-teach[n][idx]).square().sum(1).mean() for n,idx in [(0,i),(1,j)])/2
 if model.variant=='control':
  # Ordinary geometrically warped response consistency at actual candidate positions.
  if len(i):equiv=(scores[0,ks[0][i,1].long(),ks[0][i,0].long()]-scores[1,ks[1][j,1].long(),ks[1][j,0].long()]).square().mean()
  quality=equiv
 else:
  # NMS/TopK/truth/quality targets are detached; only gathered response logits receive ranking gradients.
  with torch.no_grad():
   sim=descriptors[0].detach()@descriptors[1].detach().T
   truth_t=torch.tensor(truth,device='cuda');good=torch.where(truth_t,sim,torch.full_like(sim,-2.)).amax(1) if sim.shape[1] else torch.full((len(ks[0]),),-2.,device='cuda')
   bad=torch.where(~truth_t,sim,torch.full_like(sim,-2.)).amax(1) if sim.shape[1] else good
   q0=torch.tensor(truth.any(1),device='cuda').float()*(.25+.75*torch.sigmoid((good-bad)/.1))
   good1=torch.where(truth_t,sim,torch.full_like(sim,-2.)).amax(0) if sim.shape[0] else torch.full((len(ks[1]),),-2.,device='cuda');bad1=torch.where(~truth_t,sim,torch.full_like(sim,-2.)).amax(0) if sim.shape[0] else good1
   q1=torch.tensor(truth.any(0),device='cuda').float()*(.25+.75*torch.sigmoid((good1-bad1)/.1))
  # Preserve each teacher cell's keypoint-vs-dustbin mass. Geometry/descriptor quality
  # redistributes the existing non-dustbin mass instead of rewarding every location.
  quality_target=tl.softmax(1).clone().detach()
  for n,q in enumerate([q0,q1]):
   valid=torch.tensor(visible[n],device='cuda');q=q[valid];k=ks[n][valid].long()
   if len(k):
    label=(k[:,1]%8)*8+k[:,0]%8;quality_target[n,label,k[:,1]//8,k[:,0]//8]*=.25+1.5*q
  mass=tl.softmax(1)[:,:64].sum(1,keepdim=True).detach()
  quality_target[:,:64]*=mass/quality_target[:,:64].sum(1,keepdim=True).clamp_min(1e-12)
  # Equal cell mass is the anti-density-collapse constraint; discrete targets never carry gradients.
  qkl=(quality_target*(quality_target.clamp_min(1e-12).log()-logits.log_softmax(1))).sum(1)
  if cells.any():ranking=(qkl/(mass[:,0].clamp_min(.01))).clamp_min(0)[cells].mean()
  # Explicit empty/no-correspondence calibration is part of the quality objective, not positive SAR labels.
  noise=torch.rand_like(ims[:1]);blank=torch.zeros_like(ims[:1]) if z['seed']%2==0 else torch.full_like(ims[:1],.5)
  null_logits=model.maps(torch.cat([noise,blank],0))[0];calibration=-null_logits.log_softmax(1)[:,64].mean();quality=ranking+.25*calibration
 total=match+c['preserve_weight']*distill+c['quality_weight']*quality
 return total,dict(loss=float(total.detach()),descriptor=float(match.detach()),preserve=float(distill.detach()),ranking=float(ranking.detach()),empty=float(calibration.detach()),equiv=float(equiv.detach()),repeatable_pairs=len(i),kp0=len(ks[0]),kp1=len(ks[1]))
