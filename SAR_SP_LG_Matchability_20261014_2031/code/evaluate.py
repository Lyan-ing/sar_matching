from common import *
from net import load_model,score_map
from lightglue import LightGlue
from sarbench.matchers import Prediction,_mutual_ratio
from sarbench.estimate import estimate,perspective
from sarbench.reliability import geometry_diagnostics,gate,spatial
def truncate(f,n):
 ids=torch.argsort(f['keypoint_scores'][0],descending=True,stable=True)[:n]
 return {k:(v[:,ids] if k in ['keypoints','keypoint_scores','descriptors','octaves'] else v) for k,v in f.items()}
def geometry(p,sh0,sh1):
 e=estimate(p,sh0,model='homography',seed=42);d=geometry_diagnostics(e,p,sh0,sh1,42) if e.candidate_H is not None else {};return gate(e,d,'full'),d
class Evaluator:
 def __init__(self):self.lg=LightGlue(features='superpoint').eval().cuda().requires_grad_(False)
 @torch.no_grad()
 def features(self,method,model,im):
  if method.startswith('orb'):
   orb=cv2.ORB_create(nfeatures=1024,fastThreshold=12,nlevels=1 if method=='orb_single' else 8);k,d=orb.detectAndCompute(im,None);d=np.empty((0,32),np.uint8) if d is None else d
   return dict(keypoints=torch.tensor(np.array([p.pt for p in k],np.float32).reshape(-1,2),device='cuda')[None],keypoint_scores=torch.tensor([p.response for p in k],device='cuda')[None],descriptors=torch.tensor(d,device='cuda')[None],octaves=torch.tensor([p.octave for p in k],device='cuda')[None],image_size=torch.tensor([[im.shape[1],im.shape[0]]],device='cuda'))
  return model.extract(torch.tensor(im.copy(),device='cuda').float()[None,None]/255)
 @torch.no_grad()
 def predict(self,method,fs,shapes):
  # ONLY image-derived features and image shapes. No GT/metadata/masks accepted.
  if method.startswith('orb'):
   ds=[f['descriptors'][0].cpu().numpy() for f in fs];i,j=_mutual_ratio(*ds,cv2.BFMatcher(cv2.NORM_HAMMING),.8);ids=np.c_[i,j];scores=np.ones(len(i))
  else:
   out=self.lg(dict(image0=fs[0],image1=fs[1]));ids=out['matches'][0].cpu().numpy();scores=out['scores'][0].cpu().numpy()
  ks=[f['keypoints'][0].cpu().numpy() for f in fs];p=Prediction(method,ks[0][ids[:,0]],ks[1][ids[:,1]],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1]);e,d=geometry(p,*shapes);return p,e,d,scores
 def assess(self,method,p,e,d,fs,ims,gt,meta,ms):
  H=gt['H_AB'];static=[gt['static0'],gt['static1']];k0,k1=p.keypoints0,p.keypoints1;target=warp(H,k0);vis0=valid_at(static[0],k0)&valid_at(static[1],target);vis1=valid_at(static[1],k1)&valid_at(static[0],warp(np.linalg.inv(H),k1));dist=np.linalg.norm(target[:,None]-k1[None],axis=2);truth=(dist<=3)&vis0[:,None]&valid_at(static[1],k1)[None];rep=truth.any(1)
  ds=[f['descriptors'][0].cpu().numpy() for f in fs]
  if method.startswith('orb'):
   b0,b1=[np.unpackbits(x,axis=1).astype(np.float32) for x in ds];sim=1-(b0.sum(1)[:,None]+b1.sum(1)[None]-2*b0@b1.T)/256;norm=cv2.NORM_HAMMING
  else:sim=ds[0]@ds[1].T;norm=cv2.NORM_L2
  order=np.argsort(-sim,axis=1,kind='stable');rank={}
  for K in [1,5,10]:
   hit=np.take_along_axis(truth,order[:,:K],axis=1).any(1) if len(k1) else np.zeros(len(k0),bool);rank[f'conditional_R{K}']=float(hit[rep].mean()) if rep.any() else None;rank[f'end_to_end_R{K}']=float(hit[vis0].mean()) if vis0.any() else None
  i,j=_mutual_ratio(*ds,cv2.BFMatcher(norm),.8);nncorrect=truth[i,j] if len(i) else np.zeros(0,bool)
  predgt=warp(H,p.points0);correct=(np.linalg.norm(predgt-p.points1,axis=1)<=3)&valid_at(static[0],p.points0)&valid_at(static[1],p.points1)&valid_at(static[1],predgt);support=spatial(p.points0[correct],ims[0].shape)
  h,w=ims[0].shape;yy,xx=np.mgrid[12:h-12:12,12:w-12:12];q=np.c_[xx.ravel(),yy.ravel()];g=warp(H,q);keep=valid_at(static[0],q)&valid_at(static[1],g);epe=None
  if e.candidate_H is not None and keep.any():err=np.linalg.norm(perspective(e.candidate_H,q[keep])-g[keep],axis=1);epe=float(err.mean()) if np.isfinite(err).all() else None
  row=dict(**meta,method=method,n_keypoints0=len(k0),n_keypoints1=len(k1),n_matches=e.n_matches,n_inliers=e.n_inliers,accepted=e.accepted,reason=e.reason,decision='accepted' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),elapsed_ms=ms,geometry=d,repeatable=int(rep.sum()),visible_detected0=int(vis0.sum()),repeatability=float(rep.sum()/vis0.sum()) if vis0.any() else None,correct_matches=int(correct.sum()),match_precision=float(correct.mean()) if len(correct) else None,matching_recall=float(correct.sum()/vis0.sum()) if vis0.any() else None,correct_inliers=int((correct&e.inlier_mask).sum()) if e.inlier_mask is not None else 0,correct_hull=support['hull_fraction'],correct_grid=support['grid_coverage'],NN_matches=len(i),NN_correct=int(nncorrect.sum()),epe=epe,**rank)
  for t in [3,5,10]:row[f'success{t}']=bool(e.accepted and epe is not None and epe<=t) if not meta['no_overlap'] else None
  row['false_accept']=bool(meta['no_overlap'] and e.accepted);row['wrong_accept']=bool(not meta['no_overlap'] and e.accepted and not row['success5']);return clean(row),correct
 def run(self,models,plan,tag,save=False,modes=('equal','natural'),cap=128):
  rows=[]
  for number,s in enumerate(plan):
   folder=ROOT/'validation_data'/s['id'];ims=[cv2.imread(str(folder/f'image{i}.png'),0) for i in [0,1]];meta=json.loads(output('metadata',folder=folder).read_text());features={};times={}
   for method,model in models.items():
    torch.cuda.synchronize();start=time.perf_counter();features[method]=[self.features(method,model,im) for im in ims];torch.cuda.synchronize();times[method]=1000*(time.perf_counter()-start)
   budgets=[min(cap,*[len(f[i]['keypoints'][0]) for f in features.values()]) for i in [0,1]]
   for mode in modes:
    for method in models:
     fs=[truncate(f,budgets[i] if mode=='equal' else 1024) for i,f in enumerate(features[method])];torch.cuda.synchronize();start=time.perf_counter();p,e,d,scores=self.predict(method,fs,[im.shape for im in ims]);torch.cuda.synchronize();ms=times[method]+1000*(time.perf_counter()-start)
     gt=np.load(output('gt',ext='npz',folder=folder));row,correct=self.assess(method,p,e,d,fs,ims,gt,meta,ms);row.update(protocol=mode,common_budget=budgets,native_counts=[len(f['keypoints'][0]) for f in features[method]],GPU_peak_MiB=torch.cuda.max_memory_allocated()/2**20);rows.append(row)
     if save:
      target=ROOT/'predictions'/tag/s['id'];target.mkdir(parents=True,exist_ok=True)
      with output(f'{method}_{mode}',ext='npz',folder=target).open('xb') as f:np.savez_compressed(f,keypoints0=p.keypoints0,keypoints1=p.keypoints1,points0=p.points0,points1=p.points1,scores=scores,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(scores),bool),correct=correct,candidate_H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
   if number%50==0:print(f'{now()} {tag} {number+1}/{len(plan)}',flush=True)
  return rows
def summary(rows):
 pos=[r for r in rows if not r['no_overlap']];neg=[r for r in rows if r['no_overlap']]
 def avg(k):a=[r[k] for r in pos if r.get(k) is not None];return float(np.mean(a)) if a else None
 cats={r['category'] for r in pos};cscores=[]
 for c in cats:
  a=[r for r in pos if r['category']==c];cscores.append(np.mean([np.mean([r['success5'] for r in a if r['source_component']==comp]) for comp in {r['source_component'] for r in a}]))
 return dict(n_positive=len(pos),n_negative=len(neg),category_component_S5=float(np.mean(cscores)),**{f'S{t}':avg(f'success{t}') for t in [3,5,10]},correct_matches=avg('correct_matches'),repeatability=avg('repeatability'),NN_correct=avg('NN_correct'),end_to_end_R1=avg('end_to_end_R1'),conditional_R1=avg('conditional_R1'),keypoints0=avg('n_keypoints0'),no_overlap_FPR=float(np.mean([r['false_accept'] for r in neg])) if neg else 0,wrong_accept=avg('wrong_accept'),elapsed_ms=avg('elapsed_ms'))
def probes(model):
 rng=np.random.default_rng(42);rows=[]
 for name,im in [('black',np.zeros((256,256),np.uint8)),('gray',np.full((256,256),128,np.uint8)),('noise',rng.integers(0,256,(256,256),dtype=np.uint8))]:
  f=model.extract(torch.tensor(im,device='cuda').float()[None,None]/255);rows.append(dict(probe=name,n_keypoints=len(f['keypoints'][0])))
 return rows
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--diagnostic',action='store_true');p.add_argument('--final',action='store_true');a=p.parse_args();gpu(a.gpu);plan=json.loads(output('validation_plan').read_text());models={m:load_model(m) for m in ['m0','m2','b']};models['orb']=None
 if a.diagnostic:
  subset=[]
  for cat in ['inshore','offshore','background','ssdd']:subset.extend([s for s in plan if s['category']==cat and s['regime']!='none'][:16])
  subset.extend([s for s in plan if s['regime']=='none'][:4]);models['orb_single']=None;plan=subset;tag='orb_diagnostic'
 elif a.final:
  for m in ['control','joint','frozen']:models[m]=load_model(m,ROOT/'runs'/m/'best.ckpt')
  tag='final'
 else:tag='baseline'
 ev=Evaluator();rr=ev.run(models,plan,tag,save=True);export_csv(output(tag,ext='csv'),rr);write_new(output(tag),dict(rows=rr,summaries={mode:{m:summary([r for r in rr if r['method']==m and r['protocol']==mode]) for m in models} for mode in ['equal','natural']}));print(json.dumps({mode:{m:summary([r for r in rr if r['method']==m and r['protocol']==mode]) for m in models} for mode in ['equal','natural']}))
if __name__=='__main__':main()
