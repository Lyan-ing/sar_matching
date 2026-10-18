"""Pure-image prediction API; synthetic GT is read exclusively by post-prediction evaluator."""
from ss_common import *
from network import load
from model import load_model
from lightglue import LightGlue
from sarbench.matchers import Prediction,_mutual_ratio
from sarbench.estimate import estimate,perspective
from sarbench.reliability import geometry_diagnostics,gate,spatial
import time
def geometry(p,s0,s1):
 e=estimate(p,s0,model='homography',seed=42);d=geometry_diagnostics(e,p,s0,s1,42) if e.candidate_H is not None else {};return gate(e,d,'full'),d
def pred_summary(p,e,d,ms):
 return clean(dict(n_keypoints0=p.n_keypoints0,n_keypoints1=p.n_keypoints1,n_matches=e.n_matches,n_inliers=e.n_inliers,accepted=e.accepted,reason=e.reason,has_candidate=e.candidate_H is not None,decision='accepted' if e.accepted else ('rejected' if e.candidate_H is not None else 'no_output'),elapsed_ms=ms,geometry=d))
class Evaluator:
 def __init__(self):
  self.models={m:load_model(m,FIX/'runs/m2/best.ckpt' if m=='m2' else None) for m in ['m0','m2']}
  self.matcher=LightGlue(features='superpoint').eval().cuda().requires_grad_(False)
 @torch.no_grad()
 def predict(self,model,a,b,tiled=False):
  # ONLY native pixel arrays. All methods get identical inputs, NMS/threshold/topK/geometry.
  torch.cuda.synchronize();t=time.perf_counter();fs=[]
  for im in [a,b]:
   image=torch.from_numpy(im.copy()).float()[None,None]/255
   fs.append(tiled_extract(model.sp,image) if tiled else model.extract(image.cuda()))
  out=self.matcher(dict(image0=fs[0],image1=fs[1]));ids=out['matches'][0].cpu().numpy();ks=[f['keypoints'][0].cpu().numpy() for f in fs]
  p=Prediction('SP_LG',ks[0][ids[:,0]],ks[1][ids[:,1]],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1]);e,d=geometry(p,a.shape,b.shape)
  torch.cuda.synchronize();ms=1000*(time.perf_counter()-t)
  return p,e,d,out['scores'][0].cpu().numpy(),ms,fs
 def evaluate(self,model,method,tag,save=True):
  allrows=[];plan=self.plan
  for idx,spec in enumerate(plan):
   folder=ROOT/'validation_data'/spec['id'];meta=json.loads((folder/'metadata.json').read_text());out=ROOT/'predictions'/tag/spec['id'];js=out/(method+'_'+RUNSTAMP+'.json')
   if save and js.exists():allrows.append(json.loads(js.read_text()));continue
   ims=[cv2.imread(str(folder/f'image{i}.png'),0) for i in [0,1]]
   p,e,d,scores,ms,fs=self.predict(model,*ims)
   # GT loaded strictly after prediction; no evaluator fields enter predict/model.
   gt=np.load(folder/'gt.npz');H=gt['H_AB'];static=[gt['static0'],gt['static1']];k0,k1=p.keypoints0,p.keypoints1
   target=warp(H,k0);visible=valid_at(static[0],k0)&valid_at(static[1],target)
   vis1=valid_at(static[1],k1)&valid_at(static[0],warp(np.linalg.inv(H),k1))
   dist=np.linalg.norm(target[:,None]-k1[None,:],axis=2);truth=(dist<=3)&visible[:,None]&valid_at(static[1],k1)[None]
   repeat=truth.any(1);rev=(np.linalg.norm(warp(np.linalg.inv(H),k1)[:,None]-k0[None],axis=2)<=3)&vis1[:,None]&valid_at(static[0],k0)[None]
   ds=[f['descriptors'][0].cpu().numpy() for f in fs];sim=ds[0]@ds[1].T;order=np.argsort(-sim,axis=1,kind='stable')
   rank={}
   for K in [1,5,10]:
    hit=np.take_along_axis(truth,order[:,:K],axis=1).any(1) if len(k1) else np.zeros(len(k0),bool)
    rank[f'descriptor_conditional_recall{K}']=float(hit[repeat].mean()) if repeat.any() else None
    rank[f'descriptor_end_to_end_recall{K}']=float(hit[visible].mean()) if visible.any() else None
   i,j=_mutual_ratio(ds[0],ds[1],cv2.BFMatcher(cv2.NORM_L2),.8);nncorrect=truth[i,j] if len(i) else np.zeros(0,bool)
   predtarget=warp(H,p.points0);correct=(np.linalg.norm(predtarget-p.points1,axis=1)<=3)&valid_at(static[0],p.points0)&valid_at(static[1],p.points1)&valid_at(static[1],predtarget)
   support=spatial(p.points0[correct],ims[0].shape);rs=spatial(k0[repeat],ims[0].shape)
   yy,xx=np.mgrid[12:244:12,12:244:12];q=np.c_[xx.ravel(),yy.ravel()];v=warp(H,q);keep=valid_at(static[0],q)&valid_at(static[1],v);ep=None
   if e.candidate_H is not None and keep.any():
    er=np.linalg.norm(perspective(e.candidate_H,q[keep])-v[keep],axis=1);ep=float(er.mean()) if np.isfinite(er).all() else None
   row=dict(**meta,method=method,**pred_summary(p,e,d,ms),visible_detected0=int(visible.sum()),visible_detected1=int(vis1.sum()),repeatable_detected0=int(repeat.sum()),repeatable_detected1=int(rev.any(1).sum()),detector_repeatability_3px=float(repeat.sum()/visible.sum()) if visible.any() else None,repeatable_hull=rs['hull_fraction'],repeatable_grid=rs['grid_coverage'],**rank,positive_cosine=float(sim[truth].mean()) if truth.any() else None,negative_cosine=float(sim[(dist>=8)&visible[:,None]&valid_at(static[1],k1)[None]].mean()) if ((dist>=8)&visible[:,None]&valid_at(static[1],k1)[None]).any() else None,NN_matches=len(i),NN_correct_3px=int(nncorrect.sum()),LG_correct_3px=int(correct.sum()),LG_correct_inliers_3px=int((correct&e.inlier_mask).sum()) if e.inlier_mask is not None else 0,LG_end_to_end_recall=float(correct.sum()/visible.sum()) if visible.any() else None,LG_correct_hull=support['hull_fraction'],LG_correct_grid=support['grid_coverage'],has_enough_correct_support=bool(correct.sum()>=8 and support['hull_fraction']>=.01),candidate_epe_mean=ep,n_geometry_checks=int(keep.sum()))
   for T in [3,5,10]:row[f'success_{T}px']=bool(e.accepted and ep is not None and ep<=T) if not meta['no_overlap'] else None
   row['false_accept']=bool(meta['no_overlap'] and e.accepted);row['incorrect_accept_positive']=bool(not meta['no_overlap'] and e.accepted and not row['success_5px']);row['decision_status']='success' if row['success_5px'] else ('incorrect_accept' if e.accepted else row['decision'])
   row=clean(row);allrows.append(row)
   if save:
    out.mkdir(parents=True,exist_ok=True)
    with (out/(method+'_'+RUNSTAMP+'.npz')).open('xb') as f:np.savez_compressed(f,keypoints0=k0,keypoints1=k1,points0=p.points0,points1=p.points1,scores=scores,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(correct),bool),correct=correct,repeatable=repeat,descriptors0=ds[0],descriptors1=ds[1],candidate_H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
    atomic_json(js,row)
   if idx%50==0:print(f'{now()} validation {tag} {method} {idx+1}/{len(plan)}',flush=True)
  return allrows
def summarize(rr):
 positive=[r for r in rr if not r['no_overlap']];negative=[r for r in rr if r['no_overlap']];scenes={r['scene'] if r['dataset']=='hrsid' else r['dataset'] for r in positive};cat={}
 for scene in sorted(scenes):
  p=[r for r in positive if (r['scene'] if r['dataset']=='hrsid' else r['dataset'])==scene];cs=sorted({r['source_component'] for r in p})
  cat[scene]=float(np.mean([np.mean([r['success_5px'] for r in p if r['source_component']==c]) for c in cs]))
 def mean(k):
  x=[r[k] for r in positive if r.get(k) is not None];return float(np.mean(x)) if x else None
 return dict(n_positive=len(positive),n_negative=len(negative),category_component_S5=float(np.mean(list(cat.values()))),category_component_S5_by_category=cat,**{f'S{t}':mean(f'success_{t}px') for t in [3,5,10]},no_overlap_false_accept=float(np.mean([r['false_accept'] for r in negative])) if negative else 0,positive_incorrect_accept=mean('incorrect_accept_positive'),reject_rate=float(np.mean([not r['accepted'] for r in rr])),repeatability=mean('detector_repeatability_3px'),repeatable_count=mean('repeatable_detected0'),keypoints0=mean('n_keypoints0'),correct_LG=mean('LG_correct_3px'),correct_support_pairs=sum(r['has_enough_correct_support'] for r in positive),end_to_end_R1=mean('descriptor_end_to_end_recall1'),conditional_R1=mean('descriptor_conditional_recall1'),elapsed_ms=mean('elapsed_ms'))
def selection(s):return (s['category_component_S5'],-s['no_overlap_false_accept'],-s['positive_incorrect_accept'],s['correct_support_pairs'],s['repeatability'] or 0)

class FairEvaluator(Evaluator):
 def __init__(self,plan):
  self.plan=plan;self.matcher=LightGlue(features='superpoint').eval().cuda().requires_grad_(False)
  rr=json.loads(output('baseline_records').read_text());self.budgets={tuple(r['input_sha256']):(r['n_keypoints0'],r['n_keypoints1']) for r in rr if r['method']=='m2'}
  self.common_budgets=None;self.equal=True
 @torch.no_grad()
 def predict(self,model,a,b,tiled=False):
  torch.cuda.synchronize();start=time.perf_counter();fs=[]
  hashes=tuple(hashlib.sha256(im.tobytes()).hexdigest() for im in [a,b]);budget=(self.common_budgets or self.budgets).get(hashes)
  for i,im in enumerate([a,b]):
   image=torch.tensor(im.copy(),device='cpu').float()[None,None]/255
   f=model.extract(image if max(im.shape)>2048 else image.cuda())
   if self.equal and budget:
    ids=torch.argsort(f['keypoint_scores'][0],descending=True,stable=True)[:budget[i]]
    f={k:(v[:,ids] if k in ['keypoints','descriptors','keypoint_scores'] else v) for k,v in f.items()}
   fs.append(f)
  out=self.matcher(dict(image0=fs[0],image1=fs[1]));ids=out['matches'][0].cpu().numpy();ks=[f['keypoints'][0].cpu().numpy() for f in fs]
  p=Prediction('SP_LG',ks[0][ids[:,0]],ks[1][ids[:,1]],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1]);e,d=geometry(p,a.shape,b.shape);torch.cuda.synchronize()
  return p,e,d,out['scores'][0].cpu().numpy(),1000*(time.perf_counter()-start),fs

def probes(m):
 rng=np.random.default_rng(42);ims={'black':np.zeros((256,256),np.uint8),'gray':np.full((256,256),128,np.uint8),'noise':rng.integers(0,256,(256,256),dtype=np.uint8)};rr=[]
 with torch.no_grad():
  for name,im in ims.items():
   inp=torch.tensor(im.copy(),device='cuda').float()[None,None]/255;f=m.extract(inp);s=score_map(m.maps(inp)[0]);rr.append(dict(probe=name,n_keypoints=len(f['keypoints'][0]),max_response=float(s.max()),mean_response=float(s.mean())))
 return rr
from m3_model import score_map

def quality_key(summary,sanity,baseline):
 hard=all(r['n_keypoints']==0 for r in sanity if r['probe'] in ['black','gray']) and summary['keypoints0']<=2*baseline['keypoints0'] and summary['no_overlap_false_accept']<=.05
 return (int(hard),summary['category_component_S5'],summary['correct_LG'],summary['repeatability'] or 0)
