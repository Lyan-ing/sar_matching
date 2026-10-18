"""Image-only predictions first, disjoint version/role-specific GT assessment second."""
from support22 import *
from sarbench.matchers import Prediction,_mutual_ratio
from sarbench.estimate import estimate,perspective
from sarbench.reliability import geometry_diagnostics,gate,spatial
from load_prepared_pair import load_training_pair as load_v1
from dataset_v2 import load_training_pair as load_v21
def features(model,im):
 if model is None:
  orb=cv2.ORB_create(nfeatures=1024,fastThreshold=12);k,d=orb.detectAndCompute(im,None);d=np.empty((0,32),np.uint8) if d is None else d
  return dict(keypoints=torch.tensor(np.array([p.pt for p in k],np.float32).reshape(-1,2),device='cuda')[None],keypoint_scores=torch.tensor([p.response for p in k],device='cuda')[None],descriptors=torch.tensor(d,device='cuda')[None],image_size=torch.tensor([[im.shape[1],im.shape[0]]],device='cuda'))
 return model.extract(torch.as_tensor(im.copy(),device='cuda',dtype=torch.float32)[None,None]/255)
def trim(f,n):
 ids=torch.argsort(f['keypoint_scores'][0],descending=True,stable=True)[:n]
 return {k:(v[:,ids] if k in ['keypoints','keypoint_scores','descriptors'] else v) for k,v in f.items()}
@torch.no_grad()
def predict(lg,fs,shapes,orb=False):
 if orb:
  ds=[f['descriptors'][0].cpu().numpy() for f in fs];i,j=_mutual_ratio(*ds,cv2.BFMatcher(cv2.NORM_HAMMING),.8);ij=np.c_[i,j];confidence=np.ones(len(i))
 else:
  r=lg(dict(image0=fs[0],image1=fs[1]));ij=r['matches'][0].cpu().numpy();confidence=r['scores'][0].cpu().numpy()
 ks=[f['keypoints'][0].cpu().numpy() for f in fs];p=Prediction('ORB' if orb else 'SP+LG',ks[0][ij[:,0]],ks[1][ij[:,1]],n_keypoints0=len(ks[0]),n_keypoints1=len(ks[1]),keypoints0=ks[0],keypoints1=ks[1])
 est=estimate(p,shapes[0],model='homography',seed=42);diag=geometry_diagnostics(est,p,*shapes,42) if est.candidate_H is not None else {};est=gate(est,diag,'full')
 return p,est,diag,confidence
def load(row,version):
 if version=='v1':return load_v1(row)
 if version=='v21':return load_v21(row,max_points=None)
 return load_pair(row,use_ship_aux=True)
def global_error(H,gt,p,q):
 if H is None or not len(p):return None,None,None
 try:
  a=np.linalg.norm(transform_points(H,p)-q,axis=1);b=np.linalg.norm(transform_points(np.linalg.inv(H),q)-p,axis=1);v=np.r_[a,b]
  return (float(v.mean()),float(np.median(v)),float(np.quantile(v,.9))) if np.isfinite(v).all() else (None,None,None)
 except np.linalg.LinAlgError:return None,None,None
def assess(fs,p,e,d,z,row,version):
 k0,k1=p.keypoints0,p.keypoints1;H=z['H_AB'];ms=z['static'];target=transform_points(H,k0);back=transform_points(np.linalg.inv(H),k1)
 v0=bilinear_at(ms[0],k0)&bilinear_at(ms[1],target);v1=bilinear_at(ms[1],k1)&bilinear_at(ms[0],back)
 dd=np.linalg.norm(target[:,None]-k1[None],axis=2);truth=(dd<=3)&v0[:,None]&v1[None];rep=truth.any(1)
 orb=fs[0]['descriptors'].dtype==torch.uint8
 ds=[f['descriptors'][0].cpu().numpy() for f in fs]
 if orb:
  a,b=[np.unpackbits(x,axis=1).astype(np.float32) for x in ds];sim=1-(a.sum(1)[:,None]+b.sum(1)[None]-2*a@b.T)/256;norm=cv2.NORM_HAMMING
 else:sim=ds[0]@ds[1].T;norm=cv2.NORM_L2
 order=np.argsort(-sim,axis=1,kind='stable');rank={}
 for K in [1,5]:
  hit=np.take_along_axis(truth,order[:,:K],axis=1).any(1) if len(k1) else np.zeros(len(k0),bool)
  rank[f'detected_R{K}']=float(hit[v0].mean()) if v0.any() else None
  rank[f'conditional_R{K}']=float(hit[rep].mean()) if rep.any() else None
 i,j=_mutual_ratio(*ds,cv2.BFMatcher(norm),.8);nn_correct=int(truth[i,j].sum()) if len(i) else 0
 gt1=transform_points(H,p.points0);correct=(np.linalg.norm(gt1-p.points1,axis=1)<=3)&bilinear_at(ms[0],p.points0)&bilinear_at(ms[1],p.points1)&bilinear_at(ms[1],gt1)
 support=spatial(p.points0[correct],z['images'][0].shape)
 err,median,p90=global_error(e.candidate_H,H,z['points0'],z['points1']);negative=row['profile']=='none'
 eligible=not negative if version=='v1' else bool(row.get('global_eligible',row['loss_role']=='structural_positive'))
 role='no_overlap' if negative else row.get('loss_role','legacy_geometric_positive')
 common=set(json.loads(out('protocol').read_text())['v21_common_ids']) if version=='v21' else set()
 r=dict(sample_id=row['sample_id'],version=version,source_path=row['source_path'],component=row['known_component'],scene=row['source_scene_category'],angle_bin=row.get('angle_bin_2D','not_binned'),angle_2D=row.get('image_rotation_2D_deg'),size=row['width'],profile=row['profile'],overlap=min(row['overlap_fractions']),role=role,global_eligible=eligible,no_overlap=negative,common133=row['sample_id'] in common,
  n_keypoints0=len(k0),n_keypoints1=len(k1),n_matches=e.n_matches,n_inliers=e.n_inliers,accepted=e.accepted,reason=e.reason,state='accepted' if e.accepted else 'rejected' if e.candidate_H is not None else 'no_output',
  ground_points=len(z['points0']),repeatable=int(rep.sum()),visible_detected0=int(v0.sum()),repeatability=float(rep.sum()/v0.sum()) if v0.any() else None,correct_static_matches=int(correct.sum()),precision=float(correct.mean()) if len(correct) else None,NN_correct=nn_correct,correct_grid=support['grid_coverage'],correct_hull=support['hull_fraction'],local_bidir_error=err,median_error=median,p90_error=p90,**rank)
 for t in [3,5,10]:r[f'success{t}']=bool(e.accepted and err is not None and err<=t) if eligible else None
 r['wrong_accept']=bool(e.accepted and not r['success5']) if eligible else None;r['no_overlap_false_accept']=bool(e.accepted) if negative else None
 if version=='v22':
  for i in [0,1]:r[f'calibration_keypoints{i}']=int(bilinear_at(z['calibration_mask'][i],[k0,k1][i]).sum());r[f'unknown_keypoints{i}']=int(bilinear_at(z['supervision_ignore'][i],[k0,k1][i]).sum())
  # This is explicitly synthetic auxiliary, never part of static ground match counts.
  sk0,sk1=z['ship_points0'],z['ship_points1'];r['ship_anchor_count']=len(sk0)
  shipok=(np.linalg.norm(gt1-p.points1,axis=1)<=3)&bilinear_at(z['ship_aux_masks'][0],p.points0)&bilinear_at(z['ship_aux_masks'][1],p.points1)
  r['synthetic_ship_correct_matches']=int(shipok.sum())
 return clean(r),correct
@torch.no_grad()
def oracle(model,z):
 if model is None or (len(z['points0'])<4 and len(z.get('ship_points0',[]))<4):return dict(oracle_R1=None,oracle_R5=None,ship_oracle_R1=None)
 ims=torch.as_tensor(np.stack(z['images']).copy(),device='cuda',dtype=torch.float32)[:,None]/255;dense=model.maps(ims)[1]
 def calc(p,q):
  if len(p)<4:return None,None
  xy=[torch.as_tensor(a,device='cuda')[None] for a in [p,q]];d=[sample_descriptors(a,dense[i:i+1],8)[0].T for i,a in enumerate(xy)];sim=d[0]@d[1].T
  near=(torch.cdist(xy[0][0],xy[0][0])<8)|(torch.cdist(xy[1][0],xy[1][0])<8);near.fill_diagonal_(False);sim=sim.masked_fill(near,-1e4);idx=sim.argsort(1,descending=True);lab=torch.arange(len(p),device='cuda')[:,None]
  return float((idx[:,:1]==lab).any(1).float().mean()),float((idx[:,:min(5,len(q))]==lab).any(1).float().mean())
 a,b=calc(z['points0'],z['points1']);s,_=calc(z.get('ship_points0',[]),z.get('ship_points1',[]))
 return dict(oracle_R1=a,oracle_R5=b,ship_oracle_R1=s)
def summarize(rr):
 pos=[r for r in rr if r['global_eligible']];neg=[r for r in rr if r['no_overlap']];local=[r for r in rr if r['role']=='limited_structure'];desc=[r for r in rr if r['ground_points']>=4]
 def avg(a,k):v=[r[k] for r in a if r.get(k) is not None];return float(np.mean(v)) if v else None
 return dict(n=len(rr),global_n=len(pos),limited_n=len(local),limited_descriptor_n=sum(r['ground_points']>=4 for r in local),no_overlap_n=len(neg),**{f'S{t}':avg(pos,f'success{t}') for t in [3,5,10]},wrong_accept=avg(pos,'wrong_accept'),no_overlap_FPR=avg(neg,'no_overlap_false_accept'),rejection_rate=avg([dict(r,reject=not r['accepted']) for r in rr],'reject'),oracle_R1=avg(desc,'oracle_R1'),local_R1=avg(local,'oracle_R1'),detected_R1=avg(pos,'detected_R1'),repeatability=avg(pos,'repeatability'),correct_static_matches=avg(pos,'correct_static_matches'),NN_correct=avg(pos,'NN_correct'),keypoints0=avg(rr,'n_keypoints0'),calibration_keypoints0=avg([r for r in rr if r['role']=='no_structural_positive'],'calibration_keypoints0'),ship_correct=avg([r for r in rr if r.get('ship_anchor_count',0)>=4],'synthetic_ship_correct_matches'),elapsed_ms=avg(rr,'elapsed_ms'))
@torch.no_grad()
def probes(m):
 rg=np.random.default_rng(42);rr=[]
 for name,im in [('black',np.zeros((256,256),np.uint8)),('gray',np.full((256,256),128,np.uint8)),('noise',rg.integers(0,256,(256,256),dtype=np.uint8))]:
  fs=features(m,im);rr.append(dict(probe=name,n_keypoints=len(fs['keypoints'][0])))
 return rr
class Evaluation:
 def __init__(self):self.lg=LightGlue(features='superpoint').eval().cuda().requires_grad_(False)
 @torch.no_grad()
 def run(self,models,records,version,tag,save_predictions=False,equal=True):
  result=[]
  for n,row in enumerate(records):
   z=load(row,version);fs={};times={};om={}
   for method,m in models.items():
    torch.cuda.synchronize();t=time.perf_counter();fs[method]=[features(m,im) for im in z['images']];torch.cuda.synchronize();times[method]=(time.perf_counter()-t)*1000;om[method]=oracle(m,z)
   budgets=[min(128,*[len(f[i]['keypoints'][0]) for f in fs.values()]) for i in [0,1]]
   for method,m in models.items():
    for mode in (['natural','equal'] if equal else ['natural']):
     selected=[trim(f,budgets[i] if mode=='equal' else 1024) for i,f in enumerate(fs[method])]
     torch.cuda.synchronize();t=time.perf_counter();p,e,d,confidence=predict(self.lg,selected,[im.shape for im in z['images']],m is None);torch.cuda.synchronize();ms=times[method]+1000*(time.perf_counter()-t)
     r,c=assess(selected,p,e,d,z,row,version);r.update(method=method,protocol=mode,common_budget=budgets,elapsed_ms=ms,**om[method]);result.append(r)
     if save_predictions:
      folder=ROOT/'predictions'/tag/version/row['sample_id'];folder.mkdir(parents=True,exist_ok=True);path=out(f'{method}_{mode}',ext='npz',folder=folder)
      with path.open('xb') as f:np.savez_compressed(f,keypoints0=p.keypoints0,keypoints1=p.keypoints1,points0=p.points0,points1=p.points1,confidence=confidence,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(e.n_matches,bool),correct_static=c,H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)))
   if n%100==0:print(now(),tag,version,n+1,len(records),flush=True)
  return result
