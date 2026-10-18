"""CPU-only frozen-prediction analysis, with independent GT read after matching."""
from analyze import *
from sarbench.matchers import _mutual_ratio

def run(tag='baseline',limit_pairs=None):
 last=None;g=None
 rr=read_scores(tag)
 if limit_pairs:
  keep=set(list(dict.fromkeys(r['pair_id'] for r in rr))[:limit_pairs]);rr=[r for r in rr if r['pair_id'] in keep]
 lookup={r['pair_id']:r for r in records('val')};extra=[]
 for i,r in enumerate(rr):
  row=lookup[r['pair_id']];folder=ROOT/'predictions'/tag
  with np.load(folder/f"{r['pair_id']}_{r['method']}_{r['protocol']}_{TAG}.npz") as z:pred={k:z[k] for k in ['keypoints_A','keypoints_B','matches_A','matches_B','inliers']}
  if last!=r['pair_id']:
   with np.load(row['gt_path']) as f:g={k:f[k] for k in f.files}
   last=r['pair_id']
  if True:
   a,b=pred['keypoints_A'],pred['keypoints_B'];H=g['H_AB'];Ha=g['H_BA'];sa=g['static_A']&~g['ignore_A']&g['valid_A'];sb=g['static_B']&~g['ignore_B']&g['valid_B'];q=dc.transform_points(H,a)
   v=dc.bilinear_at(sa,a)&dc.bilinear_at(sb,q);vb=dc.bilinear_at(sb,b)
   dist=np.linalg.norm(q[:,None]-b[None],axis=2);truth=(dist<=3)&v[:,None]&vb[None]
   # Reverse 3px condition is part of legal static correctness as in score.py.
   reverse_ok=np.linalg.norm(dc.transform_points(Ha,b)[None]-a[:,None],axis=2)<=3 if len(a) and len(b) else np.ones(truth.shape,bool)
   truth &= reverse_ok
  valid0=g['valid_A']&~g['ignore_A'];valid1=g['valid_B']&~g['ignore_B'];gv=dc.bilinear_at(valid0,a)&dc.bilinear_at(valid1,q);gvb=dc.bilinear_at(valid1,b);gtruth=(dist<=3)&gv[:,None]&gvb[None]
  gtruth &= reverse_ok
  # Top128 descriptors are sufficient to reproduce the Equal protocol. Never infer full1024 NN from truncated cache.
  if r['protocol']=='equal':
   with np.load(folder/f"{r['pair_id']}_{r['method']}_top128_{TAG}.npz") as f:
    da,db=f['descriptors0'][0,:len(a)],f['descriptors1'][0,:len(b)]
   rawda,rawdb=da,db
   if r['method']=='orb':da,db=np.unpackbits(da,axis=1).astype(np.float32),np.unpackbits(db,axis=1).astype(np.float32);sim=2*da@db.T-da.sum(1)[:,None]-db.sum(1)[None]
   else:sim=da@db.T
   recall=[];conditional=[];geometric_recall=[];repeat=truth.any(1)
   for K in [1,5]:
    rank=np.argsort(-sim,axis=1,kind='stable')[:,:K];hit=np.take_along_axis(truth,rank,axis=1).any(1) if len(b) else np.zeros(len(a),bool);recall.append(float(hit[v].mean()) if v.any() else None);conditional.append(float(hit[repeat].mean()) if repeat.any() else None);ghit=np.take_along_axis(gtruth,rank,axis=1).any(1) if len(b) else np.zeros(len(a),bool);geometric_recall.append(float(ghit[gv].mean()) if gv.any() else None)
   ni,nj=_mutual_ratio(rawda,rawdb,cv2.BFMatcher(cv2.NORM_HAMMING if r['method']=='orb' else cv2.NORM_L2),.8);nn_correct=int(truth[ni,nj].sum()) if len(ni) else 0;nn_matches=len(ni);nn_geometric=int(gtruth[ni,nj].sum()) if len(ni) else 0
  else:recall=[None,None];conditional=[None,None];geometric_recall=[None,None];nn_correct=None;nn_matches=None;nn_geometric=None
  ma,mb=pred['matches_A'],pred['matches_B'];mq=dc.transform_points(H,ma);mp=dc.transform_points(Ha,mb)
  c=dc.bilinear_at(sa,ma)&dc.bilinear_at(sb,mb)&dc.bilinear_at(sb,mq)&(np.linalg.norm(mq-mb,axis=1)<=3)&(np.linalg.norm(mp-ma,axis=1)<=3)
  allgeo=dc.bilinear_at(valid0,ma)&dc.bilinear_at(valid1,mb)&dc.bilinear_at(valid1,mq)&(np.linalg.norm(mq-mb,axis=1)<=3)&(np.linalg.norm(mp-ma,axis=1)<=3)
  if r['no_overlap']:c[:]=False
  assert int(c.sum())==r['correct_structure_matches'],(r['pair_id'],r['method'],r['protocol'])
  assert int(allgeo.sum())>=int(c.sum())
  ship_proxy=dc.bilinear_at(g['ship_A'],ma)|dc.bilinear_at(g['ship_B'],mb)
  sp=spatial(pred['matches_A'][c],(row['height'],row['width']))
  isp=spatial(pred['matches_A'][pred['inliers'].astype(bool)],(row['height'],row['width']))
  extra.append(dict(inlier_grid=isp['grid_coverage'],inlier_hull=isp['hull_fraction'],inlier_isotropy=isp['isotropy'],pair_id=r['pair_id'],method=r['method'],protocol=r['protocol'],repeatable_bidir=int(truth.any(1).sum()),detected_R1=recall[0],detected_R5=recall[1],conditional_R1=conditional[0],conditional_R5=conditional[1],NN_correct_static=nn_correct,NN_matches=nn_matches,NN_geometry_consistent_including_nonstatic=nn_geometric,geometry_R1_including_nonstatic=geometric_recall[0],geometry_R5_including_nonstatic=geometric_recall[1],matches_geometry_consistent_including_nonstatic=int(allgeo.sum()),matches_geometry_consistent_ship_proxy=int((allgeo&ship_proxy).sum()),correct_match_grid=sp['grid_coverage'],correct_match_hull=sp['hull_fraction'],**{k:r[k] for k in ['keypoints0','keypoints1','correct_structure_matches','state','reason','Success@5px']}))
  if i%1000==0:print(now(),tag,i,len(rr),flush=True)
 label=tag if limit_pairs is None else f'{tag}_smoke{limit_pairs}pairs'
 csvwrite(ROOT/f'detected_descriptor_diagnostics_{label}_{stamp()}.csv',extra)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--tag',default='baseline');p.add_argument('--limit-pairs',type=int);a=p.parse_args();run(a.tag,a.limit_pairs)
