"""Train-only frozen image embeddings and traceable compact supervision, no optimization."""
from mixed_io import *
from collections import Counter
OFF=np.stack(np.meshgrid(np.arange(-2,3),np.arange(-2,3)),axis=-1).reshape(-1,2).astype(np.float32)
def supervision(row):
 with np.load(row['gt_path'],allow_pickle=False) as g:
  p=[g['structure_points_'+L].astype(np.float32) for L in 'AB'];raw=len(p[0]);size=row['width']
  safe=[g['safe_static_'+L]&g['valid_'+L]&~g['ignore_'+L]&~g['ship_'+L] for L in 'AB']
  ok=dc.bilinear_at(safe[0],p[0])&dc.bilinear_at(safe[1],p[1]);p=[q[ok] for q in p]
  if len(p[0])>128:ids=np.linspace(0,len(p[0])-1,128,dtype=int);p=[q[ids] for q in p]
  patches=[];keep=[];cal=[]
  for j,L in enumerate('AB'):
   xy=p[j][:,None]+OFF[None];patches.append(dc.bilinear_at(safe[j],xy.reshape(-1,2)).reshape(-1,25))
   eligible=safe[j]&~g['supervision_ignore_'+L];h=size//8
   keep.append(eligible.reshape(h,8,h,8).all((1,3)))
   c=g['calibration_'+L]&eligible;cal.append(c.reshape(h,8,h,8).all((1,3)))
  return dict(points=np.stack(p),patch_legal=np.stack(patches),keep=np.stack(keep),calibration=np.stack(cal),H_AB=g['H_AB'],raw_count=np.array(raw),local_weight=np.array(row['local_weight'],np.float32))
if __name__=='__main__':
 state=setup();m=Model('b').cuda();rr=records('train');counts=Counter();start=time.time();folder=ROOT/'cache/train_supervision';folder.mkdir(exist_ok=True)
 for i,row in enumerate(rr):
  if not row['structural_anchor_count']:counts['empty_not_cached']+=1;continue
  x,rs,_=shared_cache(row,m);q=supervision(row);p=folder/f"{row['pair_id']}_{TAG}.npz"
  if not p.exists():np.savez_compressed(p,**q)
  counts['eligible_cached']+=1;counts['post_legal_empty']+=int(not q['points'].shape[1]);counts['anchors_used']+=q['points'].shape[1]
  if i%50==0:print(now(),i,len(rr),dict(counts),flush=True)
 report=dict(state=state,counts=dict(counts),seconds=time.time()-start,manifests={k:sha(v) for k,v in MANIFEST.items()},cache_features_dtype='float32',encoder_frozen=True,original_gt_not_modified=True)
 write(ROOT/f'train_cache_audit_{stamp()}.json',report);print(json.dumps(report),flush=True)
