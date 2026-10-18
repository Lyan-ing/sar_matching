"""Predict image arrays first; frozen version-specific labels only enter posthoc score."""
from fusion_io import *
import argparse,sys,torch
from feature_runtime import initialize,infer_images
from fusion_core import near
def score(p,e,meta,z,baseline,version):
 H=z['H_AB'];target=frozen.transform_points(H,p.points0);correct=(np.linalg.norm(target-p.points1,axis=1)<=3)&frozen.bilinear_at(z['static'][0],p.points0)&frozen.bilinear_at(z['static'][1],p.points1)&frozen.bilinear_at(z['static'][1],target)
 err,med,p90=frozen.global_error(e.candidate_H,H,z['points0'],z['points1']);src=meta['sources'];support=frozen.spatial(p.points0[correct],z['images'][0].shape)
 r={k:baseline[k] for k in ['sample_id','version','source_path','component','scene','angle_bin','size','profile','overlap','role','global_eligible','no_overlap']}
 r.update(n_keypoints0=len(p.keypoints0),n_keypoints1=len(p.keypoints1),n_matches=len(p.points0),n_inliers=e.n_inliers,accepted=e.accepted,reason=e.reason,state='accepted' if e.accepted else 'rejected' if e.candidate_H is not None else 'no_output',correct_static_matches=int(correct.sum()),precision=float(correct.mean()) if len(correct) else None,correct_grid=support['grid_coverage'],correct_hull=support['hull_fraction'],local_bidir_error=err,median_error=med,p90_error=p90,orb_retained_correct=int((correct&(src==1)).sum()),sp_retained_correct=int((correct&(src==0)).sum()))
 for t in [3,5,10]:r[f'success{t}']=bool(e.accepted and err is not None and err<=t) if r['global_eligible'] else None
 r['wrong_accept']=bool(e.accepted and not r['success5']) if r['global_eligible'] else None;r['no_overlap_false_accept']=bool(e.accepted) if r['no_overlap'] else None
 return r,correct
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,required=True);parser.add_argument('--limit',type=int);parser.add_argument('--version',choices=['v1','v21','v22']);parser.add_argument('--smoke',action='store_true');a=parser.parse_args();decision=read(next(ROOT.glob('STAGE2_DECISION_*.json')));assert decision['permit_bounded_fusion']
 models,lg=initialize(a.gpu);tag=stamp();versions=[a.version] if a.version else ['v1','v21','v22'];opening={'active':False};reads=[]
 def audit(event,args):
  if event=='open' and opening['active'] and isinstance(args[0],(str,bytes)):
   p=str(Path(args[0]).resolve())
   if any(s in p for s in ['SyntheticSAR_','DualSAR_DatasetPrep_','RealSAR_','MaritimeSARRegDataset/data']):raise RuntimeError('Forbidden data/label read in image-only prediction: '+p)
   reads.append(p)
 sys.addaudithook(audit)
 for v in versions:
  data=read(OLD/f'main_{v}_20261015_1525.json');allrows=rows(data['manifest']);idx={(x['sample_id'],x['method'],x['protocol']):x for x in data['rows']};selected=allrows[:a.limit] if a.limit else allrows
  group=('smoke_' if a.smoke else '')+'fusion_'+v;folder=ROOT/'predictions'/group;folder.mkdir(parents=True,exist_ok=True);metrics=[];completed=folder/f'COMPLETED_{tag}.json';log=ROOT/f'{group}_rows_{tag}.jsonl'
  with log.open('x') as f:
   for n,row in enumerate(selected):
    sid=row['sample_id'];done=list((folder/sid).glob('record_*.json')) if (folder/sid).exists() else []
    if done:metrics.extend(read(done[0])['rows']);continue
    images=[cv2.imread(row.get(f'image{i}',row.get(f'view{i}')),0) for i in [0,1]]
    opening['active']=True
    try:outputs,features=infer_images(images,models,lg)
    finally:opening['active']=False
    # Every recomputed detection must equal frozen positions; full-match inference is not repeated.
    for m in ['M2','B','M2_V22','ORB']:
     with np.load(cache(v,sid,m,'natural')) as old:
      for i in [0,1]:assert np.array_equal(features[m][i]['keypoints'][0].cpu().numpy(),old[f'keypoints{i}']),(v,sid,m,'detection mismatch')
    for x in outputs:assert x['budgets']==[idx[(sid,'M2','equal')][f'n_keypoints{i}'] for i in [0,1]]
    del features
    # All predictions are now frozen; only this evaluator reads old masks/analytic H.
    z=frozen.load(row,v);pairrows=[];dest=folder/sid;dest.mkdir(exist_ok=True)
    for x in outputs:
     old=idx[(sid,x['method'],'equal')];op=x['orb'];ot=frozen.transform_points(z['H_AB'],op.points0);oc=(np.linalg.norm(ot-op.points1,axis=1)<=3)&frozen.bilinear_at(z['static'][0],op.points0)&frozen.bilinear_at(z['static'][1],op.points1);unique=~near(op,x['sp'])
     for method,(p,e,d,meta) in x['outputs'].items():
      r,c=score(p,e,meta,z,old,v);src=meta['sources'];ii=meta['pool_indices'];r.update(sp_method=x['method'],method=method,protocol='equal_total128',split=x['split'],sp_quota0=x['sp_n'][0],sp_quota1=x['sp_n'][1],orb_quota0=x['orb_n'][0],orb_quota1=x['orb_n'][1],sp_pool_matches=len(x['sp'].points0),orb_pool_matches=len(op.points0),orb_correct_pool=int(oc.sum()),orb_unique_correct_pool=int((oc&unique).sum()),orb_retained_unique_correct=sum(bool(c[j] and unique[ii[j]]) for j in range(len(c)) if src[j]==1),route=meta['route'],feature_all3SP_ORB_ms=x['feature_ms'],sp_match_ms=x['sp_match_ms'],orb_match_ms=x['orb_ms'],geometry_ms=meta['geometry_ms'],elapsed_ms=x['base_ms']+meta['geometry_ms'],torch_peak_MiB=torch.cuda.max_memory_allocated()/2**20,previous_sp_success5=old['success5'],previous_orb_success5=idx[(sid,'ORB','equal')]['success5'],computed='new_frozen_split_matching',physical_GT=False)
      path=dest/f'{x["method"]}_{x["split"]}_{method}_{tag}.npz'
      with path.open('xb') as pf:np.savez_compressed(pf,keypoints0=p.keypoints0,keypoints1=p.keypoints1,points0=p.points0,points1=p.points1,sources=src,pool_indices=ii,correct_static=c,inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(c),bool),H=e.candidate_H if e.candidate_H is not None else np.empty((0,3)),sp_points0=x['sp'].points0,sp_points1=x['sp'].points1,orb_points0=op.points0,orb_points1=op.points1,sp_confidence=x['sp_confidence'],orb_confidence=x['orb_confidence'])
      r['prediction_file']=str(path);r['diagnostics']=d;r['fusion_meta']={k:v for k,v in meta.items() if k not in ['sources','pool_indices']};pairrows.append(frozen.clean(r))
    write(dest/f'record_{tag}.json',dict(rows=pairrows,manifest_row_sha=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()));metrics.extend(pairrows)
    for r in pairrows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    f.flush()
    if n%25==0:print(stamp(),v,n+1,len(selected),flush=True)
   csvwrite(ROOT/f'{group}_PAIRWISE_{tag}.csv',metrics);write(completed,dict(at=stamp(),samples=len(selected),rows=len(metrics),gpu=a.gpu,no_label_reads_in_predict=True,sha_code={str(p):sha(p) for p in (ROOT/'code').glob('*.py')},read_files_during_prediction=sorted(set(reads)),manifest_sha=sha(data['manifest']),frozen_weights={str(p):sha(p) for p in __import__('feature_runtime').WEIGHTS.values()}))
  print('COMPLETED',v,len(metrics),flush=True)
if __name__=='__main__':main()
