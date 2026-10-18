"""Locked-model post-hoc detector sanity (blank/noise) and unique GT-repeat counts, never selection."""
from m3_evaluate import *
from finish_m3 import export_csv
import argparse
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();gpu(a.gpu);lock=json.loads((ROOT/'runs/m3/selection_locked.json').read_text());assert sha(ROOT/'runs/m3/best.ckpt')==lock['best_sha256'];ev=Evaluator();ev.models['m3']=load_m3(ROOT/'runs/m3/best.ckpt')
 sanity=[];rng=np.random.default_rng(77123);ims={'black':np.zeros((256,256),np.uint8),'constant_gray':np.full((256,256),128,np.uint8),'independent_noise':rng.integers(0,256,(256,256),dtype=np.uint8)}
 for name,im in ims.items():
  for method,model in ev.models.items():
   with torch.no_grad():f=model.extract(torch.tensor(im.copy(),device='cuda').float()[None,None]/255)
   k=f['keypoints'][0].cpu().numpy();s=f['keypoint_scores'][0].cpu().numpy();sanity.append(dict(probe=name,method=method,n_keypoints=len(k),mean_score=float(s.mean()) if len(s) else None,max_score=float(s.max()) if len(s) else None,notes='blank/constant has no stable physical landmarks; synthetic image-only sanity, not validation accuracy'))
 atomic_json(ROOT/'detector_sanity.json',sanity);export_csv(ROOT/'tables/detector_sanity.csv',sanity)
 unique=[]
 for r in json.loads((ROOT/'validation_results.json').read_text()):
  folder=ROOT/'predictions'/('selected' if r['method']=='m3' else 'baseline')/r['id'];z=np.load(folder/(r['method']+'.npz'));gt=np.load(ROOT/'validation_data'/r['id']/'gt.npz');k0,k1=z['keypoints0'],z['keypoints1'];target=warp(gt['H_AB'],k0)
  d=np.linalg.norm(target[:,None]-k1[None],axis=2);count=0;support=dict(hull_fraction=0,grid_coverage=0)
  if len(k0) and len(k1):
   j=d.argmin(1);i=d.argmin(0);ids=np.arange(len(k0));keep=(i[j]==ids)&(d[ids,j]<=3)&valid_at(gt['static0'],k0)&valid_at(gt['static1'],target)&valid_at(gt['static1'],k1[j]);count=int(keep.sum());support=spatial(k0[keep],(256,256))
  unique.append(dict(id=r['id'],method=r['method'],no_overlap=r['no_overlap'],unique_mutual_GT_repeat_count=count,unique_repeat_hull=support['hull_fraction'],unique_repeat_grid=support['grid_coverage'],repeat_count_nonunique=r['repeatable_detected0']))
 export_csv(ROOT/'tables/unique_repeatability.csv',unique);atomic_json(ROOT/'post_checks.json',dict(post_selection_only=True,detector_sanity=sanity,unique_repeat_means={m:float(np.mean([r['unique_mutual_GT_repeat_count'] for r in unique if r['method']==m and not r['no_overlap']])) for m in ['m0','m2','m3']}))
 print(json.dumps(sanity,indent=2))
if __name__=='__main__':main()
