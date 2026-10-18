"""Tiny frozen cross-detector diagnostic; changed feature distribution is explicit."""
from fusion_io import *
import argparse,torch
from feature_runtime import initialize,match
from lightglue.superpoint import sample_descriptors
from fusion_core import geometry
def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();plan=read(next(ROOT.glob('DETECTOR_SWAP_PLAN_*.json')));models,lg=initialize(a.gpu);tag=stamp();result=[]
 data={v:read(OLD/f'main_{v}_20261015_1525.json') for v in ['v1','v22']};man={v:{r['sample_id']:r for r in rows(d['manifest'])} for v,d in data.items()}
 for item in plan['pairs']:
  v,sid=item['version'],item['sample_id'];row=man[v][sid];ims=[cv2.imread(row.get(f'image{i}',row.get(f'view{i}')),0) for i in [0,1]]
  with np.load(cache(v,sid,'ORB','equal')) as f:ks=[f[f'keypoints{i}'] for i in [0,1]]
  predictions=[];descriptors={}
  with torch.no_grad():
   for method in ['M2','B']:
    fs=[]
    for i,im in enumerate(ims):
     dense=models[method].maps(torch.tensor(im.copy(),device='cuda').float()[None,None]/255)[1];k=torch.tensor(ks[i],device='cuda');fs.append(dict(keypoints=k[None],keypoint_scores=torch.ones((1,len(k)),device='cuda'),descriptors=sample_descriptors(k[None],dense,8).transpose(1,2),image_size=torch.tensor([[im.shape[1],im.shape[0]]],device='cuda')))
    pred,confidence=match(lg,fs);e,d=geometry(pred,[im.shape for im in ims]);predictions.append((method,'LG',pred,e))
    dd=[f['descriptors'][0].cpu().numpy() for f in fs];descriptors[method]=dd;i,j=frozen._mutual_ratio(*dd,cv2.BFMatcher(cv2.NORM_L2),.8);pred=frozen.Prediction('cross_NN',ks[0][i],ks[1][j],keypoints0=ks[0],keypoints1=ks[1]);e,d=geometry(pred,[im.shape for im in ims]);predictions.append((method,'NN',pred,e))
  # Ground truth is read only after all actual-point predictions are fixed.
  z=frozen.load(row,v)
  for method,matcher,pred,e in predictions:
   q=frozen.transform_points(z['H_AB'],pred.points0);correct=(np.linalg.norm(q-pred.points1,axis=1)<=3)&frozen.bilinear_at(z['static'][0],pred.points0)&frozen.bilinear_at(z['static'][1],pred.points1)&frozen.bilinear_at(z['static'][1],q);err,_,_=frozen.global_error(e.candidate_H,z['H_AB'],z['points0'],z['points1'])
   t=frozen.transform_points(z['H_AB'],ks[0]);visible=frozen.bilinear_at(z['static'][0],ks[0])&frozen.bilinear_at(z['static'][1],t);truth=(np.linalg.norm(t[:,None]-ks[1][None],axis=2)<=3)&visible[:,None]&frozen.bilinear_at(z['static'][1],ks[1])[None];rep=truth.any(1);sim=descriptors[method][0]@descriptors[method][1].T;best=sim.argmax(1) if len(ks[1]) else np.zeros(len(ks[0]),int);hit=truth[np.arange(len(ks[0])),best] if len(ks[1]) else np.zeros(len(ks[0]),bool)
   result.append(dict(**item,descriptor=method,matcher=matcher,kp0=len(ks[0]),kp1=len(ks[1]),matches=len(pred.points0),correct_static=int(correct.sum()),repeatable=int(rep.sum()),actual_detected_R1=float(hit[visible].mean()) if visible.any() else None,conditional_R1=float(hit[rep].mean()) if rep.any() else None,accepted=e.accepted,reason=e.reason,success5=bool(e.accepted and err is not None and err<=5),input='same_actual_ORB_points_not_GT_points',distribution_shift=True))
  old=next(x for x in data[v]['rows'] if x['sample_id']==sid and x['method']=='ORB' and x['protocol']=='equal');result.append(dict(**item,descriptor='ORB',matcher='Hamming_cached',kp0=len(ks[0]),kp1=len(ks[1]),matches=old['n_matches'],correct_static=old['correct_static_matches'],repeatable=old['repeatable'],actual_detected_R1=old['detected_R1'],conditional_R1=old['conditional_R1'],accepted=old['accepted'],reason=old['reason'],success5=old['success5'],input='same_actual_ORB_points',distribution_shift=False))
 csvwrite(ROOT/f'DETECTOR_SWAP_{tag}.csv',result);write(ROOT/f'DETECTOR_SWAP_{tag}.json',dict(plan=plan,rows=result,weights_frozen=True,not_strict_single_factor=True));print('DETECTOR_SWAP_COMPLETE',len(result),tag)
if __name__=='__main__':main()
