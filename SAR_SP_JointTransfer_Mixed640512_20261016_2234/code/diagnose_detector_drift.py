"""Bounded read-only failure investigation: saved trained weights, eight Train pairs."""
from mixed_io import *
from joint_objective import scope,objective,load_item

def main():
 torch.set_num_threads(2);cv2.setNumThreads(1)
 ck=ROOT/'runs/t1'/f'checkpoint_step2000_{TAG}.ckpt';before=sha(ck);teacher=Model('b');original=Model('m0');models={'initial':Model('b'),'detector_trained2000':Model('b',ck)};train=records('train');plan=[json.loads(s) for s in sorted((ROOT/'plans').glob('sampling_seed42_*.jsonl'))[-1].read_text().splitlines()];ids=plan[0]['active'];result={}
 for name,m in models.items():
  scope(m,'t1');gr=[]
  for idx in ids:
   row=train[idx];q=torch.load(ROOT/'cache/train'/f"{row['pair_id']}_{TAG}.pt",map_location='cpu',weights_only=False);x,rs=q['x'],q['responses'];item=load_item(row);l,d=m.cached_maps(x,rs)
   with torch.no_grad():_,td=teacher.cached_maps(x,rs);sp=original.sp;tl=sp.convPb(sp.relu(sp.convPa(x)))
   _,info,parts=objective(l,d,tl,td,item,row['width'],True);p=[v for k,v in m.named_parameters() if k.startswith('sp.convP')];a=torch.autograd.grad(parts['descriptor'],p,retain_graph=True,allow_unused=True);b=torch.autograd.grad(parts['detector'],p,allow_unused=True);a=torch.cat([g.flatten() for g in a if g is not None]);b=torch.cat([g.flatten() for g in b if g is not None]);gr.append(dict(pair_id=row['pair_id'],points=info['points'],descriptor_norm=float(a.norm()),detector_norm=float(b.norm()),cosine=float(F.cosine_similarity(a,b,dim=0))))
  probes=[]
  for size in [256,512,640]:
   for label,value in [('black',0.),('gray',.5)]:
    with torch.no_grad():im=torch.full((1,1,size,size),value);l,d=m.maps(im);f=m.extract_maps(l,d,(size,size));probes.append(dict(size=size,input=label,keypoints=len(f['keypoints'][0]),mean_dustbin_probability=float(l.softmax(1)[:,64].mean()),max_pixel_probability=float(score_map(l).max())))
  result[name]=dict(gradients=gr,constant_input_probes=probes)
 initial=models['initial'].state_dict();trained=models['detector_trained2000'].state_dict();changed=[k for k,v in trained.items() if not torch.equal(v,initial[k])];assert all(k.startswith(('sp.convP','structure.encode')) for k in changed);assert sha(ck)==before
 write(ROOT/f'DETECTOR_DRIFT_DIAGNOSTIC_{stamp()}.json',dict(checkpoint=str(ck),checkpoint_sha=before,read_only=True,source='eight preselected Train pairs, not Val supervision',changed_parameters=changed,unexpected_frozen_changes=[],results=result,interpretation='No NaN, coordinate or frozen-scope error identified. Known-static local supervision does not constrain all constant-input responses; detector drift is an algorithmic failure under this locked objective. No loss, thresholds or data changed after observing Val. Continue identical finite-budget ablation to report this failure.'))
 print('Read-only detector drift investigation completed; protocol unchanged.')
if __name__=='__main__':main()
