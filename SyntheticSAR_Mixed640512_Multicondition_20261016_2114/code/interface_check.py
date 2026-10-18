"""Actual frozen CPU forwards, no optimizer, training, model selection or GT-guided matching."""
from reader import *
from score import assess,summaries
from inference import predict as orb_predict
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TORCH_HOME']=str(BASE/'MaritimeSARRegistrationPhase2/cache/torch')
sys.path.extend([str(BASE/'MaritimeSARRegistrationPhase2'),str(BASE/'SAR_SP_LG_StructureStable_20261013_2210/code')])
import torch
torch.set_num_threads(2)
import importlib.util
# Historical modules also use the generic name 'common'. Isolate that dependency
# during import, without editing the frozen model or replacing this dataset module.
current_common=sys.modules['common']
legacy_spec=importlib.util.spec_from_file_location('common',BASE/'SAR_SP_LG_M2_Numerics_20261012_1627/code/common.py')
legacy_common=importlib.util.module_from_spec(legacy_spec);sys.modules['common']=legacy_common;legacy_spec.loader.exec_module(legacy_common)
from network_corrected import SARNet
sys.modules['common']=current_common
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
import loss22
# Its historical zero helper assumes a trainable parameter. A frozen, no-grad
# interface check needs the identical numeric zero without enabling training.
loss22._zero=lambda model:next(model.parameters()).sum()*0
from loss22 import m2_loss

@torch.no_grad()
def predict(m,images):
    fs=[m.extract(torch.from_numpy(a.copy()).float()[None,None]/255) for a in images]
    kp=[f['keypoints'][0].cpu().numpy() for f in fs];ds=[f['descriptors'][0].cpu().numpy() for f in fs];matches=[]
    if min(map(len,ds))>=2:
        bf=cv2.BFMatcher(cv2.NORM_L2);fw=[a for a,b in bf.knnMatch(ds[0],ds[1],k=2) if a.distance<.8*b.distance];back={a.queryIdx:a.trainIdx for a,b in bf.knnMatch(ds[1],ds[0],k=2) if a.distance<.8*b.distance};matches=[a for a in fw if back.get(a.trainIdx)==a.queryIdx]
    p=np.array([kp[0][x.queryIdx] for x in matches],np.float32).reshape(-1,2);q=np.array([kp[1][x.trainIdx] for x in matches],np.float32).reshape(-1,2);H=None;inlier=None
    cv2.setRNGSeed(42)
    if len(p)>=4:H,inlier=cv2.findHomography(p,q,cv2.RANSAC,3.)
    return dict(accepted=H is not None and bool(np.isfinite(H).all()) and int(inlier.sum())>=8,
        H_AB=H.tolist() if H is not None else None,matches_A=p.tolist(),matches_B=q.tolist(),keypoints_A=kp[0].tolist(),keypoints_B=kp[1].tolist(),
        method='frozen_M2_SAR_SuperPoint_mutual_NN_.8_RANSAC3_min8_CPU_not_formal_SP_LG')

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args();rr=rows(a.manifest);chosen=[]
    for n in [512,640]:
        for kind in ['structural_positive','limited_structure','no_structural_positive','no_overlap']:
            pool=[r for r in rr if r['width']==n and r['loss_role']==kind]
            if pool:chosen.append(pool[0])
    checkpoint=BASE/'SAR_SP_LG_M2_Numerics_20261012_1627/runs/m2/best.ckpt';before=sha(checkpoint)
    model=SARNet('m2').cpu().eval().requires_grad_(False);losses=[];predictions=[];scores=[]
    for r in chosen:
        z=load_pair(r);tensor=tensors(z);assert tensor['images'].shape==(2,1,r['width'],r['width'])
        with torch.no_grad():
            # The existing V2.2 M2 loss accepts exactly the same explicit masks/points.
            loss,info=m2_loss(model,z,device='cpu');assert torch.isfinite(loss)
        losses.append(dict(pair_id=r['pair_id'],size=r['width'],role=r['loss_role'],loss=float(loss),info=info))
        for name,fn in [('ORB',lambda:orb_predict(r['image_A'],r['image_B'])),('frozen_SAR_SuperPoint',lambda:predict(model,z['images']))]:
            start=time.monotonic();pr=fn();pr.update(pair_id=r['pair_id'],baseline=name);predictions.append(pr);sc=assess(r,pr);sc.update(baseline=name,elapsed_seconds=time.monotonic()-start);scores.append(sc)
    assert not torch.cuda.is_initialized() and before==sha(checkpoint)
    t=stamp();jsonl(ROOT/f'meta/BASELINE_SMOKE_PREDICTIONS_{t}.jsonl',predictions);jsonl(ROOT/f'meta/BASELINE_SMOKE_SCORES_{t}.jsonl',scores)
    save(ROOT/f'meta/INTERFACE_CHECK_{t}.json',dict(passed=True,created=now(),cases=len(chosen),loss_cases=losses,optimizer_steps=0,cuda_initialized=False,
        frozen_checkpoint=str(checkpoint),checkpoint_hash_before=before,checkpoint_hash_after=sha(checkpoint),
        baseline_summary={b:summaries([s for s in scores if s['baseline']==b]) for b in ['ORB','frozen_SAR_SuperPoint']},
        baseline_smoke_only_not_dataset_acceptance_or_formal_performance=True,legacy_zero_helper_adapted_only_for_frozen_no_grad_check=True))
    print(json.dumps(dict(passed=True,cases=len(chosen),model_optimizer_steps=0,CPU_only=True)),flush=True)
if __name__=='__main__':main()
