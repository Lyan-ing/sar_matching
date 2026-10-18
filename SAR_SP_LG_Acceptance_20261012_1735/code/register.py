"""Pre-inference fixed protocol and targeted input/version identity."""
import json,subprocess,sys
from pathlib import Path
from core import *
from sarbench.data import discover,describe
DATA=BASE/'MaritimeSARRegDataset/output/phase1_formal_20261009_1846'
REAL=BASE/'RealSAR_MultiViewV2_20261012_1259'
def register():
 assert not (ROOT/'protocol.json').exists(),'Already registered'
 refs=discover(DATA);assert len(refs)==900
 protocol=dict(created=now(),seed=42,methods=METHODS,feature_limit=1024,native_coordinates=True,image_input='original uint8 grayscale PNG, /255, no resizing',real_input='existing native_db0/1.png: native grid, fixed robust per-image display, no geographic initialization',superpoint_threshold=.0005,nearest_neighbor=dict(ratio=.8,mutual=True),RANSAC=dict(model='homography',reprojection_px=3,max_iterations=5000,confidence=.995,seed=42,min_matches=8,min_inliers=8,min_inlier_ratio=.3,min_grid_coverage=.0625,max_fit_rmse=3),gate=dict(policy='full',denominator_margin=.01,corner_area=[.1,10],edge_scale=[.1,10],max_jacobian_condition=20,min_inlier_hull=.01,min_isotropy=.005,max_cycle_mean_px=10,max_symmetric_rmse=3),success='accepted and mean GT sparse reprojection <=3/5/10px; denominator positive pairs only; no-overlap separate',model_selection='M1 best step900 and M2 best step3300 already locked; no selection or thresholds changed',validation='existing fixed 56-pair plan; NOT fresh validation; reconstructed online with original seeds; exact synthetic GT only',real_GT='no independent qualified GT: engineering diagnosis only',formal_run='one fixed-model run; crash resume only missing predictions; no automatic repeat or tuning',timing='end-to-end extraction+matching+geometry+gate, GPU synchronized; excludes load, decode, GT and serialization; methods cyclic order across pairs',GPUs=[0,2],group_statistics='pair and source-component levels, no claim chips are independent acquisitions')
 atomic_json(ROOT/'protocol.json',protocol)
 files=[OLD/'validation_plan.json',OLD/'data_identity.json',OLD/'runs/m1/selection_locked.json',FIX/'runs/m2/selection_locked.json',*WEIGHTS.values(),DATA/'manifest.jsonl',DATA/'version_freeze_20261009_1846.json',REAL/'candidate_regions.jsonl']
 files+=list((FIX/'code').glob('*.py'))+list((ROOT/'code').glob('*.py'))
 files += [BASE/'MaritimeSARRegistrationPhase2/sarbench'/n for n in ['matchers.py','estimate.py','reliability.py','evaluate.py','data.py']]
 ck=BASE/'MaritimeSARRegistrationPhase2/cache/torch/hub/checkpoints';files += [ck/'superpoint_v1.pth',ck/'superpoint_lightglue_v0-1_arxiv.pth']
 records=[]
 for ref in refs:
  paths=[ref.folder/f'view_{i}.png' for i in [ref.view_i,ref.view_j]]
  records.append(dict(id=ref.id,paths=list(map(str,paths)),sha256=[sha(x) for x in paths],shape=[list(cv2.imread(str(x),0).shape) for x in paths],component=ref.source_component))
 atomic_json(ROOT/'formal_inputs.json',records)
 rr=[]
 for s in REAL.joinpath('candidate_regions.jsonl').read_text().splitlines():
  r=json.loads(s);paths=[REAL/'regions'/r['sample_id']/f'native_db{i}.png' for i in [0,1]]
  for i,p in enumerate(paths):assert sha(p)==r['files'][p.name]
  rr.append(dict(id=r['sample_id'],site=r['site_name'],sensor=r['sensor'],component=r['component'],look=r['look_pair_category'],paths=list(map(str,paths)),sha256=[sha(p) for p in paths],shapes=[list(cv2.imread(str(p),0).shape) for p in paths]))
 assert len(rr)==8 and sum(r['look']=='different_look_verified' for r in rr)==6
 atomic_json(ROOT/'real_inputs.json',rr)
 env=subprocess.check_output(['conda','list','-n','match','--json'],text=True);(ROOT/'dependencies.json').write_text(env)
 git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE/'MaritimeSARRegistrationPhase2',text=True).strip()
 atomic_json(ROOT/'identity.json',dict(created=now(),git_phase2=git,files={str(p):sha(p) for p in files},dataset_summary=describe(refs),checkpoints={m:dict(path=str(p),sha256=sha(p)) for m,p in WEIGHTS.items()},source_index= json.loads((OLD/'data_identity.json').read_text()),protocol_sha256=sha(ROOT/'protocol.json')))
 print(json.dumps(dict(dataset=describe(refs),real=rr),ensure_ascii=False),flush=True)
if __name__=='__main__':register()
