from support22 import *
from collections import Counter
def main():
 train=rows(E['recommended_train_reference']);val=rows(E['recommended_val_all']);local=rows(E['recommended_train_local']);tr={r['known_component'] for r in train};va={r['known_component'] for r in val};assert not tr&va
 assert len(local)==2324 and len({r['source_path'] for r in local})==677
 test=rows(BASE/'MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl')
 # Existing V2.2 provenance audit carries aliases joining background with HRSID parents.
 source_split={r['image_path']:r['split'] for r in test};assert all(source_split[r['source_path']]=='train' for r in train);assert all(source_split[r['source_path']]=='val' for r in val)
 chosen=[]
 for role,n in [('structural_positive',64),('limited_structure',32),('no_structural_positive',32),('no_overlap',16)]:
  pool=sorted([r for r in val if r['loss_role']==role],key=lambda r:stable_seed(202622,r['relation_id']));groups={}
  for r in pool:groups.setdefault(r['known_component'],[]).append(r)
  while len([r for r in chosen if r['loss_role']==role])<n:
   for c in sorted(groups):
    if groups[c] and len([r for r in chosen if r['loss_role']==role])<n:chosen.append(groups[c].pop(0))
 protocol=dict(created=now(),entry=str(ENTRY),entry_sha=sha(ENTRY),seed=202622,arm='V2-R',pool='core',effective_batch=8,microbatch=1,precision='FP32; protects historical FP16 descriptor overflow',main_cap=6000,validate_every=1000,selection_subset_count=len(chosen),selection_ids=[r['relation_id'] for r in chosen],selection_rule='Blank+gray zero and noise<=initial, then maximum natural global S@5, minimum no-overlap FPR, maximum local oracle R1. Also retain initial candidate; all failures retained.',final_equal_protocol='actual per-view shared count min(128, all compared models and ORB), separate from natural1024',global_error='Mean bidirectional reprojection on ORIGINAL saved static checks; full-Success only version-qualified global positives',gate='Unchanged historical Homography RANSAC3px + full reliability gate; no GT in prediction',training_roles=dict(M2='55/35/0/10 references, actual roles re-evaluated online',B='40/25/27/8 references, actual roles re-evaluated online'),B_loss_correction='Saved reliable anchors only for descriptor CE, N>=4; detector anchors>=1; keep old .05 positive+teacherKL+evidenced calibration+blank+.2 equivariance; masks exclude unknown',main_online=True,attribution='Three fixed-pixel M2 runs, each 6000 steps/effective batch8/lr1e-5; old strict selector vs same pixels recovered local vs only new ROI. Source-count differences reported; not a pure source-matched causal decomposition.',ship_ablation='After B selection, two same-initialization 1000-step continuations with identical stream; default off vs .15 synthetic ship auxiliary on; no ground GT relabeling',v21_common_ids=[r['sample_id'] for r in rows(E['frozen_validation']['V21_common133']['V2-R'])],no_900_rerun=True,no_real_precision=True)
 write(out('protocol'),protocol)
 identity_paths=[ENTRY,Path(E['config']),Path(E['recommended_train_reference']),Path(E['recommended_val_all']),M2,B,Path(E['frozen_validation']['V1_standard']['val_parent_core']),Path(E['frozen_validation']['V21_all']['V2-R/val/core'])]+list((V22/'code').glob('*.py'))+list((V21/'code').glob('*.py'))
 write(out('input_identity'),dict(at=now(),sha256={str(p):sha(p) for p in identity_paths},train_rows=len(train),descriptor_rows=len(local),descriptor_sources=677,train_components=len(tr),val_components=len(va),component_overlap=sorted(tr&va),roles=dict(Counter(r['loss_role'] for r in train)),note='Known component isolation only; physical acquisition independence unproven'))
 cfg=dict(seed=202622,steps=6000,effective_batch=8,lr=1e-5,detector_lr=1e-5,structure_lr=1e-4,weight_decay=1e-4,checkpoint_every=500,validate_every=1000,workers=4,online=True,early_stop_validations=4,min_steps=3000,wall_seconds=28800)
 write(out('config_precheck'),cfg)
 print(json.dumps(dict(prepared=True,train=len(train),val=len(val),selection=len(chosen),known_component_overlap=[])))
if __name__=='__main__':main()
