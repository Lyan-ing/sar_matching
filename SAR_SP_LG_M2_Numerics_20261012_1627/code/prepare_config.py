"""Freeze the initial model/training/selection choices before any training."""
import json
from common import ROOT,BASE,sha,write_new,readrows,now
def main():
 n=len(readrows(ROOT/'train_sources.jsonl'))
 for variant,gpu,lr in [('m1',0,1e-4),('m2',1,1e-5)]:
  cfg=dict(run_id=ROOT.name+'_'+variant,variant=variant,physical_gpu=gpu,seed=26101201+(variant=='m2'),input_size=256,microbatch=1,batch_accumulation=2,learning_rate=lr,weight_decay=1e-4,temperature=.1,preserve_weight=.1,max_steps=6000,validate_every=300,checkpoint_every=100,early_stop_min_steps=1200,early_stop_patience=6,train_sources=n,detector_max_keypoints=1024,detector_threshold=.0005,geometry='homography+frozen_full_gate',selection_rule=['maximize fixed validation Success@5px','minimize no-overlap false accept rate','maximize descriptor Recall@1'],curriculum='0:600 easy;600:1800 medium;1800+ 20%easy60%medium20%low',source_sampling='55%nearshore35%offshore10%background;uniform component then source',augmentation='independent gamma/contrast and Gamma30 amplitude speckle; static supervision excludes original ship bbox+5px; later 15% simulated dynamic-region occlusion',first_validation_step=25,training_is_synthetic_SAR_adaptation_not_real_view_supervision=True)
  write_new(ROOT/'configs'/f'{variant}.json',cfg)
 write_new(ROOT/'configs/m0.json',dict(variant='m0',weights='official cached SuperPoint + LightGlue',frozen=True,input_protocol='image_only',resize=None,max_keypoints=1024,detection_threshold=.0005,geometry='homography+frozen_full_gate',validation_plan_sha256=sha(ROOT/'validation_plan.json')))
 weights=BASE/'MaritimeSARRegistrationPhase2/cache/torch/hub/checkpoints'
 write_new(ROOT/'experiment_identity.json',dict(created_at=now(),source_index=json.loads((ROOT/'data_identity.json').read_text()),train_index_sha256=sha(ROOT/'train_sources.jsonl'),val_index_sha256=sha(ROOT/'val_sources.jsonl'),validation_plan_sha256=sha(ROOT/'validation_plan.json'),models={p.name:sha(p) for p in weights.glob('*.pth')},frozen900_used_for_selection=False,old_data_modified=False))
 print('CONFIG FROZEN before training')
if __name__=='__main__':main()
