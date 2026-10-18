from ss_common import *
from collections import Counter

def main():
 train=rows(OLD/'train_sources.jsonl');vals=rows(OLD/'val_sources.jsonl');index=rows(BASE/'MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl')
 prior=json.loads((M3ROOT/'validation_plan.json').read_text());old=json.loads((OLD/'validation_plan.json').read_text())
 used={component(s['source']) for s in prior+old};allvc={component(r) for r in vals};fresh=allvc-used
 tc={component(r) for r in train};testc={component(r) for r in index if r['split']=='test'}
 assert not tc&allvc and not tc&testc and not allvc&testc
 # No untouched known Val components remain. Reuse frozen pixels, new component-separated internal protocol.
 cs=sorted({component(s['source']) for s in prior},key=lambda c:hashlib.sha256((c+'26101321').encode()).hexdigest())
 selects=set(cs[:int(.65*len(cs))]);hold=set(cs)-selects
 plans={'select':[s for s in prior if component(s['source']) in selects],'holdout':[s for s in prior if component(s['source']) in hold]}
 for name,plan in plans.items():write_new(output('plan_'+name),plan)
 (ROOT/'validation_data').symlink_to(M3ROOT/'validation_data',target_is_directory=True)
 write_new(output('data_identity'),dict(train_sources=str(OLD/'train_sources.jsonl'),train_sha=sha(OLD/'train_sources.jsonl'),source_index_sha=sha(BASE/'MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl'),train_components=len(tc),val_components=len(allvc),test_components=len(testc),fresh_val_components=len(fresh),intersection_train_val=[],intersection_train_test=[],intersection_select_holdout=[],splits={k:dict(pairs=len(v),components=len({component(s['source']) for s in v}),strata=dict(Counter(s['source']['dataset']+':'+s['regime'] for s in v))) for k,v in plans.items()},caveat='All Val components already observed in M1/M2/M3 development. New internal holdout is untouched by THIS experiment selection, not globally unseen or independent raw acquisitions. No 900 benchmark used. SSDD raw acquisition independence unproven.'))
 config=dict(seed=26101321,physical_gpu=1,input_size=256,steps=3000,validate_every=500,early_stop_min=1000,patience=3,microbatch=1,accumulation=2,structure_lr=1e-4,descriptor_control_lr=1e-5,detector_lr=1e-6,weight_decay=1e-4,precision='FP32 TF32 off',structure_scales=[1,2,4],structure_residual='zero final 1x1 initialization into pre-ReLU convDa',descriptor_loss='actual detached detector point GT projection, symmetric InfoNCE T=.1 + .1 M2 preserve, exclude neighbors <8px',detector_loss='.05 canonical structural positive CE + 1 teacher KL +1 weak/invalid nondynamic dustbin CE +1 black/gray dustbin CE +.2 linear response consistency',variants=['control','a','b'],B_initialization='same M2 as A, joint training from start; no extra A warmstart steps',primary_selection='hard gates: black/gray 0 points, max density <=2x M2 mean, no-overlap FPR<=.05; then equal-M2-count component/category S5, correct matches, repeatability',NMS_radius=4,threshold=.0005,max_keypoints=1024,geometry='frozen historical Homography/RANSAC3/full',training_data='same original Train index, component-uniform stratified; ships excluded',no_real_for_selection=True,no900=True)
 write_new(output('config'),config)
 write_new(output('baseline_records'),json.loads((M3ROOT/'validation_results.json').read_text()))
 write_new(output('real_inputs'),json.loads((M3ROOT/'real_inputs.json').read_text()))
 print(json.dumps(dict(root=str(ROOT),fresh_components=len(fresh),split=config['primary_selection'],plans={k:len(v) for k,v in plans.items()})))
if __name__=='__main__':main()
