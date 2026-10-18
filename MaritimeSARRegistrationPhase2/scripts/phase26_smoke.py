"""Verify the new orchestration against frozen OLD dev predictions, never confirm."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from phase26_benchmark import old,single_pipeline
from sarbench.data import discover,load_images,load_gt
from sarbench.evaluate import evaluate
from sarbench.reports import read_csv,save_json

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
run=Path(a.run);out=run/'old_dev_smoke.json'
if out.exists():raise FileExistsError(out)
phase25=Path('output/phase25_20261010_0000');selected=json.loads((phase25/'development/selection.json').read_text())
ref=discover(phase25/'data_v2/dev',require_test=False)[0];im0,im1=load_images(ref)
cv2.setNumThreads(1);torch.set_num_threads(1);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
prior=read_csv(phase25/'development/results.csv');checks=[]
for algorithm in ['sift','orb','lightglue']:
    for settings in [{'variant':algorithm+'_base','model':'homography','policy':'baseline'},selected['selected'][algorithm]]:
        variant=settings['variant'];matcher=old.make_variant(variant,2048)
        old.predict_variant(variant,matcher,im0,im1);torch.cuda.synchronize()
        decision=single_pipeline(variant,matcher,im0,im1,settings['model'],settings['policy'],42)
        row=next(r for r in prior if r['id']==ref.id and r['variant']==variant and r['model']==settings['model'] and r['policy']==settings['policy'])
        with np.load(row['cache_file']) as frozen:
            for key in ['points0','points1','keypoints0','keypoints1']:
                assert np.array_equal(getattr(decision['pred'],key),frozen[key]),(variant,key)
        ev=evaluate(decision['estimate'],load_gt(ref),im0.shape)
        assert ev['decision_status']==row['decision_status'],variant
        assert decision['estimate'].n_inliers==int(row['n_inliers']),variant
        if decision['estimate'].H is not None:assert np.allclose(decision['estimate'].H,json.loads(row['pred_H']),atol=1e-8,rtol=1e-8)
        checks.append({'variant':variant,'model':settings['model'],'policy':settings['policy'],'exact_cached_features':True,'frozen_decision_identical':True})
save_json(out,{'accepted':True,'input_role':'old_dev_only','id':ref.id,'checks':checks,'script_sha256':old.checksum(__file__)})
print(json.dumps({'accepted':True,'checks':len(checks),'input_role':'old_dev_only'},indent=2))
