"""Read-only Phase 2 format/coordinate checks and bounded, stratified smoke inference."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from sarbench.data import discover,describe,load_images,load_gt
from sarbench.estimate import perspective,estimate
from sarbench.matchers import make_matcher
from sarbench.evaluate import evaluate
from sarbench.reports import save_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--smoke-methods',nargs='*',default=['sift','orb','phase']);a=p.parse_args();root=Path(a.root);out=Path(a.out)
if out.exists():raise FileExistsError(out)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=list(root.glob('file_checksums_*.json'));checked=0
for file in checks:
 for r in json.loads(file.read_text()):
  assert sha(root/r['path'])==r['sha256'],r['path'];checked+=1
receipt=json.loads(next(root.glob('ACCEPTED_*.json')).read_text());assert receipt['accepted'] and sha(receipt['validation_report'])==receipt['sha256']
refs=discover(root);info=describe(refs);assert len(refs)==900 and info['groups']==860 and info['missing_component']==0
worst=0.;sizes=Counter();positives=negatives=0
for r in refs:
 im0,im1=load_images(r);gt=load_gt(r);assert im0.shape==im1.shape;sizes[im0.shape[0]]+=1
 if gt['no_overlap']:
  assert gt['count']==0;negatives+=1
 else:
  assert gt['count']==512;positives+=1
  worst=max(worst,float(np.max(np.linalg.norm(perspective(gt['H'],gt['points0'])-gt['points1'],axis=1))))
assert positives==860 and negatives==40 and worst<1e-3
smoke=[]
# A fixed first representative of each profile: functionality checks only, no threshold selection.
for name in a.smoke_methods:
 matcher=make_matcher(name)
 for profile in sorted({r.profile for r in refs}):
  r=next(r for r in refs if r.profile==profile);im0,im1=load_images(r)
  pred=matcher.predict(im0,im1);est=estimate(pred,im0.shape);gt=load_gt(r);ev=evaluate(est,gt,im0.shape)
  smoke.append({'method':name,'id':r.id,'profile':profile,'accepted':est.accepted,'reason':est.reason,'n_keypoints0':pred.n_keypoints0,'n_keypoints1':pred.n_keypoints1,'n_matches':est.n_matches,'n_inliers':est.n_inliers,'decision_status':ev['decision_status']})
report={'dataset':info,'checked_frozen_files':checked,'positive':positives,'negative':negatives,'max_gt_direction_error_px':worst,'native_size_relations':dict(sizes),'smoke':smoke,'scope':'read-only compatibility/hash checks; smoke not scientific performance, no tuning','manifest_sha256':sha(root/'manifest.jsonl'),'receipt_sha256':sha(next(root.glob('ACCEPTED_*.json'))),'source_components':len({r.source_component for r in refs})}
save_json(out,report);print(json.dumps(report,indent=2))
