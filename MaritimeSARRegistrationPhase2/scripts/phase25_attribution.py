"""Post-hoc dynamic/boundary attribution; no tuning, network inference, or new gates."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sarbench.data import discover, load_gt, load_images
from sarbench.matchers import Prediction
from sarbench.estimate import estimate, perspective
from sarbench.reports import read_csv, write_csv, save_json, _float


def at(mask, points):
    xy = np.rint(points).astype(int); result = np.zeros(len(xy), bool); h,w = mask.shape
    inside = (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    result[inside] = mask[xy[inside,1],xy[inside,0]]>0
    return result


def fraction(bits): return float(np.mean(bits)) if len(bits) else None


parser = argparse.ArgumentParser(); parser.add_argument('--run',required=True); parser.add_argument('--out',required=True)
args=parser.parse_args(); run=Path(args.run).resolve(); out=Path(args.out).resolve()
if out.exists(): raise FileExistsError(out)
cv2.setNumThreads(1); out.mkdir()
records=[]; dynamic=[]; selection=json.loads((run/'development/selection.json').read_text())
for split, directory in [('dev','development'),('val','validation')]:
    refs={r.id:r for r in discover(run/'data_v2'/split,require_test=False)}
    lookup={(r['id'],r['method']):r for r in read_csv(run/directory/'results.csv')}
    for row in read_csv(run/directory/'features.csv'):
        ref=refs[row['id']]
        with np.load(row['cache_file']) as cache: p0=cache['points0'];p1=cache['points1'];k0=cache['keypoints0'];k1=cache['keypoints1']
        v=[cv2.imread(str(ref.folder/f'valid_{i}.png'),0) for i in [0,1]]
        d=[cv2.imread(str(ref.folder/f'dynamic_{i}.png'),0) for i in [0,1]]
        assert all(x is not None for x in v+d)
        boundary=[cv2.morphologyEx((x>0).astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)) for x in v]
        dyn=at(d[0],p0)|at(d[1],p1)
        records.append({'role':split,'id':ref.id,'variant':row['variant'],'profile':row['profile'],'source_scene':row['source_scene'],
                        'raw_matches':len(p0),'dynamic_raw_match_fraction':fraction(dyn),
                        'invalid_raw_point_fraction':fraction(~at(v[0],p0)|~at(v[1],p1)),
                        'invalid_keypoint_fraction0':fraction(~at(v[0],k0)),'invalid_keypoint_fraction1':fraction(~at(v[1],k1)),
                        'boundary_keypoint_fraction0':fraction(at(boundary[0],k0)),'boundary_keypoint_fraction1':fraction(at(boundary[1],k1))})
        algorithm=row['algorithm']; chosen=selection['selected'][algorithm]
        if split!='val' or row['profile']!='dynamic' or row['variant']!=chosen['variant']: continue
        method='@'.join(chosen[k] for k in ['variant','model','policy']); saved=lookup[(ref.id,method)]
        # Refit only persisted image-only points with the identical frozen seed to recover
        # original RANSAC inlier bits, which were not written separately. Assert candidate equality.
        pred=Prediction(algorithm,p0,p1); fitted=estimate(pred,(ref.meta['output_size'],)*2,model=chosen['model'],seed=42+int(ref.folder.name.split('_')[-1]))
        if saved['candidate_H']:
            assert np.allclose(fitted.candidate_H,np.asarray(json.loads(saved['candidate_H'])),rtol=1e-10,atol=1e-10)
        else: assert fitted.candidate_H is None
        assert fitted.n_inliers==int(saved['n_inliers'])
        # GT is consulted only AFTER refitting, exclusively for attribution.
        gt=load_gt(ref); projected=perspective(gt['H'],p0)
        static=at(v[0],p0)&at(v[1],p1)&~dyn&~at(d[1],projected)
        correct=static&at(v[1],projected)&(np.linalg.norm(projected-p1,axis=1)<=3)
        inliers=fitted.inlier_mask if fitted.inlier_mask is not None else np.zeros(len(p0),bool)
        dynamic.append({'id':ref.id,'method':method,'status':saved['decision_status'],'reason':saved['reason'],
                        'raw_matches':len(p0),'n_inliers':fitted.n_inliers,'dynamic_raw_match_fraction':fraction(dyn),
                        'dynamic_inlier_fraction':fraction(dyn[inliers]),'GT_correct_static_inliers3':int(correct[inliers].sum()),
                        'GT_correct_static_raw_matches3':int(correct.sum()),'epe_mean':_float(saved,'epe_mean')})
        if ref.id=='pair_000362:0-1' and algorithm=='lightglue':
            im0,im1=load_images(ref);fig,axes=plt.subplots(1,2,figsize=(8,4),layout='constrained')
            for ax,im,pts,mask in zip(axes,[im0,im1],[p0,p1],d):
                ax.imshow(im,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                ax.contour(mask>0,levels=[.5],colors='white',linewidths=.7)
                ax.scatter(pts[~inliers,0],pts[~inliers,1],marker='x',s=18,color='#E69F00',label='RANSAC outlier')
                ax.scatter(pts[inliers,0],pts[inliers,1],marker='o',s=18,facecolors='none',edgecolors='#56B4E9',label='RANSAC inlier')
                ax.set_xlim(0,im.shape[1]-1);ax.set_ylim(im.shape[0]-1,0);ax.set_xlabel('x (px)');ax.set_ylabel('y (px)');ax.legend(fontsize=7)
            fig.suptitle('Post-hoc dynamic support: selected LightGlue; white contour = GT dynamic region\nImage-only gate accepted; static GT EPE = 23.33 px')
            fig.savefig(out/'dynamic_inlier_support.png',dpi=200);plt.close(fig)
write_csv(out/'boundary_dynamic_feature_attribution.csv',records);write_csv(out/'dynamic_selected_inliers.csv',dynamic)
summary=[]
for method in sorted({r['method'] for r in dynamic}):
    for status in ['success','incorrect_accept','rejected','no_output']:
        subset=[r for r in dynamic if r['method']==method and r['status']==status]
        if not subset: continue
        s={'method':method,'status':status,'pairs':len(subset)}
        for key in ['dynamic_raw_match_fraction','dynamic_inlier_fraction','GT_correct_static_inliers3','GT_correct_static_raw_matches3']:
            values=[r[key] for r in subset if r[key] is not None];s[key+'_mean']=float(np.mean(values)) if values else None
        summary.append(s)
write_csv(out/'dynamic_attribution_summary.csv',summary)
save_json(out/'ATTRIBUTION_COMPLETE.json',{'accepted':True,'features':len(records),'selected_dynamic_refits':len(dynamic),'same_candidate_verified':True,
                                         'note':'GT masks only post-hoc attribution; no acceptance/model/parameter selection was changed. Boundary proximity is not proof of a synthetic shortcut.',
                                         'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},
                                         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
print(json.dumps({'features':len(records),'selected_dynamic_refits':len(dynamic),'summary':summary},indent=2))
