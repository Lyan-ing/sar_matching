"""Map disagreement of H diagnostics; same cached matches, NO B/GT promotion."""
import sys
sys.dont_write_bytecode=True
import json,csv
import numpy as np
from study import ROOT,rows,context,sample_valid,Prediction,estimate,regular_grid,perspective,save

def main():
    source={r['sample_id']:r for r in rows()};records=json.loads((ROOT/'analysis/graded_records.json').read_text());results=[];contexts={}
    for r in records:
        h=r.get('homography_diagnostic',{})
        if not h.get('accepted_full_gate'):continue
        sid=r['sample_id']
        if sid not in contexts:contexts[sid]=context(source[sid])
        valid,mapper=contexts[sid];z=np.load(r['cache_path']);a=z['points0'];b=z['points1_native'];H=np.array(h['candidate_H'])
        if 'homography_inlier_mask' in z:mask=z['homography_inlier_mask']
        else:
            # Old 0deg cache did not include H mask. Replay ONLY the frozen
            # estimator on these identical feature correspondences, not features.
            est=estimate(Prediction(r['method'],a.astype(np.float32),b.astype(np.float32)),(512,512),model='homography',seed=42);mask=est.inlier_mask
            assert np.allclose(est.candidate_H,H,atol=1e-7)
        eligible=mask&sample_valid(valid[0],a)&sample_valid(valid[1],b);e=np.linalg.norm(b[eligible]-mapper(a[eligible]),axis=1)
        grid=regular_grid((512,512),16);gmap=mapper(grid);gh=perspective(H,grid);ok=sample_valid(valid[0],grid)&sample_valid(valid[1],gmap)&sample_valid(valid[1],gh);ge=np.linalg.norm(gh[ok]-gmap[ok],axis=1)
        results.append({'sample_id':sid,'method':r['method'],'angle':r['angle'],'sensor':source[sid]['sensor'],'look_pair_category':source[sid]['look_pair_category'],'affine_gate':r['accepted'],'homography_gate':True,'H_native_valid_inliers':int(eligible.sum()),'H_match_map_difference_median_px':float(np.median(e)) if len(e) else None,'H_match_map_difference_p90_px':float(np.quantile(e,.9)) if len(e) else None,'common_native_grid_support':int(ok.sum()),'H_prediction_map_difference_p90_px':float(np.quantile(ge,.9)) if len(ge) else None,'independent_GT':False})
    file=ROOT/'analysis/homography_geographic_diagnostics.csv'
    with file.open('x',newline='') as f:
        w=csv.DictWriter(f,list(results[0]));w.writeheader();w.writerows(results)
    special=[r for r in results if not r['affine_gate']];save(ROOT/'analysis/H_without_Affine.json',special);print('H map diagnostics',len(results),'H without Affine',len(special),flush=True)

if __name__=='__main__':main()
