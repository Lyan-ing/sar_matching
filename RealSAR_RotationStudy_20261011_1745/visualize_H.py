"""Four H-only diagnostic cases. No model choice or pseudo-label promotion."""
import sys
sys.dont_write_bytecode=True
import json,numpy as np,cv2
from study import ROOT,rows,OLD,save,Prediction,estimate
from visualize import plt,png,image,output

def main():
    folder=ROOT/'geometry_figures';folder.mkdir(exist_ok=False);special=json.loads((ROOT/'analysis/H_without_Affine.json').read_text());records=json.loads((ROOT/'analysis/graded_records.json').read_text());lookup={(r['sample_id'],r['method'],r['angle']):r for r in records};source={r['sample_id']:r for r in rows()};manifest=[]
    for s in special:
        sid=s['sample_id'];r=lookup[sid,s['method'],s['angle']];a=png(source[sid]['reference_native_png']);b=png(source[sid]['moving_native_png']);z=np.load(r['cache_path']);p0=z['points0'];p1=z['points1_native'];H=np.array(r['homography_diagnostic']['candidate_H']);mask=z['homography_inlier_mask'] if 'homography_inlier_mask' in z else estimate(Prediction(r['method'],p0.astype(np.float32),p1.astype(np.float32)),(512,512),model='homography',seed=42).inlier_mask
        fig,axs=plt.subplots(2,2,figsize=(14,10),layout='constrained');fig.suptitle(f"{sid} | {s['method']} CCW{s['angle']}° native | Affine gate=False / H gate=True / B=False\nH-point/map difference median={s['H_match_map_difference_median_px']:.2f}px; NOT independent GT error",fontsize=13)
        image(axs[0,0],np.concatenate([a,b],1),'Homography inlier proxy, restored native coordinates; not GT')
        for i in np.flatnonzero(mask)[::max(1,int(mask.sum())//50)]:axs[0,0].plot([p0[i,0],p1[i,0]+512],[p0[i,1],p1[i,1]],color='#E69F00',lw=.8)
        warped=cv2.warpPerspective(b,np.linalg.inv(H),(512,512));geo=png(OLD/'georeference_correspondences'/sid/'moving_native_geo_assisted.png');image(axs[0,1],np.stack([a,warped,warped],-1),'H image overlay, red A/cyan warped B');image(axs[1,0],np.stack([a,geo,geo],-1),'Map-coordinate overlay for comparison ONLY');axs[1,1].set_axis_off();axs[1,1].text(.01,.95,json.dumps(s,indent=2),va='top',family='monospace',fontsize=9)
        file=folder/f'{sid}_{s["method"]}_{s["angle"]}.png';output(fig,file);manifest.append(dict(s,figure=str(file)))
    save(folder/'manifest.json',manifest);print('H diagnostic figures',len(manifest),flush=True)

if __name__=='__main__':main()
