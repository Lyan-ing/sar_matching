"""Publication-style exploratory figures. No physical correspondence/GT claims."""
import sys
sys.dont_write_bytecode=True
import json,os
from pathlib import Path
from collections import defaultdict
import numpy as np,cv2
from PIL import Image
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplcache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from study import ROOT,OLD,MANUAL,rows,cfg,save,perspective,selection

def checker(a,b,cell=32):
    y,x=np.indices(a.shape);return np.where(((x//cell+y//cell)%2)==0,a,b)
def png(path):return np.array(Image.open(path))
def image(ax,a,title):ax.imshow(a,cmap='gray',vmin=0,vmax=255,interpolation='nearest');ax.set_title(title,fontsize=10);ax.set_axis_off()
def output(fig,path):fig.savefig(path,dpi=160);plt.close(fig)
def matches(ax,a,b,record):
    canvas=np.concatenate([a,b],axis=1);ax.imshow(canvas,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    z=np.load(record['cache_path']);p0=z['points0'];p1=z['points1_native'];mask=z['eligible_inlier_mask'];step=max(1,len(p0)//50)
    for i in range(0,len(p0),step):
        color='#009E73' if mask[i] else '#D55E00';ax.plot([p0[i,0],p1[i,0]+512],[p0[i,1],p1[i,1]],color=color,lw=.6,alpha=.65)
    ax.set_xlim(0,1024);ax.set_ylim(512,0);ax.set_axis_off()
    d=record.get('distribution',{});hulls=[d.get('native_inlier'+str(i),{}).get('hull_fraction',0) for i in [0,1]]
    ax.set_title(f"CCW {record['angle']}° | kp={record.get('n_keypoints0')}/{record.get('n_keypoints1')} matches={record.get('n_matches')} inliers={record.get('n_inliers')}\ngate={record['accepted']} B={record['pseudo_quality']!='unreliable_candidate'} hull={min(hulls):.3f} reason={record['reason']}",fontsize=10)
    ax.text(.01,.02,'native coordinates; green=inlier proxy, orange=other matches; NOT GT',transform=ax.transAxes,color='white',fontsize=8,bbox={'facecolor':'black','alpha':.65})

def main():
    directory=ROOT/'figures';directory.mkdir(exist_ok=False);source={r['sample_id']:r for r in rows()};records=json.loads((ROOT/'analysis/graded_records.json').read_text());lookup={(r['sample_id'],r['method'],r['angle']):r for r in records};ids=list(dict.fromkeys(cfg()['diagnostic8']+cfg()['representative24']));manifest=[]
    for j,sid in enumerate(ids):
        row=source[sid];a=png(row['reference_native_png']);b=png(row['moving_native_png']);g=OLD/'georeference_correspondences'/sid;warp=png(g/'moving_native_geo_assisted.png');mask=png(g/'geographic_valid_overlap0.png')>0;points=json.loads((g/'correspondences.json').read_text())['points'];folder=directory/sid;folder.mkdir()
        fig,axs=plt.subplots(2,3,figsize=(15,10),layout='constrained');fig.suptitle(f"{sid} | {row['sensor']} | {row['look_pair_category']}\nGeographic reference ONLY; static structures/scatter centers require independent confirmation",fontsize=13)
        image(axs[0,0],a,'Reference native');image(axs[0,1],b,'Moving native');image(axs[0,2],warp,'Moving native -> reference geographic warp')
        visible=np.stack([a]*3,-1);visible[mask,1]=255;image(axs[1,0],visible,'Green: geographic valid overlap, NOT GT support')
        image(axs[1,1],np.stack([a,warp,warp],-1),'Transparent/complementary overlay: red A, cyan B');image(axs[1,2],checker(a,warp),'32px checkerboard: native A / geo-warped B')
        for p in points[::max(1,len(points)//12)]:
            fig.add_artist(ConnectionPatch(p['xy0'],p['xy1'],coordsA='data',coordsB='data',axesA=axs[0,0],axesB=axs[0,1],color='#E69F00',alpha=.5,lw=.7))
        output(fig,folder/'geographic_structure.png')
        # Display only: choose geographic grid patches with large shared display
        # variation, far apart. These are structure CANDIDATES, not landmarks/GT.
        centers=[]
        for p in points:
            x,y=np.rint(p['xy0']).astype(int)
            if 32<=x<480 and 32<=y<480 and mask[y-32:y+32,x-32:x+32].mean()>.8:
                score=min(np.std(a[y-32:y+32,x-32:x+32]),np.std(warp[y-32:y+32,x-32:x+32]));centers.append((score,x,y))
        chosen=[]
        for score,x,y in sorted(centers,reverse=True):
            if all(np.hypot(x-p[0],y-p[1])>=100 for p in chosen):chosen.append((x,y))
            if len(chosen)==3:break
        fig,axs=plt.subplots(2,3,figsize=(12,8),layout='constrained');fig.suptitle('64px patches enlarged for display; structure candidates ONLY, physical identity unconfirmed')
        for col in range(3):
            if col>=len(chosen):
                for ax in axs[:,col]:ax.text(.1,.5,'Insufficient valid patch support');ax.set_axis_off()
                continue
            x,y=chosen[col];image(axs[0,col],a[y-32:y+32,x-32:x+32],f'A candidate center ({x},{y})');image(axs[1,col],warp[y-32:y+32,x-32:x+32],'B at same map patch; bilinear geographic resampling')
        output(fig,folder/'structure_candidate_patches.png')
        for method in ['sift','orb','lightglue']:
            group=[lookup[sid,method,angle] for angle in [0,90,180,270]];fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained');fig.suptitle(f'{sid} | {method} | native input; integer CCW rotations, no interpolation; restored native points; NOT GT')
            for ax,r in zip(axs.ravel(),group):matches(ax,a,b,r)
            output(fig,folder/f'{method}_four_angles.png')
            chosenrec=selection(group) or group[0];H=chosenrec.get('candidate_H');alt=chosenrec.get('homography_diagnostic',{});hh=alt.get('candidate_H');fig,axs=plt.subplots(2,3,figsize=(15,10),layout='constrained');fig.suptitle(f"{sid} {method} native | prediction-side first-gate angle {chosenrec['angle']}°\nAffine and Homography use SAME matches; geographic agreement/fit NOT accuracy")
            image(axs[0,0],np.stack([a,warp,warp],-1),'Geographic overlay (map prior only)')
            for col,matrix,title in [(1,H,'Affine'),(2,hh,'Homography diagnostic')]:
                if matrix is None:
                    for ax in axs[:,col]:ax.text(.08,.5,title+' unavailable',fontsize=13);ax.set_axis_off()
                    continue
                warped=cv2.warpPerspective(b,np.linalg.inv(np.asarray(matrix)),(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                accepted=chosenrec['accepted'] if col==1 else alt.get('accepted_full_gate',False)
                image(axs[0,col],np.stack([a,warped,warped],-1),f'{title} native overlay | gate={accepted}')
                image(axs[1,col],checker(a,warped),f'{title} checkerboard; no GT verified')
            if chosenrec.get('pseudo_support_path'):image(axs[1,0],png(chosenrec['pseudo_support_path']),'Pseudo support proxy ONLY; NOT verified GT mask')
            else:image(axs[1,0],mask.astype(np.uint8)*255,'Geographic support only; no B pseudo region')
            output(fig,folder/f'{method}_geometry.png')
        manifest.append({'sample_id':sid,'provider':row['sensor'],'input_protocol':'native','angles_ccw':[0,90,180,270],'candidate_patch_centers':chosen,'patch_selection':'large shared display variance at fixed geographic grid, >100px separation; no static identity verification','files':[str(p.relative_to(ROOT)) for p in folder.glob('*.png')],'human_confirmed':False,'GT':False})
        if (j+1)%5==0:print('Figure pairs',j+1,'/',len(ids),flush=True)
    summary=json.loads((ROOT/'analysis/rotation_summary.json').read_text());fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for col,method in enumerate(['sift','orb','lightglue']):
        g=[r for r in summary if r['method']==method];x=np.arange(4)
        for ax,key,label in [(axs[0,col],'gate_accepted','Original image-only gate'),(axs[1,col],'B_general_eligible','Frozen B candidate conditions')]:
            values=[r[key] for r in g];ax.bar(x,values,color='#0072B2');ax.set_xticks(x,[str(r['angle'])+'°' for r in g]);ax.set_ylim(0,max(35,max(values)+5));ax.set_title(method+' '+label);ax.set_ylabel('crop pairs / 191');ax.bar_label(ax.containers[0])
    fig.suptitle('Exploratory real SAR rotations: acceptance/candidates, NOT true registration accuracy');output(fig,directory/'rotation_acceptance.png')
    fig,axs=plt.subplots(3,3,figsize=(14,11),layout='constrained')
    for col,method in enumerate(['sift','orb','lightglue']):
        g=[r for r in summary if r['method']==method];angles=[r['angle'] for r in g]
        for ax,key,label in [(axs[0,col],'keypoints1','moving keypoints'),(axs[1,col],'matches','descriptor matches'),(axs[2,col],'inliers','Affine inliers')]:
            median=[r[key]['median'] for r in g];p90=[r[key]['p90'] for r in g];ax.plot(angles,median,'o-',label='median');ax.plot(angles,p90,'s--',label='p90');ax.set_xticks(angles);ax.set_ylim(bottom=0);ax.set_title(method+' '+label);ax.set_xlabel('moving CCW degrees');ax.set_ylabel('count per crop pair');ax.legend(fontsize=8)
    fig.suptitle('Counts over 191 crops per angle: median / p90 (not uncertainty intervals or independent replicates)');output(fig,directory/'feature_counts.png')
    stages=json.loads((ROOT/'analysis/failure_stages.json').read_text());fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained');colors=['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#777777']
    for ax,method in zip(axs,['sift','orb','lightglue']):
        group=[r for r in stages if r['protocol']=='native_0' and r['method']==method];ax.barh([r['stage'].split('_')[0] for r in group],[r['prediction_rows'] for r in group],color=colors[:len(group)]);ax.set_xlim(0,191);ax.set_xlabel('prediction rows / 191 pairs');ax.set_title(method+' baseline');ax.bar_label(ax.containers[0])
    fig.suptitle('Failure stages: A detection, B matching, C geometry, D QA; physical structure E remains unconfirmed');output(fig,directory/'failure_stages.png')
    negatives=json.loads((ROOT/'analysis/negative_summary.json').read_text());fig,axs=plt.subplots(1,3,figsize=(14,5),layout='constrained')
    for ax,method in zip(axs,['sift','orb','lightglue']):
        g=[r for r in negatives if r['method']==method];ax.bar([str(r['angle']) for r in g],[r['wrong_geographic_accept_pairs'] for r in g],color='#D55E00');ax.set_ylim(0,30);ax.set_ylabel('wrong accepts / 30 known geographic negatives');ax.set_title(method);ax.bar_label(ax.containers[0])
    fig.suptitle('Negative diagnostics: no common geographic support, repeated sources; no threshold tuning');output(fig,directory/'negative_acceptance.png')
    save(directory/'manifest.json',manifest);print('Representative plots',len(manifest),'unique chip pairs',flush=True)

if __name__=='__main__':main()
