"""Truthful native scientific previews; all transforms and candidates labeled."""
import os,sys
sys.dont_write_bytecode=True
from pathlib import Path
os.environ['MPLCONFIGDIR']=str(Path(__file__).resolve().parent/'.mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from experiment import *
ATTR='Umbra SAR CC BY4.0; Copyright Capella Space CC BY4.0. Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026'
def finish(fig,path,title):
    fig.suptitle(title+' | ENGINEERING CANDIDATES; NO INDEPENDENT GT',fontsize=11)
    fig.text(.01,.003,ATTR,fontsize=5.5);fig.savefig(path,dpi=130);plt.close(fig)
def matches(ax,im0,im1,p0,p1,inliers,title):
    canvas=np.concatenate([im0,im1],axis=1);ax.imshow(canvas,cmap='gray',vmin=0,vmax=255);ax.set_title(title,fontsize=8);ax.set_axis_off()
    n=len(p0);indices=np.arange(n)[::max(1,(n+59)//60)]
    for i in indices:
        color='#0072B2' if inliers[i] else '#D55E00';ax.plot([p0[i,0],p1[i,0]+im0.shape[1]],[p0[i,1],p1[i,1]],color=color,lw=.6,alpha=.75)
    ax.text(0,.005,'blue: fitted inliers; orange: others. <=60 displayed; not verified.',transform=ax.transAxes,fontsize=5,color='white',bbox=dict(facecolor='black',alpha=.6))
def main():
    folder=ROOT/'figures';folder.mkdir(exist_ok=False);records=lines(ROOT/'analysis/all_records.jsonl');models=lines(ROOT/'geometry/models.jsonl');locals=lines(ROOT/'geometry/local.jsonl');manifest=[]
    for row in selected():
        sid=row['sample_id'];d=folder/sid;d.mkdir();raw,valid,old,_,mapper=load_pair(row)
        # Raw panels use explicit robust LINEAR display, no implication of physical calibration.
        fig,axes=plt.subplots(4,5,figsize=(15,10),layout='constrained');kinds=['raw_linear_display','original','linear','asinh','clahe']
        for side in [0,1]:
            for col,kind in enumerate(kinds):
                if kind=='raw_linear_display':im,meta=normalize(raw[side],valid[side],'linear',row[('reference' if side==0 else 'moving')+'_input_units'])
                elif kind=='original':im=old[side]
                else:im=np.array(Image.open(ROOT/'data'/sid/f'{kind}_{side}.png'))
                axes[side*2,col].imshow(im,cmap='gray',vmin=0,vmax=255);axes[side*2,col].set_axis_off();axes[side*2,col].set_title(f'{side}: {kind}',fontsize=8)
                if kind=='raw_linear_display':
                    a=raw[side][valid[side]];axes[side*2+1,col].hist(a,bins=64,color='#0072B2');axes[side*2+1,col].set_yscale('log');axes[side*2+1,col].set_xlabel('Raw source values; count log',fontsize=7)
                else:
                    axes[side*2+1,col].hist(im[valid[side]],bins=np.arange(-.5,256.5,1),color='#0072B2');axes[side*2+1,col].set_xlim(0,255);axes[side*2+1,col].set_xlabel('uint8; valid pixels only',fontsize=7)
                axes[side*2+1,col].tick_params(labelsize=6)
        finish(fig,d/'radiometry.png',sid+' | per-image display rules fixed; raw p1/p99 illustration')
        fig,axes=plt.subplots(3,4,figsize=(17,9),layout='constrained')
        for i,method in enumerate(METHODS):
            for j,c in enumerate(['original',*REPRESENTATIONS]):
                r=next(r for r in records if r['sample_id']==sid and r['method']==method and r['condition']==c);z=np.load(r['cache_path']);im=old if c=='original' else [np.array(Image.open(ROOT/'data'/sid/f'{c}_{s}_inference.png')) for s in [0,1]]
                matches(axes[i,j],*im,z['points0'],z['points1_native'],z['inlier_mask'],f'{method}/{c}: kp={r["n_keypoints0"]}/{r["n_keypoints1"]} M/I={r["n_matches"]}/{r["n_inliers"]}\ngate={r["accepted"]}; B={r["pseudo_quality"]}')
        finish(fig,d/'features.png',sid+' | unchanged detectors and Affine/full')
        contextfolder=context_path(sid);ci=read(contextfolder/'metadata.json');fig,axes=plt.subplots(4,2,figsize=(13,15),layout='constrained')
        for i in [0,1]:
            axes[0,i].imshow(old[i],cmap='gray',vmin=0,vmax=255);axes[0,i].set_title(f'native512 view{i}',fontsize=9);axes[0,i].set_axis_off()
            if ci['available']:
                im=np.array(Image.open(contextfolder/f'view{i}.png'));axes[1,i].imshow(im,cmap='gray',vmin=0,vmax=255);axes[1,i].add_patch(Rectangle((255.5,255.5),512,512,fill=False,edgecolor='#E69F00',lw=2));axes[1,i].set_title(f'actual1024 view{i}; old512 rectangle, no upsampling',fontsize=9);axes[1,i].set_axis_off()
        # Two method examples preserve accepted and rejected cases; all CSV covers three methods.
        for i,method in enumerate(['orb','lightglue']):
            r=next(r for r in records if r['sample_id']==sid and r['method']==method and r['condition']=='context_direct')
            if r.get('coarse_cache'):
                z=np.load(r['coarse_cache']);im=[np.array(Image.open(contextfolder/f'view{s}.png')) for s in [0,1]];matches(axes[2,i],*im,z['points0'],z['points1'],z['inlier_mask'],f'{method} image-only1024 coarse M/I={r["coarse_matches"]}/{r["coarse_inliers"]}\ncoarsegate={r["coarse_accepted"]};512directgate={r["accepted"]}')
            f=next(r for r in records if r['sample_id']==sid and r['method']==method and r['condition']=='context_fine')
            if f.get('cache_path'):
                z=np.load(f['cache_path']);matches(axes[3,i],*old,z['points0'],z['points1_native'],z['inlier_mask'],f'{method} restored native512 fine M/I={f["n_matches"]}/{f["n_inliers"]}; gate={f["accepted"]}')
            else:axes[3,i].text(.05,.45,'No local refinement output\n'+f['reason'],fontsize=10,wrap=True);axes[3,i].set_axis_off()
        finish(fig,d/'context.png',sid+' | image-only context -> coarse gate -> native512 warp/fine')
        fig,axes=plt.subplots(3,3,figsize=(15,9),layout='constrained')
        for i,method in enumerate(METHODS):
            r=next(r for r in records if r['sample_id']==sid and r['method']==method and r['condition']=='original');z=np.load(r['cache_path']);p0,p1=z['points0'],z['points1_native']
            for j,model in enumerate(['affine','homography']):
                g=next(r for r in models if r['sample_id']==sid and r['method']==method and r['condition']=='original' and r['model']==model);e=estimate(Prediction(method,p0,p1),(512,512),model=model,seed=42);mask=e.inlier_mask if e.inlier_mask is not None else np.zeros(len(p0),bool)
                matches(axes[i,j],*old,p0,p1,mask,f'{method}/{model}: I={g["n_inliers"]}; gate={g["accepted"]}\nmap median={g["geographic_difference_median_px"]}')
            ax=axes[i,2];ax.imshow(old[0],cmap='gray',vmin=0,vmax=255);ax.scatter(p0[:,0],p0[:,1],s=4,c='#E69F00');tiles=[r for r in locals if r['sample_id']==sid and r['condition']=='original' and r['method']==method]
            for r in tiles:
                ax.add_patch(Rectangle((r['tile_x']*128,r['tile_y']*128),128,128,fill=False,edgecolor='#0072B2' if r['accepted_basic'] else '#D55E00'))
                ax.text(r['tile_x']*128+3,r['tile_y']*128+15,f'I={r["n_inliers"]}',fontsize=6,color='yellow')
            ax.set_axis_off();ax.set_title(f'{method} local4x4, {len(tiles)} attempted\nSelf-fit; no whole-image GT',fontsize=8)
        finish(fig,d/'geometry.png',sid+' | SAME original matches; local extrapolation unverified')
        manifest.append({'sample_id':sid,'figures':[str(d/f'{name}.png') for name in ['radiometry','features','context','geometry']],'GT':'none'})
    summary=read(ROOT/'analysis/summary.json');fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained');conditions=['original',*REPRESENTATIONS,'context_direct','context_fine']
    for ax,m in zip(axes,METHODS):
        z=[next(r for r in summary if r['method']==m and r['condition']==c) for c in conditions];x=np.arange(len(z));ax.bar(x-.18,[r['accepted'] for r in z],.36,label='Affine/full accepted',color='#0072B2');ax.bar(x+.18,[r['B_eligible'] for r in z],.36,label='B eligible candidate',color='#E69F00',hatch='//');ax.set_xticks(x,conditions,rotation=45,ha='right');ax.set_ylim(0,12);ax.set_ylabel('crop pairs /12');ax.set_title(m);ax.legend(fontsize=7)
    finish(fig,folder/'ablation_acceptance.png','Single-factor diagnostic on12pairs /12acquisition pairs /8components')
    save(folder/'manifest.json',manifest);print('figures',len(manifest)*4+1,flush=True)
if __name__=='__main__':main()
