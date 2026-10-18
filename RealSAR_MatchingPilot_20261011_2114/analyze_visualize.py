"""Engineering diagnostics and truthful visual inspection; never independent GT."""
from core import *
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
BLUE='#0072B2';ORANGE='#D55E00';MAP='#CC79A7'
def csvwrite(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader()
        w.writerows({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
def imagepair(ax,images):
    ax.imshow(np.hstack(images),cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(-.5,1023.5,511.5,-.5))
    ax.axvline(511.5,color='white',lw=1);ax.set_xlim(-.5,1023.5);ax.set_ylim(511.5,-.5)
    ax.set_xlabel('reference x (native px) | moving x +512 (native px)');ax.set_ylabel('native y (px)')
def subset(ids,n=60):return ids[np.linspace(0,len(ids)-1,min(len(ids),n)).astype(int)] if len(ids) else []
def savefig(fig,p,caption):
    if p.exists():raise FileExistsError(p)
    fig.savefig(p,dpi=130,facecolor='white');plt.close(fig)
    manifest.append(dict(file=str(p.relative_to(ROOT)),sha256=sha(p),caption=caption,independent_GT=False))
def matchfigure(folder,r,c,images):
    p0,p1,ins=c['points0'],c['points1_native'],c['inlier_mask'];good=c['native_valid_mask']
    fig,axs=plt.subplots(3,1,figsize=(14,19),layout='constrained')
    title=f"{r['sample_id']} | {r['method']} | {r['look_pair_category']}\n{r['protocol']} | matches={len(p0)}, RANSAC inliers={ins.sum()}, native-valid={good.sum()}\n{r['decision']} ({r['reason']}); C=0; all plotted support is UNVERIFIED"
    fig.suptitle(title,fontsize=12)
    for ax,flag,color,kind in [(axs[0],ins,BLUE,'RANSAC inliers (candidate, NOT correct GT)'),(axs[1],~ins,ORANGE,'RANSAC outliers / no model (retain rejected outputs)')]:
        imagepair(ax,images);ids=np.flatnonzero(flag)
        for i in subset(ids):
            ax.plot([p0[i,0],p1[i,0]+512],[p0[i,1],p1[i,1]],color=color,lw=.6,alpha=.7)
            ax.scatter([p0[i,0],p1[i,0]+512],[p0[i,1],p1[i,1]],s=9,marker='o' if flag is ins else 'x',color=color)
        ax.set_title(f'{kind}: all {len(ids)} counted; at most60 deterministic evenly-indexed lines shown')
        if not len(ids):ax.text(256,256,'No matches in this category',color='yellow',ha='center')
    imagepair(axs[2],images);g=c['geographic_moving']
    for i in subset(np.arange(len(p0)),40):
        axs[2].plot([p0[i,0],p1[i,0]+512],[p0[i,1],p1[i,1]],color=BLUE,lw=.7)
        axs[2].plot([p0[i,0],g[i,0]+512],[p0[i,1],g[i,1]],color=MAP,lw=.7,ls='--')
        axs[2].scatter(g[i,0]+512,g[i,1],color=MAP,marker='s',s=14,facecolors='none')
    axs[2].set_title('Solid blue = method; dashed magenta square = MAP PROXY, NOT GT; outside targets clipped by viewport')
    savefig(fig,folder/'matches.png',title)
    fig,axs=plt.subplots(1,2,figsize=(13,6.5),layout='constrained');fig.suptitle(title,fontsize=10)
    for side,ax in enumerate(axs):
        ax.imshow(images[side],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        pts=[p0,p1][side]
        ax.scatter(pts[~ins,0],pts[~ins,1],s=7,color=ORANGE,marker='x',label='RANSAC outlier')
        ax.scatter(pts[ins,0],pts[ins,1],s=9,facecolors='none',edgecolors=BLUE,label='RANSAC inlier')
        for v in [127.5,255.5,383.5]:ax.axvline(v,color='yellow',lw=.4);ax.axhline(v,color='yellow',lw=.4)
        ax.set_xlim(-.5,511.5);ax.set_ylim(511.5,-.5);ax.set_title(f'view{side}; ALL candidate points; 4x4 diagnostic cells');ax.legend(fontsize=8)
    savefig(fig,folder/'coverage.png','All candidate points; colors are model-fit labels, not correctness')

def zoomfigure(folder,r,c,images):
    p0,p1=c['points0'],c['points1_native'];n=len(p0)
    ids=subset(np.arange(n),4) if n else range(4)
    fig,axs=plt.subplots(4,2,figsize=(10,15),layout='constrained')
    for k in range(4):
        if n and k<len(ids):index=int(ids[k]);ps=[p0[index],p1[index]];label=f'candidate#{index}; inlier={bool(c["inlier_mask"][index])}'
        else:ps=[np.array([128+128*(k%2),128+128*(k//2)])]*2;label='no extra candidate; same-index display windows NOT correspondence'
        for side in [0,1]:
            p=ps[side];center=np.clip(np.rint(p),[48,48],[463,463]).astype(int);x,y=center-48
            ax=axs[k,side];ax.imshow(images[side][y:y+96,x:x+96],cmap='gray',vmin=0,vmax=255,interpolation='nearest',extent=(x-.5,x+95.5,y+95.5,y-.5))
            if n and k<len(ids):ax.scatter(p[0],p[1],marker='+',s=120,color=BLUE if c['inlier_mask'][index] else ORANGE)
            ax.set_title(f'view{side} native96px window | '+label,fontsize=8)
    fig.suptitle(f'{r["sample_id"]} {r["method"]} {r["protocol"]}\nAlgorithm-proposed native zooms; C=0; no static landmark certification',fontsize=11)
    savefig(fig,folder/'local_zoom.png','96 native-pixel algorithm-proposed windows; no added image detail or human GT')

def overlayfigure(folder,r,local,images,valid,mapper):
    mapxy=mapper(geo.pixel_grid((512,512))).reshape(512,512,2).astype(np.float32)
    mapwarp=cv2.remap(images[1],mapxy[:,:,0],mapxy[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    models=[('Global '+r['reason'],r['candidate_H'],None)]
    candidates=[t for t in local if t['output_available']]
    if candidates:
        tile=max(candidates,key=lambda t:(t['accepted'],t['n_inliers'],-t['gy'],-t['gx']))
        models.append((f'Local tile({tile["gx"]},{tile["gy"]}) {tile["reason"]}; NOT full-image model',tile['candidate_H'],tile))
    else:models.append(('No local Affine output',None,None))
    fig,axs=plt.subplots(2,3,figsize=(15,10.5),layout='constrained')
    for k,(name,H,tile) in enumerate(models):
        axs[k,0].imshow(images[0],cmap='gray',vmin=0,vmax=255,interpolation='nearest');axs[k,0].set_title('Reference native PNG')
        if H is not None:
            try:
                inv=np.linalg.inv(H);warped=cv2.warpPerspective(images[1],inv,(512,512),flags=cv2.INTER_LINEAR)
                mask=cv2.warpPerspective(valid[1].astype(np.uint8),inv,(512,512),flags=cv2.INTER_NEAREST)>0
                if tile:
                    ys,xs=np.indices((512,512));mask&=(xs//128==tile['gx'])&(ys//128==tile['gy'])
                overlay=images[0].astype(float)/255;overlay[mask]=.5*overlay[mask]+.5*warped[mask]/255
                axs[k,1].imshow(overlay,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                axs[k,1].set_title(name+'\n50% predicted moving / reference; bilinear display; valid support only',fontsize=9)
            except np.linalg.LinAlgError:H=None
        if H is None:axs[k,1].text(.5,.5,'No invertible model\nFailure retained',ha='center',va='center',transform=axs[k,1].transAxes);axs[k,1].set_title(name)
        axs[k,2].imshow(.5*images[0]+.5*mapwarp,cmap='gray',vmin=0,vmax=255,interpolation='nearest');axs[k,2].set_title('50% MAP PROXY / reference\ngeocoding overlay is NOT independent GT',fontsize=10)
        for ax in axs[k]:ax.set_xticks([0,128,256,384,511]);ax.set_yticks([0,128,256,384,511])
    fig.suptitle(f'{r["sample_id"]} {r["method"]} {r["protocol"]} | {r["look_pair_category"]}\n{r["n_matches"]} matches/{r["n_inliers"]} inliers; {r["decision"]}; C=0; candidate H retained even when rejected',fontsize=11)
    savefig(fig,folder/'geometry_overlay.png','Candidate Affine overlays, including rejected models; no accuracy certification')

manifest=[]
def main():
    if sys.argv[1:]==['--finish']:
        figs=ROOT/'visualizations';records=json.loads((ROOT/'analysis/records.json').read_text())
        for p in sorted(figs.rglob('*.png')):
            manifest.append(dict(file=str(p.relative_to(ROOT)),sha256=sha(p),
                caption='Retained completed diagnostic figure; method/model-fit/map labels are not independent GT',independent_GT=False))
        complete(records,figs);print('finished existing analysis; figures',len(manifest));return
    out=ROOT/'analysis';out.mkdir(exist_ok=False);figs=ROOT/'visualizations';figs.mkdir(exist_ok=False)
    records=[]
    for name in ['route_A','route_B','baselines']:
        for p in sorted((ROOT/name).glob('*/record.json')):records.append(json.loads(p.read_text()))
    rows={r['sample_id']:r for r in selected()};tables=[];local_tables=[];failure_tables=[]
    for r in records:
        sp=r['spatial'];tables.append(dict(sample_id=r['sample_id'],sensor=r['sensor'],component=r['component'],look=r['look_pair_category'],
            method=r['method'],protocol=r['protocol'],matches=r['n_matches'],inliers=r['n_inliers'],valid_matches=r['n_native_valid_matches'],
            inlier_ratio=r['inlier_ratio'],candidate_affine=r['output_available'],accepted=r['accepted'],decision=r['decision'],reason=r['reason'],
            matches_spatial_support=min(sp['matches0']['hull_fraction'],sp['matches1']['hull_fraction'])>=.01 and min(sp['matches0']['isotropy'],sp['matches1']['isotropy'])>=.005,
            inlier_hull0=sp['inliers0']['hull_fraction'],inlier_hull1=sp['inliers1']['hull_fraction'],
            local_outputs=r['local_output_tiles'],local_accepted=r['local_accepted_tiles'],geo_all_median=r['geo_all_median_px'],
            geo_inlier_median=r['geo_inlier_median_px'],geo_inlier_p90=r['geo_inlier_p90_px'],
            bidirectional_cycle=r['geometry'].get('bidirectional_cycle_mean_px'),symmetric_rmse_NOT_GT=r['geometry'].get('symmetric_transfer_rmse'),
            end_to_end_ms=r['end_to_end_ms'],GPU_allocated_MiB=r['timing'].get('GPU_allocated_MiB'),
            timing_origin='cached_old_process' if r['method'] in ['sift','orb','lightglue'] else 'new_actual_run',C=0))
        tables[-1].update(keypoints0=r['extra'].get('original_n_keypoints0',r['extra'].get('n_candidates0')),
            keypoints1=r['extra'].get('original_n_keypoints1',r['extra'].get('n_candidates1')),
            inlier_grid0=sp['inliers0']['grid_coverage'],inlier_grid1=sp['inliers1']['grid_coverage'],
            matches_hull0=sp['matches0']['hull_fraction'],matches_hull1=sp['matches1']['hull_fraction'],
            geo_all_p90=r['geo_all_p90_px'],fit_rmse_NOT_GT=r['fit_inlier_rmse_NOT_GT'])
        with np.load(r['cache_path']) as z:c={k:z[k] for k in z.files}
        local=json.loads((Path(r['cache_path']).parent/'local.json').read_text())
        for t in local:
            ids=np.asarray(t['inlier_indices'],int);err=c['geo_difference_px'][ids];err=err[np.isfinite(err)]
            local_tables.append(dict(sample_id=r['sample_id'],method=r['method'],protocol=r['protocol'],gx=t['gx'],gy=t['gy'],
                matches=t['n_matches'],inliers=t['n_inliers'],inlier_ratio=t['inlier_ratio'],candidate_affine=t['output_available'],
                accepted=t['accepted'],reason=t['reason'],scope=t['scope'],
                geo_inlier_median_NOT_GT=float(np.median(err)) if len(err) else None,
                bidirectional_cycle=t['diagnostics'].get('bidirectional_cycle_mean_px'),
                symmetric_rmse_NOT_GT=t['diagnostics'].get('symmetric_transfer_rmse'),
                candidate_H=t['candidate_H']))
        failure_tables.append(dict(sample_id=r['sample_id'],method=r['method'],protocol=r['protocol'],reason=r['reason'],
            decision=r['decision'],matching_stage_details=r['extra']))
        row=rows[r['sample_id']];raw,valid,images,infos,mapper=load_pair(row)
        # Deliver native LightGlue AND geo-assisted control, plus all new outputs.
        if r['method'] not in ['A1','A2','B1','B2','lightglue']:continue
        folder=figs/(r['sample_id']+'_'+r['method']+'_'+r['protocol']);folder.mkdir()
        matchfigure(folder,r,c,images);zoomfigure(folder,r,c,images);overlayfigure(folder,r,local,images,valid,mapper)
    csvwrite(out/'per_sample.csv',tables);save(out/'records.json',records)
    csvwrite(out/'local_geometry.csv',local_tables);csvwrite(out/'failure_stages.csv',failure_tables)
    groups=defaultdict(list)
    for r in tables:
        groups[(r['method'],r['protocol'],'all')].append(r)
        groups[(r['method'],r['protocol'],'different' if r['look']=='different_look_verified' else 'same')].append(r)
    summary=[]
    for (method,protocol,look),rs in groups.items():
        summary.append(dict(method=method,protocol=protocol,look=look,pairs=len(rs),with_matches=sum(r['matches']>0 for r in rs),
            spatial_supported=sum(r['matches_spatial_support'] for r in rs),affine_outputs=sum(r['candidate_affine'] for r in rs),
            full_gate=sum(r['accepted'] for r in rs),local_supported_pairs=sum(r['local_accepted']>0 for r in rs),
            local_supported_tiles=sum(r['local_accepted'] for r in rs),matches_total=sum(r['matches'] for r in rs),inliers_total=sum(r['inliers'] for r in rs),
            no_output=sum(r['decision']=='no_output' for r in rs),rejected=sum(r['decision']=='rejected' for r in rs),
            median_ms=float(np.median([r['end_to_end_ms'] for r in rs])),max_GPU_MiB=max((r['GPU_allocated_MiB'] or 0 for r in rs)) or None))
    csvwrite(out/'summary.csv',summary);save(out/'summary.json',summary)
    # Time panels contain raw four-pair observations, never borrowed error bars.
    for sid in rows:
        rs=[r for r in tables if r['sample_id']==sid];fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
        labels=[r['method']+' '+r['protocol'] for r in rs];times=[r['end_to_end_ms'] for r in rs]
        ax.bar(np.arange(len(rs)),times,color=[BLUE if r['timing_origin']=='new_actual_run' else '#999999' for r in rs]);ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(len(rs)),labels,rotation=40,ha='right',fontsize=8);ax.set_ylabel('milliseconds per pair (reported end-to-end)')
        ax.set_title(sid+' | blue=new actual; gray=historical cached timings\nDifferent processes/I-O protocols; not a controlled causal speedup comparison',fontsize=11)
        savefig(fig,figs/(sid+'_runtime.png'),'Actual new timing and historical cached times explicitly distinguished')
    complete(records,figs)
    print('analysis records',len(records),'figures',len(manifest),'summarygroups',len(summary))

def complete(records,figs):
    # Official batch=1 certainty keeps a leading singleton batch axis.
    # Remove that display axis only; preserve the raw inference cache unchanged.
    for p in (ROOT/'route_B').glob('*_dense/dense.npz'):
        with np.load(p) as c:cert=np.squeeze(c['certainty'])
        if cert.ndim!=2:raise ValueError('unexpected certainty display shape')
        fig,ax=plt.subplots(figsize=(12,6),layout='constrained');im=ax.imshow(cert,vmin=0,vmax=1,cmap='cividis',interpolation='nearest');fig.colorbar(im,ax=ax,label='RoMa raw model certainty (NOT calibrated correctness)')
        ax.set_title(p.parent.name+' | symmetric forward/reverse output; C=0');savefig(fig,figs/(p.parent.name+'_certainty.png'),'Uncalibrated model certainty, not true correct region')
    save(figs/'manifest.json',manifest)
    html=['<!doctype html><meta charset="utf-8"><title>SAR fixed pilot</title><style>body{font:16px sans-serif;max-width:1250px;margin:auto}img{max-width:100%}details{margin:1em}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:5px}</style>',
        '<h1>Fixed real SAR pilot: engineering candidates; C=0</h1><p>All inputs/predictions frozen. Colored lines, support, map overlays and certainty are NOT human ground truth. At most60 evenly indexed lines/category; coverage panels retain all matches. No independent Success metrics.</p>',
        '<p><a href="../analysis/per_sample.csv">Per-pair CSV</a> · <a href="../analysis/summary.csv">Grouped CSV</a></p>']
    for r in records:
        if r['method'] not in ['A1','A2','B1','B2','lightglue']:continue
        key=r['sample_id']+'_'+r['method']+'_'+r['protocol']
        html.append(f'<details><summary>{key}: matches{r["n_matches"]}, inliers{r["n_inliers"]}, {r["reason"]}, local full-gate tiles{r["local_accepted_tiles"]}</summary>')
        for kind in ['matches','coverage','local_zoom','geometry_overlay']:html.append(f'<p>{kind}</p><a href="{key}/{kind}.png"><img loading="lazy" src="{key}/{kind}.png" alt="{key} {kind}; algorithm candidate, no independent GT"></a>')
        html.append('</details>')
    (figs/'index.html').write_text('\n'.join(html))
if __name__=='__main__':main()
