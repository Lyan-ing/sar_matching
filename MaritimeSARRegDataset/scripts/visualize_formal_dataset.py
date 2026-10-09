"""Representative fixed-stratum figures; 0..255 shared grayscale, nearest display."""
import argparse
import json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sarreg.io import jsonl_read,save_json
from scripts.summarize_phase1 import read


def panel(ax,image,title):
    ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    ax.set_title(title,fontsize=9);ax.set_axis_off()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--stamp',required=True);a=p.parse_args()
    root=Path(a.root);dest=Path('reports')/f'previews_phase1_formal_{a.stamp}'
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir();manifest=jsonl_read(root/'manifest.jsonl');records=[]
    for stratum in sorted({r['stratum'] for r in manifest}):
        options=[r for r in manifest if r['stratum']==stratum];chosen=options[0]
        if chosen['dataset']=='ssdd':
            for r in options:
                f=root/r['folder'];v=read(f/'valid_0.png');s=read(f/'sea_0.png')
                if np.any((v>0)&(s==0)) and np.any(s):chosen=r;break
        f=root/chosen['folder'];meta=json.loads((f/'meta.json').read_text());n=meta['num_views'];sz=meta['output_size'];pair=meta['pairs'][0]
        fig,axes=plt.subplots(2,n+2,figsize=(3*(n+2),6.5),layout='constrained')
        for i in range(n):
            panel(axes[0,i],read(f/f'view_{i}.png'),f'View {i}: {sz} x {sz} px')
            for ship in meta['ship_annotations'][i]:
                x,y,w,h=ship['bbox_xywh'];axes[0,i].add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='#E69F00',linewidth=1))
        b=read(f/'view_1.png');H=np.asarray(pair['H_j_to_i']);warp=cv2.warpPerspective(b,H,(sz,sz))
        panel(axes[0,n],warp,'GT warp 1 -> 0 (linear resampling)')
        panel(axes[0,n+1],read(f/pair['overlap_mask_i']),f'Overlap in 0: {pair["overlap_fraction_i"]:.3f}')
        images=[read(f/'valid_0.png'),read(f/'ships_1.png'),read(f/'dynamic_1.png')]
        titles=['Valid 0: white = source pixels','Ships 1: white = retained instance','Dynamic 1: white = excluded from GT']
        if (f/'sea_0.png').exists():
            images.append(read(f/'sea_0.png'));titles.append('Sea 0: white = sea; consult valid 0')
        else:
            images.append(read(f/pair['overlap_mask_j']));titles.append('Overlap in 1: white = common region')
        if n==3:images.append(read(f/'valid_2.png'));titles.append('Valid 2 (all three pair GT saved)')
        for ax,im,title in zip(axes[1],images,titles):panel(ax,im,title)
        fig.suptitle(f'{stratum} | {chosen["sample_id"]} | {chosen["dataset"]} | 2-D synthetic proxy',fontsize=11)
        target=dest/f'{stratum}_{a.stamp}.png';fig.savefig(target,dpi=140,facecolor='white');plt.close(fig)
        records.append({'figure':str(target),'sample_id':chosen['sample_id'],'stratum':stratum,'caption':'Native generated intensities on fixed 0..255 grayscale. Nearest display; image scaling for presentation adds no information. Boxes overlay retained polygon-derived labels. GT warp uses linear interpolation. Missing sea masks are not inferred.'})
    # Pixel-supported cross-split crop evidence: deterministic examples from both sources.
    prov=json.loads((root/'provenance.json').read_text());audit=json.loads(Path(prov['audit']).read_text())
    accepted=[e for e in audit['accepted'] if len(set(e['input_splits']))>1 and 'H_b_to_a' in e]
    chosen=[accepted[0]]+[e for e in accepted if '/SSDD/' in e['a']][:3]
    fig,axes=plt.subplots(len(chosen),3,figsize=(12,3*len(chosen)),layout='constrained')
    for ax,e in zip(axes,chosen):
        im0,im1=read(e['a']),read(e['b']);h,w=im0.shape
        panel(ax[0],im0,Path(e['a']).name+' / '+e['input_splits'][0]);panel(ax[1],im1,Path(e['b']).name+' / '+e['input_splits'][1])
        warped=cv2.warpPerspective(im1,np.asarray(e['H_b_to_a']),(w,h));panel(ax[2],warped,f'B -> A: nonship NCC {e["raw_nonship_ncc"]:.3f}')
    fig.suptitle('Confirmed related crops: source images unchanged; aligned B uses linear resampling',fontsize=12)
    target=dest/f'source_relationship_evidence_{a.stamp}.png';fig.savefig(target,dpi=140,facecolor='white');plt.close(fig)
    records.append({'figure':str(target),'evidence':chosen,'caption':'Four deterministic pixel-confirmed cross-input-split related crops. Acceptance excludes annotated ship pixels and requires common textured background; pHash alone never establishes identity.'})
    # Every group is displayed; no smoothing, jitter or omitted zero-overlap values.
    profiles=['easy','medium','low','none','dynamic','three'];counts=Counter(r['profile'] for r in manifest)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    axes[0].bar(profiles,[counts[p] for p in profiles],color='0.4');axes[0].set_ylabel('Fixed sample groups');axes[0].set_ylim(bottom=0)
    for i,k in enumerate(profiles):axes[0].text(i,counts[k]+4,str(counts[k]),ha='center')
    for i,k in enumerate(profiles):
        yy=[r['min_overlap'] for r in manifest if r['profile']==k];axes[1].plot(np.full(len(yy),i),yy,'o',markersize=2,color='0.2',alpha=.4)
    axes[1].set_xticks(range(len(profiles)),profiles);axes[1].set_ylim(-.02,1);axes[1].set_ylabel('Minimum bidirectional overlap across group pairs')
    axes[1].set_title('All 860 groups; repeated sources are correlated')
    target=dest/f'stratification_{a.stamp}.png';fig.savefig(target,dpi=160,facecolor='white');plt.close(fig)
    records.append({'figure':str(target),'caption':'Counts are groups, not independent acquisitions. Each point is a group minimum across all bidirectional pair fractions; triple secondary-pair overlap is unconstrained. No jitter/smoothing; identical values overplot. Manifest and statistics CSV provide all values.'})
    save_json(dest/f'figure_manifest_{a.stamp}.json',records);print(f'{len(records)} figures in {dest}')


if __name__=='__main__':main()
