"""Small cached-result summaries and diagnostic plots; no network inference."""
import csv,json,hashlib,sys,collections
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parents[1];BASE=ROOT/'SAR_SP_LG_StructureStable_20261013_2210'
TS=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def writecsv(name,rows):
 with (OUT/'tables'/f'{name}_{TS}.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def records(path):
 with open(path) as f:return list(csv.DictReader(f))
def main():
 real=json.load(open(BASE/'real_equal_results_20261014_0028.json'))['rows'];val=records(BASE/'tables/validation_per_pair_20261013_2210.csv');summary=[];stages=[]
 for m in ['m0','m2','a','b']:
  for kind in ['same_look_control','other_observed_pairs']:
   r=[x for x in real if x['method']==m and (x['look_category']=='repeat_pass_same_look')==(kind=='same_look_control')]
   row=dict(protocol='strict_common_count_Native',method=m,group=kind,n=len(r),kp0_median=float(np.median([x['n_keypoints0'] for x in r])),matches_median=float(np.median([x['n_matches'] for x in r])),inliers_median=float(np.median([x['n_inliers'] for x in r])),with_8_matches=sum(x['n_matches']>=8 for x in r),with_8_inliers=sum(x['n_inliers']>=8 for x in r),accepted=sum(x['accepted'] for x in r),no_output=sum(x['decision']=='no_output' for x in r),independent_GT=False,real_accuracy='unavailable');summary.append(row)
 for m in ['m0','m2','a','b','control']:
  r=[x for x in val if x['method']==m and x['no_overlap']=='False'];fail=[x for x in r if x['success_5px']=='False']
  row=dict(method=m,n_positive=len(r),failed_at_5px=len(fail),failed_repeat0_lt8=sum(float(x['repeatable_detected0'])<8 for x in fail),failed_correctLG_lt8=sum(float(x['LG_correct_3px'])<8 for x in fail),failed_with_correct_support=sum(x['has_enough_correct_support']=='True' for x in fail),conditional_detected_R1=float(np.mean([float(x['descriptor_conditional_recall1']) for x in r if x['descriptor_conditional_recall1']])),actual_detected_e2e_R1=float(np.mean([float(x['descriptor_end_to_end_recall1']) for x in r if x['descriptor_end_to_end_recall1']])),LG_e2e_R=float(np.mean([float(x['LG_end_to_end_recall']) for x in r if x['LG_end_to_end_recall']])),mean_repeat0=float(np.mean([float(x['repeatable_detected0']) for x in r])),mean_correctLG=float(np.mean([float(x['LG_correct_3px']) for x in r])),failure_reasons=json.dumps(dict(collections.Counter(x['reason'] for x in fail))));stages.append(row)
 writecsv('real_failure_stages',summary);writecsv('synthetic_failure_stages',stages)
 m3=json.load(open(ROOT/'SAR_SP_LG_M3_Detector_20261012_1932/real_results.json'));writecsv('m3_historical_native', [{k:r.get(k) for k in ['id','method','n_keypoints0','n_keypoints1','n_matches','n_inliers','accepted','reason','look_category','source_component','elapsed_ms']} for r in m3])
 pilot=json.load(open(ROOT/'RealSAR_MatchingPilot_20261011_2114/analysis/records.json'));writecsv('historical_four_pair_pilot',[{k:r.get(k) for k in ['sample_id','method','protocol','n_matches','n_inliers','inlier_ratio','accepted','reason','local_accepted_tiles','look_pair_category','end_to_end_ms','cache_sha256']} for r in pilot])
 fig,axes=plt.subplots(1,2,figsize=(9,3.5))
 for ax,group in zip(axes,['other_observed_pairs','same_look_control']):
  r=[x for x in summary if x['group']==group];x=np.arange(len(r));w=.25
  for j,(field,label,color) in enumerate([('with_8_matches','>=8 matches','#0072B2'),('with_8_inliers','>=8 H support','#E69F00'),('accepted','full gate','#009E73')]):
   ax.bar(x+(j-1)*w,[z[field] for z in r],w,color=color,label=label)
  ax.set_xticks(x,[z['method'].upper() for z in r]);ax.set_ylim(0,r[0]['n']+1);ax.set_ylabel('Number of image pairs');ax.set_title(group.replace('_',' '));ax.legend(fontsize=7)
 fig.suptitle('Native cached pipeline counts: self-fit support is NOT physical correctness');fig.tight_layout();fig.savefig(OUT/'figures'/f'failure_stages_{TS}.png',dpi=220);plt.close(fig)
 g=records(sorted((OUT/'tables').glob('geometry_counterfactual_*.csv'))[-1]);fig,ax=plt.subplots(figsize=(9,3.6));xs=np.arange(6)
 ids=list(dict.fromkeys(x['id'] for x in g));labels=['Rotterdam 18.9','Rotterdam 46.0','Jebel Ali 36.4','Antwerp 88.4','SF same','Capella same']
 for model,marker,color in [('affine','o','#0072B2'),('homography','s','#D55E00')]:
  r=[next(z for z in g if z['id']==id and z['method']=='m2' and z['model']==model) for id in ids]
  ax.plot(xs,[float(z['held_median_NOT_GT']) for z in r],marker=marker,color=color,label=model)
 ax.set_yscale('log');ax.set_xticks(xs,labels,rotation=15);ax.set_ylabel('Held-match median fit deviation (native px)');ax.set_title('M2 cached matches: fit 60%, inspect ALL remaining 40% (seed42)');ax.legend();fig.text(.5,.01,'Not independent GT; one partition; local fit cannot establish physical correctness',ha='center',fontsize=8);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(OUT/'figures'/f'held_consistency_{TS}.png',dpi=220);plt.close(fig)
 refs={r['id']:r for r in json.load(open(BASE/'real_inputs_20261013_2210.json'))}
 for id in [ids[0],ids[4]]:
  fig,axes=plt.subplots(2,2,figsize=(10,9));ims=[]
  for path in refs[id]['paths']:
   with Image.open(path) as im:im.thumbnail((900,900));ims.append(np.asarray(im).copy())
  for ri,m in enumerate(['m2','b']):
   r=next(z for z in real if z['id']==id and z['method']==m);d=np.load(r['output_npz']);sh=r['input_shapes'];p0,p1=d['points0'],d['points1'];ins=d['inliers'].astype(bool)
   for side in [0,1]:
    h,w=sh[side];ax=axes[ri,side];ax.imshow(ims[side],cmap='gray',extent=[0,w,h,0]);pts=[p0,p1][side];ax.scatter(pts[:,0],pts[:,1],s=7,c=np.where(ins,'#009E73','#D55E00'));ax.set_xlim(0,w);ax.set_ylim(h,0);ax.set_aspect('equal');ax.set_title(f'{m.upper()} view{side}; {len(pts)} matches, {int(ins.sum())} self-fit support')
   # Display up to 60 deterministic, index-spaced matches. No confidence/GT cherry-pick.
   for i in np.linspace(0,len(p0)-1,min(60,len(p0)),dtype=int):
    line=ConnectionPatch(p0[i],p1[i],coordsA='data',coordsB='data',axesA=axes[ri,0],axesB=axes[ri,1],color='#009E73' if ins[i] else '#D55E00',lw=.45,alpha=.6);fig.add_artist(line)
  fig.suptitle(id+'\nFrozen common-count pools; lines are unverified proposals, NOT GT',fontsize=11);fig.tight_layout();fig.savefig(OUT/'figures'/f'match_lines_{id}_{TS}.png',dpi=180);plt.close(fig)
 with (OUT/f'supplement_summary_{TS}.json').open('x') as f:json.dump(dict(real=summary,synthetic=stages,seed=42,new_inference=0),f,indent=2)
 print('Completed cached summaries and four figures',TS)
if __name__=='__main__':main()
