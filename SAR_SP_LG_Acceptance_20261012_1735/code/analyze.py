"""Post-hoc tables and faithful prediction figures; never runs a matcher."""
import csv,json,html,statistics
from collections import Counter,defaultdict
from core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data import valid_at

def flat(r,prefix=''):
 out={}
 for k,v in r.items():
  k=prefix+k
  if isinstance(v,dict):out.update(flat(v,k+'.'))
  elif isinstance(v,list):out[k]=json.dumps(v,ensure_ascii=False)
  else:out[k]=v
 return out
def table(name,rows):
 rows=[flat(r) for r in rows]
 for r in rows:
  # Legacy evaluator label refers to warped GT probe points, not match precision.
  if 'gt_match_precision_3px' in r:r['gt_probe_fraction_within_3px']=r.pop('gt_match_precision_3px')
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (ROOT/'tables'/name).open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def avg(rows,k):
 vals=[r[k] for r in rows if r.get(k) is not None];return float(np.mean(vals)) if vals else None
def summary(rows):
 pos=[r for r in rows if not r['no_overlap']];neg=[r for r in rows if r['no_overlap']]
 errors=[r['epe_mean'] for r in rows if r.get('epe_mean') is not None]
 return dict(n=len(rows),positive=len(pos),none=len(neg),success3=sum(bool(r.get('success_3px')) for r in pos),success5=sum(bool(r.get('success_5px')) for r in pos),success10=sum(bool(r.get('success_10px')) for r in pos),S3=avg(pos,'success_3px'),S5=avg(pos,'success_5px'),S10=avg(pos,'success_10px'),accepted=sum(r['accepted'] for r in rows),accept_rate=avg(rows,'accepted'),rejected=sum(r['decision']=='rejected' for r in rows),no_output=sum(r['decision']=='no_output' for r in rows),incorrect_positive=sum(r['accepted'] and not r.get('success_5px') for r in pos),incorrect_positive_rate=float(sum(r['accepted'] and not r.get('success_5px') for r in pos)/len(pos)) if pos else None,none_false_accepts=sum(r['accepted'] for r in neg),none_FPR=avg(neg,'accepted'),error_pairs=len(errors),accepted_positive_epe_mean=float(np.mean(errors)) if errors else None,accepted_positive_epe_median=float(np.median(errors)) if errors else None,candidate_epe_mean=avg(rows,'candidate_epe_mean'),ms_mean=avg(rows,'elapsed_ms'),ms_median=float(np.median([r['elapsed_ms'] for r in rows])),kp0_mean=avg(rows,'n_keypoints0'),kp1_mean=avg(rows,'n_keypoints1'),matches_mean=avg(rows,'n_matches'),inliers_mean=avg(rows,'n_inliers'))

def matching_canvas(a,b,z,title):
 h=max(a.shape[0],b.shape[0]);w=a.shape[1]+b.shape[1]+20
 out=np.full((h+90,w,3),22,np.uint8)
 out[90:90+a.shape[0],:a.shape[1]]=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
 off=a.shape[1]+20;out[90:90+b.shape[0],off:off+b.shape[1]]=cv2.cvtColor(b,cv2.COLOR_GRAY2BGR)
 n=len(z['points0']);indices=np.linspace(0,n-1,min(100,n)).round().astype(int) if n else []
 for i in indices:
  p=tuple(np.rint(z['points0'][i]+[0,90]).astype(int));q=tuple(np.rint(z['points1'][i]+[off,90]).astype(int));color=(255,220,40) if z['inlier_mask'][i] else (40,110,250)
  cv2.line(out,p,q,color,1,cv2.LINE_AA);cv2.circle(out,p,3,color,1);cv2.circle(out,q,3,color,1)
 for j,line in enumerate([title,'Cyan=RANSAC inliers; orange=others. Not independent GT.',f'Display {len(indices)}/{n}; native pixel coordinates; no geographic prior.']):
  width=cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,.46,1)[0][0];font=min(.46,.46*(w-16)/max(width,1))
  cv2.putText(out,line,(8,23+27*j),cv2.FONT_HERSHEY_SIMPLEX,font,(255,255,255),1,cv2.LINE_AA)
 return out
def geometry_canvas(a,b,z,title):
 H=z['candidate_H'];wa=None
 if H.shape==(3,3):
  try:wa=cv2.warpPerspective(b,np.linalg.inv(H),(a.shape[1],a.shape[0]))
  except (cv2.error,np.linalg.LinAlgError):pass
 overlay=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
 if wa is not None:overlay=np.stack([wa,a,wa],axis=-1)
 support=cv2.cvtColor(a,cv2.COLOR_GRAY2BGR)
 for p in z['keypoints0']:cv2.circle(support,tuple(np.rint(p).astype(int)),1,(255,255,255),-1)
 points=z['points0'][z['inlier_mask']]
 if len(points)>=3:cv2.polylines(support,[cv2.convexHull(points.astype(np.float32)).astype(int)],True,(255,220,40),2)
 for x in np.linspace(0,a.shape[1]-1,5).astype(int):cv2.line(support,(x,0),(x,a.shape[0]-1),(120,100,80),1)
 for y in np.linspace(0,a.shape[0]-1,5).astype(int):cv2.line(support,(0,y),(a.shape[1]-1,y),(120,100,80),1)
 out=np.concatenate([support,overlay],axis=1);out=cv2.copyMakeBorder(out,80,0,0,0,cv2.BORDER_CONSTANT,value=(22,22,22))
 cv2.putText(out,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
 cv2.putText(out,'Left: detected points & consensus hull. Right: ref=green, predicted moving=magenta.',(8,47),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
 cv2.putText(out,'Candidate fit overlay only; rejected fits retained; no physical correctness inferred.',(8,69),cv2.FONT_HERSHEY_SIMPLEX,.44,(255,255,255),1,cv2.LINE_AA)
 return out
def local_canvas(a,b,z,title):
 n=len(z['points0'])
 if not n:return None
 i=int(np.argmin(np.linalg.norm(z['points0']-np.array([a.shape[1]/2,a.shape[0]/2]),axis=1)))
 patches=[];coords=[]
 for image,p in zip([a,b],[z['points0'][i],z['points1'][i]]):
  x=int(np.clip(round(float(p[0]))-64,0,max(image.shape[1]-128,0)));y=int(np.clip(round(float(p[1]))-64,0,max(image.shape[0]-128,0)))
  patch=cv2.cvtColor(image[y:y+128,x:x+128],cv2.COLOR_GRAY2BGR);cv2.circle(patch,tuple(np.rint(p-[x,y]).astype(int)),4,(255,220,40),1)
  patches.append(cv2.resize(patch,(384,384),interpolation=cv2.INTER_NEAREST));coords.append((x,y))
 out=np.concatenate(patches,axis=1);out=cv2.copyMakeBorder(out,75,0,0,0,cv2.BORDER_CONSTANT,value=(22,22,22))
 cv2.putText(out,title,(6,22),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
 cv2.putText(out,f'128px native windows x3 display; origin0={coords[0]} origin1={coords[1]}',(6,45),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
 cv2.putText(out,f'Center-nearest candidate #{i}, not verified GT; no selection by correctness.',(6,66),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
 return out

def combine(images,titles,path):
 fig,axes=plt.subplots(len(images),1,figsize=(14,4*len(images)),squeeze=False)
 for ax,im,title in zip(axes[:,0],images,titles):ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB));ax.set_title(title,fontsize=10);ax.axis('off')
 fig.tight_layout();fig.savefig(path,dpi=160);plt.close(fig)

def visualizations(formal,val,real):
 rr=json.loads((ROOT/'real_inputs.json').read_text());gallery=[]
 for ref in rr:
  ims=[cv2.imread(p,0) for p in ref['paths']];previews=[];titles=[]
  for r in [r for r in real if r['id']==ref['id']]:
   z=np.load(ROOT/'real_predictions'/r['id']/(r['method']+'.npz'),allow_pickle=False)
   title=f'{r["id"]} {r["method"].upper()} | {r["look"]} | {r["n_matches"]} matches / {r["n_inliers"]} inliers | {r["reason"]} | GT=0'
   prefix='real_'+r['id']+'_'+r['method'];match=matching_canvas(*ims,z,title)
   cv2.imwrite(str(ROOT/'figures'/(prefix+'_matches.png')),match)
   cv2.imwrite(str(ROOT/'figures'/(prefix+'_geometry.png')),geometry_canvas(*ims,z,title))
   local=local_canvas(*ims,z,title)
   if local is not None:cv2.imwrite(str(ROOT/'figures'/(prefix+'_local.png')),local)
   scale=min(1.,1500/match.shape[1]);previews.append(cv2.resize(match,(round(match.shape[1]*scale),round(match.shape[0]*scale)),interpolation=cv2.INTER_AREA));titles.append(r['method'].upper())
   gallery.append(dict(id=r['id'],method=r['method'],prefix=prefix,reason=r['reason']))
  combine(previews,titles,ROOT/'figures'/('real_'+ref['id']+'_comparison.png'))
 # Prespecified categories, lexicographic first example; visualization selection does not alter metrics.
 byid=defaultdict(dict)
 for r in formal:byid[r['id']][r['method']]=r
 categories={
 'all_models_success':lambda d:all(d[m]['success_5px'] for m in ['m0','m1','m2']),
 'm2_gain':lambda d:bool(d['m2']['success_5px'] and not d['m0']['success_5px']),
 'm2_loss':lambda d:bool(d['m0']['success_5px'] and not d['m2']['success_5px']),
 'low_overlap_failure':lambda d:0<d['m2']['overlap']<.3 and not d['m2']['success_5px'],
 'dynamic':lambda d:d['m2']['dynamic'],
 'no_overlap':lambda d:d['m2']['no_overlap']}
 selected=[]
 from sarbench.data import discover,load_images
 refs={r.id:r for r in discover(BASE/'MaritimeSARRegDataset/output/phase1_formal_20261009_1846')}
 for category,predicate in categories.items():
  ids=[i for i,d in sorted(byid.items()) if predicate(d)]
  if not ids:continue
  id=ids[0];ims=load_images(refs[id]);images=[];titles=[]
  for m in METHODS:
   r=byid[id][m];z=np.load(ROOT/'formal_predictions'/id.replace(':','__')/(m+'.npz'))
   title=f'{m.upper()} | {r["n_matches"]}/{r["n_inliers"]} matches/inliers | {r["decision_status"]} | EPE={r["epe_mean"]} | {r["reason"]}'
   images.append(matching_canvas(*ims,z,title));titles.append(title)
  filename='formal_'+category+'.png';combine(images,titles,ROOT/'figures'/filename);selected.append(dict(category=category,id=id,figure=filename))
 for id in ['val_010','val_023','val_025','val_026','val_037','val_043']:
  folder=ROOT/'validation_predictions'/id;ims=[cv2.imread(str(folder/f'image{i}.png'),0) for i in [0,1]];images=[];titles=[]
  for m in ['m0','m1','m2']:
   r=next(r for r in val if r['id']==id and r['method']==m);z=np.load(folder/(m+'.npz'))
   title=f'{m.upper()} | repeatable {r["repeatable_detected0"]}/{r["visible_detected0"]} | correct static LG {r["LG_correct_3px"]} | {r["reason"]}'
   images.append(matching_canvas(*ims,z,title));titles.append(title)
  combine(images,titles,ROOT/'figures'/('validation_'+id+'.png'))
 atomic_json(ROOT/'visualization_selection.json',dict(formal=selected,real=gallery,meaning='Consensus and fit overlays are not independent GT. All real methods retained, including rejected predictions.'))
 return selected,gallery

def main():
 formal=json.loads((ROOT/'formal_results.json').read_text());val=json.loads((ROOT/'validation_results.json').read_text());real=json.loads((ROOT/'real_results.json').read_text())
 # Align LG GT eligibility exactly with NN/ranking, without changing predictions,
 # acceptance, success, or original raw diagnostic records. Two boundary cases
 # per model are excluded because projected GT lands in ship-excluded pixels.
 corrections=[]
 for r in val:
  folder=ROOT/'validation_predictions'/r['id'];z=np.load(folder/(r['method']+'.npz'));gt=np.load(folder/(r['method']+'_GT_diagnostics.npz'))
  eligible=valid_at(gt['static0'],z['points0'])&valid_at(gt['static1'],perspective(gt['GT_H'],z['points0']))
  correct=gt['LG_correct_3px']&eligible;before=r['LG_correct_3px'];r['LG_correct_3px']=int(correct.sum())
  if before!=r['LG_correct_3px']:corrections.append(dict(id=r['id'],method=r['method'],raw_count=before,strict_count=r['LG_correct_3px'],reason='GT projected coordinate enters ship-excluded boundary; strict common static scope'))
  r['LG_correct_inliers_3px']=int((correct&z['inlier_mask']).sum());r['LG_precision_3px']=float(correct.mean()) if len(correct) else None
  r['LG_end_to_end_recall_3px']=float(correct.sum()/r['visible_detected0']) if r['visible_detected0'] else None
  r['LG_static_evaluable_matches']=int(eligible.sum());r['LG_static_evaluable_precision_3px']=float(correct.sum()/eligible.sum()) if eligible.any() else None
  sp=spatial(z['points0'][correct],(256,256));r.update(LG_correct_hull=sp['hull_fraction'],LG_correct_grid=sp['grid_coverage'],LG_correct_isotropy=sp['isotropy'])
 atomic_json(ROOT/'validation_evaluated.json',val);atomic_json(ROOT/'evaluation_scope_corrections.json',corrections)
 table('formal_per_pair.csv',formal);table('validation_module_per_pair.csv',val);table('real_native_per_pair.csv',real)
 totals=[dict(method=m,**summary([r for r in formal if r['method']==m])) for m in METHODS];table('formal_summary.csv',totals)
 strata=[]
 for dimension in ['scene','dataset','profile','dynamic','low_texture','overlap_band']:
  groups=defaultdict(list)
  for r in formal:
   label=r.get(dimension)
   if dimension=='overlap_band':label='none' if r['no_overlap'] else ('low:<0.3' if r['overlap']<.3 else ('medium:0.3-0.6' if r['overlap']<.6 else 'high:>=0.6'))
   groups[(r['method'],str(label))].append(r)
  for (method,label),rs in groups.items():strata.append(dict(dimension=dimension,group=label,method=method,**summary(rs)))
 table('formal_stratified.csv',strata)
 comps=[]
 for method in METHODS:
  groups=defaultdict(list)
  for r in formal:
   if r['method']==method:groups[r['source_component']].append(r)
  comps += [dict(method=method,source_component=g,**summary(rs)) for g,rs in groups.items()]
 table('formal_source_components.csv',comps)
 macro=[dict(method=m,n_components=len([r for r in comps if r['method']==m]),component_macro_S5=avg([r for r in comps if r['method']==m],'S5')) for m in METHODS];table('formal_component_macro.csv',macro)
 fail=[dict(method=m,reason=k,count=v) for m in METHODS for k,v in Counter(r['reason'] for r in formal if r['method']==m).items()];table('formal_failure_reasons.csv',fail)
 lookup={(r['id'],r['method']):r for r in formal};paired=[]
 for m in ['m1','m2']:
  a=[lookup[(r['id'],'m0')] for r in formal if r['method']==m and not r['no_overlap']];b=[lookup[(r['id'],m)] for r in formal if r['method']==m and not r['no_overlap']]
  paired.append(dict(method=m,positive=len(a),gain=sum(not x['success_5px'] and y['success_5px'] for x,y in zip(a,b)),loss=sum(x['success_5px'] and not y['success_5px'] for x,y in zip(a,b)),same_success=sum(x['success_5px'] and y['success_5px'] for x,y in zip(a,b)),same_failure=sum(not x['success_5px'] and not y['success_5px'] for x,y in zip(a,b))))
 table('formal_paired_changes.csv',paired)
 diagnostic_keys=['oracle_GT_location_recall1','oracle_GT_location_recall5','detector_repeatability_3px','detector_repeatability_reverse_3px','detected_descriptor_conditional_recall1','detected_descriptor_conditional_recall5','detected_descriptor_end_to_end_recall1','NN_matches','NN_correct_3px','NN_precision_3px','NN_end_to_end_recall_3px','LG_correct_3px','LG_precision_3px','LG_end_to_end_recall_3px','LG_correct_inliers_3px','LG_correct_hull','positive_cosine_detected','negative_cosine_detected']
 ds=[]
 for group,predicate in [('all_positive',lambda r:True),('low_overlap_regime',lambda r:r['regime']=='low'),('offshore',lambda r:r['scene']=='offshore'),('low_texture',lambda r:r['low_texture']),('dynamic',lambda r:r['dynamic']),('rejected_or_no_output',lambda r:not r['accepted'])]:
  for m in ['m0','m1','m2']:
   rs=[r for r in val if r['method']==m and not r['no_overlap'] and predicate(r)]
   ds.append(dict(group=group,method=m,n=len(rs),success5=sum(bool(r['success_5px']) for r in rs),**{k:avg(rs,k) for k in diagnostic_keys},recall_pairs=sum(r['detected_descriptor_conditional_recall1'] is not None for r in rs),repeatability_pairs=sum(r['detector_repeatability_3px'] is not None for r in rs),repeatable_fewer8=sum(r['repeatable_detected0']<8 for r in rs),correct_LG_fewer8=sum(r['LG_correct_3px']<8 for r in rs)))
 table('validation_module_summary.csv',ds)
 vc=[]
 for m in ['m0','m1','m2']:
  groups=defaultdict(list)
  for r in val:
   if r['method']==m:groups[r['source_component']].append(r)
  vc += [dict(method=m,component=g,n=len(rs),positive=sum(not r['no_overlap'] for r in rs),success5=avg([r for r in rs if not r['no_overlap']],'success_5px'),oracle=avg(rs,'oracle_GT_location_recall1'),detector_repeatability=avg(rs,'detector_repeatability_3px')) for g,rs in groups.items()]
 table('validation_source_components.csv',vc)
 real_summary=[dict(method=m,group=g,n=len(rs),accepted=sum(r['accepted'] for r in rs),matches_mean=avg(rs,'n_matches'),inliers_mean=avg(rs,'n_inliers'),mean_ms=avg(rs,'elapsed_ms')) for m in ['m0','m1','m2'] for g in ['different_look_verified','repeat_pass_same_look'] for rs in [[r for r in real if r['method']==m and r['look']==g]]];table('real_native_summary.csv',real_summary)
 atomic_json(ROOT/'analysis_summary.json',dict(formal=totals,strata=strata,components_macro=macro,paired=paired,validation=ds,real=real_summary,created=now()))
 # Compact statistical figures, fixed identical axes, all failures remain in denominators.
 fig,ax=plt.subplots(figsize=(10,5));x=np.arange(5)
 for i,(k,l) in enumerate([('S3','3 px'),('S5','5 px'),('S10','10 px')]):ax.bar(x+(i-1)*.24,[100*r[k] for r in totals],.24,label=l)
 ax.set_xticks(x,[m.upper() for m in METHODS]);ax.set_ylabel('Success / 860 positive pairs (%)');ax.set_ylim(0,100);ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'figures/formal_success.png',dpi=200);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for i,m in enumerate(['m0','m1','m2']):
  row=next(r for r in ds if r['group']=='all_positive' and r['method']==m)
  axes[0].bar(np.arange(4)+(i-1)*.24,[row[k]*100 for k in ['oracle_GT_location_recall1','detector_repeatability_3px','detected_descriptor_end_to_end_recall1','LG_end_to_end_recall_3px']],.24,label=m.upper())
  axes[1].bar(m.upper(),row['LG_correct_3px'])
 axes[0].set_xticks(range(4),['GT-position\noracle R@1','Detector\nrepeatability','Detected NN\nrank R@1','LightGlue\nend-to-end recall']);axes[0].set_ylim(0,100);axes[0].set_ylabel('Mean pair diagnostic (%)');axes[0].legend();axes[1].set_ylabel('Correct static synthetic matches / positive pair');fig.tight_layout();fig.savefig(ROOT/'figures/validation_modules.png',dpi=200);plt.close(fig)
 selected,gallery=visualizations(formal,val,real)
 print('ANALYSIS_COMPLETED',flush=True)
if __name__=='__main__':main()
