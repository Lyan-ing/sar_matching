"""Post-selection analysis. Never chooses weights or changes prediction acceptance."""
from common import *
from evaluate import summary
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from collections import Counter
def main():
 rr=json.loads(output('final').read_text())['rows'];ts=stamp();table=[];strata=[];methods=['m0','m2','b','orb','control','joint','frozen']
 for split in ['all','select','holdout']:
  for protocol in ['equal','natural']:
   for method in methods:
    rows=[r for r in rr if r['method']==method and r['protocol']==protocol and (split=='all' or r['split']==split)];s=summary(rows)
    if not s['n_negative']:s['no_overlap_FPR']=None
    pos=[r for r in rows if not r['no_overlap']];s.update(split=split,protocol=protocol,method=method,components=len({r['source_component'] for r in pos}),match_precision=float(np.mean([r['match_precision'] for r in pos if r['match_precision'] is not None])),correct_grid=float(np.mean([r['correct_grid'] for r in pos])),mean_matches=float(np.mean([r['n_matches'] for r in pos])),mean_inliers=float(np.mean([r['n_inliers'] for r in pos])),reject_rate=float(np.mean([not r['accepted'] for r in rows])));table.append(s)
 for field in ['category','regime','low_texture','dynamic','input_size']:
  for value in sorted({str(r[field]) for r in rr if not r['no_overlap']}):
   for protocol in ['equal','natural']:
    for method in methods:
     a=[r for r in rr if not r['no_overlap'] and str(r[field])==value and r['method']==method and r['protocol']==protocol];strata.append(dict(field=field,value=value,method=method,protocol=protocol,**summary(a)))
 paired=[];rng=np.random.default_rng(42)
 for protocol in ['equal','natural']:
  for split in ['all','holdout']:
   base={r['id']:r for r in rr if r['method']=='m2' and r['protocol']==protocol and not r['no_overlap'] and (split=='all' or r['split']==split)}
   for method in ['control','joint','frozen','b','orb']:
    a=[r for r in rr if r['method']==method and r['protocol']==protocol and r['id'] in base];cs=sorted({r['source_component'] for r in a});deltas=[np.mean([int(r['success5'])-int(base[r['id']]['success5']) for r in a if r['source_component']==c]) for c in cs];boot=np.mean(rng.choice(deltas,(2000,len(cs)),replace=True),axis=1)
    paired.append(dict(protocol=protocol,split=split,method=method,n=len(a),components=len(cs),gained=sum(r['success5'] and not base[r['id']]['success5'] for r in a),lost=sum(not r['success5'] and base[r['id']]['success5'] for r in a),component_mean_delta_pp=100*float(np.mean(deltas)),lo95=100*float(np.quantile(boot,.025)),hi95=100*float(np.quantile(boot,.975)),caveat='development component bootstrap, single training seed, raw SSDD acquisition identity not fully known'))
 export_csv(ROOT/f'comparison_{ts}.csv',table);export_csv(ROOT/f'strata_{ts}.csv',strata);export_csv(ROOT/f'paired_components_{ts}.csv',paired)
 # Effective checkpoint exposure versus physical replayed work are different quantities.
 training=[]
 for variant in ['joint','control','frozen']:
  run=ROOT/'runs'/variant;locked=json.loads(output('selection_locked',folder=run).read_text());state=json.loads(output('status',folder=run).read_text());all_metrics=[];sample_calls=0
  for f in sorted(run.glob('metrics_*.csv')):
   with f.open() as stream:all_metrics.extend(list(csv.DictReader(stream)))
  for f in run.glob('sampling_*.csv'):
   with f.open() as stream:sample_calls+=sum(1 for _ in csv.DictReader(stream))
  ck=torch.load(run/'best.ckpt',map_location='cpu',weights_only=False);base=torch.load(FIX/'runs/m2/best.ckpt',map_location='cpu',weights_only=False)['model_state'];changes={n:float((v-base[n]).abs().max()) for n,v in ck['model_state'].items()};expected=['sp.convP','sp.convD']+(['sp.conv4'] if variant!='frozen' else []);assert all(v==0 for n,v in changes.items() if not any(n.startswith(s) for s in expected));assert any(v>0 for v in changes.values())
  training.append(dict(variant=variant,best_step=locked['best']['step'],completed_steps=5000,optimizer_parameter_step_counts=sorted(set(locked['steps_actual'].values())),unique_sources_last=locked['unique_sources'],unique_sources_best=sum(x>0 for x in ck['sampling_counts']),logical_pair_calls_last=10000,physical_pair_calls=sample_calls,view_instances_physical=2*sample_calls,elapsed_seconds=state['elapsed_seconds'],peak_MiB=max(float(r['GPU_peak_MiB']) for r in all_metrics),best_sha=locked['best_sha'],hard_gate_pass=locked['best']['hard_gate_pass'],best_probes=locked['best']['probes'],changed_tensors={n:v for n,v in changes.items() if v>0},unselected_frozen_tensors_intact=True))
 export_csv(ROOT/f'training_coverage_{ts}.csv',training)
 figs=ROOT/'figures';figs.mkdir(exist_ok=True);fig,axs=plt.subplots(1,3,figsize=(12,3.5))
 for variant,color in [('joint','#0072B2'),('control','#999999'),('frozen','#D55E00')]:
  run=ROOT/'runs'/variant;ms=[]
  for f in run.glob('metrics_*.csv'):
   with f.open() as stream:ms.extend(csv.DictReader(stream))
  ms={int(r['step']):r for r in ms};xs=sorted(ms);axs[0].plot(xs,[float(ms[x]['loss']) for x in xs],label=variant,color=color,alpha=.8);axs[1].plot(xs,[float(ms[x]['descriptor']) for x in xs],color=color)
  vals=[]
  for f in run.glob('validation_step*.json'):
   d=json.loads(f.read_text());vals.append((d['step'],d['summary']['equal']['category_component_S5']))
  vals.sort();axs[2].plot([x[0] for x in vals],[x[1]*100 for x in vals],'-o',color=color,label=variant)
 axs[0].set_title('Training total loss (variable geometry)');axs[1].set_title('Actual-candidate descriptor loss');axs[2].set_title('Select category/component S@5');axs[0].legend()
 for ax in axs:ax.set_xlabel('optimizer step');ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(figs/f'learning_curves_{ts}.png',dpi=220);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(10,3.5))
 for ax,protocol in zip(axs,['equal','natural']):
  a=[next(r for r in table if r['split']=='holdout' and r['protocol']==protocol and r['method']==m) for m in methods];ax.bar(np.arange(7),[r['S5']*100 for r in a],color=['#999999','#0072B2','#CC79A7','#E69F00','#777777','#009E73','#D55E00']);ax.set_xticks(np.arange(7),[m.upper() for m in methods],rotation=20);ax.set_ylim(0,100);ax.set_title(protocol+' held components (111 positives)');ax.set_ylabel('Synthetic Success@5px (%)')
 fig.tight_layout();fig.savefig(figs/f'held_comparison_{ts}.png',dpi=220);plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,4));ax.set_axis_off();boxes=[(.02,.58,.2,.22,'SAR images <=512\nNo geographic inputs','#eeeeee'),(.27,.58,.2,.22,'Conv1-3 frozen\nConv4 trainable *','#cce5f2'),(.53,.58,.2,.22,'Detector + descriptor\nOriginal 256D interface','#d4eac8'),(.78,.58,.2,.22,'Frozen LightGlue\nRANSAC + full gate','#eeeeee'),(.27,.1,.45,.24,'Training only: GT-confirmed actual candidates\nQuality redistributes teacher cell mass; empty/noise negatives\nNMS/TopK targets detached; gradients via logits/descriptors','#ffe5bf')]
 for x,y,w,h,label,color in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor=color,edgecolor='#555555'));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
 for x in [.23,.48,.74]:ax.annotate('',(x+.035,.69),(x,.69),arrowprops=dict(arrowstyle='->'))
 ax.annotate('',(.62,.58),(.62,.36),arrowprops=dict(arrowstyle='->',linestyle='--'));ax.text(.5,.015,'* Conv4 frozen in the backbone ablation; no additional inference parameters',ha='center',fontsize=9);fig.tight_layout();fig.savefig(figs/f'architecture_{ts}.png',dpi=220);plt.close(fig)
 event('analysis',dict(comparison=table,paired=paired,training=training,strata=strata,figure_stamp=ts,no_new_training_or_inference=True))
 print(ts)
if __name__=='__main__':main()
