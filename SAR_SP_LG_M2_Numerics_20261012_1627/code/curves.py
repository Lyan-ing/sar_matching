"""Small non-GPU plot refresh at validation checkpoints."""
import csv,os
from pathlib import Path
def plot_metrics(run):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 p=Path(run)/'metrics.csv'
 if not p.exists():return
 rows=list(csv.DictReader(p.open()));fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 groups=[['train_loss','match_loss'],['train_recall1','validation_recall1','validation_recall5'],['validation_success3','validation_success5','validation_success10','validation_false_accept_none']]
 for ax,keys,title in zip(axes,groups,['Synthetic training loss','Known-H descriptor retrieval','Known-H LightGlue geometry']):
  for k in keys:
   valid=[r for r in rows if r.get(k)]
   if valid:ax.plot([int(r['step']) for r in valid],[float(r[k]) for r in valid],marker='.' if 'validation' in k else None,label=k)
  ax.set(xlabel='Optimizer step',title=title);ax.grid(alpha=.2)
  if ax.lines:ax.legend(fontsize=7)
  if ax!=axes[0]:ax.set_ylim(0,1.04)
 fig.suptitle('SAR adaptation: synthetic validation only; not real different-look accuracy')
 temp=Path(run)/'curves.tmp.png';fig.savefig(temp,dpi=160);plt.close(fig);os.replace(temp,Path(run)/'curves.png')
