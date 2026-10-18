"""Frozen candidate illustrations; colors distinguish GT evidence from unknown regions."""
from diagnose_frozen import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
def main():
 t=stamp();details=list(csv.DictReader(sorted(ROOT.glob('FROZEN_FAILURE_DETAILS_*.csv'))[-1].open()));focused=list(csv.DictReader(sorted(ROOT.glob('FOCUSED_FAILURES_WITH_DRIFT_*.csv'))[-1].open()));frozen=json.loads((REF/'final_scores_20261017_0908.json').read_text());look={(r['pair_id'],r['method'],r['protocol']):r for r in frozen};vr={r['pair_id']:r for r in [json.loads(s) for s in (BASE/'SyntheticSAR_Mixed640512_Multicondition_20261016_2114/manifests/val_CANONICAL_20261016_2147.jsonl').read_text().splitlines()]};figpaths=[]
 wrong=[r for r in focused if r['wrong_accept_case']=='True'];fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for i,r in enumerate(wrong):
  axs[0].scatter(float(r['candidate_inlier_rmse']),float(r['global_error']),marker='o',facecolors='none',edgecolors='#0072B2');axs[0].annotate(r['pair_id'][-5:],(float(r['candidate_inlier_rmse']),float(r['global_error'])),fontsize=7)
  axs[1].bar(i,int(r['ransac_inliers']),color='#BDBDBD',label='其余内点' if i==0 else None);axs[1].bar(i,int(r['synthetic_H_consistent_inliers']),color='#E69F00',label='符合合成H，静态证据另计' if i==0 else None);axs[1].bar(i,int(r['correct_static_inliers']),color='#009E73',hatch='//',label='静态GT一致内点' if i==0 else None)
 axs[0].set(xlabel='候选H的内点自拟合RMSE / px',ylabel='共同有效区GT check双向误差 / px',title='10个错误接受：局部拟合好 ≠ 全局变换正确');axs[0].axhline(5,color='black',linestyle='--',lw=.8);axs[0].grid(alpha=.15)
 axs[1].set_xticks(range(len(wrong)),[r['pair_id'][-5:] for r in wrong],rotation=45);axs[1].set(ylabel='内点数',title='内点的几何一致性与静态证据分别计数');axs[1].legend(fontsize=7)
 p=ROOT/f'figures/WRONG_ACCEPT_SUPPORT_{t}.png';fig.savefig(p,dpi=180);plt.close(fig);figpaths.append(str(p))
 for pid in ['MIX_val_00144','MIX_val_00381','MIX_val_00052','MIX_val_01027']:
  row=vr[pid];w=row['width'];ims=[cv2.imread(row[k],cv2.IMREAD_GRAYSCALE) for k in ['image_A','image_B']];canvas=np.concatenate(ims,1)
  with np.load(row['gt_path']) as z:gt={k:z[k] for k in z.files}
  fig,axs=plt.subplots(3,2,figsize=(11,10),layout='constrained')
  for j,m in enumerate(['b','t2','keep']):
   r=look[pid,m,'natural'];d=next(r for r in details if r['pair_id']==pid and r['method']==m and r['protocol']=='natural')
   with np.load(r['frozen_prediction_path']) as z:p={k:z[k] for k in z.files}
   a,b=p['matches_A'],p['matches_B'];geom,eligible,good,error=masks(gt,a,b);ax=axs[j,0];ax.imshow(canvas,cmap='gray',vmin=0,vmax=255);ax.axis('off')
   for mask,c,style in [(good,'#009E73','-'),(geom&~good,'#E69F00','-'),(~geom,'#999999','--')]:
    ids=np.flatnonzero(mask);ids=ids[np.linspace(0,len(ids)-1,min(35,len(ids)),dtype=int)] if len(ids) else ids
    for k in ids:ax.plot([a[k,0],b[k,0]+w],[a[k,1],b[k,1]],color=c,ls=style,lw=.5,alpha=.7)
   ax.set_title(f'{m}: 候选{len(a)} / H一致{geom.sum()} / 静态{good.sum()} / RANSAC {r["inliers"]}\n{r["state"]}:{r["reason"]} / check误差{r["bidirectional_error_px"]}',fontsize=7)
   ax=axs[j,1];ax.imshow(ims[0],cmap='gray',vmin=0,vmax=255);points=np.stack(np.meshgrid(np.linspace(0,w-1,6),np.linspace(0,w-1,6)),axis=-1)
   ax.clear();ax.imshow(ims[1],cmap='gray',vmin=0,vmax=255)
   for H,c,ls,name in [(gt['H_AB'],'#0072B2','-','合成GT网格'),(p['H_AB'],'#CC79A7','--','候选H网格')]:
    if not len(H):continue
    q=perspective(H,points.reshape(-1,2)).reshape(6,6,2)
    for u in range(6):
     ax.plot(q[u,:,0],q[u,:,1],color=c,ls=ls,lw=.5,label=name if u==0 else None);ax.plot(q[:,u,0],q[:,u,1],color=c,ls=ls,lw=.5)
   ax.set(xlim=(0,w-1),ylim=(w-1,0));ax.legend(fontsize=6);ax.axis('off');ax.set_title(f'相同reference网格映到moving；不用于预测\n内点凸包{float(d["inlier_hull_fraction"]):.3f}，合成一致凸包{float(d["synthetic_good_hull_fraction"]):.3f}',fontsize=7)
  fig.suptitle(f'{pid} / {row["source_scene"]} / Native {w}px\n绿实线=静态GT一致；橙实线=仅合成H一致、未获静态资格；灰虚线=不符3px合成H。每类至多35线。',fontsize=8)
  p=ROOT/f'figures/FROZEN_CASE_{pid}_{t}.png';fig.savefig(p,dpi=180);plt.close(fig);figpaths.append(str(p))
 write(ROOT/f'DIAGNOSIS_FIGURES_{t}.json',figpaths);print('\n'.join(figpaths))
if __name__=='__main__':main()
