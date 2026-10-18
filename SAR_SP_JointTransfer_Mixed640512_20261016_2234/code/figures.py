"""Static scientific figures with fixed scales and visible failures."""
from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})
COL=['#0072B2','#009E73','#E69F00','#CC79A7','#56B4E9','#D55E00','#999999','#332288']

def comparison(rr,tag):
 methods=list(dict.fromkeys(r['method'] for r in rr));fig,axs=plt.subplots(1,2,figsize=(12,4.3),layout='constrained')
 for ax,mode in zip(axs,['natural','equal']):
  for j,th in enumerate([3,5,10]):
   vals=[100*np.mean([r[f'Success@{th}px'] for r in rr if r['method']==m and r['protocol']==mode and r['geometry_eligible']]) for m in methods]
   ax.bar(np.arange(len(methods))+(j-1)*.24,vals,.24,label=f'@{th}px',color=COL[j])
  ax.set_xticks(range(len(methods)),[NAMES[m] for m in methods],rotation=30,ha='right',fontsize=8);ax.set_ylim(0,100);ax.set_ylabel('解析几何正例成功比例 (%)');ax.set_title(mode+' / 同一几何正例分母');ax.legend();ax.grid(axis='y',alpha=.2)
 p=ROOT/f'figures/geometry_comparison_{tag}_{stamp()}.png';fig.savefig(p,dpi=180);plt.close(fig);return p

def match_cases(rr,tag,methods=None,case_rows=None):
 rows=records('val');available={r['pair_id'] for r in rr};lookup={(r['pair_id'],r['method'],r['protocol']):r for r in rr};chosen=[]
 for scene in ['offshore_vessels_low_texture_sea','nearshore_vessels','port_quay_dense_vessels','other_land_structures']:
  pool=[r for r in rows if r['pair_id'] in available and r['source_scene']==scene and r['overlap_bin']!='zero']
  if pool:chosen.append(pool[0])
 if case_rows is not None:chosen=case_rows
 methods=methods or list(dict.fromkeys(r['method'] for r in rr));paths=[]
 for row in chosen:
  ims=images(row);w=row['width'];canvas=np.concatenate(ims,1);fig,axs=plt.subplots(len(methods),2,figsize=(13,3.1*len(methods)),squeeze=False,layout='constrained')
  for j,m in enumerate(methods):
   r=lookup[(row['pair_id'],m,'natural')];path=ROOT/'predictions'/tag/f"{row['pair_id']}_{m}_natural_{TAG}.npz"
   with np.load(path) as f:p={k:f[k] for k in ['matches_A','matches_B','keypoints_A','keypoints_B']}
   good=static_correct(row,p)
   for ax in axs[j]:ax.imshow(canvas,cmap='gray',vmin=0,vmax=255);ax.axis('off')
   ax=axs[j,0]
   for mask,color in [(good,'#009E73'),(~good,'#D55E00')]:
    ids=np.flatnonzero(mask);ids=ids[np.linspace(0,len(ids)-1,min(50,len(ids)),dtype=int)] if len(ids) else ids
    for k in ids:a=p['matches_A'][k];b=p['matches_B'][k]+[w,0];ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=.5,alpha=.75,linestyle='-' if mask is good else '--',marker='o' if mask is good else 'x',markersize=1.2)
   ax.set_title(f"{NAMES[m]} | 匹配{r['match_count']}，静态GT≤3px {r['correct_structure_matches']}\n{r['state']}: {r['reason']}; 全域S5={r['Success@5px']}",fontsize=9)
   ax=axs[j,1]
   for k,off in [('keypoints_A',0),('keypoints_B',w)]:
    q=p[k];ax.scatter(q[:,0]+off,q[:,1],s=4,facecolors='none',edgecolors='#0072B2',linewidths=.3)
   ax.set_title(f"实际检测覆盖 {r['keypoints0']}/{r['keypoints1']}；原生坐标，仅显示预测关键点",fontsize=9)
  fig.suptitle(f"{row['pair_id']} / {row['source_scene']}（来源语境）\n原生{w}像素，二维旋转{row['image_rotation_2D_deg']:.1f}°，重叠{row['overlap_ratio']:.2f}；绿色=解析静态GT一致，橙色=其余（含不可监督）\n每类最多显示50条线，所有匹配保存在NPZ；不是独立真实SAR精度",fontsize=10)
  p=ROOT/f"figures/matches_{tag}_{'_'.join(methods)}_{row['pair_id']}_{stamp()}.png";fig.savefig(p,dpi=150);plt.close(fig);paths.append(p)
 return paths
if __name__=='__main__':
 rr=read_scores('baseline');print(comparison(rr,'baseline'));print(match_cases(rr,'baseline'))
