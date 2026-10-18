"""Add a genuine accepted-but-wrong case; never invent a no-overlap failure."""
from fusion_io import *
from report_fusion import parsed_csv,latest,tab,num
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def main():
 tag=stamp();rr=parsed_csv(latest('FUSION_PAIRWISE_*.csv'));bad=sorted([x for x in rr if x['method']=='F3' and x['split']=='64_64' and x['sp_method'] in ['M2','B'] and x['wrong_accept']],key=lambda x:(x['version'],x['sample_id'],x['sp_method']));assert bad
 r=bad[0];v,sid,m=r['version'],r['sample_id'],r['sp_method'];d=read(OLD/f'main_{v}_20261015_1525.json');row=next(x for x in rows(d['manifest']) if x['sample_id']==sid);ims=[cv2.imread(row.get(f'image{i}',row.get(f'view{i}')),0) for i in [0,1]];h,w=ims[0].shape
 singles=[dict(x,prediction_file=str(cache(v,sid,x['method'],'equal'))) for x in d['rows'] if x['sample_id']==sid and x['protocol']=='equal' and x['method'] in [m,'ORB']];f2=next(x for x in rr if x['version']==v and x['sample_id']==sid and x['sp_method']==m and x['split']=='64_64' and x['method']=='F2');items=singles+[f2,r];fig,axs=plt.subplots(2,2,figsize=(12,10),layout='constrained')
 for ax,x in zip(axs.flat,items):
  z=np.load(x['prediction_file']);ax.imshow(np.hstack(ims),cmap='gray',vmin=0,vmax=255);ids=np.linspace(0,len(z['points0'])-1,min(55,len(z['points0'])),dtype=int)
  for j in ids:
   p,q=z['points0'][j],z['points1'][j]+[w,0];ax.plot([p[0],q[0]],[p[1],q[1]],color='#009E73' if z['correct_static'][j] else '#D55E00',ls='-' if z['inliers'][j] else ':',lw=.7,alpha=.7)
  ax.set_title(f"{x['method']} accepted={x['accepted']}; inliers={x['n_inliers']}; GT-static={x['correct_static_matches']}\nSynthetic bidirectional mean error={x['local_bidir_error']:.2f}px",fontsize=9);ax.axis('off')
 fig.suptitle(f'{v}/{sid}: accepted geometry is NOT correct geometry; no physical GT',fontsize=11);p=ROOT/f'figures/positive_wrong_accept_{tag}.png';fig.savefig(p,dpi=180);plt.close(fig)
 source=latest('FAILURE_CASES_*.md');extra=['## 接受但错误的实际案例（补充）',f'补充于{tag}，按版本/样本ID固定排序选择主64/64方案的第一个正例错误接受。原报告和原始预测保留。',tab(['样本','主体','接受/理由','RANSAC内点','静态GT一致匹配','双向均值误差'],[[v+'/'+sid,m,str(r['accepted'])+'/'+r['reason'],r['n_inliers'],r['correct_static_matches'],num(r['local_bidir_error'])+'px']]),'该例通过了原full门控，但没有达到S@5；RANSAC自一致内点不能代替解析GT，更不能代替跨采集物理正确性。绿色/橙色只标原合成静态GT一致/其他，实线/虚线为自拟合内/外点。',f'![正例错误接受的实际匹配图]({p.relative_to(ROOT)})','无重叠样本的误接受数量见逐版本完整表；没有错误接受的组保持0，不为展示案例制造负例。']
 out=ROOT/f'FAILURE_CASES_{tag}.md'
 with out.open('x') as f:f.write(source.read_text()+'\n\n'+'\n\n'.join(extra)+'\n')
 write(ROOT/f'ERROR_CASE_EVIDENCE_{tag}.json',dict(report=str(out),figure=str(p),row=r,source_report=str(source),no_prediction_changed=True));print(out)
if __name__=='__main__':main()
