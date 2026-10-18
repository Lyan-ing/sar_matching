"""Cached complementarity + fixed fusion evaluation; no model selection or inference."""
from fusion_io import *
from collections import Counter,defaultdict
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def latest(pattern):return max(ROOT.glob(pattern),key=lambda p:p.stat().st_mtime)
def parsed_csv(p):
 result=[]
 for r in csv.DictReader(Path(p).open()):
  q={}
  for k,v in r.items():
   if v=='':q[k]=None;continue
   if v in ['True','False']:q[k]=v=='True';continue
   try:q[k]=json.loads(v)
   except (ValueError,TypeError):q[k]=v
  result.append(q)
 return result
def avg(rr,key):
 a=[r[key] for r in rr if r.get(key) is not None];return float(np.mean(a)) if a else None
def pct(x):return 'NA' if x is None else f'{100*x:.2f}%'
def num(x):return 'NA' if x is None else f'{x:.2f}'
def tab(header,rs):return '\n'.join(['|'+'|'.join(header)+'|','|'+'|'.join(['---']*len(header))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs])+'\n'
def md(name,tag,text):
 p=ROOT/f'{name}_{tag}.md'
 with p.open('x') as f:f.write('\n\n'.join(text)+'\n')
 return p
def summary(rr):
 pos=[x for x in rr if x['global_eligible']];neg=[x for x in rr if x['no_overlap']];local=[x for x in rr if x['role']=='limited_structure'];cal=[x for x in rr if x['role']=='no_structural_positive'];st=Counter(x['state'] for x in rr)
 return dict(n=len(rr),global_n=len(pos),negative_n=len(neg),limited_n=len(local),unknown_calibration_n=len(cal),success3_n=sum(x['success3'] for x in pos),success5_n=sum(x['success5'] for x in pos),success10_n=sum(x['success10'] for x in pos),S3=avg(pos,'success3'),S5=avg(pos,'success5'),S10=avg(pos,'success10'),wrong_accept_n=sum(x['wrong_accept'] for x in pos),wrong_accept_rate=avg(pos,'wrong_accept'),no_overlap_fp_n=sum(x['no_overlap_false_accept'] for x in neg),no_overlap_FPR=avg(neg,'no_overlap_false_accept'),no_output_n=st['no_output'],rejected_n=st['rejected'],accepted_n=st['accepted'],rejection_rate=(st['rejected']+st['no_output'])/len(rr),mean_keypoints0=avg(rr,'n_keypoints0'),mean_keypoints1=avg(rr,'n_keypoints1'),mean_matches=avg(rr,'n_matches'),correct_static_mean=avg(pos,'correct_static_matches'),correct_grid_mean=avg(pos,'correct_grid'),correct_hull_mean=avg(pos,'correct_hull'),orb_retained_correct_mean=avg(pos,'orb_retained_correct'),orb_retained_unique_correct_mean=avg(pos,'orb_retained_unique_correct'),oracle_new_orb_correct_pool_mean=avg(pos,'orb_unique_correct_pool'),elapsed_ms=avg(rr,'elapsed_ms'),median_elapsed_ms=float(np.median([x['elapsed_ms'] for x in rr])),torch_peak_MiB=(max(x['torch_peak_MiB'] for x in rr if x.get('torch_peak_MiB') is not None) if any(x.get('torch_peak_MiB') is not None for x in rr) else None),routes=dict(Counter(x.get('route','cached_single') for x in rr)))
def main():
 tag=stamp();comp=parsed_csv(latest('COMPLEMENTARITY_SUMMARY_*.csv'));pair=parsed_csv(latest('COMPLEMENTARITY_PAIRWISE_*.csv'));new=[];datasets={};manifest={};old=[]
 for v in ['v1','v21','v22']:
  complete=list((ROOT/'predictions'/('fusion_'+v)).glob('COMPLETED_*.json'));assert complete,v
  d=read(OLD/f'main_{v}_20261015_1525.json');datasets[v]=d;manifest[v]={r['sample_id']:r for r in rows(d['manifest'])}
  pp=latest('fusion_'+v+'_PAIRWISE_*.csv');rr=parsed_csv(pp);assert len(rr)==len(manifest[v])*18;new.extend(rr)
  for r in d['rows']:
   if r['method'] not in ['M0','M2','B','M2_V22','ORB']:continue
   old.append(dict(r,sp_method=r['method'],split='single',variant=r['method'],computed='old_cache_reused',prediction_file=str(cache(v,r['sample_id'],r['method'],r['protocol']))))
 branch_meta={(x['version'],x['sample_id'],x['sp_method'],x['split']):x['fusion_meta'] for x in new if x['method']=='F3'}
 state_corrections=[]
 for r in new:
  r['variant']=r['sp_method']+'_'+r['split']+'_'+r['method'];r['selected_branch_state']=r['state'];r['selected_branch_reason']=r['reason']
  if r['method'] in ['F2','F3']:
   attempts=branch_meta[(r['version'],r['sample_id'],r['sp_method'],r['split'])].get('attempts',[]);r['any_seed_candidate']=any(x.get('reason')!='no_seed_H' for x in attempts)
   if r['state']=='no_output' and r['any_seed_candidate']:
    r['state']='rejected';r['reason']='all_seed_candidates_rejected';state_corrections.append({k:r[k] for k in ['version','sample_id','sp_method','split','method','selected_branch_state','state','selected_branch_reason','reason']})
 csvwrite(ROOT/f'DECISION_STATE_CLARIFICATION_{tag}.csv',state_corrections)
 combined=old+new;groups=defaultdict(list)
 for r in combined:groups[(r['version'],r['protocol'],r['variant'])].append(r)
 sums=[dict(version=v,protocol=mode,variant=variant,**summary(rr)) for (v,mode,variant),rr in groups.items()]
 csvwrite(ROOT/f'FUSION_COMPARISON_{tag}.csv',sums);csvwrite(ROOT/f'FUSION_PAIRWISE_{tag}.csv',new)
 strata=[]
 for (v,mode,variant),rr in groups.items():
  for dim in ['scene','profile','angle_bin','role']:
   for val in sorted(set(str(x[dim]) for x in rr)):
    a=[x for x in rr if str(x[dim])==val];strata.append(dict(version=v,protocol=mode,variant=variant,dimension=dim,stratum=val,**summary(a)))
 csvwrite(ROOT/f'STRATIFIED_FUSION_{tag}.csv',strata)
 transitions=[]
 for v in datasets:
  oldidx={(x['sample_id'],x['method']):x for x in datasets[v]['rows'] if x['protocol']=='equal'}
  for method in ['M2','B','M2_V22']:
   for split in ['64_64','96_32']:
    a=[x for x in new if x['version']==v and x['sp_method']==method and x['split']==split and x['method']=='F3'];idx2={x['sample_id']:x for x in new if x['version']==v and x['sp_method']==method and x['split']==split and x['method']=='F2'}
    for scene in ['ALL',*sorted(set(x['scene'] for x in a))]:
     rr=[x for x in a if x['global_eligible'] and (scene=='ALL' or x['scene']==scene)]
     if not rr:continue
     for baseline in ['SP_full_equal','F2_same_pool']:
      b=[oldidx[(x['sample_id'],method)] if baseline=='SP_full_equal' else idx2[x['sample_id']] for x in rr];restored=sum(x['success5'] and not y['success5'] for x,y in zip(rr,b));lost=sum(not x['success5'] and y['success5'] for x,y in zip(rr,b))
      transitions.append(dict(version=v,sp_method=method,split=split,scene=scene,baseline=baseline,n=len(rr),restored=restored,lost=lost,net=restored-lost,F3_S5=avg(rr,'success5'),baseline_S5=avg(b,'success5'),F3_correct_matches=avg(rr,'correct_static_matches'),baseline_correct_matches=avg(b,'correct_static_matches'),F3_correct_grid=avg(rr,'correct_grid'),baseline_correct_grid=avg(b,'correct_grid')))
 csvwrite(ROOT/f'ABLATION_RESULTS_{tag}.csv',transitions)
 stage_rows=[]
 for (v,mode,variant),rr in groups.items():
  counter=Counter()
  for x in rr:
   if not x['global_eligible']:continue
   if x['success5']:stage='success'
   elif x.get('visible_detected0',9999)<8:stage='fewer8_visible_static_detections'
   elif x.get('repeatable',9999)<8:stage='fewer8_repeated_static_detections'
   elif x['n_matches']<8:stage='fewer8_candidate_matches'
   elif x['correct_static_matches']<8:stage='fewer8_GT_consistent_matches'
   elif x['correct_hull']<.01 or x['correct_grid']<.0625:stage='correct_support_concentrated'
   elif not x['accepted']:stage='rejected_despite_enough_GT_matches'
   else:stage='incorrect_accept'
   counter[stage]+=1
  for stage,n in counter.items():stage_rows.append(dict(version=v,protocol=mode,variant=variant,stage=stage,n=n,positive_denominator=sum(x['global_eligible'] for x in rr)))
 csvwrite(ROOT/f'FAILURE_STAGES_{tag}.csv',stage_rows)
 # Honest complementarity panels: oracle union is explicitly marked unattainable.
 fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for ax,mode in zip(axs,['natural','equal']):
  rr=[x for x in comp if x['version']=='v1' and 'offshore' in x['scene'] and x['protocol']==mode];x=np.arange(len(rr));bottom=np.zeros(len(rr))
  for key,label,col,hatch in [('both_success','Both S5','#0072B2',''),('sp_only','SP only','#009E73','//'),('orb_only','ORB only','#E69F00','xx'),('both_fail','Both fail','#aaa','..')]:
   heights=[r[key]/r['n']*100 for r in rr];ax.bar(x,heights,bottom=bottom,label=label,color=col,hatch=hatch);bottom+=heights
  ax.set_xticks(x,[r['method'] for r in rr]);ax.set_ylim(0,100);ax.set_title(mode+' / V1 offshore n=544');ax.set_ylabel('Pairs (%)');ax.legend(fontsize=7)
 p=ROOT/f'figures/complementarity_{tag}.png';fig.savefig(p,dpi=180);plt.close(fig);compfig=p
 fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
 for ax,v in zip(axs,['v1','v21','v22']):
  labels=['M2','B','ORB','M2+ORB F2','M2+ORB F3','B+ORB F2','B+ORB F3'];variants=['M2','B','ORB','M2_64_64_F2','M2_64_64_F3','B_64_64_F2','B_64_64_F3'];values=[]
  for m in variants:values.append(next(s['S5']*100 for s in sums if s['version']==v and s['variant']==m and s['protocol']==('equal' if m in ['M2','B','ORB'] else 'equal_total128')))
  ax.bar(range(len(values)),values,color=['#0072B2','#009E73','#E69F00','#56B4E9','#CC79A7','#56B4E9','#CC79A7']);ax.set_xticks(range(len(values)),labels,rotation=65,fontsize=7);ax.set_ylim(0,100);ax.set_title(v+' / own positive denominator');ax.set_ylabel('Synthetic S@5 (%)')
 p=ROOT/f'figures/fusion_comparison_{tag}.png';fig.savefig(p,dpi=180);plt.close(fig);fusionfig=p
 # Examples: deterministic first recovery, regression, wrong acceptance, plus offshore ORB-only.
 selected=[];reference={(x['version'],x['sample_id'],x['method']):x for x in old if x['protocol']=='equal'}
 for label,condition in [('recovery',lambda r:r['global_eligible'] and r['success5'] and not reference[(r['version'],r['sample_id'],r['sp_method'])]['success5']),('regression',lambda r:r['global_eligible'] and not r['success5'] and reference[(r['version'],r['sample_id'],r['sp_method'])]['success5']),('wrong_accept',lambda r:r['no_overlap'] and r['accepted'])]:
  rr=sorted([r for r in new if r['method']=='F3' and r['split']=='64_64' and r['sp_method'] in ['M2','B'] and condition(r)],key=lambda r:(r['version'],r['sample_id'],r['sp_method']))
  if rr:selected.append((label,rr[0]))
 rr=sorted([r for r in new if r['version']=='v1' and 'offshore' in r['scene'] and r['method']=='F3' and r['split']=='64_64' and r['sp_method']=='M2' and reference[('v1',r['sample_id'],'ORB')]['success5'] and not reference[('v1',r['sample_id'],'M2')]['success5']],key=lambda r:r['sample_id'])
 if rr:selected.append(('offshore_orb_only',rr[0]))
 casefigs=[];case_records=[]
 for label,r in selected:
  v,sid,m=r['version'],r['sample_id'],r['sp_method'];row=manifest[v][sid];ims=[cv2.imread(row.get(f'image{i}',row.get(f'view{i}')),0) for i in [0,1]];h,w=ims[0].shape;fig,axs=plt.subplots(2,2,figsize=(12,10),layout='constrained')
  f2=next(x for x in new if x['version']==v and x['sample_id']==sid and x['sp_method']==m and x['split']=='64_64' and x['method']=='F2');items=[('SP full equal',reference[(v,sid,m)]),('ORB full equal',reference[(v,sid,'ORB')]),('F2 split64/64',f2),('F3 split64/64',r)]
  for ax,(title,x) in zip(axs.flat,items):
   z=np.load(x['prediction_file']);ax.imshow(np.hstack(ims),cmap='gray',vmin=0,vmax=255);n=len(z['points0']);ids=np.linspace(0,n-1,min(n,55),dtype=int)
   for j in ids:
    a=z['points0'][j];b=z['points1'][j]+[w,0];good=bool(z['correct_static'][j]);inlier=bool(z['inliers'][j]);ax.plot([a[0],b[0]],[a[1],b[1]],color='#009E73' if good else '#D55E00',ls='-' if inlier else ':',lw=.65,alpha=.7)
    if 'sources' in z and len(ids):
     for origin,marker in [(0,'o'),(1,'^')]:
      selected_ids=ids[z['sources'][ids]==origin]
      if len(selected_ids):ax.scatter(*z['points0'][selected_ids].T,s=10,marker=marker,facecolors='none',edgecolors='white',lw=.6)
    ax.set_title(f"{title}: K={x['n_keypoints0']}/{x['n_keypoints1']} M={x['n_matches']} S5={x['success5']}\n{x['reason']}",fontsize=8);ax.axis('off')
  fig.suptitle(f'{label}: {v}/{sid}; synthetic geometry only, no physical GT',fontsize=11);p=ROOT/f'figures/{label}_{tag}.png';fig.savefig(p,dpi=160);plt.close(fig);casefigs.append(p);case_records.append(dict(label=label,version=v,sample_id=sid,sp_method=m,F3_success5=r['success5'],F3_reason=r['reason'],figure=str(p)))
 write(ROOT/f'CASE_SELECTION_{tag}.json',case_records)
 # Point coverage and local zoom from one fixed offshore candidate; no automatic physical interpretation.
 if selected:
  label,r=selected[-1];v,sid,m=r['version'],r['sample_id'],r['sp_method'];row=manifest[v][sid];im=cv2.imread(row.get('image0',row.get('view0')),0);op=np.load(cache(v,sid,'ORB','equal'));sp=np.load(cache(v,sid,m,'equal'));unique=nearest(op['keypoints0'],sp['keypoints0'])>3;novel=op['keypoints0'][unique];center=novel[0] if len(novel) else np.array(im.shape[::-1])/2
  fig,axs=plt.subplots(1,2,figsize=(10,5),layout='constrained')
  for ax in axs:
   ax.imshow(im,cmap='gray',vmin=0,vmax=255);ax.scatter(*sp['keypoints0'].T,s=9,marker='o',facecolors='none',edgecolors='#0072B2',label='SP actual');ax.scatter(*novel.T,s=14,marker='^',facecolors='none',edgecolors='#E69F00',label='ORB >3px from SP');ax.legend(fontsize=7)
  axs[0].set_title(f'{v}/{sid}: actual equal-budget locations');axs[1].set_xlim(center[0]-64,center[0]+64);axs[1].set_ylim(center[1]+64,center[1]-64);axs[1].set_title('Native 128px local view; no physical GT')
  p=ROOT/f'figures/point_coverage_{tag}.png';fig.savefig(p,dpi=180);plt.close(fig);pointfig=p
 else:pointfig=None
 primary=[x for x in comp if x['version']=='v1' and 'offshore' in x['scene'] and x['method'] in ['M2','B']]
 comptext=['# ORB与SAR-SuperPoint互补性：缓存实证',f'生成于{tag}。仅复用上一轮冻结预测；没有重新提取本节单模型特征或重跑900基准。',
 '## 输入与公平性',
 '核查表INPUT_CHECK_TABLE列出M0、历史M2、历史B、新M2-V2.2的实际权重与三个版本数据入口。B取历史结构增强checkpoint，未把B_V22 step0回退重复计作新模型。原生图像、H方向、RANSAC3px/最少8匹配和8内点/0.3比例及full可靠性门控完全沿用旧接口。V1 691几何正例+36无重叠；V2.1 135充分结构+50无重叠；V2.2 220充分结构+57无重叠。完整缓存包括有限、无结构等共2647对，不丢弃失败。',
 'Natural为各方法最多1024点，实际数量可不同；Equal沿用旧全组每幅实际共同最小点数，上限128。成功同时要求冻结门控接受和原静态GT双向均值误差≤5px。缓存中M0与M2检测位置相同，但描述子不同。',
 '## 关键问题：544个V1离岸正例',
 tab(['预算','SP主体','共同成功','SP独有','ORB独有','均失败','Oracle并集'],[[x['protocol'],x['method'],x['both_success'],x['sp_only'],x['orb_only'],x['both_fail'],pct(x['oracle_union_S5'])] for x in primary]),
 'Oracle并集只是已知GT后挑选的上限，不是可执行预测。ORB的Natural独有成功大部分在等点数后消失，不能把它全部归因于传统描述子更适合SAR。',
 tab(['预算','主体','ORB独有GT一致匹配/对','双端均遗漏的ORB正确匹配/对','新增正确4×4格/对'],[[x['protocol'],x['method'],num(x['orb_distinct_correct_mean']),num(x['orb_correct_at_sp_omitted_both_mean']),num(x['orb_new_correct_cells_mean'])] for x in primary]),
 '位置近邻容差3px；匹配重复要求双端都在3px内。独有检测位置、独有匹配和独有成功是三个不同层次。细粒度字段均保存在COMPLEMENTARITY_PAIRWISE，包含两图点数、正确匹配、状态/失败原因、网格支持及来源组件。不同版本/场景完整交叉表见COMPLEMENTARITY_SUMMARY，不能混合分母。',
 f'![离岸成功集合]({compfig.relative_to(ROOT)})',
 '## 物理解释边界与准入决定',
 'V1静态掩膜排除已知舰船，但海面同源残余散斑可产生精确二维几何一致性；不能因此称为跨采集稳定地物。V2.1/2.2结构掩膜也只是冻结自动结构监督，不是人工物理GT。仅凭局部图像外观不能自动认定道路/岸线或散射中心稳定。样例保留原灰度与原生局部放大，不生成新标签。',
 '全量分析完成后STAGE2_DECISION记录了弱但非零的等点数互补性，因此只允许一个小型对应级融合可行性实验。它不足以直接支持ORB教师训练、复杂融合网络或真实物理正确性声明。融合规则在FUSION_PROTOCOL预先锁定，仅64/64与96/32两个配置。']
 if pointfig:comptext.append(f'![独有点空间覆盖与原生局部图]({pointfig.relative_to(ROOT)})')
 cp=md('COMPLEMENTARITY_REPORT',tag,comptext)
 methodtext=['# 固定异构对应融合方法与复现',f'实现与协议注册于{read(next(ROOT.glob("FUSION_PROTOCOL_*.json")))["registered_at"]}，在全量互补性分析之后、正式融合推理之前。',
 '## 固定流程',
 'SP检测/描述→冻结LightGlue得到A；ORB检测/二值描述→Hamming双向ratio0.8得到B。绝不拼接二值/浮点描述子。独立RANSAC假设→异源非重复支持检查→冲突处理→空间轮询/来源平衡→共同重估→原full门控→接受或拒绝。所有函数只接收图像数组、预测坐标和尺寸。',
 '### 总预算',
 '每幅T=min(128,三个冻结SP模型及ORB的实际可用点数)。64/64时SP=floor(T/2)，ORB=T-SP；96/32时SP=floor(3T/4)，ORB=T-SP；T不足128时同比缩小，不向另一来源补点。两者原始选中点数合计严格T，与缓存单模型Equal相同。位置重复不被假装成新特征，另统计重合率。Natural只用于缓存互补性；没有再跑大预算Natural融合。',
 '### 对照与F3',
 '- F0：ORB单独、F1：SP单独，直接读取旧严格Equal与Natural缓存。\n- F2：同一拆分总预算内独立拟合两种假设，仅在原门控接受者中按双图内点最小凸包、内点比例、残差依次选择，SP固定平局优先；不读GT。\n- Concat：同一拆分候选直接拼接，原RANSAC/full门控，作为朴素对照。\n- F3：从两个独立候选H出发；异源至少4个非双端重复支持、比例≥0.3且覆盖≥2格，两向投影误差均≤3px。几何一致池内空间重复保留一个，一对多冲突两端候选都删除。按4×4格轮询、每源截到两者较少数，不由匹配数量多的一方主导重估。重估后仍要求每源至少4个支持和原full门控。两个合格假设在双向5×5格平均投影差>3px则拒绝；否则按平衡支持/几何质量固定排序。没有合格融合假设时回退F2。',
 '新增条件是融合的额外拒绝条件，不放宽旧RANSAC/门控。F3与Concat比较涉及“异源验证+冲突处理+来源平衡”的整体，不声称单独证明其中每个组成项的因果增益；F3与F2使用相同特征与匹配池，才能判断新对应融合本身价值。两比例仅为预定开发消融，不能选更好的再称未见测试。',
 '## 实际执行、成本与接口',
 '旧单模型缓存全部复用。新推理只重提必要描述子，因为旧NPZ没有保存描述子/分数；所有重提检测坐标逐对、逐方法与旧Natural缓存完全相同，且总T与旧Equal逐对完全一致。两个拆分的LightGlue必须重新匹配，不能裁剪旧全池匹配冒充64点匹配。训练参数更新数=0。',
 'GPU0，match，batch1，TF32关闭，固定种子42，官方本地SP/LG权重。新运行分别记录all3SP+ORB共享提取、当前SP匹配、ORB匹配和各策略几何时间；为保持总预算定义可复现，每方案耗时保守计入三模型预算提取，不能当作仅一个SP网络的最优部署延迟。F3几何时间含F2假设但不含Concat对照，三策略没有互相重复计时。Torch峰值是三模型+LG驻留分配，不含CUDA上下文。',
 '历史M2 checkpoint含teacherDa/Db训练教师头，推理仅严格加载sp.*，与旧入口相同；B和新M2严格全推理状态加载。初次smoke的兼容异常和修复日志保留。未改旧权重。',
 '```bash\ncd /home/yl/matching/sar_matching\n# 实际完成的全量评价，已有输出不得覆盖/重复运行\nCUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n match python -B SAR_ORB_SP_Fusion_20261015_2349/code/run_fusion.py --gpu 0\n# 独立双图；两图任一边不超过512，输出目录必须是新的\nCUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n match python -B SAR_ORB_SP_Fusion_20261015_2349/code/predict_pair.py LEFT.png RIGHT.png --gpu 0 --model M2 --split 64_64 --method F3 --output /tmp/sar_fusion_new_pair\n# 独立预测几何检查，无训练\nconda run --no-capture-output -n match python -B SAR_ORB_SP_Fusion_20261015_2349/code/test_core.py\n```',
 'CLI为了复用完全同一实现会计算全部冻结变体，只保存指定输出；status明确记录全部变体总耗时，不把它伪装成单变体速度。Python文件打开审计禁止GT、地理元数据与数据入口；它不是操作系统级追踪。纯图像接口未使用mask、H_gt、场景或方位选择方法。']
 mp=md('FUSION_METHOD',tag,methodtext)
 # Full numerical report focused on primary methods, while all variants remain in tables.
 primary_transition=[x for x in transitions if x['version']=='v1' and 'offshore' in x['scene'] and x['split']=='64_64' and x['sp_method'] in ['M2','B'] and x['baseline']=='SP_full_equal']
 primary_verdict='；'.join(f"{x['sp_method']}：恢复{x['restored']}、损失{x['lost']}，净增{x['net']}/{x['n']}" for x in primary_transition)
 primary_improved=any(x['net']>0 for x in primary_transition)
 verdict=('主要离岸方向存在探索性净增，仍须检查相对F2的收益与误接受，不能由此认定可训练物理教师。' if primary_improved else '**主要离岸方向未通过融合可行性验收，停止这轮融合开发，不推荐把ORB永久并入推理或直接用其海面候选训练SP。**')
 comparison=['# 固定融合评价与最终研究判断',f'生成于{tag}。本轮实际新训练0；全量缓存复用及一次固定融合评价已完成。',verdict,primary_verdict,
 '## 统一结果与代价',
 'S@5分母只含各版本合格全幅正例，FPR分母仅原解析无重叠负例；未知/有限结构不混入。原生尺寸、几何门控及每图总实际预算相同。F0/F1的Equal直接复用旧结果。',
 f'报告阶段发现{len(state_corrections)}条多分支记录的原state只描述了最后选中分支：该分支无候选H，但另一分支已经产生H并被拒绝。新导出将整体state明确记为rejected，原值保留selected_branch_state/reason，逐条见DECISION_STATE_CLARIFICATION。接受决定、H、匹配、S@阈值、FPR、总拒绝率均未变化，没有重跑推理或修改原始record/NPZ。Concat只按自身估计状态分类。']
 for v in ['v1','v21','v22']:
  a=[x for x in sums if x['version']==v and ((x['protocol']=='equal' and x['variant'] in ['M0','M2','B','M2_V22','ORB']) or (x['protocol']=='equal_total128' and any(x['variant'].startswith(m+'_') for m in ['M2','B'])))];comparison.append('### '+v);comparison.append(tab(['方法','正/负n','S3','S5','S10','正例错接受n','负例误接受n','正确静态匹配/正例'],[[x['variant'],f"{x['global_n']}/{x['negative_n']}",pct(x['S3']),pct(x['S5']),pct(x['S10']),x['wrong_accept_n'],x['no_overlap_fp_n'],num(x['correct_static_mean'])] for x in a]))
 comparison.extend([f'![固定预算主结果]({fusionfig.relative_to(ROOT)})','## 恢复、损失与局部互补价值',tab(['版本/主体/配额','场景','对照','n','恢复','损失','净增','F3 S5'],[[x['version']+'/'+x['sp_method']+'/'+x['split'],x['scene'],x['baseline'],x['n'],x['restored'],x['lost'],x['net'],pct(x['F3_S5'])] for x in transitions if x['sp_method'] in ['M2','B'] and x['split']=='64_64' and (x['scene']=='ALL' or 'offshore' in x['scene'])]),'F3如果只是回退F2，不能称为对应级融合收益。routes与orb_retained_unique_correct、pool可用正确匹配分别保留，恢复和损失都统计，不能只展示净增。更多GT一致匹配也不保证更广支持。完整场景、二维角度/重叠profile/角色分层见STRATIFIED_FUSION；新M2-V2.2所有两配额结果也在完整表，不以其不利场景换模型。'])
 comparison.extend(['### 新增ORB正确对应的实际保留',tab(['版本/主体/配额','正例n','池内独有ORB正确/对','最终保留独有ORB正确/对','正确网格/对'],[[x['version']+'/'+x['variant'],x['global_n'],num(x['oracle_new_orb_correct_pool_mean']),num(x['orb_retained_unique_correct_mean']),num(x['correct_grid_mean'])] for x in sums if x['variant'] in ['M2_64_64_F3','B_64_64_F3','M2_96_32_F3','B_96_32_F3']]),'“池内正确”只在预测完成后用原GT计数，没有用于筛选候选。按先检测/重复点、再候选/正确匹配、再空间/门控的顺序失败分类见FAILURE_STAGES；其分类互斥但不是独立因果归因。'])
 pool_stats=[]
 for v in ['v1','v21','v22']:
  for m in ['M2','B']:
   a=[x for x in new if x['version']==v and x['sp_method']==m and x['split']=='64_64' and x['method']=='F3' and x['global_eligible']];original=[x for x in old if x['version']==v and x['method']==m and x['protocol']=='equal' and x['global_eligible']]
   pool_stats.append([v+'/'+m,len(a),num(avg(original,'n_matches')),num(avg(a,'sp_pool_matches')),num(avg(a,'orb_pool_matches')),pct(sum(x['orb_pool_matches']<8 for x in a)/len(a)),sum(x['route']=='cross_supported_balanced' for x in a),sum(x['route']=='F2_fallback' for x in a),sum(x['route']=='reject_conflicting_fused_hypotheses' for x in a)])
 comparison.extend(['### 预算代价与失败发生位置',tab(['版本/主体','正n','原SP匹配均值','拆分SP匹配','拆分ORB匹配','ORB不足8比例','融合支路n','回退n','冲突拒绝n'],pool_stats),'拆分后SP候选匹配减少、ORB池可能不足以独立形成假设；这与海事低纹理损失一致，但不是严格分离检测器与描述子影响的因果证明。F3的来源平衡、冲突拒绝和全幅一致条件还可能丢弃原先可接受的近似几何。该固定设计的失败不能推广为所有异构融合都不可能；本轮不据此放宽门槛或再调配额。'])
 comparison.append('## 实际时间（保守共享提取口径）')
 comparison.append(tab(['版本/模型/配额/策略','均值ms','中位ms','峰值MiB','平均左右总点'],[[x['version']+'/'+x['variant'],num(x['elapsed_ms']),num(x['median_elapsed_ms']),num(x['torch_peak_MiB']),num(x['mean_keypoints0'])+'/'+num(x['mean_keypoints1'])] for x in sums if x['version']=='v1' and x['variant'] in ['M2_64_64_F2','M2_64_64_F3','M2_64_64_Concat','B_64_64_F3']]))
 comparison.append('旧缓存耗时来自历史并发运行，与本轮新耗时不能直接算加速比。新方案计入共同提取三SP+ORB的实际成本；匹配与几何分项留在逐对表。不是独立部署优化后的吞吐，不能以此宣称融合几乎免费。')
 swapfiles=list(ROOT.glob('DETECTOR_SWAP_*.json'))
 if swapfiles:
  sd=read(max(swapfiles,key=lambda p:p.stat().st_mtime));ss=[]
  for group in sorted(set(x['group'] for x in sd['rows'])):
   for desc,matcher in [('ORB','Hamming_cached'),('M2','NN'),('M2','LG'),('B','NN'),('B','LG')]:
    rr=[x for x in sd['rows'] if x['group']==group and x['descriptor']==desc and x['matcher']==matcher];ss.append([group,desc+'/'+matcher,len(rr),num(avg(rr,'correct_static')),pct(avg(rr,'actual_detected_R1')),pct(avg(rr,'conditional_R1')),sum(x['success5'] for x in rr)])
  comparison.extend(['## 12对冻结检测位置交换诊断',tab(['组','描述子/匹配器','n','GT一致匹配均值','实际点R1','有对应条件R1','S5例数'],ss),'这些12例由旧缓存成功类别各排序取4例，先固定、后执行；不是随机测试集，不估计总体收益。在同一实际ORB点上换M2/B描述并用NN或LG，包含描述子/匹配器输入分布变化，不能把差异都归因于检测位置或当作严格单因素消融。没有使用理想GT坐标提特征。',
  '**遗漏检测位置并非唯一瓶颈。** 4个仅ORB成功的离岸样本，原二值描述/Hamming平均7.5个静态GT一致匹配、4/4成功；在同一实际ORB坐标上改用M2描述子+冻结LG后为1.75个、0/4；B为2.0个、0/4。在4个仅ORB成功的港口样本，10.0个降到M2/B各0.5个，均0/4。NN亦未恢复。此证据否定“直接补ORB位置即可解决”的简单假设，但不能区分ORB尺度/方向处理、二值描述和学习描述/匹配器的输入分布各自贡献，未执行这些因素的独立消融。'])
 comparison.extend(['## 科研边界与下一阶段决定','所有结果来自已观察的合成开发Val。同源二维H一致不等于真实跨采集物理对应；残余共享散斑特别影响V1离岸，不能声称已获得真实稳定特征。V2.2无结构类别保持未知，不自动成为无重叠负例。本轮没有新真实图像推理，没有真实Success或真实正确匹配率。真实Train定性演示为可选，未以它调整规则。','ORB确有少量独有的合成GT一致位置和成功，但Natural优势具有明显预算因素。下一步是否值得引入ORB引导必须同时看等点数恢复/损失、误接受以及检测位置交换结果；即使F3局部有效，也不意味着完成了鲁棒关键点网络研发。不能直接把海面散斑上的ORB点当物理教师。','本轮执行到两个预定配额即停止，不添加网络、门控学习或新比例，不延长搜索。没有训练或微调SP、B或LightGlue；不存在新的学习权重。'] )
 comparison.extend(['### 明确的研究选择',
 '**不启动ORB引导的SAR检测头训练，不把本轮F3替换历史M2。** ORB的少量独有几何正确位置值得保留作诊断来源，但本轮既未获得低纹理固定预算的整体融合收益，也未证明学习描述子能有效利用这些位置。结构丰富区域F2有收益，F3却不胜同池F2或ORB单独，收益不能归功于共同对应重估。',
 '继续以SAR-SuperPoint为研究主体时，优先验证“真实检测候选位置上的描述与匹配兼容性”，而不是再增加检测密度或盲目蒸馏ORB响应。若后续考虑ORB辅助，前提是先在可核验静态结构上证明其位置/尺度对应能被学习描述子利用，并有跨采集证据排除共享散斑；本轮12例交换尚未通过这一前提。该判断不要求换主架构，也没有在本轮启动新训练或设计新模块。'])
 for p in casefigs:comparison.append(f'![恢复/失败对照 {p.stem}]({p.relative_to(ROOT)})')
 final=md('FINAL_DECISION',tag,comparison)
 failure=['# 失败案例与停止边界',f'生成于{tag}。所有失败逐对保留，以下是固定排序选例，未用于修改算法。',tab(['类别','版本/样本','主体','F3 S5','原因'],[[x['label'],x['version']+'/'+x['sample_id'],x['sp_method'],str(x['F3_success5']),x['F3_reason']] for x in case_records]),'绿色线只表示原合成静态GT一致，橙色为其他；实线/虚线表示RANSAC内/外点，二者并不等同。海面细粒散斑不自动解释为固定设施。未知样本上的拒绝不能被称为配准失败率。']
 for p in casefigs:failure.append(f'![失败及恢复对照]({p.relative_to(ROOT)})')
 fp=md('FAILURE_CASES',tag,failure)
 files=[cp,mp,final,fp];freeze=read(next(ROOT.glob('CODE_FREEZE_*.json')));assert all(sha(p)==h for p,h in freeze['source_sha'].items());original=read(next(ROOT.glob('INPUT_AUDIT_*.json')));unchanged={p:sha(p)==h for p,h in original['sha256'].items()};assert all(unchanged.values())
 write(ROOT/f'REPORT_DELIVERY_{tag}.json',dict(at=tag,reports=[str(p) for p in files],cached_pairwise=len(pair),cached_single_rows=len(old),new_fusion_rows=len(new),all_failures_retained=True,unchanged_old_inputs=unchanged,unchanged_initial_runtime=True,training_steps=0,real_inference_pairs=0,summary=sums,transitions=transitions))
 print('REPORTS_COMPLETED',tag,[str(p) for p in files])
if __name__=='__main__':main()
