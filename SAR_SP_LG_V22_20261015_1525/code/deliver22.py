"""Offline delivery audit and evidence report. Never changes models or predictions."""
from support22 import *
from collections import Counter, defaultdict
import math
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS=['m2_core','b_core_evidence','attr_old','attr_local','attr_roi','ship_off','ship_on']
GROUPS=['main_v1','main_v21','main_v22','main_v1_ssdd','main_v21_ssdd','main_v22_ssdd','attribution_v22','ship_v22','terminal_b_v22']
def latest(pattern):return max(ROOT.glob(pattern),key=lambda p:p.stat().st_mtime)
def average(rr,key):
 vals=[r[key] for r in rr if r.get(key) is not None]
 return float(np.mean(vals)) if vals else None
def percent(v):return 'NA' if v is None else f'{v*100:.2f}%'
def number(v):return 'NA' if v is None else f'{v:.2f}'
def table(head,rr):return '\n'.join(['|'+'|'.join(head)+'|','|'+'|'.join(['---']*len(head))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rr])+'\n'
def consumed(run):return [r for p in sorted((ROOT/'runs'/run).glob('sampling_from*.jsonl')) for r in rows(p)]
def online(run,indices):
 result={}
 for p in (ROOT/'runs'/run).glob('online_w*.jsonl'):
  for line in p.open():
   x=json.loads(line)
   if x.get('type')=='sample' and x['spec']['index'] in indices:
    k=x['spec']['index'];assert k not in result;result[k]=x
 assert result.keys()==indices
 return result
def main():
 torch.set_num_threads(2);tag=stamp()
 if (ROOT/f'delivery_audit_{tag}.json').exists():
  for revision in range(2,10000):
   candidate=f'v{revision}_{tag}'
   if not (ROOT/f'delivery_audit_{candidate}.json').exists():tag=candidate;break
 ds={g:json.loads(out(g).read_text()) for g in GROUPS};audit=dict(at=now(),groups={},runs={})
 locks={n:json.loads(out('selection_locked',folder=ROOT/'runs'/n).read_text()) for n in RUNS}
 training=[]
 for n in RUNS:
  rr=consumed(n);active=[x for x in rr if x.get('active')];mm=[x for p in (ROOT/'runs'/n).glob('metrics_from*.csv') for x in csv.DictReader(p.open())];lock=locks[n]
  assert len(mm)==lock['actual_steps'] and len(active)==8*len(mm)
  assert len({x['call_index'] for x in rr})==len(rr)
  assert all(math.isfinite(float(x['loss'])) and float(x['grad_norm'])>0 and math.isfinite(float(x['grad_norm'])) and int(x['active_pairs'])==8 for x in mm)
  assert all(x.get('ground_points_used',0)==0 or x['ground_points_used']>=4 for x in rr)
  assert lock['frozen_intact'] and lock['lightglue_frozen']
  ground=[x for x in active if x.get('ground_points_used',0)>=4]
  a=dict(steps=len(mm),calls=len(rr),active_pairs=len(active),skipped=len(rr)-len(active),zero_gradient_updates=0,all_effective_batch8=True,min_gradient_norm=min(float(x['grad_norm']) for x in mm),first100_loss=average([{k:float(v) for k,v in x.items()} for x in mm[:100]],'loss'),last100_loss=average([{k:float(v) for k,v in x.items()} for x in mm[-100:]],'loss'),active_roles=dict(Counter(x['actual_role'] for x in active)),sizes=dict(Counter(x['size'] for x in active)),descriptor_calls=len(ground),descriptor_sources=len(set(x['source_path'] for x in ground)),source_files=len(set(x['source_path'] for x in rr)),ship_points_backprop=sum(x.get('ship_points_used',0) for x in active),best_step=lock['best']['step'],elapsed_seconds=lock['elapsed_seconds'],peak_MiB=lock['peak_MiB'],best_sha=sha(ROOT/'runs'/n/'best.ckpt'),last_sha=sha(ROOT/'runs'/n/'last.ckpt'))
  assert a['best_sha']==lock['best_sha'] and a['last_sha']==lock['last_sha'];audit['runs'][n]=a;training.append(dict(run=n,**a))
 off,on=consumed('ship_off'),consumed('ship_on');keys=['call_index','step','source_path','component','relation_id','actual_role','active','size','ground_points_available','ground_points_used']
 assert len(off)==len(on) and all(all(a.get(k)==b.get(k) for k in keys) for a,b in zip(off,on))
 ids={x['call_index'] for x in off if not x.get('error')};aa,bb=online('ship_off',ids),online('ship_on',ids)
 identity_keys=['source_sha','spec','image_sha','mask_sha','ground_point_sha','H_AB']
 assert all(all(aa[k].get(f)==bb[k].get(f) for f in identity_keys) for k in ids)
 init=[json.loads(out('run_identity',folder=ROOT/'runs'/n).read_text())['init_sha'] for n in ['ship_off','ship_on']];assert init[0]==init[1]
 audit['ship_pairing']=dict(consumed_calls=len(off),usable_calls=len(ids),same_consumed_fields=keys,same_pixel_identity_fields=identity_keys,passed=True,same_initial_sha=init[0],effective_pairs_each=8000,terminal_steps_each=1000)
 initial={k:Model(k).state_dict() for k in ['m2','b']}
 model_states={}
 for name,kind,path in [('M2_V22','m2',ROOT/'runs/m2_core/best.ckpt'),('B_selected_fallback','b',ROOT/'runs/b_core_evidence/best.ckpt'),('B_trained6000','b',ROOT/'runs/b_core_evidence/last.ckpt')]:
  ck=torch.load(path,map_location='cpu',weights_only=False);state=ck['model_state'];changed=[k for k,v in state.items() if not torch.equal(v,initial[kind][k])]
  model_states[name]=dict(path=str(path),sha=sha(path),changed_parameters=changed,model_state_identical_to_initial=not changed,checkpoint_keys=sorted(ck))
 assert model_states['B_selected_fallback']['model_state_identical_to_initial']
 assert model_states['B_trained6000']['changed_parameters'] and model_states['M2_V22']['changed_parameters']
 audit['checkpoint_states']=model_states
 for g,d in ds.items():
  if 'manifest' in d:
   assert sha(d['manifest'])==d['manifest_sha'];n=len(rows(d['manifest']))
  else:
   # Terminal-B diagnostics reuse the exact fixed V2.2 core set.
   assert g=='terminal_b_v22';n=len(rows(E['recommended_val_all']))
   assert {x['sample_id'] for x in d['rows']}=={x['sample_id'] for x in ds['main_v22']['rows']}
  methods=list(d['summaries']['natural']);assert len(d['rows'])==n*len(methods)*2
  assert len({(x['sample_id'],x['method'],x['protocol']) for x in d['rows']})==len(d['rows'])
  for mode in ['natural','equal']:
   rr=[x for x in d['rows'] if x['protocol']==mode]
   if mode=='equal':
    by=defaultdict(list)
    for x in rr:by[x['sample_id']].append((x['n_keypoints0'],x['n_keypoints1']))
    assert all(len(set(v))==1 for v in by.values())
  audit['groups'][g]=dict(samples=n,methods=methods,prediction_rows=len(d['rows']),all_pairs_and_failures_retained=True,common_actual_budget_verified=True)
 old=json.loads(out('input_identity').read_text());oldpaths=old.get('files',old)
 # Reuse the completed direct-input audit, then independently recheck every recorded path.
 before=json.loads(latest('completion_identity_*.json').read_text())['old_inputs_unchanged'];assert all(before.values())
 recorded={}
 def visit(x):
  if isinstance(x,dict):
   for k,v in x.items():
    if k.startswith('/') and isinstance(v,str) and len(v)==64:recorded[k]=v
    else:visit(v)
  elif isinstance(x,list):
   for v in x:visit(v)
 visit(old)
 audit['old_inputs_rechecked']={p:sha(p)==v for p,v in recorded.items()};assert all(audit['old_inputs_rechecked'].values()) and recorded
 # Hash only actual runtime training dependencies; report/evaluation additions are separate.
 deps=['support22.py','graded_loss.py','stream22.py','train22.py']
 depcheck={}
 for n in RUNS:
  ident=json.loads(out('run_identity',folder=ROOT/'runs'/n).read_text());depcheck[n]={p:sha(ROOT/'code'/p)==ident['code_sha'][str(ROOT/'code'/p)] for p in deps}
  assert all(depcheck[n].values())
 audit['training_dependencies_unchanged']=depcheck
 extra=[]
 for g in ['main_v1','main_v21','main_v22','ship_v22','terminal_b_v22']:
  for m in ds[g]['summaries']['natural']:
   for mode in ['natural','equal']:
    allr=[x for x in ds[g]['rows'] if x['method']==m and x['protocol']==mode]
    for scene in sorted(set(x['scene'] for x in allr)):
     a=[x for x in allr if x['scene']==scene];pos=[x for x in a if x['global_eligible']];ships=[x for x in a if x.get('ship_anchor_count',0)>=4]
     extra.append(dict(group=g,method=m,protocol=mode,scene=scene,n=len(a),global_n=len(pos),success5=average(pos,'success5'),correct_static=average(pos,'correct_static_matches'),detected_R1=average(pos,'detected_R1'),keypoints0=average(a,'n_keypoints0'),ship_n=len(ships),ship_H_consistent=average(ships,'synthetic_ship_correct_matches'),ship_oracle_R1=average(ships,'ship_oracle_R1')))
 csvwrite(ROOT/f'delivery_scene_metrics_{tag}.csv',extra);csvwrite(ROOT/f'delivery_training_metrics_{tag}.csv',training);write(ROOT/f'delivery_audit_{tag}.json',audit)
 timing=json.loads(latest('isolated_timing_*.json').read_text())
 figures=[]
 fig,axs=plt.subplots(1,2,figsize=(10,3.6),layout='constrained')
 for ax,mode in zip(axs,['natural','equal']):
  x=np.arange(3)
  for offset,m in [(-.18,'M2'),(.18,'M2_V22')]:ax.bar(x+offset,[ds['main_'+v]['summaries'][mode][m]['S5']*100 for v in ['v1','v21','v22']],.35,label=m)
  ax.set_xticks(x,['V1 n=691','V2.1 n=135','V2.2 n=220']);ax.set_ylabel('Global synthetic S@5 (%)');ax.set_title(mode+'; own version denominator');ax.set_ylim(0,100);ax.legend();ax.grid(axis='y',alpha=.2)
 p=ROOT/f'figures/domain_tradeoff_{tag}.png';fig.savefig(p,dpi=180);plt.close(fig);figures.append(p)
 # A deterministic cached V1 offshore regression, not new inference or model selection.
 oldr={x['sample_id']:x for x in ds['main_v1']['rows'] if x['method']=='M2' and x['protocol']=='natural'}
 newr={x['sample_id']:x for x in ds['main_v1']['rows'] if x['method']=='M2_V22' and x['protocol']=='natural'}
 regress=sorted(k for k in oldr if 'offshore' in oldr[k]['scene'] and oldr[k]['success5'] and not newr[k]['success5'])
 example=regress[0];row=next(x for x in rows(ds['main_v1']['manifest']) if x['sample_id']==example)
 from evaluate22 import load
 images=load(row,'v1')['images'];fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
 for ax,m in zip(axs,['M2','M2_V22']):
  pred=np.load(ROOT/f'predictions/main_v1/v1/{example}/{m}_natural_20261015_1525.npz');h,w=images[0].shape;ax.imshow(np.hstack(images),cmap='gray',vmin=0,vmax=255)
  for j in np.linspace(0,len(pred['points0'])-1,min(60,len(pred['points0'])),dtype=int):
   a,b=pred['points0'][j],pred['points1'][j]+[w,0];ax.plot([a[0],b[0]],[a[1],b[1]],color='#16867b' if pred['correct_static'][j] else '#d87925',alpha=.65,lw=.65)
  r=oldr[example] if m=='M2' else newr[example];ax.set_title(f"{example} {m}: matches={r['n_matches']}, static-GT-consistent={r['correct_static_matches']}, S5={r['success5']}",fontsize=10);ax.axis('off')
 p=ROOT/f'figures/V1_offshore_regression_{tag}.png';fig.savefig(p,dpi=180);plt.close(fig);figures.append(p)
 # Report uses data-derived tables throughout. The initial auto report remains untouched.
 text=['# SAR V2.2受控训练：最终结果、代价与失败验收',f'完成记录：{now()}。本轮已完成实际训练、消融与冻结权重评价；不再延长训练或搜索参数。',
 '## 1. 结论先行',
 '**V2.2确实产生可测的结构区合成收益，但未得到可全面替换历史M2的海事模型。** 新M2在V2.2充分结构220例上，Natural S@5由71.36%升至86.82%（+15.45个百分点）；严格等实际点数由71.36%升至85.00%（+13.64个百分点）。与此同时，旧V1的691个几何正例由68.89%降至48.05%（−20.84个百分点），退化集中在离岸和弱结构背景。该代价不能被新版本筛选后的高分抵消。',
 '**B联合训练未通过本轮验收。** 真正训练到6000步的last模型保留并单独评价；选模best停留step0，权重与原始B逐张量相同，只是失败时的回退，不是新训练成果。原始B本身亦未通过噪声密度门槛，不建议部署这个回退。',
 '数据归因显示，同预算旧筛选继续训练已经达到很强的收益，恢复局部监督或新增ROI没有超过该对照。舰船辅助带来少量同源合成船匹配增加，但误接受更高，没有形成可靠离岸改进证据。没有训练LightGlue，没有新结构模块，没有重跑历史900条关系，没有任何真实跨角精度结论。',
 '## 2. 数据、隔离与监督边界',
 '权威入口为`../SyntheticSAR_V22_20261015_1352/DATASET_ENTRYPOINTS_V22_20261015_1434.json`。只训练V2.2-R/core；固定Train6420关系中，充分结构1729、有限结构849、无结构3605、无重叠237。地面描述子可用2324关系=1729充分+595有限，涉及677个源发布文件；其余有限关系不足4锚点，不能生成描述标签。已知Train/Val父组件101/15不交叉，原源文件4175/705不交叉。父组件不等于已证实独立原始采集。',
 '在线配对会重算监督资格；源文件数与固定关系数分别统计，不能把多次增强计作新采集。离岸4297是全部R中的关系数量，地面描述正例为0。B在线生成可在新视野得到有效监督，实际地面描述来源692；这是重新验收后的在线来源，不是把固定677改写为692。',
 '监督规则：充分结构与有限结构分别按1和min(1,N/16)加权；双向对应CE温度0.1，排除正例附近8px负例，另加0.1冻结初始M2描述保持项。N<4完全不计算描述CE。B的检测正监督只来自保存的静态锚点；低纹理dustbin只用于整8×8单元都有依据的校准区，未知、无效和动态区排除。解析无重叠负对单独以0.1×余弦hinge处理。可选船辅助权重0.15，默认关闭，绝不并入静态地面GT。',
 '发现并修复：交付B桥接在1–3锚点时可能从支持区重采样出更多点计算描述CE；新实验统一用保存锚点。最初B pilot另有仅人工空白约束被计作有效图像配对的问题，在520步停止并保留，从历史B重新启动`b_core_evidence`。没有修改数据交付reader、历史标签、原权重或冻结评价门槛。',
 '## 3. 训练实现与实际覆盖',
 'M2仅训练convDa/convDb（360960参数）；B仅训练convPa/convPb及原结构分支（325409参数）。共享编码、B原描述子头、LightGlue冻结。两模型需要梯度的角色不同，因此具有相同有效batch和优化步预算，但不声称逐角色曝光或浮点运算量完全相等。',
 'AdamW，描述/检测学习率1e−5、结构分支1e−4、weight decay1e−4，cosine，microbatch1×累积8；FP32，未启用AMP，避免历史半精度溢出。seed202622，TF32关闭，确定性算法warn-only；CUDA采样反传仍可能有微小舍入差异，不声称跨设备逐字节确定。每500步保存、每1000步固定Val选模，各主/归因6000步，船辅助各1000步。所有任务使用当时空闲GPU0，共卡运行，未干扰其他七卡任务。',
 table(['任务','步/最佳步','消费/有效配对','源/描述源','跳过率','耗时min','峰值MiB'],[[n,f"{a['steps']}/{a['best_step']}",f"{a['calls']}/{a['active_pairs']}",f"{a['source_files']}/{a['descriptor_sources']}",percent(a['skipped']/a['calls']),number(a['elapsed_seconds']/60),number(a['peak_MiB'])] for n,a in audit['runs'].items()]),
 '跳过率不是空梯度优化率：无监督/拒绝调用留下记录并推进索引，但不占有效batch；全部正式优化步有有限正梯度范数，空梯度更新0。B结构分支5999次更新、检测头6000次，是一个合法无描述分支跳过，并非伪造更新。每次调用的角色、点数、损失与文件路径在sampling日志；不同尺寸256/384/512及实际计数见delivery_training_metrics。Train缓存中未被消费的预取不能计入覆盖。',
 '短程检查：M2与B各400步，M2包含200→400真实恢复。M2首50步loss1.0449→末50步0.8938；B总loss没有下降，校准项下降但匹配未改善，负面证据保留。修正后的正式B再次验证effective8。预检查覆盖N=0/1/3/4/8/16/128、冻结/梯度/累积/重载/LG；10个实际消费在线配对回放像素、mask、H与锚点SHA一致，解析坐标误差上界约1.56e−5px，不是物理精度。额外计算包括两次400步smoke、无效B pilot520步，未混入正式6000步对照。',
 '## 4. 选模与统一评价协议',
 '固定V2.2开发选模144项：64全幅、32有限、32无结构、16无重叠，训练前锁定。先过黑/灰零点、噪声不高于M2、已证实校准区密度与无重叠误接受门槛，再按等点数S@5、误接受和有限结构oracle R1选模。M2最佳4000；B所有训练候选未合格，best是step0失败回退。三组归因best4000/6000/4000；船辅助依预定协议比较两个终点last1000，不事后换成各自best0。',
 '预测仅读取图像。Native最大512，不强行放大；NMS4、响应阈值0.0005、Natural上限1024。Equal对每一对、每一幅取本组所有方法实际点数最小值，上限128，逐样本已验证完全相同；不同实验组Equal预算不同，不能跨组作直接差值。ORB使用Hamming双向ratio0.8；学习特征用冻结官方LightGlue。Homography RANSAC3px/5000次/0.995、最少8匹配及8内点/0.3比例、原full可靠性门控均不变。',
 'S@3/5/10是通过接受门槛且保存静态GT点上的平均双向重投影误差≤阈值；不是角点或全图每像素误差。V1沿用旧版静态几何定义，含大量海面同源纹理；V2.1与V2.2全幅仅统计充分结构，有限局部不当作全幅正例，无结构未知既不算失败也不自动算负例。错误接受为已接受但S@5失败/全部合格正例；无重叠FPR另用解析负例分母。所有无输出、拒绝及失败均保留。',
 '## 5. 三版本核心结果（各自分母）']
 for version in ['v1','v21','v22']:
  d=ds['main_'+version];text.append('### '+version)
  text.append(table(['预算','模型','正例/无重叠','S3','S5','S10','正例错接受','无重叠FPR'],[[mode,m,f"{s['global_n']}/{s['no_overlap_n']}",*[percent(s[k]) for k in ['S3','S5','S10','wrong_accept','no_overlap_FPR']]] for mode in ['natural','equal'] for m,s in d['summaries'][mode].items()]))
 text.extend(['B_V22列为原始B回退，重复行保留用以核对冻结选模策略，不能称为训练成功。SSDD扩展各版本单列在unified_comparison与逐对main_*_ssdd文件；其来源独立性未证实，不混入core分母。V2.1原common133保持原ID，另见V21_common133统计，不用新增2例改变旧口径。',
 f'![三版本收益与退化]({figures[0].relative_to(ROOT)})',
 '## 6. 关键瓶颈：结构区改善与海事遗忘同时发生',
 'M2冻结检测路径，点数、重复性与校准响应均不变。因此本轮结构收益来自描述子/冻结LG接入后的关联改善，而不是多输出点。V2.2 Natural静态GT一致匹配均值37.53→43.67，实际检测点描述R1 45.55%→60.59%；有限结构102个可描述局部的oracle R1 71.39%→92.24%。oracle是在保存GT坐标提特征，与端到端实际检测Recall不同。',
 table(['版本/场景','正例n','M2 S5','新M2 S5','M2正确匹配均值','新M2正确匹配均值'],[[v,s,0,0,0,0] for v,s in []])])
 scene_rows=[]
 for v in ['main_v1','main_v22']:
  for a in extra:
   if a['group']==v and a['protocol']=='natural' and a['method']=='M2' and a['global_n']:
    b=next(x for x in extra if x['group']==v and x['protocol']=='natural' and x['method']=='M2_V22' and x['scene']==a['scene']);scene_rows.append([v+'/'+a['scene'],a['global_n'],percent(a['success5']),percent(b['success5']),number(a['correct_static']),number(b['correct_static'])])
 text[-1]=table(['版本/场景','正例n','M2 S5','新M2 S5','M2正确匹配','新M2正确匹配'],scene_rows)
 text.extend(['V1离岸544例：358→235成功，正确静态GT一致匹配47.07→27.92；同一检测路径下描述匹配下降，支持“结构监督适配伴随旧海事弱结构能力遗忘”。V1静态定义仍可能奖励同源散斑，这种下降不能被等同为真实物理对应退化的直接证明，但它明确否定跨版本无代价提升。二维角度、重叠profile、低纹理与场景的完整分层见stratified_metrics；没有将二维旋转称为雷达观测角。',
 f'![缓存中的V1离岸退化实例]({figures[1].relative_to(ROOT)})',
 '图中绿色仅表示旧合成静态mask内解析H误差≤3px，橙色为其余匹配；不代表跨采集真实物理正确/错误。该例按样本ID排序选取第一个M2成功而新M2失败的离岸缓存，不参与选模。',
 'failure_stages按“可见静态检测不足→重复检测不足→正确匹配不足→空间支持→门控”顺序分类，是工程诊断而非互相独立的因果分解；candidate_error_and_decisions含有限候选误差均值/中位数、无输出/拒绝/接受数。未产生矩阵者不填虚假误差，不仅统计成功样本。known-component bootstrap附带探索性区间，不将同父裁片当独立采集，不作未见测试显著性宣称。',
 '## 7. B实际训练终点、噪声与拒绝代价',
 table(['方法/同池预算','S5','检测重复性','正确静态匹配','无重叠FPR','校准区均值点数'],[[m+'/'+mode,percent(s['S5']),percent(s['repeatability']),number(s['correct_static_matches']),percent(s['no_overlap_FPR']),number(s['calibration_keypoints0'])] for mode in ['natural','equal'] for m,s in ds['terminal_b_v22']['summaries'][mode].items()]),
 'B训练到6000步后，随机噪声点由1010降至678，仍远高于M2的126；黑/均匀灰均0。V2.2 Natural S5从原B71.82%降至70.45%，无重叠由0/57升到1/57；同池Equal下降至68.18%。减少密度不自动等于更好检测，重复性和正确匹配同时下降。原B、新终点与M2的随机噪声都不是真实纯海面GT；真实无结构海面只统计有依据的校准mask，不给未知海面造负标签。',
 '## 8. 数据归因：更多监督不是已证明的增益来源',
 table(['方法','Natural S5','同组Equal S5','有限局部oracle R1','Natural无重叠FPR'],[[m,percent(s['S5']),percent(ds['attribution_v22']['summaries']['equal'][m]['S5']),percent(s['local_R1']),percent(s['no_overlap_FPR'])] for m,s in ds['attribution_v22']['summaries']['natural'].items()]),
 '旧严格筛选1078关系/452文件；同图恢复局部监督1465关系/611文件；仅新增ROI857关系/580文件。相同M2初始化、解冻范围、6000步/effective8和选模，固定像素保留“同图”意义。恢复局部增加159个可监督文件，新增ROI与局部池还有来源交叠，故并非完全文件配对的因果实验。主实验是在线几何，不能与固定像素归因直接作单因素差。',
 '恢复局部与新ROI均显著优于未继续训练的M2，但没有超过旧筛选同预算对照。可确认额外SAR描述训练有效，不能确认V2.2的局部恢复或新增ROI本身优于旧选择。旧筛选Natural出现1/57误接受；不允许根据最终Val倒选归因模型替代已锁定主模型。',
 '## 9. optional_ship_aux有界消融',
 f'OFF/ON均从选模B回退（原始B）出发，各1000步、8000有效配对；{len(off)}消费调用和{len(ids)}可读取配对的来源、spec、双图像素、mask、H和静态锚点逐项一致，初始化SHA一致。该核对已通过，见delivery_audit。只有ON计算0.15船辅助，终点last1000固定，未以最终结果换权重。船辅助不能掩盖基础B未合格。',
 table(['版本','Natural S5','Natural FPR','Equal S5','Equal FPR','船H一致匹配均值','噪声点'],[[m,percent(s['S5']),percent(s['no_overlap_FPR']),percent(ds['ship_v22']['summaries']['equal'][m]['S5']),percent(ds['ship_v22']['summaries']['equal'][m]['no_overlap_FPR']),number(s['ship_correct']),ds['ship_v22']['probes'][m][2]['n_keypoints']] for m,s in ds['ship_v22']['summaries']['natural'].items()]),
 '在166个保存≥4合成船锚点的局部中，ON平均船H一致匹配比OFF约多0.87个；Natural静态S5反降1/220，无重叠1/57→2/57，Equal误接受1/57→3/57。未达到可靠性收益，不建议开启。下面按真实源场景标签单列离岸；这里的船对应只是同源二维变换辅助，不是跨时相静态船舶物理GT。',
 table(['离岸方法/预算','样本n','≥4船锚点n','船H一致均值','船oracle R1'],[[a['method']+'/'+a['protocol'],a['n'],a['ship_n'],number(a['ship_H_consistent']),percent(a['ship_oracle_R1'])] for a in extra if a['group']=='ship_v22' and 'offshore' in a['scene']]),
 '## 10. 时间、部署与推理入口',
 table(['模型','独立小样均值ms','中位ms','Torch峰值MiB'],[[m,number(s['mean_ms']),number(s['median_ms']),number(s['peak_MiB'])] for m,s in timing['summaries'].items()]),
 '上述为所有训练结束后GPU0空闲时，256/384/512各2对、每对2次、2次warmup，图像已加载，计入特征提取、匹配及几何门控，排除磁盘/模型加载；样本很少且LG自适应提前退出，不能把B与相同权重B回退的计时差称为算法改进，也不是完整数据集延迟估计。Torch内存不含CUDA上下文/其他进程；原逐对耗时含本项目并发竞争，另行保留。',
 '独立CLI仅打开两幅图与本地权重，不导入V2.2入口/reader，不读取GT或地图。Python open审计已实际运行，但不是系统调用追踪器；接口不使用GeoTransform、观测角、初始地图对应。输入需灰度可读、任一边≤512，不隐式上采样。',
 '```bash\ncd /home/yl/matching/sar_matching\nCUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n match python -B \\\n  SAR_SP_LG_V22_20261015_1525/code/predict_frozen22.py LEFT.png RIGHT.png \\\n  --kind m2 --gpu 0 \\\n  --checkpoint SAR_SP_LG_V22_20261015_1525/runs/m2_core/best.ckpt \\\n  --output /tmp/sar_v22_new_pair_run\n```',
 '实际部署依赖、官方SP/LG权重SHA见deployment_identity；本轮没有安装/卸载/升级依赖。初始权重与数据哈希见input_identity；每个run_identity记录配置、冻结范围、PID/物理GPU及源码SHA。全部正式运行的训练依赖在最终审计再次核对未变；新增报告/缓存分析与训练源码区分记录。',
 '## 11. 交付索引与复现',
 '- 主训练权重：`runs/m2_core/best.ckpt`（4000，新训练），`runs/m2_core/last.ckpt`（6000）；B真正训练权重`runs/b_core_evidence/last.ckpt`（6000），其`best.ckpt`是step0失败回退。三归因与船辅助best/last、分步恢复状态均保留。\n- 源码：`code/train22.py`、`graded_loss.py`、`stream22.py`；正式B有效batch入口`train_evidence22.py`，配对船消融入口`train_ship_evidence22.py`；未改历史桥接。\n- 数据/固定协议：`config_20261015_1540.json`、`protocol_locked_20261015_1540.json`、`SHIP_ABLATION_PROTOCOL_20261015_1631.json`；选择与完整Val清单/哈希可查。\n- 全部逐对：`main_v1/v21/v22_20261015_1525.csv`，SSDD单列`main_*_ssdd_20261015_1525.csv`；另有`attribution_v22`、`terminal_b_v22`与`ship_v22`；所有预测NPZ在predictions。\n- 汇总/分层：`unified_comparison_20261015_1525.csv`、`stratified_metrics_20261015_1525.csv`；本次delivery_*审计、component_comparisons、V21_common133及failure_stages使用实际生成时间戳。\n- 执行日志：logs目录；每run的sampling、metrics、source_coverage、selection_locked与status保留。checkpoint包含model/optimizer/scheduler/scaler/RNG/call游标/配置/身份，可恢复未完成任务；已完成锁禁止误重训。',
 '复现应指定全新run目录，先检查GPU资源；不要直接重跑旧调度器或覆盖结果：',
 '```bash\n# M2主训练；runs/m2_reproduction_new_run必须不存在，正式配置已锁定\nCUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n match python -B \\\n  SAR_SP_LG_V22_20261015_1525/code/train22.py \\\n  --kind m2 --gpu 0 --name m2_reproduction_new_run\n# 正式B使用同参数但入口train_evidence22.py及--kind b\n# 只对未完成run续训：在相同入口增加--resume，原配置/数据池不变\nbash SAR_SP_LG_V22_20261015_1525/check_training.sh\n```',
 '原run真实执行命令见run_experiments.sh、finish_experiments.sh及logs；独立评估入口final_evidence22.py以权重和manifest身份验证完整缓存，存在则跳过。不能用它改变数据池或阈值重新选模。',
 '## 12. 最终科研判断',
 '保留历史M2作为通用海事参照；新M2_V22是结构区描述适配候选，提供真实训练权重和明确局部收益，但因V1离岸退化，不作全面替代。B新终点与船辅助均未达到可靠性验收，不继续加步或增加模块。下一步若再训练，应先解决地面结构监督与海事弱结构泛化的平衡，并以固定跨版本开发验证约束遗忘；当前证据不支持直接微调LightGlue或宣称解决真实异向SAR。',
 '全部Val是既有来源中的开发验证，包含用于选模的子集；新版本结果不能称作未见确认测试。已知源组件不是原始采集独立性的完整证明。没有新增真实物理GT，没有真实Success@3/5/10，也未利用已经观察的18真实样本或900基准再挑权重。',
 '## 13. 完整训练曲线与结构样例',
 '![训练、固定开发验证、局部Recall与噪声响应](figures/training_curves_20261015_1525.png)',
 '![三版本六方法](figures/version_comparison_20261015_1525.png)'])
 for p in sorted((ROOT/'figures').glob('p*_20261015_1525.png')):text.append(f'![合成结构匹配示例 {p.stem}]({p.relative_to(ROOT)})')
 report=ROOT/f'V22_CONTROLLED_TRAINING_REPORT_{tag}.md';report.write_text('\n\n'.join(text)+'\n');print('FINAL_REPORT',report)
 write(ROOT/f'final_delivery_identity_{tag}.json',dict(at=now(),report=str(report),report_sha=sha(report),audit=str(ROOT/f'delivery_audit_{tag}.json'),code_sha={str(p):sha(p) for p in (ROOT/'code').glob('*.py')},outputs={g:sha(out(g)) for g in GROUPS},checkpoints=model_states,completed=True))
if __name__=='__main__':main()
