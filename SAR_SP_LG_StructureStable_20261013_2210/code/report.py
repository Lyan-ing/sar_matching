"""Figures/report from retained outputs. No training, inference or model selection."""
from ss_common import *
import html,re,fcntl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def latest(name,folder=None):
 ps=list((Path(folder) if folder else ROOT).glob(name+'_*.json'));assert ps,name
 return json.loads(max(ps,key=lambda p:p.stat().st_mtime_ns).read_text())
def table(cols,rr):return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(str(x) for x in r)+' |\n' for r in rr)
def pct(x):return '—' if x is None else f'{100*x:.2f}%'
def num(x):return '—' if x is None else f'{x:.2f}'
def main():
 stampnow=stamp();figdir=ROOT/'figures';figdir.mkdir(exist_ok=True)
 def save(fig,name):
  p=figdir/f'{name}_{stampnow}.png';assert not p.exists();fig.savefig(p,dpi=160);plt.close(fig);return p.relative_to(ROOT).as_posix()
 comp=latest('comparison');rr=latest('validation_results')['rows'];real=latest('real_results')['rows'];san=latest('sanity')['rows'];density=latest('density_budget')['rows'];native=latest('native_validation')['rows'];locks={v:json.loads(output('selection_locked',folder=ROOT/'runs'/v).read_text()) for v in ['a','b','control']}
 fig,ax=plt.subplots(figsize=(14,6),layout='constrained');ax.set(xlim=(0,14),ylim=(0,6));ax.axis('off')
 boxes=[(0.2,3.5,2,1,'SAR pixels\n256x256 / Native', '#DDDDDD'),(3,3.5,2.4,1,'SP shared encoder\n1/8 x128, FROZEN','#DDDDDD'),(6.2,3.5,2.7,1,'M2 convDa/convDb\n256-D, FROZEN A/B','#DDDDDD'),(10,3.5,3,1,'Original LightGlue\nFROZEN','#DDDDDD'),(3,.5,2.4,1.7,'log1p intensity\nGaussian 1/2/4\nScharr magnitude','#DDDDDD'),(6.2,.5,2.7,1.7,'Conv 3->16->32->256\nZero-init last 1x1\nA/B: TRAIN 13,536','#C9E7F2'),(10,.5,3,1.7,'convPa/convPb\nA: FROZEN\nB: TRAIN 311,873','#F5D8B6')]
 for x,y,w,h,t,c in boxes:ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.1',facecolor=c,edgecolor='black'));ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=10)
 for start,end in [((2.2,4),(3,4)),((5.4,4),(6.2,4)),((8.9,4),(10,4)),((1.2,3.5),(3,1.4)),((5.4,1.4),(6.2,1.4)),((7.5,2.2),(7.5,3.5)),((5.4,3.8),(10,2.2)),((11.5,2.2),(11.5,3.5))]:ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.4))
 ax.text(7.6,2.7,'residual +',fontsize=10);ax.set_title('Controlled SAR structure and stable feature learning | no backbone / LightGlue update')
 structure=save(fig,'network_structure')
 fig,axs=plt.subplots(1,3,figsize=(15,4),layout='constrained')
 for v in ['a','b','control']:
  series=[]
  for file in (ROOT/'runs'/v).glob('metrics*.csv'):
   with file.open() as f:series+=list(csv.DictReader(f))
  axs[0].plot([float(r['step']) for r in series],[float(r['loss']) for r in series],label=v,alpha=.8)
  vv=[json.loads(p.read_text()) for p in (ROOT/'runs'/v).glob('validation_step*.json')];vv=sorted(vv,key=lambda r:r['step'])
  axs[1].plot([r['step'] for r in vv],[100*r['summary']['category_component_S5'] for r in vv],marker='o',label=v)
  axs[2].plot([r['step'] for r in vv],[100*r['summary']['repeatability'] for r in vv],marker='o',label=v)
 for ax,title in zip(axs,['Train loss (raw, unsmoothed)','Selection component/category S@5 (%)','Selection repeatability@3 (%)']):ax.set_title(title);ax.set_xlabel('Optimizer step');ax.legend();ax.grid(alpha=.2)
 curves=save(fig,'training_curves')
 images=[]
 # Fixed representative IDs before looking at model success: first/middle/last, low texture, low overlap.
 ids=list(dict.fromkeys(r['id'] for r in rr));chosen=list(dict.fromkeys([ids[0],ids[len(ids)//2],ids[-1],next(r['id'] for r in rr if r['low_texture'] and not r['no_overlap']),next(r['id'] for r in rr if r['regime']=='low')]))
 def matchfig(id,rows,ims,realflag=False):
  methods=['m0','m2','a','b'];h0,w0=ims[0].shape;h1,w1=ims[1].shape;fig,axs=plt.subplots(4,1,figsize=(14,18),layout='constrained')
  for ax,v in zip(axs,methods):
   row=next(r for r in rows if r['method']==v);folder=ROOT/'real_predictions'/id if realflag else ROOT/'predictions/equal_final'/id;p=folder/f'{v}_{RUNSTAMP}.npz';z=np.load(p) if p.exists() else None
   for im,x in zip(ims,[0,w0]):
    s=min(1,1100/max(im.shape));a=cv2.resize(im,(round(im.shape[1]*s),round(im.shape[0]*s))) if s<1 else im;ax.imshow(a,cmap='gray',vmin=0,vmax=255,extent=(x,x+im.shape[1],im.shape[0],0))
   if z is not None:
    for k,x in [('keypoints0',0),('keypoints1',w0)]:kp=z[k];ax.scatter(kp[:,0]+x,kp[:,1],s=2,c='#999999',alpha=.4)
    p0,p1=z['points0'],z['points1'];ii=z['inliers'];good=ii if realflag else z['correct'];indices=np.linspace(0,len(p0)-1,min(80,len(p0)),dtype=int) if len(p0) else []
    for i in indices:ax.plot([p0[i,0],p1[i,0]+w0],[p0[i,1],p1[i,1]],color='#0072B2' if good[i] else '#D55E00',lw=.6,ls='-' if ii[i] else ':',alpha=.75)
   ax.set(xlim=(0,w0+w1),ylim=(max(h0,h1),0),aspect='equal');ax.set_title(f'{id} | {v.upper()} | points={row.get("n_keypoints0")}/{row.get("n_keypoints1")} matches={row.get("n_matches")} inliers={row.get("n_inliers")} | {row["decision"]} {row["reason"]}\n'+('Native real: blue=RANSAC self-consistency, independent GT=0' if realflag else 'Equal-count synthetic: blue=GT-correct static@3px, orange=wrong/nonstatic; max80 lines'))
  return save(fig,('real_' if realflag else 'validation_')+id)
 for id in chosen:ims=[cv2.imread(str(ROOT/'validation_data'/id/f'image{i}.png'),0) for i in [0,1]];images.append(matchfig(id,[r for r in rr if r['id']==id],ims))
 for ref in json.loads(output('real_inputs').read_text()):images.append(matchfig(ref['id'],[r for r in real if r['id']==ref['id']],[cv2.imread(p,0) for p in ref['paths']],True))
 # Explicit blank tests, no colored support claims.
 fig,axs=plt.subplots(2,4,figsize=(13,6),layout='constrained')
 for i,probe in enumerate(['black','gray']):
  for j,v in enumerate(['m0','m2','a','b']):row=next(r for r in san if r['method']==v and r['probe']==probe);axs[i,j].imshow(np.full((64,64),0 if i==0 else 128),cmap='gray',vmin=0,vmax=255);axs[i,j].set_title(f'{v}: {probe}\n{row["n_keypoints"]} detections; max={row["max_response"]:.2g}');axs[i,j].axis('off')
 blankfig=save(fig,'blank_acceptance')
 methods=['m0','m2','control','a','b','orb'];rows=[]
 for split in ['select','holdout','all']:
  for v in methods:
   s=comp[split][v];rows.append([split,v,s['n_positive'],s['n_negative'],pct(s['S3']),pct(s['S5']),pct(s['S10']),pct(s['repeatability']),num(s['correct_LG']),s['correct_support_pairs'],pct(s['no_overlap_false_accept']),pct(s['positive_incorrect_accept']),pct(s['reject_rate']),num(s['elapsed_ms'])])
 caps={v:np.mean([r['keypoints0'] for r in density if r['method']==v and not r['no_overlap']]) for v in ['m0','m2','a','b','control']}
 main_s=comp['all'];hold=comp['holdout'];b_blank=all(r['n_keypoints']==0 for r in san if r['method']=='b' and r['probe'] in ['black','gray']);b_density=caps['b']<=2*caps['m2']
 net={v:dict(holdout_S5_delta_pp=100*(hold[v]['S5']-hold['m2']['S5']),all_S5_delta_pp=100*(main_s[v]['S5']-main_s['m2']['S5']),holdout_correct_LG_delta=hold[v]['correct_LG']-hold['m2']['correct_LG']) for v in ['a','b','control']}
 candidate=[v for v in ['a','b','control'] if net[v]['holdout_S5_delta_pp']>0 and net[v]['holdout_correct_LG_delta']>0 and (v!='b' or (b_blank and b_density))]
 recommendation=max(candidate,key=lambda v:net[v]['holdout_S5_delta_pp']) if candidate else 'm2'
 conclusion=dict(completed=True,benefits=net,B_blank_gate=b_blank,B_native_density_gate=b_density,native_density=caps,recommendation= recommendation,independent_real_GT=0,real_accuracy_evidence=False,scientific_caveat='synthetic development data; no untouched source components; holdout already seen in M3; single seed; gains are exploratory, no raw acquisition significance')
 event('scientific_conclusion',conclusion)
 realtab=[]
 for v in ['m0','m2','a','b']:
  a=[r for r in real if r['method']==v];realtab.append([v,len(a),sum(r['accepted'] for r in a),sum(r['decision']=='rejected' for r in a),sum(r['decision']=='no_output' for r in a),sum(r['decision']=='exception' for r in a), '不可评价'])
 cfg=json.loads(output('config').read_text());identity=json.loads(output('data_identity').read_text())
 body=f'''# 多尺度SAR结构增强与稳定特征联合学习：受控实验报告\n\n生成时间：{now()}。所有Python使用match，正式训练和评价物理GPU1；原M0/M1/M2/M3、数据、900评价均保持冻结。\n\n## 技术结论\n\n建议保留：**{recommendation.upper()}**。A/B/继续训练对照相对M2的内部留出S@5变化分别为{net['a']['holdout_S5_delta_pp']:+.2f}/{net['b']['holdout_S5_delta_pp']:+.2f}/{net['control']['holdout_S5_delta_pp']:+.2f}个百分点。此为单seed、既有二维代理开发数据上的结果，不能证明真实异向散射中心对应改善。B空白门槛={b_blank}，未经截断的平均检测密度门槛={b_density}。\n\n## 动机、结构与训练范围\n\nM2检测可重复点不足；M3失去响应校准，空白图992/994点、等点数劣于M2。本轮新增仅三尺度对数显示强度梯度分支，不加入Log-Gabor/DCN/Transformer。当前源输入为uint8 SAR显示强度，log1p不是恢复原始辐射量或复杂幅度。三个固定Gaussian sigma1/2/4，Scharr梯度幅值，8倍avgpool，Conv3→16→32→256，最后1×1零初始化；残差注入convDa输出、ReLU前。起始A/B描述子与M2严格一致。残差是可学习结构特征融合，没有额外独立门控网络。A只训练新增13,536参数；B同时训练convPa/convPb，共325,409参数；继续训练对照仅原convDa/convDb，共360,960参数。共享骨干与LG始终冻结。B从同一M2初始化联合训练，未额外享用A预训练步数。\n\n![网络结构]({structure})\n\n## 损失与M3修正\n\nA/对照：实际检测点经精确H投影得到对应，reference位置来自实际检测，moving正对应为已知H位置双线性采样；评估则使用双图实际检测点，二者不能混淆。双向InfoNCE temperature.1，8px邻近替代正对应排除，+.1固定M2描述子保持。B使用当前检测点（位置detach）训练描述子；检测训练采用固定平滑结构角点与M2可描述锚点，已知H投影，排除动态bbox+5和无效像素。检测损失=.05正锚点CE+1原M2分布KL+1无结构/无效且非动态单元dustbin CE+1黑/均匀灰图dustbin CE+.2正锚点响应线性一致性。与M3不同，正点不压倒背景校准、显式训练空白负例、低纹理没有强制补点、硬验收空白0点。结构响应支持阈值.012固定在训练前，未在留出或真实数据调整。空对应样本安全返回有梯度零描述损失；不伪造正例。\n\n同seed、相同Train组件采样、同增强课程、batch累积2、FP32，A/B分支lr1e-4、B检测lr1e-6、对照描述lr1e-5，AdamW weight_decay1e-4/clip1。各初始最多3000步，每500验证（首次100），至少1000后连续3次不改善早停。方案间优化参数量/学习率不同，属于方法对照而非全参数严格计算量相等。\n\n## 数据与证据范围\n\nTrain668组件，原Val112组件，索引test{identity['test_components']}组件，已知train/val/test交集0。全新Val组件为0。复用M3冻结294对像素，分179选模、115内部留出，按组件交集0。两部分早在M3研究中已观察；内部留出仅未参与本轮A/B选模，不能叫全新独立确认。SSDD原采集身份不完整，known-component隔离不等于证明独立雷达采集。没有新背景组件、近岸来源极少，局限不通过混入Train修补。动态舰船监督排除。旧900不参与训练/验证/本轮推理。\n\n## 统一协议与等点数\n\n相同256×256冻结PNG，NMS4、阈值.0005、上限1024、官方冻结LG、原生坐标和冻结Homography/RANSAC3/full门控。主表每幅图预算=min(M0,M2,A,B,继续训练对照的实际点数)，各方法按原响应分数取同样N；预算只依赖纯图像检测，不读GT。ORB在同一上限下采用fastThreshold12、Hamming/ratio.8/双向互检，可能实际点数更少；其正确匹配列是ORB输出，不是LG输出。另保留未经共同截断的native-budget结果，避免掩盖B密度变化。\n\n选模采用固定M2点数上限（B不足时不补点），空白0点与未截断点数≤M2两倍为硬门槛，组件/类别S@5为主。初版实现误对截断后的点数检查密度；发现后保留原代码、日志和checkpoint，用修正版对B已有best重算门槛并从step800恢复。最终验收仍独立检查native点数；如果所有训练checkpoint都未过硬门槛，提交训练结果但明确不合格，不能为了交付而自动降低门槛。主表严格共同等点数，未按最终留出重新选择checkpoint。\n\n{table(['划分','方法','正例','无重叠','S@3','S@5','S@10','重复性@3','正确静态匹配/对','正确支持对数','无重叠误接受','正例错接受','拒绝率','均时ms'],rows)}\n\nSuccess要求门控接受且均匀共同静态GT格网平均误差≤阈值，失败保留分母。重复性基于实际检测GT3px邻域；descriptor条件Recall和端到端Recall逐对CSV区分。空间支持至少8个正确静态点且凸包≥1%，不是仅看RANSAC内点。各场景、实际重叠、低纹理、动态及组件见分层CSV。\n\n## 实际训练、权重和曲线\n\nB初版在step873第二个micro样本遇到全舰船区域，nondynamic集合为空，KL均值为NaN；自动停止，没有带NaN继续优化。固定种子26118782与来源P0130_2400_3200_4800_5600复现，修正为空集合有梯度零损失并通过回归检查。从step800完整状态恢复；仅修正数值空集合与实际密度验收，不修改损失权重/学习率/预算。原失败日志、原best/last备份和源码版本均保留。\n\n'''
 for v,lock in locks.items():body+=f'- {v}: 优化{lock["final_step"]}步，best{lock["best"]["step"]}，耗时{lock["elapsed_seconds"]/60:.2f}分钟，checkpoint `runs/{v}/best.ckpt` SHA256 `{lock["best_sha256"]}`，结束原因 `{lock["reason"]}`。\n'
 body+=f'\n![训练与选模]({curves})\n\n## 空白、噪声与未经截断密度\n\n'+table(['模型','输入','点数','响应最大值','响应均值'],[[r['method'],r['probe'],r['n_keypoints'],f'{r["max_response"]:.7g}',f'{r["mean_response"]:.7g}'] for r in san])+f'\n![空白验收]({blankfig})\n\n未截断正例左图平均点数：{json.dumps(caps)}。低纹理SAR分层与各点数记录保留在CSV；随机噪声出点不能视为静态地物。\n\n## 真实Native诊断\n\n'+table(['方法','项数','接受','拒绝','无输出','异常','真实精度'],realtab)+'\n\n18项已观察真实Native，仅图像输入，原始像素不缩放；真实对照也按四模型各图实际点数的最小值严格统一点数，不读GT设置预算。单独推理CLI采用自然最多1024点，因此不能将CLI自然点数结果和此等点数对照混用。大图采用经过训练后等价检查的1024/64卷积分块、全局NMS和全局坐标采样。后续附加site/look标签不进入预测。没有独立物理GT；自拟合残差、RANSAC内点和接受不代表真实正确性。预测保留所有拒绝候选。真实结果不用于权重或阈值选择。\n\n## 可视化、失败及局限\n\n'
 for image in images:body+=f'![保存的匹配与点分布]({image})\n\n'
 body+='## 部署与复现\n\n见README及`code/infer_pair.py`。依赖没有安装升级，环境版本见environment记录。完整source/data/code身份与preflight保存；训练仅更新列出的参数，冻结参数最终逐张量验证。所有checkpoint保留optimizer/scheduler/scaler/RNG/配置/代码数据身份，completed模型禁止重新启动。无独立真实GT、无全新来源、单seed及同源散斑共享限制均影响学术主张；本轮不自动启动新网络。\n'
 md=ROOT/f'structure_stable_report_{stampnow}.md';md.write_text(body)
 # Reuse existing stdlib markdown renderer, no new packages.
 source=(BASE/'SAR_SP_LG_TrainingReview_20261012_1651/code/export_pdf.py').read_text();scope={'html':html,'re':re};exec(source[source.index('def inline'):source.index('def main')],scope)
 renderer=scope.get('render') or scope.get('render_markdown');assert renderer
 text=renderer(body);hp=md.with_suffix('.html');hp.write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1500px;margin:30px auto;padding:20px;font:16px sans-serif}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}pre{white-space:pre-wrap}</style>'+text)
 event('delivery',dict(markdown=str(md),html=str(hp),figures=[structure,curves,blankfig]+images,conclusion=conclusion));print(str(md))
if __name__=='__main__':main()
