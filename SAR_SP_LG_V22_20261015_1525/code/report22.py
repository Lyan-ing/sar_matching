"""Direct evidence tables and figures, no pandas/new dependencies."""
from support22 import *
from evaluate22 import summarize
from collections import Counter,defaultdict
os.environ['MPLCONFIGDIR']=str(ROOT/'mpl_cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def coverage(run):
 used=[]
 for p in sorted(run.glob('sampling_from*.jsonl')):used.extend(rows(p))
 bycall={r['call_index']:r for r in used};assert len(bycall)==len(used),'Repeated consumed calls'
 rr=list(bycall.values());files=Counter(r['source_path'] for r in rr);positive=Counter(r['source_path'] for r in rr if r.get('ground_points_used',0)>=4);roles=Counter(r.get('actual_role','rejected') for r in rr);active=[r for r in rr if r.get('active')]
 lock=json.loads(out('selection_locked',folder=run).read_text());assert len(active)==lock['actual_steps']*8
 stats=dict(run=run.name,steps=lock['actual_steps'],best_step=lock['best']['step'],calls=len(rr),active_pairs=len(active),inactive_calls=len(rr)-len(active),inactive_fraction=1-len(active)/len(rr),zero_gradient_optimizer_steps=0,source_files=len(files),ground_descriptor_sources=len(positive),max_source_calls=max(files.values()),roles=dict(roles),roles_active=dict(Counter(r['actual_role'] for r in active)),ground_points_used=sum(r.get('ground_points_used',0) for r in rr),ship_points_used=sum(r.get('ship_points_used',0) for r in rr),peak_MiB=lock['peak_MiB'],elapsed_seconds=lock['elapsed_seconds'],best_sha=lock['best_sha'],best_hard_gate=lock['best']['hard_gate'])
 csvwrite(out('source_coverage',ext='csv',folder=run),[dict(source_path=s,consumed_calls=n,ground_descriptor_calls=positive[s]) for s,n in sorted(files.items())]);write(out('coverage_summary',folder=run),stats)
 return stats
def figures(resultsets,trainstats):
 folder=ROOT/'figures';folder.mkdir(exist_ok=True);paths=[];fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
 for run in trainstats:
  rr=[]
  for p in (ROOT/'runs'/run['run']).glob('metrics_from*.csv'):
   with p.open() as f:rr.extend(list(csv.DictReader(f)))
  rr=sorted(rr,key=lambda r:int(r['step']));x=np.array([int(r['step']) for r in rr]);y=np.array([float(r['loss']) for r in rr]);w=min(100,len(y));axs[0,0].plot(x,y,alpha=.12,lw=.3);axs[0,0].plot(x[w-1:],np.convolve(y,np.ones(w)/w,'valid'),label=run['run'])
  vv=sorted((ROOT/'runs'/run['run']).glob('validation_step*.json'),key=lambda p:json.loads(p.read_text())['step']);d=[json.loads(p.read_text()) for p in vv];axs[0,1].plot([s['step'] for s in d],[s['summary']['S5'] for s in d],marker='.',label=run['run']);axs[1,0].plot([s['step'] for s in d],[s['summary']['local_R1'] for s in d],marker='.',label=run['run']);axs[1,1].plot([s['step'] for s in d],[s['probes'][2]['n_keypoints'] for s in d],marker='.',label=run['run'])
 for ax,title in zip(axs.flat,['Training loss (raw faint, 100-step moving mean)','Fixed selection S@5 (budget specified in protocol)','Limited-structure oracle descriptor R1','Random-noise detections / 256x256']):ax.set_title(title,fontsize=10);ax.set_xlabel('Optimizer step');ax.grid(alpha=.2);ax.legend(fontsize=6)
 p=out('training_curves',ext='png',folder=folder);fig.savefig(p,dpi=180);plt.close(fig);paths.append(p)
 fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
 for ax,version in zip(axs,['v1','v21','v22']):
  d=resultsets['main_'+version]['summaries'];methods=list(d['natural']);x=np.arange(len(methods));ax.bar(x-.18,[d['natural'][m]['S5']*100 for m in methods],.35,label='Natural <=1024');ax.bar(x+.18,[d['equal'][m]['S5']*100 for m in methods],.35,label='Shared actual <=128');ax.set_xticks(x,methods,rotation=60,fontsize=8);ax.set_ylim(0,100);ax.set_title(version+' (own denominator)');ax.set_ylabel('Synthetic global S@5 (%)');ax.legend(fontsize=7)
 p=out('version_comparison',ext='png',folder=folder);fig.savefig(p,dpi=180);plt.close(fig);paths.append(p)
 return paths
def previews(data):
 folder=ROOT/'figures';rr=[r for r in data['rows'] if r['protocol']=='natural'];pairs=defaultdict(dict)
 for r in rr:pairs[r['sample_id']][r['method']]=r
 examples=[]
 for scene in ['nearshore_vessels','port_quay_dense_vessels','other_land_structures','offshore_vessels_low_texture_sea']:
  p=[v for v in pairs.values() if next(iter(v.values()))['scene']==scene]
  p.sort(key=lambda v:(not bool(v.get('M2_V22',{}).get('success5')) or bool(v.get('M2',{}).get('success5')),next(iter(v.values()))['sample_id']))
  if p:examples.append(next(iter(p[0].values()))['sample_id'])
 lookup={r['sample_id']:r for r in rows(E['recommended_val_all'])};paths=[]
 for sid in examples:
  row=lookup[sid];ims=[cv2.imread(row[f'image{i}'],0) for i in [0,1]];fig,axs=plt.subplots(3,1,figsize=(12,13),layout='constrained')
  for ax,method in zip(axs,['M2','M2_V22','B_V22']):
   z=np.load(out(f'{method}_natural',ext='npz',folder=ROOT/'predictions/main_v22/v22'/sid));canvas=np.concatenate(ims,axis=1);ax.imshow(canvas,cmap='gray',vmin=0,vmax=255);p,q=z['points0'],z['points1'].copy();q[:,0]+=ims[0].shape[1];correct=z['correct_static'];ids=np.arange(len(p));ids=ids[np.linspace(0,len(ids)-1,min(70,len(ids))).astype(int)] if len(ids) else ids
   for j in ids:ax.plot([p[j,0],q[j,0]],[p[j,1],q[j,1]],color='#009E73' if correct[j] else '#D55E00',alpha=.55,lw=.65)
   r=pairs[sid][method];ax.set_title(f'{method}: matches={r["n_matches"]}, known-static correct={r["correct_static_matches"]}, {r["state"]}; {r["reason"]}',fontsize=10);ax.axis('off')
  fig.suptitle(sid+' / '+row['source_scene_category']+'\nGreen: synthetic static GT <=3px; orange: remaining (not necessarily physically incorrect).',fontsize=11);p=out(sid,ext='png',folder=folder);fig.savefig(p,dpi=150);plt.close(fig);paths.append(p)
 return paths
def main():
 resultsets={p.stem.rsplit('_',2)[0]:json.loads(p.read_text()) for p in ROOT.glob('*_v[12]*_'+TAG+'.json') if p.stem.startswith(('main_','attribution_','ship_'))};assert all('main_'+v in resultsets for v in ['v1','v21','v22'])
 trainstats=[coverage(p) for p in sorted((ROOT/'runs').iterdir()) if not p.name.startswith('smoke') and out('selection_locked',folder=p).exists()];csvwrite(out('training_coverage',ext='csv'),trainstats)
 summaries=[];layers=[]
 for name,a in resultsets.items():
  for mode,models in a['summaries'].items():
   for m,s in models.items():summaries.append(dict(experiment=name,protocol=mode,method=m,**s))
  for key in ['scene','angle_bin','size','profile','role']:
   for value in sorted({str(r[key]) for r in a['rows']}):
    for mode in ['natural','equal']:
     for m in a['summaries'][mode]:
      rr=[r for r in a['rows'] if str(r[key])==value and r['protocol']==mode and r['method']==m];layers.append(dict(experiment=name,dimension=key,value=value,protocol=mode,method=m,**summarize(rr)))
 csvwrite(out('unified_comparison',ext='csv'),summaries);csvwrite(out('stratified_metrics',ext='csv'),layers);paths=figures(resultsets,trainstats)+previews(resultsets['main_v22'])
 identity=json.loads(out('input_identity').read_text());intact={p:sha(p)==h for p,h in identity['sha256'].items()};assert all(intact.values())
 t=stamp();report=ROOT/f'V22_TRAINING_REPORT_{t}.md';lines=['# V2.2 SAR受控训练与独立分母评价',f'生成时间：{now()}。全部是同源二维合成GT，不证明真实跨采集方位配准。','', '## 执行与可复现性','使用match环境与既有M2/B结构，LightGlue及共享骨干冻结。新目录内graded_loss统一使用保存锚点，充分/有限分别加权；少于4点不计算描述CE，未知区不设负标签。有证据低纹理检测校准仅训练B。解析无重叠与可选舰船辅助独立。','', '交付B接口实测1–3个锚点也可能得到4个以上重采样点并计算描述CE，原实现/数据均保留；新桥接修正这一训练准入问题，不能将新B训练与旧B分数称为单因素损失无变化复现。','', '## 正式训练与覆盖','|运行|实际步数|最佳步|有效配对|不同源文件|地面描述来源|跳过调用|峰值MiB|','|---|---:|---:|---:|---:|---:|---:|---:|']
 for s in trainstats:lines.append(f'|{s["run"]}|{s["steps"]}|{s["best_step"]}|{s["active_pairs"]}|{s["source_files"]}|{s["ground_descriptor_sources"]}|{s["inactive_calls"]}|{s["peak_MiB"]:.1f}|')
 lines+=['','固定core Train 6420关系中，地面描述监督2324关系/677原始发布文件；在线新几何的实际可用来源另以日志统计。重复增强不是不同原始采集。B/M2因冻结范围不同使用不同角色参考配额，实际角色以新配对重算。全部有效batch为8；失败和空监督推进调用索引并留痕，不产生伪描述标签或空优化步。','', '## 统一全幅合成配准','各版本保留各自分母。V1沿用原静态点；V2.1与V2.2仅充分结构计算全幅Success。有限结构、未知与低纹理不进入正负全幅分母。Equal为本组所有模型实际共同点数（上限128），Natural上限1024；不同实验组的Equal预算不得混减。','', '|验证版本/点数协议|方法|全幅正例|S@3|S@5|S@10|正例错误接受|无重叠误接受|','|---|---|---:|---:|---:|---:|---:|---:|']
 def pct(v):return 'NA' if v is None else f'{100*v:.2f}%'
 for s in summaries:
  if not s['experiment'].startswith('main_'):continue
  lines.append('|'+ '|'.join([s['experiment']+'/'+s['protocol'],s['method'],str(s['global_n']),pct(s['S3']),pct(s['S5']),pct(s['S10']),pct(s['wrong_accept']),pct(s['no_overlap_FPR'])])+ '|')
 lines+=['','## 局部监督、校准及舰船辅助','局部oracle Recall使用保存GT位置，actual detected Recall在实际关键点上计算，两者不同。所有逐样本值、有限点数、校准区检测、无输出/拒绝/错误接受见各版本CSV；场景、二维旋转、尺寸、重叠profile及角色见stratified_metrics。离岸仍没有合格地面描述正例，不报告该类的虚假全幅地面Success；船辅助仅为同源合成、分割内部的独立指标。','', '|实验（V2.2/Natural）|方法|有限结构oracle R1|校准区平均点数|合成船正确匹配均值|','|---|---|---:|---:|---:|']
 for s in summaries:
  if s['experiment'].endswith('v22') and s['protocol']=='natural':lines.append(f'|{s["experiment"]}|{s["method"]}|{pct(s["local_R1"])}|{s["calibration_keypoints0"]}|{s["ship_correct"]}|')
 lines+=['','## 数据归因与限制','归因三臂网络、学习率、有效batch与最大优化预算相同：旧严格筛选1078固定关系/452源；同像素恢复局部监督1465关系/611源；仅新增ROI 857关系/580源。它们来自同一原Train池但可监督源数不同，不能将差异全归因于裁片数量或声称源完全匹配的因果实验。主训练使用在线重新生成，归因采用固定像素以保持“同图”含义；在线与固定差异不作单因素直接相减。','', '模型只在固定V2.2开发Val子集选模；全部V1标准/V2.1冻结/V2.2固定Val只用于版本分别汇报。Val均是既有Val来源，并非新未见最终测试；已知父组件隔离不能证明所有原始采集绝对独立。历史900未运行，真实人工GT未产生。训练loss下降或更多点数均不构成真实跨角改善证据。','', '## 实际图表']
 for p in paths:lines+=['',f'![{p.stem}]({p.relative_to(ROOT)})']
 report.write_text('\n'.join(lines)+'\n');write(ROOT/f'completion_identity_{t}.json',dict(at=now(),report=str(report),old_inputs_unchanged=intact,models=trainstats,code_sha={str(p):sha(p) for p in (ROOT/'code').glob('*.py')}));print(report)
if __name__=='__main__':main()
