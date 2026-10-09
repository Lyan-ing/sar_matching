"""Audit saved predictions, export evidence and write a timestamped Phase 2 report.
No matcher inference, parameter selection or Phase 1 writes occur here.
"""
import argparse, collections, datetime, hashlib, importlib.metadata, json, os, subprocess, sys
from pathlib import Path
from zoneinfo import ZoneInfo
import cv2
import numpy as np
from sarbench.data import discover, load_images, load_gt
from sarbench.estimate import Estimated
from sarbench.evaluate import evaluate
from sarbench.reports import read_csv, save_json, _truth

p=argparse.ArgumentParser();p.add_argument('--comparison',required=True);p.add_argument('--log',required=True);a=p.parse_args()
base=Path.cwd();out=Path(a.comparison).resolve();summary=json.loads((out/'summary.json').read_text());rows=read_csv(out/'results.csv')
methods=['sift','orb','phase','lightglue'];names={'sift':'SIFT + RANSAC','orb':'ORB + RANSAC','phase':'Phase correlation','lightglue':'SuperPoint + LightGlue'}
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
sources=[Path(x).parent for x in summary['comparison_sources']]
provs=[json.loads((x/'provenance.json').read_text()) for x in sources]
root=Path(provs[0]['phase1_root']);refs=discover(root);byid={r.id:r for r in refs}
assert len(rows)==3600 and len(refs)==900 and len({r.folder for r in refs})==860
assert set(r['method'] for r in rows)==set(methods)
assert sha(out/'results.csv')==json.loads((out/'RUN_COMPLETE.json').read_text())['results_sha256']
for src in sources:
 receipt=json.loads((src/'RUN_COMPLETE.json').read_text());assert receipt['results_sha256']==sha(src/'results.csv') and receipt['method_exceptions']==0
for m in methods:
 subset=[r for r in rows if r['method']==m];assert len(subset)==900 and {r['id'] for r in subset}==set(byid)
 assert sum(not _truth(r['no_overlap']) for r in subset)==860 and sum(_truth(r['no_overlap']) for r in subset)==40
assert all(r['source_split']=='test' and not r['error'] and not r.get('preview_error') for r in rows)
checked=0
for check in root.glob('file_checksums_*.json'):
 for record in json.loads(check.read_text()):assert sha(root/record['path'])==record['sha256'];checked+=1
assert checked==11093
# Recompute all metrics from recorded H and held-out native-coordinate GT.
cache={};max_metric_difference=0.
for r in rows:
 ref=byid[r['id']]
 if ref.id not in cache:
  im0,im1=load_images(ref);assert im0.shape==im1.shape
  cache[ref.id]=(im0.shape,load_gt(ref))
 shape,gt=cache[ref.id];assert shape==(int(r['image_height']),int(r['image_width']))
 H=np.asarray(json.loads(r['pred_H']),float) if r['pred_H'] else None
 if _truth(r['accepted']):assert H is not None and H.shape==(3,3) and np.isfinite(H).all()
 est=Estimated(H,_truth(r['accepted']),r['reason'],int(r['n_matches']),int(r['n_inliers']),float(r['inlier_ratio']),None,None,None,r['model'],np.eye(3) if _truth(r['has_estimate']) else None)
 ev=evaluate(est,gt,shape)
 for key in ['no_overlap','success_3px','success_5px','success_10px','decision_status','false_accept','has_estimate']:
  expected=ev[key];actual=r[key]
  assert (actual=='' if expected is None else _truth(actual)==expected if isinstance(expected,bool) else actual==expected),(r['id'],r['method'],key)
 if ev['epe_mean'] is not None:
  d=abs(float(r['epe_mean'])-ev['epe_mean']);max_metric_difference=max(max_metric_difference,d);assert d<1e-8
lg=next(q for q in provs if 'lightglue' in q['methods']);lgmeta=lg['lightglue'];lgroot=Path(lgmeta['source_path'])
original=subprocess.check_output(['git','-C',str(lgroot),'show','HEAD:requirements.txt'])
assert (lgroot/'requirements.txt').read_bytes()==original.replace(b'\nopencv-python\n',b'\nopencv-python-headless\n')
assert subprocess.check_output(['git','-C',str(lgroot),'diff','HEAD','--name-only'],text=True).splitlines()==['requirements.txt']
assert lgmeta['source_commit']=='eb42fee2d71449efb0aa5c10549752b5d75384d8'
weights=lgmeta['weights_sha256'];assert weights['superpoint_v1.pth']=='52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e'
assert weights['superpoint_lightglue_v0-1_arxiv.pth']=='6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a'
for filename,expected in weights.items():assert sha(base/'cache/torch/hub/checkpoints'/filename)==expected
packages={d['name']:d['version'] for d in json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True))};assert [x for x in packages if x.lower().startswith('opencv-')]==['opencv-python-headless']
pipcheck=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True);assert pipcheck.returncode==0,pipcheck.stdout+pipcheck.stderr

# Deterministic post-hoc examples expose successes and failures; they are not a performance sample.
by_pair=collections.defaultdict(dict)
for r in rows:by_pair[r['id']][r['method']]=r
chosen=[]
def choose(label,predicate):
 found=next((k for k in sorted(by_pair) if predicate(by_pair[k])),None)
 if found:chosen.append((label,found))
choose('Inshore success',lambda d:d['orb']['source_scene']=='inshore' and d['orb']['decision_status']=='success' and d['lightglue']['decision_status']=='success')
choose('Low overlap: SIFT success',lambda d:d['sift']['profile']=='low' and d['sift']['decision_status']=='success' and d['orb']['decision_status']!='success')
choose('Background: ORB / LightGlue success',lambda d:d['orb']['source_scene']=='background' and d['orb']['decision_status']=='success' and d['lightglue']['decision_status']=='success')
choose('No overlap: LightGlue false accept',lambda d:_truth(d['lightglue']['false_accept']))
worst=max((r for r in rows if r['method']=='lightglue' and r['epe_mean']),key=lambda r:float(r['epe_mean']))
worst_H=np.asarray(json.loads(worst['pred_H']));wh=int(worst['image_height']);ww=int(worst['image_width'])
corner_z=np.array([[0,0,1],[ww-1,0,1],[0,wh-1,1],[ww-1,wh-1,1]])@worst_H[2]
worst_gt=cache[worst['id']][1];gt_z=np.c_[worst_gt['points0'],np.ones(worst_gt['count'])]@worst_H[2]
chosen.append(('Dynamic: extreme LightGlue extrapolation',worst['id']))
choose('LightGlue estimate rejected',lambda d:d['lightglue']['decision_status']=='rejected' and not _truth(d['lightglue']['no_overlap']))
choose('Three-view pair',lambda d:d['lightglue']['num_views']=='3' and d['lightglue']['decision_status']=='success')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
art=out/'delivery_figures'
if art.exists():raise FileExistsError(art)
art.mkdir()
fig,axes=plt.subplots(len(chosen),7,figsize=(18,2.5*len(chosen)),layout='constrained')
case_records=[]
for i,(label,k) in enumerate(chosen):
 ref=byid[k];im0,im1=load_images(ref);gt=load_gt(ref);h,w=im0.shape
 columns=[('View 0',im0),('View 1',im1)]
 def overlay(H):
  warped=cv2.warpPerspective(im1,np.linalg.inv(H),(w,h))
  # Whole-image linear mapping [0,255]->[0,1]; no local contrast enhancement.
  return np.stack([im0,warped,warped],axis=-1)/255.
 columns.append(('GT warp (evaluation only)',overlay(gt['H']) if not gt['no_overlap'] else None))
 for m in methods:
  r=by_pair[k][m];H=np.asarray(json.loads(r['pred_H'])) if r['pred_H'] else None
  e=f"EPE={float(r['epe_mean']):.2f}px" if r['epe_mean'] else 'EPE=N/A'
  columns.append((f"{m}: {r['decision_status']}\n{e}",overlay(H) if H is not None else None))
 for j,(title,img) in enumerate(columns):
  ax=axes[i,j]
  if img is None:ax.text(.5,.5,'NO ACCEPTED ESTIMATE' if j>2 else 'NO OVERLAP',ha='center',va='center',fontsize=8,transform=ax.transAxes)
  else:ax.imshow(img,cmap='gray',vmin=0,vmax=255 if img.ndim==2 else 1,interpolation='nearest')
  ax.set_title(title,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
  if j==0:ax.set_ylabel(label+'\n'+k,fontsize=8)
 case_records.append({'label':label,'id':k,'dataset':ref.dataset,'profile':ref.profile,'methods':{m:{field:by_pair[k][m][field] for field in ['decision_status','epe_mean','reason','n_matches','n_inliers','coverage','inlier_rmse']} for m in methods}})
fig.savefig(art/'representative_cases.png',dpi=140);plt.close(fig)
# Comparison plot retains zero baseline, labels counts and has a table in the report.
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
styles=['/','\\','x','.'];colors=['#0072B2','#E69F00','#009E73','#CC79A7']
for i,m in enumerate(methods):
 s=summary['by_method'][m];v=100*s['success_5px_rate'];ax.bar(i,v,color=colors[i],hatch=styles[i],edgecolor='black');ax.text(i,v+1,f'{v:.2f}%\n({s["decision_counts"].get("success",0)}/860)',ha='center',fontsize=9)
ax.set_xticks(range(4),[names[m] for m in methods]);ax.set_ylim(0,105);ax.set_ylabel('Success @ 5 px (% of all positive pairs)');ax.set_title('2D synthetic SAR proxy; fixed test, no tuning');fig.savefig(art/'four_method_success.png',dpi=180);fig.savefig(art/'four_method_success.pdf');plt.close(fig)

stamp=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');reports=base/'reports';report=reports/f'phase2_final_report_{stamp}.md';log_final=reports/f'phase2_execution_{stamp}.log'
if report.exists():raise FileExistsError(report)
quality={'accepted':True,'scope':'engineering acceptance of complete fixed 2D synthetic-proxy benchmark; no real multi-angle validation','rows':len(rows),'relations_per_method':900,'positive_per_method':860,'negative_per_method':40,'groups':860,'checked_unchanged_phase1_files':checked,'all_metric_rows_recomputed':3600,'max_metric_difference_px':max_metric_difference,'exceptions':0,'test_source_only':True,'shared_protocol':summary['shared_protocol'],'phase1_manifest_sha256':sha(root/'manifest.jsonl'),'phase1_receipt_sha256':sha(next(root.glob('ACCEPTED_*.json'))),'comparison_results_sha256':sha(out/'results.csv'),'model_inference_source_unmodified':True,'lightglue_metadata_patch':lgmeta['source_diff'],'python_executable':sys.executable,'pip_check':pipcheck.stdout.strip(),'cases':case_records,'generated_at':datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')}
quality_path=reports/f'phase2_acceptance_{stamp}.json';save_json(quality_path,quality)
quality['worst_lightglue_geometry']={'id':worst['id'],'corner_projective_denominators':corner_z.tolist(),'minimum_abs_gt_denominator':float(np.min(np.abs(gt_z))),'projective_pole_crosses_image':bool(corner_z.min()<0<corner_z.max())}
save_json(quality_path,quality)
env_path=reports/f'phase2_environment_{stamp}.json';save_json(env_path,{'python':sys.version,'executable':sys.executable,'packages':packages,'conda_list':json.loads(subprocess.check_output(['conda','list','-n','match','--json'],text=True)),'gpu_provenance':lgmeta,'source_runs':[str(x) for x in sources]})
freeze=reports/f'phase2_pip_freeze_{stamp}.txt';freeze.write_text(subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True))
lock=reports/f'phase2_requirements_lock_{stamp}.txt';lock.write_text('# Linux Python 3.10; torch/torchvision cu121 require official extra-index.\n'+'\n'.join(f'{k}=={v}' for k,v in sorted(packages.items(),key=lambda x:x[0].lower()) if k.lower() not in ('lightglue','maritime-sar-registration-phase2'))+'\n')
patch=reports/f'phase2_lightglue_metadata_{stamp}.patch';patch.write_text(lgmeta['source_diff'])

def pct(v):return 'N/A' if v is None else f'{100*v:.2f}%'
def num(v):return 'N/A' if v is None else f'{v:.3f}'
def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in data])
text=[]
def add(x):text.append(x)
add(f'# 第二阶段正式实验报告 — {stamp}\n\n生成时间：{quality["generated_at"]}（Asia/Shanghai）。工程验收通过：四类方法各 900 对，共 3,600 条保存结果，无推理异常。传统方法结果复用已完成运行，第一阶段冻结数据保持原样。')
add(f'## 数据与证据位置\n\n- 冻结数据：`{root}`。860 组：840 双视图、20 三视图；900 条关系，860 有重叠、40 无重叠；全部来源为 test。\n- 传统方法：`{sources[0]}`。\n- LightGlue：`{sources[1]}`。\n- 四方法比较：`{out}`；逐对结果 `results.csv`、分层 `summary.json`、80 条三视图诊断 `cycle_consistency.csv`。\n- 验收证据：[{quality_path.name}]({quality_path.name})；环境：[{env_path.name}]({env_path.name})；执行日志：[{log_final.name}]({log_final.name})。\n- 所有定量表由上述 CSV/JSON 生成；图由 `scripts/finalize_phase2.py` 读取原始图像和已保存预测生成。正式集不调参，不训练或微调网络。')
add('## 实验协议与指标\n\n匹配器接口仅接收两幅原生灰度图，GT、重叠/海陆/舰船掩膜和真实 H 不进入预测或接受门限；GT 仅用于推理结束后的评价与展示。视图原生边长 128–384，不额外放大或统一 resize。预测 H 为 view_i → view_j；LightGlue `resize=None`，关键点保持原坐标。\n\nSIFT/ORB 使用双向 mutual ratio=0.8，ORB fastThreshold=12；三种特征方法使用同一 Homography RANSAC：3px、最多 5,000 次、confidence=0.995；最低匹配/内点各 8、内点率 0.3、4×4 网格覆盖至少 1/16、拟合 RMSE≤3px。Phase 使用 Hanning 窗和 response≥0.20，只估计平移，是模型能力受限的诊断基线。参数沿用代码包默认值，未按正式测试成绩选择。\n\nSuperPoint 最大 2,048 点，官方 detector threshold=0.0005；LightGlue 官方默认 9 层、flash=True、mp=False、depth=0.95、width=0.99、filter=0.1，没有 compile 或微调。完整配置见 LightGlue provenance。seed=42；RANSAC 每对 seed=42+固定排序索引；Python/NumPy/Torch 也固定种子。OpenCV 和 Torch CPU 线程各 1。GPU 默认内核的逐位确定性未强制启用，不能承诺不同驱动/硬件下逐位相同。\n\nSuccess@Npx：**接受且 512 个静态 GT 对应点的平均重投影误差≤Npx**，分母为全部 860 条正样本。拒绝或没有 H 均计失败。无重叠 40 条独立计算错误接受率。EPE 均值/中位数仅在已接受且可评价的正样本上计算，包含错误接受，不替代总体成功率。动态区域已由冻结 GT 排除；匹配器仍看到动态舰船。\n\n`no_output` 表示共享估计器没有输出可用候选 H（可能仍有少量匹配）；`rejected` 表示已有候选 H 被预测门限拒绝；`incorrect_accept` 包含正样本 EPE>5px 和无重叠错误接受；`success` 为正样本 Success@5px。两类未接受不得混称算法拒绝。')
add('## 四方法总体结果\n\n'+table(['方法','Success@3','Success@5','Success@10','正样本接受率','正样本失败率@5','无重叠错误接受'],[[names[m],pct(summary['by_method'][m]['success_3px_rate']),pct(summary['by_method'][m]['success_5px_rate']),pct(summary['by_method'][m]['success_10px_rate']),pct(summary['by_method'][m]['acceptance_positive_rate']),pct(summary['by_method'][m]['failure_5px_positive_rate']),f"{summary['by_method'][m]['false_accept_count']}/40 ({pct(summary['by_method'][m]['false_accept_rate'])})"] for m in methods]))
add(table(['方法','接受正样本数','EPE 均值 / 中位数 px','平均时间 ms','中位 / P95 ms','关键点 图0 / 图1','匹配数 / 内点数均值'],[[names[m],summary['by_method'][m]['epe_evaluable_accepted_count'],num(summary['by_method'][m]['mean_epe_on_accepted_estimates'])+' / '+num(summary['by_method'][m]['median_epe_on_accepted_estimates']),num(summary['by_method'][m]['runtime_ms_mean']),num(summary['by_method'][m]['runtime_ms_median'])+' / '+num(summary['by_method'][m]['runtime_ms_p95']),num(summary['by_method'][m]['avg_keypoints0'])+' / '+num(summary['by_method'][m]['avg_keypoints1']),('N/A' if m=='phase' else num(summary['by_method'][m]['avg_matches'])+' / '+num(summary['by_method'][m]['avg_inliers']))] for m in methods]))
add('计时排除图像解码、GT 评价、预览、模型初始化与每方法一次 warmup，包含特征提取、匹配、CPU RANSAC/接受门限，LightGlue 包含张量传输和前后 CUDA 同步。传统方法为 CPU，LightGlue 在 RTX4090 GPU；共享服务器负载随时间变化，这些耗时不能解释为统一硬件的纯算法速度排名。\n\n![四方法 Success@5，分母均为 860 条正样本；包含所有失败](../'+os.path.relpath(art,base)+'/four_method_success.png)')
add(table(['方法','成功@5','错误接受','没有 H 输出','估计被拒绝','无重叠无输出 / 拒绝'],[[names[m]]+[summary['by_method'][m]['decision_counts'].get(k,0) for k in ['success','incorrect_accept','no_output','rejected']]+[f"{summary['by_method'][m]['negative_no_output_count']} / {summary['by_method'][m]['negative_rejected_estimate_count']}"] for m in methods]))
add('## 难度、场景及重叠分层\n\n以下为 Success@5px；N 为正样本数。无重叠只在总体表和独立负样本统计中评价。')
for field,title in [('profile','难度'),('source_scene','原始来源场景'),('overlap_bin','双向有效重叠率较小值'),('has_ship','端点视图是否存在舰船标注'),('dataset','来源数据集'),('scene','正式设计场景层')]:
 key='by_method_and_'+field if field not in ('has_ship','overlap_bin') else ('by_method_and_ship' if field=='has_ship' else 'by_method_and_overlap_bin')
 vals=sorted({r[field] for r in rows});data=[]
 for v in vals:
  stats=summary[key][f'sift|{v}'];data.append([v,stats['positive'],stats['negative']]+[pct(summary[key][f'{m}|{v}']['success_5px_rate']) for m in methods])
 add(f'### {title}\n\n'+table(['层','正 N','负 N']+methods,data))
add('场景来源标签与正式设计场景层分别报告：困难层包含 `any` 90 条关系，不能将其当成已知近岸或离岸；原始场景还有 7 条 unknown 正样本，保留未知状态。`has_ship=False` 包含端点没有舰船的图像，不能一律等同于 HRSID background 来源。动态 40 条是独立 profile，不是把所有有船图像视为动态。\n\n本固定代理集上 ORB 总体最好，动态条件下为 70%，LightGlue 为 52.5%；但低重叠 40 对 SIFT 为 25%，LightGlue 为 10%，ORB 为 2.5%，不存在统一优胜的方法。LightGlue 背景层成功率约 90.6%，同时出现 1 条纯背景无重叠错误接受。不能用总体平均掩盖困难层失败。')
add('## 失败原因与几何核查\n\n'+table(['方法','记录的估计/拒绝原因'],[[names[m],'; '.join(f'{k}: {v}' for k,v in summary['by_method'][m]['reject_reasons'].items())] for m in methods]))
add(f'LightGlue 最大误差样本 `{worst["id"]}` 属 dynamic / offshore：15 个匹配、13 个内点，拟合 RMSE={float(worst["inlier_rmse"]):.3f}px、覆盖率={float(worst["coverage"]):.4f}，但静态 GT 平均重投影误差={float(worst["epe_mean"]):.3f}px。保存 H 的评价重算一致，故不是 H 方向或尺寸换算错误。预测投影分母在图像四角范围为 [{corner_z.min():.6f}, {corner_z.max():.6f}]，跨过零；GT 点最小绝对分母为 {np.min(np.abs(gt_z)):.8g}，说明图内投影极点导致误差暴涨。现有局部拟合门限没有排除这种整幅不稳定 Homography。动态舰船主导或局部退化可能解释错匹配来源，需要开发集进一步检验，不能仅凭此结果确认为唯一原因。极端有限误差保留在均值中，不截断或删除。\n\n无重叠误接受 `pair_000792:0-1` 是 background、192px：20 匹配、9 内点、内点率 0.45、拟合 RMSE 1.421px、覆盖率 0.3125，满足既有门限但不存在真实可配准重叠。未据此修改正式集阈值。三种传统方法观察到 0/40，不等于真实未知分布错误接受率为零。')
add('## 三视图循环一致性\n\n每方法保留全部 20 个三视图组，包括未完成三边估计的组。独立估计 0→1、1→2、0→2，计算 `inv(H02) @ H12 @ H01` 在原生整幅 5×5 网格的循环误差；它与 GT 对应点误差的采样范围不同。')
cycles=summary['three_view_cycles']['entries'];data=[]
for m in methods:
 c=[x for x in cycles if x['method']==m];assert len(c)==20
 complete=[x for x in c if x['complete_accepted']];values=[x['cycle_mean_px'] for x in complete if x['cycle_mean_px'] is not None]
 data.append([names[m],len(complete),20-len(complete),sum(x['all_edges_gt_success_5px'] for x in complete),num(float(np.median(values))) if values else 'N/A',num(float(np.max(values))) if values else 'N/A'])
add(table(['方法','三边均接受 /20','不完整 /20','三边均 GT 成功@5','闭环均值的中位 px','最大闭环均值 px'],data))
add('Phase 唯一完整三角形的闭环误差约 0.785px，但三边未全部满足 GT Success@5；自洽不等于正确。LightGlue 也存在三边 GT 成功但整幅循环外推误差较大的组，说明重叠区域精度与整幅几何稳定性需要分开解释。')
add('## 来源相关性与不确定性\n\n'+table(['方法','正样本组件数','配对加权 Success@5','组件等权 Success@5','描述性组件 bootstrap 95% 区间'],[[names[m],summary['cluster_bootstrap'][m]['source_components'],pct(summary['cluster_bootstrap'][m]['point_estimate_pair_weighted']),pct(summary['cluster_bootstrap'][m]['point_estimate_component_weighted']),' – '.join(pct(x) for x in summary['cluster_bootstrap'][m]['ci95_cluster_bootstrap'])] for m in methods]))
add('组件 bootstrap 1,000 次，seed=42。HRSID 背景与对应 HRSID 父场景共享命名空间，避免伪造独立组件。背景只来自 2 个父场景，且未知来源关联可能残留，区间为描述性探索结果，不做独立配对显著性检验，也不能推广到独立真实观测。')
add('## 代表性图与检查方式\n\n图中原图使用相同 0–255 灰度范围，未增强或裁去失败区域；叠加图红色为 view0、青色为 view1 按预测 H 的逆变换回 view0。GT warp 仅为评价参照，不进入任何算法。缺少已接受估计时明确显示空格和状态；无重叠不显示伪 GT 配准。案例按写明的条件取固定顺序首例，另保留最大误差例，属于事后诊断展示。\n\n![七组代表案例：近岸、低重叠、背景、无重叠误接受、动态极端误配、拒绝和三视图](../'+os.path.relpath(art,base)+'/representative_cases.png)\n\n'+table(['案例条件','配对 ID'],chosen))
add('## 质量验收与代码修改\n\n- 900 条关系 / 860 组完整读取；每方法 900 个相同且唯一 ID；合并检查冻结 receipt、manifest、source index、结果 SHA 和共享参数一致。\n- 3,600 行预测的状态、Success@3/5/10 与 EPE 全部用保存 H 和 GT 重算一致；最大差异 '+num(max_metric_difference)+'px。之前全对 GT 方向检查最大误差 2.149e-5px。\n- 第一阶段 11,093 个冻结文件在运行前后 SHA256 校验通过，没有修改、覆盖或重新生成。\n- 15 项软件测试通过，包含坐标、状态、元数据、三视图、覆盖保护与共享协议不一致拒绝合并；pip check 无依赖问题；仅一个 OpenCV 发行包。\n- 修复 authoritative source_group 到 component 映射、background 与父场景聚合、按端点视图统计有船、关键点计数、候选 H 与四种状态、完整三角形记录和整幅循环网格、计时同步及环境/权重来源记录。\n- 增加共享协议合并校验，GPU 模型种子与默认配置记录；README/quickstart 统一 match 环境及 Phase2 输出目录；提供固定版本安装与新目录复现脚本。未按 test 成绩修改算法阈值。')
add('## 环境、模型版本与复现\n\n'+table(['项目','实际版本'],[['Python',sys.version.split()[0]],['环境',str(Path(sys.executable).parent.parent)],['NumPy',packages['numpy']],['OpenCV headless',packages['opencv-python-headless']],['PyTorch',packages['torch']],['torchvision',packages['torchvision']],['Kornia',packages['kornia']],['LightGlue 官方提交',lgmeta['source_commit']],['传统评测代码',provs[0]['code_commit']],['LightGlue 评测代码',lg['code_commit']],['报告/交付代码',subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()]]))
add('GPU：物理卡 7，`GPU-06d4dc94-7d75-0580-5608-12ccfa27496a`，RTX4090；驱动 535.288.01；Torch CUDA build=12.1。完整 GPU inventory 与精度设置保存在环境 JSON 和原运行 provenance。原始 LightGlue 进程的 importlib.metadata 枚举包含 Torch 引入的 setuptools 内置副本；实际安装版本以本次独立 pip list/freeze 的环境快照为准，后续 provenance 已改用该方式。官方网络文件未修改；唯一上游差异是 requirements 中 `opencv-python` 改为 `opencv-python-headless`，避免共享 cv2 模块冲突，补丁已保存。\n\n权重 SHA256：\n\n'+table(['官方缓存文件','SHA256'],list(weights.items())))
add(f'完整依赖见 [{lock.name}]({lock.name}) 与 [{freeze.name}]({freeze.name})；模型补丁见 [{patch.name}]({patch.name})。用户手动完成 Torch 下载；执行日志保留此前网络失败，不将失败安装当作成功。\n\n在已有 match 环境直接读取结果无需再次运行算法：\n\n```bash\ncd {base}\nconda run -n match python -c \'import csv; from pathlib import Path; p=Path("{out}/results.csv"); rows=list(csv.DictReader(p.open())); print(len(rows), sorted({{r["method"] for r in rows}}))\'\n```\n\n完整复现（会执行全部四方法；本次没有再次运行已完成传统方法）：\n\n```bash\ncd {base}\n# 新服务器先创建 match；当前服务器已有该环境，不必重复创建。\n# conda create -n match python=3.10.22 -y\nconda run -n match python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r reports/{lock.name}\nconda run -n match python -m pip install --no-deps -e .\nbash scripts/install_official_lightglue.sh\nconda run -n match python -m pytest -q\nSAR_GPU=7 bash scripts/reproduce_phase2.sh\n```\n\n脚本创建带实际时间戳的新目录，已有路径直接失败，不覆盖旧结果。实际历史 run 命令保存在日志及各自 settings.json。生成图表和报告可从已有比较结果独立执行 `conda run -n match python scripts/finalize_phase2.py --comparison <比较目录> --log <执行日志>`，其 delivery_figures 也拒绝覆盖。')
add('## 残余风险与下一步研究建议\n\n1. 本结果仅验证二维合成 SAR 配准代理任务。共享源散斑、边界与几何生成过程可能产生捷径；缺少真实跨观测角的散射变化、叠掩、阴影和非平面效应验证。必须用独立真实多视角观测检验外推能力。\n2. 本代理集 ORB 足以作为强基线；LightGlue 特征点均值明显更少、动态与离岸困难层弱于 ORB，先在 train/val 独立开发集分析检测覆盖、关键点空间分布、动态响应与几何退化，再决定是否需要 SAR 特征适配。当前证据支持调查域差异，不能直接证明微调一定有效。\n3. 优先在独立开发集研究预测侧几何稳定性、支持范围与无重叠拒绝，避免局部小残差的错误 Homography 外推；不得在这 900 对上挑选新 gate 阈值。若比较 affine 或新特征，另立实验并明确变换模型差异。\n4. 只有当独立 SAR 开发集与真实多视角测试显示预训练特征限制，才考虑受控 SAR 域微调，保留冻结主测试集并按源场景隔离；本次没有训练。\n5. 动态区域只在静态 GT 评价中排除，并非算法知道舰船/动态掩膜；13 个组既有来源标注边界差异仍是检测/融合研究风险。背景父场景仅 2 个、40 条负样本及困难层样本量有限，不能将本表接受率直接当作真实部署可靠性。\n6. GPU 共享负载和未强制逐位确定性限制运行时间与微小数值的复现；需依赖版本、种子、原生坐标、模型/权重 SHA 和协议联合复现，不保证跨平台逐位一致。\n\n官方模型接口和默认配置依据 [LightGlue 官方仓库](https://github.com/cvg/LightGlue/tree/eb42fee2d71449efb0aa5c10549752b5d75384d8)。上述 SAR 研究路线是基于本次分层结果的推断，不是已验证的微调收益。')
report.write_text('\n\n'.join(text)+'\n',encoding='utf8')
# Caller copies the completed historical log after run_logged has written its exit line.
save_json(reports/f'phase2_delivery_{stamp}.json',{'report':str(report),'execution_log':str(log_final),'historical_log':str(Path(a.log).resolve()),'quality':str(quality_path),'environment':str(env_path),'requirements_lock':str(lock),'comparison':str(out),'figure':str(art/'representative_cases.png'),'accepted':True})
print(json.dumps({'report':str(report),'execution_log':str(log_final),'quality':str(quality_path),'comparison':str(out),'rows':len(rows),'phase1_unchanged_files':checked},ensure_ascii=False,indent=2))
