"""Seal actual completed evidence, preserve prior report versions and handoff snapshots."""
from report import *
import hashlib

def main():
 ts=stamp();old=latest('delivery_final');source=Path(old['markdown']);body=source.read_text()
 prior='A/B模型可恢复和加载；独立的CLI GPU冒烟执行请求被拒绝，未运行该额外测试。预检查及正式评价已实际完成A/B权重加载、LG调用、纯图像预测和训练后分块等价检查；此事实不能改写成CLI独立测试已通过。'
 results=[]
 for v in ['a','b']:
  folder=max(ROOT.glob(f'cli_smoke_{v}_*'),key=lambda p:p.stat().st_mtime_ns);p=max(folder.glob('prediction_*.json'),key=lambda p:p.stat().st_mtime_ns);r=json.loads(p.read_text());assert r['image_only'] and r['process']['physical_gpu']==5;assert r['checkpoint_sha']==sha(ROOT/'runs'/v/'best.ckpt');assert (folder/f'{v}_{RUNSTAMP}.npz').exists();results.append(dict(method=v,folder=str(folder),process=r['process'],n_keypoints0=r['n_keypoints0'],n_keypoints1=r['n_keypoints1'],n_matches=r['n_matches'],n_inliers=r['n_inliers'],decision=r['decision'],elapsed_ms=r['elapsed_ms'],exit_code=0))
 text='A/B模型可恢复和加载。初次独立CLI GPU执行请求曾被拒绝；用户随后明确重新授权八卡及再次CLI测试。重新查实时余量，物理GPU5剩余显存最大，在match环境完成A/B两条独立纯图像CLI实际推理，均退出码0；权重SHA与锁定模型一致，成功调用冻结LG，保存点/内点/候选H。此前报告和日志不覆盖。单次CLI含模型首次推理暖机，耗时不能与已暖机批量均时混为同一协议。CLI自然点数与正式等点数对照分开记录。'
 assert prior in body;body=body.replace(prior,text).replace('正式训练和评价物理GPU1','正式训练及合成评价物理GPU1；严格等点数真实补充与独立CLI实时选用物理GPU5')
 body+='\n## 最终独立CLI测试记录\n\n'+table(['模型','物理GPU','点数左/右','匹配','内点','工程状态','退出码'],[[r['method'],5,f"{r['n_keypoints0']}/{r['n_keypoints1']}",r['n_matches'],r['n_inliers'],r['decision'],0] for r in results])+'\n\n所有训练和评价已结束，没有后续自动训练或重复benchmark。环境、配置、来源、源码和权重身份保留；原始报告版本不覆盖。\n'
 md=ROOT/f'structure_stable_report_{ts}.md';hp=md.with_suffix('.html');assert not md.exists() and not hp.exists();body=re.sub(r'生成时间：[^。]+。','生成时间：'+now()+'。',body,count=1);md.write_text(body)
 renderer=(BASE/'SAR_SP_LG_TrainingReview_20261012_1651/code/export_pdf.py').read_text();scope={'html':html,'re':re};exec(renderer[renderer.index('def inline'):renderer.index('def main')],scope)
 hp.write_text('<!doctype html><meta charset="utf-8"><style>body{max-width:1500px;margin:30px auto;padding:20px;font:16px sans-serif}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:6px}pre{white-space:pre-wrap}</style>'+scope['render'](body))
 rows=latest('validation_results')['rows'];real=latest('real_equal_results')['rows'];native=latest('native_validation')['rows'];neural=['m0','m2','a','b','control'];checks={}
 checks['validation_1764_rows']=len(rows)==1764;checks['real_72_rows']=len(real)==72;checks['native_1470_rows']=len(native)==1470;checks['real_no_exceptions']=not any(r['decision']=='exception' for r in real)
 checks['exact_equal_point_counts']=all(len({r[f'n_keypoints{i}'] for r in rows if r['id']==id and r['method'] in neural})==1 for id in {r['id'] for r in rows} for i in [0,1])
 checks['real_exact_equal_counts']=all(len({r[f'n_keypoints{i}'] for r in real if r['id']==id})==1 for id in {r['id'] for r in real} for i in [0,1])
 checks['frozen_training_source']=all(sha(ROOT/k)==v for k,v in json.loads(output('code_identity').read_text())['files'].items())
 for v in ['a','b','control']:
  lock=json.loads(output('selection_locked',folder=ROOT/'runs'/v).read_text());checks[v+'_best_sha']=sha(ROOT/'runs'/v/'best.ckpt')==lock['best_sha256']
 checks['m2_old_unchanged']=sha(FIX/'runs/m2/best.ckpt')=='69e671f743adaf134983a3c01c3f9f1a152c387f8a999218b089b422799ac1ef'
 checks['m3_old_unchanged']=sha(M3ROOT/'runs/m3/best.ckpt')=='6983c3e1c5d419cb76eaec9712b66d36be80b0b120c5a21558f58679970d99e9'
 figures=re.findall(r'!\[[^]]*\]\(([^)]+)\)',body);checks['all_report_images_exist']=all((ROOT/f).exists() for f in figures);checks['CLI_both_exit0']=all(r['exit_code']==0 for r in results);assert all(checks.values()),checks
 files=list((ROOT/'code').glob('*.py'))+[md,hp]+list((ROOT/'tables').glob('*.csv'))
 evidence=event('delivery_verified',dict(markdown=str(md),html=str(hp),report_images=len(figures),checks=checks,CLI=results,file_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},engineering_complete=True,scientific_recommendation='retain M2',B_scientific_gate_pass=False,no_new_training=True))
 # Update live shared handoff under the existing lock; preserve exact previous content as a timestamped snapshot.
 with open('/tmp/sar_project_handoff.lock','a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);path=BASE/'PROJECT_HANDOFF.md';previous=path.read_text();snapshot=ROOT/f'project_handoff_before_completion_{ts}.md';assert not snapshot.exists();snapshot.write_text(previous)
  title='## 三尺度结构增强与稳定性联合学习已完成'
  section=f'''\n\n{title}\n\n更新{now()}。目录`{ROOT.name}`；A/B完成3000步，best各3000；M2继续训练对照实际2000步早停，best500。正式训练/评价GPU1，最后独立CLI验证实时选择GPU5，A/B退出码均0。三模型均锁定，所有任务完成；不要重训或重新执行这些评价，GPU占用必须实时查。\n\n179选模/115内部留出，均复用已观察M3开发像素，组件隔离；全新Val组件0。严格共同实际点数的270正例S@5 M0/M2/A/B/对照=61.11/65.56/67.78/73.70/67.04%；内部留出99正例=54.55/62.63/63.64/65.66/62.63%。B同预算匹配/重复性有探索性收益、空白0点，但native平均557.48点(约M2的5.77倍)、噪声1010点，未过预注册密度门槛。A留出仅净增1成功且正确静态匹配均值下降；最终建议继续保留M2，不声明真实异向精度改善。\n\n1764等点数验证行、1470自然预算诊断行、72严格等点数真实Native行（18×4，无异常），另保留原72自然上限诊断行；发现旧真实结果实际点数不一致后新增固定模型对照，未重选权重；四模型仍仅接受2同向对照，独立物理GT0；ORB自然1024上限S@5=92.22%，不能仅引用约96点预算的低ORB成绩。B step873全动态掩膜空集合KL NaN，修正后从step800恢复；源文件/失败日志/checkpoint备份、数值回归及native密度门控修正记录保留，未改损失超参数。旧900及旧权重不修改、不重跑。\n\n最终报告`{md.relative_to(BASE)}`、同名HTML；证据`{evidence.relative_to(BASE)}`；CLI`{ROOT.name}/code/infer_pair.py`，指令见README；locked权重`runs/a,b,control/best.ckpt`，禁止覆盖。下一步仅建议改善结构/噪声判别与响应校准，未启动其他网络训练。源码原版code_identity和修正版implementation_revision记录并存；全部Python使用match，依赖没有升级。\n'''
  assert title not in previous;tmp=path.with_name('PROJECT_HANDOFF.md.tmp.structure_complete');tmp.write_text(previous+section);os.replace(tmp,path)
 print(json.dumps(dict(report=str(md),html=str(hp),verification=str(evidence),checks=checks,CLI=results),ensure_ascii=False))
if __name__=='__main__':main()
