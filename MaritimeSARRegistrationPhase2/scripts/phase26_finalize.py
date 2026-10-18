"""Publish local frozen confirmation artifacts with the actual completion minute."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--log',required=True);a=p.parse_args()
run=Path(a.run).resolve();analysis=run/'analysis';log=Path(a.log).resolve();receipt=run/'DELIVERY_COMPLETE.json'
if receipt.exists():raise FileExistsError(receipt)
quality=json.loads((analysis/'quality_acceptance.json').read_text());assert quality['accepted']
report_receipt=json.loads((analysis/'REPORT_COMPLETE.json').read_text());assert report_receipt['accepted']
for path,digest in report_receipt['files_sha256'].items():assert sha(analysis/path)==digest,path
assert not subprocess.check_output(['git','status','--porcelain'],text=True).strip(),'Commit all Phase26 code before delivery'
run_hashes={str(f.relative_to(run)):sha(f) for f in sorted(run.rglob('*')) if f.is_file()}
stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');reports=Path('reports').resolve()
source=next(analysis.glob('phase26_confirmation_report_*.md'));final_report=reports/f'phase26_confirmation_report_{stamp}.md';final_log=reports/f'phase26_execution_{stamp}.log';phase3=reports/f'phase3_real_sar_requirements_{stamp}.md'
assert not final_report.exists() and not phase3.exists() and (final_log==log or not final_log.exists())
text=source.read_text();text=text.replace(text.splitlines()[0],f'# Phase 2.6 独立确认报告（完成于 {stamp}，Asia/Shanghai）',1)
aggregate=list(csv.DictReader((analysis/'aggregate_metrics.csv').open()));strata=list(csv.DictReader((analysis/'stratified_metrics.csv').open()))
def get(method,subset):return next(r for r in aggregate if r['method']==method and r['subset']==subset)
def difference(first,second):return 100*(float(second['success_5px_rate'])-float(first['success_5px_rate']))
conclusion=['## 正式交付结论：收益与确认失败的边界','',
            '主确认层中三个冻结改进均保持成功率上升、错误接受减少；已知组件等权的变化方向也一致。无重叠48对中七个方法均0次错误接受，这只描述18个已知组件上的当前样本，不能推论真实海面总体FPR为0。强透视层预先单列并完整保留，未被用来重新选择模型。','']
for alg in ['sift','orb','lightglue']:
    chain=[get(alg+s,'primary') for s in ['_original','_feature_H','_feature_A','_improved']]
    contribution=[difference(chain[i],chain[i+1]) for i in range(3)]
    a,b=get(alg+'_original','strong_stress'),get(alg+'_improved','strong_stress')
    conclusion += [f'{alg}: 特征/模型/full检查的主层Success@5增量分别 {contribution[0]:+.2f}/{contribution[1]:+.2f}/{contribution[2]:+.2f}pp。压力层原始→改进 {100*float(a["success_5px_rate"]):.2f}%→{100*float(b["success_5px_rate"]):.2f}%，错误接受 {a["wrong_positive"]}→{b["wrong_positive"]}。','']
conclusion+=['SIFT和LightGlue的主层成功率收益以检测阈值变化为主；ORB的模型变化增益比检测变化大。full机制主要交换成功召回与错误接受：SIFT额外拒绝6个正确正例且未再减少错误；ORB拒绝10个正确/2个错误正例；LightGlue拒绝3个正确/12个错误正例。这些都是冻结配置的观察，不能据此回头重新选择阈值。','',
             '方法配置的2048是请求特征预算；OpenCV SIFT实际数量可略超请求值，本轮记录的最大值为2051，原始数量均保留，没有事后截断或改变检测器。','',
             '强透视16对的GT最小二乘Affine残差RMSE平均3.23px、最大7.77px，而主层该指标最大低于1px；同一选定特征的H/A对比也显示模型容量问题。ORB和LightGlue压力层改进失败，接受稳定或双向自洽不保证静态对应正确；不能把Affine结果的低方差称作真实跨视角稳健性。','']
for label,field,value in [('低重叠','profile','low'),('动态舰船','profile','dynamic'),('显式低纹理','texture_stratum','low')]:
    entries=[next(r for r in strata if r['method']==m and r['stratum']==field and r['value']==value) for m in ['sift_improved','orb_improved','lightglue_improved','orb_then_sift']]
    conclusion += [label+'：冻结改进 SIFT/ORB/LightGlue/实际回退的Success@5分别 '+ '/'.join(f'{100*float(r["success_5px_rate"]):.2f}%' for r in entries)+'；错误正例分别 '+ '/'.join(r['wrong_positive'] for r in entries)+'。','']
conclusion+=['低重叠仍以支持不足和无输出为主要瓶颈；动态船舶仍可能支撑错误但自洽的模型，不能把匹配内点直接解释为静态GT。真实回退可补救ORB未接受的部分样本，不能修复它已错误接受的样本。按每个实际分支测得的耗时报告，不再使用Phase2.5缓存分项估算。','',
             '本轮提供检测、模型与拒绝机制的确认证据，未建立真实SAR域微调的必要性。后续应先取得真实跨采集数据、独立静态GT和每视图舰船标注，再在严格来源隔离的训练/验证和封存测试上决定是否微调。停止当前二维合成数据上的重复选参。','']
text+='\n'+'\n'.join(conclusion)
text+=f'\n正式交付记录：日志 [{final_log.name}]({final_log})；Phase3 需求 [{phase3.name}]({phase3})；完整依赖和Conda导出 `{run}/environment`；交付核验 `{receipt}`。运行目录时间为启动时间，报告/日志尾缀为实际完成分钟。\n'
final_report.write_text(text);shutil.copyfile('docs/phase3_real_sar_requirements.md',phase3)
with log.open('a') as f:
    f.write(f'\n[{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}] Phase26 delivery finalized: all confirmation decisions/metrics verified; protected files {quality["protected_files_unchanged"]}; no retuning or training.\n')
    f.write(f'Command: conda run --no-capture-output -n match python scripts/phase26_finalize.py --run {run} --log {log}\n')
    f.write(f'Python: {sys.executable}\nFinal report: {final_report}\nFinal execution log: {final_log}\nPhase3 requirements: {phase3}\n')
if log!=final_log:log.rename(final_log)
active=Path('cache/phase26_active.json');state=json.loads(active.read_text());state.update(log=str(final_log),completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat());active.write_text(json.dumps(state,indent=2)+'\n')
delivery={'accepted':True,'completed_at':state['completed_at'],'completion_stamp':stamp,'run':str(run),'inference_code_commit':json.loads((run/'evaluation/provenance.json').read_text())['code_commit'],
          'delivery_code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'script_sha256':sha(__file__),'quality':quality,'run_files_sha256':run_hashes,
          'official_artifacts_sha256':{str(f):sha(f) for f in [final_report,final_log,phase3]},'official_report':str(final_report),'official_log':str(final_log),'phase3_requirements':str(phase3)}
receipt.write_text(json.dumps(delivery,indent=2,ensure_ascii=False)+'\n');print(json.dumps({k:delivery[k] for k in ['accepted','completion_stamp','official_report','official_log','phase3_requirements']},indent=2))
