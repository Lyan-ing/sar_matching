"""Publish a timestamped local delivery from completed, immutable experiments.

Copy the execution-log snapshot only after the enclosing run_logged.py exits.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from zoneinfo import ZoneInfo
import cv2
from sarbench.reports import read_csv, save_json, _float


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(x) for x in r)+' |' for r in rows])
def fmt(v): return 'NA' if v is None else f'{float(v):.2f}'


p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--log',required=True)
a=p.parse_args();run=Path(a.run).resolve();analysis=run/'analysis';attribution=run/'attribution'
receipt=run/'DELIVERY_COMPLETE.json'
if receipt.exists(): raise FileExistsError(receipt)
for folder, name in [(analysis,'REPORT_COMPLETE.json'),(attribution,'ATTRIBUTION_COMPLETE.json')]:
    complete=json.loads((folder/name).read_text());assert complete['accepted']
    for path,digest in complete['files'].items(): assert sha(folder/path)==digest
quality=json.loads((analysis/'quality_acceptance.json').read_text());assert quality['accepted']
replay=json.loads((run/'replay_verification.json').read_text());assert replay['accepted'] and replay['groups']==10
freeze=json.loads((run/'generation_code_sha256.json').read_text())
assert all(sha(path)==digest for path,digest in freeze.items())
data=run/'data_v2';prov=json.loads((data/'provenance.json').read_text())
config=prov['config']; index=Path(config['source_index']);audit=Path(config['source_audit'])
assert sha(index)==prov['source_index_sha256'];assert sha(audit)==prov['source_audit_sha256']
assert sha(config['phase1_recipe'])==prov['phase1_recipe_sha256']
assert not (data/'confirmation').exists()
source_files=0
for split in ['dev','val']:
    for path,digest in json.loads((data/split/'source_decoded_hashes.json').read_text()).items():
        im=cv2.imread(path,0);assert im is not None
        assert hashlib.sha256(str(im.shape).encode()+im.tobytes()).hexdigest()==digest
        source_files+=1
for path,digest in json.loads((run/'preserved_artifacts.json').read_text())['files'].items(): assert sha(path)==digest
model=json.loads((run/'model_freeze.json').read_text())
assert subprocess.check_output(['git','-C','cache/LightGlue','rev-parse','HEAD'],text=True).strip()==model['lightglue_commit']
assert subprocess.check_output(['git','-C','cache/LightGlue','diff','HEAD'],text=True)==model['lightglue_diff']
for name,digest in model['weights_sha256'].items(): assert sha(Path('cache/torch/hub/checkpoints')/name)==digest
quality['checks'].update(fixed_seed_replay_groups_byte_identical=10,selected_source_native_images_rehashed=source_files,
                         source_index_audit_generation_code_weights_unchanged=True,confirmation_images_and_predictions_not_generated=True,
                         posthoc_selected_dynamic_inlier_refits_same_candidate=120)
stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M'); reports=Path('reports').resolve()
paths={name:reports/f'phase25_{name}_{stamp}{suffix}' for name,suffix in [('report','.md'),('execution','.log'),('quality_acceptance','.json'),('requirements_lock','.txt'),('environment','.json')]}
assert all(not path.exists() for path in paths.values())
rows=read_csv(analysis/'validation_primary.csv'); dynamic=read_csv(attribution/'dynamic_attribution_summary.csv')
text=next(analysis.glob('phase25_report_*.md')).read_text()
text=re.sub(r'# Phase 2.5 配准可靠性与失败机制报告（\d{8}_\d{4}）',f'# Phase 2.5 配准可靠性与失败机制报告（{stamp}）',text,count=1)
text=text.replace('512 个独立静态 GT 对应点','512 个预先生成的静态 GT 对应点')
text=text.replace('可复现 requirements、Conda 导出另存报告目录。',f'实际 requirements 锁文件：`{paths["requirements_lock"]}`；完整 Conda 导出在 `{run/"environment/conda_environment.yml"}`，环境摘要为 `{paths["environment"]}`。')
text=text.replace('](validation_comparison.png)',f']({analysis/"validation_comparison.png"})').replace('](accepted_error_cdf.png)',f']({analysis/"accepted_error_cdf.png"})')
counts=table(['方法','KP0均值','KP1均值','匹配均值','内点均值'],
             [[r['label']]+[fmt(r[k]) for k in ['avg_keypoints0','avg_keypoints1','avg_matches','avg_inliers']] for r in rows])
text=text.replace('## 特征检测与匹配失败证据\n','## 特征检测与匹配失败证据\n\n'+counts+'\n\n内点均值包括所有 384 对（无输出按 0），同一特征在 H/A 下可能得到不同内点集合。\n')
dynamic_table=table(['选定方法','动态状态','n','动态区域内点占比 %','GT正确静态内点均值'],
                    [[r['method'],r['status'],r['pairs'],fmt(100*float(r['dynamic_inlier_fraction_mean'])) if r['dynamic_inlier_fraction_mean'] else 'NA',fmt(r['GT_correct_static_inliers3_mean'])] for r in dynamic if r['status'] in ['success','incorrect_accept']])
addendum='''## 动态舰船错误接受机制（离线归因）

仅从缓存点恢复 120 个验证动态样本的原 RANSAC 内点位：相同图像尺寸、seed、模型和点，候选矩阵与已有输出全部一致。恢复后才用 GT Mask 和静态 H 标记动态/正确静态内点；未改参数、模型、接受状态，未运行新网络。

'''+dynamic_table+'''

错误接受组的动态内点占比高于成功组的描述均值；动态组规模较小且共享来源，这不是因果或显著性检验。该结果支持“局部目标运动可形成自洽模型，但不代表静态背景配准”的失败机制。

具体 `pair_000362:0-1`：原 LightGlue H 接受，静态 EPE=1417.05px，原 H 的 full 检查因极点拒绝；选定 LightGlue Affine 有 23 匹配/12 内点，7/12（58.33%）内点落在动态区域，正确静态内点=0。双向循环均值3.06px、对称RMSE1.37px、两侧凸包3.20%/2.22%，均满足封存门控，却仍被接受，静态EPE=23.33px。循环与分布检查只能排除部分错误，不能证明语义正确；不得按 GT 动态区域在预测时过滤。

'''+f'[动态内点支持图]({attribution/"dynamic_inlier_support.png"})；原图范围0–255，无增强。白色轮廓仅用于事后说明动态区。\n\n'+'''
另保存全部 4,992 个特征缓存的有效区/合成边界与动态支持占比于 `attribution/boundary_dynamic_feature_attribution.csv`。单像素有效区边界邻域的关键点占比不是“利用合成捷径”的证明；仍需真实 SAR 和替代生成过程验证。

'''
text=text.replace('## Affine 容量与 ORB/SIFT 互补',addendum+'## Affine 容量与 ORB/SIFT 互补')
extra=f'\n\n追加复现核查：dev/val 每个 profile 各重放 1 组，共 10 组/120 个文件逐字节一致，旧文件未被替换。{source_files} 个已选原生来源图像重新解码哈希一致；源索引、审计、生成代码和官方权重哈希保持不变。详见 `replay_verification.json` 和最终验收 JSON。\n'
text+=extra
text+=f'\n最终执行日志：`{paths["execution"]}`；完整验收：`{paths["quality_acceptance"]}`。报告与日志尾缀采用实际交付分钟 `{stamp}`，历史运行目录及日志标识不改写。\n'
text+=f'\n报告/离线归因代码版本：`{subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}`；推理代码仍按 dev 封存 SHA 校验。\n'
text+='\n复现离线归因：`conda run --no-capture-output -n match python scripts/phase25_attribution.py --run "$RUN" --out "$RUN/attribution"`；本地最终交付：`conda run --no-capture-output -n match python scripts/finalize_phase25.py --run "$RUN" --log <完整执行日志>`（需先保存模型/生成代码/旧成果保护快照及重放验收；只复现数据和算法可直接使用前面的命令）。日志快照须在外层 logger 写完退出行后复制。\n'
paths['report'].write_text(text,encoding='utf8');save_json(paths['quality_acceptance'],quality)
shutil.copyfile(run/'environment/requirements_freeze.txt',paths['requirements_lock'])
save_json(paths['environment'],{'actual_packages':json.loads((run/'environment/pip_list.json').read_text()),
                              'conda_environment':str(run/'environment/conda_environment.yml'),
                              'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                              'inference_code_commit':'280643888e3f59e5ab34c53a9f6a31d6a31c0f77','model':model})
save_json(receipt,{'accepted':True,'created_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'timestamp':stamp,
                  'paths':{k:str(v) for k,v in paths.items()},'source_log':str(Path(a.log).resolve()),
                  'files_sha256':{str(v):sha(v) for k,v in paths.items() if k!='execution'},
                  'log_snapshot_pending':True})
print(json.dumps({'delivered':str(receipt),'paths':{k:str(v) for k,v in paths.items()},'quality':quality['checks']},indent=2))
