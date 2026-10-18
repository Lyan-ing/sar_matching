"""Post-prediction diagnostics and descriptive reports; never alters gate decisions."""
from mixed_io import *
from collections import Counter,defaultdict
import argparse

def read_scores(tag):
 p=ROOT/'predictions'/tag/f'scores_{TAG}.jsonl';return [json.loads(s) for s in p.read_text().splitlines() if s]
def static_correct(row,pred):
 with np.load(row['gt_path']) as z:
  a,b=pred['matches_A'],pred['matches_B'];sa=z['static_A']&~z['ignore_A']&z['valid_A'];sb=z['static_B']&~z['ignore_B']&z['valid_B'];q=dc.transform_points(z['H_AB'],a);p=dc.transform_points(z['H_BA'],b)
  return dc.bilinear_at(sa,a)&dc.bilinear_at(sb,b)&dc.bilinear_at(sb,q)&(np.linalg.norm(q-b,axis=1)<=3)&(np.linalg.norm(p-a,axis=1)<=3)
def diagnose(rr):
 out=[]
 for r in rr:
  if r['no_overlap']:stage='无重叠负例'
  elif not r['structure_anchor_count']:stage='无保存结构锚点：无法仅凭结构指标归因检测失败'
  elif not r['visible_detected_structure_points']:stage='可评价静态区未检测到点'
  elif r['repeatable_structure_points']<8:stage='重复静态检测少于8'
  elif r['correct_structure_matches']<8:stage='重复点存在但正确静态匹配少于8'
  elif r['state']!='accepted':stage='正确匹配存在但几何估计或门控拒绝'
  elif not r['Success@5px']:stage='接受但全域误差超5px'
  else:stage='全域5px成功'
  out.append(dict(r,diagnostic_stage=stage))
 return out

def report_baseline():
 rr=diagnose(read_scores('baseline'));assert len(rr)==12000 and len(set(r['pair_id'] for r in rr))==1200
 t=stamp();csvwrite(ROOT/f'baseline_failure_stages_{t}.csv',rr)
 grouped={f'{k}/{mode}':scorer.summaries([r for r in rr if r['method']==k and r['protocol']==mode]) for k in ['m0','m2','b','m2v22','orb'] for mode in ['natural','equal']}
 write(ROOT/f'baseline_diagnostics_{t}.json',dict(summary=grouped,stages={key:dict(Counter(r['diagnostic_stage'] for r in rr if r['method']==key.split('/')[0] and r['protocol']==key.split('/')[1])) for key in grouped}))
 txt=['# 冻结合成验证集：原生640/512统一基线','',f'生成时间：{now()}。完整1200对、1140几何正例、60无重叠负例；为开发验证，非独立测试。','', '|方法|预算|S@3|S@5|S@10|结构S@5|无重叠误接受|结构重复性|', '|---|---|---:|---:|---:|---:|---:|---:|']
 def fmt(x):return 'NA' if x is None else f'{100*x:.2f}%'
 for key,v in grouped.items():
  name,mode=key.split('/');z=v['all'];txt.append('|'+ '|'.join([NAMES[name],mode,*[fmt(z[k]) for k in ['Success@3px','Success@5px','Success@10px','StructureSuccess@5px','no_overlap_FPR','structure_repeatability']]])+'|')
 txt+=['','Natural最多1024实际点；Equal每图为五方法实际检测数和128的共同最小值。相同原生图像、Homography/RANSAC/full门控。成功误差来自原score.py合法共同域网格双向投影，结构Precision/Recall仅限可监督静态区域。场景为来源语境，未经逐片人工核实。','', '所有失败保留；没有输出、候选被拒绝、错误接受分别记录。阶段诊断按可见静态检测→重复点→正确静态匹配→几何顺序分类，是工程定位，不足以证明唯一因果。','', '512/640、重叠、二维旋转、来源语境、增强类型及尺寸交叉分层见完整JSON和CSV。二维旋转不是雷达观测方位。计时分离缓存共享特征、特征头和匹配几何，不将缓存计时冒充端到端。','', '历史结构分支仅接入描述路径；检测头独立读取冻结共享特征。其描述头与历史SAR描述子迁移逐张量相同。最新局部监督描述子训练不等于结构模型最近一次step0回退。']
 p=ROOT/f'BASELINE_REPORT_{t}.md';p.write_text('\n'.join(txt));print(p)
if __name__=='__main__':report_baseline()
