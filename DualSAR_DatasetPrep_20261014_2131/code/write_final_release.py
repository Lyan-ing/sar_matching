from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import json,re,csv
R=Path(__file__).resolve().parents[1];T=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
qc=json.loads((R/'FINAL_INPUT_QC_20261015_0044.json').read_text());entry=json.loads((R/'DATASET_ENTRYPOINTS_20261015_0044.json').read_text())
data=[]
for s in ['train','val']:data.extend(json.loads(l) for l in Path(entry['synthetic_manifests'][s+'_all_known_components']).read_text().splitlines())
assert len(data)==5948
report=(R/'DUAL_DATASET_REPORT_20261015_0044.md').read_text()
cn={'offshore_vessels_low_texture_sea':'离岸舰船/低纹理海面来源','nearshore_vessels':'近岸舰船来源','port_quay_dense_vessels':'港口、码头及密集舰船（AI初筛）','other_land_structures':'其他陆地结构（AI初筛）','unclassified_background':'海陆/场景未定背景'}
scene='|场景|Train配对|Val配对|视图实例|已知组件总数|\n|---|---:|---:|---:|---:|\n'+'\n'.join(f"|{cn[r['scene_category']]}|{r['train_pairs']}|{r['val_pairs']}|{r['views']}|{r['known_components']}|" for r in qc['usable_scenes'])
report=re.sub(r'### 场景分布（来源上下文分层，生成总量）\n\n\|场景.*?(?=\n\n海面/舰船)','### 最终可用场景分布（来源上下文分层）\n\n'+scene,report,flags=re.S)
maritime=sum(r['pairs'] for r in qc['usable_scenes'] if r['scene_category'] in ['offshore_vessels_low_texture_sea','nearshore_vessels','port_quay_dense_vessels'])
report=report.replace('合计5602组，占93.98%；其余陆地18组和未分类背景341组',f'合计{maritime}组，占{maritime/len(data):.2%}；其余陆地18组和未分类背景339组')
report=report.replace('上方旧生成量和下方原场景/难度表包含隔离样本，不应把生成量当可训练量。最终可用场景、角度及数量另见','上方生成量包含隔离样本，下方场景/难度/角度表统一使用最终可用5948组；不应把生成量当可训练量。可用场景、角度及数量另见')
profiles=Counter(r['profile'] for r in data);dyn=sum(r['synthetic_dynamic_events']>0 for r in data)
profile='|生成配置|最终可用配对|实际发生合成舰船移动的配对|\n|---|---:|---:|\n'+'\n'.join(f"|{p}|{profiles[p]}|{sum(r['profile']==p and r['synthetic_dynamic_events']>0 for r in data)}|" for p in ['easy','medium','rotation','low','none','dynamic'])
report=re.sub(r'\|生成配置\|实际配对.*?(?=\n\n其中319组)',profile,report,flags=re.S)
report=report.replace('动态profile276组中只有214组实际发生移动，余62组',f"最终可用动态profile{profiles['dynamic']}组中只有{dyn}组实际发生移动，余{profiles['dynamic']-dyn}组")
angles=list(csv.DictReader((R/'SYNTHETIC_USABLE_2D_ANGLES_20261015_0044.csv').open(encoding='utf-8-sig')))
angle='|合成二维图像旋转段（度）|最终可用配对|视图实例|\n|---|---:|---:|\n'+'\n'.join(f"|{a['image_2D_rotation_bin']}|{a['usable_synthetic_pairs']}|{a['views']}|" for a in angles)
report=re.sub(r'\|合成二维图像旋转段（度）\|配对关系.*?(?=\n\n源图不预上采样)',angle,report,flags=re.S)
size512=sum(r['width']==512 for r in data)
report=report.replace('3334组为512×512',f'最终可用{size512}组为512×512').replace('DATA_USAGE_20261015_0044.md',f'DATA_USAGE_{T}.md')
report=report.replace('|---|---:|---:|---:|---:|---:|---:|---:|','|---|---:|---:|---:|---:|---:|---:|').replace(' 输入强度门控前','输入强度门控前')
rp=R/f'DUAL_DATASET_REPORT_{T}.md'
with rp.open('x') as f:f.write(report)
usage=(R/'DATA_USAGE_20261015_0044.md').read_text().replace('其他5642组有静态几何点',f"其他{len(data)-profiles['none']}组有静态几何点")
usage=usage.replace('动态profile276组中实际发生移动214组，其余62组',f"最终可用动态profile{profiles['dynamic']}组中实际发生移动{dyn}组，其余{profiles['dynamic']-dyn}组")
with (R/f'DATA_USAGE_{T}.md').open('x') as f:f.write(usage)
final=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),synthetic=dict(generated_unique_sources=5961,generated_pairs=5961,generated_views=11922,
 usable_unique_source_files=len(data),usable_pairs=len(data),usable_views=2*len(data),recommended_parent_core_pairs=5188,recommended_parent_core_views=10376,
 quarantined_pairs=13,usable_ssdd_extension_pairs=760,usable_ssdd_extension_views=1520,usable_train_pairs=5086,usable_val_pairs=862,
 maritime_source_context_pairs=maritime,maritime_source_context_fraction=maritime/len(data),scene_counts=qc['usable_scenes'],profiles=dict(profiles),actual_dynamic_pairs=dyn,
 scenes_are_source_context_not_independent_physical_GT=True,default_entrypoints=str(R/'DATASET_ENTRYPOINTS_20261015_0044.json')),
 real=json.loads((R/'REAL_RESOURCE_SUMMARY_20261014_2329.json').read_text()),report=str(rp),data_usage=str(R/f'DATA_USAGE_{T}.md'),no_training_started=True,new_SAR_downloads=0)
with (R/f'DUAL_DATASET_FINAL_COUNTS_{T}.json').open('x') as f:json.dump(final,f,ensure_ascii=False,indent=2)
with (R/f'README_{T}.md').open('x') as f:f.write(f'# 数据交付入口\n\n最终报告：[MD]({rp.name}) / [PDF]({rp.with_suffix(".pdf").name})\n\n数据入口：[DATASET_ENTRYPOINTS](DATASET_ENTRYPOINTS_20261015_0044.json)；默认只读USABLE manifest。\n\n使用说明：[DATA_USAGE](DATA_USAGE_{T}.md)；补充来源：[TARGETED_SOURCE_PLAN](TARGETED_SOURCE_PLAN_20261015_0007.md)。\n\n实际生成11922视图；隔离13组后可用11896视图，推荐核心10376视图。旧0007/0044报告及原生成manifest保留作过程版本，采用本文链接版本。真实独立GT仍0。\n')
print(json.dumps(dict(timestamp=T,report=str(rp),usable_pairs=len(data),usable_views=2*len(data),core_views=10376,actual_dynamic=dyn,size512_pairs=size512),ensure_ascii=False))
