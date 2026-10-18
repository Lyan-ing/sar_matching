"""Reject information-free valid regions without regenerating/overwriting any view."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json,csv,re,hashlib,sys
import cv2,numpy as np
R=Path(__file__).resolve().parents[1]
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def text(p,s):
 with p.open('x',encoding='utf-8') as f:f.write(s)
def jsonl(p,rs):
 with p.open('x',encoding='utf-8') as f:
  for r in rs:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def table(p,rs):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)

def main():
 cv2.setNumThreads(1)
 src={s:R/f'SYNTHETIC_{s}_all_known_components_manifest_20261014_2340.jsonl' for s in ['train','val']}
 data=rows(src['train'])+rows(src['val'])
 def inspect(r):
  d=[]
  for i in [0,1]:
   im=cv2.imread(r[f'view{i}'],0);valid=cv2.imread(r['valid_masks'][i],0)>0;values=im[valid]
   assert values.size and im.shape==(r['height'],r['width'])
   d.append(dict(valid_min=int(values.min()),valid_max=int(values.max()),valid_std=float(values.std())))
  bad=any(v['valid_min']==v['valid_max'] for v in d)
  return dict(r,input_qc_state='quarantine_constant_valid_region' if bad else 'usable',valid_region_intensity=d,
   low_texture_valid_std_below_1=any(v['valid_std']<1 for v in d),input_qc_basis='pixel intensities on known valid support only; no matching model or success read',
   allow_static_descriptor_positive_supervision=(r['allow_static_descriptor_positive_supervision'] and not bad))
 with ThreadPoolExecutor(max_workers=4) as ex:checked=list(ex.map(inspect,data))
 good=[r for r in checked if r['input_qc_state']=='usable'];bad=[r for r in checked if r['input_qc_state']!='usable'];T=stamp()
 manifests={};counts=[]
 for split in ['train','val']:
  for kind in ['parent_core','ssdd_extension','all_known_components']:
   selected=[r for r in good if r['split']==split and (kind=='all_known_components' or (r['construction_pool']=='parent_component_core')==(kind=='parent_core'))]
   p=R/f'SYNTHETIC_{split}_{kind}_USABLE_manifest_{T}.jsonl';jsonl(p,selected);manifests[f'{split}_{kind}']=str(p)
   counts.append(dict(split=split,pool=kind,usable_source_images=len(selected),usable_pairs=len(selected),usable_views=2*len(selected),known_components=len({r['known_component'] for r in selected})))
 assert len({r['source_path'] for r in good})==len(good)
 core=[r for r in good if r['construction_pool']=='parent_component_core'];assert 2*len(core)>10000
 assert {r['known_component'] for r in good if r['split']=='train'}.isdisjoint({r['known_component'] for r in good if r['split']=='val'})
 qp=R/f'SYNTHETIC_INPUT_QUARANTINE_{T}.jsonl';jsonl(qp,bad)
 table(R/f'USABLE_TRAIN_VAL_COUNTS_{T}.csv',counts)
 scenes=[]
 for name in sorted({r['source_scene_category'] for r in good}):
  a=[r for r in good if r['source_scene_category']==name]
  scenes.append(dict(scene_category=name,train_pairs=sum(r['split']=='train' for r in a),val_pairs=sum(r['split']=='val' for r in a),pairs=len(a),views=2*len(a),known_components=len({r['known_component'] for r in a})))
 table(R/f'SYNTHETIC_USABLE_SCENES_{T}.csv',scenes)
 def angle_bin(a):
  for lo,hi,label in [(0,15,'0-15'),(15,30,'15-30'),(30,60,'30-60'),(60,90,'60-90'),(90,120.00001,'90-120')]:
   if lo<=a<hi:return label
  return '>120'
 angular=[]
 for b in ['0-15','15-30','30-60','60-90','90-120','>120']:
  a=[r for r in good if angle_bin(r['image_rotation_2D_deg'])==b];angular.append(dict(image_2D_rotation_bin=b,usable_synthetic_pairs=len(a),views=2*len(a),real_observation_azimuth=False))
 table(R/f'SYNTHETIC_USABLE_2D_ANGLES_{T}.csv',angular)
 quality=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),generated_pairs=len(data),generated_views=2*len(data),
  usable_pairs=len(good),usable_views=2*len(good),usable_core_pairs=len(core),usable_core_views=2*len(core),
  quarantined_pairs=len(bad),quarantined_sources_not_regenerated=True,quarantine_manifest=str(qp),
  gate='either view constant on valid pixel support -> quarantined; nonconstant low texture remains challenging data',
  nonconstant_low_texture_pairs=sum(r['low_texture_valid_std_below_1'] for r in good),
  usable_count_table=counts,usable_scenes=scenes,all_generated_images_and_previous_manifests_preserved=True,
  bilinear_static_support_filtered_at_training_read=True,no_model_matching_used=True,real_C_frozen_unchanged=True,
  new_real_GT=0,no_training_started=True,new_downloads=0)
 save(R/f'FINAL_INPUT_QC_{T}.json',quality)
 oldentry=R/'DATASET_ENTRYPOINTS_20261015_0007.json';entry=json.loads(oldentry.read_text())
 entry.update(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),generated_manifests=entry['synthetic_manifests'],
  synthetic_manifests=manifests,input_qc=str(R/f'FINAL_INPUT_QC_{T}.json'),quarantined_input_manifest=str(qp),
  default_synthetic_pool='parent_core after valid-region constant-input QC')
 ep=R/f'DATASET_ENTRYPOINTS_{T}.json';save(ep,entry)
 # New report version; retain the previous MD and PDF unchanged.
 original=(R/'DUAL_DATASET_REPORT_20261015_0007.md').read_text()
 opening='**合成数据已实际构建5961组配对、11922个视图实例。**'
 cq='\n\n**最终输入质量门控：已生成数量与可训练数量分别报告。** '
 cq+=f"按每幅图有效像素的最小/最大值检查，隔离{len(bad)}组恒定输入，未重新生成补数；最终推荐核心池{len(core)}组/{2*len(core)}视图，全部可用池{len(good)}组/{2*len(good)}视图。"
 cq+=f"另有{quality['nonconstant_low_texture_pairs']}组非恒定低纹理输入保留，几何有GT不保证纹理可辨。隔离清单、有效区强度统计和最终manifest见 `{ep.name}`。旧0007版本是输入强度检查前的生成/几何验收记录，保留但不再作默认数据入口。\n\n"
 original=original.replace(opening,opening+cq,1)
 original=original.replace('推荐主池为已知HRSID父场景隔离的5200组/10400视图；另有SSDD扩展761组/1522视图，其原采集身份不足，必须单列限制。','输入强度门控前的父场景核心生成量为5200组/10400视图；SSDD扩展生成量761组/1522视图。最终可用数量见下表，SSDD身份限制不变。',1)
 original=original.replace('## 2. 合成数据实际数量与划分','## 2. 合成数据生成数量与最终可用划分',1)
 usable_table='|最终可用池|Train关系|Train视图|Val关系|Val视图|组件Train/Val|\n|---|---:|---:|---:|---:|---|\n'
 for kind,title in [('parent_core','推荐父场景核心'),('ssdd_extension','限制性SSDD扩展'),('all_known_components','全部可用池')]:
  a,b=[next(x for x in counts if x['split']==s and x['pool']==kind) for s in ['train','val']]
  usable_table+=f"|{title}|{a['usable_pairs']}|{a['usable_views']}|{b['usable_pairs']}|{b['usable_views']}|{a['known_components']} / {b['known_components']}|\n"
 original=original.replace('### 场景分布（来源上下文分层）','### 最终可用数量（默认训练/验证入口）\n\n'+usable_table+'\n上方旧生成量和下方原场景/难度表包含隔离样本，不应把生成量当可训练量。最终可用场景、角度及数量另见 `SYNTHETIC_USABLE_SCENES`、`SYNTHETIC_USABLE_2D_ANGLES`、`USABLE_TRAIN_VAL_COUNTS`。\n\n### 场景分布（来源上下文分层，生成总量）',1)
 original=original.replace('DATASET_ENTRYPOINTS_20261015_0007.json',ep.name)
 for key,value in manifests.items():
  old=entry['generated_manifests'][key];original=original.replace(Path(old).name,Path(value).name)
 original=original.replace('DATA_USAGE_20261015_0007.md',f'DATA_USAGE_{T}.md')
 original=original.replace('立即可用**：合成核心10400视图及扩展1522视图',f"立即可用**：合成核心{2*len(core)}视图及扩展{2*(len(good)-len(core))}视图")
 rp=R/f'DUAL_DATASET_REPORT_{T}.md';text(rp,original)
 usage=(R/'DATA_USAGE_20261015_0007.md').read_text()
 for key,value in manifests.items():usage=usage.replace(entry['generated_manifests'][key],value).replace(Path(entry['generated_manifests'][key]).name,Path(value).name)
 usage=usage.replace('DATASET_ENTRYPOINTS_20261015_0007.json',ep.name)
 ctrain=next(c for c in counts if c['split']=='train' and c['pool']=='parent_core');cval=next(c for c in counts if c['split']=='val' and c['pool']=='parent_core')
 etrain=next(c for c in counts if c['split']=='train' and c['pool']=='ssdd_extension');evals=next(c for c in counts if c['split']=='val' and c['pool']=='ssdd_extension')
 usage=usage.replace('4473组、8946视图',f"{ctrain['usable_pairs']}组、{ctrain['usable_views']}视图")
 usage=usage.replace('合计5200张按已知重复组去重的来源文件、5200关系、10400视图',f"合计{len(core)}张按已知重复组去重的来源文件、{len(core)}关系、{2*len(core)}视图")
 usage=usage.replace('626训练组、135验证组',f"{etrain['usable_pairs']}训练组、{evals['usable_pairs']}验证组")
 usage=usage.replace('合计5961组/11922视图',f"合计{len(good)}组/{2*len(good)}视图")
 usage='> 最终可用版本：恒定有效像素输入已隔离，原生成5961组仍保留。低纹理不是无重叠负例。\n\n'+usage
 text(R/f'DATA_USAGE_{T}.md',usage)
 save(R/f'FINAL_DATA_FREEZE_{T}.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
  input_qc=quality,entrypoints=str(ep),report=str(rp),original_final_verification=str(R/'DELIVERY_VERIFICATION_20261015_0007.json'),
  files={str(Path(p).relative_to(R)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [*manifests.values(),str(ep),str(qp)]},
  source_independence_not_upgraded=True,original_images_and_history_not_overwritten=True))
 print(json.dumps(dict(report=str(rp),entrypoints=str(ep),qc=quality),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
