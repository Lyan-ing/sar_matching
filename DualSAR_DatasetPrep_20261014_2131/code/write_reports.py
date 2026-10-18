"""Publish auditable counts, final entrypoints and concise research reports."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter,defaultdict
import json,csv,hashlib,sys
import numpy as np
from PIL import Image,ImageDraw
BASE=Path('/home/yl/matching/sar_matching');R=Path(__file__).resolve().parents[1]
T=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def latest(pattern):return max(R.glob(pattern),key=lambda p:p.stat().st_mtime_ns)
def rows(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def text(p,s):
 with p.open('x',encoding='utf-8') as f:f.write(s)
def table(p,rs):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def binof(x):
 for a,b,name in [(0,15,'0-15'),(15,30,'15-30'),(30,60,'30-60'),(60,90,'60-90'),(90,120.0000001,'90-120')]:
  if a<=x<b:return name
 return '>120'
def rel(p):return str(p.relative_to(R))

def main():
 paths={f'{split}_{kind}':latest(f'SYNTHETIC_{split}_{kind}_manifest_*.jsonl') for split in ['train','val'] for kind in ['parent_core','ssdd_extension','all_known_components']}
 synthetic=rows(paths['train_all_known_components'])+rows(paths['val_all_known_components'])
 resource=json.loads(latest('RESOURCE_AUDIT_*.json').read_text());real=json.loads(latest('REAL_RESOURCE_SUMMARY_*.json').read_text())
 supervision=json.loads(latest('STATIC_SUPERVISION_VERIFICATION_*.json').read_text())
 relations=rows(latest('REAL_ACQUISITION_RELATIONS_*.jsonl'));realcrops=rows(latest('REAL512_EXISTING_CROPS_*.jsonl'))
 candidates=rows(latest('REAL512_CANDIDATES_*.jsonl'));proposals=json.loads(latest('TARGETED_SOURCE_PROPOSALS_*.json').read_text())
 humans=[]
 old=BASE/'RealSAR_IndependentBenchmark_20261011_2015/annotation_adapter/annotations'
 for d in sorted(old.iterdir()):
  fs=sorted(d.glob('v*.json'))
  if not fs:continue
  doc=json.loads(fs[-1].read_text());humans.append(dict(sample_id=d.name,latest_actual_record=str(fs[-1]),
   historical_version_count=len(fs),actual_clicked_points=len(doc.get('points',[])),
   independent_review_records=sum(bool(p.get('review')) for p in doc.get('points',[])),
   actor=doc.get('actor'),saved_at=doc.get('saved_at'),GT_qualified=doc.get('gt_quality',{}).get('status')=='gt_qualified',
   coordinates_not_moved_or_recertified=True))
 save(R/f'EXISTING_HUMAN_EVIDENCE_{T}.json',humans)
 hb={h['sample_id']:h for h in humans}
 for c in candidates:
  c['prior_actual_human_click_evidence']=hb.get(c['sample_id'])
  if c['sample_id'] in hb:c['physical_correspondence_status']='historical_human_clicks_unreviewed_coordinate_frame_not_recertified'
 cp=R/f'REAL512_CANDIDATES_WITH_EVIDENCE_{T}.jsonl'
 with cp.open('x',encoding='utf-8') as f:
  for c in candidates:f.write(json.dumps(c,ensure_ascii=False)+'\n')
 scene=[];difficulty=[];angles=[];splitrows=[]
 for name in sorted({r['source_scene_category'] for r in synthetic}):
  a=[r for r in synthetic if r['source_scene_category']==name]
  scene.append(dict(scene_category=name,train_pairs=sum(r['split']=='train' for r in a),val_pairs=sum(r['split']=='val' for r in a),
   total_pairs=len(a),view_instances=2*len(a),known_components=len({r['known_component'] for r in a}),
   evidence='AI provisional source-context screening' if name in ['port_quay_dense_vessels','other_land_structures'] else 'unknown' if name=='unclassified_background' else 'publisher coarse / SL-SSDD mask derived; source context not guaranteed crop label'))
 for profile in ['easy','medium','rotation','low','none','dynamic']:
  a=[r for r in synthetic if r['profile']==profile];difficulty.append(dict(profile=profile,pairs=len(a),views=2*len(a),
   min_overlap=float(min(min(r['overlap_fractions']) for r in a)),max_overlap=float(max(max(r['overlap_fractions']) for r in a)),
   actual_dynamic_events_pairs=sum(r['synthetic_dynamic_events']>0 for r in a),physical_viewpoint_change=False))
 for b in ['0-15','15-30','30-60','60-90','90-120','>120']:
  a=[r for r in synthetic if binof(r['image_rotation_2D_deg'])==b];angles.append(dict(image_2D_rotation_bin=b,synthetic_pairs=len(a),
    views=2*len(a),source_components=len({r['known_component'] for r in a}),interpretation='2D image-map local polar rotation ONLY, not radar azimuth'))
 for split in ['train','val']:
  for kind in ['parent_core','ssdd_extension','all_known_components']:
   a=rows(paths[f'{split}_{kind}']);splitrows.append(dict(split=split,pool=kind,distinct_known_deduplicated_source_images=len({r['source_path'] for r in a}),
     pair_relations=len(a),view_instances=2*len(a),known_source_components=len({r['known_component'] for r in a}),manifest=str(paths[f'{split}_{kind}'])))
 table(R/f'SYNTHETIC_SCENE_DISTRIBUTION_{T}.csv',scene);table(R/f'SYNTHETIC_DIFFICULTY_{T}.csv',difficulty)
 table(R/f'SYNTHETIC_2D_ANGLE_DISTRIBUTION_{T}.csv',angles);table(R/f'TRAIN_VAL_COUNTS_{T}.csv',splitrows)
 # Scope the archive metadata against actual readable pair assets, not stale raster flags.
 inv=rows(BASE/'RealSAR_AngleExpansion_20261012_1641/acquisition_inventory.jsonl');retained={x for r in relations for x in r['acquisition_ids']}
 flags=Counter((r.get('raster_status','none'),r['acquisition_id'] in retained) for r in inv)
 maritime=sum(s['total_pairs'] for s in scene if s['scene_category'] in ['offshore_vessels_low_texture_sea','nearshore_vessels','port_quay_dense_vessels'])
 summary=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),synthetic=dict(raw_indexed_files=7164,known_deduplicated_original_files=7135,
   distinct_sources_used=len(synthetic),pairs=len(synthetic),view_instances=2*len(synthetic),parent_core_pairs=5200,parent_core_views=10400,
   component_limited_ssdd_extension_pairs=761,component_limited_ssdd_extension_views=1522,
   train_pairs=5099,val_pairs=862,marine_source_context_pairs=maritime,marine_source_context_fraction=maritime/len(synthetic),
   original_source_scene_labels_not_crop_ground_truth=True,unclassified_background_pairs=341,scene_table=scene,
   train_val_component_intersection=[],formal900_untouched_and_not_counted=True,independent_raw_acquisition_count='unestablished'),
  real=real,real_source_catalog_crosscheck={str(k):v for k,v in flags.items()},
  old_human_actual_clicks=sum(h['actual_clicked_points'] for h in humans),old_independent_review_records=sum(h['independent_review_records'] for h in humans),
  independent_real_GT=0,actual_network_training_started=False,new_SAR_downloads=0)
 save(R/f'DUAL_DATASET_COUNTS_{T}.json',summary)

 # Small actual material previews: source views are never enlarged, rotated or recreated.
 previews=[]
 for category,label in [('offshore_vessels_low_texture_sea','offshore'),('port_quay_dense_vessels','port')]:
  row=next(r for r in synthetic if r['source_scene_category']==category and r['width']==512 and r['profile'] in ['easy','medium'])
  im=Image.new('RGB',(1040,550),(22,22,22));d=ImageDraw.Draw(im)
  d.text((8,5),f"{row['sample_id']} | {category} | 2D synthetic proxy, NOT real viewpoint GT",fill='white')
  for i in [0,1]:
   with Image.open(row[f'view{i}']) as tile:im.paste(tile.convert('RGB'),(i*528,30))
  path=R/f'SYNTHETIC_PREVIEW_{label}_{T}.png';im.save(path);previews.append((label,path,row['sample_id']))

 # Freeze verification is scoped to the existing 3-candidate delivery, not a new large original-data SHA audit.
 froot=BASE/'RealSAR_Fixed512_20261014_1743/release_20261014_1756';fr=json.loads((froot/'FREEZE_20261014_1810.json').read_text())
 for name,value in fr['files'].items():assert sha(froot/name)==value['sha256'],name
 assert len(synthetic)==5961 and supervision['static_QA_passed_count']==5961
 assert {r['known_component'] for r in synthetic if r['split']=='train'}.isdisjoint({r['known_component'] for r in synthetic if r['split']=='val'})
 entry=dict(created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),default_synthetic_pool='parent_core',
  synthetic_manifests={k:str(v) for k,v in paths.items()},source_inventory=str(latest('SOURCE_INVENTORY_*.jsonl')),
  real_relation_registry=str(latest('REAL_ACQUISITION_RELATIONS_*.jsonl')),real_existing512_inventory=str(latest('REAL512_EXISTING_CROPS_*.jsonl')),
  real512_curated_candidates=str(cp),real_component_roles={s:str(latest(f'REAL512_{s}_UNLABELED_*.jsonl')) for s in ['research_development','candidate_validation']},
  real_supervised_eligible_manifests={s:str(latest(f'REAL512_{s}_SUPERVISED_ELIGIBLE_*.jsonl')) for s in ['research_development','candidate_validation']},
  frozen3_original_package=str(froot/'blind512_materials_20261014_1810.zip'),frozen3_package_not_regenerated=True,
  pair_adapter=str(R/'code/load_prepared_pair.py'),input_rule='load_inference_images returns only two arrays; training geometry and masks are targets, not prediction inputs',
  no_training_started=True)
 ep=R/f'DATASET_ENTRYPOINTS_{T}.json';save(ep,entry)

 scenes_cn={'offshore_vessels_low_texture_sea':'离岸舰船/低纹理海面来源','nearshore_vessels':'近岸舰船来源','port_quay_dense_vessels':'港口、码头及密集舰船（AI初筛）','other_land_structures':'其他陆地结构（AI初筛）','unclassified_background':'海陆/场景未定背景'}
 scene_table='\n'.join(f"|{scenes_cn[s['scene_category']]}|{s['train_pairs']}|{s['val_pairs']}|{s['view_instances']}|{s['known_components']}|" for s in scene)
 syn_angle_table='\n'.join(f"|{a['image_2D_rotation_bin']}|{a['synthetic_pairs']}|{a['views']}|" for a in angles)
 real_angle_table='\n'.join(f"|{a['angle_bin']}|{a['actual_acquisition_relations']}|{a['all_source_components']}|{a['maritime_relations']} / {a['maritime_source_components']}|{a['curated_search_candidates']}|{a['fixed512_prepared_candidates']}|0|" for a in real['angle_coverage'])
 profile_table='\n'.join(f"|{d['profile']}|{d['pairs']}|{d['actual_dynamic_events_pairs']}|" for d in difficulty)
 usage=f'''# 已准备数据的最简使用说明

以 `{ep.name}` 为入口，路径均指实际文件。没有启动网络训练。

## 默认使用核心训练/验证池

- Train：`{paths['train_parent_core'].name}`，4473组、8946视图、101个已知HRSID父场景组件。
- Val：`{paths['val_parent_core'].name}`，727组、1454视图、15个已知父场景组件。
- 合计5200张按已知重复组去重的来源文件、5200关系、10400视图；同一来源仅一组双图。父场景组件与旧test均不交叉。

SSDD扩展单列：626训练组、135验证组，原采集父场景不完整，只能主张已知像素组件隔离。`all_known_components` 合计5961组/11922视图，不应把SSDD组件当真实独立原采集；论文可将核心作为主实验、扩展作为补充，并明确限制。

## 读取现成配对

```python
import json, sys
sys.path.insert(0, {str(R/'code')!r})
from load_prepared_pair import load_training_pair, load_inference_images
manifest = {str(paths['train_parent_core'])!r}
with open(manifest) as f:
    row = json.loads(next(f))
training_target = load_training_pair(row)  # 与现有data.pair字典主要字段一致
images_only = load_inference_images(row)  # 模型预测只接收这两幅图
```

现有训练源码/配置未改；该适配器只读数据，不训练网络。尺寸64–512、均为32的倍数，采用batch=1或按尺寸分桶；不能把小图无说明上采样到512。旧训练入口固定在线256生成，不能声称未经改动已在训练本数据。未来切换需单独锁定模型、配置和数据版本。

监督使用最终manifest指向的 `static_all_vessels` 掩膜和静态NPZ。它们从原始所有舰船标注生成，包括低可见比例/被截断实例，并排除动态区域及5px保护带；保留旧掩膜为历史。适配器再过滤双线性2×2支持无效的查询点。有效区/静态/海陆标签只用于训练目标或离线评价，不传入纯图像预测和接受门控。

`none` 319组是已知二维无重叠合成关系，不能当真实SAR无重叠证据。其他5642组有静态几何点，但纯海面点的解析几何正确不证明真实跨角可匹配。动态profile276组中实际发生移动214组，其余62组没有可移动可见实例，均如实记录。

## 真实数据入口

`REAL512_CANDIDATES_WITH_EVIDENCE` 仅是12项搜索候选：冻结3项保持原盲包，其余9项仅引用旧512图，不增加裁片或标注系统。真实监督合格train/val manifest目前为空；UNLABELED清单不可自动作为正对应训练。

C003/C002/C001仍按你已审查的原ANNOTATION_ACTIONS分发和回收，本轮不重新裁切/打包；真人标注由你安排。旧5任务22个实际点击另有引用表，独立复核记录0，没有迁移坐标或自动升级。辅助观察、旧点击、独立标注和合格GT分别保存。
'''
 text(R/f'DATA_USAGE_{T}.md',usage)

 supplement_rows='\n'.join(f"|{p['site']}|{p['azimuth_delta_deg']:.3f}°|{p['time_gap_days']*24:.4f}h|{p['existing_local_acquisition_count']} / 2|" for p in proposals)
 plan=f'''# 缺失真实采集的针对性补充方案（未下载）

优先级由当前真实角度/站点缺口决定，不由M2/B匹配结果决定。完整ID、task、UTC、URL、方位/入射角和元数据文件见 `{latest('TARGETED_SOURCE_PROPOSALS_*.json').name}`。以下仅有供应商元数据及产品足迹共同场景证据，**没有新增影像、没有可辨识共同地面结构证明、没有GT**。

公开入口已核对：[Umbra开放数据登记](https://registry.opendata.aws/umbra-open-data/)和[Capella开放数据登记](https://registry.opendata.aws/capella_opendata/)均说明匿名访问及CC BY 4.0。具体产品URL和身份以本地存档供应商元数据为准，本轮没有读取/下载这些候选栅格；不能保证目录可用就一定适合512物理标注。

|站点|真实观测方位差|采集时间间隔|已有可用本地采集|
|---|---:|---:|---:|
{supplement_rows}

1. **15°–30°优先补来源组件**：当前只有鹿特丹。安特卫普17.093°与杰贝阿里24.937°的间隔约19/22秒，若未来实际产品中存在固定地面结构，可增加该角度段的站点支持。它们不是当前冻结C001/C002/C003的重新裁切版本，而是另外的真实采集身份。
2. **60°–90°先复用已下载的鹿特丹75.635°与安特卫普88.394°**：已有人工辅助A类场景线索，但固定512的分散地面点不足。本轮不生成新裁片，先由研究人员审查现有Native上下文的共同固定结构，再决定是否值得独立有界准备；不需要先下载相同来源。
3. **90°–120°补短时、多站点关系**：当前仅3关系/2组件，其中海事只有Kingston一个组件且物理对应困难。鹿特丹90.674°约11小时与杰贝阿里104.988°约24.2小时提供不同站点元数据线索；高度、阴影及观测可见性风险仍须实际影像审查。现有>120°困难样本继续保留，不挪入目标角度段。
4. **新加坡40.623°作为新真实站点候选**：约22.8小时；用于增加30°–60°的港口站点多样性，而不是继续增加鹿特丹裁片数。新站点若未来读取，先确定整组件的开发/验证角色，再审查结构。不能宣称它与所有身份不完整的HRSID/SSDD绝对独立。
5. **0°–15°可选补充**：安特卫普11.345°和釜山9.838°增加可核验的小角差站点；已有同向/近同向对照单独统计，不能凭0°样本堆数量覆盖真正异向难度。

时间间隔≤2天、产品凸足迹交集覆盖较小足迹≥50%只作离线目录筛选，不是正GT门槛或物理共同可见性证明。供应商几何投到站点局部UTM，复用OpenCV凸多边形交集，仅用于提案。站点/collect/task全部保留；若已有一侧，只复用该侧，不重复下载或复制原资产。

**新增下载前须单独明确具体采集ID、缺口、预计读取窗口/体量与组件角色，再作下载决定。此报告不触发下载。** 没有短时固定岸线结构的纯海面/动态舰船关系保持无可靠静态配准证据；不能为填补比例强制生成正GT。
'''
 text(R/f'TARGETED_SOURCE_PLAN_{T}.md',plan)

 report=f'''# 合成SAR与真实多视角SAR：512任务数据准备结果

## 实际完成与证据边界

**合成数据已实际构建5961组配对、11922个视图实例。** 推荐主池为已知HRSID父场景隔离的5200组/10400视图；另有SSDD扩展761组/1522视图，其原采集身份不足，必须单列限制。合成图尺寸64–512，不预放大来源，全部5961组通过现有几何/有效区验收及新版静态监督检查。没有网络训练或模型推理。

**真实数据完成库存去重、来源/角度审计及候选清单，不声称已建成物理GT集。** 191旧512配对、18 Native关系及3早期Sentinel-1配对合并为78个实际采集关系、73个采集身份、15个站点组件；现存197组512材料，包含原冻结3项。精选12项仅是搜索候选（11项海事来源上下文），其中3项沿用原盲包等待真人；独立合格GT仍0。

本轮所有产物在独立目录；C003/C002/C001、原盲包、原始SAR、GT/人工历史、旧900关系和算法结果不修改。最终入口是 `{ep.name}`，训练/验证manifest指向实际图像与标签文件，非仅计划数量。

## 1. 原始资源：文件、组件、视图分别计数

现有目录中SAR图像来自HRSID、SSDD及HRSID背景。SL-SSDD是1160张SSDD的海陆掩膜，新增SAR图像数为0；SSDD的VOC/COCO/实例掩膜等重复表示也不重复算图。未引入其他未下载数据。

|来源|原图索引文件数|Train原文件|Val原文件|Test保留原文件|
|---|---:|---:|---:|---:|
|HRSID|5604|4163|681|760|
|SSDD|1160|636|137|387|
|SAR背景|400|312|47|41|
|合计|7164|5111|865|1188|

7164文件均已实际解码可用。复用原26个已知重复组、去除29份已知重复后，全库存按已知像素重复组去重为7135张；本轮Train/Val去重为5961张。**它们是不同原始文件/像素图像计数，不是7135或5961个独立原采集。** HRSID与背景共有133个已知父场景组件，Train101/Val15/Test17；SSDD为已知像素关联组件，原采集身份仍不完整。未重新大规模SHA或相似匹配搜索，因此未知重复/同采集风险不能宣称为零。

SSDD的1160个海陆掩膜按原发布split、同名stem和尺寸核验，值为0陆地/255海面。其来源级海面比例≥98%记为离岸/海面背景，否则有海面则近岸；这是分层规则，不代表精确离岸距离。HRSID沿用发布方近岸/离岸粗分类，无像素海陆掩膜；背景没有可靠海陆标签的保持未知。港口与陆地结构的42张来源图经本轮有界AI视觉复看作**暂定场景标签**，不是人工真值，也不能保证每个变换裁片完整保留来源场景。

## 2. 合成数据实际数量与划分

|池|Train来源/关系|Train视图|Val来源/关系|Val视图|已知组件 Train/Val|
|---|---:|---:|---:|---:|---|
|推荐父场景核心|4473|8946|727|1454|101 / 15|
|SSDD限制性扩展|626|1252|135|270|567 / 97|
|全部已知组件池|5099|10198|862|1724|668 / 112|

每张去重来源仅生成一组双视图，关系数等于来源数，视图数恰为2倍；预检查12组/24视图和旧900关系均未计入11922。新Train/Val按修复后的源索引保留已知父场景/组件隔离，交集0，且与源Test组件交集0。核心池已超过10000视图目标，不需要用重复增强或SSDD弱身份组件凑够主池。

“核心隔离”仅指现有HRSID文件父场景与已知像素关联，仍不等于完整真实采集身份审计。旧900基准及已多轮观察的历史确认集不会因本数据版本重新成为未见测试。真实来源也未拆单图为合成训练来污染候选验证组件。

### 场景分布（来源上下文分层）

|场景|Train配对|Val配对|视图实例|已知组件总数|
|---|---:|---:|---:|---:|
{scene_table}

海面/舰船、近岸与港口来源上下文合计{maritime}组，占{maritime/len(synthetic):.2%}；其余陆地18组和未分类背景341组不混入海面比例。**港口24组、陆地18组为AI暂定标签**，需论文使用前语义复核；来源组件可跨多个场景，不能把表中组件数相加当独立总数。Val港口只有1组、陆地只有2组，场景覆盖仍偏薄；240个密集近岸候选中只有限复看24项，其他不能自动称港口。源级分类、裁片舰船可见数量及SSDD裁片海面比例在manifest分别记录。

### 变化难度与二维角度

|生成配置|实际配对|实际发生合成舰船移动的配对|
|---|---:|---:|
{profile_table}

其中319组是**解析已知二维无重叠负关系**；动态profile276组中只有214组实际发生移动，余62组如实记录无移动。全部为同一SAR图像的二维几何/辐射代理，不能叫真实跨角采集。局部二维旋转由已知H在视图中心的Jacobian极分解计算，含仿射/透视局部旋转意义，和真实雷达方位差完全分开。

|合成二维图像旋转段（度）|配对关系|视图实例|
|---|---:|---:|
{syn_angle_table}

源图不预上采样；按真实尺寸和已有headroom规则自适应输出。3334组为512×512，其余64–480且为32的倍数。合成warp当然包含二维几何插值，这与真实裁片必须不旋转/重采样的要求不同。原解析H和有效掩膜保留，静态监督排除所有**已标注**舰船及动态区域，未标注舰船仍是局限。

原实例生成器会过滤低可见/低面积舰船，不能直接用于“排除全部舰船”的静态监督。本轮另存新版旁文件，从所有源标注（含被截断实例）传播支持，再加5px保护带；1097组因此多排除了边缘舰船像素。旧图像/旧掩膜不覆盖。训练适配器进一步过滤双线性2×2支持无效的点。低纹理海面即便解析GT正确，也可能无可定位特征，不能据合成几何成功推断真实跨视角能力。

![离岸来源合成双图，二维代理非真实跨角GT]({previews[0][1].name})

![港口来源合成双图，二维代理非真实跨角GT]({previews[1][1].name})

### SuperPoint数据入口

默认使用 `{paths['train_parent_core'].name}` 与 `{paths['val_parent_core'].name}`；SSDD扩展另列，不悄悄混作原采集独立主结果。`code/load_prepared_pair.py` 返回既有训练pair字典的图像、H_AB、静态/有效掩膜与查询点；仅为只读数据适配，不改旧训练网络代码。实际读取41个训练代表和32个验证代表，覆盖6配置、各场景及两种池，几何与尺寸检查通过。

预测函数只返回两张图像；GT、海陆/静态掩膜、来源与裁片信息均为训练目标或离线审查，不能进入预测、初始化或接受门控。尺寸不一用batch=1或分桶；旧在线256训练入口需要未来单独切换并锁定配置，本轮没有启动训练。最简命令和示例见 `DATA_USAGE_{T}.md`。

## 3. 真实数据：实际关系、场景与可评价性

旧191组是65个真实采集关系；18 Native与其共享8关系，另增10关系；此前3个Sentinel-1实际配对再加入，合计78关系。固定3项是这些关系的512窗口版本，不能再增加独立采集关系数。现存512材料是191+3早期pilot+3固定=197组，394个视图实例。身份去重得73采集、15站点组件，按站点和采集关联合并，保留各旧namespace的组件别名，不能只按数字后缀拼接。

采集元数据库存402项不等于402已下载影像。338项带有“仅元数据”旧标记，其中12项后来已有当前可用影像，按实际资产纠正计数；326项未在可用关系资产中出现。另6项有旧header/元数据但不在当前可用配对中，不声称为合格512。额外3个pilot采集身份也分别保留，不能重复计算同一RTC图的不同备份。

按**来源场景上下文**：港口30关系/63组512，近岸或沿海城市9关系/27组512，其他陆地39关系/107组512。合计海事上下文39/78关系=50%，90/197组512≈45.69%，原始真实库存尚未达到海事多数。这里没有按站点名称断言每张图都含舰船：不少港口裁片是纯水、孤船、箱区或工业地面，实际船舶可见性/海陆像素分层尚未全量人核。离岸动态舰船关系的共同静态支撑尤其缺乏，不能宣称已具备离岸正GT。

复用既有初筛，形成12项候选引用清单：原冻结3项、旧8项判读候选和1项农田道路补充。11/12为海事上下文，6站点组件；这是面向真人搜索的子集，不是12个合格正例。旧候选中的同向对照单列，>120°困难候选保留，未新增真实裁片、盲包或标注系统。

### 真实雷达观测方位差（按采集关系去重）

角度用供应商方位字段的环形最小差：`abs((a-b+180)%360-180)`，负角/360°绕回正确归一。区间左闭右开，最后90°–120°包含120°；未知不补0，>120°不截断。

|真实方位段（度）|实际采集关系|全部组件|海事关系 / 组件|搜索候选512|原冻结可交真人512|独立GT可评价|
|---|---:|---:|---:|---:|---:|---:|---:|
{real_angle_table}

0°/近同向关系是对照，不能用它们代替真实异向证据；元数据同向、场景可见、可形成精确点是三种不同状态。15°–30°只有1个来源组件；60°–90°与90°–120°没有当前首批固定512候选；各段合格GT均0。多裁同一采集不解决组件缺口，二维旋转也不能填真实角度段。

### 人工证据与真实划分状态

Geo Assist的13项A7/B2/C4、40版本属于辅助观察；原冻结3项没有在本目录回收新真人输入。旧5任务确有22个实际点击、22历史版本，独立复核记录0，已另存**引用清单**；没有删掉它们，也没有把坐标迁移/重认证为新512独立GT。真实独立标注、独立复核通过、最终合格GT不混为一谈。

新清单按整站点组件分配**研究用途角色**：development 7组件/48关系/115组512；candidate-validation 8组件/30关系/82组512，组件交集0。原split角色也逐项保留，没有重写旧数据。两者都是已观察资产，不是模型未见的最终测试；目前真实监督合格train/val manifest均为空，UNLABELED清单不能自动当正对应监督。

没有可靠共同地面结构、只有动态舰船或低纹理海面的关系保持“无可靠静态配准证据/待核验”。若两名真人按任务规则确认缺乏可靠静态支撑，可另列有原因的不可评价/拒绝任务；不能仅因匹配失败就贴无重叠负例，也不能把船体运动误差当静态配准误差。原独立GT协议和已固定512协议继续适用。

## 4. 缺口与补充顺序

1. **立即可用**：合成核心10400视图及扩展1522视图的图像、解析几何、有效/静态监督、隔离manifest；不是训练已完成。
2. **已有候选，仍待人证**：冻结C003/C002/C001和旧512搜索子集；你已安排的真人工作继续沿原盲包流程。本轮不要求重做工具或重打盲包。
3. **已有资产优先**：60°–90°先审查鹿特丹75.635°和安特卫普88.394°现有Native固定结构；本轮不再裁切，不能据A类观察直接保证多个分散点。
4. **仅提出元数据级补充**：15°–30°补安特卫普/杰贝阿里短时关系；90°–120°补鹿特丹/杰贝阿里不同站点关系；新加坡40.623°增加新的港口站点。共8条具体候选，均保留ID、时间/姿态、公开URL和足迹证据，未读取或下载新栅格。详见 `TARGETED_SOURCE_PLAN_{T}.md`。
5. **论文仍缺少**：各真实角度段的合格双人对应、足够海事来源组件，以及较均衡的港口/陆地Val语义覆盖。纯海面无法凭增加图像数量产生共同静态地面GT；应保留可拒绝任务，与有岸线/码头的局部正例分开评价。

新增来源公开性已核对[Umbra开放数据登记](https://registry.opendata.aws/umbra-open-data/)及[Capella开放数据登记](https://registry.opendata.aws/capella_opendata/)。本轮仅形成有针对性的元数据提案，新增下载前须单独说明具体采集、必要性和读取规模；没有启动下载。

## 5. 交付与核验

- 数据总入口：`{ep.name}`；来源库存、各池Train/Val、真实关系与512候选、已有人工证据均可追溯。
- 场景/难度/角度/划分统计：`SYNTHETIC_SCENE_DISTRIBUTION`、`SYNTHETIC_DIFFICULTY`、`SYNTHETIC_2D_ANGLE_DISTRIBUTION`、`REAL_ANGLE_COVERAGE`、`TRAIN_VAL_COUNTS`，文件均带分钟尾缀。
- 原生成器及QA复用，没有改旧库；新增静态监督旁文件补齐被截断舰船排除。未使用M2/B结果选数据，未运行191/900算法历史实验，未启动网络训练或GPU任务。
- 原3项冻结清单中的{len(fr['files'])}个文件当前指纹全部一致，原盲包未重新生成；原标签/人工记录未写入。环境缺少Shapely的首次真实审计导入失败已保留日志，改用现有Rasterio/OpenCV的局部UTM凸足迹交集完成元数据筛选，没有安装环境依赖。

合成数据数量目标已经完成；真实部分完成了可追溯库存和候选准备，尚缺真实双人证据及角度段组件覆盖，不能对外称真实精度评价集已完成。
'''
 report_path=R/f'DUAL_DATASET_REPORT_{T}.md';text(report_path,report)
 # Final identities cover new manifests/summaries/code; no mass hash scan of original imagery.
 tracked=[ep,*paths.values(),cp,latest('REAL_ACQUISITION_RELATIONS_*.jsonl'),latest('REAL512_EXISTING_CROPS_*.jsonl'),
  latest('SYNTHETIC_SOURCE_POOL_*.jsonl'),latest('STATIC_SUPERVISION_VERIFICATION_*.json'),*sorted((R/'code').glob('*.py'))]
 save(R/f'DELIVERY_VERIFICATION_{T}.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
  passed=True,synthetic_geometry_QA_pairs=5961,static_supervision_QA_pairs=5961,reader_representatives_train=41,reader_representatives_val=32,
  frozen_C_files_verified=len(fr['files']),frozen_C_package_not_regenerated=True,
  actual_human_annotations_generated=0,real_GT_generated=0,model_inference_runs=0,network_training_runs=0,GPU_used=False,
  additional_downloads=0,ports_started=[],original_large_SHA_audit=False,
  source_index_expected_sha256=resource['input_index_sha256'],
  files={str(p.relative_to(R)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in dict.fromkeys(tracked)},
  report=str(report_path),counts=str(R/f'DUAL_DATASET_COUNTS_{T}.json')))
 print(json.dumps(dict(report=str(report_path),entrypoints=str(ep),timestamp=T),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
