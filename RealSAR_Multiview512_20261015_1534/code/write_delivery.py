"""Write evidence-linked delivery documents; never manufactures annotation evidence."""
from common import *
from collections import Counter
import shutil

def textfile(p,s):
 with Path(p).open('x',encoding='utf-8') as f:f.write(s.strip()+'\n')

def main():
 t=stamp();statsfile=latest('FINAL_DATASET_STATISTICS_*.json');s=json.loads(statsfile.read_text());st=s['statistics_version']
 pairs=Path(s['pair_manifest']);allrr=rows(pairs);rr=[r for r in allrr if r['qa_state']=='passed'];pmap={p['sample_id']:p for p in rr}
 delivery_checks=latest('DELIVERY_CHECKS_*.json');exclusions=latest('BLIND_PACKAGE_EXCLUSIONS_*.csv');quarantine=latest('HISTORICAL_RESAMPLING_QUARANTINE_*.jsonl')
 ai=rows(ROOT/'AI_PHYSICAL_SCREENING_20261015_1642.jsonl');index=ROOT/'annotation_20261015_1644/PACKAGE_INDEX_20261015_1644.jsonl';packages=rows(index)
 intake=sorted(ROOT.glob('gt_intake_*/GT_QA_RESULTS_*.json'))[-1];gt_dir=intake.parent;gt=next(gt_dir.glob('REAL_GT_MANIFEST_*.jsonl'));gtstatus=next(gt_dir.glob('GT_STATUS_*.jsonl'));q=json.loads(intake.read_text())
 assert q['actual_return_versions']==q['qualified_pairs']==0 and gt.stat().st_size==0
 gt_root=ROOT/f'REAL_GT_MANIFEST_{t}.jsonl'
 with gt_root.open('xb') as f:f.write(gt.read_bytes())
 report=ROOT/f'REAL_MULTIVIEW_512_REPORT_{t}.md';usage=ROOT/f'REAL_ANNOTATION_USAGE_{t}.md';qa=ROOT/f'REAL_GT_QA_REPORT_{t}.md';entry=ROOT/f'REAL_MULTIVIEW_512_ENTRYPOINTS_{t}.json'
 preview=ROOT/f'previews_{t}';preview.mkdir()
 # Administrator illustrations only. Independent packages never include these figures.
 figures=[]
 for title,ids in [('FIXED_STRUCTURE_CANDIDATES',['V040','V058']),('VESSELS_AND_PHYSICAL_LIMITS',['V053','V078'])]:
  sheet=Image.new('RGB',(1044,1138),'white');draw=ImageDraw.Draw(sheet)
  for row,rid in enumerate(ids):
   a=next(x for x in ai if x['review_id']==rid);p=pmap[a['sample_id']];y=row*569
   draw.text((10,y+8),f'{rid}  {p["sample_id"]}  AI {a["provisional_class"]} | azimuth delta {p["azimuth_delta_deg"]:.3f} deg',fill='black')
   draw.text((10,y+28),'Native pixels shown 1:1 | Administrator illustration | NOT physical GT',fill='black')
   for side in [0,1]:
    with Image.open(p[f'image{side}']) as im:sheet.paste(im.convert('RGB'),(8+side*520,y+52))
  dest=preview/f'{title}_{t}.jpg';sheet.save(dest,quality=94,subsampling=0);figures.append(dest)
 angle='\n'.join('| '+ ' | '.join(str(x[k]) for k in ['angle_bin','pairs','relations','acquisitions','site_components','AI_local_candidates','independent_GT'])+' |' for x in s['angle'] if x['split']=='All')
 splitangle='\n'.join('| '+ ' | '.join(str(x[k]) for k in ['split','angle_bin','pairs','relations','site_components'])+' |' for x in s['angle'] if x['split']!='All')
 components='\n'.join(f'| {a["site"]} | {a["split"]} | {a["pairs"]} | {a["relations"]} | {a["acquisitions"]} | {a["AI_local_candidates"]} |' for a in s['site'])
 target=list(csv.DictReader((ROOT/f'PRIORITY_SOURCE_OUTCOMES_{st}.csv').open(encoding='utf-8-sig')))
 targettable='\n'.join(f'| {r["site"].replace("Port of ","")} | {float(r["angle_degrees"]):.3f}° | {r["generated_512_pairs"]} | 已读取双侧 | 0 |' for r in target)
 freeze=FIXED/'FREEZE_20261014_1810.json';protocol=FIXED/'EVALUATION_PROTOCOL_20261014_1754.json'
 textfile(report,f'''# 真实多视角SAR-512建设与可评价性报告

生成时间：{now()}。结果目录：`{ROOT}`。本报告只描述真实不同采集；未改动合成V2.2，未训练、运行匹配器或重复历史模型实验。

## 1. 当前可交付的结果与尚未完成的目标

当前严格原生候选为**1033组、2066个视图实例**：194组历史引用加839组新原生裁片，来自**137个实际采集关系、126个在用采集身份、20个站点组件**。总归档仍为1036组/140关系，未删除任何文件；其中3组旧Sentinel配对有既往双线性对齐记录，已隔离出正式纯图像入口。此修正取代19:36草稿的1036组全通过表述，不改变原数据或历史评价。

本轮新增取得57个不同采集的原生窗口；全库存有130个影像采集身份，其中3个只在上述隔离关系中使用，1个新取得的汉堡采集尚未组成配对。459条元数据中329条仅有元数据，不能计为影像下载。

**合格独立物理GT、正式精度可评价配对均为0。** 本轮没有真实人员提交标注，没有模拟人员、点击、复核或地理GT。图像数量目标已经达到，但“数百组可靠GT、论文定量评价集”的目标尚未完成。当前材料可用于纯图像运行及人工标注，不能据此计算真实Success或配准误差。

84组实际双图AI浏览得到B类局部结构候选40组、C类弱静态支持候选37组、E类不确定7组；其余949组未逐组判读，原生候选整体为B=40、C=37、E=956，A=0、D=0。**这些是AI辅助研究判断，不是真人标注或共同物理点认证。** 不根据对比度代理或港口名称自动宣称可定位6个地面点。

已制作1033项匿名离线材料：42批，每批不超过25项，A/B共84个包。其中3项历史Sentinel任务已列匿名跳过清单`{exclusions.name}`，保留包而不重建；其余1030项用于本轮原生候选。C003/C002/C001只引用原冻结盲包，不重新裁切或重新打包。首批50组不含上述隔离任务，涵盖46个关系、17个站点：40组B候选优先，另10组E。50是待人工先导任务数，不是GT合格数。

## 2. 来源获取与真实身份

起点是历史197组、78个关系、73个采集身份和15个站点。身份按原始collect ID统一，产品别名、处理副本、不同裁片不增加采集计数。新增62个关系后归档140个关系，隔离3个既往预对齐关系后为137个原生关系，不能理解为137个独立场景；原生共享采集图有34个连通组件，按整个站点合并则20个更保守的划分组件。

定向公开获取计划为62个新增采集，实际取得57个；余5个汉堡来源的访问失败和已恢复的失败历史均保留。下载使用公开COG范围请求，只落56个4096窗口与1个3072窗口，未下载整库或完整大产品。失败后未伪称成功，也未用副本凑身份数。Capella公开资产和本地Sentinel来源已核查，本轮新增栅格均为Umbra；没有重新下载既有17份Native。

许可证：Umbra与Capella公开数据沿用各自[Umbra AWS来源](https://registry.opendata.aws/umbra-open-data/)和[Capella AWS来源](https://registry.opendata.aws/capella_opendata/)的CC-BY-4.0及源元数据；历史Sentinel RTC沿用原Indigo Atlas许可，不推断其为相同许可。匿名盲包的管理员映射保留原始身份和来源，用于公开论文时补全归属。

已有八项优先关系均取得实际双侧栅格并生成裁片，结果如下；每个关系的12组是不同窗口，不能当作12个独立来源。

| 来源 | 原生观测方位差 | 512配对 | 栅格状态 | 合格GT |
|---|---:|---:|---|---:|
{targettable}

已下载鹿特丹75.635°和安特卫普88.394°也纳入实际浏览与有界裁片；不同鹿特丹关系共享采集，仍统一归入同一站点Train组件。完整采集ID、时间、URL、原始元数据、组件与关系见`ACQUISITION_SOURCE_RELATION_{st}.csv`、`REAL_RELATION_COUNTS_{st}.csv`和入口所指采集清单。

## 3. 角度覆盖：真实元数据，但供应商定义不能混用

下表是同供应商原始观测方位字段的最小圆周差；不是二维图像旋转，也不冒称全部为统一目标点雷达视线差。按[Umbra官方几何说明](https://docs.canopy.umbra.space/docs/tasking-geometries)和[采集元数据说明](https://docs.canopy.umbra.space/docs/collect-metadata)，Umbra字段以星下点朝场景的方位为定义；[Capella官方参数](https://support.capellaspace.com/what-are-capellas-tasking-parameters)则描述目标到卫星的方位。没有可靠目标方位字段时，Umbra的`radar_los_azimuth_degrees`留空，不擅自加180°。平台航向不从升降轨或左右视猜测。入射角、轨向、侧视和原始字段分别保存；无依据者留空。

| 方位差范围 | 裁片对 | 真实关系 | 采集身份 | 站点组件 | AI局部候选 | 合格GT |
|---|---:|---:|---:|---:|---:|---:|
{angle}

0°–120°主范围878组，其中60°以下673组；>120°额外挑战141组；未知14组独立保留。另3组隔离的旧未知角度任务不计入此表。0°–15°中有18组、4个真实关系的元数据差为0°，属于不同时间同方位控制，不能冒称非零跨角。采用左闭右开分箱，120°包含在90°–120°。

与参考目标150/150/300/240/160相比，小角度较多，30°–60°少68组、60°–90°少152组、90°–120°少43组。并未为了凑目标复制、旋转或移动十几个像素重复裁片。尤其Val的90°–120°仅来自1个站点、2个关系，仍不足以做多站点大角度泛化结论。

| 分区 | 范围 | 裁片对 | 关系 | 站点组件 |
|---|---|---:|---:|---:|
{splitangle}

## 4. 场景与舰船：已确认内容和地点背景分开

929/1033组具有海事来源背景，占89.93%，**超过期望70%–80%，也不等于929组可见舰船**。其余104组属于其他陆地来源背景。这是来源统计；多数512局部尚未完成语义审核，不能仅凭位于港口就分入密集舰船类。

84组AI实际浏览：港口固定结构候选28、地面/工业结构16、低纹理水面35、近岸舰船2、港口密集舰船2、水面舰船且岸基支持未确认1。AI认为至少一侧可见舰船5组，其中双侧4组；真人确认数量0。949组未判读，不能推断有船或无船。Train已浏览54组，Val已浏览30组；目前5组可见船线索均在Train，Val舰船覆盖仍是缺口。

离岸纯海面/舰船、近岸舰船、港口密集舰船、其他陆地必须继续按**图像内容**核查，而不是按站点预填。船可能运动，船体相似或两图均有船不能成为静态地面GT。C类37组保留为`OVERLAP_WITHOUT_RELIABLE_SUPPORT_AI`，属于待真人核验困难关系，**不是已确认负例**。本轮没有经独立几何证据确认的无重叠关系，`NO_OVERLAP_CONFIRMED`清单为空，未虚构D类。

![原生双图：固定结构候选](previews_{t}/{figures[0].name})

图1（管理员研究材料，不交盲标者）：上为杰贝阿里24.937°附近的道路/围界候选，下为鹿特丹75.635°附近的码头转角候选。两者仍需真实人员排除储罐/建筑、船与方向散射风险，不含已确认物理对应点。

![原生双图：舰船及物理对应限制](previews_{t}/{figures[1].name})

图2：上为鹿特丹可见长船但缺可靠岸基支持的C候选；下为Sunny Isles同方位码头控制，双图可见密集船只，仅固定岸基才可能作为GT。船只可见性不授予静态GT资格。图像为原生512像素，图版文字及JPEG仅用于报告。

## 5. 裁片、几何与技术验收

839组新增候选来自已有Native和新增范围读取窗口，以地理共同视场、时间、方位覆盖及粗尺度结构支持进行有界搜索。粗对比度只能发现内容，不能证明同一静态物体。使用640像素候选步距，每真实关系总数上限12，任一侧与同关系既有窗口IoU超过0.2即不选；裁片方案先落盘，再输出图像，后续没有因GT或算法结果重裁。所有历史失败、未明与弱结构候选均保留。

每侧保存原始采集ID、产品窗口原点、源上下文局部偏移、原生TIFF、有效掩膜、像素间距、产品分辨率、CRS/Affine和显示参数。原生值与父图实际窗口逐像素一致；算法PNG仅作逐像素辐射显示转换，未空间重采样、上采样、旋转或地理预对齐。原始像素间距与物理分辨率分开保存，不将GEC投影间距当成独立定位精度。

地理信息只用于离线选区。51个新增采集采用厂商GEC参考高程下的粗Affine，6个明确使用原RPC及厂商参考高度来搜索共同视场；不是DEM正射产品，也不是实测地面高度。依据[Umbra GEC产品规范](https://help.umbra.space/product-guide/umbra-products/umbra-product-specifications/geo-ellipsoid-corrected-gec-geotiff)，高度/叠掩及散射中心变化仍需实物核验。数学往返误差只说明坐标计算自洽，不是物理对应误差。

技术验收累计实际读取归档1036组/2072视图：首1000组完成读取，追加36组读取后，旧1000组依据既有通过记录及未变化文件路径/时间复用；不是再次全盘读取或重复SHA原始SAR。有效像素比例最低0.98025，常量输入0，读取/像素技术失败0。839组新增裁片均通过原始窗口值/掩膜一致性检查。

随后追溯旧Sentinel源清单，发现3组PILOT关系的部分影像有`bilinear_to_first_scene_map_grid`记录。因此“可读取”不等于“原生合规”：本版保留1036组归档，但原生入口只发1033组。证据为`real_sar_pilot/sentinel1_20261010_1731/triplet_verified/manifest.json`，隔离表为`{quarantine.name}`。原影像、旧结果、旧包均未更改；未重新下载或用地图同网格推断精确物理GT。

空间去重：原生候选同关系双侧窗口完全重复0，任一侧IoU>0.2重复0。跨不同关系但共享同次采集时，2066视图实例中有1882个不同原生窗口；记录606条窗口IoU>0.2依赖边，其中268条>0.7、229条为完全复用的窗口边。这些不是新增独立观测，全部在同一分区。不同对手采集形成的真实关系保留，统计/置信区间必须按关系与站点聚合，不能把1033裁片当成1033独立场景。详见`CROSS_RELATION_WINDOW_DEPENDENCY_{st}.jsonl`。

实际Chromium 145只读测试了首2批A/B共4个离线包，图像自然尺寸512、可切换任务、地理/算法提示不可见、JavaScript错误0；没有填写身份、点击点或触发保存。回收与GT流程实际空输入运行结果为0人/0版本/0GT。身份核查、同人别名拒绝、冻结512空间规则、缺不确定度拒绝和纯图像输入白名单均有自动检查。

曾发生公开源TLS/超时及GEC/RPC诊断阻断，原失败记录保留；确认官方GEC与RPC参考面语义后，对6个窗口显式采用RPC参考面搜索，没有把诊断误差当GT或放宽物理门槛。剩余5个计划来源未取得影像。未使用GPU或启动训练，未修改8772/8773服务。

## 6. Train/Val隔离与评价资格

Train：747组、112关系、92采集、15历史站点。Val：286组、25关系、34采集、5新增站点（新加坡、巴拿马运河、香港、萨凡纳、汉堡）。隔离的旧3组另作历史保留，不进入Train算法输入。跨分区的站点、采集身份、关系交集均为0。旧诊断或辅助使用过的站点均归开发Train；旧`candidate_validation`字段只保留历史含义，不能覆盖本版`split`。

| 站点 | 分区 | 512组 | 关系 | 采集 | AI局部候选 |
|---|---|---:|---:|---:|---:|
{components}

Val是来源隔离的候选验证池，尚无独立GT，不能称为已完成独立精度验证。人工质量审查不等于算法验证；一旦用某组件GT调整模型，必须登记开发暴露，不再宣称该组件为未见验证。最终算法推理入口`IMAGE_ONLY_Train_{st}.jsonl`、`IMAGE_ONLY_Val_{st}.jsonl`每行仅含不透明ID和两个图像路径；`code/reader.py`只返回双图数组，禁止把管理员完整manifest送入模型。16:23等旧入口为保留历史，不是本版正式入口。

## 7. 物理GT规则与下一步实际操作

严格沿用冻结协议`{protocol}`，SHA-256为`{sha(protocol)}`，没有调整门槛。

| 资格 | 检查点 | 双侧凸包 | 双轴跨度 | 空间及结构支持 |
|---|---:|---:|---:|---|
| 主全幅稀疏 | ≥6 | ≥10% | 各≥50% | ≥3结构组，每组≤2点；点距≥32px；≥4个4×4网格、≥3象限 |
| 次局部稀疏 | ≥4 | ≥5% | 各≥25% | ≥3结构组，每组≤2点；点距≥32px；≥3个4×4网格 |

均要求两名真实不同人员独立定位；身份/别名与既往地理、算法、提示和他人点位暴露由管理员核查；每点提供地面固定结构、可见性、物理理由、两侧不确定度、风险排除和二维平面适用性。已看过Geo Assist者可做辅助开发，换裁片/改ID/关闭模式不能恢复原站点独立资格。

保留A原点击及B独立重点击，不自动平均。争议>3px需实际仲裁；合并不确定度U按原协议计算，只有U≤阈值/3才能支持3/5/10px对应阈值。少量可靠点未达空间支持时保留稀疏证据，不能强迫凑点或授予全幅H资格。非平面/高度风险区域不得硬算全图配准成功；稀疏、局部和全幅分开。

检查点不得用于拟合或初始化待评价预测。若另有至少3个control与4个check，可按原协议作控制点拟合、独立check诊断；这只是二维适用性检查，不是纯图像算法成绩。

**现在需要用户安排两名未接触对应辅助信息的真实人员。** 人员A接收B001/B002的A包，人员B接收同批B包，各自解压双击HTML、如实填身份/暴露、独立标记或拒绝；回收所有JSON版本和物理证据CSV。管理员核实身份与别名，并依据物理文字证据关联两人的同一结构，再运行收集与QA。完整可执行命令见[{usage.name}]({usage.name})。未安排人员前，不启动8772或替用户虚构标注。

历史13项辅助筛选（A7/B2/C4）、40版本、26旧浏览中心、旧5任务22次点击均保留在原位置，既未清空也未补造；它们未被自动升级或移植为独立GT。新GT状态表按本版未回收状态记录，与旧辅助记录分离。

## 8. 仍然缺少什么

第一缺口是**真实人员对共同地面结构的独立点击与复核**，其次是窗口本身是否有足够空间分散的地面结构。40个AI B候选不保证达到局部门槛，500组GT的可达性目前没有真人通过率证据。先导50组结束后，应公开通过/拒绝及各场景、站点、角度的分母；原失败窗口不移动、不抹除。

数据覆盖还缺：经图像确认的离岸/近岸舰船内容、Val可见舰船、大角度多站点地面支持，以及经可靠几何核验的无重叠负例。若后续需定向新增，应寻找短时、不同观测方向、低矮岸基与船共存、512原生范围可容纳多个地面交汇结构的新站点，优先补60°–120°，而不是继续从同一屋顶/水面反复裁片。本轮不再追加下载或裁片以掩盖这些缺口。

当前可直接交付算法方的是冻结纯图像及来源隔离清单，可作运行与拒绝行为观察；正式物理误差、正确匹配数量、Success必须等待相应GT资格通过。所有主要入口见[{entry.name}]({entry.name})，GT状态和技术图像合格数量始终分别报告。
''')
 textfile(usage,f'''# 真实SAR-512人工标注：立即可执行的分发与回收

时间：{now()}。当前真人提交0，合格GT0。所有路径以`{ROOT}`为结果目录。无需服务器或端口，不重启8772/8773。

## 1. 管理员先做的事

安排两名实际、不同且未看过本样本地理/算法/他人点提示的人员。A/B只是分发槽位，人员ID必须由真实人员确定，不自动生成身份。核查稳定ID、别名、既往暴露及独立工作条件；在`annotation_20261015_1644/PERSONNEL_ASSIGNMENTS_TEMPLATE_20261015_1644.csv`的**新副本**填写实际信息，保留原空模板。`identity_verified`、`exposure_audited`、`independent_work_verified`逐项写yes或no，并填`verified_by`、`verified_at`。actor必须与实际分配的person_id一致；同一人别名不能充当两个人。

Geo Assist历史actor=1的相关站点暴露保留，改窗口不能恢复该站点盲标资格。已暴露人员仍可提交辅助开发标注，必须如实声明，不能纳入独立GT。实际身份管理员应检查更广的历史算法提示，自动记录只能提供已有暴露证据，不能替代人员访谈。

只把各人的ZIP交给本人；**不要给管理员报告、图版、源manifest、地理数据、PACKAGE_INDEX或另一人的点文件**。不得先向人员发送本报告中的候选结构位置再称其为盲标。

## 2. 首批50组的实际文件

| 给谁 | 第1批25组 | 第2批25组 |
|---|---|---|
| 人员A | [B001 A](annotation_20261015_1644/B001_20261015_1644_A.zip) | [B002 A](annotation_20261015_1644/B002_20261015_1644_A.zip) |
| 人员B | [B001 B](annotation_20261015_1644/B001_20261015_1644_B.zip) | [B002 B](annotation_20261015_1644/B002_20261015_1644_B.zip) |

全部42批、84包已存在，管理员索引为`{index}`。当前先分发上述两批，取得真实通过率后再顺序开展，50不是合格GT配额。首两批不含隔离任务；后续批次须附`{exclusions.name}`，人员按匿名task_id跳过3项历史预对齐任务（A/B各列一条）。这些任务包保留但不能用于本轮原生GT。旧C001/C002/C003不在新包中；只使用原`{FIXED/'blind512_materials_20261014_1810.zip'}`，不得新裁或重生成盲包。

## 3. 人员A和B各自的最短操作

1. 解压自己每个ZIP，保留同目录的PNG与3个JS，双击`blind_annotation_20261015_1644.html`，用Chrome/Chromium或Edge离线打开。填写本人的稳定ID、别名、暴露声明并确认实际本人操作。
2. 选择匿名任务，在左图和右图独立寻找同一个低矮固定地面结构，点击两侧。坐标为0基像素中心，x列/y行，0–511。图像不旋转、不地理对齐。
3. 点击“逐点填写结构依据”，填写地物文字、静态和双侧可见确认、两侧实际不确定度和中/高置信。仅按Enter快存但缺这些字段的点仍然不合格。不能为了凑数点船、集装箱、屋顶亮斑或阴影边界。
4. 填同包`POINT_EVIDENCE_TEMPLATE_20261015_1644.csv`的新副本，每个点一行：package_id、匿名task_id、point_id必须与导出JSON一致；填结构组/类型/物理理由，地面固定/低高度/二维适用性，以及高度、叠掩、阴影、动态和散射歧义。每项必须如实yes/no，不可空填当成无风险。
5. 保存会生成浏览器下载JSON，必须确认下载落盘，保留**所有版本**。完成或不可判读时也通过任务结束判断导出，写具体原因。刷新/关闭会清空页面内存，本工具不承诺导入恢复旧点；因此建议一项任务在同一会话完成，下载文件全部回收。
6. B只在自己的干净包独立寻找和点击，不看A的坐标、带点截图、CSV位置答案。二人可以独立选择不同点；后续由真实管理员按结构描述判断哪些确为同一物理结构，而不是按坐标最近强配。

每人交回：`human_observations_*.json`所有版本、本人填写的点证据CSV、实际ID/别名与暴露说明。找不到可靠同点允许拒绝，0点且有原因也应回收，不能隐去失败任务。

## 4. 管理员关联、收集、QA

先在`PHYSICAL_POINT_LINKS_TEMPLATE_20261015_1644.csv`的新副本填写实际关联。`sample_id`由管理员任务索引映射；A/B_package_id及各自point_id来自实际返回。依据物理结构文字与双人实际确认关联，填`physical_identity_confirmed=yes`、真实`structure_group`、`linked_by`、`linked_at`。B没有点到同一结构就暂缺独立复核，不靠近邻或地理预测补齐。分歧>3px要真实仲裁并留下记录；即使填写conflict_resolved也不自动降低不确定度。

以下命令中的`/path/to/...`必须替换为真实回收路径。输出目录以执行时北京时间命名，工具拒绝覆盖已有目录；不要输入虚构标注进行“通过演示”。

```bash
cd /home/yl/matching/sar_matching
conda activate match
export CUDA_VISIBLE_DEVICES=''
run_tag=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
result_dir=/home/yl/matching/sar_matching/RealSAR_Multiview512_20261015_1534
python -B "$result_dir/code/annotation.py" collect \\
  --inputs /path/to/returned_A /path/to/returned_B \\
  --out "$result_dir/received_$run_tag"
python -B "$result_dir/code/gt_qa.py" \\
  --pairs "$result_dir/{pairs.name}" \\
  --package-index "$result_dir/annotation_20261015_1644/PACKAGE_INDEX_20261015_1644.jsonl" \\
  --received "$result_dir/received_$run_tag" \\
  --registry /path/to/ACTUAL_PERSONNEL.csv \\
  --evidence /path/to/ACTUAL_POINT_EVIDENCE_A.csv /path/to/ACTUAL_POINT_EVIDENCE_B.csv \\
  --links /path/to/ACTUAL_PHYSICAL_POINT_LINKS.csv \\
  --out "$result_dir/gt_review_$run_tag"
```

每个包的回收目录顶层放实际JSON；不递归扫描。collector保留原内容和SHA，不自动认证；相同时间重复收集须用新的输出目录。QA保留所有实际返回版本与原点，只选每任务最高revision审核，其他版本仍在原始索引中；如同revision出现不同内容，应由管理员先明确版本，不把冲突当作稳定GT。

正式门槛按原冻结`{protocol}`。输出`REAL_GT_MANIFEST_*`可能包含少量经双人核验但空间不足的稀疏记录，**仍须筛选GT_qualified=true**，并区分primary/local。`GT_STATUS_*`说明未标注、辅助、身份待核、独立待二人/关联、独立复核空间不足或通过；`point_failures`列出点拒绝原因。不能直接把返回JSON中自带的资格字段用于评价。

## 5. 直接交给算法方

先交纯图像入口`IMAGE_ONLY_Train_{st}.jsonl`和`IMAGE_ONLY_Val_{st}.jsonl`、`code/reader.py`；函数`load_images`返回两个uint8数组，不提供地理、姿态、GT初始化。当前仅能运行无标签测试或不使用对应标签的开发，不能以无GT记录训练有监督物理对应或算真实精度。获取实际合格GT后，管理员另发资格清单给评价程序，GT不得进入推理或拟合预测。
''')
 textfile(qa,f'''# 真实物理GT验收状态

生成：{now()}。本轮真实人返回版本0、人员0、实际独立复核点0、全幅合格0、局部合格0、正式精度可评价0。历史辅助记录未导入、未抹去、未重新认证。

本次实际执行空回收检查：`{intake}`。`{gtstatus}`中1033个原生候选为unannotated，3个旧任务为quarantined_input_not_formal；空物理清单`{gt_root}`不是缺失文件，也不是零误差标签。`GT_FULL_QUALIFIED_{st}.jsonl`、`GT_LOCAL_QUALIFIED_{st}.jsonl`、`GT_REJECTED_ACTUAL_{st}.jsonl`均为空。AI判为C的37组另列辅助困难表，不能作为确认负例；确认无重叠表亦空。

已运行的技术/规则检查见`{delivery_checks.name}`；实际浏览器检查`BROWSER_QA_20261015_1704.json`。它们验证输入、加载和拒绝规则，不是对真人标注真实性的替代验收。没有伪造人员来测试通过路径，真实返回后仍必须运行完整流程并由管理员复核语义判断。

冻结协议：`{protocol}`，SHA-256 `{sha(protocol)}`。主要门槛：主全幅≥6check、双侧凸包≥10%、双轴各≥50%、≥3组/每组≤2、点距≥32、4网格/3象限；局部≥4check、凸包≥5%、双轴各≥25%、≥3组/每组≤2、点距≥32、3网格。声明与重点击不确定度合并，不默认填sigma；按U≤threshold/3区分3/5/10px可支持阈值。主/局部分母单独统计。

QA先核实际包/图像SHA、身份/别名、管理员核查及历史暴露，再核两次真实坐标、有效掩膜、显式结构证据和风险、独立重点击分歧。保留A原始坐标及B复核，不平均；地理点或浏览中心不进入GT。实际管理员按物理结构关联并记录第三方仲裁，自动几何接近不能授予物理身份。

可靠稀疏点未达空间条件时保留为待支持或稀疏对应，不算配准成功。二维平面不适用时不得强算局部/全幅H。只用control做几何适用性诊断、check只评价，二者不可混用；算法模型/预测选择必须先于打开GT固定。

需要用户安排的真实工作见[{usage.name}]({usage.name})。目前第一阻塞是实际人员及共同地面结构证据，不是工具数量。
''')
 pilot={slot:[p['zip'] for p in packages if p['slot']==slot and p['batch_id'].startswith(('B001_','B002_'))] for slot in ['A','B']}
 e=dict(created_at=now(),release_id=ROOT.name,root=str(ROOT),state='image_QA_complete_independent_physical_GT_pending_actual_people',report_md=str(report),report_pdf=str(report.with_suffix('.pdf')),annotation_usage=str(usage),gt_qa_report=str(qa),counts=str(statsfile),pair_manifest=str(pairs),acquisition_manifest=s['acquisition_manifest'],GT_manifest=str(gt_root),GT_status=str(gtstatus),GT_actual_intake_QA=str(intake),qualified_GT_count=0,formal_evaluable_pairs=0,train_image_only=str(ROOT/f'IMAGE_ONLY_Train_{st}.jsonl'),val_image_only=str(ROOT/f'IMAGE_ONLY_Val_{st}.jsonl'),train_admin_unlabeled=str(ROOT/f'REAL_Train_UNLABELED_{st}.jsonl'),val_admin_unlabeled=str(ROOT/f'REAL_Val_UNLABELED_{st}.jsonl'),prediction_reader=str(ROOT/'code/reader.py'),prediction_allowed_keys=['sample_id','image0','image1'],prediction_arrays_only=True,supervised_physical_training_ready=False,pure_image_unlabeled_running_ready=True,full_GT_qualified=str(ROOT/f'GT_FULL_QUALIFIED_{st}.jsonl'),local_GT_qualified=str(ROOT/f'GT_LOCAL_QUALIFIED_{st}.jsonl'),actual_GT_rejected=str(ROOT/f'GT_REJECTED_ACTUAL_{st}.jsonl'),no_overlap_confirmed=str(ROOT/f'NO_OVERLAP_CONFIRMED_{st}.jsonl'),weak_support_AI_only=str(ROOT/f'OVERLAP_WITHOUT_RELIABLE_SUPPORT_AI_{st}.jsonl'),uncertain_or_unreviewed=str(ROOT/f'UNCERTAIN_OR_UNREVIEWED_{st}.jsonl'),original_frozen_protocol=str(protocol),original_frozen_protocol_sha256=sha(protocol),frozen_C001_C002_C003_package=str(FIXED/'blind512_materials_20261014_1810.zip'),old_freeze=str(freeze),annotation_package_index=str(index),annotation_pilot50_zips=pilot,assignment_template=str(index.parent/'PERSONNEL_ASSIGNMENTS_TEMPLATE_20261015_1644.csv'),physical_link_template=str(index.parent/'PHYSICAL_POINT_LINKS_TEMPLATE_20261015_1644.csv'),return_collector=str(ROOT/'code/annotation.py'),GT_QA_code=str(ROOT/'code/gt_qa.py'),browser_QA=str(ROOT/'BROWSER_QA_20261015_1704.json'),pixel_QA=[str(ROOT/'REAL_PAIR_QA_20261015_1620.json'),str(ROOT/'REAL_PAIR_QA_20261015_1623.json')],delivery_checks=str(delivery_checks),historical_resampling_quarantine=str(quarantine),annotation_excluded_task_ids=str(exclusions),native_usable_pairs=1033,archived_including_quarantine_pairs=1036,spatial_dependency_audit=str(ROOT/f'CROSS_RELATION_WINDOW_DEPENDENCY_{st}.jsonl'),statistics_csv=[str(p) for p in sorted(ROOT.glob(f'*_{st}.csv'))],preview_figures=list(map(str,figures)),source_limits=['port context is not vessel evidence','0-120 main878, >120 challenge141, unknown14; 3 prior resampled tasks archived separately','Umbra and Capella vendor-native angle frames are not interchanged','GT remains0 until actual independent people return evidence','Val 90-120 has only one station; visible vessel Val evidence still absent'],training_started=False,GPU_used=False)
 save(entry,e)
 textfile(ROOT/f'README_{t}.md',f'''# 真实SAR-512结果入口

1033原生配对，137真实关系，126在用采集身份，20站点。Train747/Val286。另3组历史预对齐配对保留并隔离。原生技术QA通过；合格独立物理GT0，需要真实人员。

入口：[{entry.name}]({entry.name})。报告：[{report.name}]({report.name})；同名PDF。人工操作：[{usage.name}]({usage.name})。冻结历史未修改，GPU/训练/8772/8773未动。
''')
 print('ENTRYPOINT',entry,'REPORT',report,'USAGE',usage,flush=True)

if __name__=='__main__':main()
