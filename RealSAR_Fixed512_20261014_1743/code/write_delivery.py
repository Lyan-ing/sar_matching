"""Write bounded research delivery and freeze existing final crop/protocol inputs."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,csv,hashlib,zipfile,shutil
BASE=Path('/home/yl/matching/sar_matching')
TOP=Path(__file__).resolve().parents[1]
R=TOP/'release_20261014_1756'
T=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def text(p,s):
 with p.open('x',encoding='utf-8') as f:f.write(s)
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p,rows,fields=None):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)

sources_path=next((R/'custodian').glob('candidate_sources_*.json'))
rows=json.loads(sources_path.read_text());S=sources_path.stem[-13:]
protocol=R/f'EVALUATION_PROTOCOL_{S}.json'
status=R/f'FIXED_CANDIDATES_{S}.csv'
prepare=json.loads((R/f'preparation_verification_{S}.json').read_text())
source_checks=[]
for row in rows:
 for v in row['views']:
  a=v['acquisition'];meta=json.loads(Path(a['metadata_path']).read_text());pr=meta['properties']
  asset=json.loads(Path(v['original_product_profile_metadata']).read_text())
  assert pr['umbra:collect_id']==a['acquisition_id']==asset['acquisition']['acquisition_id']
  assert pr['view:azimuth']==a['azimuth_degrees']
  assert pr.get('start_datetime',pr['datetime'])==a['utc']
  assert asset['original_window_xywh']==v['original_native_parent_window_xywh']
  source_checks.append(dict(candidate_id=row['candidate_id'],side=v['side'],collect_id=a['acquisition_id'],
   metadata_collect_id_and_azimuth_and_start_agree=True,parent_product_offset_agrees=True,
   local_header_pixel_checks_in=f'preparation_verification_{S}.json',no_large_product_SHA_scan=True))
save(R/f'source_identity_verification_{T}.json',source_checks)
table(R/'blind_release'/f'REVIEWER_RESULTS_template_{T}.csv',[],[
 'blind_code','first_person_point_id','reviewer_real_id','reviewer_aliases','reviewed_at','structure_group_id',
 'physical_structure_description','conclusion_confirmed_uncertain_rejected','static_confirmed','visible_both_confirmed',
 'reviewer_sigma_left_px','reviewer_sigma_right_px','height_shadow_layover_risk','reason','own_observation_file','own_point_id'])

instruction=f'''# 固定512双图：两名真实人员的操作说明

本包只采集实际人工观察，尚无GT。仅显示两幅固定512原生网格SAR图；没有地理投影、观测方位、采集身份、算法匹配或候选点。对比度显示不改变坐标。不要从文件管理员索取辅助预览、来源表或地理信息。

1. 解压全部文件，保留同目录文件。用Chrome、Edge或Firefox打开 `blind_annotation_{S}.html`，无需安装环境、服务器或端口。页面第一次进入须填写本人真实姓名或稳定ID、全部既往别名，逐项声明地理、算法、候选点及他人坐标暴露。不得换名字恢复资格。
2. 两名人员各使用独立解压目录和干净浏览器会话。先由第一人自行搜索共同固定地面结构；第二人独立搜索，不接收第一人坐标或画点截图。任务C001/C002/C003的结构好坏未知，不要求每组成功。
3. 左图点一下、右图点一下；点击坐标是该512图的零基原生像素中心（x=列、y=行，0..511）。滚轮放大，Shift或中键拖动，局部放大镜只放大显示。不要估算截图坐标，不从管理员提供的地理中心推点。
4. 进入“高级模式”，逐点填写地物说明（建议以G01/G02等真实物理结构组开头）、静态/两图可辨认、各侧真实定位不确定度、置信度，以及备注中的高度、阴影、叠掩和误认风险。优先地面道路交汇、固定围界/低矮设施边界；罐顶、屋顶亮点、船舶、起重机、集装箱和阴影不能因相似直接对应。默认check，不能把这些点用于模型初始化、选模或接受判断。
5. 点击“导出人工观察”（或Enter）会下载 `human_observations_<C编号>_<本人ID>_v<版本>_<北京时间分钟>.json`。必须确认文件实际落盘，并允许浏览器多文件下载。保存新版本而不是覆盖旧版本。刷新/关页会清空内存，下载文件才是可回收材料；页面合格数始终0。
6. 单独使用“结束任务”说明局部可辨、清楚对应、无法可靠确认或暂停；此步骤也下载新版本。找不到点可以下载零点判断，不要填中心凑数。区域级线索写在CSV/任务说明里，明确“仅区域，不是点级GT”。不能确认静态性时不要勾选确认，不确定度不能填自动零值。
7. 第二人先独立观察。若需要逐点复核，管理员只交第一人的点ID和物理结构文字，不含像素坐标、方向性提示或带点截图。第二人在自己的页面重新点击，导出自己的JSON；逐项填写 `REVIEWER_RESULTS_template_{T}.csv`，包括confirmed/uncertain/rejected、自己的不确定度和风险。如无法找到同一结构，留下疑义或拒绝，不填伪造坐标。
8. 两人分别把**所有版本JSON、任务结束判断及各自CSV**交给管理员。第二人额外交回复核CSV。点ID相同不是同一物理结构证明，管理员应按双方描述和真实证据核对，保留拒绝和争议。

独立资格由管理员核对真实身份、别名、历史暴露和交付过程后确认，本人声明不能代替审核。已暴露者可以如实进行辅助开发标注，文件必须保留assisted状态，不能升级独立GT。两组任务可能共享图像，重复点击不能计为新的独立场景。

本包复用现有离线采集代码，没有自动独立复核/GT升级。页面“全幅”仅指本512窗口的稀疏支持，不能说明8192整景或稠密全像素精度。请只提供实际可确认的点，空间门槛是否达到由管理员核验；不要为凑门槛标不确定点。
'''
text(R/'blind_release'/f'ANNOTATOR_INSTRUCTIONS_{T}.md',instruction)

protocol_md=f'''# 真实512跨角物理对应评价协议 V1（正式标注及算法评价前固定）

机器可读规则：`{protocol.name}`。适用对象是两张已经固定的512×512真实SAR原生裁片，最终推理只输入这两图。此协议是新的512评价范围声明，不修改原8192 Native协议、数据库或历史结果。当前没有真实标注，合格GT=0。

## 1. 冻结对象与算法输入

候选清单、两侧窗口整数原点、DN原生裁片、有效掩膜、固定显示PNG、来源、身份/暴露记录以及本协议一起冻结。无重采样、上采样、旋转预对齐。512不是把8192缩到512；同一512场景也可能有不同方向、入射角与视野。

纯图像算法输入只使用 `image_only_inputs/` 中固定DN PNG。原始像素值与来源TIFF窗口相等，PNG不含CRS、Affine和身份标签。盲标只显示同网格的既有固定dB显示裁片；它由原生DN按已有各图固定显示映射得到，不是新增分辨率，也不能充当绝对辐射可比证据。不得把带CRS的TIFF、地理覆盖、裁片偏移、掩膜、辅助点或GT传入预测/初始化/接受门控。

候选位置只由采集元数据、此前辅助场景线索及本轮静态地面结构复看决定，未读取M2/B等匹配成败作选择。标注后不可移动窗口、改变对比映射或替换样本以获得合格门槛。失败、点不足和未知都保留；如将来确有制作错误，另建有原因的新版本并保留原分母，不悄悄替换。

## 2. 物理证据、坐标与两人独立性

坐标：512图零基像素中心x=列/y=行，0..511；窗口范围用半开区间 `[x0,x0+512)×[y0,y0+512)`。返回保留Native和完整原产品，只分别加对应整数偏移一次。有效像素由双方原生掩膜与有限值检查；NoData点不可用。

逐点必须有真实标注者和时间、明确同一物理结构文字、静态/双方可见确认、中/高置信、两侧定位不确定度。第二名真实独立人员需独立重新定位并确认，身份和全部别名经实际审核。标注或复核人员看过该来源相关地理提示、算法对应或辅助同点提示，正式独立资格不成立；裁片或模式变化不能清除暴露。

岸线/低矮永久码头的固定拐角、地面道路/设施交汇可核验；潮汐水线、阴影边界、抬高屋顶/罐顶、方向敏感强亮点、船舶、箱堆或可动部件不能作为当然地面对应。附近结构相似也不证明同名。真实高度/地面属性无法确认，留疑义、区域级或拒绝。

同一个物理边界/交汇须记录结构组Gxx。一转角附近的密集点击不算多个独立地物。未确认结构数、结构组和坐标不可由Codex填补。

## 3. 预定空间支持分层（两侧分别检查）

|项目|原Native局部规则|原Native全幅稀疏规则|新512主评价：窗口分散稀疏|新512次评价：局部稀疏|
|---|---:|---:|---:|---:|
|独立合格check数|≥3|≥6|≥6|≥4|
|check凸包/评价图像面积|≥0.001|≥0.10|≥0.10|≥0.05|
|横、纵跨度/评价宽高|未单独要求|每轴≥0.50|每轴≥0.50|每轴≥0.25|
|4×4网格占用|原规则记录，不设此门槛|原规则记录，不设此门槛|每侧≥4格，且覆盖≥3象限|每侧≥3格|
|独立物理结构组|需物理证据|需物理证据|≥3组，每组计入check最多2个|≥3组，每组计入check最多2个|
|check间像素距离|原规则拒绝坐标重复|原规则拒绝坐标重复|任意两check在每侧≥32px|任意两check在每侧≥32px|

凸包分母固定为512×512=262144px²，不改为标注小区域、共同地理足迹或掩膜面积。主评价凸包至少26214.4px²，横纵跨度均至少256px；局部至少13107.2px²，跨度至少128px。只有一个转角、共线边界或集中于一小块的点不足以成为主评价正例。

这是任务范围从8192整景改为真实512窗口后的定义变化：新任务不要求窗口外6.7千万像素的支撑，也**不降低**原全幅稀疏评价的点数、比例、跨度或定位精度标准，并新增独立结构与分散性门槛。旧Native全幅指标和新512指标不得拼为同一成功率。

仅有1–3个可信物理点时保留真实证据，不强迫补点，但不授予上述正式空间支持资格。满足局部层但不满足窗口分散层，单列局部，不混入主Success分母。区域级对应不能计算点级像素误差。

## 4. 定位精度与算法指标

沿用原规则：每点令 `u0=max(第一人sigma0,第二人sigma0,左侧实际重点击差)`，右侧同理，`U=hypot(u0,u1)`。它是保守证据界限，不是经标定95%置信区间。分别支持3/5/10px指标，要求所有计入检查点的U≤对应阈值/3；不知道sigma就保留未知，不填0。

原规则对重点击差>3px提示复核，超过 `max(3,3×四个声明sigma的最大值)` 的未解决差异不合格。人核不确定点不可通过几何拟合/算法内点挽救。

模型、预处理和预测变换类型必须在打开GT前锁定；GT check不参与预测、拟合、初始化、选模或接受判断。以冻结纯图像预测映射计算右图原生像素欧氏误差，报告检查点数、mean/median/p90/p95、空间支持与定位不确定度。`sparse Success@t` 沿用accepted且mean≤t的定义；没有输出计为该合格层失败，错误接受如实记录，另列p90不隐藏局部坏点。不具备该阈值定位证据的样本报告unsupported，不能强当成功或失败。

主分母=合格窗口分散稀疏样本，次分母=合格局部样本，另报告候选总数、未标注、辅助、争议、不可评价、资格不足和各阈值可评价数量。未知不是无重叠负例。可按来源组件等权报告，但本轮仅2组件，不能声明广泛泛化或显著性。

## 5. 局部二维几何是否适用

小窗口并不消除高度、叠掩、阴影或方向散射变化。固定地面上可能适合局部二维仿射，但本轮没有人核点证明；Homography也不能保证纠正不可见/非同散射结构。

若真人自然获得≥3个单独指定control且≥4个check，可做预定的地面局部Affine诊断：只用control拟合，check留出；check p90>10px提示点冲突、高度效应或非均匀几何，需物理复核。它不是GT资格证明，不能用check拟合，不据拟合残差删去困难的真实物理点。点不够则不拟合，不强迫标control。此诊断只作为数据审查，不传入图像模型或算法接受判断。

每组最终明确：窗口分散稀疏可评价、局部可评价、仅区域、证据不足、存在可信点但统一二维映射有风险。几何风险不抹去真实物理对应，统一变换局限应保留在评价解释中。凸包只是采样支持，不能宣称其内外全像素都有稠密GT。

## 6. 冻结及实际执行状态

JSON是本批固定规则；完整文件身份见FREEZE清单。原 `gt_quality/quality.py` 不修改。旧后端返回gt_qualified不自动满足本512新增门槛，须在本批记录中逐项核验；当前仅提供输入和协议，没有人核数据可执行正式资格审核或精度评价。将来真实输入的导入/核验写入独立512工作目录，不写旧Native/旧8769数据库。
'''
text(R/f'EVALUATION_PROTOCOL_{T}.md',protocol_md)

actions=f'''# 固定512候选：分发、回收与后续动作

当前交付3组冻结候选，合格GT为0；人核点、第二人复核和算法评价均未执行。请先安排两名真实、未接触相关辅助提示的人员。只把 `blind512_materials_{T}.zip` 发给他们；不要发送本报告、来源清单、custodian目录、辅助预览或旧地理截图。

## 可以直接执行的入口

两人各自解压后打开 `blind_annotation_{S}.html`，阅读包内 `ANNOTATOR_INSTRUCTIONS_{T}.md`。无需Python、match环境或端口；没有启动8772/8773。全部文件必须同目录，不能只复制HTML。先做C003，再做C002，C001为条件性尝试；管理员不用向标注者解释角度、地点或具体候选位置。

第一人：独立点击双方同一静态地面结构，逐点填写物理依据、结构组、不确定度、静态/可见判断和风险，导出实际JSON，结束任务另存判断。第二人：用独立目录/会话自行观察并重点击；必要时只接收第一人结构文字和点ID，不能接收坐标、带点截图和定位提示。第二人在CSV明确confirmed/uncertain/rejected。

## 回收材料

分别建立“第一人原样下载”“第二人原样下载”目录，回收每人全部 `human_observations_*.json` 版本、物理观察CSV和结束任务判断。第二人还回收 `REVIEWER_RESULTS_template_{T}.csv` 的填写版，关联其本人JSON/点ID。保存前后各版本，不能只留页面截图。浏览器刷新会清空内存，不会自动写入项目GT数据库。

管理员填写 `custodian/personnel_distribution_registry_template_{S}.csv`，真实核对身份、别名、包版本、分发时间与各项暴露，关联已有exposure历史。此前Geo Assist操作者1已经暴露，不能凭换ID、关辅助或换512窗口取得同来源盲标资格。已暴露研究人员仍可辅助开发标注，但如实登记assisted，独立GT不合格。

## 回收后的核验顺序

1. 核验包/裁片版本，坐标0..511和有效掩膜；按custodian整数偏移溯源到保留Native与原产品。C002/C003同一左裁片，共享采集与站点；不能跨开发/验证/测试拆分，重复物理结构不能当独立样本。
2. 依据双方真实结构说明确认同一地面物理位置；保留高度、叠掩、阴影、车辆/船舶、箱堆、模糊与重复结构风险。点ID或地理投影相似不足以确认。
3. 对照两人的实际点击和sigma，保留每人原值、拒绝和疑义，不用平均、投影或模型拟合自动修成GT。记录reviewer身份、重点击时间、conclusion、static/visible和实际差异。
4. 按固定 `{protocol.name}` 及本目录协议MD检查空间支持、结构组、定位精度和二维几何风险。空间不足就局部/证据不足，不重新裁切，不凑点。
5. 实际资格通过后，在独立512工作目录保存正式GT及审核版本；原Native/8769标签和结果不改。管理员再将无GT的固定DN双图输入清单交给算法负责人，先锁定预测配置，再打开检查点计算精度。该批已经用于项目诊断与候选观察，不能称未见最终测试。

GT实际入库须真实两人证据。本轮未运行prepare_gt创建假批次，未覆盖原A/B记录的recommend=false，没有自动导入、自动配对或自动资格升级。现有离线工具已可收集实际输入；新增512空间门槛是本批固定审核要求，尚不能由旧后端自动认证。

## 谁还需要做什么

你/材料管理员：安排两名人员、审核资格，只分发盲ZIP，收回实际文件。第一人：点击与物理依据。第二人：独立重点击并确认/疑义/拒绝。回收后本数据任务可继续检查坐标、来源、独立性和固定协议，保存可核验对应。算法负责人：在合格数据形成后按固定双图输入及锁定算法设置评价。没有真人提交之前，无法宣称精度评价集已完成。
'''
text(R/f'ANNOTATION_ACTIONS_{T}.md',actions)

mdtable='\n'.join(f"|{x['candidate_id']}|{x['site'].split(',')[0]}|{x['azimuth_delta_deg']:.3f}°|{x['views'][0]['crop_offset_in_retained_native_xy']}|{x['views'][1]['crop_offset_in_retained_native_xy']}|" for x in rows)
preview='\n\n'.join(f"### {x['candidate_id']} 原生512双图\n\n![{x['candidate_id']} 双图固定显示，非GT](previews/{x['candidate_id']}_preview_{S}.png)" for x in rows)
report=f'''# 首批真实512跨角候选与评价准备结果

本轮完成3组**固定512候选**，来自5个唯一真实采集、2个站点/来源组件；优先2组鹿特丹，条件性1组杰贝阿里。原生DN、有效掩膜、窗口坐标与来源已经核验，离线工具复用既有代码。**这是待真实人员标注的候选，不是已经通过物理核验的正例；新增人工对应0，独立复核0，合格GT0。**

## 既有191组与18项的核查结论

旧191组来自65采集关系、12来源组件：151组/51关系记录为different_look_verified，36组same_look，4组方向未知。151组异向来自7组件，含小方位差；metadata类别不能直接解释为大角差或物理可配准。读取了既有索引、191项工程核查/分类及候选报告，对异向原生双图做按组件的缩略复看；另以原生512复看Umbra_0032__chip_000、Umbra_0066__chip_000、Umbra_0068__chip_001和旧8候选图集。没有重跑191算法实验或大规模SHA核查。

旧审核优先4组精确候选主要是Sunny Isles/SF同向；本轮不把它们混进跨角正例。旧跨角港口512多是箱区、亮线、单条边界或不够明确的方向散射，尚未确认多个分散地面点。Johnston农田0032有跨角道路/地块交会候选，但地块/作业痕迹稳定性未确认，且属于非海事场景，本轮不纳入海事首批主候选。矿区和山地地形不适合仅凭线状相似认定平面点。旧191保持冻结，未把历史22点击迁移为新GT。

最新18关系的真实采集及辅助观察继续引用原记录：13项人工辅助筛选A7/B2/C4、40版本；这些不是精确同名点。重点四组的旧浏览中心和辅助区域只作搜索入口。随后对四组做**19个有界双窗口方案**的原生静态结构复看（10个原辅助区域方案、9个补充道路/围界方案），在标注前选择3个；候选选择未依赖M2/B等匹配成功与否。新区域不是旧用户点击位置的补造，窗口原点也不是物理对应坐标。

## 冻结候选及可定位来源

|代码|站点|方位差|左512原点（保留Native）|右512原点（保留Native）|
|---|---|---:|---|---|
{mdtable}

每侧窗口均512×512。裁片像素(u,v)加上述原点得到保留Native坐标，再加存档parent_window原点得到完整产品坐标。完整5采集身份、STAC/collect/task、时间、方位/入射角、升降轨、左右观测、许可/URL、产品父窗口、CRS/Affine、原生格距和供应商物理分辨率均在 `{sources_path.relative_to(R)}`。像素格距约0.2187m，512边长约112m；它不等于实际空间分辨率，供应商本批range约0.35m、azimuth约0.175m，两者分别保留，不能把像素数当分辨率。

C003与C002共享左采集d1f1f1ba…和同一个左512窗口，两张右图也属于同一鹿特丹站点/task来源组件。其原组件v2_component_011与旧real_component_011是同一站点依赖，不能随机拆分或作为两个独立场景。C001为另一站点组件。全部定位为diagnostic_development_seen，独立人员标注不自动产生模型未见的最终测试。

## 共同静态结构与几何可评价性

**C003，鹿特丹18.882°，首选核验。** 两图可追踪多个地面围界/设施区边界的转折和道路交汇拓扑，比旧港池单转角窗口提供更多二维分散结构候选；18秒间隔减少时相变化。所选位置邻近罐区，地面围界必须由真人区分于圆形罐体轮廓、罐顶亮边、管架和阴影；本轮并没有认定具体亮点为物理同点。候选可能支持局部平面稀疏评价，能否满足6个分散检查点仍未知。

**C002，鹿特丹45.984°，次选核验。** 固定同一左侧场景，在40秒后的另一观测方位可见类似围界与道路组织。它可提供同场景角度增加的研究对照，不能提供新独立场景数。较大视角下设施边缘和高度响应变化明显，需要第二人证明是真实地面交汇而非抬高设施。若只有少量真点，单列局部；不旋转右图预对齐。

**C001，杰贝阿里36.369°，条件性候选。** 补充区域中两图可追踪道路/设施分区边界及交会，避免原辅助窗口仅一处明显暗区转折。但B类真人观察已指出模糊和位置偏离，约23.49小时的时间间隔及邻近罐体散射仍限制可靠定位；它比两组鹿特丹更可能停留局部或区域级。不能把直线交叉的数学交点、箱堆端点或亮点直接当同名地面点。

**安特卫普88.394°暂不纳入本批。** 已查区域的512双图中罐体、连续强亮设施与阴影响应占据较多支持，地面交汇看起来有限；尚未确认可提供多个空间分散的清晰地面锚点。另存在入射角差约20.81°与升降轨/左右观测变化，不能将所有差异只归因于方位角。它仍保留原18队列，不因本轮未选而标无重叠/不可配准负例。

> 上述是Codex在辅助条件下对图像拓扑和风险的研究观察，不是未暴露真人的物理点确认。当前可肯定的是实际图像、来源和搜索线索；至少6个可重复定位的独立点及二维几何适用性必须由两名真人给出证据。没有任何自动生成对应点。

两侧有效像素比例均100%；仅按产品地理足迹计算的覆盖约为C003 88.11%、C002 82.85%、C001 83.40%。这些数值仅供离线制作审查，不含共同散射可见性，也不能作为GT或真实配准精度。纠正版本中两侧裁片中心与原预选的地理中心舍入关系一致，数学一致不代表物理点无偏。

{preview}

## 新512协议与原Native协议的区别

主评价延续原全幅稀疏的≥6个独立合格check、两侧凸包≥10%、横纵跨度各≥50%和定位不确定度≤3/5/10px阈值的1/3，并增加独立结构组和网格分散性条件。局部层单列，要求≥4check、凸包≥5%、两轴跨度≥25%；不与主分母混合。完整固定规则见 `EVALUATION_PROTOCOL_{T}.md` 和机器JSON。

原8192全幅空间要求属于8192任务；新规则的分母是预先冻结的512窗口，不是标注后缩出的区域。没有降低原主评价比例、点数和精度门槛，也不在标注后再重裁满足覆盖。只有一个转角不会成为合格主正例；不够数量/支持时保留已有真人点但保持资格不足。小窗口不保证二维变换适用，局部仿射只在低矮地面假设有实际证据时使用；拟合残差不能充当GT。

## 文件用途及人工下一步

- `FIXED_CANDIDATES_{S}.csv`：3组固定状态；当前全为未标注。`custodian/`：来源与旧暴露，**不给盲标人员**。
- `raw_crops/`：原生数值TIFF与有效掩膜，只作保管/核验；`image_only_inputs/`：对应原生DN PNG，供未来纯图像算法固定输入，不能同时发送地理字段。
- `previews/`：管理员辅助预览，非GT；盲标人员仅收到ZIP中无位置提示的同网格显示图。
- `blind512_materials_{T}.zip`：真实人员可直接解压打开的离线采集包；行为JS与旧包逐字节相同，仅替换3项任务与5幅显示图，不新增平台。
- `ANNOTATION_ACTIONS_{T}.md`：你如何安排两名真实人员、回收哪些文件，以及回收后的逐项审核；包内也有最简说明和空白CSV。

距离第一批精度可评价数据，还缺两名真实合格人员的实际双图点击、物理结构描述/不确定度、独立重点击与确认，以及管理员按固定512协议核验空间支持和来源/暴露。你安排人并回收实际文件，本数据任务再核验保存；未达到规则者保留局部或不可评价。未启动训练、算法推理、来源下载或任何判读端口，原人工记录/GT/评价结果均不改。

## 制作核验与保留的纠正记录

准备阶段右侧原点出现一次抄录错误：C002草稿写成6334，原预选应为6634，相差300px。1748草稿文件全部保留；本目录为纠正后发布候选，窗口在真人标注和算法评价前纠正，原因记录于顶层DRAFT_WITHDRAWN。不得使用顶层旧草稿做评价。最终6侧裁片的DN逐像素相等、PNG原始DN相等、固定显示裁片相等、有效掩膜与Affine/CRS均已实际核验；来源collect/方位/时间/父窗口与存档元数据相符，没有再次整景SHA扫描。

Chromium本地只读测试检查3任务图像/画布均512、未知暴露拒绝进入、无地理/算法图层、原生坐标转换、刷新空记录、无外部请求，无真人身份或保存。测试脚本初读折叠元数据用innerText导致断言失败，改为textContent后复验；执行沙箱限制下浏览器保留自身沙箱并在批准的外层环境运行。实际结果以本目录blind_browser_check文件为准，浏览器测试不是人工物理核验。
'''
text(R/f'REAL512_PREPARATION_REPORT_{T}.md',report)
text(R/f'PREVIEW_{T}.html','<!doctype html><meta charset="utf-8"><title>固定512候选，非GT</title><style>body{font-family:sans-serif;max-width:1100px;margin:auto;background:#171d26;color:white}img{width:100%;height:auto}p{line-height:1.5}</style><h1>固定512候选 · 管理员辅助预览，不是GT</h1><p>不要将此页或来源报告发给盲标人员。仅分发blind512 ZIP。</p>'+''.join(f'<h2>{x["candidate_id"]} · {x["azimuth_delta_deg"]:.3f}°</h2><img src="previews/{x["candidate_id"]}_preview_{S}.png">' for x in rows))

# Explicitly supersede without changing or deleting the first draft.
save(TOP/f'DRAFT_WITHDRAWN_{T}.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
 affected_draft='top-level raw_crops/image_only_inputs/blind_release and candidate/protocol files dated 20261014_1748',
 reason='C002 right x origin transcription 6334 instead of 6634; caught by geographic preparation consistency before annotation/model evaluation',
 preserved=True,not_formal_frozen_candidate=True,use_release=str(R),human_annotations_at_correction=0,model_evaluations_at_correction=0))
text(TOP/f'README_{T}.md',f'# 固定512交付入口\n\n仅使用纠正后冻结目录 `{R.name}/`。顶层1748文件为保留的未发布草稿，C002有已记录的偏移抄录错误，不作为评价输入。\n\n报告：[{report and "REAL512_PREPARATION_REPORT_"+T+".md"}]({R.name}/REAL512_PREPARATION_REPORT_{T}.md)\n\n盲包：[{"blind512_materials_"+T+".zip"}]({R.name}/blind512_materials_{T}.zip)\n\n不改旧数据/记录/结果，没有新GT。\n')

archive=R/f'blind512_materials_{T}.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
 for p in sorted((R/'blind_release').iterdir()):z.write(p,'blind512_materials/'+p.name)
paths=[protocol,status,sources_path,*sorted((R/'raw_crops').rglob('*')),*sorted((R/'image_only_inputs').iterdir()),
 *sorted((R/'blind_release').iterdir()),archive,R/f'EVALUATION_PROTOCOL_{T}.md']
manifest={str(p.relative_to(R)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in paths if p.is_file()}
save(R/f'FREEZE_{T}.json',dict(frozen_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
 candidate_ids=['C001','C002','C003'],candidate_count=3,unique_acquisitions=5,site_components=2,
 no_human_annotation_or_algorithm_evaluation_before_freeze=True,independent_GT=0,
 not_independent_unseen_final_test=True,no_recrop_after_human_annotation=True,files=manifest,
 algorithm_input_contract='image_only_inputs DN PNG only; no provenance, GT, masks or geographic initialization',
 preparation_only_geo_centres_are_NOT_GT=True))
print(json.dumps(dict(release=str(R),report_stamp=T,material_stamp=S,zip=str(archive),freeze=f'FREEZE_{T}.json'),ensure_ascii=False))
