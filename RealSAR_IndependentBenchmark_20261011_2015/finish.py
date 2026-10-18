"""Verify the limited prepared inputs, export pending evidence and seal preparation."""
import sys
sys.dont_write_bytecode=True
import json, hashlib, subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from export_annotations import export
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

GUIDES={
'Capella_0003__chip_001': '''任务：Kingston 真实异向城市。2025-03-06 → 2025-03-07；图0 descending/left、方位208.8°、入射39.6°；图1 ascending/right、方位320°、入射24.9°。原生像素间距分别约0.318×0.319和0.414×0.416米，不等于物理分辨率。

实际显示检查：两图能辨认街块及弯曲道路网络；图0相对灰白，图1有显著暗区域和方向性建筑亮边。局部显示能查看路网轮廓，不能据此宣称屋顶尖端是同一物理散射中心。

独立操作：先分别看两图整体道路拓扑，描述可能的道路分叉、稳定地面边界转角，再判断双图是否可见同一结构。地面路口也可能因叠掩无法精确定位，不要把暗路面中心和另一图亮墙端点对应。屋顶亮边、阴影末端、孤立亮斑均不默认可用。两图原生尺度不同，点仍填写各自原生坐标。

无法在两图独立确认同一结构，或位置只能落入宽泛散射带时，不添加高置信点；可保留不确定点但不能进入正式精度评价。仅局部可靠时选择 local_only，无法辨认时选择不可评价并写原因。''',
'Umbra_0014__chip_002': '''任务：Bingham Copper Mine 真实异向矿区。2024-05-06 → 2025-10-21，间隔约17.5个月；图0 ASCENDING/RIGHT、方位235.143°、入射16.478°；图1 descending/left、方位294.017°、入射28.552°。两图原生像素间距约0.312米。

实际显示检查：图0有斜向亮暗条带，图1近竖向纹理与不同明暗结构；96像素局部放大后，斑点及连续边界仍难精确分离。工具显示正常，但没有发现足以自动确认的静态对应点。

独立操作：若确实能看出同一固定道路分支、基岩边界或长期固定工程结构，可依据其物理拓扑定位。开采台阶、堆料边缘、车辆、施工道路、阴影边界随时间或视向变化，不能默认静态。相似条带或相同曲率不能证明是同一台阶，不能在两图不同阴影末端配点。

这对允许最终零点或不可评价；先独立判断共同结构存在性，再考虑点位。确有局部结构但不足全幅支持时只选择 local_only。不能用地图、DEM或算法输出把未辨認的点“补出来”。''',
'Umbra_0061__chip_001': '''任务：Long Beach 相近方向港口，2025-02-02 → 2025-08-03；两图 descending/right；方位85.726°与85.156°、入射39.769°与40.460°。原生像素间距约0.219米；这不是本轮异向样本。

实际显示检查：全幅有重复长条亮结构和交界带，局部斑点强、相似长条容易混淆。能够看结构形态，尚不能判断其中哪些是长期固定设施。相近方向只降低辨认难度，不保证无时间变化。

独立操作：优先寻找固定地面道路交叉、固定防波堤/码头实体拐角、永久建筑的地面边界，并分别确认两图可见同一结构。船体、船头、靠泊位置、货物堆、可移动集装箱、起重机吊臂及阴影末端排除。不要仅凭平行亮条或重复纹理选择点。

先写物理结构说明，再填点位、静态属性和不确定度。若可靠点都在一条狭长结构上，应记录几何支持不足，不能强行扩散点来满足全局规则。''',
'S1SF_EXTENSION_000__chip_002': '''任务：San Francisco 同向对照，2020-06-02 → 2020-06-14，两个不同 Sentinel-1 真实采集。原生像素间距约19.960×20.044米；两图由同一地图网格构建不构成精确GT。

实际显示检查：大部分为海面，静态海岸及右侧城市结构较一致；靠岸中部有可辨认轮廓。8倍放大只是20米像素的显示放大，不能定位不存在的亚像素建筑角点。

独立操作：可在两图清晰辨认的固定海岸拐角、岸边地面道路分叉或较大人工边界处判断对应。潮位、水面、海面亮点及模糊城市散射体排除。自然海岸线可能随潮位移动，应优先固定实体设施；无法区分自然边界与固定设施时记录不确定或跳过。

静态区域集中在右侧，可能只能满足局部稀疏评价。不要为满足全局横向跨度在海面造点；不得因共用地图网格直接填写 xy0=xy1。定位不确定度按原生像素判断，而非按放大后的屏幕像素。''',
'Umbra_0071__chip_000': '''保留的第5对：Rotterdam 相近方向港口，2025-01-29 → 2025-02-02；这是补充任务，本轮优先级低于指定4对。沿用相同独立标注规则：只寻找双图可确认的长期静态实体边界，不默认船舶、货物或孤立亮点静态。未安排人工操作时保持 pending；没有可靠共同地物则由真人写不可评价原因。'''
}

GUIDE='''# 最小独立核验基准：人工操作入口

本轮保留已有5对任务，重点4对；不改变原任务顺序和ID。软件准备完成，人工标注尚未开始。这里的静态地物描述只是检索建议，不是已确认对应点。预览里的ROI也是两幅图各自的显示窗口，同一列、同一像素窗口不表示对应。

## 启动与访问

在服务器运行：
```bash
cd /home/yl/matching/sar_matching
conda run --no-capture-output -n match python RealSAR_IndependentBenchmark_20261011_2015/serve_review.py --port 8768
```
服务器有桌面时打开 http://127.0.0.1:8768 。远程使用，在自己的电脑建立隧道（把 `yl@服务器地址` 换成实际登录地址）：
```bash
ssh -N -L 8768:127.0.0.1:8768 yl@服务器地址
```
随后在电脑浏览器打开 http://127.0.0.1:8768 。程序只绑定服务器回环地址；本轮测试服务已经退出，需用以上命令启动。

## 第一人员独立标注

1. 填真实姓名或稳定身份ID；保持默认“独立标注”。不要打开算法结果图或辅助层，也不要阅读路线报告中的预测诊断再挑点。选择任务时重点4对为 Capella_0003__chip_001、Umbra_0014__chip_002、Umbra_0061__chip_001、S1SF_EXTENSION_000__chip_002。
2. 先整体看两图，再滚轮/+/-缩放，Shift拖动平移，悬停看8倍局部放大。原生像素中心坐标为0..511，缩放不改变坐标。亮度/对比度仅显示调节，可恢复默认；不生成新尺度或辐射表示实验。
3. 仅在能判断同一静态地物、双图清晰可见时分别点击。写具体物理结构、静态和双图可见属性、置信度及两图定位不确定度；别因希望达到3/5px阈值而压低不确定度。散射峰、阴影端、船舶或重复条带都不能默认对应。
4. 默认角色是“独立保留检查点”。若要另拟合模型，事先把专用点标为control，不得把检查点移作拟合点；不能根据算法误差删除难点。可靠点不够时停止，不强行补点。每对初看最多20分钟、最多12点，只是工作预算，零点完全允许。
5. 点位先添加草稿，再点击“保存图像对”。若只能建立局部对应，选local_only；如果无法确认，选“视角差异过大/没有可靠共同静态地物/其他不可评价”并写原因后保存。数量不足可选insufficient_points并记录原因。pending保留未判断任务。

## 真实第二人员复核

第二位真实人员以自己的ID进入“独立复核”模式；点击点ID只读物理描述，原点位在界面隐藏，必须在两图重新点击，填写自己的静态/可见判断、不确定度及确认/不确定/拒绝结论，保存复核。每对建议最多15分钟。工具只能检查不同ID，不能认证真是两个人；不要用另一ID自审。无第二真人时保留未复核结果，不得升级C。

原有规则不变：局部至少3个合格check、两图凸包面积≥0.001；global_sparse至少6个check、两图凸包≥0.1且各方向跨度≥0.5。稀疏Success还至少4个check，联合不确定度≤相应阈值/3；不确定度包含实际两人重点击分歧。3点局部合格并不意味着允许报告Success@3/5/10。所有结果只支持相应稀疏范围，不能证明整幅稠密精度。

## 辅助暴露与导出

默认算法和地图候选全部隐藏。主动切换辅助模式并点击显示后会留下 `annotation_adapter/annotations/assistance_events.jsonl`，记录身份、时间、生成器和文件哈希；“隐藏”不删除暴露。已暴露人员在该对上新增/修改的点或复核不能取得独立资格。需要辅助诊断时先完成独立操作，辅助证据另记，不把伪标签改称人工GT。

完成操作后停止编辑（建议Ctrl-C停止服务），执行：
```bash
conda run --no-capture-output -n match python RealSAR_IndependentBenchmark_20261011_2015/export_annotations.py
```
输出自动保存到 `exports/annotations_当前年月日_时分/`；同一分钟已存在会拒绝覆盖，可等下一分钟或用 `--out 新目录`。其中 `points.csv` 是逐点坐标/属性/复核/不确定度/资格表，`documents/` 保存完整记录，`history/` 保留版本链和辅助暴露，`index.csv`汇总状态，`EXPORT_COMPLETE.json`标志一致快照及哈希。没有该完成标志的目录不能视作成功导出。

本轮已有零点导出可验证接口，不是人工GT。不设置人工截止等待；接下来真正需要你安排的只有一位标注人员与另一位独立复核人员。

许可：Umbra 与 Copyright Capella Space CC BY 4.0；Sentinel-1 预览保留声明：Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026。保持原数据使用范围。
'''

REPORT='''# 真实 SAR 最小独立核验基准准备与特征匹配技术路线

完成时间：{completed_at}（Asia/Shanghai）。本轮完成的是工程准备与有针对性的原始文献/官方实现核验；不是已经建立了可评价人工GT，也不是全面系统综述。所有新内容位于 `RealSAR_IndependentBenchmark_20261011_2015`。

## 实际交付与旧结果保护

复用 RealSAR_GeometryGT 标注系统与 RepresentationContext 的全部5对任务，任务清单、候选载荷和3个UI文件逐字节复制；旧程序只读引用。新标注只写新目录。5对/5个来源组件/10个不同采集，重点4对/4组件/8采集。其中重点异向2对、同向2对，第5对也是相近方向。原split保留；这些样本早已被看过和研究过，不能称为新的无偏确认集。

10张512×512灰度原生图尺寸、文件哈希及显示有效性通过；提供5张原生双图预览、5张局部视窗预览、5张真实浏览器截图和1张缩放/平移/局部放大截图。ROI没有对应点、预测连线或地理辅助点。实际人工点0、独立复核0、辅助暴露0、gt_qualified/C=0。零点并非失败；静态共同结构的存在性和可定位性仍须真人判断。

| 重点任务 | 方向与时间 | 显示检查与独立标注风险 |
|---|---|---|
| Capella_0003__chip_001 | 真实异向，2025-03-06→03-07 | 路网/街块可辨认；尺度差、建筑亮边和阴影差大，不能默认同一散射中心 |
| Umbra_0014__chip_002 | 真实异向，2024-05-06→2025-10-21 | 条带与阴影方向改变，开采地貌可能变化；允许局部、零点或不可评价 |
| Umbra_0061__chip_001 | 相近方向，2025-02-02→08-03 | 港口重复结构和斑点多；船舶、货物、吊臂排除，新增B候选仍未独立证实 |
| S1SF_EXTENSION_000__chip_002 | 同向，2020-06-02→06-14 | 岸边大结构较一致，但约20m像素且海面占多，不强求全幅分布 |

各任务独立说明见 `guides/`；入口和质量规则见本轮 `annotation_guide_*.md`。在UI默认独立模式下，地图和算法预测均隐藏；暴露需显式操作并永久留痕。原质量规则、第二人员重点击、修改使复核失效等未改变。独立复核隐藏的是界面坐标，API返回记录含原坐标，身份亦由操作者声明；这不是抗恶意访问的双盲系统。操作者不得查看API原点位或提前查看辅助输出。来源选择已受旧研究影响，亦不能宣称严格完全盲法。

新增薄层服务入口、完整导出接口与显示/浏览器验收；没有重建标注系统。导出包括全部实际点（即使不合格）、原生坐标、物理说明、静态/可见属性、定位不确定度、真实复核记录、资格理由、不可评价原因、版本链、暴露日志、任务清单与文件哈希。导出不推断人工意见、不升级GT、拒绝旧目录覆盖，并检测导出期间修改。模拟测试只在已清除的 `/tmp` 目录，不能算人工标注或第二人员复核。

只验证本轮使用的47个输入文件哈希，没有重复前轮5万余条保护审计。47/47保持不变，旧数据/伪标签/结果未改写。Phase2基线Git为 `ef0c6fd1c4203fef67d5f291c139c39c9c70676f`，本轮新代码以本目录冻结SHA记录，未提交或推送。实际match环境Python3.10.22、torch2.5.1+cu121、torchvision0.20.1+cu121、opencv-python-headless4.12.0.88；GPU读取确认RTX4090、可见显存23.650GiB。未安装依赖、未加载新权重、未推理或训练新模型。

## 技术路线比较：证据和适用边界

下面的“能尝试”仅是输入、实现和预算可行性；不是已证明本项目异向SAR精度。SAR–光学、自然图像视角变化与SAR–SAR异向的散射/叠掩/阴影变化不能等同。512高分辨率小窗只有约百余米范围，可能仅含重复屋顶或可移动物体，甚至没有可评价共同静态结构。更强描述子也不能恢复未共同可见的物理地物。

| 方向与代表方法 | 主要解决的问题 | 训练与GT需求 | 公开实现、match/4090条件 | 512异向SAR的条件与限制 |
|---|---|---|---|---|
| SAR局部特征：SAR-SIFT | 乘性斑点下更稳定的梯度、检测与描述 | 推理无训练；精度判断要独立点 | 作者论文可核验；找到第三方MATLAB版本，未确认作者维护Python包及许可证，不直接装 | 有稳定角点/边界时有价值；不是入射角不变或叠掩纠正器 |
| 辐射鲁棒结构：RIFT/RIFT2 | 相位/最大索引结构描述、非线性辐射和面内旋转变化 | 无训练；评价仍需独立点 | 作者RIFT2为MATLAB，根目录无明确许可证，Linux match不能直接运行；Python移植不是作者原实现 | SAR–光学证据不证明本项目真实异向；强几何变化仍是难点 |
| 密集结构/局部模板：CFOG | 梯度方向结构相似、局部模板相关搜索 | 推理无训练；地理先验不是GT | 作者MATLAB/MEX包声明64位Windows且仅限研究；需另审移植/使用条件 | 局部实体结构共同可见时可试；重复纹理会假匹配，不以相关峰自证精度 |
| 深度强外观鲁棒性：RoMa（冻结outdoor） | DINOv2粗特征+细化，密集对应和可见性/置信 | 官方预训练推理无需本项目训练；独立点检验迁移 | 作者MIT，DINOv2 Apache2；Python≥3.9、torch≥2.5.1，现有基础版本满足，但缺timm/einops等；需依赖白名单避免新增OpenCV | 自然图像与SAR–光学迁移有依据，真实异向SAR收益尚未知；密集输出不等于稠密GT，置信未作SAR校准 |
| 多谱专用深度：XPoint | 跨谱检测、描述与自监督适配 | 冻结SAR版可试；SAR–SAR训练需真实/弱监督标签及独立评价 | 作者MIT，有VIS–SAR权重链接；测试Python3.11/推荐CUDA12.4，numpy1.26.3、kornia0.1.4与当前不符，VMamba需编译 | SAR版训练任务是VIS–SAR，不是当前异向SAR–SAR；不优先冒险改环境 |
| SAR专用开集：CroR-OSIR | 把不可匹配/OOD关键点拒绝，减少错误接受 | 论文有监督对比特征调整与交叉拒绝训练，不是冻结通用匹配器 | 作者Apache2，依赖清单混入Windows包、多个OpenCV及异常包名；权重/完整推理步骤未核实，不直接安装 | 研究问题与本项目缺对应/错接受很相关，但不能直接当作“无训练即用”方案 |
| 道路/岸线/线结构：OpenCV LSD；可选DeepLSD | 以长线、边界拓扑补充稀疏亮点，降低局部外观依赖 | LSD无训练；DeepLSD可用预训练；训练边界网络另需标注/弱监督 | 现有cv2已有LSD；DeepLSD作者MIT及权重，但完整精化需Ceres/GFlags/GLog、旧环境依赖且双OpenCV，暂不装 | 地面固定线状结构优先；屋顶亮线、阴影线、自然潮岸不是静态实体线；单线沿线方向不可定位 |
| 地理约束局部搜索 | 缩小候选空间、处理小窗低重叠和重复纹理 | 影像+现有CRS/Affine/spacing即可生成候选；准确性要独立点 | 复用已有PixelMap，rasterio/numpy/OpenCV可完成，CPU即可 | 地理偏差/地形位移会越出窗口；窗外必须允许无输出；不可把地图对应作正样本真值 |
| 稀疏可靠匹配+局部几何 | 避免强行全幅H；局部平面/不均匀几何支持 | 无需训练；独立check验证局部正确性和范围 | 复用冻结Affine/可靠性；OpenCV有USAC_MAGSAC，作者MAGSAC BSD，暂不另装 | 鲁棒估计不能创造正确对应；局部模型只声明局部支持，控制点拟合残差不算精度 |

原始证据（检索/核验日期2026-10-11；官方仓库版本、原文件与SHA见 `sources/`）：

- SAR-SIFT以SAR斑点鲁棒梯度改造检测与描述。[作者机构的2015论文记录](https://researchportal.ip-paris.fr/en/publications/sar-sift-a-sift-like-algorithm-for-sar-images/)，[作者2012先行论文](https://perso.telecom-paristech.fr/tupin/PUB/Dellinger12.pdf)。第三方[MATLAB实现](https://github.com/ZeLianWen/Image-Registration)不能冒充作者官方包。
- RIFT2改进RIFT旋转不变描述计算；不以其跨模态实验推断SAR异向可靠性。[论文](https://arxiv.org/abs/2303.00319)，[作者MATLAB代码](https://github.com/LJY-RS/RIFT2-multimodal-matching-rotation)。CFOG使用方向结构模板框架，作者明确研究用途和Windows/MATLAB条件。[CFOG作者包](https://github.com/yeyuanxin110/CFOG)。
- RoMa融合DINOv2粗特征及细尺度卷积匹配；公开基础实现和依赖可检查。[RoMa论文](https://arxiv.org/abs/2305.15404)，[作者代码](https://github.com/Parskatt/RoMa)，[实际依赖定义](https://raw.githubusercontent.com/Parskatt/RoMa/77f8d68803526dcddfd9b7a46bc76125bdc25f15/pyproject.toml)。2026年SAR–光学零样本研究支持“先测冻结匹配器”的低成本选择，但任务和评价协议不同，不能搬用其3px结果或运行时间作本项目预计值。[CVPRW2026论文v4](https://arxiv.org/abs/2604.10217v4)。
- XPoint有VIS–SAR适配，不是异向SAR–SAR保证；其训练涉及对齐多谱数据和伪标签。[论文](https://arxiv.org/abs/2411.07430)，[作者代码与权重入口](https://github.com/canyagmur/XPoint)。CroR-OSIR面向SAR开集错配，包含特征调整/训练过程；可借鉴拒绝思想，暂缓整套复现。[CVPR2025论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Mao_Cross-Rejective_Open-Set_SAR_Image_Registration_CVPR_2025_paper.pdf)，[作者代码](https://github.com/XDyaoshi/CroR-OSIR-main)。本轮仅核验可访问摘要与代码元数据，未做全文训练标签/权重完整性核验。
- 线结构可以补充点，但线检测不包含线匹配或地面物理对应判断。[OpenCV LSD接口](https://docs.opencv.org/4.13.0/db/d73/classcv_1_1LineSegmentDetector.html)，[DeepLSD论文](https://arxiv.org/abs/2212.07766)，[官方代码](https://github.com/cvg/DeepLSD)。鲁棒估计改变采样/评分而非特征的可匹配性。[MAGSAC++论文](https://arxiv.org/abs/1912.05909)，[官方代码](https://github.com/danini/magsac)。

## 三类可实施性

**I：现有影像与元数据就能运行候选实验。** 自编最小SAR比值梯度/方向结构描述、LSD线段、地理限窗及局部Affine；经过必要兼容性核验的冻结RoMa也不需要本项目GT来预测。这里的“可以运行”只允许统计输出、耗时、候选分布和拒绝；地图偏差、循环一致、自拟合残差都不能证明精度。

**II：少量独立对应点才能判断效果。** I类的正确匹配率、局部残差、错误接受、Success、置信校准及“几何还是特征失败”均属于此类。至少保留专用check，先判断人工质量和可评价范围；未合格或两图不共同可见的点不进入精度分母，必须报告不可评价数量。少量点只能验证这些点附近，不给所有机器匹配随意标正确/错误；稀疏点不能形成完整匹配precision/recall真值。

**III：需要真实训练标签或弱监督。** SAR描述子/匹配器域适配、XPoint/CroR式调整及实体边界网络。需跨方向共同可见的静态正对、难负例、不可对应/动态/OOD标记、来源组件隔离与独立检查集。地图和现有B/弱伪标签只作带置信权重的弱监督，不能充当独立C；单图合成H、面内旋转训练只表示二维增广，不模拟真实视角散射。不能把本轮4对同时用于训练、阈值选择和确认。

## 优先路线（最多两条）与固定预算

**路线1：训练无关的SAR结构描述＋地理限窗＋稀疏局部几何。** 用比值梯度的方向结构描述与固定地面边界/线拓扑，减少对孤立散射峰的依赖；借鉴SAR-SIFT/RIFT/CFOG的问题分解，但不擅自复制许可证不明代码，也不把一个自编小组件称为完整复现。先用现有numpy/cv2实现最小可读基线，再判断线拓扑是否有额外信息；本轮只制定方案。

**路线2：冻结RoMa outdoor的一次零样本诊断。** 使用已核验官方commit、官方outdoor/DINOv2权重，固定默认内部560→864推理分辨率、输出转回各图512原生坐标，无尺度搜索。先测能否跨辐射/方向产生更多可独立确认的静态对应，再决定是否值得SAR域训练；不直接转向MASt3R等自然透视3D模型，不把SAR当针孔相机。

预算配置见 `pilot_plan.json`，状态为 `planned_not_run`，不是新实验结果：

| 项目 | 预先固定设置 |
|---|---|
| 数据 | 仅指定4对，2异向+2同向，4来源组件；第5对只保留标注任务，不进本次方法选择。所有样本已研究，定位为诊断开发小基准 |
| 输入表示 | 路线1固定使用已有原生GeoTIFF幅度数据计算比值梯度，Sentinel-1声明为power时固定取平方根；不把对数显示PNG当线性幅度。路线2和旧缓存固定原生PNG；不增加CLAHE/滤波/尺度/旋转组合。跨路线属于完整方法栈比较，输入不同，收益不能单独归因特征；各路线内部限窗消融保持输入完全相同 |
| 路线1 | 初版固定一个描述子：单个3px半窗比值梯度、8方向bin、48px描述窗口；确定性16×16双图采样网格各最多256候选；无参数网格。双向搜索、固定比值0.8、cycle≤3px仅作图像一致性拒绝，置信不作独立准确率；两图像素间距不同的物理窗口尺度差作为局限记录，不追加多尺度选择 |
| 两路线共同消融 | 每条仅2个版本：不设地理候选窗 vs 与旧PixelMap中心相距≤64 moving原生像素。地图只限制搜索/过滤，不作点标签；64px是一项预注册研究假设，无法证明覆盖所有地形偏移，失败如实记录，不在4对上扩窗再试 |
| 路线2 | 单个官方冻结outdoor版本，batch1，最多2048匹配，seed42；匹配/采样默认阈值由固定commit保存，不以基准点调整；记录权重SHA、缺依赖清单及完整配置后方可推理 |
| 几何 | 同一匹配池比较冻结全局Affine与固定4×4局部Affine诊断；最多16局部单元，不新增H/TPS模型搜索。仍按旧规则检查退化/空间支持；仅局部支持就不接受成全幅配准 |
| 规模 | 4对×2路线×2版本=最多16次新增配对推理；同匹配池几何复用不重提特征。SIFT/ORB/LightGlue原生与geo_assisted对照读取现有24条缓存，不重新跑191对 |
| 时间 | 实现/兼容性工作最多2工作日；CPU推理≤2小时、GPU推理≤2小时、单对超时120秒；显存预算≤20GiB；到上限停止并保留失败，不换仓库/模型或重搜索参数 |
| 时间测量 | 端到端读图/描述/搜索/估计，CUDA同步；冷启动与首次权重加载另报。热运行只在首个同向对照额外3次计时，最多19次总配对执行；不重采样选择最好结果 |
| 人工预算 | 重点4对每对一标注者20分钟+另一真人15分钟，总计约140人分钟、每对最多12点，无最低凑数；第5对可保持pending |

若安装RoMa，应在match先检查 `pip --dry-run`/依赖列表，禁止照搬默认依赖同时装opencv-python与现有headless；固定commit源码路径导入或受控`--no-deps`安装，仅补确实需要且兼容的包，保留torch/CUDA与单一cv2。源码中若调用contrib功能，先定位需要的具体接口，不能盲目安装第二个OpenCV。官方权重当前未下载、哈希未取得，未来运行前必须记录；不能假称已完整复现。4090基础资源可试batch1，真实显存/耗时仍未知，超预算立即报告。

## 评价与停止规则

人工check只给 evaluator，匹配器只接收允许的图像/地理元数据；check不能决定接受或参与局部拟合。control若另用于人工模型可解释性，只是单列诊断，不能掺入公平算法预测。所有变换方向固定reference→moving原生坐标，反向重点击、地图映射半像素约定、RoMa[-1,1]转像素均在独立小测试验证后执行。

对实际gt_qualified任务按现有quality.quantitative记录check mean/median/p90与允许阈值的稀疏Success@3/5/10；局部预测单列仅覆盖支持内检查点的误差与覆盖数/总合格check数，未覆盖点不可删除以抬高成功率，不宣称全幅Success。记录 no_output、rejected、accepted_unassessable、incorrect_acceptance、sparse_success，并报告人工不可评价任务和阈值资格分母。按来源组件列出，4组件样本太少，不给大范围泛化显著性结论。本次4对没有预先核验无重叠任务，因此无重叠错误接受率标N/A；不可把“无可靠静态点”当无重叠负例。

先比较独立检查点上的结构可匹配性，再在完全相同匹配池比较全局/局部几何，避免把模型差异归因特征。内点数、地图残差、自循环作为诊断不作为精确GT，新增B数量也不能替代正确对应数量。遇到异向共同地物根本不可见，应该标不可评价，转向更大原生场景/可靠地面控制或更合适真实采集，不再调整预处理追求接受。

若两条路线都不能在可评价异向点上改善，保留失败并停止本轮小窗探索。只有完成独立核验且发现跨方向重复可见静态结构存在、当前特征仍系统性失败，才有证据考虑SAR专用描述/检测训练；若几何/可见性/地物变化是主因，应先补观测元数据、地形和局部结构标签。后续训练与确认需另建来源隔离数据，本轮已看过的4对不得转称未见确认。

## 验收、复现与失败记录

Chromium145.0.7632.6实测11项流程检查通过，控制台错误0；真实任务0 POST、0新标注；临时测试验证同人复核拒绝、另人重点击、辅助留痕、修改失效及导出历史/哈希/拒绝覆盖。状态出口在真实5任务成功生成零点快照，C=0；这只是接口验收。

准备命令是 `conda run --no-capture-output -n match python 本目录/prepare.py`，只允许创建新adapter；完成后的同目录重跑会拒绝覆盖。`fetch_sources.py`只读下载官方文本，不执行远程代码；官方元数据7/7取得，无网络下载失败。复现浏览器检查需 `PYTHONPATH=/tmp/sar_gt_browser_deps PLAYWRIGHT_BROWSERS_PATH=/tmp/sar_gt_browsers_verified NODE_TLS_REJECT_UNAUTHORIZED=1 conda run --no-capture-output -n match python 本目录/browser_check.py`，已有ui_checks目录会拒绝覆盖；需要重复工程检查时在另一个准备根目录执行，不能覆盖本次证据。

文献浏览过程中，猜测CFOG的RegCFOG地址取回失败，已改为核验作者真实CFOG仓库；CVF MAGSAC HTML返回403、CroR PDF工具直接打开失败，使用作者arXiv/可访问出版摘要和官方仓库继续核验。未把工具访问失败描述为服务器无法联网。外层cwd没有Git仓库，已从实际Phase2仓库读取版本。CroR requirements存在异常`charset==normalizer`/`huggingface==hub`和Windows项，属于源码静态检查发现，不是本机安装失败；没有尝试安装这些包。

仍需人工操作：安排一位真实标注人员与另一位独立复核人员，按指南在不看辅助对应的条件下操作；缺第二真人就保留预备证据。自动工作已结束，不等待人类完成，不产生所谓人工GT或再次调参。
'''

def main():
    at=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=at.strftime('%Y%m%d_%H%M')
    protected=json.loads((ROOT/'input_sha256.json').read_text())
    changed=[p for p,h in protected.items() if sha(Path(p))!=h]
    assert not changed,changed
    browser=json.loads((ROOT/'ui_checks/browser_acceptance.json').read_text());assert browser['passed']
    assert not list((ROOT/'annotation_adapter/annotations').rglob('*'))
    tasks=[json.loads(l) for l in (ROOT/'annotation_adapter/annotation_tasks/tasks.jsonl').read_text().splitlines()]
    assert len(set(t['component'] for t in tasks))==5
    assert len(set(t[f'{s}_acquisition_id'] for t in tasks for s in ['reference','moving']))==10
    result=export(ROOT/'annotation_adapter',ROOT/'exports'/f'annotations_{stamp}')
    assert result['points']==result['gt_qualified']==0
    (ROOT/'guides').mkdir(exist_ok=False);(ROOT/'reports').mkdir(exist_ok=False)
    for sid,body in GUIDES.items():
        (ROOT/'guides'/f'{sid}_{stamp}.md').write_text(f'# {sid} 独立标注说明\n\n'+body+'\n\n全幅预览：../previews/'+sid+'_native.png；局部显示：../previews/'+sid+'_local_inspection.png。ROI只是显示窗口，不是对应点。\n')
    guide=ROOT/'reports'/f'annotation_guide_{stamp}.md';guide.write_text(GUIDE)
    report=ROOT/'reports'/f'independent_benchmark_routes_report_{stamp}.md';report.write_text(REPORT.replace('{completed_at}',at.isoformat()))
    save(ROOT/'pilot_plan.json',dict(status='planned_not_run',seed=42,priority_pairs=json.loads((ROOT/'priority.json').read_text())['priority_order'],
        development_diagnostic_only=True,unbiased_confirmation=False,old_results_frozen=True,
        routes=['SAR_structure_geo_local','RoMa_outdoor_frozen'],variants=['without_geographic_window','geographic_window_64_moving_native_px'],
        primary_pair_executions_max=16,extra_timing_pair_executions_max=3,total_pair_executions_max=19,
        GPU_hours_max=2,CPU_hours_max=2,pair_timeout_seconds=120,GPU_allocated_GiB_max=20,
        formal_GT_at_preparation=0,no_training=True,no_preprocessing_or_scale_search=True,
        checkpoints_downloaded=False,checkpoint_hashes=None,
        actual_repositories={r['repository']:r['commit'] for r in json.loads((ROOT/'sources/repositories.json').read_text())}))
    log=ROOT/'reports'/f'execution_{stamp}.log'
    log.write_text('\n'.join([
        'Preparation began: 2026-10-11 20:15 Asia/Shanghai',
        'Actual source application/code/quality policy inspected; copied 5 tasks and UI without edits.',
        'match prepare.py: 10 native 512 images checked, 5 tasks, 4 priority, 47 protected inputs.',
        'All 4 priority native pairs and local 96px viewports visually inspected; no correspondences created.',
        'Official primary paper/implementation sources browsed; 7 repository metadata snapshots, no installs/inference/training.',
        'match CUDA read-only check: torch2.5.1+cu121, torchvision0.20.1+cu121, RTX4090, cuda_available=True.',
        'match browser_check.py: exit0; Chromium145, 11 checks passed, real0POST, simulated interactions only/tmp.',
        'Protected input files rechecked: 47/47 unchanged. No repeated full historical audit.',
        f'match export_annotations.py library: 5 real pending tasks, 0 points, 0 qualified; exports/annotations_{stamp}.',
        'Static source issues: licenses/platform/dependency constraints documented; no installation failures.',
        'Web lookup failures: guessed RegCFOG URL; CVF MAGSAC HTML403; CroR direct PDF open failed. Primary alternatives used.',
        'Parent workspace git status failed: not a git repository; correct Phase2 repository HEAD/clean status inspected.',
        'Pilot plan only, ≤2 routes, ≤19 executions proposed; not run; checkpoints not downloaded.',
        'Human independent annotation/review pending; no artificial C upgrade.',
        'Completed: '+at.isoformat(),''
    ]))
    # Future annotations, quality sidecars and exports are explicitly mutable append-only products.
    excluded=('annotation_adapter/annotations/','annotation_adapter/gt_quality/','exports/')
    files={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='PREPARATION_ACCEPTANCE.json' and not str(p.relative_to(ROOT)).startswith(excluded)}
    receipt=dict(completed_at=at.isoformat(),preparation_complete=True,human_GT_complete=False,
        tasks=5,priority_pairs=4,components=5,acquisitions=10,native_images_checked=10,
        real_human_points=0,real_independent_reviews=0,real_assistance_exposures=0,C=0,
        browser_checks=len(browser['checks']),browser_passed=True,protected_inputs_verified=len(protected),
        old_results_modified=False,dependencies_installed=[],new_model_inference=False,training=False,
        report=str(report.relative_to(ROOT)),annotation_guide=str(guide.relative_to(ROOT)),execution_log=str(log.relative_to(ROOT)),
        mutable_append_only_prefixes=list(excluded),sha256=files)
    save(ROOT/'PREPARATION_ACCEPTANCE.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='sha256'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
