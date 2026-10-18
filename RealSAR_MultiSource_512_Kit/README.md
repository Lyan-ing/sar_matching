# 多来源真实 SAR 配准数据构建工具（512 像素默认）

**工程性质：** 真实卫星/机载 SAR 不同采集配对的*候选生成与可追溯审计*工具。默认每幅裁片 512×512，图像真实源不得由单图变换伪装为跨采集。当前工具不内置任何论文、论文 PDF 或商业影像。

**验收边界：** 公开源目录可用于发现和粗几何对齐；**地理编码重投影不构成独立像素级配准 GT**。没有独立静态检查点/可靠几何观测时，输出一律标为 `COARSE_GEO_ONLY__INDEPENDENT_GCP_REQUIRED`。本工具不训练网络，不会在正式测试集选参数。

## 文件和能力

| 文件 | 职责 |
| --- | --- |
| `discover_s1_rtc.py` | 从匿名 Indigo Sentinel-1 RTC S3 按 MGRS tile / 年度列出真实采集并生成候选配对清单；支持多个 tile、多个年、每 tile 数量上限 |
| `download_s1_rtc.py` | 沿用已验证的单个 Sentinel-1 RTC 三时相试点下载与窗口读取程序，供人工快速查看 |
| `discover_stac.py` | 遍历 Umbra / Capella 公开 STAC 静态目录，输出 GeoTIFF 资源清单与空间/时间重叠配对**建议**，不跳过产品元数据复核 |
| `from_existing_pilot.py` | 将既有 Sentinel-1 真实试点三景整理成候选 Manifest，用于离线 smoke |
| `build_real_pairs.py` | 从本地 GeoTIFF 或公开 HTTPS COG 创建不同真实采集的 512×512 配对，同时保存未经像素重投影的移动影像本地裁片 `moving_native_float.tif`，以及用于预对齐的地图网格裁片 `moving_float.tif`、PNG、valid mask、元数据；可选择生成 128×128 正负 patch **候选** |
| `review_gcps.py` | 独立人工控制点 / 保留检查点的拟合精度检查（只是验收辅助，不能自行生成 GT） |
| `audit_candidates.py` | 统计已生成候选的来源、场景、失败原因及文件完整性 |
| `CODEX_HANDOFF.md` | 给原 Codex 对话的直接执行任务书 |
| `tests/` | 不依赖互联网的小规模合成**单元测试**，测试数据不是公开真实 SAR 样本 |

## 快速启动（先离线验证）

从服务器 `/home/yl/matching/sar_matching/` 解压该 ZIP。**所有 Python 都走 `match`**，不触碰其他 Conda 环境。

```bash
cd /home/yl/matching/sar_matching/RealSAR_MultiSource_512_Kit
conda run --no-capture-output -n match python -m pytest -q tests/
```

若缺依赖，可在 `match` 环境里安装 `requirements.txt`，先核验已有库版本，避免更改 `torch` / `opencv` 的已锁定依赖。

### 1. 利用现有已下载三景验证候选构建

根据远端真实试点路径（不复制旧实验数据，不覆写）：

```bash
BASE=/home/yl/matching/sar_matching
RUN="$BASE/real_sar_pilot/multisource512_$(date +%Y%m%d_%H%M)"
mkdir -p "$RUN"
conda run --no-capture-output -n match python from_existing_pilot.py \
  --pilot-manifest "$BASE/real_sar_pilot/sentinel1_20261010_1731/triplet_verified/manifest.json" \
  --out "$RUN/pilot_manifest.csv"
conda run --no-capture-output -n match python build_real_pairs.py \
  --manifest "$RUN/pilot_manifest.csv" --output "$RUN/pilot_512" \
  --chip-size 512 --chips-per-pair 1 --seed 260512
conda run --no-capture-output -n match python audit_candidates.py \
  --data "$RUN/pilot_512" --out "$RUN/pilot_audit.json"
```

以上仍是同轨重复观测，**只检验工程可运行性**，不是大角度 SAR GT 基准。

### 2. 批量发现 Sentinel-1 RTC 候选

```bash
conda run --no-capture-output -n match python discover_s1_rtc.py \
  --tiles 14TPN --years 2020 --max-pairs-per-tile 12 \
  --out "$RUN/s1_discovery"
```

`14TPN/2020` 已经在此前试点验证可访问；更多 tile、年份需要 Codex 按 **不同真实地理场景**核对有效覆盖后再添加，不要臆造 tile。输出 `pairs_manifest_CANDIDATE.csv` 的采集时间暂为日期占位，须用原始 Sentinel-1 productInfo/SAFE 校正采集 UTC、轨道及 datatake 身份。候选对可能没有共同有效覆盖，构建脚本会拒绝这些窗口并记录失败。

```bash
conda run --no-capture-output -n match python build_real_pairs.py \
 --manifest "$RUN/s1_discovery/pairs_manifest_CANDIDATE.csv" \
 --output "$RUN/s1_512" --chips-per-pair 4 --chip-size 512
```

同一 tile 的多个日期不是多个独立站点，不得随机把它们跨 train/val/test 划分。**不能因多裁几个 512 patch 就声称获得等量独立采集。**

### 3. 批量发现 Umbra / Capella 高分辨率数据

```bash
conda run --no-capture-output -n match python discover_stac.py \
 --provider umbra --max-documents 1200 --max-pairs 60 \
 --out "$RUN/umbra_discovery"

conda run --no-capture-output -n match python discover_stac.py \
 --provider capella --max-documents 1200 --max-pairs 60 \
 --out "$RUN/capella_discovery"
```

这是 **STAC 资源扫描和候选配对提议**，不保证所有提供方资产是单波段浮点 GeoTIFF、无需登录或适合配准。特别是 `pairs_PROPOSALS_REVIEW_REQUIRED.csv` 中 `split=REVIEW_REQUIRED` 且 `input_units` 为空是刻意设计：必须确认场景/来源/原始采集 ID、产品级别、极化、量纲、时空覆盖、观测角与许可证后，才能将其变成正式输入 Manifest。不可盲目批量读取或把不同传感器/极化/不同辐射量纲的产品混为相同样本。

数据源可查：[Sentinel-1 RTC](https://registry.opendata.aws/sentinel-1-rtc-indigo/)、[Umbra Open Data](https://registry.opendata.aws/umbra-open-data/)、[Capella Open Data](https://www.capellaspace.com/community/capella-open-data-program/)；另可探索 Sentinel-1 GRD 的 Earth Search STAC，ALOS-PALSAR 需额外核实产品授权和配对可得性。多源覆盖应以**不同真实场景、不同已核实采集身份、不同观测条件**为主要目标，不单纯追求裁片数量。

### 4. 正式构建 512×512 候选

自行创建经审计的 Manifest；格式参考 `examples/source_pairs_MANUAL_TEMPLATE.csv`，其中 `input_units` 必须明确为：`linear_power`、`linear_amplitude` 或 `db`（不能自行假定所有提供方都是 Gamma0 功率）。

```bash
conda run --no-capture-output -n match python build_real_pairs.py \
  --manifest "$RUN/verified_sources.csv" \
  --output "$RUN/real512_candidates" \
  --chip-size 512 --chips-per-pair 4 --seed 260512 \
  --min-valid 0.98 --min-texture-db 0.5 \
  --positive-per-chip 0 --negative-per-chip 0
```

只在需要训练小补丁匹配器时，显式添加 `--positive-per-chip 2 --negative-per-chip 2 --patch-size 128`。这些仍是**未核验正负 patch 候选**，需要独立人工/几何复核。原生参考栅格以512像素窗口读取，同时保存第二次采集的**原生像素裁片** `moving_native_float.tif`（以参考区域的地图中心定位裁片，但不进行像素重投影）与**地图网格粗对齐裁片** `moving_float.tif`（发生重采样），二者不能混为一谈；不会把小尺寸原图简单放大作为独立高分辨率数据。保留 GeoTIFF 的原始浮点数值；PNG 仅展示用途。

### 5. 独立检查点与放行

在每个裁片独立标注静态、可重复定位的点，建议至少4个 control 和4个 check 点，使用不同区域分布。不得将 SIFT/ORB/LightGlue 自动预测结果直接当成独立 GT。先使用 `review_gcps.py` 查看 holdout 残差及空间覆盖；对大视角/非平面场景不能强迫一个 Affine，需单独评估局部变换或基于地理传感器模型的 GT。

```bash
conda run --no-capture-output -n match python review_gcps.py \
 --gcp-csv "$RUN/independent_gcps.csv" \
 --output "$RUN/gcp_review.json" --chip-size 512
```

只有通过来源审计与独立 GT 核查的样本才能标记为可正式评价或训练的真值样本。**保持“真实采集图像对”与“人工对齐训练 patch”分开存储，保留未经预配准的图像坐标系和原始来源。**

## 分辨率与旧数据集的关系

- **已冻结的旧合成测试集主要是边长128–384像素的原生采样视图。** 这是像素尺寸，不是空间分辨率。旧数据/预测/GT/哈希**一律不修改**，不能把旧测试集插值成512后和旧成功率比较。
- **新真实数据默认512×512**，只要原始 GeoTIFF 有真实512个采样像素；不同传感器须另外报告像素间隔、实际空间分辨率、几何校正、重采样比例。
- 如希望同样建立合成512对照集：让 Codex 检查第一阶段生成器对源图512裁剪、变换支持、GT点、valid mask、动态Mask的能力；优先选足够大的原生来源（例如 HRSID 的800像素裁片），沿用既有来源隔离规则，在**全新输出目录另建 synthetic512_v1**，重新生成 GT 和完整 QA。较小源图不够512时应跳过或单独列为 upsample 压力测试，不能混入主指标；不改变已有冻结 test。
- 如大角度真实跨视角数据的公共 overlap 非正方形或可见地物不足，允许额外保留256、768等裁片供诊断，但主协议仍以512为标准，不能强行放大无效内容。

## 当前验收边界

本包的离线单测可以验证程序逻辑，不表示已下载 Umbra/Capella 多景，也不等于已获得像素级 GT。外部 STAC API 和 COG 访问可能变动。请让 Codex 在原服务器上执行网络验证、真实来源身份审计、传感器几何/极化与许可检查后再发布数据集。
