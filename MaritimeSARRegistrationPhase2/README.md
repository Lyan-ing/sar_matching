# MaritimeSARRegistrationPhase2

在第一阶段冻结的二维合成 SAR 代理数据上比较 SIFT、ORB、Phase correlation、官方预训练 SuperPoint + LightGlue。仅图像预测，不使用 GT 或掩膜辅助估计，不训练或在正式测试集调参。860 组、900 条关系（860 正、40 无重叠）；840 双视图和 20 三视图。此任务不等同于真实跨观测角 SAR 配准。

## 本服务器的已完成实验

- 传统方法：`output/phase2_classic_20261009_2213`，2,700 行，三个方法各 900 对。
- LightGlue：`output/phase2_lightglue_20261009_2235`，900 行，官方原始推理代码/权重，无微调。
- 合并：`output/phase2_comparison_20261009_2237`，3,600 行。
- 最终交付以 `reports/phase2_final_report_*.md` 和同分钟的 `phase2_execution_*.log` 为准；历史日志与运行目录的标识保留。

已完成实验直接读取 CSV/JSON。不要为了继续分析而再次运行已完成评测。

## 环境与安装

**本项目所有 Python、pip、测试与推理统一使用 `conda run -n match`。** 当前环境 Python 3.10.22；不使用第一阶段 `.venv`、rfrock 或 Conda base。

```bash
cd /home/yl/matching/sar_matching/MaritimeSARRegistrationPhase2
# 新机器才需要创建，当前环境已经存在：
# conda create -n match python=3.10.22 -y
conda run -n match python -m pip install \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
# 完整实际版本见最终报告对应的 phase2_requirements_lock_YYYYMMDD_HHMM.txt。
# 使用该锁文件安装后，再执行：
conda run -n match python -m pip install --no-deps -e .
bash scripts/install_official_lightglue.sh
conda run -n match python -m pytest -q
```

锁文件安装命令：`conda run -n match python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 -r reports/<对应锁文件>`。只安装 `opencv-python-headless`，不与其它 OpenCV 发行包共存。官方 LightGlue 固定提交 `eb42fee2d71449efb0aa5c10549752b5d75384d8`，只将其 requirements 的 OpenCV 发行名换为 headless；网络实现未改动，补丁及权重 SHA 保存在最终证据中。安装脚本校验缓存权重，避免盲目重下。

驱动 535.288.01 / RTX4090 已验证 Torch 2.5.1+cu121 可用。当前自动执行环境的普通沙箱看不到 CUDA，正式模型推理在主机权限下执行；这不表示 match 安装失败。GPU 7 的物理 UUID 和实时负载保存在 provenance；复现前按实际服务器负载选卡，不中断他人任务。

## 冻结数据及实验复现

```bash
ROOT=/home/yl/matching/sar_matching/MaritimeSARRegDataset/output/phase1_formal_20261009_1846
INDEX=/home/yl/matching/sar_matching/MaritimeSARRegDataset/data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl
conda run -n match python -m sarbench inspect --root "$ROOT"
```

必须为全部 test 来源，原生图像边长 128–384px，GT 和预测均为 view_i→view_j。`source_group` 只在权威 v2 split_protocol 下转换为来源组件；HRSID 背景与父场景共享组件命名空间。

只有确实需要新的全量复现实验时执行：

```bash
SAR_GPU=7 bash scripts/reproduce_phase2.sh
```

它创建 `output/phase2_reproduce_YYYYMMDD_HHMM`，已有目录直接失败；运行固定协议的 CPU 方法、GPU LightGlue、安全合并及独立验收，所有命令记入日志。可设 `SAR_PHASE2_OUT` 为新的路径，或设 `SAR_PHASE1_ROOT` / `SAR_PHASE1_INDEX`，但不得改变正式测试集来选择超参数。只跑传统方法可用 `scripts/quickstart_server.sh`，同样使用 match 和 Phase2 目录。

每次运行保存 `settings.json`、`provenance.json`、`results.csv`、`summary.json`、`cycle_consistency.csv`、图及 `RUN_COMPLETE.json`。合并严格检查结果 SHA、每方法唯一 ID、冻结 receipt/manifest/index 和共同评价/拒绝参数一致。缺少模型依赖/权重直接报错，不跳过或伪造第四种方法。

已有结果的合并、验收与可视化不重新推理：

```bash
conda run -n match python -m sarbench compare \
  --runs output/phase2_classic_20261009_2213 output/phase2_lightglue_20261009_2235 \
  --out /tmp/new_phase2_comparison
conda run -n match python scripts/finalize_phase2.py \
  --comparison /tmp/new_phase2_comparison \
  --log reports/phase2_execution_20261009_2140.log
```

验收脚本重算全部 EPE/状态与 Success@3/5/10，核对 11,093 个冻结文件 SHA，导出实际依赖、权重、源代码补丁、案例和报告。报告和最终日志名称用实际交付分钟；执行日志快照须在 logger 写完退出行后复制到报告指定路径。

## 指标和科学边界

Success@Npx 分母为所有 860 条正样本，要求算法接受且 512 个静态 GT 同名点平均重投影误差≤Npx；无输出和拒绝计失败。40 条无重叠独立计算错误接受率。EPE 均值/中位仅针对已接受且可评价的正样本，**包含错误接受**。

逐对记录四类状态：没有候选 H（no_output）、已有估计被拒绝（rejected）、错误接受（incorrect_accept）、GT Success@5（success）。少量匹配但不足协议最低点数仍可能为 no_output。无重叠错误接受无 GT EPE，不能用数值自洽 H 将其当作成功。

SIFT/ORB mutual ratio=0.8，ORB fastThreshold=12；特征方法共享 Homography RANSAC（3px、5000 次、confidence=.995），最低匹配/内点 8、内点率 .3、4×4 覆盖 1/16、RMSE≤3px；Phase response≥.20，模型仅平移。seed42，OpenCV 单线程；LightGlue 官方默认配置、2048 点、resize=None、无 compile/微调。参数未在正式集选择。

三视图全部 20 组保留，三条边独立估计；只有三边接受时计算整幅 5×5 网格闭环误差，同时报告三边 GT 成功，不能把闭环小当作正确。动态区域仅在冻结静态 GT 中排除；算法本身不读取动态掩膜。

共享源图/场景存在相关性，使用来源组件的描述性 bootstrap，不做伪独立配对置信区间。背景只有两个父场景，来源关联可能不完全；少量标注边界差异保留。CPU 与 GPU/shared load 下的时延不是相同硬件的纯算法比较。未来调参/域微调应另建 source-safe train/val 开发集，真实多视角验证必须独立开展。

官方实现与许可证见 [LightGlue](https://github.com/cvg/LightGlue)，尤其注意其 SuperPoint 文件/权重的上游条款。本工程复用官方代码，不重新实现网络。

## Phase 2.5 独立开发与可靠性研究

本轮数据及缓存结果位于 `output/phase25_20261010_0000`。`data_v2/dev`、`data_v2/val` 各 384 对，来源分别为修正版 train/val；原 900 对 test 未用于选择参数。`development/selection.json` 在验证前封存了参数、数据及推理代码哈希，验证阶段不再选参。`data_v2/confirmation_sources.jsonl` 仅预留 133 个未用于本轮匹配的来源组件，未生成确认图像。

`scripts/phase25_benchmark.py` 缓存图像匹配，复用同一批匹配点比较 Homography/Affine 和八种几何接受策略；所有接受检查均不读 GT/Mask。`scripts/phase25_report.py` 独立核查旧成果及本轮数据、重算指标、输出消融与案例；`scripts/phase25_attribution.py` 仅离线分析动态内点和合成边界。完整复现指令、实际环境、失败和局限见 `reports/phase25_report_*.md`，执行日志同分钟命名。

已有分析直接读取 `analysis/validation_primary.csv`、`stratified_metrics.csv`、`feature_diagnostics.csv`、`failure_mechanisms.csv` 和 `attribution/dynamic_selected_inliers.csv`，无需重跑匹配器。各执行脚本拒绝覆盖输出目录。下一轮不得用已分析的验证结果继续挑选参数；需要收益确认时使用进一步审计的封存确认来源。没有训练或微调网络。
