# Maritime SAR Matching

面向海事SAR图像的来源隔离数据构建、二维合成监督、特征迁移学习与纯图像匹配代码。

本仓库只发布自主开发代码、静态配置、测试和代码使用说明。数据、来源索引、人工标注、伪标签、模型权重、缓存、日志、实验报告、统计表和生成图表均不发布。研究工作区与Git发布目录分开；原有时间戳目录用于保留代码版本与跨阶段依赖，不在发布时重写冻结实验。

## 主要入口

|目录|内容|
|---|---|
|`sar_match_portable/`|最终研究模型的双图/外部多视角批处理代码；需要使用者合法取得匹配权重，见独立README|
|`SAR_SP_LowTextureKeep_20261017_0908/code/`|可靠结构描述学习与低纹理旧特征保持训练；只更新描述子头|
|`SAR_SP_KeepAblation_20261017_1343/code/`|同采样、同预算的保持约束控制实现|
|`SAR_SP_LG_StructureStable_20261013_2210/code/`|三尺度结构增强分支、特征提取与联合训练代码|
|`SAR_SP_JointTransfer_Mixed640512_20261016_2234/code/`|混合尺寸检测/描述迁移代码|
|`SAR_SP_LG_Train_20261012_1544/`、`SAR_SP_LG_M2_Numerics_20261012_1627/`|早期描述子适配、微调及数值稳定实现|
|`SAR_SP_LG_M3_Detector_20261012_1932/`|检测头适配代码|
|`MaritimeSARRegDataset/`|SAR来源适配、划分审计、二维合成几何及标签传播|
|`SyntheticSAR_V2_20261015_1255/`、`SyntheticSAR_V22_20261015_1352/`|不同合成监督版本的生成、读取、采样与损失接口|
|`SyntheticSAR_Mixed640512_Multicondition_20261016_2114/`|混合原生尺寸、多条件合成数据接口及评价代码|
|`MaritimeSARRegistrationPhase2/`|传统方法、冻结学习式基线、统一几何与离线评价|
|`RealSAR_*`|真实数据获取、原生裁片、元数据检查、人工标注和诊断代码|
|`SAR_ORB_SP_Fusion_20261015_2349/`|历史互补性/融合实现，仅作代码归档|
|`Report_Figures_*/code/`、`tools/`|图表生成、文档导出、存储迁移及代码发布工具；不含生成结果|

进一步的模块索引和环境说明见[docs/README.md](docs/README.md)，方法组件关系见[docs/MODEL_CODE_GUIDE.md](docs/MODEL_CODE_GUIDE.md)。

## 环境

项目开发使用独立Conda环境`match`。不要修改base，不要同时安装多个OpenCV发行包。

```bash
conda create -n match python=3.10 -y
# 先根据本机GPU驱动选择PyTorch构建；以下为项目CUDA12.1依赖示例
conda run -n match python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements-match.txt
conda run -n match python -m pip install --no-deps -e MaritimeSARRegDataset
conda run -n match python -m pip install --no-deps -e MaritimeSARRegistrationPhase2
```

真实数据处理工具和独立推理包各有requirements，按实际需要安装，不一次安装全部研究仓库。LightGlue/SuperPoint的外部准备见各模块README。权重缺失时不能把公开模型当作已训练研究模型的替代。

## 从代码开始

```bash
# 数据工程与传统几何的轻量测试，不启动训练
(cd MaritimeSARRegDataset && conda run -n match python -m pytest -q tests/test_geometry.py)
(cd MaritimeSARRegistrationPhase2 && conda run -n match python -m pytest -q tests/test_reliability.py)

# 外部数据只检查命名和配对，不需要权重，不产生预测
conda run -n match python sar_match_portable/scripts/test_external_dataset.py \
  --data-root /path/to/views --mode all --scan-only
```

训练源码保留原阶段相对目录、输入校验与不可覆盖约束；并非每个历史脚本都可在没有对应私有数据、权重和索引的情况下直接执行。迁移时先提供合法数据、固定来源划分与所需权重，再核对脚本内的数据入口和资源配置，不在新机器直接运行历史自动收尾/训练启动脚本。

## 科学与许可边界

同一SAR源图的二维几何配对不能证明真实不同雷达观测方位下的泛化。纯图像推理不使用GeoTransform、角度、地图位置或人工点作为先验；缺少独立像素GT时，RANSAC支持与几何接受率不能解释为真实配准准确率。

第三方组件遵守原许可，见[docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)。尤其SuperPoint源码及预训练/派生权重不能随本仓库公开再分发；仅提供官方固定来源与本地准备方式。不对整个组合工程施加覆盖第三方条款的宽泛许可证。
