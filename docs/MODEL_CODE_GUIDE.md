# 方法组件与源码入口

## 网络与监督

- 基础特征为SuperPoint，输出检测响应和256维归一化描述子；匹配器为LightGlue。后者在本仓库这些SAR迁移阶段保持冻结，不能因训练了SuperPoint便称训练了LightGlue。
- 描述子适配代码在`SAR_SP_LG_Train_20261012_1544/code/`及`SAR_SP_LG_M2_Numerics_20261012_1627/code/`，分开冻结检测与描述路径，支持精确二维几何监督。
- 三尺度结构分支见`SAR_SP_LG_StructureStable_20261013_2210/code/network_corrected.py`。对输入执行固定多尺度结构响应，用轻量卷积形成描述分支残差；保持原生坐标、8倍描述特征尺度及LightGlue接口。`network.py`为历史实现保留，受控正式版本优先查`network_corrected.py`及对应训练入口。
- 低纹理保持见`SAR_SP_LowTextureKeep_20261017_0908/code/keep_objective.py`。可靠结构位置的描述对应损失与合法同图位置的教师余弦保持组成训练目标。后者不创造跨采集匹配标签；仅更新convDa/convDb，教师、检测器、共享编码与结构分支冻结。
- `keep_data.py`、`mixed_io.py`负责按真实合法监督组织结构流和保持流；`keep_train.py`负责优化、状态及checkpoint。`SAR_SP_KeepAblation_20261017_1343/code/`保留同样本顺序的零保持梯度对照，不将保持流转换为额外结构监督。

## 推理与评价

优先使用`sar_match_portable/`的独立推理入口。其`pipeline.py`只接受两张图像数组，严格校验模型与配置SHA；`estimate.py`和`reliability.py`为冻结几何估计与接受门控；`external.py`负责场景分组和预测之后的可选标注诊断；`visualize.py`仅生成输出，不修改匹配点。

历史训练脚本会跨目录导入前阶段数据、网络或评价辅助函数，因此保留项目相对结构。配置中的数据路径和模型SHA用于复现约束，不是随仓库提供的数据或权重。不得通过清空SHA、修改门槛或随机初始化绕过缺失模型。

## 测试范围

无数据/权重测试验证几何方向、配对、输入验证和接口逻辑，不构成模型性能实验。只有在取得相应数据、固定划分与训练权重后，才运行对应模型评价。开发集、历史已观察基准和真实GT不足的数据应分别描述。
