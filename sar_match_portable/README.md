# SAR配准推理代码

多尺度结构增强SAR-SuperPoint＋低纹理特征保持迁移的纯图像推理与外部多视角批处理入口。源码沿用正式模型的灰度归一化、检测、256维描述子、冻结LightGlue以及Homography/full验收流程。

**Git版本只包含代码、配置和说明。** 数据、示例图、训练权重、报告和生成输出均不发布。完整内部迁移包与公开源码仓库不是同一交付物。

## 环境与外部组件

使用Python3.10，按本目录`requirements.txt`安装依赖；或使用`environment.yml`。不要同时安装多个OpenCV发行包，不修改原服务器base环境。

LightGlue的Apache-2.0源码与许可在仓库中。SuperPoint源码及预训练/派生权重适用不同的限制性研究许可，未纳入Git。自行阅读`licenses/SuperPoint_RESEARCH_LICENSE.txt`，确认有权使用后，准备固定版本的本地源码：

```bash
conda run -n match python scripts/prepare_superpoint.py --accept-superpoint-license
```

也可通过`--source-file /path/to/legally_obtained/superpoint.py`离线准备。脚本严格核对官方固定commit的源码SHA，只移除构造函数的自动权重下载，不更改网络计算。

通过有权方获取本项目的最终研究权重，按照`configs/formal_protocol_20261018_0142.json`中的相对路径放置于`checkpoints/`。默认final、structure和pretrained三种模型只需准备所选特征模型及LightGlue；程序必须通过对应SHA校验。不提供公开原始模型冒充本研究最终模型的回退功能。严禁将受限源码或权重重新提交到本仓库。

## 双图与批处理

本仓库不附带图像，必须传入自己的图像：

```bash
conda run -n match python scripts/smoke_test.py \
  --reference /path/A.png --moving /path/B.png --device cpu --offline-check

conda run -n match python scripts/test_external_dataset.py \
  --data-root /path/to/external_dataset --mode all --scan-only

conda run -n match python scripts/test_external_dataset.py \
  --data-root /path/to/external_dataset --mode smoke \
  --scene scene_000001 --device cuda:0

conda run -n match python scripts/test_external_dataset.py \
  --data-root /path/to/external_dataset --mode all \
  --device cuda:0 --output-dir outputs/external_eval
```

输入为同尺寸8位SAR PNG，原生每边64～640像素，不自动重采样或重新选择辐射归一化。默认每图至多1024个点；`--budget 128`为可选固定截断预算。其他匹配与几何配置锁定，不按新数据结果调整。

递归扫描`scene_000001_inc20_az000.png`命名；不跨scene配对。同一场景默认inc={20,30,45}、az={0,60,120,180,240,300}，18视角生成153对。`inc/az`只作分组标签，不作为网络输入或二维旋转先验。支持`--pair-type same_inc|same_az|both_different|all`，每类固定首对快速模式、完整配对、缺失/重复/不可读警告。

每次输出新的时间戳目录，包含逐对CSV、场景摘要、总体JSON、NPZ和有数量上限的匹配图。`H_AB`方向为A图原生像素→B图原生像素，候选H与`accepted`分开保存。无独立GT时只报告匹配数、内点支持、几何接受率和内部残差，不输出真实配准准确率或Success@N。

## 可选区域诊断

`--annotations annotations.json`仅用于预测完成后的舰船/背景统计，不参与匹配、几何估计或接受。JSON顶层`images`按相对data-root路径索引，每幅图可提供`boxes:[{"xyxy":[x0,y0,x1,y1],"instance_id":"ship_1"}]`，或`mask`相对路径及`instance_ids`映射。`complete:true`才声明未标区域为背景；否则为unknown。mask为同尺寸整数图，0非舰船、正整数为船区域标签。没有ID不判断同船，没有标注不自动猜测区域；同一实例也不等于像素对应正确。

只安装源码依赖即可运行命名/配对、几何与标注结构单元测试；需要权重的测试在缺权重时明确跳过。推理需要事先合法准备权重，缺失即报错。测试命令：

```bash
conda run -n match python -m unittest discover -s tests -v
```
