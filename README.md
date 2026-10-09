# SAR Matching

海面 SAR 二维合成配准数据与图像配准基线代码。

- `MaritimeSARRegDataset/`：真实 SAR 来源适配、来源重复与划分检查、二维多视图合成、标注传播及 GT 验证。
- `MaritimeSARRegistrationPhase2/`：SIFT/ORB + RANSAC、相位相关、官方预训练 SuperPoint + LightGlue，以及统一评价和可视化工具。
- `inspect_sar_dataset.py`：检查本地 SAR 数据目录与标注结构。

这里仅包含代码、测试、配置和使用说明，不包含数据集、源图索引、权重、实验输出、报告或日志。代码可以生成报告，生成文件由 `.gitignore` 排除。原始数据与模型权重遵守各发布者的条款；本仓库不重新分发。

**科学边界：** 单幅 SAR 源图生成的二维代理视图不是真实跨雷达观测角图像，不能将代理测试成绩视为真实多视角性能。

## 环境

所有 Python 操作使用独立 Conda 环境 `match`。不要安装到 base，避免同时安装多个 OpenCV 发行包。

```bash
conda create -n match python=3.10.22 -y
conda run -n match python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu121 \
  -r requirements-match.txt
conda run -n match python -m pip install -e './MaritimeSARRegDataset[dev]'
conda run -n match python -m pip install --no-deps -e ./MaritimeSARRegistrationPhase2
bash MaritimeSARRegistrationPhase2/scripts/install_official_lightglue.sh
```

`requirements-match.txt` 固定依赖版本，PyTorch / torchvision 使用 CUDA 12.1 构建。安装前应检查目标服务器驱动；CPU 或其他 CUDA 平台需安装对应 PyTorch 构建并另行记录环境。LightGlue 安装脚本固定官方提交及权重哈希，仅将其 OpenCV 依赖声明改为 headless，网络代码未修改。

## 使用

两个子目录的 README 提供数据生成、读入与评测入口。正式测试数据保持冻结；匹配器只接收图像，GT 与掩膜仅用于离线评价，不能在正式测试集上选择超参数。

```bash
cd MaritimeSARRegistrationPhase2
conda run -n match python -m pytest -q
conda run -n match python -m sarbench inspect --root /path/to/frozen_dataset
conda run -n match python -m sarbench run \
  --root /path/to/frozen_dataset --source-index /path/to/source_safe_index.jsonl \
  --methods sift orb phase --seed 42 --threads 1 --out output/new_classic_run
```

使用 LightGlue 时设置 `TORCH_HOME` 指向该子项目的 `cache/torch`，或通过 `scripts/run_logged.py` 执行，它会设置项目缓存和完整日志。所有新实验写入新的输出目录，已有目标路径拒绝覆盖。
