# Maritime SAR Registration Dataset Toolkit

**中文版 / Chinese README** — Version 0.3.0 | Python 3.10+

这是一个**不自动下载数据集**的、可直接运行的 SAR 海洋场景配准数据集合成工程。先手动从数据发布者处下载并解压 HRSID、SSDD；项目会自动读取 COCO JSON 或 VOC XML 舰船标注，构建双/多视角合成图像、精确二维几何变换、重叠掩膜、匹配点、舰船框/掩膜，以及质量检查报告。

> **科学边界：** 本工程模拟的是二维视图几何、覆盖变化和 SAR 风格扰动，是 `synthetic_2d_registration_proxy`，**不是真实的跨雷达观测角度电磁散射仿真**。标注的 `rotation_degrees` 是**图像平面旋转**，不是雷达观测方位角差。不得将 synthetic benchmark 结果当成真实 multi-view SAR benchmark。真正真实多视角验证仍需 ICEYE Dwell 或其他数据另行处理。

## 服务器实测目录及使用

第一阶段正式验收已完成，使用 **[正式报告](reports/phase1_formal_report_20261009_2106.md)** 和 `output/phase1_formal_20261009_1846`：860 组固定样本（840 双视图、20 三视图），展开 900 对，860/860 独立验收通过。来源关联审计合并 442 条确认关联，正式索引为 `data/index/phase1_formal_20261009_1846/maritime_source_safe_v2.jsonl`。SSDD 输出自适应 128–256 像素，源图像预上采样为零。

第二阶段直接读取已冻结数据，不必重跑生成：

```python
from sarreg.formal import iter_pairs, load_images, load_groundtruth
root = 'output/phase1_formal_20261009_1846'
for pair in iter_pairs(root):
    image0, image1 = load_images(pair)  # matcher 仅输入 uint8 灰度图
    # 保存算法预测后，单独读取离线评测 GT。
    gt = load_groundtruth(pair)
```

完整分层、固定种子、版本、依赖、31 张预览和 SIFT/LightGlue 输入方式见正式报告。旧 88 组 smoke 与新来源协议不兼容，保留为历史工程验证；不能将旧 train/val 混入正式测试协议。后续 train/val 必须使用新索引的相应 split，test 不用于调参。无重叠 40 对单独评价误配拒绝。517 源图及重复裁片不是 860 次独立采集，背景只有 2 个父组件；近重复检索并非穷尽证明。本数据仍是二维合成代理，不能视为真实跨观测角 SAR。

上一批 7,164 条规范源记录、1,160 张 SL-SSDD 海陆掩膜和 88 组预检见 **[历史 Phase 1 报告](reports/phase1_report_20261009_1637.md)**。下面的原流程保留用于重现历史工程检查，正式数据使用上面的固化入口。

### 一条命令重现 Phase 1

```bash
# 先建立依赖完整的 .venv，或设置 PHASE1_PYTHON 指向现有 Python 3.10+。
bash scripts/run_phase1.sh ../dataset
```

脚本只读取原始数据，使用新的分钟时间戳目录，导出审计、四类 smoke、固定近岸/离岸/背景样例、静态三视图检查、预览和汇总。CPU 线程固定为 1。已有输出会拒绝覆盖。它保留官方划分索引，并另建 `maritime_leakage_safe.jsonl`：由相同文件连接的父场景整体并入已有 test/val；该文件是**新的显式实验协议**，不是官方 SSDD benchmark split。当前默认有放回采样仍然保留。

分类 mask 均采用最近邻插值，`sea` 必须与 `valid` 联用。`validate` 检查对应点的有效、重叠和动态区域支持；`scripts/summarize_phase1.py` 额外独立核对逐像素反采样、源实例传播及三视图组合。预览新增有效/双向重叠/海陆面板，颜色语义为红色参考、青色对齐次视图。

每次生成的精确配置写入 `generation_config_<split>.json`；根目录 `generation_config.json` 保留第一次调用，避免不同 split 改写旧来源记录。重跑已有目标目录会在写入 manifest/config 前失败，显式 `--overwrite` 除外。生成器不改原始图像。

## 一、手动获取数据

| 数据 | 官方入口 | 提取码 | 用途 |
|---|---|---|---|
| HRSID | https://pan.baidu.com/s/1EitHYJ7tpGh0q9Qb3QA5SQ | vxrs | 主数据、COCO/舰船标注 |
| SSDD | https://pan.baidu.com/s/1Lpg28ZvMSgNXq00abHMZ5Q | 2021 | 补充数据、VOC/COCO |
| SL-SSDD | https://github.com/Han-Ke/SL-SSDD/blob/main/SL-SSDD.zip | 无 | 与 SSDD 同名的海陆掩膜 |
| LS-SSDD | https://radars.ac.cn/web/data/getData?dataType=DatasetinthePaper | 平台下载 | 后续大场景扩展；当前无专门适配器，可先用 generic 模式 |

HRSID 备选 Google Drive：https://drive.google.com/file/d/1NY3ovgc-woDlNoQdyqzRB3t9McOBH5Ms/view

SSDD 备选 Google Drive：https://drive.google.com/file/d/1glNJUGotrbEyk43twwB9556AdngJsynZ/view

下载后**自行解压**，建议组织为：

```text
MaritimeSARRegDataset/
  data/
    raw/
      HRSID/                   # 解压后内容，可有 HRSID_JPG/JPEGImages/annotations 嵌套
      SSDD/                    # 解压后内容，可有 BBox_SSDD/voc_style/... 等嵌套
      SL-SSDD/                # 官方 train/*.png、test/*.png 海陆掩膜
    index/
```

HRSID 和 SSDD **不需要改名、不要求手动转格式**。`prepare` 从指定 root 扫描，支持 COCO `images/annotations` 和 VOC `JPEGImages/Annotations`。v0.2 针对 HRSID 与 Official-SSDD-OPEN 选择**唯一的规范标注源**，避免多个 SSDD 分支重复计数。**只放压缩包不解压无法解析**；下载失败不会自动生成或替换成假 SAR 数据。

SL-SSDD 本身不需要作为单独的图像配准源；将它作为 SSDD 的 `--sea-mask-root`。确保该参数指向**仅含海陆 mask 的目录**，不要指到同时包含 SSDD 图像的父级目录，否则可能错配。v0.3 会严格按照 SSDD 官方 train/test 关联 `SL-SSDD/train/*.png` 与 `SL-SSDD/test/*.png`；默认只要有一张缺少对应 mask 就失败，避免静默丢失。需要允许不完整掩膜时显式添加 `--allow-partial-sea-masks`。

## 二、环境配置

```bash
cd MaritimeSARRegDataset
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m pytest -q
```

没有 GPU 要求。依赖：NumPy、OpenCV-headless、PyYAML、tqdm。需要正确解码 COCO 压缩 RLE 实例时可额外安装 `pycocotools`；没有该包会明确使用 bbox 代理掩膜（不是像素级真值）。

## 三、下载前可运行的软件自检（非 SAR 数据）

```bash
bash scripts/quickstart.sh
# 或分开执行：
python -m sarreg demo --out data/demo --count 24
python -m sarreg prepare --dataset hrsid --root data/demo --out data/index/demo.jsonl
python -m sarreg generate --index data/index/demo.jsonl --config configs/easy.yaml --split all --count 8 --out output/demo
python -m sarreg validate --root output/demo
python -m sarreg visualize --root output/demo --out output/previews --count 8
```

**`demo` 只生成程序自检的虚构图像，绝对不是 SAR 数据，不能用于正式论文或性能实验。**

## 四、读取真实 HRSID / SSDD

### 单独 HRSID

```bash
python -m sarreg prepare \
  --dataset hrsid --root data/raw/HRSID \
  --out data/index/hrsid.jsonl
```

### 单独 SSDD，附加海陆掩膜（可省略 `--sea-mask-root`，且不能使用 SSDD 自带舰船实例 mask 当作 sea-land mask）

```bash
python -m sarreg prepare \
  --dataset ssdd --root data/raw/SSDD \
  --sea-mask-root data/raw/SL-SSDD \
  --out data/index/ssdd.jsonl
```

### 合并两套数据源

```bash
python -m sarreg merge-index \
  --indexes data/index/hrsid.jsonl data/index/ssdd.jsonl \
  --out data/index/maritime.jsonl
```

输出 index.jsonl 每行包含：`dataset`、原始图像路径、`split`、舰船 annotation、`sea_mask_path`、`source_group`。HRSID 默认按原始场景前缀（如 `P0001`）做 hash 分组切分，`official_split` 仍保存在索引；SSDD 默认使用官方 train/test，训练部分划出 val。可通过 `--split-policy official` 强制 HRSID 官方划分，但同一原始场景的不同裁片可能跨 train/test。没有可识别 parent-scene ID 的 SSDD 只能按已有官方 train/test 划分或图像 ID 分组，**无法保证隐藏的同航次/同原景裁剪不存在跨 split 重复**。做严格论文实验时应按原始场景归属进行人工复核。

如果只有无标注本地图像：

```bash
python -m sarreg prepare --dataset generic --root /path/to/images --out data/index/unlabeled.jsonl
```

这类数据不会编造舰船框、分割 mask。`--dataset hrsid` / `ssdd` 若找不到标注会明确报错。单独导入 HRSID 的 400 张纯背景 PNG 时使用 `--dataset background`。

## 五、生成数据

```bash
# 第一轮建议先生成 10 组小规模样本
python -m sarreg generate \
  --index data/index/hrsid.jsonl \
  --config configs/default.yaml \
  --split train --count 10 --seed 42 \
  --out output/hrsid_reg

python -m sarreg validate --root output/hrsid_reg
python -m sarreg visualize --root output/hrsid_reg --out output/hrsid_preview --count 10

# 生成更多数据（按各自 split 调用，可指向同一根目录）
python -m sarreg generate --index data/index/hrsid.jsonl --config configs/default.yaml --split val --count 100 --out output/hrsid_reg
python -m sarreg generate --index data/index/hrsid.jsonl --config configs/default.yaml --split test --count 200 --out output/hrsid_reg
```

如果切换配置或种子重新生成同一 split，须显式指定 `--overwrite`，或使用新的 `--out`。原始图像从索引对应 split 采样，可重复采样同一源图来合成不同扰动；输出目录根据 split 隔离。

配置：`configs/easy.yaml`、`configs/default.yaml`、`configs/hard.yaml`。

主配置参数：

| 参数 | 含义 |
|---|---|
| `views` | 2～6 张图像组成一组 |
| `output_size` | 输出方形尺寸（默认 384） |
| `rotation_degrees` | 图像平面旋转最大幅度 |
| `scale_min/max` | 图像仿射尺度 |
| `shear_degrees` | 剪切幅度 |
| `perspective_fraction` | 相对图像尺寸的投影角点扰动 |
| `min_overlap/max_overlap` | 每个新视图相对参考视图的最小双向重叠比例范围 |
| `min_valid_fraction` | 输出图像必须有多少比例来自原始场景（防止大片填充边缘） |
| `max_sampling_attempts` | 重叠采样失败的最大重试次数 |
| `augmentation` | 两个视图分别进行强度、散斑、模糊仿真 |
| `dynamic_ships.enabled` | 是否启用近似船舶局部位移（困难模式，默认关闭） |

注意：视图 1、2 的重叠比例可因有效像素数量不同而略有差异；所有真实的重叠比例均在元信息中记录。若配置要求在某张源图尺寸条件下不可行，生成器会**报失败样本，不会伪造达标的 GT**。为了支持 HRSID/SSDD 小图，默认可能对源图做上采样，并在元信息 `source_resize_factors_xy` 明确记载。

## 六、输出格式

```text
output/hrsid_reg/
  generation_config.json
  manifest_train.jsonl
  generation_report_train.json
  manifest_val.jsonl
  generation_report_val.json
  train/
    pair_000000/
      view_0.png
      view_1.png
      valid_0.png
      valid_1.png
      ships_0.png
      ships_1.png
      dynamic_0.png
      dynamic_1.png
      overlap_0_1_in_0.png
      overlap_0_1_in_1.png
      matches_0_1.npz
      sea_0.png             # 存在 SL-SSDD mask 时才生成
      sea_1.png
      meta.json
  val/
  test/
```

`meta.json` 提供：

- `H_source_to_view[k]`：真实源图（如有上采样，则对应 resize 后坐标）到第 k 张视图的单应矩阵。
- `pairs[].H_i_to_j`：列向量约定的 `H_j @ inverse(H_i)`，**坐标顺序**是 `(x,y)`。
- `pairs[].overlap_fraction_i/j`、`pairs[].overlap_mask_i/j`。
- `pairs[].matches`：NPZ 文件 `points_i` / `points_j`，均为 (N,2) 的浮点型 `(x,y)` 对应；如果启用局部船舶运动，则从静态 GT 中排除变更区域。
- `ship_annotations[k]`：舰船实例 ID、裁剪之后的 `bbox_xywh`、保留估计比例、是否为 bbox 代理掩膜（VOC XML）。
- `ships_k.png` 是来自 polygon/RLE 或 bbox 代理的掩膜。若只有 VOC box，它并非真实船舶分割标注。
- `sea_k.png` 来自外部掩膜，在原尺寸一致时按同样几何变换传播。**0 还可能表示视图的填充无效区域**，因此使用时务必与 `valid_k.png` 联合判断。

源 SL-SSDD mask 必须与对应 SSDD 图像宽高相同，且像素仅含 0（land）/255（sea）；v0.3 在生成时验证这两个条件，并在 `validate` 阶段检查 `sea_k.png` 的二值性、文件完整性及其没有越出 `valid_k.png`。海陆 mask 仅作场景分组/评测分析，不会自动作为静态同名点 GT。

若 `views=3`，所有组合 `0-1,0-2,1-2` 都会输出各自变换与匹配 GT；不过只强制视图 k 和 view_0 满足指定重叠条件，不强制其它两两视图都重叠。

## 七、评价与验收

```bash
python -m pytest -q
python -m sarreg validate --root output/hrsid_reg
python -m sarreg visualize --root output/hrsid_reg --out output/previews --count 12
```

`validate` 检查逆变换一致性、相对 Homography、重叠掩膜、导出匹配点的重投影误差，以及 bbox 越界；支持无网络运行，不依赖模型。`visualize` 导出 view0、view1（绿色框为舰船标注），以及变换之后的红/青叠加图。

这只是软件 GT 和数值一致性验证，不证明合成的 SAR 视角散射是否接近真实观测。

## 八、限制与后续扩展

1. 输入只需要灰度 SAR 图像，若下载的是复杂 SLC 数据，需要先独立完成成像或幅度转换；本包不提供电磁仿真或 SAR 原始回波成像。
2. synthetic 同源裁剪包含共享的原始散斑指纹；视图独立增加噪声不等同于真实独立观测的散斑去相关。
3. 默认只同步处理轴对齐检测框和可用实例像素 mask；对源数据的旋转框不会保留最小旋转矩形方向信息。
4. 可选的舰船局部位移使用近似图像处理，运动区域不作为静态几何 GT；不代表真实舰船动力学或 SAR 运动成像。
5. 本包没有真实多视角配准数据的下载或读取功能，也不包含 LightGlue 模型训练代码；这一版只承担“用户下载真实单视角 SAR 舰船数据 → 构建合成配准训练/测试集”的任务。
6. 生成后的 source paths 是本机绝对路径；迁移机器前需保留来源元信息或重新 `prepare`。
7. 第三方数据本身不随本 ZIP 分发，请遵守各数据集发布方许可与学术引用要求。

## 九、排错

- `No COCO/VOC labels`：检查解压后的 HRSID `annotations/*.json` 或 SSDD `Annotations/*.xml` 是否存在。
- `Could not sample overlap`：降低 `min_valid_fraction`、减少 `output_size`、放宽 `min_overlap/max_overlap` 或增加 `max_sampling_attempts`。
- 没有海陆 mask：检查 `--sea-mask-root` 是否指向正确的 PNG mask 文件夹，且与 SSDD 同名；不会自动将海面推断出来。
- GitHub 下载的仓库只有 README：官方大数据通常存放在百度/Google Drive，不是 `git clone` 就会自动包含完整训练集。

## 十、快速命令速查

```bash
# 先做软件自检
bash scripts/quickstart.sh

# 真 HRSID 实验
python -m sarreg prepare --dataset hrsid --root data/raw/HRSID --out data/index/hrsid.jsonl
python -m sarreg generate --index data/index/hrsid.jsonl --out output/hrsid_reg --split train --count 1000 --seed 42
python -m sarreg validate --root output/hrsid_reg
python -m sarreg visualize --root output/hrsid_reg --out output/hrsid_vis --count 12
```
