# MaritimeSARRegistrationPhase2

在冻结二维合成 SAR 配准数据上评价四类图像方法：SIFT + RANSAC、ORB + RANSAC、Phase correlation（仅平移）、官方预训练 SuperPoint + LightGlue + RANSAC。不训练模型、不在正式测试集选参数，不用 GT 或掩膜辅助预测。

## 安装与测试

依赖和 Conda `match` 环境说明见仓库根 README。

```bash
cd MaritimeSARRegistrationPhase2
conda run -n match python -m pip install --no-deps -e .
bash scripts/install_official_lightglue.sh
conda run -n match python -m pytest -q
```

官方 LightGlue 固定提交 `eb42fee2d71449efb0aa5c10549752b5d75384d8`。安装脚本缓存官方权重并验证 SHA256，只更改 OpenCV 的依赖发行名，避免 headless 与 GUI 包共同覆盖 `cv2`。查看 [官方实现和许可证](https://github.com/cvg/LightGlue)，尤其 SuperPoint 的上游条款。

## 输入及运行

数据来自第一阶段的冻结输出，每个 `pair_*` 文件夹含 `meta.json`、灰度 `view_i.png`、`pairs[*]` 对应的 GT 和有效区域信息。保持原生像素尺寸；预测 H 为 view_i→view_j。正式输入需要 `ACCEPTED_*.json` 验收文件和 test 来源。`--allow-unfrozen` / `--allow-non-test` 仅供独立开发或软件自测。

```bash
ROOT=/path/to/frozen_dataset
INDEX=/path/to/source_safe_index.jsonl
conda run -n match python -m sarbench inspect --root "$ROOT"
conda run -n match python -m sarbench run \
  --root "$ROOT" --source-index "$INDEX" --methods sift orb phase \
  --model homography --seed 42 --threads 1 --out output/new_classic_run
TORCH_HOME="$PWD/cache/torch" CUDA_VISIBLE_DEVICES=0 \
  conda run -n match python -m sarbench run \
  --root "$ROOT" --source-index "$INDEX" --methods lightglue --device cuda \
  --model homography --seed 42 --threads 1 --out output/new_lightglue_run
conda run -n match python -m sarbench compare \
  --runs output/new_classic_run output/new_lightglue_run \
  --seed 42 --out output/new_comparison
```

日志包装器：`conda run -n match python scripts/run_logged.py --log reports/new_run.log -- conda run -n match python -m sarbench ...`。它设置固定 CPU 线程及项目缓存，不覆盖冻结数据。

针对正式第一阶段固定格式的全量工作流可用 `scripts/reproduce_phase2.sh`，通过 `SAR_PHASE1_ROOT`、`SAR_PHASE1_INDEX`、`SAR_GPU`、`SAR_PHASE2_OUT` 设置服务器路径和设备。输出目录必须是新路径；GPU 卡号需要按实际服务器选择。输入未准备好时不要启动生成或全量评测。

## 指标与协议

- Success@3/5/10px：接受且静态 GT 点平均重投影误差不超过阈值；全部正样本为分母，拒绝和无输出都计失败。
- 无重叠独立评价错误接受率，不计算虚假的 GT 成功。
- 明确区分 `no_output`、`rejected`、`incorrect_accept`、`success`。EPE 汇总包括错误接受，且注明仅在已接受估计上计算。
- 特征方法共享 Homography RANSAC（3px、5000 次、confidence=.995）及接受门限；相位相关只估计平移。完整参数见 CLI `run --help` 和生成的 `settings.json`。
- 三视图每条边独立估计，保留不完整三角形；循环一致性与 GT 正确性分开报告。
- 合并检查样本唯一 ID、数据/来源索引哈希、结果文件 SHA 和共同实验参数一致。
- 按来源组件进行描述性 bootstrap，不能把相关源裁片视为独立观测。

每次运行保存配置、逐对预测、分层汇总、环境/模型版本、权重 SHA、耗时及失败原因。运行产物均由仓库 `.gitignore` 排除。CPU/GPU 耗时不是统一硬件速度排名；二维代理数据不验证真实跨视角散射变化。
