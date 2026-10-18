# Sentinel-1 真实多次采集 SAR 小样本下载工具

**用途：**在 Linux 服务器上自动检索并截取同一地理网格的 2–3 幅**真实 Sentinel-1 SAR 多次采集**图像，用于开始研究真实 SAR 图像匹配，而不是继续使用同一张 SAR 图的二维变换合成样本。

**重要说明：**这个工具包只有脚本和说明，不含真实 SAR 影像字节；生成本包的环境无法连接外部数据文件服务器。运行脚本后，才会在你的服务器上下载数据。请不要把“有下载工具包”理解成“原始数据已经下载完成”。

## 数据来源

- 官方索引：[AWS Registry of Open Data — Analysis Ready Sentinel-1 Backscatter Imagery](https://registry.opendata.aws/sentinel-1-rtc-indigo/)
- 官方读取指南：[Indigo Sentinel-1 RTC — Dataset Access](https://sentinel-s1-rtc-indigo-docs.s3-us-west-2.amazonaws.com/data_access.html)
- 数据格式：Sentinel-1 真实采集、RTC 地形/辐射校正、VV 通道、浮点 Gamma0、Cloud Optimized GeoTIFF。
- S3 桶：`sentinel-s1-rtc-indigo`，`us-west-2`。官方 AWS 数据目录列明可以**免 AWS 账号匿名访问**（`--no-sign-request`）。
- 数据覆盖：主要是**美国本土**，不是全球覆盖，也不保证最新年份有更新。
- **许可：**遵守 Indigo Atlas Sentinel License；如需重新分发原始图像/衍生数据，先核对许可条款和项目使用条件。

默认试点使用 **`14TPN`** 网格、2020 年的数据，原因是官方指南实际列有该网格 2020-08-01 的 COG 文件路径。它是陆地场景，不包含已验证的舰船标签，目的是先完成真实多时相配准数据下载与验证。

## 第一步：解压到服务器

例如把 `RealSAR_Sentinel1_DownloadKit.zip` 放入 `/home/yl/matching/sar_matching/`：

```bash
cd /home/yl/matching/sar_matching
unzip RealSAR_Sentinel1_DownloadKit.zip
cd RealSAR_Sentinel1_DownloadKit
```

## 第二步：只使用 `match` 环境安装少量依赖

```bash
conda run -n match python -m pip install 'rasterio>=1.4,<2' 'requests>=2.31' 'Pillow>=10'
conda run -n match python -m pytest -q tests
```

`rasterio` 能对远端 COG 发起字节范围请求，只读取小窗口的数据。**不需要先下载几百 MB 的整景图像**。但窗口远程读取可能需要多次 HTTP 请求；实际数据流量取决于 TIFF 内部切片和压缩方式。

## 第三步：先测试能否访问数据索引

```bash
conda run -n match python download_s1_rtc.py \
  --tile 14TPN --years 2020 --discover-only
```

如果成功，控制台会列出真实 Sentinel-1 场景的日期、升降方向和 VV GeoTIFF 数据链接。**这一步只查目录，不下载图像。**

## 第四步：下载三幅真实 SAR 小裁片

```bash
conda run -n match python download_s1_rtc.py \
  --tile 14TPN --years 2020 --n 3 --size 512 \
  --out output/sentinel1_14TPN_2020_pilot
```

选取策略：优先找两次同一升/降轨**方向**、不同日期的真实采集；若有另一观测方向则补充一幅，若没有则继续选用不同时刻的相同方向采集。

**注意：同一升降方向并不代表相同相对轨道，也不等于已知的精确同视角。** 匹配前还需要根据原始产品的相对轨道、入射角等元数据进一步核对。

如果某年可用数据不足，可以扩大检索年份：

```bash
conda run -n match python download_s1_rtc.py \
  --tile 14TPN --years 2019 2020 2021 --n 3 --size 512 \
  --out output/sentinel1_14TPN_multi_year_pilot
```

如果希望只下载两幅先验证，设置 `--n 2`。

## 成功后会得到什么

```text
output/sentinel1_14TPN_2020_pilot/
  manifest.json                      # 数据来源、日期、轨道方向、GeoTIFF哈希
  quicklook_strip.png               # 多次采集并排预览
  S1A_YYYYMMDD_14TPN_ASC_display.png
  S1A_YYYYMMDD_14TPN_ASC_linear_gamma0.tif
  S1A_YYYYMMDD_14TPN_ASC_display.png
  S1A_YYYYMMDD_14TPN_ASC_linear_gamma0.tif
  ...
```

- `*_linear_gamma0.tif`：**科学分析优先使用**，保留浮点后向散射量、坐标系及地理变换。
- `*_display.png`：使用统一 dB 范围 `[-25,0]` 制作的 8-bit 显示图，方便接入已有 OpenCV SIFT/ORB/LightGlue 或查看影像。**显示图不能替代浮点原数据。**
- `manifest.json`：保留每个文件的原始 HTTPS 链接、真实采集日期、轨道**方向**、GeoTIFF 哈希以及公共地理参考网格。

## 能做什么、暂时不能做什么

可做：
1. 对真实多时相 SAR 影像运行 SIFT/ORB/SuperPoint+LightGlue 特征检测及匹配。
2. 评估特征点数量、匹配数、分布、匹配稳定性、投影几何异常及对时相变化的敏感性。
3. 可初步利用同一个地理参考网格做粗对应诊断。

不应直接声称：
1. 图像具有完全独立人工测量的**像素级配准 GT**（RTC 已经地理编码，同网格位置仍受定位误差影响）。
2. 数据一定具有**很大的观测角差异**（同轨方向可能只是重复观测）。
3. 该陆地数据能直接证明海面舰船检测效果。
4. 数据来自未经处理的原始雷达复数回波（这里是地形校正后的强度产品）。

## 常见问题

**`Name or service not known`、连接拒绝或超时**：服务器可能无法访问 AWS S3；先运行：

```bash
curl -I --connect-timeout 10 https://sentinel-s1-rtc-indigo.s3.us-west-2.amazonaws.com
```

如果服务器依赖网络代理，确保当前 shell 具有有效 `HTTPS_PROXY` 环境变量；不能把本地客户端的 `127.0.0.1` 代理端口直接套在远程 Linux 服务器上。

**`403 Forbidden`**：先检查是否使用官方公开桶、URL 及许可政策；此公共镜像的访问策略可能变化。此时不要尝试绕过访问控制，可转用有账号的 NASA ASF/Copernicus 正式下载渠道。

**`no two acquisitions` 或 `Insufficient temporally distinct acquisitions`**：减少 `--n`，扩大 `--years`，或更换 MGRS 网格。

**`Low usable content`**：所选窗口可能落在无效或纹理极弱的区域；用新的输出目录调整 `--x-fraction 0.35 --y-fraction 0.6`，不要在同一个输出目录覆盖原始失败记录。

**读取 COG 报错**：需支持 HTTPS range 的 GDAL/rasterio 编译；可升级 rasterio 或先让 Codex 检查 TIFF 是否可通过普通 GET/Range 访问，不要默认先下载整个影像。

## 数据隔离与后续配准评价

该试点不必马上并入原来已经冻结的合成正式测试集。应另设 `real_sar_pilot/`，先验证真实采集身份、不同日期/不同轨道和有重叠的静态地物，再决定配准真值获取方案。

参考项目已有 `phase3_real_sar_requirements` 中的准入原则，尤其保留采集 ID、相对轨道、入射角、原始产品来源、处理链、静态匹配点的独立验证及舰船单视图标注（后两者在该工具中**尚不存在**）。
