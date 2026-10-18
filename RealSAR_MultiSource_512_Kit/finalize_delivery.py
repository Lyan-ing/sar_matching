"""Assemble accepted candidate indices and a truthful Chinese acquisition report.

Requires completed engineering QA. Independent-GT status remains unverified.
"""
import argparse,csv,datetime,difflib,hashlib,json,re,subprocess,sys,zipfile
from collections import Counter,defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np,rasterio
from PIL import Image,ImageDraw
from prepare_sources import lines
from build_real_pairs import display

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def finish(run,base):
    main_quality=run/('quality_retry1' if (run/'quality_retry1/ACCEPTANCE.json').exists() else 'quality')
    main=json.loads((main_quality/'ACCEPTANCE.json').read_text());assert main['engineering_QA_passed']
    entries=lines(main_quality/'release_index.jsonl');quality_sets=[main_quality];sources=lines(run/'sources.jsonl');extras=[]
    ext=run/'s1_sf_extension'
    if (ext/'quality_retry1/ACCEPTANCE.json').exists():
        q=json.loads((ext/'quality_retry1/ACCEPTANCE.json').read_text())
        if q['engineering_QA_passed']:
            entries+=lines(ext/'quality_retry1/release_index.jsonl');quality_sets.append(ext/'quality_retry1');sources+=lines(ext/'sources.jsonl');extras.append(q)
    city=run/'melbourne_zero_aware'
    city_quality=city/('quality_retry1' if (city/'quality_retry1/ACCEPTANCE.json').exists() else 'quality')
    if (city_quality/'ACCEPTANCE.json').exists():
        q=json.loads((city_quality/'ACCEPTANCE.json').read_text())
        if q['engineering_QA_passed']:
            entries+=lines(city_quality/'release_index.jsonl');quality_sets.append(city_quality);extras.append(q)
    ports=run/'ports_recovery'
    if (ports/'quality/ACCEPTANCE.json').exists():
        q=json.loads((ports/'quality/ACCEPTANCE.json').read_text());assert q['engineering_QA_passed']
        entries+=lines(ports/'quality/release_index.jsonl');quality_sets.append(ports/'quality');extras.append(q)
    source_map={s['acquisition_id']:s for s in sources};split_keys=defaultdict(set);decoded=defaultdict(list);source_pairs=set();files={};failures=[]
    for row in entries:
        folder=Path(row['folder']);row.update(reference_native_png=str(folder/'reference.png'),moving_native_png=str(folder/'moving_native.png'),moving_coarse_png=str(folder/'moving.png'),reference_native_geotiff=str(folder/'reference_float.tif'),moving_native_geotiff=str(folder/'moving_native_float.tif'),moving_coarse_geotiff=str(folder/'moving_float.tif'),provider_metadata_reference=source_map[row['reference_acquisition_id']]['metadata_path'],provider_metadata_moving=source_map[row['moving_acquisition_id']]['metadata_path'])
        metadata=json.loads((folder/'metadata.json').read_text());lo,hi=metadata['display_db_range'];units=metadata['input_units'];native_masks=[]
        source_pairs.add(tuple(sorted((row['reference_acquisition_id'],row['moving_acquisition_id']))))
        split_keys[str(('component',row['component']))].add(row['split'])
        for side in ['reference','moving']:
            aid=row[side+'_acquisition_id'];s=source_map[aid]
            for field in ['datetime','end_datetime','platform','relative_orbit','absolute_orbit','orbit_direction','observation_direction','azimuth_degrees','incidence_degrees','polarizations','input_units','license','license_url','asset_url']:
                row[side+'_'+('utc' if field=='datetime' else field)]=s.get(field)
            row[side+'_physical_resolution_metadata']={k:v for k,v in s.items() if 'resolution' in k or k=='base_ipr_m'}
            for key in [('collect',aid),('site',s['provider'],s['site_name'])]+[('original',x) for x in s.get('original_products',[])+s.get('datatake_ids',[])]:split_keys[str(key)].add(row['split'])
            path=Path(row[side+'_native_geotiff'])
            with rasterio.open(path) as ds:
                a=ds.read(1);valid=ds.read_masks(1)>0;native_masks.append(valid)
                assert ds.width==ds.height==512
                assert np.array_equal(np.asarray(Image.open(row[side+'_native_png'])),display(a,valid,lo,hi,units))
                mask_file='reference_valid.png' if side=='reference' else 'moving_native_valid.png'
                assert np.array_equal(np.asarray(Image.open(folder/mask_file))>0,valid)
            decoded[hashlib.sha256(a.tobytes()).hexdigest()].append({'pair':row['pair_id'],'chip':row['chip_id'],'side':side,'split':row['split']})
        with rasterio.open(folder/'moving_float.tif') as ds:
            a=ds.read(1);valid=ds.read_masks(1)>0
            assert ds.width==ds.height==512
            assert np.array_equal(np.asarray(Image.open(folder/'moving.png')),display(a,valid,lo,hi,units))
            assert np.array_equal(np.asarray(Image.open(folder/'common_valid.png'))>0,native_masks[0]&valid)
        for path in folder.iterdir():
            if path.is_file():files[str(path.relative_to(run))]=sha(path)
    leaks={k:sorted(v) for k,v in split_keys.items() if len(v)>1};duplicates=[v for v in decoded.values() if len(v)>1];duplicate_leaks=[v for v in duplicates if len({x['split'] for x in v})>1];assert not leaks and not duplicate_leaks
    assert len({(r['pair_id'],r['chip_id']) for r in entries})==len(entries)
    preserved=json.loads((run/'protected_artifacts.json').read_text());changed=[]
    for n,(path,value) in enumerate(preserved.items()):
        f=base/path
        if not f.exists() or sha(f)!=value:changed.append(path)
        if (n+1)%5000==0:print('Preservation check',n+1,'/',len(preserved),flush=True)
    assert not changed,changed[:5];save(run/'evidence/preservation_after.json',{'checked':len(preserved),'mismatches':changed})
    # Completion timestamps use the actual Asia/Shanghai clock, including day rollover.
    now=datetime.datetime.now(ZoneInfo('Asia/Shanghai'));stamp=now.strftime('%Y%m%d_%H%M')
    (run/'dataset_index.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in entries))
    pinned=[{'metadata_relative_path':str((Path(r['folder'])/'metadata.json').relative_to(run)),'expected_file_sha256':{f.name:sha(f) for f in Path(r['folder']).iterdir() if f.is_file()}} for r in entries]
    (run/'released_windows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in pinned))
    fields=[k for k in entries[0] if k not in ['native_reference_footprint','native_moving_footprint']]
    with (run/'dataset_index.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in entries)
    used_ids={r[k] for r in entries for k in ['reference_acquisition_id','moving_acquisition_id']};used=[source_map[k] for k in sorted(used_ids)];(run/'downloaded_acquisitions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in used))
    grouped=defaultdict(list)
    for r in entries:grouped[(r['sensor'],r['site_name'],r['scene_class'],r['split'])].append(r)
    statistics=[]
    for key,g in sorted(grouped.items()):statistics.append({'provider':key[0],'site':key[1],'scene':key[2],'split':key[3],'source_pairs':len({r['pair_id'] for r in g}),'crop_pairs':len(g),'unique_acquisitions':len({r[k] for r in g for k in ['reference_acquisition_id','moving_acquisition_id']}),'different_look_pairs':len({r['pair_id'] for r in g if r['look_pair_category']=='different_look_verified'}),'same_look_pairs':len({r['pair_id'] for r in g if r['look_pair_category']=='repeat_pass_same_look'}),'unknown_look_pairs':len({r['pair_id'] for r in g if r['look_pair_category']=='orientation_unknown'})})
    save(run/'source_statistics.json',statistics)
    with (run/'source_statistics.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(statistics[0]));w.writeheader();w.writerows(statistics)
    for path in [run/'real512_candidates_v2/SUMMARY.json',ext/'data/SUMMARY.json',ext/'data_retry1/SUMMARY.json',city/'data/SUMMARY.json',ports/'data/SUMMARY.json']:
        if path.exists():failures.append({'stage':str(path.relative_to(run)),'failures':json.loads(path.read_text()).get('failures',[])})
    pair_looks=Counter(next(r['look_pair_category'] for r in entries if r['pair_id']==pid) for pid in sorted({r['pair_id'] for r in entries}))
    quality={'engineering_QA_passed':True,'candidate_crop_pairs':len(entries),'real_acquisition_pairs':len(source_pairs),'pair_manifest_rows':len({r['pair_id'] for r in entries}),'unique_real_acquisitions':len(used_ids),'providers':dict(Counter(s['provider'] for s in used)),'sites':len(grouped),'source_components_used':len({r['component'] for r in entries}),'splits_crop_pairs':dict(Counter(r['split'] for r in entries)),'look_acquisition_pairs':dict(pair_looks),'native_geographic_overlap_strata':dict(Counter(r['geographic_overlap_stratum'] for r in entries)),'cross_split_known_identity_leaks':leaks,'cross_split_exact_pixel_leaks':duplicate_leaks,'exact_duplicate_groups':duplicates,'all_native_pixels_equal_remote_originals':True,'minimum_coarse_valid_fraction':min(r['coarse_valid_fraction'] for r in entries),'independent_GT_verified_pairs':0,'Success_metrics_computed':False,'frozen_files_checked_unchanged':len(preserved),'stage_failures_retained':failures,'quality_sets':[str(p) for p in quality_sets],'prior_synthetic_acquisition_independence':'Cannot prove: HRSID/SSDD original acquisition metadata is incomplete; old real 14TPN pilot belongs to same train site component','prospective_confirmation_is_not_sealed_test':True,'completed_at':now.isoformat()}
    save(run/f'quality_report_{stamp}.json',quality)
    # A single contact sheet containing accepted sources only, with true images and attribution.
    panels=[]
    for key,g in sorted(grouped.items()):
        r=g[0];f=Path(r['folder']);c=Image.new('RGB',(1536,600),'white');d=ImageDraw.Draw(c)
        for i,name in enumerate(['reference.png','moving_native.png','moving.png']):c.paste(Image.open(f/name).convert('RGB'),(i*512,58))
        d.text((5,5),f"{r['sensor']} | {r['site_name']} | {r['look_pair_category']} | native / native / coarse | NO INDEPENDENT GT",fill='black');d.text((5,580),'Umbra SAR CC BY 4.0; Copyright Capella Space Corp CC BY 4.0; Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery, 2026',fill='black');panels.append(c.resize((768,300)))
    sheet=Image.new('RGB',(1536,math_ceil(len(panels)/2)*300+40),'white');d=ImageDraw.Draw(sheet);d.text((5,5),'Distinct real SAR acquisitions, 512 native samples. Contact sheet downsized for overview; full images preserved. NO INDEPENDENT PIXEL GT.',fill='black');d.text((5,20),'Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026; Umbra SAR / Capella Space CC BY 4.0.',fill='black')
    for i,c in enumerate(panels):sheet.paste(c,(768*(i%2),40+300*(i//2)))
    sheet.save(run/f'preview_{stamp}.png')
    diagnostic=json.loads((run/'image_only_diagnostics/SUMMARY.json').read_text());syn=json.loads((run/'synthetic512_dev_v1/feasibility.json').read_text())
    rows=['| 来源 | 站点 / 场景 | 划分 | 真实采集对 | 512 裁片对 | 唯一采集 | 明确异向对 |','|---|---|---|---:|---:|---:|---:|']
    rows += [f"| {s['provider']} | {s['site']} / {s['scene']} | {s['split']} | {s['source_pairs']} | {s['crop_pairs']} | {s['unique_acquisitions']} | {s['different_look_pairs']} |" for s in statistics]
    methods=['| 冻结方法 | 原生图像诊断对数 | 图像门控接受数（不是成功数） | 平均耗时 ms | 中位耗时 ms |','|---|---:|---:|---:|---:|']
    for method,m in diagnostic['methods'].items():methods.append(f"| {method} | {m['evaluated_source_pairs']} | {m['image_only_gate_accepted']} | {m['mean_ms'] if m['mean_ms'] is not None else '未运行'} | {m['median_ms'] if m['median_ms'] is not None else '未运行'} |")
    env=json.loads((run/'environment/environment.json').read_text());probe=json.loads((run/'s1_verified/summary.json').read_text());discovery=json.loads((run/'evidence/static_stac_limitations.json').read_text())
    report=f"""# 多来源真实 SAR 512 数据工程报告

完成时间：{now.isoformat()}。目录开始于 2026-10-10 23:52；报告、日志使用实际完成分钟 {stamp}，跨日后未沿用旧时间戳。

本轮交付 **{len(source_pairs)} 个不同真实采集对、{len(entries)} 对 512×512 原生候选裁片**，涉及 {len(used_ids)} 次真实采集、{len(grouped)} 个站点、{len(quality['providers'])} 类来源。原生指提供方发布的 RTC/GEC 栅格采样，不是未处理 SLC。两个视图来自不同采集，均保留未经本工具重投影的原生裁片；另存移动影像粗地理重投影版本。工程 QA 通过；**独立人工 GCP/保留检查点为 0，全部真实对仍为 unverified，不能报告 Success@3/5/10px 或精确 GT 配准误差。**

## 数据入口与分层

统一入口 `dataset_index.jsonl` / `dataset_index.csv`；采集身份、UTC、传感器/卫星、轨道、左右视、方位/入射角、极化、产品分辨率、许可证、原始侧车和公开 URL 见 `downloaded_acquisitions.jsonl`、`verified_sources.csv`、`sources.jsonl` 及 `s1_sf_extension/`。未知轨道/角度字段明确为空，未用日期推断方向。`source_statistics.csv` 给出站点级统计；多裁片和重复使用的采集对不能充当独立采集样本。

场景名称是来源站点类别，不是逐裁片人工地表标注。实际查看的 Sunny_Isles 首张示例主要为海面散斑，不能因站点标签 urban_coast 就宣称每张都有城市建筑；逐裁片的城市/水域/山地含量尚未人工标注。`evidence/visual_review.json` 保存实际预览检查观察，AI 预览检查不替代独立人工 GCP 复核。

{chr(10).join(rows)}

采集对角度类别：{dict(pair_looks)}。异向规则在构建前固定：提供方明确升降轨/左右视变化，或两次方位角的圆周差≥15°；同向要求同相对轨道，或方位及入射差均≤2°。其余保守归入 orientation_unknown，包括角度缺失和未满足两端规则的中间变化；原值保留供后续审查。原生裁片的**粗地理**重叠分层：{quality['native_geographic_overlap_strata']}。分辨率不同会使两幅固定 512 原生裁片的物理覆盖不同；`common_valid.png` 是移动完整来源重投影后的网格有效区，`quality*/overlap_masks` 另外保存两幅实际原生裁片的地理有效重叠。它们都不是独立对应 GT。

划分按原始 collect/product/datatake、站点、相交来源范围和空间邻域关联组件处理。同站点所有采集、派生格式和裁片留在一个组件。当前已知身份跨划分边为 0、原生像素精确重复跨划分为 0。旧 14TPN 真数据试点与新 South_Dakota 来源同归 train；HRSID/SSDD 的历史采集元数据不完整，**无法证明新真实集与旧合成源完全没有未知采集或地理关联**。组件独立性只限已公布元数据；当前 confirmation 是工程候选分区，已被查看，不是严格封存无偏测试集。

## 获取与工程质量验收

- match Python：{env['version'].split()[0]}；关键依赖：{env['packages']}。本轮没有安装或改动依赖，不使用 .venv/rfrock，不修改 base；完整锁定见 `environment/`。
- 原包校验 15 文件通过，原有 13 个离线测试通过；修复后检查整型 nodata、裁片分离、旋转网格插值一致性及空集拒绝，最终测试日志随交付保留。
- 旧三时相 512 烟雾测试 3/3 配对通过，未把它们重复计入上述新数据规模。
- 每个正式裁片均重新读取提供方原始窗口，验证浮点导出数值逐像素一致、原生 CRS/变换/窗口一致、512 尺寸与 PNG 展示变换一致；所有接受原生视图有效率≥98%，粗网格共同有效率最小 {quality['minimum_coarse_valid_fraction']:.4f}。固定纹理门槛为输入量纲对数标准差≥0.5；对于非校准 DN，这是展示/筛查量，不是物理后向散射指标。
- Umbra 新版的 COLLECT_ID 与原始侧车 collect_id 核对；旧 GEC 无内置采集 ID 时，单幅原始 METADATA 侧车固定 SHA、collect/task UUID、同目录公开 URL 和声明 GEC 行列数须全部匹配，并单列缺少嵌入 ID 的证据等级。Capella GeoTIFF 内置采集 ID 与 extended sidecar 核对；Sentinel-1 原始 SAFE 精确 UTC、相对/绝对轨道与原始 GRD 身份核对。若 RTC 同一日期含多幅 GRD 镶嵌，本轮主来源选择保守排除，不把混合来源假装成单幅采集。
- 原生参考裁片在每个源对内不重叠；移动原生窗口可能因栅格尺度差异而空间部分重叠，详见独立 QA 记录，不能据此增加独立采集数。遥感来源本身已做 GEC/RTC 采样；本工具原生导出未扩大尺寸或插值放大。
- 有效性筛查额外要求功率/振幅大于 0。部分旧 Umbra uint8 产品未声明 nodata，但含量化零值；这些像素也被保守排除为 -9999，而不是宣称它们均为原始 nodata。Melbourne 中心诊断窗口正值比例约 95.3%，未通过固定 98% 门槛。此规则可能减少暗阴影区域的代表性；未为凑数量降低阈值，保留原来源、排除规则和探测证据。
- 针对已查明的量化零值另建 `melbourne_zero_aware` 小补充批次：对主批次已处理且没有裁片的两对来源固定 seed260513，按原始声明掩膜保留非负样本，声明 nodata=0 的像素仍排除。原生有效零值保存为 0；对数纹理及显示百分位仅使用正样本，避免 log(0) 的截断制造纹理或扭曲色阶。有效率仍要求≥98%，主批次及其已生成裁片没有更改。索引中 `allow_declared_valid_zero` 区分两种规则，新增本地测试覆盖 nodata=None 与 nodata=0。
- 通过检查的原始裁片文件 SHA、全新输出 receipt、配置/源码快照、Git 版本、模型权重 SHA、依赖和实际命令均保留。遥远整幅产品只下载所需窗口，完整产品 SHA 明确为 null；没有把 URL 或 S3 ETag 当成整幅文件 SHA。
- 第一阶段、Baseline v1、Phase2.5、Phase2.6 和旧三时相试点的 **{len(preserved):,} 个冻结文件** 结束时逐一 SHA 校验未变。

## 失败、修复和未通过内容

Umbra 静态 STAC 读取到 {discovery['umbra_static_features']} 个真实元数据条目，但资产链接为空，原扫描器可下载清单为 0。已用真实公开 S3 任务目录与原始 METADATA / STAC v2 侧车补充，固定 seed 抽取六站点 48 个来源。第一次 22 条侧车命名兼容失败全部保留，修复新版 `.stac.v2.json` 后 48/48 侧车核验通过。Capella 扫描 1,499 个 GEC 元数据条目，截断剩余 16 条，绝不将整个扫描数声称为下载量；选定三个多时相站点，逐项核实极化、采集 ID、尺度因子和栅格。

Sentinel-1 初始 30 条目标来源探测选中 {probe['sources']} 条，排除日志详见 `source_retrieval_outcomes.json`，主要为多幅 GRD 镶嵌或目标区域无效。旧金山补充使用目录确实存在的 ±12 天重复观测；2020-06-26 候选未满足单幅/有效性检查，未发布。2020-06-02 / 06-14 下载先遇 TLS EOF，按相同来源和 seed 在新目录重试成功。旧空输出的错误验收结果已通过 INVALIDATED.json 明确作废，新增回归测试要求空数据集不能通过。

整型 GeoTIFF 在转浮点前填 NaN 的兼容性错误、UTC 日期占位与 Capella 纳秒时间解析、移动原生有效率门槛、旋转网格采样间隔计算、裁片重复采样问题均已修复。远程整幅 band 重投影耗时较高，改为带 2 像素支持边界的窗口读取；发现 GDAL 自动插值尺度与分块有关，明确固定 XSCALE=YSCALE=1 的点双线性插值并验证整幅/窗口一致。两次中断的构建中间文件保留，不进入正式统一索引；没有覆盖这些尝试，也没有按匹配结果调参。

Melbourne 的首次补充 QA 被旧产品缺少 COLLECT_ID 拒绝；首次整体验收为 false，未发布。修复为原始同目录侧车/task/尺寸的严格替代核验，同时将源缓存登记移到所有身份检查通过之后，防止失败后后续裁片绕过该检查。首次失败结果保留，重新验收写入 quality_retry1；新增测试确保尺寸或采集 ID 不一致仍被拒绝。

执行被中断时，旧后台进程使用的本地代理失联；主构建余下 20 个港口来源对报连接失败，第一次主 QA 的 122 个裁片全部拒绝。新会话匿名 HEAD 恢复 HTTP 200 后，主裁片在 quality_retry1 重新验收，固定的港口来源表另写 ports_recovery，seed260514 / 30 尝试 / 3 裁片，使用已验证的声明有效零值规则；没有修改旧批次、放宽 98% 门槛或更换来源以冒充成功。第一次失败结果和 32 条主构建拒绝/网络记录全部保留。

主构建与补充尝试的拒绝记录：

```json
{json.dumps(failures,ensure_ascii=False,indent=2)}
```

## 原生图像配准诊断与独立 GT 边界

沿用 Phase2.5 冻结特征设置（SIFT contrast=.01；ORB edge=16；SuperPoint threshold=1e-4，官方 LightGlue 缓存权重），2048 特征请求、Affine、full 可靠性门控、seed42。每个主批次实际源对取一张固定原生裁片，仅给 matcher `reference.png` 与 `moving_native.png`，不读取真实 H、Mask、角度或地图坐标进行预测；不对当前数据选择参数，不训练模型。端到端时间包含 PNG 读取、预测和几何判断，模型初始化单列，首调用冷启动注明；旧金山、Melbourne 零值补充和恢复的港口批次不在这次主批次诊断分母。

{chr(10).join(methods)}

完整原因、特征/匹配/内点、拟合残差、真实耗时、预测矩阵及示例覆盖图在 `image_only_diagnostics/`。门控接受和 RANSAC 内点残差不是独立精度，不能用它们给方法排真实 SAR 成功率。观察到不同方向、地形叠掩、散斑/亮度和固定像素物理范围变化，现有全局 Affine 不能保证有效；需要独立静态对应点后才能区分匹配与几何模型错误。

本轮无真实人工静态点输入，`independent_gcps_PENDING.csv` 只有表头，未伪造标注。下一步独立人工标注分散的控制点及保留检查点、记录 reviewer/source/不确定性；分别在各原生视图定位，不能从粗地理编码或预测匹配抄 GT。复杂山地可仅释放稀疏静态对应点或局部几何，不能强行用全局 H。真实舰船的实例、运动与动态排除区也未获得人工标注，不能声称已验证舰船检测或动态配准性能。

## 512 合成分支

新建 `synthetic512_dev_v1`，从修正版 train 的三个 HRSID 来源组件选 800px 原始图像，output_size=512、min_source_side_factor=1.35、禁止来源上采样。实际生成 {syn['generated']} 组，全部 GT/Mask/标注传播基础 QA 通过，source_resize_factors_xy=[1,1]。旧生成器可直接支持这一新尺寸，未修改冻结源码；小 SSDD 不应放大凑量。它们仍是单幅 SAR 的二维合成代理，不能计入真实多采集数量，也不构成真实角度证据；这是开发可行性样本，不能与旧 900 对 Success 直接混合比较。

## 使用与复现

```python
import json, cv2
from pathlib import Path
root = Path({str(run)!r})
row = json.loads((root / 'dataset_index.jsonl').read_text().splitlines()[0])
image0 = cv2.imread(row['reference_native_png'], cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread(row['moving_native_png'], cv2.IMREAD_GRAYSCALE)
# 给 SIFT/ORB/LightGlue 只传 image0/image1。moving_coarse_png 是预处理版本，勿混入原生配准实验。
```

原生 float32 TIFF 保留提供方 DN/功率值。PNG：Sentinel-1 固定 [-25,0] 展示；X 波段振幅 DN 用两幅原生图联合 2/98 百分位的对数展示，确切界限写在 metadata.json。不同时间的非校准 DN 尺度/量化可能不同，PNG 跨景亮度不能解释成物理散射变化。Capella 的 sidecar scale_factor 保留，sigma0_dB=20log10(scale_factor*DN)；Umbra 未假定绝对 sigma0 校准。

顺序随机采样遇网络失败时，后续 RNG 消耗会改变；仅 seed + 来源 CSV 不能保证逐裁片重现。这一限制已处理：`released_windows.jsonl` 锁定每张实际下载窗口、原始 metadata 的确切显示范围及 10 个导出文件 SHA，`REPRODUCE.sh` 使用 `code_snapshot_final/replay_released_windows.py` 在全新目录重放并逐文件核对 SHA，不进行随机补量。已实际重放一张真实旋转网格 Umbra 裁片，10 个文件字节哈希全部一致；离线测试也验证整个导出流程字节一致。完整数据重放需来源对象不变且使用锁定依赖，否则会明确拒绝哈希不一致的输出。

`verified_sources.csv` / 各补充来源表、`code_snapshot`（实际主构建版本）、`code_snapshot_final`、实际命令和 `REPRODUCE_PLANNED_BATCHES.sh` 保存原始采样计划；重新扫描会遇提供方新上传记录，不能用它替代已锁定来源和窗口。新数据以原生图像/来源隔离进行开发，先补独立 GT 与辐射/地形规范，再研究是否需要 SAR 域微调，不据本轮门控接受率决定训练方案。

## 许可与官方依据

Umbra Open Data：CC BY 4.0，来源与变更注明；[AWS 官方目录](https://registry.opendata.aws/umbra-open-data/)、[Umbra GEC 产品说明](https://help.umbra.space/product-guide/umbra-products/umbra-product-specifications/geo-ellipsoid-corrected-gec-geotiff)。Capella Open Data：CC BY 4.0、保留提供方署名和裁剪/展示变更；[Capella 许可](https://www.capellaspace.com/legal/data-licensing)、[产品量纲说明](https://support.capellaspace.com/what-sar-imagery-products-are-available-with-capella)。Indigo RTC：按其研究/非营利许可使用，不能默认视作同一开放商业许可；[官方数据格式](https://sentinel-s1-rtc-indigo-docs.s3-us-west-2.amazonaws.com/data_format.html)、[Indigo Atlas Sentinel License](https://www.indigoag.com/forms/atlas-sentinel-license)。

图像署名：Umbra SAR Open Data；© Capella Space Corp, All Rights Reserved；Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026。所有裁片/预览处理方式及来源保留。
"""
    report_path=run/f'multisource512_report_{stamp}.md';report_path.write_text(report)
    # Rename each completed stage log, then assemble a readable full execution log.
    for f in sorted(run.rglob('*.log')):
        if not re.search(r'_\d{8}_\d{4}$',f.stem):f.rename(f.with_name(f.stem+'_'+stamp+f.suffix))
    logs=sorted(run.rglob('*.log'));log_path=run/f'execution_{stamp}.log'
    with log_path.open('x') as f:
        f.write(now.isoformat()+' COMPLETED real candidate engineering; independent GT pending\n')
        for stage in logs:f.write('\nSTAGE '+str(stage.relative_to(run))+'\n'+stage.read_text(errors='replace'))
        f.write('\nFINAL ACCEPTANCE '+json.dumps(quality,ensure_ascii=False)+'\n')
    kit=base/'RealSAR_MultiSource_512_Kit';snapshot=run/'code_snapshot_final';snapshot.mkdir()
    codefiles=list(kit.glob('*.py'))+list((kit/'tests').glob('*.py'))+[kit/'README.md',kit/'CODEX_HANDOFF.md'];codehash={}
    for f in codefiles:
        t=snapshot/f.relative_to(kit);t.parent.mkdir(parents=True,exist_ok=True);t.write_bytes(f.read_bytes());codehash[str(f.relative_to(kit))]=sha(f)
    with zipfile.ZipFile(base/'RealSAR_MultiSource_512_Kit_v2.zip') as original:
        originals={Path(n).name:original.read(n).decode() for n in original.namelist() if n.endswith('.py')}
    changes=[]
    for f in codefiles:
        if f.suffix=='.py':
            changes.extend(difflib.unified_diff(originals.get(f.name,'').splitlines(keepends=True),f.read_text().splitlines(keepends=True),fromfile='uploaded/'+f.name,tofile='delivered/'+str(f.relative_to(kit))))
    (run/'code_changes_final.diff').write_text(''.join(changes))
    save(run/'code_provenance_final.json',{'snapshot_sha256':codehash,'phase2_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=base/'MaritimeSARRegistrationPhase2',text=True).strip(),'original_package_sha256':sha(base/'RealSAR_MultiSource_512_Kit_v2.zip'),'frozen_core_modified':False})
    checkpoint=json.loads((run/'CHECKPOINT.json').read_text());checkpoint.update(updated_at=now.isoformat(),status='Engineering delivery completed; independent precision GT remains unverified',next='Independent human static GCP/check-point and crop-semantic review; no training or retuning in this run',report=str(report_path),execution_log=str(log_path));save(run/'CHECKPOINT.json',checkpoint)
    save(run/'DELIVERY_COMPLETE.json',{'engineering_accepted':True,'independent_GT_accepted':False,'completed_at':now.isoformat(),'report':str(report_path),'execution_log':str(log_path),'dataset_index':str(run/'dataset_index.jsonl'),'quality':quality,'frozen_files_unchanged':len(preserved),'file_sha256':{str(f.relative_to(run)):sha(f) for f in run.rglob('*') if f.is_file() and f.name!='DELIVERY_COMPLETE.json'}})
    print('DELIVERED',len(source_pairs),'real acquisition pairs',len(entries),'native crop pairs',report_path,flush=True)

def math_ceil(x):return int(np.ceil(x))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--base',required=True);a=p.parse_args();finish(Path(a.run).resolve(),Path(a.base).resolve())
