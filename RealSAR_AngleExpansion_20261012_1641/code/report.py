"""Evidence-backed data delivery, AI triage explicitly separate from human GT."""
import sys
sys.dont_write_bytecode=True
import json,csv,platform
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'RealSAR_MultiViewV2_20261012_1259/code'))
from geo_tools import lines,save,jsonl,sha

NOTES={
 'ANGLE_e76cc299fa7b':(1,'promising_native_structures','两侧港池与突出码头、地面道路形状可辨；18秒同平台相邻采集，仍属一个依赖组件。'),
 'ANGLE_00b88634c45f':(1,'promising_native_structures','两侧有相似港池、码头地面轮廓；22秒同任务不同collect，不把它们算独立站点。'),
 'ANGLE_e3bfffe1139e':(1,'promising_native_structures','40秒、约46°的港区岸线与永久码头值得优先真人核验；不使用罐顶或船舶亮点。'),
 'ANGLE_5baa3ae0a7c1':(1,'promising_native_structures','约57分钟、约76°，两侧存在多段港池和码头轮廓；需区分低矮地面边界与叠掩。'),
 'ANGLE_75c261e08eae':(1,'promising_native_structures','约一天、约36°，两侧工业半岛岸线与突出的永久设施轮廓值得真人局部核验；屋顶、储罐顶不作地面GT。'),
 'ANGLE_b28cf6a2cdc9':(2,'local_structure_needs_review','约一天，防波堤/道路曲线与局部支路可辨；方位差小但入射角差约15°，不是完全同视角控制。'),
 'ANGLE_5da40401a853':(1,'promising_native_structures','约1.54天、约88°，岸线转角、港池与工业区地面道路值得真人查看；水面亮度和阴影显著变化。'),
 'ANGLE_e17fe5f59130':(2,'local_structure_needs_review','约5.59天、约51°，固定防波堤曲线与局部支路可辨，但唯一可定位转角少；保留局部范围限制。'),
 'ANGLE_325c0714a279':(3,'FOV_needs_manual_check','约3.97天、约127°；右侧仅复用2048原生上下文，FOV明显小于左侧。只能真人判断局部共见，不宣称全幅对应。'),
 'ANGLE_cc7e4d5fcb78':(2,'hard_local_structure_needs_review','约13小时、约145°，两侧港口锯齿状岸线与设施网格可辨，但泊位重复、船舶和强亮线带来身份歧义；保留困难样本。'),
}

def run():
    now=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=now.strftime('%Y%m%d_%H%M')
    rows=lines(ROOT/'candidate_regions_complete.jsonl');review=lines(ROOT/'review/candidate_regions.jsonl')
    delivery=json.loads((ROOT/'delivery_summary.json').read_text());audit=json.loads((ROOT/'angle_audit.json').read_text())
    discovery=json.loads((ROOT/'discovery/summary.json').read_text())
    qualified=list((ROOT/'reports').glob('verification_*.json'));assert qualified
    verification=json.loads(sorted(qualified)[-1].read_text());assert verification['passed']
    triage=[]
    for r in rows:
        priority,status,note=NOTES[r['pair_id']]
        triage.append(dict(sample_id=r['sample_id'],pair_id=r['pair_id'],priority=priority,AI_triage_status=status,note=note,
            author='Codex AI',source='actual native overview images inspected, no matcher or Common-FOV',
            human_verdict=None,human_readability_passed=False,point_coordinates=[],gt_qualified=False))
    jsonl(ROOT/'ai_native_triage.jsonl',triage)
    byid={r['pair_id']:r for r in rows};ordered=[byid[r['pair_id']] for r in sorted(triage,key=lambda r:(r['priority'],r['pair_id']))]
    jsonl(ROOT/'human_review_priority.jsonl',[dict(sample_id=r['sample_id'],priority=NOTES[r['pair_id']][0],human_verdict=None) for r in ordered])
    # Predictors can use this index without loading any geographic/GT metadata.
    jsonl(ROOT/'image_only_pairs.jsonl',[dict(sample_id=r['sample_id'],image0=r['views'][0]['native_tif'],image1=r['views'][1]['native_tif']) for r in review])
    acquisition_components={};stations={};pairs=set()
    for r in review:
        sensor={'UmbraZero':'Umbra'}.get(r['sensor'],r['sensor']);stations[r['site_name']]=r['component']
        ids=[]
        for v in r['views']:
            k=sensor.lower()+':'+v['acquisition_id'];ids.append(k)
            assert k not in acquisition_components or acquisition_components[k]==r['component']
            acquisition_components[k]=r['component']
        key=tuple(sorted(ids));assert key not in pairs;pairs.add(key)
    save(ROOT/'source_isolation.json',dict(station_components=stations,acquisition_components=acquisition_components,
        combined_pairs=len(pairs),combined_unique_acquisitions=len(acquisition_components),components=len(set(stations.values())),
        split='diagnostic_development_seen',final_unseen_test=0,shared_acquisition_and_station_must_not_split=True,
        same_task_multiple_collects_correlated=True,random_chip_split_forbidden=True))
    fields=['pair_id','site','azimuth_delta_deg','angle_bin','days','incidence_delta_deg','native0','native1','component','AI_priority','GT']
    table=[]
    for r in rows:
        table.append(dict(pair_id=r['pair_id'],site=r['site_name'],azimuth_delta_deg=r['azimuth_delta_deg'],angle_bin=r['angle_bin'],days=r['days'],
            incidence_delta_deg=r['incidence_delta_deg'],native0=f"{r['views'][0]['native_info']['width']}x{r['views'][0]['native_info']['height']}",
            native1=f"{r['views'][1]['native_info']['width']}x{r['views'][1]['native_info']['height']}",component=r['component'],AI_priority=NOTES[r['pair_id']][0],GT=0))
    with (ROOT/'candidate_summary.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(table)
    text=[f'# 真实 SAR 角度补充与独立 GT 准备 — {stamp}',f'\n完成记录：{now.isoformat()}，Asia/Shanghai。独立目录 `{ROOT}`。',
        '\n## 实际交付与边界',
        f'\n已构建 **{len(rows)} 个此前65对清单中不存在的真实采集对**，五个方位差档各2对；来自17个唯一采集、5个站点/组件。17个采集中12个身份不在原V2的58采集库存中，另5个复用已有身份。12个是新增采集身份，不是新增文件或裁片计数。没有合成视角、重复裁片扩充、模型训练或900/191对重跑。',
        f'\n真人判读队列共{len(review)}项＝10项新增关系＋8项旧V2推荐项，合并后{len(acquisition_components)}个唯一采集、{len(stations)}站点/{len(set(stations.values()))}组件。旧V2八项只读引用，不能再算新增关系。**真人A/B通过=0、独立几何GT=0、真实精度不可评价**。',
        '\nAI实际查看了原生预览，仅作优先级和FOV风险初筛。它不产生真人A/B结论或对应点；AI具体结构提示不显示在独立Native判读页面。看过辅助对应提示的人员仍需申明暴露，不因新任务或新FOV自动恢复独立资格。',
        '\n## 角度、时差与原生窗口',
        '\n角度是元数据最小圆周方位差，边界为[0,20)、[20,40)、[40,60)、[60,90]、(90,180]，不是完整三维视角差。入射角差、升降轨、左右视、真实collect ID、时间、许可和来源侧车继续保留。厂商区分真实collect ID与STAC处理ID，去重使用前者。[Umbra元数据定义](https://docs.canopy.umbra.space/docs/collect-metadata)。',
        '\n|范围|原65对|本轮实际新增Native对|\n|---|---:|---:|']
    for bn in ['0-20','20-40','40-60','60-90','>90','unknown']:
        text.append(f"|{bn}|{audit['original_angle_bins'].get(bn,0)}|{delivery['bins'].get(bn,0)}|")
    text+=['\n|采集对|站点|方位差°|时差天|入射角差°|Native左/右|\n|---|---|---:|---:|---:|---|']
    for t in table:text.append(f"|{t['pair_id']}|{t['site']}|{t['azimuth_delta_deg']:.3f}|{t['days']:.6f}|{t['incidence_delta_deg']:.3f}|{t['native0']} / {t['native1']}|")
    text+=['\n鹿特丹18/22/40秒三个关系共享三个同平台collect，其中两次collect共享task；不是三个独立站点或独立时间实验。约57分钟的75.63°对亦属于同站点组件。杰贝阿里约一天的36.37°对、安特卫普1.54天的88.39°对和釜山13小时的144.92°对补充跨时段与不同站点；长滩5.59天的50.60°关系复用旧采集身份。',
        '\n每采集只保存一份原生窗口，多关系通过路径/硬链接引用，不重复读取相同成功窗口。目标地面边长1500m、最大8192像素，实际范围以各侧原生尺寸/仿射/采样间距为准，不能把目标值当实际共同FOV。16份为有界较大窗口；鹿特丹127°对右侧复用旧2048上下文，FOV明显较小，保留为局部/质量待核验的困难候选。没有原生上采样、Common-FOV重采样或人工点迁移。',
        '\n原始强度与有效掩膜在TIFF中保留。新显示仅作独立线性或dB映射，P1/P99来自不超过约一百万规则采样点；不平滑、不补细节。首份显示沿用完整像素分位数。采样间距不冒充物理分辨率，地理技术覆盖不冒充共见散射体。',
        '\n## 筛选与GT任务状态',
        '\n优先真人判读鹿特丹的18/27/46/76°组、杰贝阿里36°组及安特卫普88°组：预览中存在值得核验的永久码头、港池或工业区地面轮廓。长滩以弯曲防波堤/道路为主，独特转角少，先作局部核验。釜山145°保留为困难组，重复泊位、船舶和强亮线有歧义。鹿特丹127°有明显FOV差异，不能声称全幅对应。完整AI初筛见 `ai_native_triage.jsonl`；这些结论均不是人类资格或精确GT。',
        '\n`gt_staging_manifest.jsonl`冻结18项真实Native尺寸与显示SHA，只是未启用的准备清单。没有实际真人A/B时，复用的prepare_gt已实际返回None，不创建任务/点。通过后再在本目录生成独立批次，要求真人原生点击和真实第二人独立重点击；规则见 `INDEPENDENT_GT_PROTOCOL.md`，不降低既有质量门限。',
        '\n## 搜索覆盖、失败与恢复',
        f'\n公开目录核查7站点，读取377个任务前缀、得到354条侧车记录；合并旧库存及真实身份去重后402采集。地理筛选得到9062个元数据配对候选，其中14天内港口候选五档分别201/115/42/30/325；它们不算下载影像、可评价样本或GT。6个港口枚举全部列出任务，Panama仅80/225；{len(discovery["failures"])}个目录/侧车读取失败，不能称穷尽目录。',
        '\n原生首次阶段因通配导入带入旧V2 ROOT而被防覆盖检查拦截，旧目录未改。首景全量显示/压缩约460秒；保留已完成原生数据后，仅停止核验过命令的自有CPU PID，改用正确字节单位的128MiB GDAL缓存、较快编码和有界显示分位数。原生像素/采集选择不改。控制停止打断了54e07361窗口的一次未保存读取，已单独披露并恢复，未新增身份。',
        '\n四个公开影像读取出现TLS/分块失败。三项尚无保存像素的短间隔候选，仅在同bucket/key的官方path-style端点恢复一次；一项直接复用冻结本地2048上下文，无网络重读。所有失败、首阶段5对输出、恢复日志与最终10对清单均保留；没有按匹配结果替换样本、循环重试成功景或隐藏失败。原生TIFF约0.994GB；显示和硬链接另计。',
        '\n## 核验与资源边界',
        '\n聚焦核验通过：角度边界/未知项、9062无序真实ID配对唯一性、17原生TIFF的SHA与尺寸/CRS/仿射/掩膜块、成对显示尺寸、无像素完全重复的新原生窗口、无真人A/B时GT不生成、空准备清单不计算Success、59份直接保护的旧索引/标注历史/训练源码SHA未变。浏览器只读检查结果另见 `reports/browser_readonly_*.json`，没有模拟真人或提交点。',
        '\n全程使用match与CPU，无环境安装，不使用GPU0/1，不更改其训练代码、数据索引、Checkpoint或后台任务。训练状态只读：M1完成2700步；M2状态文件显示于16:43完成5100步。初始沙盒看不到host PID且nvidia-smi查询失败，不能把该检查解释成训练失败，本任务没有恢复训练。',
        '\n新代码以逐文件SHA为准，不冒用复用Phase2 HEAD作为本轮版本。root .git为空，采用独立目录；Phase2仓库只读且clean，Phase1已有README/SERVER_USAGE修改和报告未动。最终source_isolation限制整采集/站点/组件隔离；已经查看的组仅用于开发/诊断，最终未见test为0。',
        '\n主要入口：`candidate_regions_complete.jsonl`、`candidate_summary.csv`、`image_only_pairs.jsonl`、`review/`、`ai_native_triage.jsonl`、`source_isolation.json`。旧数据、旧报告、旧标注和训练输出均不覆盖。']
    report=ROOT/'reports'/f'data_expansion_report_{stamp}.md'
    with report.open('x') as f:f.write('\n'.join(text)+'\n')
    import rasterio,numpy,cv2,PIL
    save(ROOT/'environment_identity.json',dict(python=sys.version,executable=sys.executable,numpy=numpy.__version__,rasterio=rasterio.__version__,cv2=cv2.__version__,Pillow=PIL.__version__,GPU=False,installation=False))
    save(ROOT/'code_identity.json',dict(new_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'code').glob('*.py'))},
        critical_executed_snapshots={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'evidence').glob('build_windows*executed*.py')},
        reused_files_sha256={str(p):sha(p) for p in [BASE/'RealSAR_MultiViewV2_20261012_1259/code/geo_tools.py',BASE/'RealSAR_AutoLabels_20261011_1149/geo.py',BASE/'RealSAR_MultiViewV2_20261012_1259/code/prepare_gt.py',BASE/'RealSAR_MultiViewV2_20261012_1259/code/readability_server.py',BASE/'RealSAR_GeometryGT_20261011_0220/gt_quality/quality.py']}))
    save(ROOT/'data_identity.json',dict(at=now.isoformat(),files_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'candidate_regions_complete.jsonl',ROOT/'image_only_pairs.jsonl',ROOT/'gt_staging_manifest.jsonl',ROOT/'review/candidate_regions.jsonl',ROOT/'source_isolation.json']}))
    print('REPORT',report)

if __name__=='__main__':run()
