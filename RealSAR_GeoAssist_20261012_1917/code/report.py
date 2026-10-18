"""Actual geocoding evidence and AI search-region preparation; no human/GT fabrication."""
import sys,json,csv,platform
sys.dont_write_bytecode=True
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,rasterio,PIL
from geometry import ROOT,BASE,SOURCE,queue,lines,View,mapped,sha,save
from records import Records

# Broad search regions estimated from actually inspected Native previews, not landmarks/GT.
REGIONS={
 'ANGLE_e76cc299fa7b':([.22,.46,.40,.30],'港池内部突出码头、防波堤分叉与地面连接处；先查低矮固定边界。'),
 'ANGLE_e3bfffe1139e':([.22,.46,.40,.30],'与18°共享左采集的港池码头/防波堤；两侧形状可辨但不能把共享采集计作独立样本。'),
 'ANGLE_75c261e08eae':([.16,.20,.32,.35],'工业半岛岸线和永久码头入口；避开储罐顶和屋顶强亮点。'),
 'ANGLE_5baa3ae0a7c1':([.05,.27,.48,.30],'港池伸出的永久码头轮廓和地面接合处；必须区分船舶与码头。'),
 'ANGLE_5da40401a853':([.10,.25,.80,.20],'横向工业港池岸线、永久岸壁转角；另一侧水面亮度显著变化。'),
 'ANGLE_cc7e4d5fcb78':([.48,.15,.24,.65],'重复锯齿码头的岸侧连接与整体岸线转折；单个亮线/泊位身份歧义较大。'),
}

def run():
    now=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=now.strftime('%Y%m%d_%H%M')
    rows=queue();tasks={t['sample_id']:t for t in lines(ROOT/'tasks.jsonl')}
    previous={r['pair_id']:r for r in lines(SOURCE/'ai_native_triage.jsonl')}
    notes=[]
    for row in rows:
        sid=row['sample_id'];task=tasks[sid];left=None;right=None;note='已有V2候选；本轮核验坐标与原生入口，未新增真人可辨识性结论。'
        if row['pair_id'] in REGIONS:
            box,note=REGIONS[row['pair_id']];a,b=[View(row,i) for i in [0,1]]
            x,y,w,h=[box[0]*a.width,box[1]*a.height,box[2]*a.width,box[3]*a.height]
            left=[int(round(x)),int(round(y)),int(round(w)),int(round(h))]
            polygon=[[x-.5,y-.5],[x+w-.5,y-.5],[x+w-.5,y+h-.5],[x-.5,y+h-.5]]
            right=mapped(a,b,polygon).tolist()
        elif row['pair_id'] in previous:note=previous[row['pair_id']]['note']
        item=dict(sample_id=sid,pair_id=row['pair_id'],site=row['site_name'],component=row['component'],
            native_TIFFs=[v['native_tif'] for v in row['views']],native_display_PNGs=[str(SOURCE/'review/regions'/sid/f'native_db{i}.png') for i in [0,1]],
            native_dimensions=[[v['width'],v['height']] for v in task['views']],human_priority=task['priority'],
            AI_search_note=note,left_approx_search_bbox_xywh=left,right_geographic_reference_polygon=right,
            source='current AI Native preview inspection' if left else ('previous AI triage reused' if row['pair_id'] in previous else 'previous V2 candidate reused'),
            human_verdict=None,fixed_structure_identity_confirmed=False,GT_qualified=False,
            risks=row.get('risks',[]),region_semantics='broad search only; projected region is not corresponding physical landmarks')
        notes.append(item)
    with (ROOT/'gt_candidate_preparation.jsonl').open('x') as f:
        for item in sorted(notes,key=lambda d:d['human_priority']):f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    store=Records(ROOT/'assisted_triage',tasks)
    ready=[store.load(sid) for sid in tasks if store.load(sid).get('verdict') in ['A','B'] and store.load(sid).get('recommend_independent_annotation')]
    save(ROOT/'gt_readiness.json',dict(at=now.isoformat(),actual_assisted_A_B_recommended=len(ready),independent_GT=0,
        geometry_batches_created=False,projected_points_exported=0,
        missing='actual human triage plus two actual declared unexposed people and independent clicks/re-clicks',
        old_GT_tools_reused=['prepare_gt.py','serve_gt.py'],exposure_guard='code/gt_bridge.py; known same-acquisition exposure propagated and refreshed'))
    validation=json.loads((ROOT/'mapping_validation.json').read_text());browser=json.loads(sorted((ROOT/'reports').glob('browser_check_*.json'))[-1].read_text())
    assert validation['passed'] and browser['passed']
    cache=json.loads((ROOT/'display_cache.json').read_text())
    primary=[t for t in tasks.values() if t['priority']<=6]
    records={sid:json.loads((ROOT/'validation'/sid/'mapping_verification.json').read_text()) for sid in tasks}
    math_error=max(t['max_math_roundtrip_px'] for t in validation['pairs'])
    clicks=browser['actual_bidirectional_click_tests'];mapping_times=[t['mapping_ms'] for t in clicks];ui_times=[t['click_to_both_crops_ms'] for t in clicks]
    report=[f'# Geo Assist 地理映射核验与真实GT候选准备 — {stamp}',f'\n核验交付时间 {now.isoformat()}，Asia/Shanghai。独立目录 `{ROOT}`。',
        '\n## 访问与实际成果',
        '\nCPU服务已启动于 **http://127.0.0.1:8773/**，只监听loopback；8772未修改。18项原有队列全部可选，原始Native TIFF/PNG只读复用，新增下载、采集身份和真实GT均为0。登记真实操作者后，两侧概览均可点击，另一侧自动显示地理参考并读取原生局部窗口；局部128/256/512/1024px支持4倍/2倍/1:1/概览，局部点击与四向移动同样双向映射。',
        f'\n真实核验18对、包含{validation["mixed_CRS_pairs"]}对实际不同CRS（Melbourne EPSG:32755→4326），六组优先候选各保存Native原生显示上的覆盖轮廓与点位示意，每方向9点数学检验。最大数学往返误差{math_error:.3e}px。**这不是共同散射中心的真实位置误差；物理位置误差未测量，不能把投影当GT。**',
        '\n## 原生坐标与元数据证据',
        '\n以实际TIFF读取CRS和完整Affine，包含旋转/剪切/任意轴向；整数Native坐标表示像素中心，正向先加0.5、逆向减0.5，x为column、y为row。CRS转换使用Rasterio x/y语义，WGS84输出为lon/lat。不把parent_window_xywh的继承字段重新加到已裁片Affine；实际窗口与原产品Affine用原始窗口偏移独立核对，缺原产品时核对冻结context裁片链。',
        '\n所有本轮实际窗口的本地头检查未发现GCP、RPC或GEOLOCATION数组。六组优先源为已地理编码Umbra GEC，保存侧车的原source RPC标签为空。实现检测非线性来源：可用GCP走GCPTransformer；缺高程依据的RPC/未实现的GEOLOCATION明确拒绝，不自动退回Affine。GCP路径当前未用真实GCP影像验证；当前成功仅指现有地理编码产品坐标网格。它不验证传感器3D投影或地理定位绝对精度。',
        '\n官方依据：[Rasterio transforms](https://rasterio.readthedocs.io/en/stable/topics/transforms.html)、[Rasterio CRS transformations](https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html)、[Umbra Collect STAC Metadata](https://docs.canopy.umbra.space/docs/collect-metadata)。产品是GEC的证据及源链接保留在实际映射记录中；高度、叠掩和阴影风险不因数值往返小而消失。',
        '\n绿色轮廓为两份实际裁片边界在本地UTM中的加密凸交集；不是双方真实散射体共见区域，亦不把无效掩膜空洞算有效。每次点位另外检查Native范围和最多2×2像素支撑的掩膜/有限值（实际IO为相应栅格块）；目标坐标出界时保持原预测并提示，不夹到边界制造对应。',
        '\n|优先采集关系|方位差°|数学往返上限px|两侧地理覆盖比例|证据|\n|---|---:|---:|---|---|']
    for t in primary:
        record=records[t['sample_id']];error=max(v['max_math_roundtrip_px'] for v in record['views']);frac=record['coverage']['fractions']
        report.append(f"|{t['pair_id']}|{t['azimuth_delta_deg']:.3f}|{error:.2e}|{frac[0]:.3f}/{frac[1]:.3f}|[原生示意](../validation/{t['sample_id']}/bidirectional_mapping_preview.png)|")
    report+=['\n## 浏览器、效率与失败记录',
        f'\n实际Chromium {browser["browser"]}只读加载全部18项，并对六组优先样本及一组实际跨CRS样本完成左→右、右→左和128px局部再点击。服务器映射加有效性检查{min(mapping_times):.1f}–{max(mapping_times):.1f}ms；本机浏览器从点击到两侧局部显示{min(ui_times):.0f}–{max(ui_times):.0f}ms（包含测试轮询，不是网络部署保证）。没有POST、真人判断、暴露记录或GT点。实际截图及逐项记录见 `browser_check_20261012_1933.json`。',
        f'\n缓存来自已有dB PNG一次解码，{cache["distinct_display_windows"]}个唯一显示窗口引用到36个任务侧，np.memmap约{cache["bytes"]/1e9:.3f}GB；它们覆盖29个采集身份，不是新增来源或采样。三份重复硬链接显示缓存在本轮自有cache内去重，删除前计划/身份保留。所有Native强度原件保留；只生成显示缓存，没有地理重采样图。',
        '\n实际浏览器首轮检查错误地要求概览鼠标点击等于精确预设Native坐标；发现CSS鼠标量化后，以实际点击Native坐标验证数学预测，并单独记录概览鼠标偏差。第二轮等待复用了同侧旧结果，改为当前请求序号与局部坐标都一致后检查。两项失败、执行代码快照和修正均保留在evidence；第三轮全部通过。错误属于核验口径/异步等待，不能删除失败或当作真实SAR位置误差。',
        '\n## 辅助记录与GT隔离',
        '\n真实界面先记录操作者、采集对、时间和地理暴露，才显示辅助。判断A/B/C/D、粗略区域/位置描述、理由和下一步建议作为assisted_triage追加版本并校验hash链。记录在本目录assisted_triage，与8772独立Native判读和旧标注隔离；状态始终GT_qualified=false。预测点仅供当前浏览，不复制为真实标点；保存的窗口也是搜索区域，不是对应点。',
        '\n自动核验会话由本任务随机只读凭证限制保存；常规页面匿名辅助入口被实际浏览器验证拒绝。真实操作者没有默认身份或默认合格判断。存储输入拒绝/GT字段注入测试通过，未伪造真人正向保存；第一份实际辅助判断由用户本人完成。',
        '\n候选准备清单gt_candidate_preparation.jsonl含全部18项原生路径/尺寸、组件、已有风险；六项提供AI粗略固定结构搜索框及右侧地理投影范围，明确不是物理同点，其余复用原AI/V2说明，不编造新的可辨识性结论。独立GT仍需未看过任何辅助/算法对应的真实人员标点与真正独立第二人复核；无法确认可留C/D，不凑点。',
        '\ncode/gt_bridge.py在实际辅助A/B且人工建议后才准备新workspace，要求明确的两个不同真实未暴露身份；复制实际辅助记录时保留assisted_triage语义和源SHA，不模拟Native判读，不导出地理点。复用已有prepare_gt/serve_gt，并将已知8772/V2/GeoAssist的相关暴露传递至独立质量检查，包含共享采集关系，正式服务时持续刷新。其他历史辅助仍必须如实声明。当前实际prepare返回无资格项，未创建workspace/几何批次、未启动正式GT端口。',
        '\n## 冻结、资源与复现',
        '\n全程match、CPU、现有NumPy/Rasterio/OpenCV/Pillow和既有Chromium；未安装依赖、未使用GPU、未改训练代码/索引/checkpoint/后台任务。中央交接只读，最新M1/M2与正式验收均completed，真实异向无新的精度证据。本轮最终文件身份和旧174份直接保护文件复核见最终核验记录。',
        '\n入口与运行命令见[README](../README.md)，独立GT准备见[GT说明](../GT_CANDIDATE_PREPARATION.md)。初始条件检查脚本针对未开始真人阶段；后续记录追加后应如实核验，不能清空记录让初始零记录检查通过。']
    path=ROOT/'reports'/f'geo_assist_report_{stamp}.md'
    with path.open('x') as f:f.write('\n'.join(report)+'\n')
    save(ROOT/'environment_identity.json',dict(at=now.isoformat(),python=sys.version,executable=sys.executable,numpy=np.__version__,rasterio=rasterio.__version__,Pillow=PIL.__version__,GPU=False,new_installation=False))
    save(ROOT/'report_identity.json',dict(report=str(path),at=now.isoformat(),GT=0,downloads=0,assisted_records=len(ready)))
    print(path)

if __name__=='__main__':run()
