"""Freeze directly used data/code identities and write timestamped delivery report."""
import sys
sys.dont_write_bytecode=True
import json,subprocess
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from geo_tools import ROOT,BASE,lines,save,sha

def main():
    now=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=now.strftime('%Y%m%d_%H%M')
    rs=lines(ROOT/'candidate_regions.jsonl');allr=lines(ROOT/'regions.jsonl');smoke=lines(ROOT/'smoke/results.jsonl')
    pre=json.loads((ROOT/'preregistration.json').read_text());assert pre['config_sha256']==sha(ROOT/'protocol.json')
    assert all(sha(p)==h for p,h in pre['direct_catalogue_sha256'].items())
    assert all(sha(p)==h for p,h in json.loads((ROOT/'protected_history.json').read_text()).items())
    unit=json.loads(sorted(ROOT.glob('unit_test_results_*.json'))[-1].read_text());browser=json.loads(sorted(ROOT.glob('browser_test_results_*.json'))[-1].read_text())
    assert unit['passed'] and browser['passed'];assert len(smoke)==16 and not any(r['decision']=='runtime_failure' for r in smoke)
    def counts(rr):
        return dict(regions=len(rr),acquisition_pairs=len({r['pair_id'] for r in rr}),unique_acquisitions=len({v['acquisition_id'] for r in rr for v in r['views']}),sites=len({r['site_name'] for r in rr}),source_components=len({r['component'] for r in rr}))
    diff=[r for r in rs if r['look_pair_category']=='different_look_verified'];evidence=dict(created_at=now.isoformat(),recommended=counts(rs),different_look=counts(diff),constructed=counts(allr),geographic_only=len(rs),qualified_independent_GT=0,known_height_model=0,requires_height_evidence_for_elevated_scatterer_claims=len(rs),final_unseen_test_components=0,supervised_real_cross_view_learning_ready=False,unit_tests=unit,browser_tests=browser,smoke_calls=16,raw_geography_baselines=8,new_RoMa_calls=0,training_started=False)
    save(ROOT/('acceptance_'+stamp+'.json'),evidence)
    versions=subprocess.run([sys.executable,'-B','-m','pip','list','--format=freeze'],capture_output=True,text=True,check=True)
    with (ROOT/'dependency_versions.txt').open('x') as f:f.write(versions.stdout)
    refs=[BASE/'MaritimeSARRegistrationPhase2/sarbench'/n for n in ['matchers.py','estimate.py','reliability.py']]
    refs += [BASE/'RealSAR_AutoLabels_20261011_1149/geo.py',BASE/'RealSAR_MatchingPilot_20261011_2114/core.py',BASE/'RealSAR_IndependentBenchmark_20261011_2015/serve_review.py',BASE/'RealSAR_IndependentBenchmark_20261011_2015/benchmark_quality.py',BASE/'RealSAR_GeometryGT_20261011_0220/gt_quality/quality.py']
    code=list((ROOT/'code').glob('*.py'))+[ROOT/'protocol.json',ROOT/'TECHNICAL_PROTOCOL.md',ROOT/'quick_review.html',ROOT/'README.md']
    head=subprocess.run(['git','-C',str(BASE/'MaritimeSARRegistrationPhase2'),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    save(ROOT/('code_identity_'+stamp+'.json'),dict(reused_phase2_git_head=head,new_V2_code_uncommitted=True,new_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in code},reused_code_sha256={str(p):sha(p) for p in refs},protocol_sha256=sha(ROOT/'protocol.json'),model_identity=str(ROOT/'smoke/identity.json'),conda_environment='match'))
    data=[p for p in (ROOT/'regions').rglob('*') if p.is_file()]+[ROOT/n for n in ['regions.jsonl','candidate_regions.jsonl','candidate_screening.json','acquisition_inventory.jsonl','acquisition_pair_inventory.jsonl','source_isolation.json']]
    save(ROOT/('data_freeze_'+stamp+'.json'),dict(created_at=now.isoformat(),files=len(data),bytes=sum(p.stat().st_size for p in data),sha256={str(p.relative_to(ROOT)):sha(p) for p in data},mutable_readability_excluded=True,simulation_and_reproduction_fixtures_excluded=True))
    table=['|母区域|观测|时差d|方位差°|地面边长m|Native 左/右|地理有效覆盖 左/右|','|---|---|---:|---:|---:|---|---|']
    for r in rs:
        meta=r['acquisition_metadata'];views=r['views'];dims=['%d×%d'%(v['native_info']['width'],v['native_info']['height']) for v in views];ol=r['native_geographic_overlap'];az=meta['azimuth_delta_deg']
        table.append('|%s|%s|%.2f|%s|%.1f|%s|%.1f%% / %.1f%%|'%(r['sample_id'],'异向' if r in diff else '同向',meta['days'],'未知' if az is None else '%.2f'%az,r['ground_edge_m'],' / '.join(dims),100*ol[0]['fraction_valid'],100*ol[1]['fraction_valid']))
    srows=['|样本|方法|协议|候选/内点|状态|端到端ms|','|---|---|---|---:|---|---:|']
    for r in smoke:srows.append('|%s|%s|%s|%d/%d|%s|%.1f|'%(r['sample_id'],r['method'],r['protocol'],r['n_matches'],r['n_inliers'],r['decision'],r['end_to_end_ms_synchronized']))
    context_counts=Counter(v['context_provenance']['actual_read'] for r in allr for v in r['views']);total_s=sum(r['end_to_end_ms_synchronized'] for r in smoke)/1000
    report=f'''# RealSAR-MultiView V2 自动交付验收

生成时间：{now.isoformat()}，Asia/Shanghai。目录：`{ROOT}`。本轮完成数据与评价工程，停止于真实人工快速判读前；没有训练、微调或全量191对重推理。

## 可核实结论

最终推荐 **6个明确异向母区域＋2个同向对照**，来自8个不同采集对、15次唯一采集、6个站点/6个来源组件。异向子集来自6个采集对、11次唯一采集、4站点/4组件。三组鹿特丹采集对属于同一站点组件，不能算三个独立站点。全部8项仅有地理对应证据，独立人工可辨识性与几何核验均为0；尚不具备真实跨视角监督训练条件。

自动初始构建11项（8异向/3同向，21采集）。直接查看原生图后排除3项当前ROI主要为散斑/均匀背景的候选；未换入容易样本、未删除数据、未按匹配器成功率筛选。`candidate_screening.json`保留理由，自动边界密度会把散斑误排前列，不能作为共同结构证明。其余8项只是更值得人眼查看，不自动确认静态物理对应。

## 采集与母区域

{chr(10).join(table)}

表中方位角差为产品元数据最小圆周差，不能等同完整三维视角差。升降轨、左右视/observation_direction、入射角、极化、许可和各侧真实身份/日期均保留在metadata及采集库存；未知项保留未知。表中重叠是Native轴对齐包围窗口的技术有效/地理交集比例；旋转窗口会包含ROI之外的额外区域，因此长滩右侧60.9%、鹿特丹左侧65.5%并不表示共同ROI只有这些比例。8项Common-FOV共同网格技术有效均100%，仍不证明同一散射体可见或稠密正确。

库存复用旧65采集对/58采集的元数据，不重新跑旧来源审计。22个上下文侧：{dict(context_counts)}。其中8侧复用既有未重采样1024上下文，14侧对已有同一官方资产读取新的有界2048原生窗口；未重复下载整景或获取新采集身份。上下文都大于512，Native是整数窗口原样切片，不插值冒充新增内容。当地UTM共同地面ROI、原产品→context→Native原点、CRS和像素中心(x+.5,y+.5)、两侧Native↔Common映射及数值往返误差完整保存。跨CRS不强行视作H。

Common-FOV512是线性DN/power双线性重采样；有效掩膜要求全部插值支持有效。固定P1/P99线性或dB显示独立按图计算，无平滑、无数据覆盖；dB/幅度转换不意味着绝对辐射定标。UmbraZero原8位产品已有量化损失，增强不能恢复。地理网格间距不等于产品真实分辨率。

## 快速判读与独立GT入口

8770服务已实际启动并只读检查返回HTTP200、8任务、reviewed=0。默认Native、不显示自动点；支持上下文、固定显示、独立亮度对比度、真实像素放大和同步平移。选择A清楚共同固定结构/B仅局部/C难确认/D待查，可简短备注，无点数要求，版本追加且旧版本哈希保护。

查看Common-FOV或覆盖图记录辅助暴露；真实几何核验应由未暴露人员在Native重点击。A/B仅是进入下一步的可读性判断，不自动成为C级GT。`prepare_gt.py`在没有真实A/B时不创建任何点或任务；后续只对实际通过项生成新的批次，在8771复用极简UI、独立第二人复核和原质量规则。变量Native尺寸经浏览器测试，不将旧512点自动搬迁为新独立证据。旧8769任务和历史记录未动。

## 统一算法和评价接口的实际测试

A仅图像；B可用地图粗定位/Common-FOV及moving原生±64px限窗。简化SAR结构法直接使用原生强度/幅度。共同后处理在原生坐标采用冻结Affine及full gate，输出4×4局部诊断与采样内点支持；局部支持不外推全图，置信度不声称校准概率。A/B改变输入方案，不能将差异全归于描述子。

选Kingston异向与SF同向两项，4方法×2协议，共16次真实新母区域调用，没有参数搜索：

{chr(10).join(srows)}

`accepted_unassessable`只表示预测侧门控通过，无独立正确性证据。Kingston未得到可靠接受；SF各法得到几何支持。B筛除很多Kingston候选只表示先验不一致减少，不能据此证明真实精度提高。

本轮实际CPU运行，16调用端到端累计{total_s:.2f}s，单次约{min(r['end_to_end_ms_synchronized'] for r in smoke):.1f}–{max(r['end_to_end_ms_synchronized'] for r in smoke):.1f}ms，峰值GPU分配0GiB。此计时含影像读取/匹配/全局与局部诊断，模型初始化计时因首次汇总失败未保留，不能伪造完整冷启动耗时。不是RTX4090吞吐测试；未为此安装/改动任何依赖。

另实际输出8项原始地理映射基线及候选采样，明确`coarse_geographic_relation_only`。152条旧SIFT/ORB/LightGlue/RoMa缓存引用只读取与本轮关联的记录/缓存身份，不新跑RoMa；旧512与V2不同FOV，不作虚假定量性能提升比较。原始RoMa模型身份继续可追溯。

`evaluation.py`单独读取合格check，预测侧没有人工参数/标注访问路径。未来在同一check集合对照原地理参考与模型预测，输出原生像素、产品地理网格推导的米距离、不确定度与支持覆盖。支持范围外不悄悄剔除后声称全幅成功。没有合格GT不报告真实Success@3/5/10px；可靠拒绝/错误接受需真实证据，未知不能作负例。

## 训练接口、隔离与数据缺口

`training_examples/normalized_index.jsonl`24条符合schema：1单图512已知二维几何示范（复用Phase1，seed26101201，源放大因子1/1）、11地理弱搜索、11困难未知、1跨站点明确地理无交集示范。**真实跨采集正例0、训练0**。被视觉筛除的ROI可用于单图合成/未知例，不能作为真实对应正例。未来局部块接口只接受合格原生GT的control点并保留check评价点，所有增强继承采集/组件，不虚增独立采集数量。

当前58采集/12站点依赖图合并共享采集、同站点与已有组件；12个组件全部已被多轮观察。当前V26组件只标`diagnostic_development_seen`，没有独立最终test，不能随机分裁片形成漂亮划分。

8项的产品DEM/高度假设没有可独立核实的充分资料。**8项都不能对高建筑/叠掩散射点声称二维地图精确对应**；这不意味着地面固定边界一定不可用。真正共见结构、建筑高度、阴影及船/集装箱动态需人工或其他独立证据。异向港口时间差112–198天，Melbourne约898天；短时差1.22天仅Kingston城市，未形成短间隔港口异向验证。

建议下一步先完成本轮轻量判读，优先SF对照及LongBeach0056、Rotterdam0070/0068。若港口仍无可确认静态结构，停止匹配阈值调整，补充：

- 8–12个新增港口/工业固定设施站点，至少3–5个从未观察的整站点留作确认；每站3–4真实日期/方向，总体24–48采集是采集设计目标，非已获取数量。
- 优先1–14天内异向对，兼顾20–60°中等和60–120°强方位差，同向控制；实际左右视/升降轨及入射角共同记录，不只看azimuth差。
- 高分辨率原生产品覆盖1–3km地面（按间距读取足够像素，不能固定2048就假定覆盖足够）；当前港口共同FOV136–318m易被局部堆场动态/散斑主导。优先固定岸线转角、防波堤、低矮永久码头和道路，不以船舶身份作跨时相正例。
- 保留原始动态范围、辐射单位/标定参数、NoData、真实空间分辨率和采样间距，避免只拿8位预览图。保留地理编码类型、实际DEM/高度、地理定位误差及可能的入射/叠掩说明；必要时独立地形/建筑高度或光学参考仅辅助地物身份，不当SAR散射中心GT。
- 官方公开数据候选继续是已有Umbra/Capella目录中的短间隔重复港口采集；S1 RTC适合同向大范围对照，其粗分辨率不能替代高分辨率异向标签。本轮没有执行额外大规模下载；目录是否存在符合条件的组须后续按实际记录核实。

## 验收、复现与失败记录

最终{unit['tests']}项测试通过，覆盖零值/NoData、原生窗口、跨CRS、像素中心/尺度、ROI、Native/Common值和掩膜、来源图隔离、冻结门控、预测禁止GT、未合格不计算Success、辅助暴露失资格、追加版本及训练规范/未合格不能导出正例。8候选已从冻结本地context实际重写Native/Common并与原输出逐像素/掩膜精确对比，无网络。真实Chromium{browser['browser']}测试通过全部8项加载、四种判读/版本、无自动合格、暴露记录、A/B任务转换、变尺寸点选Enter、隐藏原复核坐标和导出；所有操作位于`browser_SIMULATED_ONLY*`，不是人工GT。

保留首次失败日志：纳秒日期在Python3.10解析失败→仅日期计算截到微秒、身份原字符串不改；16推理已完成但汇总缺少jsonl导入→只恢复已有缓存不重复推理；合成API返回summary非meta→读取实际meta不重生成；标注服务模块重名→新服务改名；任务切换加载竞态→图像未完成时禁用保存。最终均验证。实际本地HTTP首次检查被外部代理返回502，禁用该检查的代理后本机HTTP200；不是数据访问失败。

26份直接保护的旧标注版本/任务记录与3份直接目录索引SHA保持不变；没有全面历史SHA重审计。V2源码、直接复用模块、模型selection/权重哈希、数据SHA与实际包版本分别冻结。复用Phase2 Git HEAD：`{head}`；新V2代码未提交，不能把旧Git HEAD冒充新代码版本，以`code_identity_{stamp}.json`补充完整文件身份。所有日志/报告使用真实当前Asia/Shanghai分钟时间戳；实验目录名保留创建时间而不是反复改名。

复现及实际网页命令见README。主要交付：`TECHNICAL_PROTOCOL.md`/`protocol.json`、`candidate_atlas.png`、`candidate_summary.csv`、`smoke/results.csv`、`raw_geography/`、`training_examples/normalized_index.jsonl`、`acceptance_{stamp}.json`。真实快速判读后再决定是否有必要建立少量check；当前不开始SAR域监督微调。
'''
    with (ROOT/'reports'/('phase_v2_report_'+stamp+'.md')).open('x') as f:f.write(report)
    with (ROOT/'logs'/('execution_'+stamp+'.log')).open('x') as f:
        f.write(f'Finalized {now.isoformat()}\nConda=match; no dependency installation; no training\n')
        f.write('Prior command stdout/stderr: see timestamped logs in this folder. Failed attempts retained.\n')
        f.write('build initial exit=1 (nanosecond date); retry exit=0, 11 regions\n')
        f.write('registration initial exit=1 after 16 completed calls (summary import); finalize-existing exit=0, inference reruns=0\n')
        f.write('training initial exit=1 after synthetic generated; resume exit=0, synthetic regeneration=0\n')
        f.write('browser initial/retry exit=1 (module collision, readiness race); final exit=0\n')
        f.write('delivery.py exit=0: 152 old comparator references, 8 geography outputs, protected26 unchanged\nnormalize_training_index.py exit=0: 24 normalized records\n')
        f.write('test_v2.py final exit=0: 14 tests passed; see final_tests_*.log\nreproduce_data.py --out reproduction_check_20261012_1340 exit=0: 8 regions exactly rebuilt, no network\n')
        f.write('live8770 startup successful; HTTP through external proxy failed502; direct local GET /api/tasks200=8, /api/progress200=reviewed0, /200\n')
        f.write('finalize.py acceptance assertions passed, config/direct catalog/protected history unchanged\n')
    print('FINAL REPORT',ROOT/'reports'/('phase_v2_report_'+stamp+'.md'));print('ACCEPTANCE',json.dumps(evidence,ensure_ascii=False))
if __name__=='__main__':main()
