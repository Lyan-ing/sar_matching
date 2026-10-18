"""Evidence-based handoff, immutable prior results; no fake real-world GT."""
import sys,json,hashlib,csv,subprocess,statistics
from pathlib import Path
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
from prepare import ROOT,SOURCE,sha,write
sys.path.insert(0,str(ROOT/'annotation_app'))
from store import Store
from quality import POLICY

def main():
    assert not (ROOT/'ENGINEERING_ACCEPTANCE.json').exists(),'Handoff already completed; use a new output for later manual review'
    tests=sorted((ROOT/'reports').glob('tests_verified_*.log'))[-1];testlog=tests.read_text();assert '18 passed' in testlog and 'coordinate_roundtrips' in testlog
    browser=json.loads((ROOT/'tests/browser_acceptance.json').read_text());assert browser['passed'] and not browser['console_errors'];assert browser['simulated_versions']==5
    browserlog=sorted((ROOT/'reports').glob('browser_final_*.log'))[-1];assert '"passed": true' in browserlog.read_text()
    store=Store(ROOT);assert len(store.tasks)==24;assert not list((ROOT/'annotations').rglob('v*.json'));assert not store.audit.exists()
    receipt=json.loads((SOURCE/'DELIVERY_COMPLETE.json').read_text());initial=json.loads((ROOT/'selection/input_acceptance.json').read_text());assert sha(SOURCE/'DELIVERY_COMPLETE.json')==initial['receipt_sha256']
    preserved=json.loads((SOURCE/'protected_artifacts.json').read_text());changed=[]
    for i,(p,h) in enumerate(preserved.items()):
        if not Path(p).exists() or sha(p)!=h:changed.append(p)
        if (i+1)%5000==0:print('Protected old files',i+1,'/',len(preserved),flush=True)
    assert not changed,changed
    for rel,h in receipt['file_sha256'].items():
        p=SOURCE/rel
        if not p.is_file() or sha(p)!=h:changed.append(str(p))
    assert not changed,changed
    preservation={'old_frozen_files':len(preserved),'previous_real_release_files':len(receipt['file_sha256']),'previous_real_receipt_unchanged':True,'mismatches':changed,'no_real_annotations_or_simulated_GT_written':True};write(ROOT/'reports/preservation_check.json',preservation)
    tasks=list(store.tasks.values());selected=json.loads((ROOT/'selection/selection_freeze.json').read_text());assert selected['sha256']==sha(ROOT/'selection/samples.jsonl')
    candidate=json.loads((ROOT/'candidate_matches/SUMMARY.json').read_text());assert len(candidate['records'])==24
    for t in tasks:assert sha(ROOT/'candidate_matches'/f"{t['sample_id']}.json")==t['candidate_sha256']
    prior_runner=sha(ROOT/'evaluation/runner_executed.py')
    metrics=[];executions={}
    for name in ['demo_native_gpu','demo_coarse_gpu']:
        p=ROOT/'evaluation'/name;provenance=json.loads((p/'protocol.json').read_text());assert provenance['runner_sha256']==prior_runner and provenance['cuda_available']
        r=[json.loads(s) for s in (p/'predictions.jsonl').read_text().splitlines()];assert len(r)==20 and all(not x.get('error') for x in r);assert not json.loads((p/'evaluation/evaluation.json').read_text())['formal_accuracy_computed']
        executions[name]={'runner_sha256':prior_runner,'predictions_sha256':sha(p/'predictions.jsonl'),'frozen_selection_sha256':provenance['selection_sha256'],'model_freeze':provenance['model'],'init':provenance['initialization'],'formal_GT_denominator':0}
        for m in ['sift','orb','lightglue','orb_then_sift']:
            rows=[x for x in r if x['method']==m];times=[x['end_to_end_ms'] for x in rows];metrics.append({'protocol':provenance['protocol'],'method':m,'pairs':5,'geometry_outputs':sum(x['output_available'] for x in rows),'gate_accepts':sum(x['accepted'] for x in rows),'mean_ms':statistics.mean(times),'median_ms':statistics.median(times),'fallbacks':sum(x['fallback_used'] for x in rows),'formal_GT_denominator':0,'Success@3/5/10px':None})
    with (ROOT/'evaluation/demo_process_statistics.csv').open('w',newline='') as f:w=csv.DictWriter(f,list(metrics[0]));w.writeheader();w.writerows(metrics)
    # Native prediction interface was re-evaluated without rerunning the algorithm,
    # after adding supported-threshold summaries. Evidence denominator stays zero.
    a=json.loads((ROOT/'evaluation/native_interface_acceptance/evaluation.json').read_text());assert a['gt_qualified_prediction_rows']==0
    assert all('error_mean_px' not in r for r in a['records'])
    dependencies=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True);(ROOT/'reports/match_pip_freeze.txt').write_text(dependencies)
    current=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True));original=json.loads((ROOT/'evaluation/demo_native_gpu/protocol.json').read_text())['dependencies'];assert current==original,'match environment changed during engineering checks'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,driver_version','--format=csv,noheader'],text=True)
    now=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=now.strftime('%Y%m%d_%H%M');write(ROOT/'gt_quality/policy_locked.json',POLICY)
    failures=[
      {'stage':'initial selection','reason':'mixed UTC fractional precision caused naive/aware timedelta TypeError','resolution':'interval compares UTC whole seconds; original complete acquisition timestamps retained; selection frozen only after successful run'},
      {'stage':'first HTTP unit test','reason':'sandbox blocked local socket','resolution':'authorized localhost tests executed outside sandbox'},
      {'stage':'HTTP retry','reason':'server outbound proxy routed localhost and returned502','resolution':'test client uses ProxyHandler({}) only for loopback; no system proxy change'},
      {'stage':'first inference demo','reason':'GPU invisible in restricted sandbox; ran on CPU','resolution':'CPU evidence retained in demo_native; same methods additionally exercised with actual CUDA in demo_native_gpu; CPU times excluded from primary GPU table'},
      {'stage':'existing Firefox check','reason':'Snap profile/runtime paths unwritable; direct binary host GLIBC/GLIBCXX mismatch','resolution':'used temporary Playwright+Chromium, no runtime web framework or environment rebuild'},
      {'stage':'browser bootstrap','reason':'inherited NODE_TLS_REJECT_UNAUTHORIZED=0 detected','resolution':'interrupted initial download; final browser freshly downloaded under different /tmp directory with NODE_TLS_REJECT_UNAUTHORIZED=1; Chromium sandbox enabled'},
      {'stage':'initial browser interaction','reason':'automation click after page scroll missed offscreen canvas','resolution':'automation scrolls canvas into view before mapping; real UI interaction and coordinates retested'},
      {'stage':'added simulated evaluation test','reason':'simulation fixture deliberately had no GeoTIFF but requested optional map metric','resolution':'simulation explicitly mocks map metric=None; actual sensor geocoding separately tested'},
      {'stage':'handoff Git provenance','reason':'workspace parent is not itself a Git repository','resolution':'record existing Phase2 project Git commit and every new source file SHA separately; do not claim new code is committed'}]
    write(ROOT/'reports/stage_failures.json',failures)
    files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.py','.js','.html']]
    provenance={'completed_at':now.isoformat(),'python':sys.executable,'frozen_Phase2_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT.parent/'MaritimeSARRegistrationPhase2',text=True).strip(),'new_code_is_committed':False,'new_code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in files},'actual_demo_runner_sha256':prior_runner,'note':'runner_executed.py reconstructs exact measured-runner bytes and matches all recorded SHA; run.py adds evaluation aggregation only after demo inference; frozen sarbench untouched','gpu':gpu,'selection_sha256':selected['sha256'],'task_sha256':sha(ROOT/'annotation_tasks/tasks.jsonl'),'match_dependencies_unchanged':True,'test_tool_only_tmp':{'playwright':'1.58.0','chromium':browser['browser_version'],'pyee':'13.0.1','greenlet':'3.5.6','typing_extensions':'4.16.0'},'executions':executions}
    write(ROOT/'reports'/f'reproducibility_{stamp}.json',provenance)
    providers=Counter(t['sensor'] for t in tasks);readiness=Counter(t['annotation_readiness'] for t in tasks);looks=Counter(t['look_pair_category'] for t in tasks)
    table='\n'.join(f"|{r['protocol']}|{r['method']}|{r['geometry_outputs']}/5|{r['gate_accepts']}/5|{r['mean_ms']:.2f}|{r['median_ms']:.2f}|{r['fallbacks']}|" for r in metrics)
    blocked='\n'.join(f"- `{t['sample_id']}`：{t['visual_note']}" for t in tasks if t['annotation_readiness']=='possibly_not_evaluable')
    report=f'''# 真实多视角 SAR 几何对应点工程验收

完成时间：{now.isoformat()}（Asia/Shanghai）。本轮工程验收完成，停止于人工独立判断之前。
真实人工对应点 **0**，`gt_qualified` **0**，真实配准精度分母 **0**。没有声称自动匹配或地理网格为GT。

## 数据与冻结选择

来源目录 `{SOURCE}`；程序核验最终191条索引、764次原生PNG/GeoTIFF文件SHA检查、各传感器512尺寸/CRS，沿用已验收工程QA，不下载、不重裁、不重复来源审计。
新工作目录 `{ROOT}`。selection/samples.jsonl为24裁片/24不同真实采集对、12站点；来源数量 `{dict(providers)}`，观测类别 `{dict(looks)}`。
seed261011，逐站点2采集对；旧金山/南达科他各仅1可用，Johnston与长滩各增加1。按观测方向类别、地理重叠覆盖及seededSHA排序选样，匹配实验之前冻结。每采集对仅1裁片，不把同源裁片当独立观测。
完整采集ID、UTC、轨道、观测方向/角度、原生间距、提供方、许可、来源组件、原索引路径及共同地理范围均随任务保留。未知角度保持未知，日期差不等于观测角差。
图像实际内容分级在selection/visual_triage.json，不把站点名冒充逐图地物标签。visualization/selection_overview.png含全部24对；annotation_app_demo.png与demo_task_1..5.png为实际浏览器截图。

## 可进入标注与待判断样本

{readiness['ready_for_human']}对可优先人工尝试，{readiness['difficult']}对困难但保留，{readiness['possibly_not_evaluable']}对可能缺乏可靠共同静态结构。
全部24对均配置任务；**最终人工排除0对**，候选不可评价与最终不可评价严格区分。
待人工判断的6对：

{blocked}

首批顺序：`S1SF_EXTENSION_000__chip_000`、`Capella_0003__chip_001`、`Umbra_0061__chip_001`、`Umbra_0014__chip_000`、`Umbra_0041__chip_001`。
初步目标每可辨认对8～12分散静态对应，提前区分control/check。估计首批标注2.5～5人时、独立复核1.25～2.5人时，总约4～8人时；这是规划估计，非实测。异向矿区/低重叠农田可能耗时更长或最终跳过。

## 标注工具与质量控制

标准库ThreadingHTTPServer与原生Canvas，无新Web框架/数据库。回环监听、SSH转发、同源/Host与CSRF检查，原生PNG白名单路径。默认独立模式隐藏算法候选及矩阵，左右点击、亚像素输入、缩放/平移、十字放大、对比度/亮度、点增删改/草稿撤销、保存重载、前后切换均实现。
辅助候选仅存candidate_matches，每文件SHA锁定。24对共有SIFT候选{sum(r['sift_candidates'] for r in candidate['records'])}、地理候选{sum(r['geographic_candidates'] for r in candidate['records'])}，候选生成失败{sum(r['failure'] is not None for r in candidate['records'])}。候选不得自动填入GT；主动打开辅助留人员、时间、来源及SHA，资格中排除辅助来源证据。
点记录双视图坐标、地物、静态、可见性、置信、两图不确定性、人员/时间、独立复核重新点击与结论/备注。复核隐藏原标注位置，要求不同人员；保存原子发布、只追加、父SHA链、旧版本拒绝覆盖、并发旧revision拒绝。修改位置/语义失效复核。人员身份为声明式，需要真人真实分工，不能由软件证明独立人格。
质量规则在gt_quality/policy_locked.json，状态从证据自动计算，非法客户端升级被拦截。检查越界、重复ID/坐标、双视图缺失、分布集中、静态/可见未知、辅助依赖、自审/未审、定位差异、不确定性与邻域冲突。Affine控制点拟合/独立检查点残差只作全局模型适用性信号，不能由同组RANSAC误差制造GT。
局部资格最少3分散可信检查点，全幅稀疏另需6点、各图面积与跨度要求；Success阈值至少4检查点且联合定位筛查界限不超过阈值/3。界限源于人工定位估计与真实复核差，未校准为统计置信区间，仍须科研复核其实际可支持精度。山地/异向允许局部GT和全局模型不适用，不强行拟合一个H。

## 冻结方法流程演示（不是真实精度）

Phase2.5 selection、配置、推理代码、LightGlue源码提交/diff、官方两个权重SHA在每次运行前核对。沿用SIFTcontrast0.01、ORBedge16/fast12、SuperPointthreshold1e-4、2048特征、ratio0.8、Affine/full和seed42。
纯图像协议仅接收reference/moving原生PNG，不读取GT、Mask或地理变换。粗定位辅助协议明确使用既有moving粗网格PNG，预测后评价经地理转换回moving原生像素；与纯图像实验分别统计。
以下仅报告5对首批任务的几何矩阵输出与门控，以及GPU解码至决策耗时。几何输出包括被门控拒绝的candidate_H；接受不证明正确。初始化/保存/GT加载排除，冷调用已标记，样本少且未warmup，不能作为方法优劣或正式性能结论。

|协议|方法|几何输出|门控接受|平均ms|中位ms|实际回退对数|
|---|---|---:|---:|---:|---:|---:|
{table}

两协议各20行逐样本记录，实际执行回退，不以单项耗时求和估算。接受的均是首个旧金山任务，其他任务保留too_few_matches/few_inliers/low_inlier_ratio等原因。这是流程诊断，**不排除失败样本，不调参数、不计算Success@px**。
沙箱首次不可见GPU的CPU演示保留在evaluation/demo_native，GPU演示为demo_native_gpu与demo_coarse_gpu，主表仅后两者；不可混合两硬件耗时。
{gpu.strip()}

评价接口已经模拟点闭环验证：无GT不得计算正式误差；合格GT仅用独立check而非control，记录均值/中位/p90/p95、逐点误差、GT版本SHA、分阈值成功及错误接受/拒绝分母。无输出、拒绝、未核验接受、受支持阈值下错误接受/稀疏成功分别处理。
按提供方、实际内容标签、观测方向、来源组件输出分层，附真实采集对数量。支持稀疏真实点误差不能冒充旧512密集点全图评价。米单位投影CRS可输出地理网格条件距离（保留旋转项），经纬度网格无可靠地面度量时不输出米误差，任何地理粗网格均非独立绝对精度GT。

## 自动化验收与保护

18项pytest通过；180次浏览器同源JS坐标往返检查、CSS尺度/zoom/pan/像素中心检查通过。真实Chromium {browser['browser_version']}的{len(browser['checks'])}类交互验收通过，控制台错误0：真实24任务加载、5首批原生图显示、鼠标坐标映射、增删/撤销、保存重载/版本、独立复核、修改复核失效、辅助留痕与切换等。
模拟点只在/tmp的SIMULATED测试标识下；真实annotations没有点版本、没有辅助暴露事件。历史{len(preserved):,}个冻结文件与上轮真实交付{len(receipt['file_sha256']):,}文件逐一SHA未变，交付回执SHA也未变。
Python为 `{sys.executable}`，match依赖清单与GPU运行前一致；base、旧venv/rfrock未使用。Playwright及浏览器只置/tmp测试，开启TLS验证与Chromium沙箱，不是标注服务依赖。
问题与修复保存在reports/stage_failures.json，包含时间解析、localhost沙箱/代理、CPU/GPU证据区分、Firefox兼容、浏览器自动滚动测试与模拟fixture的可选地理信息。
新增代码位于本目录；未改历史生成器、冻结推理核心、旧数据与报告。代码SHA及实际演示runner（{prior_runner}）在复现记录中。评价汇总接口的后续完善独立于实际冻结预测，runner_executed.py逐字SHA吻合原实际运行版本。

## 下一步及限制

停止自动推进，等待真人静态结构判断及第二人独立重新定位。启动/SSH/浏览器/标注与复核/导出及评价命令详见根README.md。
已有confirmation参与工程QA与预览，不称为封存无偏测试；站点/采集组件标签继续追溯，未来训练或调参必须另定隔离方案。旧合成源采集关联不完整的风险沿用，不能证明不存在未知关联。
本轮没有训练、超参搜索、SAR-SIFT替代实现、扩大合成数据或正式真实精度结论。自动候选即使显示可信也不升级为GT。
请完成首批并通知“首批标注和独立复核已完成”，提供样本ID/人员/不可评价原因；后续以冻结预测和新人工证据继续核验，输出新目录，不覆盖既有结果。
'''
    reportpath=ROOT/'reports'/f'geometry_gt_report_{stamp}.md';reportpath.write_text(report)
    logpath=ROOT/'reports'/f'execution_{stamp}.log'
    log='This execution log consolidates actual commands and saved evidence; not fabricated live timestamps.\n'
    log+=f'completed_at={now.isoformat()}\npython={sys.executable}\n'
    log+='commands: prepare.py; setup_tasks.py; evaluation/run.py predict --protocol native --out evaluation/demo_native (CPU); predict --protocol native --out evaluation/demo_native_gpu; predict --protocol coarse_geo --out evaluation/demo_coarse_gpu; evaluate --predictions evaluation/demo_native_gpu/predictions.jsonl --out evaluation/native_interface_acceptance; gt_quality/export_status.py --out gt_quality/initial_status; tests/browser_smoke.py; finalize.py\n'
    log+='selection and input acceptance: '+json.dumps(initial)+'\n'+json.dumps(selected)+'\n'
    log+='failures_and_resolutions:\n'+json.dumps(failures,ensure_ascii=False,indent=2)+'\n'
    log+='FINAL_UNIT_TEST_RAW_OUTPUT:\n'+testlog+'\nFINAL_BROWSER_RAW_OUTPUT:\n'+browserlog.read_text()+'\n'
    log+='FINAL_PRESERVATION_AND_GT_CHECK:\n'+json.dumps(preservation)+'\nreal_GT_points=0; gt_qualified=0; formal_accuracy_denominator=0\n'
    logpath.write_text(log)
    acceptance={'completed_at':now.isoformat(),'engineering_accepted':True,'real_independent_GT_accepted':False,'awaiting_human_annotation':True,'selected_pairs':24,'sites':12,'providers':dict(providers),'first_tasks':[t['sample_id'] for t in tasks[:5]],'readiness':dict(readiness),'final_human_excluded':0,'real_GT_points':0,'gt_qualified':0,'formal_accuracy_denominator':0,'unit_tests_passed':18,'coordinate_cases_passed':180,'browser_checks_passed':len(browser['checks']),'preservation':preservation,'report':str(reportpath),'execution_log':str(logpath),'guide':str(ROOT/'README.md'),'file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}}
    write(ROOT/'ENGINEERING_ACCEPTANCE.json',acceptance);print(json.dumps({k:v for k,v in acceptance.items() if k!='file_sha256'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
