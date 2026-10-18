"""Write the evidence-linked pilot report and completion-minute execution log."""
from core import *
from datetime import datetime
from zoneinfo import ZoneInfo
import subprocess,re
def main():
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    records=json.loads((ROOT/'analysis/records.json').read_text());summary=json.loads((ROOT/'analysis/summary.json').read_text())
    acceptance=sorted(ROOT.glob('ACCEPTANCE_*.json'))[-1];assert json.loads(acceptance.read_text())['status'].startswith('passed')
    # Stage filenames reflect actual last-output minute; the consolidated log and
    # report use this finalization minute. Preserve all contents and failures.
    renamed={}
    for p in sorted((ROOT/'logs').glob('*.log')):
        prefix=re.sub(r'_\d{8}_\d{4}$','',p.stem)
        actual=datetime.fromtimestamp(p.stat().st_mtime,ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
        dest=p.with_name(prefix+'_'+actual+'.log')
        if p!=dest:
            if dest.exists():raise FileExistsError(dest)
            p.rename(dest)
        renamed[p.name]=dest.name
    stages=[('A','run.py A','route_A',0),('baseline','run.py baseline','baseline_postprocessing',0),
        ('B','guarded_run.py B','route_B',0),('tests','-m pytest -q test_core.py','tests_final',0),
        ('analysis first','analyze_visualize.py','analysis',1),('analysis finish','analyze_visualize.py --finish','analysis_finish',0),
        ('acceptance','verify_actual.py','acceptance',0)]
    journal=[]
    for stage,cmd,prefix,status in stages:
        logs=sorted((ROOT/'logs').glob(prefix+'_*.log'))
        if prefix=='analysis':logs=[p for p in logs if not p.stem.startswith('analysis_finish')]
        journal.append(dict(stage=stage,command='conda run --no-capture-output -n match python '+str(ROOT)+'/'+cmd if not cmd.startswith('-m') else 'conda run --no-capture-output -n match python -m pytest -q '+str(ROOT/'test_core.py'),exit_status=status,log=str(logs[-1])))
    failures=[
        '首次官方权重流式下载静默截断：初始工具退出0，但两份文件 torch.load 均失败，未用于推理；保留 weights/incomplete_initial。',
        '首次续传入口遇 identity.json 已存在，退出1并拒绝覆盖；原记录移存后重试。',
        '分段续传首轮工具退出0但 identity 明确报告 DINO truncated chunk；53段中1段不完整。重试仅该段成功；最终两份官方权重完整。',
        '试图取消已结束下载时 /proc 文件不存在，退出1；没有终止其他任务。',
        '环境查询 pyproj、scipy 元数据分别退出1：二者未安装且当前推理不需要，未为此修改环境。',
        '绘图首轮退出1：certainty 实际形状(1,864,1728)，imshow要求2D；仅 squeeze 显示batch轴，保留100张已生成图，补4张热图及索引，退出0。',
        '无实际匹配推理崩溃、超时或OOM；算法无匹配/不可靠输出记录在 failure_stages.csv，未隐藏。']
    save(ROOT/('execution_status_'+stamp+'.json'),dict(commands=journal,failures=failures,log_filename_changes=renamed,
        source_download_exit=0,local_dependency_install_exit=0,prepare_exit=0,initial_transport_exit0_does_not_imply_valid_model=True))
    log=['完成时间 '+stamp+' Asia/Shanghai','新根目录 '+str(ROOT),'全部 Python 执行使用 match。']
    for entry in journal:log.append(json.dumps(entry,ensure_ascii=False))
    log+=failures
    for p in sorted((ROOT/'logs').glob('*.log')):log+=['\n--- '+p.name+' ---',p.read_text()]
    (ROOT/('execution_'+stamp+'.log')).write_text('\n'.join(log))
    def number(v):return '—' if v is None else f'{v:.2f}'
    table=['|样本|方法/协议|匹配/内点|全局门控|局部通过格数|地图内点中位差px（非GT）|阶段计时ms|',
        '|---|---|---:|---|---:|---:|---:|']
    order={'Capella_0003__chip_001':0,'Umbra_0014__chip_002':1,'Umbra_0061__chip_001':2,'S1SF_EXTENSION_000__chip_002':3}
    for r in sorted(records,key=lambda r:(order[r['sample_id']],r['method'],r['protocol'])):
        table.append(f"|{r['sample_id']}|{r['method']}/{r['protocol']}|{r['n_matches']}/{r['n_inliers']}|{'通过（未核验）' if r['accepted'] else r['reason']}|{r['local_accepted_tiles']}|{number(r['geo_inlier_median_px'])}|{number(r['end_to_end_ms'])}|")
    grouped=['|方法/协议|有匹配对|匹配空间支持对|可估计Affine对|全局门控对|局部支持对/格数|无模型/拒绝|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        if s['look']=='all':grouped.append(f"|{s['method']}/{s['protocol']}|{s['with_matches']}/4|{s['spatial_supported']}/4|{s['affine_outputs']}/4|{s['full_gate']}/4|{s['local_supported_pairs']}/{s['local_supported_tiles']}|{s['no_output']}/{s['rejected']}|")
    versions=json.loads((ROOT/'environment_before_B.json').read_text())['packages']
    identity=dict(phase2_git=subprocess.check_output(['git','-C',str(BASE/'MaritimeSARRegistrationPhase2'),'rev-parse','HEAD'],text=True).strip(),
        new_code_sha256={p.name:sha(p) for p in sorted(ROOT.glob('*.py'))},
        official_archive=json.loads((ROOT/'vendor/source_identity.json').read_text()),
        local_dependencies=dict(einops='0.8.1',loguru='0.7.3'),
        licenses={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'vendor/RoMa/LICENSE',ROOT/'vendor/DINOv2_LICENSE']},
        old_code_modified=False,new_code_has_separate_git_commit=False)
    save(ROOT/'code_identity.json',identity)
    report=f'''# 真实SAR固定四对匹配 Pilot 实际实验报告

完成：{stamp}，Asia/Shanghai。根目录：`{ROOT}`。这是已经查看过的四对探索性诊断，不是未见独立确认集。没有训练、微调、旋转/尺度/参数搜索；未替换样本；鹿特丹不参与。所有历史预测、伪标签、人工任务与数据保持冻结。

**结论：当前两对真实异向SAR上，简化结构描述子没有得到候选；冻结RoMa显著增加候选数量，但未增加通过全局或局部可靠性门控的样本。RoMa在相近方向港口控制上新增稳定全局Affine支持，不能外推为异向SAR能力改善。当前C=0，没有独立证据证明真实正确性提高，不报告真实Success@3/5/10px，也不自动升级任何证据等级。**

## 实际执行与完整结果

A已完成4对×2版；B已完成4对官方网络推理，B1/B2共享同一密集输出与2048抽样池；额外3次预定计时仅在固定顺序中的第一同向控制（长滩港）完成。16个主版本输出、3个计时重复，未为效果搜索额外模型。旧SIFT/ORB/LightGlue仅读取24份固定缓存并重新做统一原生坐标诊断，没有再推理191对。

[逐方法CSV](analysis/per_sample.csv)、[分组CSV](analysis/summary.csv)、[640条局部区域CSV](analysis/local_geometry.csv)、[匹配阶段失败CSV](analysis/failure_stages.csv)、[完整诊断JSON](analysis/records.json)、[图集入口](visualizations/index.html)。每条预测目录均含 `record.json`（候选H、门控、时间）、`matches.npz`（原生点位、置信、有效性、地图差、内点）和 `local.json`（16格模型及索引）。没有模型、模型被拒绝、工程接受但独立精度不可评价三种状态分别保留；本轮不能进一步判定“错误接受/真实成功”。

{chr(10).join(grouped)}

匹配空间支持在两图要求凸包≥0.01、各向同性≥0.005，仅描述候选分布，不替代内点支持或正确性。下表所有地图差均是诊断代理，不是定位误差GT；局部通过格数也不是“正确区域”。

{chr(10).join(table)}

## 路线A及A1/A2受控关系

这是参考SAR-SIFT/RIFT/CFOG思路自编的简化基线，**不是论文算法完整复现**。原生float栅格、3px半窗分离矩形均值比值log梯度、8个有向bin、48px窗口、4×4空间直方图、L2-Hys。固定16×16网格，首尾边界缺完整48窗口而剔除；各视图实际196个描述子。无旋转归一或尺度归一，网格间距32px，不能把网格节点对应解释成亚像素精确定位。Sentinel-1线性power开平方，其余使用既有linear_amplitude声明；Umbra源DN仍有8bit量化，未声称跨传感器物理辐射定标等价。

技术有效mask、finite和GeoTIFF read-mask用于排除NoData；有效0保留。描述窗口及梯度支持必须完整有效。四对原生栅格本次技术有效率均1.0，真实有效0占比0；有0/NoData情形通过独立构造数组测试。详细原始范围、坐标系统及单位见 [actual_input_metadata.json](actual_input_metadata.json)。

A1/A2复用同一份点位与描述子；双向ratio0.8、精确互检、3px节点循环一致性。精确互检下节点循环为0，是索引一致性，不是独立几何验证。A2只把可候选距离矩阵限定为PixelMap预测周围moving原生坐标两轴各±64px；这是方形L∞窗，非半径64px圆窗。地图只限搜索，没有加入地图伪对应点参与拟合。

两对异向A1/A2均0匹配；长滩港A1为0、A2为1，不足估计Affine；S1为20/24匹配且均为内点，全局通过。S1 A2比A1增加4个且局部支持1格，但源地图网格一致及节点同索引并不能独立证明定位正确。详情记录了窗口无候选、少于两个候选、双向ratio和互检各级数量；没有为了输出而放宽ratio。

## RoMa真实部署与B1/B2

官方源码固定 [Parskatt/RoMa 77f8d688](https://github.com/Parskatt/RoMa/tree/77f8d68803526dcddfd9b7a46bc76125bdc25f15)，官方outdoor模型；batch1、官方默认粗560×560/升采样864×864、默认对称输出、threshold_balanced抽样、sample_thresh0.05、max2048、seed42。输入为冻结原生512灰度PNG，由官方程序转RGB和标准归一化，不做本轮新辐射表示。源码默认Linux自定义correlation扩展未安装，使用**源码正式支持的 `use_custom_corr=False` 原生PyTorch后端**，无模型结构或权重替换。

官方输出helper采用像素角点基准，本项目512像素中心0..511，因此转换后统一减0.5；已对官方helper、align_corners=False及真实保存数据校验。B2是B1精确采样池的确定性±64px两轴限窗子集，不重新抽样，不重新推理。密集输出及原始certainty保存到 `route_B/*_dense/dense.npz`；官方抽样可能将certainty阈值化为1，因此同时保存阈值化值和原始值，本报告与图使用原始值。两者都不是校准正确率。

官方权重身份（严格state加载，非随机/第三方）：

- [roma_outdoor.pth官方地址](https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth)：445647516字节，SHA256 `c7a45c80d41ad788a63c641d1b686d7cb3f297f40097c6f4e75039889e5cc8ba`。
- [DINOv2 ViT-L/14官方地址](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth)：1217586395字节，SHA256 `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428`。

[权重记录](weights/identity.json)、[真实模型身份](route_B/model_identity.json)、[推理前B冻结记录](B_run_freeze.json)。SHA是本次官方下载实际字节指纹，非独立发布方签名认证。RoMa源码README明确代码除DINOv2为MIT，DINOv2为Apache2.0；本地保留 [MIT](vendor/RoMa/LICENSE) 与 [DINOv2 Apache2](vendor/DINOv2_LICENSE)。RoMa权重下载资产未提供独立的权重许可证文本，不能把代码许可陈述扩大为已核实所有训练数据/权重再分发权。本轮科研推理使用官方公开下载权重，未另行发布权重。

## 异向失败与同向控制

Capella B1：2048候选/109内点，内点率5.32%，全局拒绝；双向独立Affine循环均值267.94px，内点地图差中位296.46px。可视化中拟合簇虽能形成线性关联，但跨越地理粗位置，不能认定为同一地物。B2只留1个（去除2047/2048，99.95%），不能拟合；这是删除粗位置不一致候选，**不是识别真实正确点的证明**。旧native LightGlue27/5，同样拒绝；A无输出。没有任何方法在该异向对通过局部完整门控。

矿区 B1：2048/133，内点率6.49%；reference/moving内点凸包分别约0.2121/0.00283，明显多对一压缩；对称自拟合RMSE45.39px、双向循环309.00px。B2只留171（去除91.65%），26内点，内点率15.20%，reference凸包0.00279、循环194.96px。地图差中位下降和自拟合残差下降没有消除集中支持和全幅不稳定。图中的局部模型也被拒绝（包括shape/support/cycle问题）。旧LG13/4，A无输出；两条路线均没有新增可靠全局/局部支持。

长滩港 B1：2048/1128，55.08%内点，全局通过，8格局部通过；B2：2015/1176，58.36%，同样8格。B2内点数上升是**对同一池重新RANSAC的共识模型/内点标签变化**，不是新生成匹配。旧nativeLG451/68因低内点率全局拒绝，但同一局部协议已有8格支持；因此RoMa主要新增全局工程稳定性和更密支持，并未在该样本增加局部通过格数。预测约25px平移，相对地图内点差仍约25.95px；不能以地图近似否定或确认真实位移。50%叠加及96px原生局部窗供诊断，不构成人工静态点确认。此前线性表示B候选及旧证据等级没有修改。

S1 B1/B2：2048/1152，56.25%，全局通过、9格局部通过，B2未删除任何点。旧SIFT、ORB、LG以及A已全局通过；RoMa增加密度和局部支持数，未证明真实精度优于旧方法。地图差约0.59px仅说明与已地理编码网格接近。海面弱纹理主要仍在拟合外，不能用海陆大面积背景证明稠密正确性。

两对异向上所有方法全局/局部通过均0。候选数量不是可比的统一检测能力指标：A是最多256网格候选，RoMa是密集网络抽样2048，旧方法为其冻结检测器；跨路线是完整方案比较，不能把差异直接归因于描述子。

## 共同几何、有效区域与证据隔离

全部reference→moving原生512坐标，共用旧 `estimate(model=affine,seed=42)` 与 `gate(full)`。RANSAC3px、最大5000、置信0.995；≥8匹配/内点、内点率≥0.3、grid覆盖≥0.0625、自拟合RMSE≤3。full继续检查投影有限性、角点形状/尺度、两图内点凸包≥0.01、各向同性≥0.005、独立反向拟合循环均值≤10px、对称残差≤3px；代码及selection输入SHA锁定。拒绝时仍保存候选H和内点，不把失败隐藏。

局部4×4以reference原生128px格取点，仍在native512坐标使用相同full门控，不将小格归一化放大支持面积、不给A/B区别门控。局部通过只支持该格，不能升级全幅。此规则偏严格，小格/稀疏A容易因空间支持不足被拒绝，零局部通过不意味着一定没有真实局部对应。

全局/局部RANSAC使用所有原始候选；技术有效性单独QA保存，**没有在看结果后追加有效mask筛选或改变接受逻辑**。RoMa边缘少量坐标在技术原生0..511外（例如港口B1有效2042/2048，内点有效1123/1128），不是实际数据NoData；边缘计数在record/matches保留。图中的拟合支持标签不是有效性或正确性保证。旧geo_assisted预测先转换原生再重新统一诊断；原来粗坐标门控结论/H另外保存，历史结果不回写。

预测接口只读图像/技术有效性/必要地理元数据及冻结模型，未读取人工check/control/复核文档。B另用Python文件访问guard禁止 GeometryGT 与 IndependentBenchmark 路径，实测阻挡事件0；底层GDAL读取路径由已检查的load_pair限制。这是工程边界证据，非操作系统级形式证明。未来 [evaluate_sparse.py](evaluate_sparse.py) 仅独立读取真实导出与冻结H，要求gt_qualified，只用check评分、不拟合或筛选预测；局部输出不会假装全幅模型参与Success。

## 时间、资源与环境

A阶段计时每版约0.24–0.44秒、CPU线程1、GPU0；A1/A2共享特征提取，表中每版计时将共享提取成本各记一次，用于单方案成本而非总作业求和。B粗推理+读取+抽样首对3.603秒，暖对0.776–0.800秒；主版本加几何后的阶段和约0.81–3.71秒。模型一次初始化12.739秒另记；峰值分配显存4357.71MiB（4.256GiB），远低于20GiB。

**计时边界限制：主表沿用字段名 `end_to_end_ms`，实际为各已测阶段之和，包含读图、特征/推理、匹配/限窗与全局+局部几何，排除主实验密集缓存字典构建、CPU结果存档/图表、模型一次初始化。这不是含全部输出落盘的作业延迟。** 旧算法时间是历史缓存记录、显存未知，不能称本轮同进程公平测速。新B实际暖重复的直接墙钟（读图→推理/抽样/转换→同一全局+局部诊断，无缓存写盘）分别873.18、851.02、855.74ms，中位855.74ms，坐标与主预测逐元素一致；见 [timing_repeats.json](route_B/timing_repeats.json)。这给出真实算法链计时，没有分项相加估算冒充直接墙钟。下载时间不算GPU推理；正式推理和重复均在120秒/对、CPU/GPU两小时预算内，无OOM/超时。

Python3.10，`match`；torch {versions['torch']}、torchvision {versions['torchvision']}、numpy {versions['numpy']}、opencv-python-headless {versions['opencv-python-headless']}、rasterio {versions['rasterio']}。CUDA构建12.1、实际RTX4090。仅本轮 `vendor/deps` 安装einops0.8.1和loguru0.7.3（--target --no-deps），推理不需要项目pyproject列出的训练依赖集合。没有安装第二个OpenCV发行版；完整版本见 [match_pip_freeze.txt](match_pip_freeze.txt)，整个match包集合前后相同。

## 验收、失败与可复现记录

9项pytest全部通过（原生power/幅度/valid0/NoData；确定性描述子；跨CRS与半像素；±64边界；双向ratio/互检/循环；官方RoMa坐标；全局/局部几何；预测接口隔离；未来GT资格与不拟合）。实际验收通过16新版本、24旧对照、640局部记录、104图SHA、B2精确子集、原始certainty映射与坐标、三次重复、完整权重及环境；见 [{acceptance.name}]({acceptance.name})。55个直接输入/旧冻结记录SHA前后相同，未执行大规模旧文件审计。新目录外没有旧输出写入。新代码身份见 [code_identity.json](code_identity.json)，复用Phase2 Git `{identity['phase2_git']}`；本轮独立目录没有虚构新commit，以代码SHA固定实际实现。初始A冻结后只在B未推理时补齐预定计时重复；core/config/samples未改，B前另固定runner SHA。

全部失败保留在 [execution_{stamp}.log](execution_{stamp}.log) 和 [execution_status_{stamp}.json](execution_status_{stamp}.json)。首次网络静默截断未检查HTTP总长度，是下载程序的实际问题；随后改成校验总长度、每段206/Content-Range/字节数及整文件结构，单段重试后成功。截断权重从未加载推理。绘图batch轴问题仅修显示，不改预测。两次不需要的依赖查询失败未引发安装。真实算法失败/拒绝逐样本保留，未选择最容易图像输出。

图集含每对旧LG native/geo_assisted、A1/A2、B1/B2的内点/拒绝点分图、全点覆盖、方法与地图区别、96px原生局部放大、全局及局部叠加；另有4张运行时间图和4张原始certainty图。匹配线每类最多60条，确定性等间隔索引选取而非按好看筛选；覆盖图保留全部点。局部叠加显示支持最多的候选格（优先门控通过，非GT最佳），所有16格模型仍保存。已实际查看Capella/Mine失败、港口局部与叠加、S1覆盖，颜色不代表真实正确性。

## 复现与人工下一步

不要在本次已冻结目录再次执行生成。使用新兄弟目录（自动复用官方源码/权重和本地依赖，只读；已有目录会拒绝覆盖）：

```bash
cd /home/yl/matching/sar_matching
conda run --no-capture-output -n match python {ROOT.name}/reproduce.py --out /home/yl/matching/sar_matching/RealSAR_MatchingPilot_reproduce_新时间戳
```

该命令用当前match同一Python依次prepare、pytest、A、B、旧baseline后处理与图集；不下载旧数据，不安装/升级环境，不使用人工标注。每步日志/退出状态写新目录。源码、依赖和模型当前已可运行；本轮没有再次执行整套复现以免额外重复推理，实际主执行、9测试和产物校验构成证据。GPU需正常可用。配置/种子/样本/输入指纹分别见config/preregistration/samples/protected_inputs；RoMa字节指纹与许可在上述identity/本地文本。

人工核验继续使用**原网页与4任务**，不改变UI。另安排尚未看算法图/辅助层的真实标注人和第二复核人；本报告读者不能再声称对已看样本的算法诊断完全盲。仅确认静态、双图可见且能赋实际定位不确定度的共同地物，允许不可评价；不得依据算法点挑人工check或替换难点。原入口：

```bash
conda run --no-capture-output -n match python RealSAR_IndependentBenchmark_20261011_2015/serve_review.py --port 8768
# 浏览器 http://127.0.0.1:8768；远程先SSH端口转发。
# 标注及第二真人复核完成后，停止编辑并导出：
conda run --no-capture-output -n match python RealSAR_IndependentBenchmark_20261011_2015/export_annotations.py
# 仅后续独立评价读取真实完成导出，禁止预测重拟合：
conda run --no-capture-output -n match python {ROOT.name}/evaluate_sparse.py --human-export /绝对路径/新的真实人工导出 --out /绝对路径/新的评价目录
```

原质量规则和辅助暴露记录继续有效，详细见 [原人工指南](../RealSAR_IndependentBenchmark_20261011_2015/reports/annotation_guide_20261011_2038.md)。目前未等人工操作、未读取人工点、未模拟复核、未计算真实Success。

**下一步判断：值得研究SAR域特征与弱监督/局部几何，但当前只有“可见光预训练迁移在两对异向失效、同向控制可用”的小样本证据，不足以证明应立即微调某网络或训练能恢复不可辨认地物。** 先完成独立静态对应/不可评价判定，再对本次冻结预测作check评价；若共同结构真实存在而全方法失败，优先研究跨方向SAR局部特征及地理约束下对应（训练来源/确认来源隔离）。若异向样本根本无可确认共同静态点，优先改进真实数据与观测/地形几何，不能拿自拟合/地理伪点直接训练并声称精确监督。停止当前四对上的传统预处理/尺度或RoMa内部搜索，本轮未开始训练。

数据许可沿用原声明：Umbra与Capella CC BY4.0；S1 Indigo使用范围保持原项目约束。Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026。
'''
    (ROOT/('matching_pilot_report_'+stamp+'.md')).write_text(report)
    print('Report and consolidated execution log completed at',stamp)
if __name__=='__main__':main()
