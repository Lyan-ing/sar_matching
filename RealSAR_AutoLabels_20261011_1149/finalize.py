"""Verify artifacts and frozen inputs, then issue truthful timestamped handoff."""
import sys
sys.dont_write_bytecode=True
import argparse,json,subprocess,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from common import ROOT,BASE,SOURCE,MANUAL,PROJECT,save,sha,configure

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])
def verify_files(root,mapping,label):
    mismatch=[]
    for i,(name,digest) in enumerate(mapping.items()):
        p=root/name
        if not p.is_file() or sha(p)!=digest:mismatch.append(name)
        if (i+1)%10000==0:print('Preservation',label,i+1,'/',len(mapping),flush=True)
    if mismatch:raise ValueError(f'Frozen files changed: {label} {mismatch[:10]}')
    return {'checked':len(mapping),'changed':0}

def main(out):
    out=Path(out).resolve();configure(out);analysis=json.loads((out/'quality/analysis_summary.json').read_text());assert analysis['crop_pairs']==191 and analysis['C']==0
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(out/'tests')]
    test=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(out/'reports'/f'tests_{stamp}.log').write_text(test.stdout);print(test.stdout,flush=True)
    if test.returncode:raise RuntimeError('Release tests failed')
    if 'skipped' in test.stdout:raise RuntimeError('Final release checks must not be skipped')
    browser=json.loads((out/'tests/browser_acceptance.json').read_text());assert browser['passed'] and browser['real_annotations_written']==0
    provenance=json.loads((out/'predictions/all_frozen/provenance.json').read_text());inference=json.loads((out/'predictions/all_frozen/SUMMARY.json').read_text());assert inference['rows']==1146
    rows=[json.loads(s) for s in (out/'dataset_index.jsonl').read_text().splitlines()];assert all(not r['has_C_independent_reference'] and not r['independent_test_GT'] for r in rows)
    assert not list((out/'annotation_adapter/annotations').rglob('*.json')) and not (out/'annotation_adapter/annotations/assistance_events.jsonl').exists()
    prereg=json.loads((out/'preregistration.json').read_text());assert prereg['old_manual_receipt_sha256']==sha(MANUAL/'ENGINEERING_ACCEPTANCE.json') and prereg['source_delivery_receipt_sha256']==sha(SOURCE/'DELIVERY_COMPLETE.json')
    protected={'historical_frozen':verify_files(BASE,json.loads((SOURCE/'protected_artifacts.json').read_text()),'historical_frozen'),'original_real_release':verify_files(SOURCE,json.loads((SOURCE/'DELIVERY_COMPLETE.json').read_text())['file_sha256'],'original_real_release'),'original_annotation_project':verify_files(MANUAL,json.loads((MANUAL/'ENGINEERING_ACCEPTANCE.json').read_text())['file_sha256'],'original_annotation_project')}
    save(out/'quality/preservation.json',protected)
    # Capture new source plus reused code identity; no Git mutations.
    code=[*out.glob('*.py'),out/'config.json',*out.glob('tests/*.py'),*out.glob('annotation_adapter/*.py'),*out.glob('annotation_adapter/*.js'),out/'annotation_adapter/index.html']
    code_hash={str(f.relative_to(out)):sha(f) for f in code}
    for f in code:
        dst=out/'code_snapshot'/f.relative_to(out);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dst)
    save(out/'code_snapshot/SHA256.json',code_hash)
    versions=provenance['dependencies'];save(out/'environment.json',{'python':provenance['python'],'packages':versions,'torch':provenance['torch'],'cuda':provenance['torch_cuda'],'device':provenance['device'],'new_packages_installed_this_task':False,'browser_test_tooling':'existing temporary Playwright 1.58.0; outside match site-packages; not production dependency'})
    (out/'requirements.lock.txt').write_text('\n'.join(f"{v['name']}=={v['version']}" for v in versions)+'\n')
    algorithms=json.loads((out/'quality/algorithm_summary.json').read_text());strata=json.loads((out/'quality/strata.json').read_text());scales=json.loads((out/'quality/scale.json').read_text());review=json.loads((out/'annotation_adapter/annotation_tasks/task_freeze.json').read_text())
    # Minute suffix reflects finalization time, not the earlier run-directory time.
    completed=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=completed.strftime('%Y%m%d_%H%M');report_name=f'autolabel_report_{stamp}.md';log_name=f'execution_{stamp}.log'
    algorithm_table=table(['协议','匹配器','尝试','门控接受','B候选','高候选','无矩阵输出','输出后拒绝','中位ms'],[[r['protocol'],r['method'],r['attempts'],r['gate_accepted'],r['B_eligible'],r['B_high'],r['no_output'],r['output_rejected'],round(r['time_ms']['median'],2)] for r in algorithms])
    def strat(key):return table([key,'裁片对','真实采集对','站点','A','B','高B','A-only','弱监督候选'],[[r['value'],r['crop_pairs'],r['real_acquisition_pairs'],r['sites'],r['A'],r['B'],r['B_high'],r['A_only'],r['weak_candidates']] for r in strata if r['stratifier']==key])
    scale_table=table(['来源','像素间距x中位(m)','像素间距y中位(m)','512边长范围(m)','不同CRS裁片对'],[[r['sensor'],round(r['spacing_x_m']['median'],4),round(r['spacing_y_m']['median'],4),f"{r['coverage_edge_m']['min']:.1f}–{r['coverage_edge_m']['max']:.1f}",r['different_crs']] for r in scales])
    failures='\n'.join(f"- {r['protocol']}/{r['method']}: gate={r['gate_reasons']}; pseudo筛选={r['pseudo_reasons']}" for r in algorithms)
    differences=table(['协议/方法','门控接受数','与地理对应的点偏差中位数之中位(px)','该点偏差p90之中位(px)','同样匹配的H门控接受数'],[[r['protocol']+'/'+r['method'],r['gate_accepted'],round(r['accepted_geo_difference_median_px']['median'],3) if r['accepted_geo_difference_median_px']['median'] is not None else '无',round(r['accepted_geo_difference_p90_px']['median'],3) if r['accepted_geo_difference_p90_px']['median'] is not None else '无',r['homography_gate_accepted']] for r in algorithms])
    body=f'''# 真实SAR自动地理对应与伪标签工程报告

完成时间：{completed.isoformat()}。运行目录的1149是启动时间；报告和执行日志使用实际验收分钟{stamp}。

## 已交付结果与证据边界

完整保留既有{analysis['crop_pairs']}对512原生裁片、{analysis['real_acquisition_pairs']}个真实采集对、{analysis['acquisitions']}次采集、{analysis['sites']}站点、{analysis['components']}来源组件；未下载、重新裁片或训练。

- A级地理参考：{analysis['A']}/191。双向覆盖均≥50%：{analysis['overlap_ge_0_5_both']}；低于50%：{191-analysis['overlap_ge_0_5_both']}。
- B级自动伪标签：{analysis['B']}，其中高候选{analysis['B_high']}、一般候选{analysis['B']-analysis['B_high']}；仅A级{191-analysis['B']}。A/B为累积证据，B仍具有A级地理信息。
- C级独立核验：0，真实独立对应点0；全部gt_status=unverified。没有计算真实Success@3/5/10px。
- 弱监督候选单位：{analysis['candidates']['weak']}。
- 伪标签候选单位：{analysis['candidates']['pseudo']}；高候选单位：{analysis['candidates']['high_pseudo']}。
- 6种协议/匹配器组合共1146条实际推理记录，运行异常{inference['runtime_errors']}。每条失败及拒绝原因均保留。

统一索引：dataset_index.jsonl / dataset_index.csv。georeference_correspondences/、pseudo_labels/、quality/与visualization/均为本轮新结果。全部191对拥有地理与伪标签状态预览；后续诊断抽查{review['tasks']}对，不是随机无偏测试集。

## 地理映射实现和质量

严格使用原生GeoTIFF和原生PNG；像素中心先+0.5，通过包含旋转/剪切项的Affine映射至地图坐标，必要时使用rasterio/GDAL CRS转换，逆Affine后-0.5。43对不同CRS采用逐点非线性转换；同CRS可以给出准确的地图Affine组合，跨CRS不伪造全局矩阵。

16×16规则网格按共同有效区域筛选，不填充无效点；双向共{analysis['geographic_point_count']}地理采样点。每视图网格有效点统计{analysis['grid_count_per_view']}。最差数值往返误差{analysis['roundtrip_max_px']:.3g}px。此误差验证数学实现，不验证产品定位精度或散射中心一致性。

双图掩膜为声明原生有效掩膜、GeoTIFF读取掩膜与有限值的交集；合法零值保留。分数像素要求参与双线性采样的邻居全部有效，越界和NoData排除。双向重叠分母同时提供全图像素和有效像素。全图最小双向覆盖统计{analysis['overlap_min_both']}。

{scale_table}

512像素不是统一地面空间尺度。表中距离来自产品地理编码的局部投影网格，不等同于传感器物理分辨率。原始分辨率元数据、日期、轨道、方位、入射角、极化及缺失项均沿用来源索引；没有编造角度或绝对定位精度。

低纹理、暗区、饱和与量化统计只作为显示/原生数值代理，逐图见quality/image_conditions.csv。不同来源的功率、振幅、量化DN及PNG不可用统一阈值解释真实散射。暗区代理不等于已验证阴影；未声称识别叠掩或准确排除动态舰船。人工工程预览抽查Umbra_0020__chip_000：reference显示中位值为0、有效暗像素约59.4%、PNG只有55个不同显示值；原生量化DN仍有126个值，不是完全空白影像。强斑点与显示压缩可能掩盖稳定结构，低结构代理不会可靠区分噪声与真实地物；保留原显示，不通过重新拉伸和重跑提高通过数量。弱监督候选状态因此仅代表基本工程筛选。

## 冻结推理与伪标签条件

复用Phase2.5锁定改进特征检测：SIFT contrastThreshold=0.01，ORB edgeThreshold=16/fastThreshold=12，SuperPoint detection_threshold=0.0001；均2048特征，LightGlue官方冻结权重，Affine与原full可靠性门控，seed42。标签制作seed261011。selection.json、源代码、配置、LightGlue版本和权重哈希在运行前核对，记录于predictions/all_frozen/provenance.json。没有新参数搜索，也未新增复杂SAR-SIFT实现。

native为仅原生图像预测；geo_assisted为把已有moving_native.png双线性重投影到reference原生网格后匹配。地理辅助输入只用原生512像素，不使用可能引入裁片外像素的旧full-scene粗对齐图。输出匹配统一转换回双图原生坐标。地理筛选只发生于标签制作与诊断；native匹配器没有读取H、重叠Mask或独立GT。geo_assisted明确使用地图先验，不能混入纯图像指标。

{algorithm_table}

这些是门控接受与候选标签数量，**不是配准正确率**。B条件在生成前写入config.json并由preregistration.json固定：原Affine/full门控通过、双图原生有效内点≥8、有效内点占比≥0.95、双图内点凸包≥0.01；地图偏差median≤32、p90≤64px。高候选还需内点≥20、内点比≥0.5、双图凸包≥0.08、各至少6/16网格单元、地图偏差median≤8/p90≤16px，以及至少2个不同算法家族在≥16有效地图格点上预测差p90≤3px。同一算法的两个协议不算两个家族。

上述阈值是未校准的候选标签工程筛选，不宣称真实误差概率。地图偏差很大时拒绝B并保留A，可能舍弃真实图像匹配但避免把无法解释的偏差直接当高可信监督；没有通过改写地图坐标掩盖偏差。标签只限内点凸包、双原生有效支持和共同地理区域交集，不把整图矩阵外推当成稠密真值。

匹配能力与几何模型诊断分开：每次匹配额外使用相同原始匹配拟合Homography，保留投影分母、全幅角点稳定性、空间分布、双向一致性和局部4×4内点重代入残差；Homography仅诊断，不参与伪标签选择。局部残差没有拟合局部模型，不是独立检查点误差。

{differences}

偏差为匹配点和同地图坐标之间的像素距离，不是真实定位或配准误差。拟合残差与算法间一致性也不证明真实性。产品定位误差、不同散射中心、变化目标、地形叠掩或全局模型不适用均可能造成偏差；现有证据无法唯一归因。即使地理辅助结果看起来吻合，也不能宣称1px GT。

## 分层及失败机制

{strat('sensor')}

{strat('look_pair_category')}

{strat('overlap_bin')}

实际结果的关键限制：Sentinel-1为13/13个B候选，Capella为6/24，Umbra为0/154；已验证不同观测方向的151裁片为0个B，相近方向重复观测为18/36，方向未知为1/4。全部19个B都来自双向地理覆盖≥80%的样本；低/中覆盖没有B候选。这支持相近方向/高覆盖数据的伪标签流程可行，但**没有为真实异向SAR提供足够的高可信自动监督**。不同来源、尺度、场景与视角强混杂，不能据此把退化唯一归因于方向。

最终19个伪标签均选自native/ORB，是固定优先级的结果；同一图像也可能具有SIFT/LightGlue可用候选，不能据此宣称ORB真实精度最好。高候选使用不同算法家族一致性证据，仍可能共同错误。现阶段更适合启动受控弱监督/数据质量探索，而不是直接依赖B级做大规模真实多视角监督训练。

{strat('scene_class')}

scene_class主要沿用站点级类别；只对旧抽查样本保留已有visual_scene，不能当作对191裁片逐张人工重标。海面、农田、城市与矿区的具体原因应结合示例核验，不能仅由站点标签推断。

{strat('low_structure_proxy')}

{strat('component')}

quality/algorithm_strata.csv给出每种算法×协议在来源、观测类别、重叠及场景下的数量、特征、内点、门控、地图偏差及耗时；quality/acquisition_pairs.csv按65个真实采集对汇总，避免把同源裁片当独立采集。类别分布、来源、尺度与站点相互混杂，不能把观测类别的数量差异直接解释为视角因果效应。

{failures}

quality/feature_diagnostics.csv保留1146条关键点数量、空间覆盖、匹配、原生有效内点、局部残差及模型诊断。无输出、算法拒绝、运行异常与伪标签附加筛选拒绝单独记录；缺乏C级时无法判定“错误接受”的真实数量。不能以门控接受率代替真实精度。

## 实际成本

PNG读取至特征提取、匹配、Affine估计与原full门控的实际GPU同步耗时已记录；包含首次调用标记，模型初始化另列。native与geo_assisted分开；Homography诊断、缓存写入与地图质量计算不在主推理时间中，额外耗时另列。地理网格/双向重投影/预览成本见每对runtime.json和quality/scale.json。

{table(['协议/算法','主推理总s','同匹配H额外s','主推理中位ms','主推理p90ms'],[[r['protocol']+'/'+r['method'],round(r['total_primary_s'],2),round(r['homography_extra_s'],2),round(r['time_ms']['median'],2),round(r['time_ms']['p90'],2)] for r in algorithms])}

设备：{provenance['device']}；Python {provenance['python']}，PyTorch {provenance['torch']}，CUDA构建 {provenance['torch_cuda']}。生产依赖实际版本见environment.json与requirements.lock.txt；未安装新生产包，OpenCV沿用headless单发行包。临时浏览器工具只用于既有UI兼容验收，不属于数据制作依赖。

## 来源隔离、许可与训练使用建议

现有component/site/acquisition到split关联检查无跨划分复用；保留原train/dev/val/confirmation追溯，但已查看的confirmation不是严格封存测试集。未来训练前需要重新按地理站点/来源组件和采集共同身份固定协议；不能把65个采集对产生的191裁片都视为独立观测，也不能把一个站点的相关裁片拆分训练/测试。已有HRSID/SSDD缺少采集身份，仍无法证明与其未知采集场景绝对无关联。

当前A级足以支撑地图弱监督的数据管道与小规模探索，B级可作为经过限定区域的伪标签候选。实际候选单位已在开头列出；58次采集、12站点且来源尺度和类别不平衡，不能以191对数量证明足够大或泛化良好。**目前不建议把本集合直接作为高可信全监督基准，更不能以这些标签自评生成器。**先按组件固定新的训练/验证隔离、完成少量C级静态地物核验、检查动态/叠掩和场景变化，再开展小规模SAR域特征学习探索；本轮没有训练。合成数据未来可用于几何预训练，但不能替代真实独立GT。

A-only、unreliable_candidate、低覆盖和显示低结构代理高的样本应保留供探索/诊断，暂不做高可信监督。B高候选也是自动证据，优先独立核验；一般B应仅用于鲁棒且带支持区域/不确定性的伪监督，不作精确评测。伪标签生成器不得拿同一伪标签评价自己的真实准确率。是否需要SAR域微调及其收益，必须由新隔离数据与C级证据验证。

Umbra/Capella沿用CC BY4.0及其原作者/产品身份署名，Capella保留Copyright Capella Space。Sentinel-1 Indigo RTC沿用来源研究/非营利使用条件，不能自动当作无条件商用数据：Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026。每条reference/moving的license与license_url原样保留；不同许可不可被统一CSV覆盖。

## 人工工具接口与验收

原人工系统文件和记录均未改动。annotation_adapter/serve.py在运行时复用原HTTP/Store/质量规则；地理图层与自动伪标签默认关闭，不导入人工点，也不升级gt_qualified。旧UI内部sift_candidates槽仅作为兼容接口，新的按钮、提示及暴露日志如实标为pseudo_correspondences，并记录实际native/geo_assisted算法、人员、时间和候选SHA。查看后再隐藏不取消暴露记录，原质量规则排除辅助暴露的独立GT资格。

{review['tasks']}个抽查任务位于annotation_adapter/diagnostic_tasks.json，覆盖{review['sites']}站点/{review['real_acquisition_pairs']}真实采集对。旧5个代表样本加3个新站点优先选B用于成功/失败诊断，选择有意识、非随机，不称无偏测试。本轮不要求立即人工标注。

最终pytest所有检查通过、没有跳过；覆盖像素中心、不同尺度、旋转/剪切、跨CRS非线性、NoData、越界、双向覆盖、191对真实点/Mask、组件隔离、伪标签资格与辅助暴露。已有UI坐标交互规则复用；新Chromium验收{len(browser['checks'])}组通过，8个真实任务加载正确，地理/伪标签图层原生坐标一致，默认不导入、来源留痕准确。模拟操作全部在/tmp；新旧真实人工点和C级均0。仅有Affine测试运算符的PendingDeprecationWarning，不影响断言。

原冻结结果SHA保护核验：{protected}。这只是确认未改动，没有重复下载、来源重复审计或历史算法实验。

## 复现与读取

```bash
cd /home/yl/matching/sar_matching
# 必须给未存在的同级目录，拒绝覆盖。本轮配置与模型保持冻结。
conda run --no-capture-output -n match python RealSAR_AutoLabels_20261011_1149/reproduce.py --out /home/yl/matching/sar_matching/RealSAR_AutoLabels_REPLAY_YYYYMMDD_HHMM
conda run --no-capture-output -n match python -m pytest -q -p no:cacheprovider RealSAR_AutoLabels_REPLAY_YYYYMMDD_HHMM/tests
# 使用原工具的薄适配入口，可用SSH隧道转发8766；默认独立模式。
conda run --no-capture-output -n match python RealSAR_AutoLabels_20261011_1149/annotation_adapter/serve.py --port 8766
```

完整流水线generate_geo.py → infer.py → labels.py → prepare_review.py → analyze.py。config.json固定标签条件；preregistration.json记录运行前配置及来源SHA；predictions/all_frozen/provenance.json保存被冻结方法与权重身份、依赖、设备和计时范围；新源码快照见code_snapshot/SHA256.json，旧代码Git版本在冻结provenance内。requirements.lock.txt为实际版本记录；GPU版PyTorch仍需匹配官方CUDA wheel来源，不建议盲目在base安装。

同样配置、权重、种子与环境可重放实验协议，但GPU算子没有额外强制逐位确定性；不承诺跨硬件浮点输出完全一致。当前1146条逐样本记录及NPZ缓存保留实际原始输出，可直接重建本次统计而无需再次推理。

读取dataset_index.jsonl逐行JSON；image0=reference_native_png、image1=moving_native_png；A级读geographic_correspondences_path中的points/reverse_sampled_points，B级读pseudo_correspondences_path和pseudo_transform_path，监督范围限定pseudo_support_path。跨CRS geo_assisted变换可能是地图非线性映射与Affine组合，native_H为空时必须按composition执行，不能强制当全局H。

伪标签置信度由内点、空间覆盖、原门控、地图差异和算法家族一致性组成；confidence_is_probability=false，dynamic_exclusion_verified=false。未来消费者必须再次固定训练划分、许可及独立测试协议；training_eligible_now保持false。
'''
    (out/'reports'/report_name).write_text(body)
    commands=['generate_geo.py (191/191 completed before inference)','infer.py (191 pairs ×2 protocols ×3 methods, real GPU)','labels.py','prepare_review.py','analyze.py','tests/browser_check.py (existing temporary Chromium)','finalize.py (focused tests + frozen SHA protection)']
    log={'completed_at':completed.isoformat(),'execution_record_type':'final evidence summary; raw inference/test logs stored separately','commands_all_python_in_match':commands,'frozen_config_sha256':sha(out/'config.json'),'source_index_sha256':sha(SOURCE/'dataset_index.jsonl'),'preregistration':prereg,'results':analysis,'inference':inference,'browser':browser,'preservation':protected,'source_changes':'new standalone scripts and thin adapter only; previous code and outputs unchanged','no_training':True,'no_independent_accuracy_claim':True,'report':report_name}
    (out/'reports'/log_name).write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n')
    readme=f'''真实SAR自动对应与训练候选数据（非独立GT）\n\n结果：A={analysis['A']}；B={analysis['B']}（高={analysis['B_high']}）；C=0。191裁片/65真实采集对/58采集/12站点。\n\n统一入口 dataset_index.jsonl；报告 reports/{report_name}；执行日志 reports/{log_name}。详见报告证据边界、原生坐标、限定支持区域、组件隔离与许可。\n\n重放：conda run --no-capture-output -n match python {out/'reproduce.py'} --out /home/yl/matching/sar_matching/RealSAR_AutoLabels_REPLAY_YYYYMMDD_HHMM\n\n标注接口：conda run --no-capture-output -n match python {out/'annotation_adapter/serve.py'} --port 8766\n\n所有原始成果保持冻结。此版本没有训练，不提供真实Success@px。\n'''
    (out/'README.md').write_text(readme)
    files={str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file() and '__pycache__' not in f.parts and '.pytest_cache' not in f.parts and f.name!='ENGINEERING_ACCEPTANCE.json'}
    receipt={'completed_at':completed.isoformat(),'engineering_accepted':True,'independent_GT_accepted':False,'A':analysis['A'],'B':analysis['B'],'B_high':analysis['B_high'],'C':0,'report':str(out/'reports'/report_name),'execution_log':str(out/'reports'/log_name),'dataset_index':str(out/'dataset_index.jsonl'),'preservation':protected,'file_sha256':files}
    save(out/'ENGINEERING_ACCEPTANCE.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='file_sha256'},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default=str(ROOT));a=p.parse_args();main(a.out)
