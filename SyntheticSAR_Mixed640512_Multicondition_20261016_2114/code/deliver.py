"""One evidence-based report, usage instructions and machine entry point."""
from common import *
from qa import contact
import platform,csv

def table(headers,rr):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rr)+'\n'

def main():
    t=stamp();release=json.loads(latest('meta/RELEASE_METADATA_*.json').read_text());cfgpath=Path(release['config']);cfg=json.loads(cfgpath.read_text())
    qpath=latest('qa_*/QA_SUMMARY_*.json');q=json.loads(qpath.read_text());assert q['passed'] and q['pairs']==6000
    rr=[r for p in release['manifests'].values() for r in rows(p)]
    old=json.loads(latest('meta/HISTORICAL_CODE_AUDIT_*.json').read_text());audit=json.loads(latest('meta/SOURCE_AUDIT_SUMMARY_*.json').read_text());build=json.loads(latest('meta/BUILD_SUMMARY_*.json').read_text())
    interface=json.loads(latest('meta/INTERFACE_CHECK_*.json').read_text());replay=json.loads(latest('meta/REPLAY_CHECK_*.json').read_text());assert interface['passed'] and replay['passed']
    detector=json.loads(latest('meta/JOINT_DETECTOR_INTERFACE_*.json').read_text());resume=json.loads(latest('meta/RESUME_CHECK_*.json').read_text());cli=json.loads(latest('meta/SCORER_CLI_CHECK_*.json').read_text());assert detector['passed'] and resume['passed'] and cli['passed']
    qa_records=rows(qpath.parent/('PAIR_QA_'+qpath.stem.replace('QA_SUMMARY_','')+'.jsonl'))
    ratios=[x['coarse_anchor_contrast_ratio_min'] for x in qa_records if x.get('coarse_anchor_contrast_ratio_min') is not None]
    manifest_hashes={s:sha(p) for s,p in release['manifests'].items()}
    selected=[];seen=set()
    # Fixed metadata coverage, never matcher successes, selects figure examples.
    for r in rr:
        keys=[('scene-size',r['source_scene'],r['width']),('overlap-size',r['overlap_bin'],r['width']),('angle',r['angle_bin']),('profile',r['enhancement_profile'])]
        if any(k not in seen for k in keys):selected.append(r);seen.update(keys)
    figures=[]
    for page in range((len(selected)+3)//4):
        path=ROOT/f'figures/RELEASE_EXAMPLES_{page:02d}_{t}.jpg';contact(selected[page*4:page*4+4],path);figures.append(path)
    controls=rows(latest('smoke_*/SMOKE_MANIFEST_*.jsonl'))[-5:]
    control=ROOT/f'figures/SAME_NATIVE_SOURCE_P0_P4_{t}.jpg';contact(controls,control)
    sizes=Counter(r['width'] for r in rr);scene_counts=Counter(r['source_scene'] for r in rr);opcounts=q['actual_ops_per_view'];none=q['actual_no_extra_operations']
    classes=[('P0','不额外外观增强'),('P1','仅轻度辐射'),('P2','轻度随机综合'),('P3','中等随机综合'),('P4','非对称，一侧P0')]
    evidence=ROOT/f'meta/DELIVERY_EVIDENCE_{t}.json'
    save(evidence,dict(created=now(),passed=True,pairs=6000,views=12000,manifests=release['manifests'],manifest_hashes=manifest_hashes,
        QA=str(qpath),source_audit=str(latest('meta/SOURCE_AUDIT_SUMMARY_*.json')),historical_audit=str(latest('meta/HISTORICAL_CODE_AUDIT_*.json')),
        interface_check=str(latest('meta/INTERFACE_CHECK_*.json')),replay=str(latest('meta/REPLAY_CHECK_*.json')),
        joint_detector_interface=str(latest('meta/JOINT_DETECTOR_INTERFACE_*.json')),resume=str(latest('meta/RESUME_CHECK_*.json')),scorer_CLI=str(latest('meta/SCORER_CLI_CHECK_*.json')),
        python=platform.python_version(),numpy=np.__version__,opencv=cv2.__version__,cpu_workers=4,
        code_hashes={str(p):sha(p) for p in (ROOT/'code').glob('*.py')},code_version='isolated source snapshot; repository has no Git metadata',
        local_structure_contrast_ratio_min=min(ratios) if ratios else None,all_GT_and_images_present=True,
        no_download_no_network_training=True,old_data_preserved=True))
    scene_names=dict(zip(SCENES,['离岸舰船／低纹理海面','近岸舰船','港口／码头／密集舰船','其他陆地结构']))
    split_table=[]
    for s in ['train','val']:
        item=q['split'][s];split_table.append([s,item['pairs'],item['pairs']*2,item['sizes'].get('640',0),item['sizes'].get('512',0),item['anchor_files'],item['visible_original_files'],item['components']])
    stat='''# Mixed640512合成SAR数据集：构建与验收报告

生成时间：{created}。

## 1. 交付结论与边界

已实际生成并读回验收 **6,000对、12,000幅图像实例**；Train 4,800对，Val 1,200对。全部同对同尺寸，只含640或512。未上采样原始图像，未扩边或填补未观测区域，未下载影像，未修改真实SAR标注、历史900基准或模型权重，未启动训练。

数据目录：`{root}`。机器入口：`DATASET_ENTRYPOINTS_MIXED_{tag}.json`。报告只统计正式release；16组首轮小样和60组矩阵小样另存，不计入6,000对。P0—P4同源对照同样不计作新增独立来源。

**这是同源二维合成几何数据，不是真实不同雷达方位数据。** 解析网格GT与保守结构监督分别保存；低纹理样本可以有几何GT而没有可靠静态结构正点。所有逐裁片`verified_visible_scene`仍为unknown；下列场景比例为已有来源语境标签，不能宣称已逐片确认海陆/港口语义。

## 2. 实际规模、尺寸与来源

{split_table}

640共 **{n640}对（{p640:.2f}%）**，512共 **{n512}对（{p512:.2f}%）**。512的逐对原因在`reason_if_512`，包含640联合视场不足、有限次数有效域/旋转/重叠拒绝。没有强迫640配额。

正式12,000幅实例中，唯一输出像素哈希 **{unique}**，重复像素实例 **{duplicates}**。选用锚发布文件 **{anchors}** 个，实际可见发布文件 **{originals}** 个，已知来源组件 **{components}** 个。文件数、源坐标域数与真实独立SAR采集数不能混称；HRSID部分原采集身份不完整。

单锚最大复用 **{anchor_reuse}次**，实际可见原始发布文件最大复用 **{raw_reuse}次**。每源贡献见QA目录`SOURCE_REUSE` CSV。Train/Val按历史原始组件先划分，再生成；父组件、源路径、文件字节哈希、解码像素哈希跨分区交集均为0，且排除了原Test组件。Val来源在历史研究中已被使用，不能称全新未见最终测试。

### 原始支持审计与扩展源坐标域

共读检5,948个已有发布文件：HRSID 4,831、SAR背景357、SSDD 760。HRSID/背景均800×800；SSDD仅1个文件尺寸达到512，其余759个不足，不能上采样凑数。尺寸达到门槛仍不保证任意角度/重叠可行。

仅靠单张800图，不能一般地构造640低重叠或严格无重叠双视场。本轮最小扩展是**源读取适配器**：同数据集、同父场景、已有文件行列范围与重叠像素一致的相邻发布裁片，可组成共同原生源坐标域。文件名实际顺序经像素验证为`row_start,row_end,column_start,column_end`，不是x/y顺序。

检查11,229条相邻关系，10,388条满足重叠区域≥99.5%像素完全相等且平均差≤0.05 DN。得到5,189个有尺寸支持的锚源域，其中4,860个可利用已验证相邻裁片，329个为单发布图。这里只是候选读取域，绝非新增真实采集。邻域最多3×3裁片，不跨数据集拼接，不平均像素，不修补缺口；确定性像素所有权、冲突区域无效化、拼缝保护区均保留。实际落盘QA要求每侧完整双线性有效比例≥98%。

逐文件SHA、有效域、尺寸、身份关联见`meta/SOURCE_AUDIT`；相邻像素证据见`COORDINATE_EDGE_AUDIT`；分场景/尺寸/重叠的上界估计及有限几何探针见`PREGEN_FEASIBILITY_ESTIMATES`和矩阵小样目录。维度上界不等于已生成容量，每对最终仍检查实际掩膜。

## 3. 实测场景、重叠与旋转

以下为**原来源语境**，继承官方粗分类及既有AI预览记录；不是逐像素语义GT。船舶多边形/框只用于排除，检测框不被当成精确船体分割。邻域视场可能包含多类背景，实际裁片语义未知保持显式unknown。

{scene_table}

{overlap_table}

重叠率用落盘有效掩膜与H重新计算双向占比，主值为`min(r_A,r_B)`；左闭右开，最后箱含100%。本轮不含(0%,10%)额外挑战样本。所有300个无重叠负例还验证了共同父源坐标下两个完整视场凸多边形交面积为0，不能与“结构少/算法失败”混同。

{angle_table}

旋转是H在A中心雅可比极分解的实际二维旋转，不是真实SAR观测方位差。刚体、相似与微小投影变化预先随机分配；尺度0.96—1.00，无原图预放大，剪切在±1°，透视角点扰动比例0.0006。H与两侧绝对源变换准确保存。

{joint_table}

两种尺寸×六个重叠箱×五类增强的60个组合全部有正式样本，512没有被集中安排为低重叠。受原始支持约束，512在30°—120°和单图陆地背景中的比例更高；这是真实存在的尺寸/条件关联，不能宣称完全去混淆，评价必须继续按尺寸与角度/场景分层。

**物理/几何不可实现格子如实保留：** 等尺寸正方形约45°相对旋转时，不能具有90%以上双向公共视场。因此90%—100%箱没有强迫15°—30°/30°—60°角度箱，其他箱承接相应角度以保留精确边际配额。严格无重叠也未分配给只有孤立800裁片的陆地语境。高重叠条件内角度不再声称均匀。完整`size×overlap×angle×scene×split×profile`交叉CSV含实际非空格，缺格按源支持和正方形视场解释，不通过旋转预对齐或黑边制造覆盖。

## 4. 实际增强与强度

{profile_table}

P0保持几何采样后的原始显示DN，不增加纹理、散斑、辐射、模糊或散射修改。P1只调用原有辐射函数。P2随机选1—2个操作；P3选2—3个；P4一侧P0、另一侧P2或P3。操作按实际随机顺序组合，A/B种子分开。所有操作在显示DN域，Gamma随机场形状不是实测ENL；原始SAR固有散斑仍共享。

{ops_table}

实际无额外散斑的视图 **{no_speckle}/{views}**；无额外纹理的视图 **{no_texture}/{views}**；无局部散射修改的视图 **{no_scatter}/{views}**；完全无外观操作的视图 **{no_appearance}/{views}**。两侧均未额外加散斑的配对为 **{no_speckle_pairs}**。这些是实际执行统计，不是P0计划比例的替代。

轻度辐射Gamma 0.96—1.04、对比0.97—1.03、偏置±0.004；中度为0.92—1.08、0.95—1.05、±0.007。细纹理残差衰减轻度0.04—0.16、中度0.12—0.30并保护粗边缘；相关场sigma轻度0.003—0.010、中度0.008—0.018；乘性对数正态扰动变异系数轻度0.02—0.06、中度0.06—0.12；小核各向异性模糊sigma≤0.65。局部软散射半径sigma 3—6，增益轻度0.85—1.15、中度0.65—1.25，区域≤2.5%，并加入双向ignore。

局部散射虽被随机选中，但找不到合法亮区时明确记skip，不计作实际变化。规范manifest保存`selected_ops`、实际`ordered_ops`、skip原因、种子、局部事件和掩膜引用；原始生成记录原样保留。此规范化只修正执行状态表达，不改变任何图像、GT、裁片或随机采样。

![同源同几何P0—P4对照：绿点为合成结构候选；灰为有效域、青为公共域、橙为ignore](../{control})

## 5. GT、结构监督与历史一致性

每对NPZ保存`T_A/T_B/H_AB/H_BA`、双方valid/overlap/ignore、局部变化掩膜、船舶排除区、static/calibration/unknown、网格检查点与结构点。T使用统一父源像素中心坐标，H=T_B×inv(T_A)，坐标x向右/y向下。落盘结构点为空是合法状态，不凑点。

{level_table}

几何正例 **{positive}** 对，其中有≥4个合法网格检查点可直接计算几何Success的 **{geometry_eligible}** 对；无重叠负例300对。含结构点 **{with_structure}** 对，≥4点局部描述子监督 **{local_desc}** 对，严格全幅结构资格 **{global_desc}** 对。全幅门槛仍为双方≥16点、4×4格≥4、凸包≥5%、特征值比≥0.03；没有因尺寸变化或数量目标降低门槛。舰船不混入静态地面点。

复用原来的`view_matrix/transform_points/warp/overlap_mask/ratio`，改变的是源读取适配、有限几何采样调度及可选增强组合。64组512/640几何往返与原辐射函数同种子逐像素一致性均通过；60组尺寸×重叠×增强矩阵全部读回通过。旧V2.1不足全幅门槛会清空局部点的问题已由V2.2分级标签修复，本轮复用V2.2原图特征缓存与语义范围，保留局部点。

历史V2.1 R共7,214对的实际掩膜重算结果为：70%—90% {old_high}、50%—70% {old_mid}、10%—30% {old_low}、严格0% {old_zero}；与旧manifest数值差为0。旧代码请求256/384/512，实际还有96—352等回退尺寸，本轮没有照搬。旧完整增强路径多数步骤必做，本轮拆成随机可跳过操作，并降低到轻中强度；不能声称新旧完整增强分布相同。

## 6. 全量质量验收及接口证据

- 6,000个GT与12,000幅PNG全部解码、哈希及尺寸检查通过，失败0。
- 双向实际重叠掩膜重算一致；300个严格无重叠均通过源域不相交检查。
- H组成、H互逆、有效域重建、点范围及ignore排除全部通过。最大双向点误差 **{maxerr:.3g}px**，逐对最大误差均值 **{meanerr:.3g}px**。
- 网格点数量均值 **{checks_mean:.2f}**、中位 **{checks_median:.1f}**；无重叠为0，不伪造空间支持。
- P0及P4干净侧逐像素等于原始几何采样结果。全量记录饱和、零像素比例、有效区std与原结构点局部粗对比保持比；有可测粗对比的结构点最小保持比 **{contrast_min}**。此测量不新增/移动结构点，也不冒充真人物理核验。
- 实际生成中的源/几何拒绝 **{rejections}** 次，最终未完成任务 **{failed_jobs}**；完整原因日志保留。正式图像没有按结构点数或模型成绩重新裁切。
- 评分器用解析H得到近零误差，60px错误H明显失败；无重叠的接受计为FPR；缺预测保留为失败分母。
- 冻结ORB与SAR-SuperPoint互近邻冒烟、M2原损失8组CPU前向通过；这是接口测试，**不是正式SP+LightGlue论文成绩**。旧`evaluate22.Model.extract`有≤512包装限制，640请直接使用本版读者和已支持任意32倍数尺寸的原网络，不改旧实验文件。首次冻结前向暴露了历史零损失辅助函数对可训练参数的依赖，新检查脚本仅提供数值相同的无梯度零值；同时隔离了历史`common`模块重名。失败日志保留，旧模型/损失文件未改。
- 额外使用原`train22`实际`graded_loss.loss22`与冻结B权重，在512和640各完成一次联合检测/描述监督前向，检测正单元非空、损失有限、权重SHA未变。没有反向传播或优化器更新。
- **{replayed}组**按冻结种子重放，图像、矩阵、掩膜和点数组逐项相同。已完成记录支持恢复且不会覆盖；Val图像、GT、manifest、参数及配置哈希另存冻结。

质量明细：`{qa}`；接口：`{interface}`；复现：`{replay}`；完整软件版本与源码SHA：`{evidence}`。所有网络参数保持不变，CUDA未初始化，优化器步数0。

## 7. 直接使用与评价定义

参见同目录`USAGE_{tag}.md`，以及根目录机器入口。`reader.py`提供现有V2.2式监督字典和`[2,1,H,W]`图像张量；混合尺寸应按尺寸组批或一次一对，不隐式resize。只有图像进入模型，H/掩膜/点仅用于显式监督及评价。

评分器接受预测H、匹配点、检测点和接受/拒绝状态，保留所有无输出。`Success@3/5/10px`使用合法公共域网格的平均双向投影误差；另外单列未放宽旧门槛的`StructureSuccess`。归一化误差除以图像对角线，512/640分别报告；不能将像素阈值解释成相同物理米级精度。投影误差均值仅对有合法H者计算，Success与拒绝率分母仍保留失败样本。

结构Precision的分母是双方位于可监督静态域的预测匹配，同时报告全匹配中正确结构匹配比例；结构Recall的分母为保存的合法结构锚点，双方3px内被正确匹配覆盖才计命中；Repeatability分母为能投影到另一侧静态有效域的实际检测点。它们不是全海面或真实跨采集物理对应的召回率。无重叠FPR分母为全部严格负例。

## 8. 代表性正式样本

以下按场景、尺寸、重叠、角度、profile的元数据覆盖选取，不按基线成绩。绿点仅为原图提取后映射的合成结构锚；右侧灰色有效、青色重叠、橙色忽略；训练PNG未绘制任何标记。逐裁片场景仍待语义确认。

{figure_md}

## 9. 尚存限制

现有原始发布裁片来源依赖较强，已知组件隔离不代表所有原始物理采集身份均已恢复；SSDD原始视场基本不支持本轮尺寸。语境标签与实际裁片语义未混称，不能用90%海事来源语境宣称90%逐片已确认舰船/港口。纯海面几何对应可能依赖共享原始散斑，不能代替真实多视角物理对应评价。需要算法Codex另行授权训练；本轮没有训练或重跑历史实验。
'''
    vals=dict(created=now(),root=ROOT,tag=t,split_table=table(['分区','配对','视图','640','512','锚文件','可见源文件','已知组件'],split_table),
        n640=sizes[640],n512=sizes[512],p640=sizes[640]/60,p512=sizes[512]/60,unique=q['unique_output_pixel_hashes'],duplicates=q['duplicate_output_pixel_instances'],
        anchors=q['anchor_original_files'],originals=q['actual_visible_original_files'],components=q['source_components'],anchor_reuse=q['max_anchor_source_reuse'],raw_reuse=q['max_visible_raw_source_reuse'],
        scene_table=table(['来源语境','Train','Val','总数'],[[scene_names[s],q['split']['train']['scenes'].get(s,0),q['split']['val']['scenes'].get(s,0),scene_counts[s]] for s in SCENES]),
        overlap_table=table(['实际重叠箱','Train','Val','总数'],[[b,q['split']['train']['overlaps'].get(b,0),q['split']['val']['overlaps'].get(b,0),sum(r['overlap_bin']==b for r in rr)] for b in OVERLAPS]),
        angle_table=table(['二维旋转角度','Train','Val','总数'],[[f'{ANGLES[i]:g}—{min(180,ANGLES[i+1]):g}°',q['split']['train']['angles'].get(str(i),0),q['split']['val']['angles'].get(str(i),0),sum(r['angle_bin']==i for r in rr)] for i in range(6)]),
        joint_table=table(['重叠箱','640对数','512对数'],[[b,sum(r['width']==640 and r['overlap_bin']==b for r in rr),sum(r['width']==512 and r['overlap_bin']==b for r in rr)] for b in OVERLAPS]),
        profile_table=table(['主类别','含义','Train','Val','总数'],[[p,desc,q['split']['train']['profiles'].get(p,0),q['split']['val']['profiles'].get(p,0),q['profiles'][p]] for p,desc in classes]),
        ops_table=table(['实际操作','视图次数'],sorted(opcounts.items())),views=12000,no_speckle=none['no_extra_speckle_views'],no_texture=none['no_extra_texture_views'],no_scatter=none['no_local_scatter_views'],no_appearance=none['no_appearance_ops_views'],no_speckle_pairs=none['no_extra_speckle_pairs_both_sides'],
        control=str(control.relative_to(ROOT)),level_table=table(['监督等级','总数'],q['levels'].items()),positive=q['positive_geometry_pairs'],geometry_eligible=q['geometry_evaluable_pairs'],with_structure=q['with_structure_points'],local_desc=q['local_descriptor_pairs'],global_desc=q['strict_global_structure_pairs'],
        old_high=old['overlaps']['70_90'],old_mid=old['overlaps']['50_70'],old_low=old['overlaps']['10_30'],old_zero=old['overlaps']['zero'],maxerr=q['geometry_max_error_px'],meanerr=q['geometry_mean_pair_max_error_px'],checks_mean=q['check_point_counts']['mean'],checks_median=q['check_point_counts']['median'],
        contrast_min=f'{min(ratios):.3f}' if ratios else '无可测点',rejections=build['source_rejections'],failed_jobs=build['failed_jobs'],replayed=len(replay['cases']),
        qa=qpath.relative_to(ROOT),interface=latest('meta/INTERFACE_CHECK_*.json').relative_to(ROOT),replay=latest('meta/REPLAY_CHECK_*.json').relative_to(ROOT),evidence=evidence.relative_to(ROOT),
        figure_md='\n\n'.join(f'![正式样本{idx+1}](../{p.relative_to(ROOT)})' for idx,p in enumerate(figures)))
    rendered=stat.format(**vals)
    reports=ROOT/'reports';reports.mkdir(exist_ok=True);report=reports/f'BUILD_REPORT_{t}.md'
    with report.open('x') as f:f.write(rendered)
    entry=dict(created=now(),version='Mixed640512-1',dataset_root=str(ROOT),pairs=6000,views=12000,split_pairs=dict(train=4800,val=1200),
        manifests=release['manifests'],config=str(cfgpath),config_hash=sha(cfgpath),reader=str(ROOT/'code/reader.py'),scorer=str(ROOT/'code/score.py'),
        pure_image_inference=str(ROOT/'code/inference.py'),replay=str(ROOT/'code/replay.py'),QA=str(qpath),delivery_evidence=str(evidence),report=str(report),
        usage=str(reports/f'USAGE_{t}.md'),source_domains=cfg['source_domains'],source_audit=cfg['source_audit'],
        geometry_GT_is_synthetic=True,real_physical_GT=False,training_started=False,Val_frozen_hashes=str(sorted(qpath.parent.glob('VAL_FREEZE_HASHES_*.jsonl'))[-1]))
    save(ROOT/f'DATASET_ENTRYPOINTS_MIXED_{t}.json',entry)
    usage=f'''# Mixed640512直接读取与评分

机器入口：`{ROOT}/DATASET_ENTRYPOINTS_MIXED_{t}.json`。环境：`match`；不要运行训练脚本。

## 读取一对

```bash
/home/yl/miniconda3/envs/match/bin/python -B {ROOT}/code/reader.py {release['manifests']['train']} --index 0
```

```python
import sys,json
from pathlib import Path
root=Path({str(ROOT)!r})
entry=json.loads((root/'DATASET_ENTRYPOINTS_MIXED_{t}.json').read_text())
sys.path.insert(0,str(root/'code'))
from reader import FixedPairs,tensors
ds=FixedPairs(entry['manifests']['train'])
z=ds[0]                       # images、masks、H、points0/1均显式分开
images=tensors(z)['images']    # [2,1,512,512]或[2,1,640,640]，浮点[0,1]
```

训练接现有M2/B监督时使用`z`字典。`local_descriptor_eligible`仅≥4点才开描述子监督，空点合法跳过；`global_eligible`另行要求旧16点与空间分布门槛。`ship_aux`此版关闭，不用框补船体点。混合尺寸按尺寸分桶，或DataLoader一次一对，不把512 resize成640。已有`evaluate22.Model.extract`的≤512包装限制应在新实验适配，而不是改写旧实验；原卷积网络和本版张量均已实际通过640 CPU前向。

## 纯图像预测与评分

`inference.py`只接受两张图像路径，不读取manifest、H或掩膜：

```bash
/home/yl/miniconda3/envs/match/bin/python -B {ROOT}/code/inference.py IMAGE_A.png IMAGE_B.png
```

外部算法每行预测JSON至少提供`pair_id`、`accepted`、`H_AB`（没有则null）。可选`matches_A/matches_B`、`keypoints_A/keypoints_B`，均为原输出像素xy，不是缩放显示坐标。预测文件必须保留失败/拒绝；遗漏的pair也会计作无输出失败。例：

```json
{{"pair_id":"MIX_val_00000","accepted":false,"H_AB":null,"matches_A":[],"matches_B":[]}}
```

```bash
/home/yl/miniconda3/envs/match/bin/python -B {ROOT}/code/score.py {release['manifests']['val']} predictions.jsonl --output-dir /tmp/mixed_scores
```

分层指标自动按512/640、场景语境、重叠、角度和增强分别输出。几何Success基于合法公共域网格；StructureSuccess只看旧门槛合格结构子集；误差归一化除以图像对角线。结构Precision/锚点Recall/实际检测重复性及无重叠FPR定义见构建报告；无结构不被误当无重叠。

## 重放、恢复和文件

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/yl/miniconda3/envs/match/bin/python -B {ROOT}/code/replay.py {release['manifests']['train']} {release['manifests']['val']}
```

正式生成命令（仅用于恢复，现已完成；不要重新创建冻结计划）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/yl/miniconda3/envs/match/bin/python -B {ROOT}/code/build.py build --config {cfgpath} --workers 4
```

每对在`pairs/{{train,val}}/PAIR_ID/`下有A/B PNG、完整GT NPZ及不可覆盖原始生成记录。规范manifest另行记录实际操作与skip，不改原始记录。父源坐标、邻接发布图、文件SHA、偏移、分区、种子、增强参数均可由manifest和source审计追溯。NPZ中的`check_points`只用于解析几何检验，不自动当结构描述子正例；`structure_points`才是已保存的保守合成结构监督，仍不是真实跨观测物理GT。

所有报告/统计文件尾缀为北京时间分钟；已有文件不覆盖。修改采样分布或扩展数据需新版本，不能基于Val模型分数反复调整本版。
'''
    with (reports/f'USAGE_{t}.md').open('x') as f:f.write(usage)
    print(json.dumps(dict(entrypoint=str(ROOT/f'DATASET_ENTRYPOINTS_MIXED_{t}.json'),report=str(report),usage=entry['usage']),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
