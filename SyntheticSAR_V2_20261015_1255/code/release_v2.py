"""Write concrete final handoff from completed build, disk QA and interface evidence."""
from common_v2 import *
from collections import Counter
import argparse,csv,platform,subprocess

def mdtable(headers,rr):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rr)
def write(p,s):
    with Path(p).open('x',encoding='utf8') as f:f.write(s)
def main():
    p=argparse.ArgumentParser();p.add_argument('summary',type=Path);args=p.parse_args();t=stamp()
    build=json.loads(args.summary.read_text());cfg=json.loads(Path(build['config']).read_text());distpath=sorted(ROOT.glob('DISTRIBUTION_SUMMARY_*.json'))[-1];dist=json.loads(distpath.read_text())
    geompath=sorted(ROOT.glob('QA_GEOMETRY_*.json'))[-1];geom=json.loads(geompath.read_text());ipath=sorted(ROOT.glob('INTERFACE_QA_*.json'))[-1];interface=json.loads(ipath.read_text());assert geom['passed'] and interface['passed']
    assert geom['read_pair_records']==dist['qa_after_usable_pair_records']
    rr=rows(build['all_generated']);usable=[r for r in rr if r['qa_state']=='usable'];rrows=[r for r in usable if r['arm']=='V2-R'];manifests=build['manifests']
    qa=rows(sorted(ROOT.glob('DISK_QA_*.jsonl'))[-1]);pixels={(r['sample_id'],r['arm']):r['calibration_pixels'] for r in qa}
    qmap={(r['sample_id'],r['arm']):r for r in qa};count_corrections=0
    for r in usable:
        q=qmap[(r['sample_id'],r['arm'])]
        count_corrections+=r['visible_ship_counts']!=q['valid_visible_ship_counts']
        r['visible_ship_counts_before_validity_margin']=r['visible_ship_counts']
        r['visible_ship_counts']=q['valid_visible_ship_counts'];r['visible_ship_instances']=q['valid_visible_ship_instances']
        r['visible_ship_count_definition']='all annotation instances with >=1 pixel inside conservative valid view; partial ships included'
    updated={(r['sample_id'],r['arm']):r for r in usable}
    manifests={k:str(ROOT/f'DELIVERY_{k.replace("/","_")}_{t}.jsonl') for k in build['manifests']}
    for k,newpath in manifests.items():jsonl(newpath,[updated[(r['sample_id'],r['arm'])] for r in rows(build['manifests'][k])])
    common_updated={};common_all={};common_ssdd={}
    for arm,oldpath in geom['common_validation_manifests'].items():
        selected_common=[updated[(r['sample_id'],r['arm'])] for r in rows(oldpath)]
        for pool,mapping in [('core',common_updated),('all',common_all),('ssdd',common_ssdd)]:
            newpath=ROOT/f'DELIVERY_COMMON_VAL_{arm}_{pool}_{t}.jsonl'
            jsonl(newpath,[r for r in selected_common if pool=='all' or (r['source_dataset']=='ssdd')==(pool=='ssdd')]);mapping[arm]=str(newpath)
    role_manifests={};role_counts={}
    for arm in cfg['arms']:
        for split in ['train','val']:
            for pool in ['core','ssdd','all']:
                selected=[r for r in usable if r['arm']==arm and r['split']==split and (pool=='all' or (r['source_dataset']=='ssdd')==(pool=='ssdd'))]
                for role in ['calibration_only','trainable','diagnostic_only']:
                    def eligible(r):
                        positive=r['loss_role']=='structural_positive';calibration=sum(pixels[(r['sample_id'],arm)])>0
                        return (calibration and not positive) if role=='calibration_only' else positive or calibration if role=='trainable' else not positive and not calibration
                    selected_role=[r for r in selected if eligible(r)];path=ROOT/f'{arm}_{split}_{pool}_{role}_{t}.jsonl';jsonl(path,selected_role)
                    role_manifests[f'{arm}/{split}/{pool}/{role}']=str(path);role_counts[f'{arm}/{split}/{pool}/{role}']=len(selected_role)
    src=rows(cfg['source_pool']);online_sources={}
    for pool in ['core','ssdd']:
        selected=[r for r in src if r['split']=='train' and (r['dataset']=='ssdd')==(pool=='ssdd')]
        path=ROOT/f'ONLINE_{pool}_ALL_TRAIN_SOURCES_{t}.jsonl';jsonl(path,selected);online_sources[pool]=str(path)
    online_sources['core_structural_recommended']=interface['online_source_pool']
    v1entry=json.loads((V1/'DATASET_ENTRYPOINTS_20261015_0044.json').read_text())
    protected=json.loads(next(ROOT.glob('PROTECTED_INPUT_IDENTITY_*.json')).read_text());changed=[p for p,h in protected.items() if sha(p)!=h];assert not changed
    sizes=Counter(r['width'] for r in rrows);angles=Counter(r['angle_bin_2D'] for r in rrows if r['profile']!='none');positiveN=sum(angles.values())
    scenes=Counter(r['source_scene_category'] for r in rrows)
    final=dict(version=cfg['version'],created_at=now(),trainability='可训练：M2用结构MATCH；B/检测校准用trainable并通过loss_bridge消费掩膜；其他只做诊断',
        recommended_arm='V2-R',config=build['config'],fixed_manifests=manifests,role_manifests=role_manifests,role_counts=role_counts,
        original_V1_standard_validation={k:v for k,v in v1entry['synthetic_manifests'].items() if k.startswith('val_')},original_V1_adapter=v1entry['pair_adapter'],
        V2_common_structural_validation=common_updated,V2_all_hard_validation={a:manifests[f'{a}/val/core'] for a in cfg['arms']},
        V2_common_structural_validation_all=common_all,V2_common_structural_validation_ssdd_extension=common_ssdd,
        fixed_reader=str(ROOT/'code/dataset_v2.py'),online_generator=str(ROOT/'code/dataset_v2.py'),online_sources=online_sources,
        legacy_B_data_bridge=str(ROOT/'code/loss_bridge.py'),validation_protocol=str(ROOT/'VALIDATION_PROTOCOL_20261015_1307.md'),
        build_summary=str(args.summary.resolve()),disk_QA=str(geompath),interface_QA=str(ipath),distribution=str(distpath),
        images_are_model_inputs_only=True,physical_multiview_GT=False,noise_is_display_domain_proxy=True,shared_original_speckle_eliminated=False,
        dataset_fully_materialized=True,network_training_started=False,old_checkpoint_or_training_config_modified=False,
        visible_count_margin_corrections=count_corrections,
        count_note='per-arm pairs/views, not three independent sources; original acquisition identity unresolved',
        source_split_note='known parent/pixel components only, prior validation observed, not a new unseen final test')
    entry=ROOT/f'DATASET_ENTRYPOINTS_V2_{t}.json';save(entry,final)
    counts=dist['counts'];ct=[c for c in counts if c['arm']=='V2-R'];Rall=sum(c['pairs'] for c in ct if c['pool']=='all');Rcore=sum(c['pairs'] for c in ct if c['pool']=='core')
    Rmatch=sum(c['match_positive'] for c in ct if c['pool']=='all');Rmatchcore=sum(c['match_positive'] for c in ct if c['pool']=='core')
    Rtrainable=sum(role_counts[f'V2-R/{s}/all/trainable'] for s in ['train','val']);Rcalib=sum(role_counts[f'V2-R/{s}/all/calibration_only'] for s in ['train','val'])
    scene_names=dict(offshore_vessels_low_texture_sea='离岸舰船/低纹理海面语境',nearshore_vessels='近岸舰船语境',port_quay_dense_vessels='港口/码头/密集舰船语境',other_land_structures='其他陆地结构语境',unclassified_background='未分类背景')
    numbertable=mdtable(['池','划分','独立文件数¹','已知组件','配对','视图','结构正例'],[[c['pool'],c['split'],c['original_images'],c['components'],c['pairs'],c['views'],c['match_positive']] for c in ct])
    scene_table=mdtable(['来源场景语境','Train','Val','总关系'],[[scene_names[s],sum(r['split']=='train' and r['source_scene_category']==s for r in rrows),sum(r['split']=='val' and r['source_scene_category']==s for r in rrows),n] for s,n in scenes.items()])
    angle_table=mdtable(['二维角度','非负几何关系','实际占比','结构正例'],[[b,angles[b],f'{100*angles[b]/positiveN:.2f}%',sum(r['angle_bin_2D']==b and r['loss_role']=='structural_positive' for r in rrows)] for b in ['00_15','15_30','30_60','60_90','90_120']])
    countroles=Counter(r['loss_role'] for r in rrows)
    v1aud=json.loads(next(ROOT.glob('V1_AUDIT_*.json')).read_text());qreasons=Counter(why for r in rr if r['arm']=='V2-R' and r['qa_state']!='usable' for why in r['qa_failures'])
    examples='\n\n'.join('![V1-matched、V2-R、V2-S同几何对照；橙色为ignore，绿色为结构锚点，不作真实物理GT](%s)'%Path(p).name for p in dist['examples'])
    commonval=len(rows(common_updated['V2-R']));tensorN=len(interface['tensor_samples'])
    report=f'''# 合成SAR训练数据集V2实际交付

生成时间：{now()}。版本{cfg['version']}。权威入口：[{entry.name}]({entry.name})。

**判断：可训练。推荐先用V2-R的core结构MATCH配对及core结构来源在线采样开展M2/B受控训练；S只作同预算消融。本轮没有启动训练。** CPU前向通过不代表学习收益或真实跨方位泛化已被证明。旧训练launcher保持冻结，应在算法方的新运行目录接入本次reader；B使用本次数据兼容桥，不能原样调用其256专用标签辅助函数。

## 实际规模与可训练含义

单独V2-R已落盘并通过全量读取验收：**{Rall}配对、{2*Rall}视图实例**；core单独{Rcore}配对、{2*Rcore}视图。不是靠三组相加达到10000。原始不同文件{dist['distinct_original_images']}个，均来自既有来源；新下载0、新独立雷达采集0。三组共用{dist['unique_geometry_relations']}个几何关系，合计{dist['qa_after_usable_pair_records']}臂配对记录/{2*dist['qa_after_usable_pair_records']}视图实例，不能计作三倍原始图像。

{numbertable}

¹ 此处为经既有像素重复修复后的不同原始文件，不是已证明独立的真实采集。同一source多次增强不增加原图数，同源重复增强全部仍在同一划分。core与SSDD以及all是子集/合集关系，不能叠加。

匹配正监督只有**{Rmatch}配对/{2*Rmatch}视图**，其中core {Rmatchcore}配对；其余不是匹配正例。R的角色计数：{dict(countroles)}。另有{Rcalib}配对可提供有范围约束的低纹理/海面检测校准，正例或校准合计{Rtrainable}配对/{2*Rtrainable}视图。其余保留诊断或拒绝研究，不应作为精确匹配标签输入M2。`trainable`中的校准样本不会给M2描述子损失制造点，需使用能消费校准区的B/检测损失。

## 场景与原生输入尺寸

{scene_table}

海事来源语境比例{100*sum(n for s,n in scenes.items() if s in ['offshore_vessels_low_texture_sea','nearshore_vessels','port_quay_dense_vessels'])/Rall:.2f}%。港口与陆地标签经现有599张来源预览的有界AI审查，其中369项作出明确语境标签，仍是AI暂定语义，不是人工场景GT，也不保证每个最终裁片都有码头或舰船。裁片内至少一侧有标注舰船{sum(any(r['visible_ship_counts']) for r in rrows)}对，双侧都有{sum(all(r['visible_ship_counts']) for r in rrows)}对；按所有标注实例在视图中至少1像素计数，包括截断舰船。未标船风险仍在。只有SSDD使用已有SL-SSDD像素海陆掩膜；HRSID/背景海面比例保持null，不能用场景语境代替像素海面比例。

尺寸关系数：{dict(sorted(sizes.items()))}。均≤512、32倍数。请求尺寸256/384/512；小来源及分离负例按原图尺寸/几何余量降到更小输入，源图从未先上采样。合成缩放、旋转和双线性插值属于显式H几何，不等于恢复更高原始分辨率。模型推理输入只有两幅灰度SAR显示图。

## 几何覆盖与可解性

{angle_table}

上述分母为{positiveN}个有解析重叠的几何关系，包含低/无结构，不把它们全部叫可配准正例。无重叠{sum(r['profile']=='none' for r in rrows)}对单列。V1公布的小角4528含319负例；正例小角实际4209/5629，V1的60–120为238。V2目标20/20/30/20/10在生成前固定，不按任何模型成功率调节。

正常重叠.50–.90，低重叠.12–.30，每幅保守有效区≥90%；高角使用较窄尺度，透视限尺寸.002。**这是二维图像角度，不是雷达观测方位。** 真正可训练结构正例须有双方≥16个持久结构锚点、至少4个4×4格、凸包≥5%和协方差特征值比≥.03。只有一个转角、直线或海纹不能靠解析H自动达标。

## 外观增强与监督修正

已核查全部5948个既有使用来源的图像头：HRSID4831 JPEG RGB、背景357 PNG RGB、SSDD760 JPEG RGB，均为8-bit发布/显示表示。可用元数据不能恢复逐图幅度、线性强度或dB到DN的映射。V2没有直接在这些DN上套用Gamma强度乘法物理模型，也没有估计真实ENL。

R使用显示域粗结构/细纹理分解：sigma .8低通，原始细纹理按.65–.85系数部分替换；粗边界保护让强结构附近保留更多残差。独立标准化Gamma随机场与局部残差RMS绑定、标准差上限.06 DN，加入轻度gamma/gain/contrast、平滑局部杂波和各向异性响应。它不是潜在物理反射率估计；仍可能平滑小目标/改变定位，原始共享散斑仍残留。Gamma场的形状8–24仅是代理参数，不是实测视数。

初版R试制在所查海面样例中共享高频相关性偏高，故保留`fixed_20261015_1303`为非推荐试制，未作为最终交付或额外独立数据计数。修订烟测6例中5例相关性低于V1-matched，另1例更高；该诊断含真实场景纹理和插值影响，不能证明独立观测或完全去相关。增强参数未使用M2/B推理结果。

S在同一R随机序列后加有界软响应增强/衰减及少量响应缺失，每视图最多3.5%原始变化区域；变化区再扩5像素并双向投影排除对应侧，更新ignore/visibility/static。所有臂还可对已标舰船做独立的软响应衰减，实际执行率见统计；这是动态可见性代理，无伪造船位GT。所有已标舰船及保护区、动态区、增强ignore均被排除。固定点数不够时清空匹配监督，不填充随机海面点。

`safe_static`表示排除已知船/动态后的安全区域，**不是结构GT**；`structure_support`是来源多尺度角点支持；`static`是已通过双方空间支持和可见性条件的匹配监督区。低监督样本static和points置空。B桥使用`static`过滤正点，并把ignore及无监督未知区域纳入legacy排除掩膜；可靠海面校准区单独消费，不能把所有非角点地物当成dustbin负例。结构锚点仍是合成训练代理，不能迁移为真实独立物理GT。

## 实际验收与隔离

尝试{build['attempted_relations']}个几何关系，成功构建{build['generated_relations']}个，生成失败{build['geometry_failures']}个。三臂QA前{len(rr)}配对记录/{2*len(rr)}视图；共同输入门控隔离{len(rr)-len(usable)}配对记录，对应{(len(rr)-len(usable))//3}个关系；QA后{len(usable)}条配对记录。隔离原因计数（单臂字段，可一例多因）：{dict(qreasons)}。不删除失败、不替换来源凑数。初次烟测曾因private函数import遗漏失败，修复后烟测和全量QA均保留，不冒充没有工程失败。

实际读回**{geom['read_pair_records']}条配对记录/{geom['read_view_instances']}幅图**，验证H/逆H、重新构造源有效区、双线性四邻域、静态/船/ignore/可见性、精确重叠和空间支持，全部通过；最大解析对应误差{geom['max_point_error']:.8f}px。这个数只是合成坐标数学误差，完全不是SAR物理对应误差。新数据读回时记录像素/掩膜摘要，不重新全量SHA旧原图。

实际tensor读取{tensorN}例，覆盖Train/Val、三臂、全部profile、尺寸和低/无结构、动态/ignore。现有M2实际CPU损失前向通过；Model-B实际网络与桥在256/384/512、负例、无结构和S变化区域等6例通过。优化器步数0、GPU使用0、权重写入0。在线{interface['online_success']}次成功调用完成图像、H及掩膜精确回放，Val来源被拒绝；日志保留每次索引/epoch/visit/种子/来源SHA/参数/结果，拒绝不静默重抽。

Train/Val已知组件{geom['component_counts']}、交集0；原始文件交集0；原修正版test已知组件保持封存。HRSID父名及已知像素关联并不证明所有原始采集/地理场景独立；来源预览也可见相似港区语境，未建立完整跨采集身份链。SSDD仅作单列扩展。历史V1、900、人工记录、真实冻结512、旧训练代码/配置/权重均未改；已选择的{len(protected)}个保护入口/代码摘要前后不变。没有新下载、真实数据扩展或网络训练。

## 固定验证与算法交接

V1标准验证按旧入口原样引用；V2困难Val保存三臂全关系及分层。新增Val仍是已观察开发来源，不是未见最终测试。三臂共同合格结构Val **{commonval}对**的ID交集已在模型运行前冻结；匹配精度对照使用同一分母，其余低结构样本保留诊断，不能自动称无重叠负例。每臂自身合格数作为补充，不允许只用S更容易的子集比较。SSDD结果单列。

交给算法Codex的最简步骤见[DATASET_USAGE](DATASET_USAGE_{t}.md)。直接读取入口中的`fixed_manifests['V2-R/train/core/match']`、`V2_common_structural_validation['V2-R']`，或`online_sources['core_structural_recommended']`。校准任务使用`role_manifests`，不要把全部可读数据视作有匹配正标签。R与V1-matched先作相同来源/基础几何/采样种子的受控实验；S在此后按同一预算比较。本轮没有训练或选模型，不能预先承诺V2优于V1或改善真实跨方位精度。

## 对照图

{examples}

图中绿色为自动多尺度结构锚点，橙色为ignore；展示缩略图，训练仍读取原生生成尺寸。图像来自真实SAR发布裁片的二维合成，不是两个真实SAR采集，也不是人为标点。详细增强参数和依据见[AUGMENTATION_AUDIT](AUGMENTATION_AUDIT_{t}.md)；实际交叉统计CSV随目录保存。
'''
    write(ROOT/f'DATASET_V2_REPORT_{t}.md',report)
    audit=f'''# V1/V2增强机制审计

时间：{now()}；配置：`{build['config']}`，代码`code/dataset_v2.py`、`build_v2.py`。所有概率是数据机制，不是真实SAR观测方向分布。

## V1实际机制

复用已有审计，V1可用5948关系11896视图不变；各profile实际数量：{v1aud['v1_profiles']}。easy/medium/rotation/low/none/dynamic名义权重35/30/15/10/5/5%，源ID稳定哈希选择。easy单幅角±5、scale .95–1.05；medium±18、.85–1.15、shear±3、perspective .005；rotation±60、.9–1.1；low单幅±12、shear2、perspective .003、overlap .08–.25；none单幅±3、overlap0；dynamic±10、overlap.6–.9。双图单幅角度差的分布不是每幅角范围本身。头部余量及拒绝采样改变最终可行分布。

warp后每视图独立gamma→中心.5的对比度+亮度偏置→均值1 lognormal乘噪声→clip→可选Gaussian blur→uint8。easy gamma/contrast .95–1.05、offset±.03、sigma .07、不模糊；medium gamma/contrast .85–1.15、offset±.08、sigma .18、模糊概率.18/sigma .1–.85。rotation/low/dynamic沿用medium外观，none沿用easy。两图各自随机数来自同一确定性RNG顺序不同抽样，原始散斑仍共享。dynamic可用275关系实际214移动，近似inpaint/cut-paste，未执行实例不能算实际移动。

原生成器的可见实例阈值会漏部分截断船，V1后处理已修正全标注并5像素保护，额外修正1097对。V2直接复用annotation_mask读取完整多边形/RLE，失败时诚实bbox代理，不设最低可见比例后再决定是否排船。所有source实例都参与union，裁片计数另按实际可见像素。

## V2-R：显示域结构/纹理代理

当前数据没有逐图可逆物理辐射映射，因此不对DN做“Gamma强度物理仿真”。处理顺序：原生源warping→受限船响应动态代理→显示域粗/细分解与部分残差移除→独立gamma/gain/contrast/offset→独立Gamma形状随机纹理残差替换→平滑局部杂波→轻度各向异性响应→可选S软散射变化→clip+uint8→有效区外置0。三臂基础几何/源/动态种子一致，R/S的全局外观随机数一致，S只在末尾加局部变化。

参数以锁定JSON为准：

```json
{json.dumps(cfg['appearance'],ensure_ascii=False,indent=2)}
```

Gamma随机场先按形状8–24生成、中心化/标准化，再混入sigma .35–.65空间相关分量（混合权重0–.5），随后标准化为单位方差。DN噪声幅度为基础.008–.016与部分移除残差局部RMS之和，上限.06；不是物理乘性Gamma模型，不把该形状叫实测ENL。纹理分解sigma .8，系数.65–.85，粗sigma2 Sobel边界保护，强边界的细纹理移除权重下限.25。局部杂波std .012、限±.03；各向异性Gaussian sigma每轴.15–.50。

原始图像小块显示DN统计保存在NATIVE_DISPLAY_PATCH_STATS；包括不同来源/场景的64像素窗口和高频std，未做海面物理ENL拟合。参数是受这些显示尺度及质量检查约束的工程范围，不是从真实独立多时相配对反演出的分布。没有读取新真实Native来标定物理噪声，也不假定海况/雷达分辨率完全同分布。

## V2-S与动态区域

每视图触发软局部变化概率.65，仅从有效内部亮响应中选1–3个Gaussian区域，sigma3–6、gain .55–1.35；其中响应缺失概率.3/gain .35–.55。单视图原始变化区总面积≤3.5%，再扩5像素排除，投影到另一图亦排除；多blob可能因面积上限不执行，实际执行数量以manifest为准。Gaussian权重>.05作为局部显著变化域，域外尾部只存在很弱辐射变化，不能把整个图视为完全未改变。没有硬矩形贴片、完全黑块或周期拼接。

三臂共享概率.2的右图已标船软衰减（含边缘平滑），不伪造船运动坐标、不提供舰船静态GT。只对确实可见标注船执行；实际事件及面积保存。局部结构损坏/丢失后，static和点集合同步收缩，低于16/空间门槛归低监督并清空正点。原始未标船、建筑散射、源影像压缩和相似地理场景仍是残余风险。

## 场景、几何与强度验收

固定profile按seed抽样：none4%，low10%，projective30%，其余affine56%；角度对所有非负profile目标20/20/30/20/10，高角尺度.96–1.02，其他.92–1.06；透视只在projective .002。输出按源原生余量自适应，不预放大。船源排除10px+视图5px，持久角点sigma2/4、距离12、一致距离≤4.5、绝对eig≥.00008和粗局部std≥.025；SSDD像素海面否决，粗离岸/未分类背景不监督匹配。

接受门槛在构建前固定：有效区std≥.5灰度、255饱和占比≤30%，任一臂失败三臂共同隔离，保留原始失败。不会因为原图存在很多0值自动把暗海删除，也不会靠噪声随机点增加锚点数量。数学往返不是结构/物理正确性。可视抽查仅是AI工程审查，不是人员独立GT审查。

## 依据与物理边界

HRSID官方说明区分JPG发布版、更高保真PNG及400背景，不能据此恢复本地裁片的物理DN函数。[HRSID官方](https://github.com/chaozhong2010/HRSID)。SSDD/SL-SSDD身份与海陆编码来自本地README及[SSDD官方](https://github.com/TianwenZhang0825/Official-SSDD)、[SL-SSDD官方](https://github.com/Han-Ke/SL-SSDD)。

SAR强度域乘性Gamma模型有统计依据，但适用对象是强度观测，不是任意压缩显示灰度；本项目选择保守的显示统计代理。[强度统计原始研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC9024813/)。散斑降噪是有损推断，不能通过一幅原图恢复两次真正独立采集。[MuLoG原始研究](https://arxiv.org/abs/1704.05335)、[ESA SNAP Multilook说明](https://step.esa.int/main/wp-content/help/versions/10.0.0/snap-toolboxes/eu.esa.microwavetbx.sar.op.sar.processing.ui/operators/MultilookOp.html)。这些来源仅支持物理边界/统计背景，不为本次DN参数提供物理标定。

旧V1-matched对照在新几何上固定使用旧medium外观，不声称复刻旧profile混合分布。原历史V1数据和标准验证入口另行完整保留。任何合成训练收益必须由算法方固定验证检验，无法预先从噪声更强或loss下降推断。
'''
    write(ROOT/f'AUGMENTATION_AUDIT_{t}.md',audit)
    usage=f'''# 给算法Codex：Synthetic SAR V2最简用法

**可训练。默认V2-R core；先用MATCH训练M2；B通过loss_bridge消费匹配/校准及ignore。不要修改/重启历史冻结run。**

权威机器入口：`{entry}`。环境match，输入均原生生成尺寸≤512；batch1或按shape分桶，不隐式resize。当前GPU训练0，本轮不提供自动启动网络训练的shell。

## 固定M2监督数据

```python
import sys,json
from pathlib import Path
sys.path.insert(0,{str(ROOT/'code')!r})
from dataset_v2 import load_training_pair,tensors,load_inference_images
entry=json.loads(Path({str(entry)!r}).read_text())
train=[json.loads(s) for s in Path(entry['fixed_manifests']['V2-R/train/core/match']).read_text().splitlines()]
val=[json.loads(s) for s in Path(entry['V2_common_structural_validation']['V2-R']).read_text().splitlines()]
z=load_training_pair(train[0])
batch=tensors(z,device='cpu')  # images:[2,1,H,W] float32[0,1]
# existing M2 model.loss(model,z,device,amp=...) accepts this dictionary.
# Model forward input only: batch['images']; H/valid/static/ignore are targets.
images=load_inference_images(train[0]) # ONLY two uint8 arrays, no GT or location
```

字段沿用`images,valid,static,ship_masks,H_A,H_B,H_AB,points0,points1,overlap,source,source_origin_xy,regime,seed,dynamic_occlusion`。`H_A/H_B`为完整原图→生成图，origin=[0,0]，不能再扣一次裁片偏移。新增`ignore,visibility,safe_static,structure_support,calibration_mask,loss_role`明确传递；其中static已包含结构及成对可见性约束。不要用safe_static替代static，不要从完整解析网格重造随机正点。

`fixed_manifests['.../core']`为全部可读关系；`.../core/match`仅结构正例；`role_manifests['.../core/calibration_only']`用于低纹理检测校准，不能用于M2描述子匹配；`.../diagnostic_only`不当作有GT训练样本。`.../ssdd`单列扩展，只能声称已知像素组件隔离。图内船数含≥1像素的截断标注实例，不等于可用于配准的静态目标数。

## 在线训练与精确回放

```python
from dataset_v2 import OnlinePairs,replay
from common_v2 import stamp,rows
# 每次实验/每个worker用独立日志路径；构造器以x模式保护旧日志。
online=OnlinePairs(entry['config'],entry['online_sources']['core_structural_recommended'],
                   f'/tmp/sar_v2_worker0_{{stamp()}}.jsonl',arm='V2-R',seed=20261015)
z=online.get(index=0,epoch=0,visit=0)
another=online.get(index=0,epoch=1,visit=0) # 新几何/外观，仍来自Train组件
# 批处理保持(index,epoch,visit)调用日志；异常已记拒绝原因，不能偷偷重抽掩盖。
# 独立重放：先核对日志首行config/source/code SHA和release身份，再：
events=rows(online.log_path);record=next(r for r in events if r.get('state')=='usable')
replayed=replay(record,online.sources[record['index']],online.cfg)
```

`core_structural_recommended`包含{interface['online_source_count']}个实际出现合格结构的Train原始文件，只做模型无关筛选；不是新增独立采集。在线每次仍可能因新裁片/低重叠变为limited/no-structure，points置空，M2应显式跳过其描述损失，不能补随机点。`core`全Train来源另外给出，可为检测校准采样；不要把海面解析对应当稳定锚点。每个worker自己构造OnlinePairs，使用不同日志，调用序号由训练run分配并保存；日志包括失败，包含来源像素文件SHA及H、外观参数、各图/掩膜摘要。Val不能用OnlinePairs；固定Val不重增强。

受控V1-matched/R/S使用相同来源调度、(index,epoch,visit)、全局seed和基础几何。所有臂使用同一共同输入门控；不可用不同重抽次数给某臂换容易的样本。R/S共享全局外观随机数，S额外改变局部可见性。日志是运行产物，不混进固定manifest。

## Model-B数据兼容

```python
from loss_bridge import legacy_B_loss
# existing Model-B network/teacher, in a NEW algorithm experiment:
loss,info=legacy_B_loss(model,teacher,z,device='cpu')
```

桥仅读取旧`network_corrected.py`中的joint_loss并在新命名空间编译：替换固定384源窗口结构标签为V2已保存points，替换256归一化为实际H/W，device显式传入。网络参数/旧源码不改。ignore和无监督未知区域进入legacy排除掩膜；物理船掩膜保留在`all_vessel_masks`。这样既不会把局部消失结构继续当正例，也不会把所有无角点陆地区当负例。旧损失权重仍保留；新的标签/掩膜口径是V2实验因素，所有受控臂一致应用，不与历史B分数直接混算。CPU实际网络6种情形前向已通过，未执行优化器或GPU训练。

## 验证及回收结果

固定V1标准Val路径在`original_V1_standard_validation`，读取使用`original_V1_adapter`，不修改旧GT/图像/掩膜。新V2三臂全Val及共同结构Val分别在入口中；共同分母{commonval}关系，先固定再训练，不能只挑S自身通过的子集。新Val仍是开发来源，原始采集独立性不完整。匹配误差只在结构支持充分的合成正例上报告，低/无结构单独报告拒绝/响应，不把它们冒充无重叠负例。历史900不训练、不选模、不重跑。

算法run需保存本次entry/config/code SHA、起始checkpoint身份、来源调度、完整online日志、seed、旧V1和新Val逐对预测/失败与分层结果。不在本数据目录写checkpoint或覆盖QA。数据集可直接读取不等于模型一定有效；出现无结构/空对应须合法跳过，不能视为训练故障后自动补点。

QA命令（新检查写新时间文件；避免重复全量审计已完成输入）：

```bash
cd {BASE}
CUDA_VISIBLE_DEVICES='' /home/yl/miniconda3/envs/match/bin/python -B \\
  {ROOT}/code/verify_v2.py {args.summary.resolve()}
```

已完成的结果见`{geompath.name}`、`{ipath.name}`，无需算法方再生成一遍数据。所有报告/统计/采样日志文件尾缀用北京时间分钟；保留历史版本。无真实GT生成，无真实采集/冻结512/8772/8773服务变更。
'''
    write(ROOT/f'DATASET_USAGE_{t}.md',usage)
    # Final reproducibility identity: small manifests, code, protocol, and reused V1 val entrances.
    paths=[entry,Path(build['config']),Path(cfg['source_pool']),Path(args.summary),geompath,ipath,distpath]+list((ROOT/'code').glob('*.py'))
    paths += [Path(v) for v in manifests.values()]+[Path(v) for m in [common_updated,common_all,common_ssdd] for v in m.values()]
    paths += [Path(v) for v in role_manifests.values()]+[Path(v) for v in online_sources.values()]
    paths += [Path(v) for v in final['original_V1_standard_validation'].values()]
    paths += [BASE/'SAR_SP_LG_StructureStable_20261013_2210/code/network_corrected.py',BASE/'SAR_SP_LG_Train_20261012_1544/code/model.py',BASE/'SAR_SP_LG_Train_20261012_1544/code/data.py']
    paths += list((BASE/'MaritimeSARRegDataset/sarreg').glob('*.py'))
    save(ROOT/f'RELEASE_IDENTITY_{t}.json',dict(at=now(),sha256={str(p.resolve()):sha(p) for p in paths},protected_inputs_verified=len(protected),protected_changed=[],
        pixel_identity='DISK_QA stores decoded new-image/mask/points hashes; old imagery not rehashed',python=sys.version,numpy=np.__version__,opencv=cv2.__version__,
        no_training=True,source_isolation='known components only',initial_nonrecommended_trial=str(ROOT/'fixed_20261015_1303')))
    save(ROOT/f'FINAL_COUNTS_{t}.json',dict(at=now(),R_pairs=Rall,R_views=2*Rall,R_core_pairs=Rcore,R_core_views=2*Rcore,R_match_pairs=Rmatch,R_match_core_pairs=Rmatchcore,R_calibration_pairs=Rcalib,R_trainable_pairs=Rtrainable,
        all_arms_pairs=len(usable),all_arms_views=2*len(usable),different_original_files=dist['distinct_original_images'],different_acquisition_count=None,common_val_pairs=commonval,entrypoint=str(entry)))
    write(ROOT/f'README_{t}.md',f'# Synthetic SAR V2\n\n权威入口：[{entry.name}]({entry.name})。\n\n可训练；先使用V2-R core MATCH，B使用数据桥。R单臂{Rall}配对/{2*Rall}视图，其中{Rmatch}结构匹配正例，低纹理不补点。\n\n[结果报告](DATASET_V2_REPORT_{t}.md) · [增强审计](AUGMENTATION_AUDIT_{t}.md) · [直接读取与在线使用](DATASET_USAGE_{t}.md)。\n\n历史V1、旧试制和所有失败保留。推荐配置1309；不使用1258初版R作为正式交付。无训练、无下载、无新真实数据。\n')
    print(json.dumps(dict(entry=str(entry),report=str(ROOT/f'DATASET_V2_REPORT_{t}.md'),R_pairs=Rall,R_views=2*Rall,R_match_pairs=Rmatch,common_val=commonval),ensure_ascii=False))
if __name__=='__main__':main()
