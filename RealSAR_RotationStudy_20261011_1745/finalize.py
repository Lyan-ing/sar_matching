"""Focused acceptance, immutable-input protection, minute-stamped scientific report."""
import sys
sys.dont_write_bytecode=True
import json,csv,subprocess,shutil
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from study import ROOT,OLD,MANUAL,BASE,rows,cfg,save,sha,frozen

def table(headers,data):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in data])
def read(name):return json.loads((ROOT/'analysis'/name).read_text())
def verify(root,hashes,label):
    for i,(name,digest) in enumerate(hashes.items()):
        path=root/name
        if not path.is_file() or sha(path)!=digest:raise AssertionError('Frozen file changed: '+str(path))
        if (i+1)%10000==0:print(label,i+1,'/',len(hashes),flush=True)
    return {'files':len(hashes),'changed':0}
def main():
    config=cfg();prereg=json.loads((ROOT/'preregistration.json').read_text());assert prereg['config_sha256']==sha(ROOT/'config.json') and prereg['negative_index_sha256']==sha(ROOT/'negative_index.json');assert prereg['old_receipt_sha256']==sha(OLD/'ENGINEERING_ACCEPTANCE.json');freeze=frozen()
    reportdir=ROOT/'reports';reportdir.mkdir(exist_ok=False);stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    command=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(ROOT/'test_study.py')];test=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(reportdir/f'tests_{stamp}.log').write_text(test.stdout);print(test.stdout,flush=True);assert test.returncode==0 and 'skipped' not in test.stdout
    stages={s:json.loads((ROOT/'runs'/s/'SUMMARY.json').read_text()) for s in ['diagnostic8','representative24','full191','strategies','negative']};assert stages['full191']['records']==2292 and stages['negative']['records']==360 and stages['strategies']['strategy_rows']==1146;assert all(s['runtime_errors']==0 for s in stages.values())
    source=BASE/'real_sar_pilot/multisource512_20261010_2352';protection={'old_auto_labels':verify(OLD,json.loads((OLD/'ENGINEERING_ACCEPTANCE.json').read_text())['file_sha256'],'old_auto_labels'),'historical':verify(BASE,json.loads((source/'protected_artifacts.json').read_text()),'historical'),'real_source':verify(source,json.loads((source/'DELIVERY_COMPLETE.json').read_text())['file_sha256'],'real_source'),'annotation_app':verify(MANUAL,json.loads((MANUAL/'ENGINEERING_ACCEPTANCE.json').read_text())['file_sha256'],'annotation_app')};save(ROOT/'preservation.json',protection)
    packages=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True));save(ROOT/'environment.json',{'python':sys.executable,'packages':packages,'runtime_provenance':json.loads((ROOT/'runs/full191/provenance.json').read_text()),'new_packages_installed':False});(ROOT/'requirements.lock.txt').write_text('\n'.join(f"{p['name']}=={p['version']}" for p in packages)+'\n')
    snapshot=ROOT/'code_snapshot';snapshot.mkdir();code=[*ROOT.glob('*.py'),ROOT/'config.json',ROOT/'TASK.txt'];hashes={p.name:sha(p) for p in code}
    for p in code:shutil.copyfile(p,snapshot/p.name)
    save(snapshot/'SHA256.json',hashes)
    summary=read('summary.json');rotation=read('rotation_summary.json');recovery=read('recoveries.json');strata=read('strata.json');strategies=read('strategy_summary.json');negative=read('negative_summary.json');failure=read('failure_stages.json');zero=read('zero_rerun_consistency.json');figures=json.loads((ROOT/'figures/manifest.json').read_text());assert len(figures)==26
    completed=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=completed.strftime('%Y%m%d_%H%M');reportname=f'rotation_failure_report_{stamp}.md';logname=f'execution_{stamp}.log'
    counts=table(['方法','角度CCW','匹配≥8','矩阵可用','原门控接受','B一般/高候选合计','高B','同匹配H门控接受'],[[r['method'],r['angle'],r['matching_ge_8'],r['model_available'],r['gate_accepted'],r['B_general_eligible'],r['B_high'],r['H_gate_accepted']] for r in rotation])
    restoration=table(['方法','比较','恢复层面','0°数量','旋转候选并集','原失败恢复裁片/采集对','原通过丢失裁片'],[[r['method'],r['comparison'],r['outcome'],r['baseline'],r['candidate_union'],f"{r['recovered']['chips']}/{r['recovered']['acquisition_pairs']}",r['lost']['chips']] for r in recovery])
    costs=table(['样本池','方法','策略','接受/样本','B候选','中位ms','p90ms','中位尝试角数','实际总ms'],[[r['population'],r['method'],r['strategy'],f"{r['accepted']}/{r['pairs']}",r['B_eligible_after_selection'],round(r['time_ms']['median'],2),round(r['time_ms']['p90'],2),r['rotations']['median'],round(r['time_ms']['sum'],2)] for r in strategies])
    negatives=table(['方法','角度/并集','地理负样本误接受/30','经验比例'],[[r['method'],r['angle'],r['wrong_geographic_accept_pairs'],f"{100*r['empirical_rate']:.2f}%"] for r in negative])
    failures=table(['协议','方法','主失败阶段','预测记录数','不同裁片/采集对','关键点min中位','匹配中位','内点中位'],[[r['protocol'],r['method'],r['stage'],r['prediction_rows'],f"{r['chips']}/{r['acquisition_pairs']}",r['median_keypoints_min']['median'],r['median_matches']['median'],r['median_inliers']['median']] for r in failure if r['protocol']!='native_all_rotations'])
    mechanisms=table(['机制代理（非互斥）','方法×裁片记录','不同裁片','真实采集对','站点/组件','例子'],[[r['case'],r['method_pair_rows'],r['chips'],r['acquisition_pairs'],f"{r['sites']}/{r['components']}",', '.join(r['example_ids'])] for r in read('mechanism_summary.json')])
    hcases=table(['H通过/Affine拒绝样本','方法/角度','H有效内点','匹配/地图偏差中位px','H预测/地图格点差p90px'],[[r['sample_id'],f"{r['method']}/{r['angle']}",r['H_native_valid_inliers'],round(r['H_match_map_difference_median_px'],3),round(r['H_prediction_map_difference_p90_px'],3) if r['H_prediction_map_difference_p90_px'] is not None else '无充分有效支持'] for r in read('H_without_Affine.json')])
    def strat(key):return table([key,'方向假设','裁片/采集对/组件','任一算法门控裁片/采集对','B裁片/采集对'],[[r['value'],r['mode'],f"{r['chips']}/{r['acquisition_pairs']}/{r['components']}",f"{r['gate_chips']}/{r['gate_acquisition_pairs']}",f"{r['B_chips']}/{r['B_acquisition_pairs']}" ] for r in strata if r['stratifier']==key and r['method']=='ANY_ALGORITHM_EXPLORATORY_UNION'])
    locals=list(csv.DictReader((ROOT/'analysis/local_affine_diagnostics.csv').open()));localaccepted=[r for r in locals if r['basic_local_estimator_accepted']=='True']
    restored180={m:next(r for r in recovery if r['method']==m and r['comparison']=='180' and r['outcome']=='gate')['recovered']['chips'] for m in ['sift','orb','lightglue']}
    B180={m:next(r for r in recovery if r['method']==m and r['comparison']=='180' and r['outcome']=='B')['recovered']['chips'] for m in ['sift','orb','lightglue']}
    gate4={m:next(r for r in recovery if r['method']==m and r['comparison']=='four' and r['outcome']=='gate')['recovered']['chips'] for m in ['sift','orb','lightglue']}
    B4={m:next(r for r in recovery if r['method']==m and r['comparison']=='four' and r['outcome']=='B')['recovered']['chips'] for m in ['sift','orb','lightglue']}
    alt={m:sum(r['H_gate_accepted'] for r in rotation if r['method']==m) for m in ['sift','orb','lightglue']}
    recommendation=('没有新增B级支持，旋转搜索不宜作为当前真实SAR数据的默认模块。' if summary['new_B']['chips']==0 else '出现新增B级候选，但尚无独立GT证实，旋转搜索只能保留为待验证模块。')
    report=f'''# 真实SAR可配准性、失败阶段与固定旋转假设研究

完成时间：{completed.isoformat()}；新目录{ROOT.name}。这是已分析真实数据的探索性研究，不是新的封存外部测试。没有训练、下载、重裁片、参数搜索或历史标签改写。

## 核心结果与十个问题的回答

1. 180°恢复到原几何门控的原失败样本：{restored180}；恢复B资格：{B180}。这里的恢复仅为工程候选状态，不是真实GT验证的配准成功。
2. 四角度的算法内候选并集新增门控：{gate4}；新增B：{B4}。三算法×四角度探索性并集的总体新增门控单位{summary['new_gate']}；新增B单位{summary['new_B']}。算法并集不是部署策略，不能把选择已知好方向的结果冒充实际搜索。
3. 明确异向样本四角度B单位{summary['different_look_B']}，异向新增门控单位{summary['different_look_new_gate']}。Umbra/Capella分层见下表；来源、尺度、场景、时间与视角存在混杂，不能唯一归因为观测角度。
4. 30个已验证地理无重叠负样本上的各方向、任一方向误接受及实际策略误接受见负样本表。是否产生额外误接受必须看新增被接受的负样本，不能由接受率推断正样本真实正确率。这30对仍共享站点/采集，经验比例不是独立采集总体风险保证。
5. SIFT/ORB在本实验中较能保留整数旋转后的匹配，而SuperPoint+LightGlue表现出明显方向敏感性：诊断S1SF_EXTENSION_000__chip_000在0°有984匹配/682内点；90°、180°、270°只有47/52/25匹配、5/5/7内点，moving关键点仍1568/1476/1591。这是描述与匹配退化的证据，不是检测点数不足，也不能据此证明具体网络层的因果机制。
6. 原0°失败阶段见下表。关键点数量充足时，SIFT/ORB常停在描述子有效匹配不足；LightGlue常有匹配但缺少Affine一致内点。D类同时区分原full门控和B附加筛选；不能把所有不可靠预测归为产品定位偏差。
7. 同一匹配下Homography原full门控通过记录总数{alt}，对照Affine表及mechanisms.csv。只有候选局部支持而没有独立对应证据时，不能宣称Affine限制了真实正确结果；增加自由度不构成精度证明。局部诊断{len(locals)}个有≥8匹配的128像素tile，其中{len(localaccepted)}个通过原基本Affine估计条件，未用全幅full门控，不升级B，不外推到整图。
8. {recommendation}实际策略成本与收益已测量，见策略表。若只增加偶然模型或计算量而不增加可靠候选，不作为默认前处理；以后可在具有独立对应的方向失配案例上单独验证。
9. 下一步先核验可共同辨认的静态地物、原生/显示量化和跨观测散射变化，再研究SAR适配的描述与匹配及局部几何；在新隔离开发数据上设计预处理，不因本轮失败临时更改冻结算法。本实验只覆盖90°整数旋转，不能排除连续角度、尺度、局部结构等其他机制。
10. 当前C级仍0；19个旧B主要来自高覆盖相近观测，而异向自动监督不足。可以继续地图弱监督管道或小规模受控探索，但没有足够证据直接把自动标签作为真实异向网络的高可信监督。优先解决独立标签质量、地理场景/采集隔离及验证协议，再谈模型微调收益。

## 协议、固定规则与执行范围

191裁片、65真实采集对、58采集、12站点/组件。8诊断与24冻结代表样本重叠6个，实际不同裁片合计26；未重复计为32。逐阶段diagnostic8 → representative24 → full191；已完成小规模的实际坐标/基线检查后扩展，重叠阶段复用本轮相同旋转缓存。

四方向消融共有2292条原生协议记录（191×3×4），0°573条直接使用原缓存；90/180/270在整数旋转后的moving_native图上完整重新检测、描述、匹配、Affine/RANSAC与full门控，不是只旋转旧特征。reference保持不变。np.rot90定义逆时针，旋转无插值、原生DN不变。

像素中心编号0..511：90°为(xrot,yrot)=(y,511-x)，180°=(511-x,511-y)，270°=(511-y,x)。预测reference→rotated-moving矩阵左乘逆旋转矩阵后恢复到moving原生坐标；所有匹配、内点、支持和地图偏差均在原生坐标计算。不同CRS地图映射仍复用既有非线性逐点函数。栅格地图方向、SAR观测方位角与升降轨不是同一概念，升降轨反向不自动要求图像旋转180°。

冻结算法、LightGlue官方权重、seed42、原Affine/full门控和B条件完全复用。新配置seed261011、固定方向顺序、失败诊断代理、负样本选择规则在旋转结果前写入config.json/preregistration.json。B资格始终用原生有效内点、双图支持、地图偏差和原阈值；高候选要求至少两个不同算法家族一致，同一算法多个方向不重复计家族。新候选支持掩膜按原内点凸包、原生有效像素与共同地理范围生成；不是GT区域。

策略A按0→180，策略B按0→180→90→270，首个通过原图像侧门控即停止。没有读取GT、地图Mask、旧B状态或地图偏差来选角度。多方向消融可能有多个候选，但部署规则按既定顺序选择，不能挑事后最好的B角度。

实际运行一次B级联并在A的真实停止前缀记录墙钟时间，两个策略共用实际前缀；没有把分项缓存耗时相加，也不宣称两个独立重复计时。时间包含PNG解码、整数旋转、完整新特征提取、估计、门控、GPU同步和选择；地图/标签QA、额外Homography诊断及落盘在选择后另做，不在部署时间内，模型初始化也不摊到每对。结果保存实际执行角数与选中矩阵。

为真实端到端计时，搜索策略重新执行0°：与被冻结0°缓存的573条比较结果为{zero}。这次实际重跑只用于部署计时和一致性检查，原0°结果不覆盖。

地理辅助旋转扩展属于可选项，本轮优先完整原生消融、真实策略与负样本；未追加1719条地理辅助旋转推理。保留并复用原573条geo_assisted 0°记录，单独分析，不混入native方向对比。原地理辅助图仅由moving原生512像素产生；未引用full-scene旧粗对齐图的裁片外像素。

## 四角度结果和恢复层级

{counts}

{restoration}

matching_ge_8只是达到原几何估计最小匹配数，不代表这些匹配正确。矩阵可用、门控接受、B资格各自有独立状态；0°与180°表同时报告恢复和丢失，避免只看净数量。

原B集合{summary['baseline_B']}；四角度B集合{summary['four_B']}。新B ID：{summary['new_B_ids']}。原19个标签原样冻结；新分析结果不是对其版本的重写。

新增SIFT门控案例Capella_0001__chip_000为相近观测方向：90°有12匹配、8内点，原生有效内点比例1.0；与地图对应的点偏差median=46.707px、p90=47.984px，原B的median≤32条件未满足。保留这个被拒绝的候选及其矩阵，不改变地图对应或阈值。缺乏独立核验时不能区分定位偏差、重复结构或错误匹配，也不能说它已恢复真实正确配准。

## 失败阶段及几何证据

{failures}

{mechanisms}

机制代理使用运行前固定的匹配增长规则（至少翻倍且增加≥8），并保留可能同时符合多种情况的样本。E“All models fail”只指当前算法/模型均拒绝，物理结构原因仍未知；不同于失败阶段E的独立“没有静态共同结构”断言。CSV保留每个算法/样本的证据，代表图对应原sample ID。

A：min关键点<8或双图关键点凸包<0.01的检测/分布代理；B：有关键点但匹配<8；C：基本RANSAC估计失败、内点不足、低内点比例、空间退化或高拟合残差；D：基本几何之后的原full检查或B额外筛选未通过；passed_B仅为候选，不是成功GT。

这些是可观察的计算阶段分类，不自动等同于物理原因。E“缺少可确认共同静态结构”的独立确认样本数0；F“物理结构证据不足”覆盖191对，可能与A/B/C/D重叠；并非191对已经被证明没有静态对应。runtime_error属于计算证据缺失，实际运行异常总数为0。原0°和地理辅助0°分开，all_rotations的完整阶段表见CSV，不把多角度重复行当独立样本。

对26代表样本提供原生图、共同覆盖、地理重采样、透明/互补颜色叠加、32px棋盘格和64px局部显示放大。局部位置来自固定地理格点的双图显示方差与空间分散规则，仅为“结构候选”；没有物理散射中心或静态地物独立确认。道路、建筑、码头、海岸、农田和矿区的站点类别与原visual_scene/visual_note可用于诊断定位，不把站点标签当逐点真实对应证据。

analysis/source_geometry.csv记录原像素间距、地面覆盖、共同有效范围、实际日期、方位/入射/轨道，以及局部栅格轴方向。512尺寸不能消除Umbra/Capella/S1的空间尺度及量纲差异。地图覆盖证明同地图位置可映射，不能证明真实散射结构可重复。地图往返数值正确也不能验证产品绝对地理定位精度。

Homography使用每个角度同一原始匹配诊断，保存投影极点/分母、角点稳定性、空间支持及双向独立拟合循环；本轮从不以H替代被冻结B生成器的Affine。4×4局部Affine仅在26代表样本上诊断，使用相同匹配与原基本估计参数，其拟合残差为重代入，不是独立检验误差。局部通过、内点更多、图看起来吻合都不能证实物理正确。

{hcases}

geometry_figures/提供这4个H诊断案例。异向Umbra_0014__chip_000在90°的H有23个有效内点、通过full门控，但匹配点与地图对应的偏差中位470.587px，格点预测与地图偏差p90=471.452px，无法直接解释为可信配准。旧0°缓存没有H内点Mask时，仅对相同原始匹配重放原冻结H估计器并核对矩阵一致，没有重新检测特征。全部180条H门控通过记录的地理偏差见homography_geographic_diagnostics.csv；不将此差异称作GT误差。

跨方向互相矛盾的门控预测对{summary['contradictions']}，涉及单位{summary['contradictory_chips']}。矛盾代理为≥16个有效地图格点上原生预测差p90>10px；没有调整原接受规则。重复结构、随机偶然匹配或模型不适用都可能导致矛盾，目前不能唯一归因。

## 来源、观测、覆盖与组件分层

{strat('sensor')}

{strat('look_pair_category')}

{strat('overlap_bin')}

{strat('component')}

每个来源×方法×方向假设及场景类别的完整结果见analysis/strata.csv。表中的ANY_ALGORITHM_EXPLORATORY_UNION只表示候选上限，没有访问GT，也不是一个可部署的算法。不能因来源/视角的失败率差异直接作观测角因果判断。相近地理区域共享裁片与采集，191裁片不等于191独立采集；65采集对也共享58采集，报告按站点/组件保留相关性。

## 实际搜索时间与负样本

{costs}

{negatives}

负样本固定为Capella/Sentinel-1/Umbra各10对，使用现有原生图片组合；不同站点、不同实际采集，双向稠密地图映射得到共同有效像素0。negative_index.json在旋转结果前固定并有SHA，没有把“匹配失败”的正样本改成负样本。每个负样本另外完整跑四角度消融，并真实执行相同A/B级联。

负样本的主要路径、采集ID、时间、有效Mask和TIF SHA均正确并用于实际实验。发现初始组合索引中一些未用于预测的moving附属角度/许可字段继承了reference记录；没有改变已固定的样本选择或索引SHA，正确双方附属元数据由analysis/negative_source_metadata.json按两个source_sample身份补充，未来prepare.py已修正继承逻辑。读取负样本角度/许可应使用该补充文件。

多假设接受可能增加偶然错误，即便当前30对没有观察到新增，也不能证实风险为0。未对负样本反复调阈值，没有C级的正样本不能计真实错误接受或Success@px；地理无重叠负样本的错误接受是单独可检查的有限诊断。

## 可视化、代码与复现

figures/共有26个不同代表样本，每对提供三种方法四角度原生匹配线、内点标记、同匹配Affine/Homography图、配准叠加/棋盘、地理结构与局部放大。标注方法、CCW角度、native输入、门控和B状态；绿色支持为伪标签或地图范围，不是真实GT。总览图rotation_acceptance.png、failure_stages.png、negative_acceptance.png附对应CSV源数据，无精度或论文标准合规宣称。

逐样本表：analysis/rotation_results.csv（2292）、strategy_results.csv（真实A/B）、negative_results.csv（360）、zero_rerun_consistency.csv（573）、recoveries.csv、mechanisms.csv、contradictions.csv、local_affine_diagnostics.csv与source_geometry.csv。失败、无矩阵、被拒绝与运行异常均保留。原/新NPZ缓存保存原生匹配、旋转坐标、内点和关键点，旧数据路径明确指向冻结输出。

```bash
cd /home/yl/matching/sar_matching
# 已有目录只读复核；禁止在已存在runs阶段上重复写入。
conda run --no-capture-output -n match python -m pytest -q -p no:cacheprovider {ROOT.name}/test_study.py
# 完整重放到尚未存在的同级目录；拒绝覆盖：
conda run --no-capture-output -n match python {ROOT.name}/reproduce.py --out /home/yl/matching/sar_matching/RealSAR_RotationStudy_REPLAY_YYYYMMDD_HHMM
```

实际命令和开始/结束时间在logs/*_YYYYMMDD_HHMM.log，最终日志reports/{logname}。所有Python在match内；实际依赖版本environment.json与requirements.lock.txt，GPU与PyTorch/CUDA身份见各stage provenance；固定selection/权重哈希与Git版本{freeze['git_version']}。新代码源码快照和SHA记录在code_snapshot/；没有安装新生产依赖。GPU没有额外强制跨硬件逐位确定性，实际缓存与逐样本结果保留用于统计复核。

最终自动化检查全部通过、无跳过；包括整数旋转/逆变换/中心坐标、跨CRS与粗定位组合、真实匹配坐标、0°复用、冻结B规则、毒化GT选择防护、30负样本地理无重叠、统计单位与输出完整性。冻结文件保护结果{protection}，旧标签/人工工具/合成/Phase2/2.5/2.6均未变。这是未改动校验，不是重新执行来源审计或历史实验。
'''
    (reportdir/reportname).write_text(report);log={'completed_at':completed.isoformat(),'config_sha256':sha(ROOT/'config.json'),'preregistration':prereg,'stages':stages,'summary':summary,'actual_strategy_timing':'single-pass cascade with independently measured A stopping prefix; no cache-time sum','zero_rerun_consistency':zero,'source_metadata_correction':'negative ancillary moving fields supplemented via source sample identity, frozen pair choice/images/geography unchanged','plot_cache_warning':'Initial Matplotlib process could not write user configuration, safely used /tmp fallback; plotting script now sets writable experiment cache BEFORE import for reproduction','frozen':freeze,'code_sha256':hashes,'preservation':protection,'tests':test.stdout,'report':reportname,'no_training':True,'C':0};(reportdir/logname).write_text(json.dumps(log,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'README.md').write_text(f'Real SAR fixed rotations and failure mechanisms. Exploratory, C=0, no true Success@px.\n\nReport reports/{reportname}\nExecution reports/{logname}\nIndex/config/preregistration unchanged; see analysis/*.csv and figures/manifest.json.\nOriginal sources and results are frozen.\n')
    receipt={'completed_at':completed.isoformat(),'engineering_accepted':True,'C':0,'report':str(reportdir/reportname),'execution_log':str(reportdir/logname),'summary':summary,'preservation':protection,'file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='ENGINEERING_ACCEPTANCE.json'}};save(ROOT/'ENGINEERING_ACCEPTANCE.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='file_sha256'},indent=2),flush=True)

if __name__=='__main__':main()
