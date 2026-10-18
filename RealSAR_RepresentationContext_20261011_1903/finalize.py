"""Evidence-based engineering acceptance and actual minute-stamped report/log."""
from experiment import *
import importlib.util
def local_module(name):
    spec=importlib.util.spec_from_file_location('representation_'+name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in r)+' |' for r in rows])
def verify(root,hashes,label):
    for i,(name,digest) in enumerate(hashes.items()):
        assert sha(root/name)==digest,'Frozen file changed:'+str(root/name)
        if (i+1)%10000==0:print('protection',label,i+1,'/',len(hashes),flush=True)
    return {'files':len(hashes),'changed':0}
def main():
    reportdir=ROOT/'reports';reportdir.mkdir(exist_ok=False);s=stamp();test=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider',str(ROOT/'test_experiment.py')],capture_output=True,text=True)
    (reportdir/f'tests_{s}.log').write_text(test.stdout+test.stderr);print(test.stdout,flush=True);assert test.returncode==0 and 'skipped' not in test.stdout
    http=subprocess.run([sys.executable,str(ROOT/'test_review.py')],capture_output=True,text=True);(reportdir/f'http_review_{s}.log').write_text(http.stdout+http.stderr);assert http.returncode==0;print(http.stdout,flush=True)
    for path,digest in read(ROOT/'protected_inputs.json').items():assert sha(path)==digest,path
    assert read(ROOT/'preregistration.json')['config_sha256']==sha(ROOT/'config.json') and read(ROOT/'preregistration.json')['samples_sha256']==sha(ROOT/'samples.jsonl');freeze=study.frozen()
    preservation={}
    for label,root,manifest,key in [('historical',BASE,SOURCE/'protected_artifacts.json',None),('original_real_SAR',SOURCE,SOURCE/'DELIVERY_COMPLETE.json','file_sha256'),('old_autolabel',OLD,OLD/'ENGINEERING_ACCEPTANCE.json','file_sha256'),('old_rotation',ROT,ROT/'ENGINEERING_ACCEPTANCE.json','file_sha256'),('old_annotation',MANUAL,MANUAL/'ENGINEERING_ACCEPTANCE.json','file_sha256')]:
        hashes=read(manifest);hashes=hashes[key] if key else hashes;preservation[label]=verify(root,hashes,label)
    save(ROOT/'preservation.json',preservation)
    packages=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json'],text=True));save(ROOT/'environment.json',{'python':sys.executable,'packages':packages,'actual_inference':read(ROOT/'runs/radiometry/provenance.json'),'frozen':freeze,'packages_installed_this_stage':False,'git':subprocess.check_output(['git','rev-parse','HEAD'],cwd=study.PROJECT,text=True).strip(),'git_status':subprocess.check_output(['git','status','--porcelain'],cwd=study.PROJECT,text=True)});(ROOT/'requirements.lock.txt').write_text('\n'.join(f"{r['name']}=={r['version']}" for r in packages)+'\n')
    analysis=local_module('analyze');records=lines(ROOT/'analysis/all_records.jsonl');summary=read(ROOT/'analysis/summary.json');source=selected();lookup={r['sample_id']:r for r in source};quality=read(ROOT/'data/quality.json');contexts=read(ROOT/'contexts_resolved.json');models=lines(ROOT/'geometry/models.jsonl');locals=lines(ROOT/'geometry/local.jsonl');timing=lines(ROOT/'timing_validation/records.jsonl');confirm=read(ROOT/'runs/confirmation/SUMMARY.json');progress=read(ROOT/'analysis/progression.json')
    assert len(records)==252 and len(models)==366 and len(timing)==72 and contexts['available']==12
    assert all(r['predictions_equal_first_run'] for r in timing);assert len(source)==12 and all(r['distinct_acquisition'] and r['different_dates'] for r in quality)
    assert len([r for r in source if r['look_pair_category']=='different_look_verified'])==9
    for r in records:
        if lookup[r['sample_id']]['look_pair_category']=='different_look_verified':assert not r.get('accepted')
    assert read(ROOT/'analysis/candidate_summary.json')['new_sample_ids']==['Umbra_0061__chip_001']
    assert not list((ROOT/'annotation_adapter/annotations').rglob('v*.json'))
    # Additional grouped local evidence; independent units retained, not tile pseudo-replication.
    localtable=[]
    for c in ['original',*REPRESENTATIONS,'context_direct','context_fine']:
        z=[r for r in locals if r['condition']==c];accepted=[r for r in z if r['accepted_basic']];ids={r['sample_id'] for r in accepted};localtable.append([c,len(z),len(accepted),len(ids),len({lookup[s]['pair_id'] for s in ids}),len({lookup[s]['component'] for s in ids}),sum(lookup[r['sample_id']]['look_pair_category']=='different_look_verified' for r in accepted)])
    csvwrite(ROOT/'geometry/local_grouped.csv',[dict(zip(['condition','attempted_tiles','accepted_basic_tiles','crop_pairs','acquisition_pairs','components','different_look_accepted_tiles'],r)) for r in localtable])
    baselines={m:next(r for r in summary if r['condition']=='original' and r['method']==m) for m in METHODS}
    sampletable=table(['样本','来源/站点','观测类别','双向最小地图覆盖','原生地面边长m(两图)','选择'],[[r['sample_id'],r['sensor']+'/'+r['site_name'],'异向' if r['look_pair_category']=='different_look_verified' else '相近方向',round(min(v['fraction_of_all_pixels'] for v in r['effective_geographic_overlap']),3),'/'.join(str(round(v['coverage_edge_lengths_m'][0],1)) for v in r['native_scale_geometry']),'元数据预定'] for r in source])
    radiometrytable=table(['样本/图','原生有效等级','旧PNG等级','PNG0比例','PNG255比例','原值1%/50%/99%','量纲'],[[r['sample_id']+'/'+str(i),v['raw_unique'],v['png_unique'],round(v['png_zero_fraction'],3),round(v['png_255_fraction'],3),'/'.join(f'{x:.5g}' for x in v['raw_quantiles'][1:4]),v['units']] for r in quality for i,v in enumerate(r['views'])])
    comparisontable=table(['条件','方法','Affine/full接受/12','B候选/12','关键匹配中位','内点中位','主失败分布'],[[r['condition'],r['method'],r['accepted'],r['B_eligible'],r['matches']['median'],r['inliers']['median'],json.dumps(r['failures'],ensure_ascii=False)] for r in summary])
    modeltable=table(['输入条件','模型','门控接受记录','尝试记录','同匹配点'],[[c,m,sum(r['accepted'] for r in models if r['condition']==c and r['model']==m),sum(r['condition']==c and r['model']==m for r in models),'是'] for c in ['original',*REPRESENTATIONS,'context_direct','context_fine'] for m in ['affine','homography']])
    localmarkdown=table(['条件','局部尝试','基本Affine接受tile','涉及裁片','采集对','组件','异向通过tile'],localtable)
    costs=table(['条件','方法','每对实测ms中位','p90ms','峰值allocated MB最大','计时来源'],[[r['condition'],r['method'],round(r['time_ms']['median'],2),round(r['time_ms']['p90'],2),max(x.get('peak_cuda_allocated_MB',0) for x in records if x['condition']==r['condition'] and x['method']==r['method']),'旧缓存' if r['condition'] in ['original','geo_assisted512_control'] else '原生读取+表示+特征+Affine/full'] for r in summary if r['condition'] not in ['context_direct','context_fine']]+[[c,m,round(float(np.median([r['actual_end_to_end_ms'] for r in timing if r['condition']==c and r['method']==m])),2),round(float(np.quantile([r['actual_end_to_end_ms'] for r in timing if r['condition']==c and r['method']==m],.9)),2),round(max(r['peak_cuda_allocated_MB'] for r in timing if r['condition']==c and r['method']==m),2),'独立新鲜整流程计时/预测严格一致'] for c in ['context_direct','context_fine'] for m in METHODS])
    confirmtable=table(['条件','方法','门控接受/4','B候选/4','耗时中位ms'],[[r['condition'],r['method'],r['accepted'],r['B_eligible'],round(r['time_ms']['median'],2)] for r in confirm['summary']])
    new=[r for r in records if r['condition']=='linear' and r['sample_id']=='Umbra_0061__chip_001'];newtable=table(['方法','原匹配/内点','线性匹配/内点','新内点比例','双向hull','地图偏差median/p90 px','级别'],[[r['method'],'/'.join(str(next(x for x in records if x['sample_id']==r['sample_id'] and x['method']==r['method'] and x['condition']=='original')[k]) for k in ['n_matches','n_inliers']),f"{r['n_matches']}/{r['n_inliers']}",round(r['inlier_ratio'],3),'/'.join(str(round(r['distribution']['native_inlier'+str(i)]['hull_fraction'],3)) for i in [0,1]),f"{r['geo_difference_median_px']:.3f}/{r['geo_difference_p90_px']:.3f}",r['pseudo_quality']] for r in new])
    changed=sum(v['files'] for v in preservation.values());localdifferent=[r for r in locals if r['accepted_basic'] and lookup[r['sample_id']]['look_pair_category']=='different_look_verified'];candidate=read(ROOT/'analysis/candidate_summary.json')
    # Freeze code snapshot after all fixes, separately from preregistration/individual runner hashes.
    snapshot=ROOT/'code_snapshot';snapshot.mkdir();code=[*ROOT.glob('*.py'),ROOT/'TASK.txt',ROOT/'config.json'];hashes={p.name:sha(p) for p in code}
    for p in code:shutil.copyfile(p,snapshot/p.name)
    save(snapshot/'SHA256.json',hashes)
    completed=now();s=stamp();reportname=f'representation_context_report_{s}.md';logname=f'execution_{s}.log'
    report=f'''# 真实 SAR 辐射表示、空间上下文与几何适配研究

完成时间：{completed}（Asia/Shanghai）。创建目录1903是本轮开始分钟；本报告和最终日志使用完成时的当前分钟{s}，没有沿用旧时间戳。

## 结论与证据范围

在固定12对/12真实采集对/8站点和来源组件中，三种冻结算法的原PNG门控各接受1对；稳健线性表示各接受2对。新增样本只有Umbra_0061__chip_001，来自长滩港**相近方向重复观测**，三个方法均为一般B候选。9对明确异向样本在原PNG、三种表示、1024上下文直接/粗到精以及同匹配Homography中均没有全局接受或新增B候选。不能把这个同向收益称为异向恢复。

旧191对、地理对应、19对旧B、旋转实验和所有合成/正式结果保持冻结；全量文件SHA核对{changed}项未变（不同保护清单中可能有重复项），本轮必要输入285项也逐一核对。C=0，没有训练、微调、旋转搜索、新来源检索、大量下载或191对配置搜索。工程验收不等于真实配准精度验证。

## 预定样本与有效性

{sampletable}

实际分类9异向、3相近方向；Umbra10、Capella1、Sentinel-1 1。原采集身份/时间/升降轨/左右视/方位入射角/许可/源产品路径详见samples.jsonl，必要文件SHA见protected_inputs.json。选择只依赖预定站点、地图有效覆盖、采集对唯一性和seed={config()['seed']}的ID散列；每站最高覆盖一对，4个预定站点再选另一采集对靠近预定覆盖目标。没有根据匹配结果换样，但最高覆盖规则并没有强制Umbra异向，所以两个港口最高覆盖样本实际同向。过程口头曾误称异向，复核后已更正，冻结清单未改。

全部两图身份及时间不同，有正共同地图覆盖；有效比例逐图见data/quality.json。观测类别和角度证据来自已验证采集侧车及冻结索引，而非PNG旋转或匹配结果。地图位置、栅格往返一致只能证明计算/产品地理范围相容，不验证散射中心身份或绝对地理定位精度。城市、矿区、农田、库区和港口是站点上下文类别；每个裁片是否具有双图共同静态结构仍需独立核验。不同产品512像素地面覆盖不同，分辨率/像素间距不能等同；源栅格旋转/CRS差异由旧PixelMap正确处理，未修改地理对应。

12主样本只覆盖8/12组件。余4组件（{config()['holdout_components']}）用于本轮因素选择之外的一次诊断；没有剩余未参与主样本的明确异向站点，且全部191历史上已分析，不存在严格无偏的真实确认集。

## 原始辐射到PNG转换审计

{radiometrytable}

旧流程由code_snapshot_final/build_real_pairs.py追踪：幅度20log10(max(DN,1e-12))、功率10log10、db直接；按metadata.json中冻结display_db_range截断/round到uint8，无效0。高分辨率产品使用旧联合原生百分位显示范围，Sentinel-1使用固定-25..0db。24张旧PNG均与原转换逐像素一致，没有发现意外转换软件错误。旧Umbra源产品本身为uint8量化幅度DN，float32保存容器不能恢复已丢失的产品精度；Capella DN幅度及Sentinel-1 Gamma0功率不是统一物理标度。Capella标定比例在源元数据保留，当前仅比较同对图的表示，未伪造跨传感器sigma0可比性。

部分Umbra旧PNG灰度级明显减少，出现大比例有效黑像素；这可能来自低DN的对数截断/量化和显示范围，并非这些像素无效。线性显示有时压低噪声背景、保留强结构，而不是简单“变亮即改善”。图像统计保留原值正/负/零/NoData、梯度、局部对比与原值局部CV。CV来自异质场景，不作为ENL或纯散斑估计；没有独立的同质静态区，不能分离散斑、纹理、阴影或叠掩的物理贡献。未使用额外滤波器搜索。

新表示固定：B逐图有效原值1%/99%线性归一化；C以有效正值中位为scale的asinh再逐图1%/99%；db测试兼容有符号值；D只对旧PNG做CLAHE clipLimit=2、8x8网格。无效区域在导出PNG仍为0并保留原Mask，派生推理输入作nearest-valid延拓，避免填充0边缘影响增强/检测。有效零从不自动当作NoData；线性量纲出现负有效值则拒绝并要求检查。全部参数在当前推理结果前固定，仅用输入图自身统计。

## 单因素匹配消融

{comparisontable}

original和geo_assisted512_control各36条来自旧缓存，没有重复历史推理。派生512表示各36条，保持max2048、SIFT contrast=.01、ORB edge16、mutual ratio=.8、SuperPoint threshold=.0001、官方LightGlue/权重、Affine RANSAC/full门控/seed42全部冻结。实际SIFT关键点在并列响应时可略超过请求上限，CSV保存真实数量。预测只使用图像，Mask/地图只用于后验原B规则；地理辅助控制与image-only结果分开，不把先验算作图像匹配收益。

{newtable}

新增B候选共1裁片/1采集对/1站点/1组件，包含三个算法版本记录，不是三对独立数据；地图差约26px，只满足原一般B，不满足高B的地图median≤8/p90≤16要求。原PNG的LightGlue已有451匹配/68内点，但内点率不足；线性表示变为296匹配/99内点，匹配数量减少而一致性改善，说明不能用匹配数作成功代理。SIFT/ORB有效匹配和空间支持同时增加。算法之间一致也不是独立真值，港口舰船/货物变化和重复设施可能产生共同偏差，未经人工核验不认定正确。

全部新增候选版本存new_candidates/，原输入条件、算法、匹配点、矩阵与支持区域可追溯；不覆盖旧19B，不具备训练资格。派生条件共{candidate['candidate_predictions']}条B候选记录涉及2裁片（还包括原S1控制），真正原样本新增只有上述1对。没有独立C，不能计算Success@3/5/10px或真实正样本错误接受率。

## 真正1024上下文与粗到精

没有本地完整大幅产品；利用既有冻结源产品URL进行有限窗口读取，未发现新采集或下载整幅产品。12/12双图1024窗口成功，原512在两图各对应(x,y)=(256,256)中心子区。原产品有效像素和有效Mask与冻结512一致，按完全相同旧db显示范围生成的中心PNG也一致，未插值制造上下文。

直接方案：全分辨率1024、仍max2048，Affine/full粗几何用1024坐标；另把原匹配移回512，仅保留双方512内匹配重新估计Affine/full，作为独立直接条件。粗到精方案：只有粗模型image-only full接受才把原生moving512按纯图像预测warp到reference512，重新检测/匹配，恢复points1=H粗×fine_points1后在原生512匹配上估计Affine/full。H原生=T(-256)@H1024@T(+256)，不把大范围H直接当512精确标签。当前只有S1控制每方法通过粗门控（各1/12），其余33条没有fine输出，记录为coarse_rejected，未伪造“零匹配的已执行细阶段”。

1024与512真实地面范围相比每边约翻倍，面积约4倍；特征预算不增加，存在特征被上下文分流的限制。这里只检验1024原生全分辨率上下文，未检验无限更大范围或多种金字塔，负面结果不能排除其他尺度的效用。原PNG表示保持冻结，未将线性收益叠加进上下文；既然上下文独立没有收益，本轮不做组合搜索。

## 同匹配点几何与局部候选

{modeltable}

{localmarkdown}

Affine和H使用完全相同匹配集合、原RANSAC参数、各自独立反向拟合和full诊断；保存内点、空间分布、重代入残差、分母/极点、角点面积/边尺度/Jacobian及循环一致性。H没有新增全局接受样本；本轮并非“放宽模型就解决异向”。context_fine只有3条真正执行后的模型记录；其余因粗阶段拒绝而无匹配集合，未纳入几何估计分母。

4x4局部≥8原匹配的基本Affine共{len(locals)}个tile尝试、{sum(r['accepted_basic'] for r in locals)}接受。基本局部支持检查弱于全幅full，仅表示同点重代入可拟合；不能外推或提升B。唯一明确异向通过局部的记录为{[(r['sample_id'],r['condition'],r['method'],r['tile_x'],r['tile_y'],r['n_inliers'],round(r['geographic_difference_median_px'],3)) for r in localdifferent]}，与地图偏差约126px，不能认为是真实正确局部配准。其余局部接受主要在同向港口和S1控制。geometry/local_contradictions.json保留不同tile模型在双方支持中心的交叉外推差异；这可能是匹配错误或非统一几何，并不能单独证明SAR高差/阴影/叠掩导致非平面关系。

## 分层、失败与一次组件确认

analysis/strata.csv逐条件/方法分来源、观测类别、场景、有效覆盖区间、来源组件，并报告裁片/真实采集对/站点/组件单位。9个异向主样本的接受均为0，不将其余同向改善混算。analysis/changes.csv保存每对匹配、内点、门控、B、收益/损失与耗时；不足匹配、几何拒绝、粗阶段拒绝、B筛选拒绝与无输出分开，原始拒绝原因全部保留。不能在没有C的正样本上用地图差直接判真错，原30负样本旋转结果未重新执行也不冒充本轮预处理的负样本验证。

由于线性表示新增了一个有空间支持的B候选，在全部单因素结束后仅锁定linear于4个剩余组件各1对做一次确认（extension_decision.json，在确认推理前写入）。同一固定算法/阈值，无重新选择或调参；确认24条=12旧缓存+12新推理。4个组件与主8组件不相交，但仅是本轮因素选择隔离，历史191已看过，且没有新的明确异向站点，无法声称无偏异向泛化。

{confirmtable}

linear让Capella_0001__chip_002的SIFT从6匹配/0内点到19/11并通过full，地图差median44.934px，原B规则拒绝；ORB/LG在该样本原已接受同样受地图差拒绝。确认中每方法B覆盖仍2/4（S1），没有新增B覆盖。选择区有局部同向收益，确认没有B扩展收益，故不再扩大191或组合/反复优化。

## 实际成本与可复现协议

{costs}

派生512计时包含读取原生raw/valid/旧图、表示、特征、Affine/full，排除初始化、同匹配H/局部诊断、地图QA和输出序列化；原基线为旧PIL PNG推理缓存，时间口径/运行进程不同，不能据表精确计算预处理百分比额外代价。分辨时原值读取与表示计算均新增，不能把全部延迟归给asinh/CLAHE算子。

上下文首轮缓存/诊断有交错，另做一次针对性完整流程计时，direct和fine分别新鲜实际执行，包含1024图读取、粗特征/门控及实际512过滤/门控或原512读取/warp/新fine特征/原生估计，排除初始化/网络取窗口/输出/地图与H诊断；不是缓存分项耗时估算。72条预测、原始粗匹配和执行后的原生匹配/内点/门控与首轮严格一致，完整记录timing_validation/records.csv。fine中位大多代表粗拒绝即停止，不能据此说执行过细阶段。CUDA峰值为torch.max_memory_allocated（含已载入网络），不含系统总显存或未被torch追踪的分配；SIFT/ORB行也包含已载入LG的驻留。1024 LG约581MB高于512约190MB，但没有新增接受收益。

实际环境见environment.json与requirements.lock.txt；Python {sys.executable}，RTX4090推理，torch2.5.1+cu121，只有opencv-python-headless。source Git版本{read(ROOT/'environment.json')['git']}，selection、核心、LightGlue Git和官方权重SHA由preregistration与各run/provenance固定；新增代码最终SHA见code_snapshot/SHA256.json。没有修改base环境或安装新依赖。

```bash
cd /home/yl/matching/sar_matching
conda run --no-capture-output -n match python -B -m pytest -q -p no:cacheprovider {ROOT.name}/test_experiment.py
# 完整重放到新的同级目录；默认复制已封存的真实1024窗口，不重复网络读取。
conda run --no-capture-output -n match python {ROOT.name}/reproduce.py --out /home/yl/matching/sar_matching/RealSAR_RepresentationContext_REPLAY_新时间戳
# 需重查原产品时加 --fetch-context；仅有限1024读取，网络失败按实记录。
```

复现默认利用既有旧PNG缓存；开发函数拒绝覆盖阶段输出，reproduce拒绝已存在目录。重放会重跑当前锁定小实验而不改旧数据；重放字段标为replay，不伪称过去结果从未观察。受网络影响原产品可能不可读，默认用本轮通过原有效像素/Mask/显示等价校验的封存1024窗口。各阶段log均含真实命令、开始/结束时间和exitcode。

## 独立核验入口与失败记录

5对任务：S1SF_EXTENSION_000__chip_002（同向控制）、Capella_0003__chip_001（异向城市）、Umbra_0061__chip_001（新线性候选、同向港口）、Umbra_0071__chip_000（同向港口失败、线性局部候选）、Umbra_0014__chip_002（异向矿区）。至少两对明确异向、至少一对同向并包含新候选；来自主元数据预定代表，而非按结果隐藏失败。review_guide/给出参考图检索区域和两图清晰预览，区域不是对应点、静态判断或自动人工GT。

```bash
conda run --no-capture-output -n match python {ROOT.name}/serve_review.py --port 8767
# 在本机（非服务器）按实际主机名建立SSH转发后浏览：
ssh -L 8767:127.0.0.1:8767 yl@服务器
# http://127.0.0.1:8767
```

复用现有网页及其审计/复核逻辑；输入是原生PNG，自动候选默认隐藏，查看辅助层永久记录，不能把辅助标签标为独立C。人工点0，C0，无伪造人工操作。本轮实际读测5任务/10原生PNG路由通过，未写点或暴露记录。各图和对应CSV已生成，人工核验是后续活动，不阻塞自动化交付。

过程失败都保留：最初S1缺省allow_declared_valid_zero引发KeyError，改用原builder同样False默认；两Umbra初次中心raw全值比较被旧-9999无效编码触发，改为有效原值与有效Mask分别核对，显示仍逐像素核对；两次retry的读取完成后汇总存在old变量被PNG数组复用而IndexError，改名并只读重建汇总，没有再次推理或调参；确认推理全部完成保存后，复用模块的sys.path使汇总错误导入旧analyze，改为显式本轮文件模块，只从24个已有记录重建SUMMARY，确认推理没有重复。原错误日志保留，新最终代码修复，当前所有数据/模型/计时/任务验收通过。初次网络有限读取均可用，未遇权限阻塞或伪造上下文。相近方向被口头误称异向的更正已记录，上表/CSV/索引是实际元数据类别。

## 当前真实异向配准失败更可能受什么限制？

**实验直接支持**：存在输入显示压缩和低DN量化问题，但合理固定表示仅恢复1对同向港口，9对异向没有恢复；所以不能把异向失败主要归为PNG软件转换错误。真实1024、旧辐射、固定2048特征没有新增粗定位/原512候选，局部上下文不足作为这套方案的主要解释未获支持。H在同匹配点没有新增全局接受，局部异向唯一拟合与地图差很大，尚无证据说明只需更灵活全局模型便能解决。

SIFT/ORB在表示调整后仍经常不足8个互检匹配；LG能输出大量描述对应，却多数只有少数或低比例几何内点。线性长滩例子的匹配数下降而内点/支持提升进一步证明对应可重复性与描述区分能力的重要性。综合本轮受控计算证据，**跨观测条件下的特征描述/匹配可重复性，是比简单旋转、这组显示规则、1024上下文和Affine→H替换更有依据的下一步重点**。

**仍须独立GT验证的推断**：这里不能证明算法错配的物理原因、同一散射中心可见性、非刚性投影是否主导，亦不能把局部模型矛盾等同地形或叠掩。12对是受控小样本且9异向集中现有7?组件（精确单位见strata.csv），不同来源与场景/角度/像素尺度耦合，不能推广成所有SAR异向场景的因果结论。原生GEC/RTC也不是原始复数SLC观测，现有DN产品会限制可恢复辐射信息；更高保真未量化数据的作用仍未检验。

值得规划小规模SAR专用特征/描述匹配学习研究，当前证据足以停止无限传统参数尝试；**尚不足以宣布应立即大规模微调或其必然有效**。先完成上述异向城市/矿区的独立静态对应、地图偏差与原生数据质量核验，建立来源组件隔离的标注/训练/验证协议，再用固定计算预算评价SAR特征学习是否提高真正对应可重复性。任何域微调都应同时保存同匹配几何诊断和外部核验，不用旧自动B或地图位置冒充真实测试GT。本轮到此停止配置扩展，不训练网络。
'''
    # Exact current primary different-look units, no hand-written approximate component count.
    different_units=analysis.units([r for r in source if r['look_pair_category']=='different_look_verified']);report=report.replace('现有7?组件',f"现有{different_units['components']}个组件")
    (reportdir/reportname).write_text(report)
    logs=sorted((ROOT/'logs').glob('*.log'));execution=f'Completed {completed}\nFINAL source/config/selection/weight hashes verified. Frozen inputs {len(read(ROOT/"protected_inputs.json"))}, historical protection entries {changed}; C0 no Success metrics, no training.\n';execution+='\n'.join('\nFILE '+str(p.relative_to(ROOT))+'\n'+p.read_text() for p in logs);execution+='\nDERIVED stages executed directly: analyze.py (252 rows), test_review.py (five real tasks read-only); final focused tests below.\n'+test.stdout+'\n'+http.stdout;(reportdir/logname).write_text(execution)
    (ROOT/'README.md').write_text(f'本轮自动化诊断完成，C=0，无真实精度GT。主样本12对/8组件（9异向、3同向），确认4对/4剩余组件。\n\n报告：reports/{reportname}\n日志：reports/{logname}\n完整逐样本：analysis/all_records.csv；真实上下文完整计时：timing_validation/records.csv。\n原数据与19B保持冻结；新增候选仅另存。\n复现：conda run --no-capture-output -n match python {ROOT}/reproduce.py --out {BASE}/RealSAR_RepresentationContext_REPLAY_新时间戳\n核验入口：conda run --no-capture-output -n match python {ROOT}/serve_review.py --port 8767\n默认隐藏自动点，人工点0，独立核验任务见review_guide/README.md。\n')
    receipt={'completed_at':completed,'engineering_accepted':True,'C':0,'independent_accuracy_verified':False,'primary_pairs':12,'confirmed_different_look_pairs':9,'confirmation_pairs':4,'report':'reports/'+reportname,'execution_log':'reports/'+logname,'preservation':preservation,'new_B_sample_ids':candidate['new_sample_ids'],'old_B_unchanged':19,'no_training':True,'Success_metrics_computed':False,'file_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*')) if p.is_file() and '.mplcache' not in p.parts and '__pycache__' not in p.parts}}
    save(ROOT/'ENGINEERING_ACCEPTANCE.json',receipt);print('ACCEPTED',receipt['completed_at'],'files',len(receipt['file_sha256']),'report',reportname,flush=True)
if __name__=='__main__':main()
