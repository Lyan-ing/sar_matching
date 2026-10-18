"""Concise scientific handoff from actual V2.2 build/QA, without changing history."""
from data22 import *
from manifests22 import csvwrite
from collections import Counter
from PIL import Image,ImageDraw
import argparse

def table(headers,rs):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(str(x) for x in r)+' |\n' for r in rs)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('summary',type=Path);args=parser.parse_args();summary=json.loads(args.summary.read_text());cfg=json.loads(Path(summary['config']).read_text());t=stamp()
    qa_path=sorted(ROOT.glob('GT_MASK_QA_*.json'))[-1];qa=json.loads(qa_path.read_text());ipath=sorted(ROOT.glob('INTERFACE_QA_V22_*.json'))[-1];interface=json.loads(ipath.read_text());assert qa['passed'] and interface['passed']
    rr=rows(summary['all_rows_with_roi_family']);rrows=[r for r in rr if r['arm']=='V2-R'];old=old_rows();protected=json.loads(next(ROOT.glob('PROTECTED_V21_*.json')).read_text());assert all(sha(p)==h for p,h in protected.items())
    assert cfg['appearance']==CFG21['appearance'] and cfg['geometry']==CFG21['geometry']
    old_sources={r['source_path'] for r in old};assert {r['source_path'] for r in rrows}==old_sources
    # Explicit counterfactual subsets for feature-supervision ablation; images remain references.
    subsets={}
    for split in ['train','val']:
        for pool in ['core','ssdd']:
            data=[r for r in rrows if r['split']==split and (r['source_dataset']=='ssdd')==(pool=='ssdd')]
            for role,select in [('same_pixels_local',lambda r:r['origin']=='reused_V21' and r['local_descriptor_eligible']),('same_pixels_old_selector_local',lambda r:r['origin']=='reused_V21' and r['local_descriptor_eligible'] and r['source_structural_candidates_v21']>0),('new_ROI_local',lambda r:r['origin']=='new_ROI' and r['local_descriptor_eligible'])]:
                path=ROOT/f'ABLATION_{split}_{pool}_{role}_{t}.jsonl';jsonl(path,[r for r in data if select(r)]);subsets[f'{split}/{pool}/{role}']=str(path)
    candidates=[
        next(r for r in rrows if r['origin']=='new_ROI' and r['global_eligible'] and r['image_rotation_2D_deg']>60),
        next(r for r in rrows if r['origin']=='reused_V21' and r['old_loss_role']=='limited_structure' and 4<=r['structural_anchor_count']<=12),
        next(r for r in rrows if r['source_scene_category']=='offshore_vessels_low_texture_sea' and r['ship_descriptor_eligible']),
        next(r for r in rrows if 0<r['structural_anchor_count']<4),
        next(r for r in rrows if r['origin']=='new_ROI' and r['profile']=='low' and r['local_descriptor_eligible']),
        next(r for r in rrows if r['loss_role']=='no_structural_positive' and sum(r['calibration_pixels'])==0)]
    figures=[]
    for page in range(2):
        canvas=Image.new('RGB',(1024,3*340+30),'white');draw=ImageDraw.Draw(canvas)
        draw.text((8,8),'Green: ground | Cyan: OPTIONAL synthetic ship | Orange: appearance ignore | NOT real physical GT',fill='black')
        for j,r in enumerate(candidates[page*3:page*3+3]):
            z=load_pair(r,use_ship_aux=True);y=j*340+30
            for side in range(2):
                im=cv2.cvtColor(z['images'][side],cv2.COLOR_GRAY2RGB)
                contours,_=cv2.findContours(z['ignore'][side].astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(255,160,0),1)
                for xy in z['points0' if side==0 else 'points1']:cv2.circle(im,tuple(np.rint(xy).astype(int)),2,(40,240,70),1)
                for xy in z['ship_points0' if side==0 else 'ship_points1']:cv2.circle(im,tuple(np.rint(xy).astype(int)),2,(0,220,255),1)
                tile=Image.fromarray(im);tile.thumbnail((500,285));canvas.paste(tile,(side*512+(512-tile.width)//2,y))
            draw.text((8,y+288),f"{r['relation_id']} {r['split']} {r['source_dataset']} {r['width']}px {r['image_rotation_2D_deg']:.1f}deg {r['loss_role']}",fill='black')
            draw.text((8,y+306),f"Ground N={r['structural_anchor_count']} (global={r['global_eligible']}) | optional ship N={r['ship_aux_count']} | overlap={min(r['overlap_fractions']):.2f}",fill='black')
        path=ROOT/f'GRADED_SUPERVISION_PREVIEW_{page:02d}_{t}.jpg';canvas.save(path,quality=94);figures.append(str(path))
    role_stats=summary['role_stats'];core_stats=[r for r in role_stats if r['arm']=='V2-R' and r['pool']=='core' and r['role'] in ['all','structural_positive','limited_structure','local_descriptor','local_detection_only','calibration_evidence','optional_ship_aux']]
    stattable=table(['划分','用途','配对','不同源文件','已知父组件'],[[r['split'],r['role'],r['pairs'],r['sources'],r['components']] for r in core_stats])
    roles=summary['R_roles'];oldroles=Counter(r['loss_role'] for r in old)
    roletable=table(['四类','V2.1 R','V2.2 R','新版占比'],[[r,oldroles[r],roles[r],f"{100*roles[r]/len(rrows):.2f}%"] for r in ['structural_positive','limited_structure','no_structural_positive','no_overlap']])
    names=dict(semantic_veto_no_accepted_source_corner='粗场景否决且候选筛选无角点',semantic_veto_with_proposals='粗场景否决；候选非零但地面身份未确认',source_selector_no_corner_after_ship_land_texture_persistence='来源经船/海陆/纹理/持久性筛选后无点',source_corners_outside_common_crop='来源有角点，共同裁片未保留',appearance_ignore_or_bilinear_support='增强ignore或插值支持损失',below16='有局部对应但未达16点',grid_below4='点数达到16但覆盖格不足',hull_below5pct='前述条件通过但凸包不足',eigen_ratio_below_03='前述条件通过但接近退化',analytic_no_overlap='解析无重叠',strict_global_pass='严格全局门槛通过')
    names['eigen_ratio_below_.03']='前述条件通过但协方差近退化'
    causetable=table(['旧版互斥主因','关系数'],[[names.get(k,k),n] for k,n in summary['primary_causes'].items()])
    stages=summary['attribution'];before=summary['old_core_training_full_sources'];after=summary['new_core_training_local_sources'];new_count=sum(r['origin']=='new_ROI' for r in rrows)
    old_reuse_full=sum(r['origin']=='reused_V21' and r['global_eligible'] for r in rrows);new_full=roles['structural_positive']-old_reuse_full
    angle=[]
    for b in ['00_15','15_30','30_60','60_90','90_120']:
        oldb=[r for r in old if r['profile']!='none' and r['angle_bin_2D']==b];newb=[r for r in rrows if r['profile']!='none' and r['angle_bin_2D']==b]
        angle.append([b,len(oldb),sum(r['loss_role']=='structural_positive' for r in oldb),len(newb),sum(r['local_descriptor_eligible'] for r in newb)])
    angletable=table(['二维角度','旧关系','旧有效匹配','新关系','新局部描述有效'],angle)
    scenes=[]
    for scene in sorted({r['source_scene_category'] for r in rrows}):
        oldc=[r for r in old if r['source_scene_category']==scene];newc=[r for r in rrows if r['source_scene_category']==scene]
        scenes.append([scene,len(oldc),sum(r['loss_role']=='structural_positive' for r in oldc),len(newc),sum(r['local_descriptor_eligible'] for r in newc),sum(r['ship_descriptor_eligible'] for r in newc)])
    scenetable=table(['原来源语境','旧关系','旧匹配','新关系','新局部描述','可选船辅助'],scenes)
    maritime=sum(r['source_scene_category'] in ['offshore_vessels_low_texture_sea','nearshore_vessels','port_quay_dense_vessels'] for r in rrows)
    sims=interface['sampling_simulation'];simtable=table(['模式','抽样调用¹','来源覆盖','覆盖率','最高单源次数'],[[s['mode'],s['calls'],f"{s['seen_source_files']}/{s['eligible_source_files']}",f"{100*s['coverage_fraction']:.2f}%",s['max_source_calls']] for s in sims])
    protocol=next(ROOT.glob('SUPERVISION_PROTOCOL_*.md'));entrypath=ROOT/f'DATASET_ENTRYPOINTS_V22_{t}.json'
    entry=dict(version='2.2',created_at=now(),trainability='可训练；M2分级描述/无重叠负校准，B检测/描述/有证据校准；不保证训练收益',
        recommended_arm='V2-R',recommended_train_local=summary['manifests']['V2-R/train/core/local_descriptor'],recommended_train_reference=summary['manifests']['V2-R/train/core/all'],
        recommended_val_all=summary['manifests']['V2-R/val/core/all'],config=summary['config'],protocol=str(protocol),manifests=summary['manifests'],ablation_subsets=subsets,
        fixed_reader=str(ROOT/'code/data22.py'),role_loss_bridge=str(ROOT/'code/loss22.py'),online_sampler=str(ROOT/'code/sampling22.py'),source_feature_index=str(next(ROOT.glob('SOURCE_FEATURE_AUDIT_*.jsonl'))),
        frozen_validation=dict(V1_standard=ENTRY['original_V1_standard_validation'],V1_reader=ENTRY['original_V1_adapter'],V21_common133=ENTRY['V2_common_structural_validation'],
            V21_all={k:v for k,v in ENTRY['fixed_manifests'].items() if '/val/' in k},V21_reader=str(OLD/'code/dataset_v2.py')),
        dataset_summary=str(args.summary.resolve()),input_GT_mask_QA=str(qa_path),interface_QA=str(ipath),R_pairs=len(rrows),R_views=2*len(rrows),R_ground_descriptor_pairs=summary['R_local_descriptor_pairs'],
        R_strict_global_pairs=roles['structural_positive'],R_optional_ship_descriptor_pairs=summary['R_optional_ship_descriptor_pairs'],R_original_files=summary['R_sources'],
        core_train_ground_descriptor_sources=after,core_train_ground_descriptor_pairs=next(r['pairs'] for r in core_stats if r['split']=='train' and r['role']=='local_descriptor'),
        ship_aux_default=False,ground_global_thresholds_unchanged=True,SSDD_extension_not_proven_raw_acquisition_independence=True,old_data_or_checkpoints_modified=False,network_training_started=False)
    save(entrypath,entry)
    save(ROOT/f'LOSS_POLICY_{t}.json',dict(at=now(),M2_local_descriptor_min=4,M2_detector_frozen=True,M2_calibration='explicit skip, no fake descriptor positives',B_detector_min=1,
        local_weight='1 if strict global else min(1,N/16)',ship_aux_default=False,ship_aux_weight=.15,negative_weight=.1,negative_cosine_margin=.8,
        negative_probes='analytic non-overlap only, never structural positive anchors',B_negative_cells='sea evidence calibration_mask only, unknown ignored',
        no_model_architecture_changes=True,not_a_claim_of_effective_training=True))
    images='\n\n'.join(f'![分级监督示例：绿为地面候选点、青为默认关闭的合成舰船辅助点、橙为增强ignore；均非真实物理GT]({Path(p).name})' for p in figures)
    report=f'''# Synthetic SAR V2.2：分级监督与有界ROI改进

交付时间：{now()}。**可训练，推荐R/core。** 本轮实际完成标签、局部裁片、来源记录、训练采样和接口验收；没有下载SAR、扩展真实数据或训练网络。机器入口：[{entrypath.name}]({entrypath.name})；[直接使用说明](DATASET_USAGE_V22_{t}.md)。

## 1. 为什么旧版只有1264组

旧7214组的互斥主因重新从原图缓存角点、实际H及原掩膜逐对重建，计数严格复现旧标签，合计7214：

{causetable}

占多数的离岸/未分类数据，在舰船排除后缺少被接受的地面结构；这不证明物理上绝无对应。复查了122项原图预览，候选非零的离岸图常是船、尾迹和海纹，不能直接取消语义否决；仅对12个可见陆地/道路/防波堤区域记录暂定AI范围，原图角点阈值不变。SSDD已有海陆掩膜中的小陆地区也可保留。未确认区域继续unknown。

16点主要过滤局部可学样本；达到16点后又被空间门槛单独拒绝的仅22组。旧代码还把未过全局门槛的点及static清空，所以692个limited标签本身没有实际描述子监督。原图审查中core Train有747张来源满足旧候选点数≥4，而实际全局合格来源452张，确有裁片和训练准入瓶颈；并非其余全部源图都能转为地面正例。

## 2. 实际构建结果

{roletable}

单独R：**{len(rrows)}配对/{2*len(rrows)}视图实例**；不同原始文件仍为{summary['R_sources']}，不是新独立采集。复用原7214组图像、另生成并保留{new_count}组不同原生FOV。R/S两臂共{len(rr)}条配对记录/{2*len(rr)}视图，不能当双倍来源。1406个新ROI计划一次冻结，1381个完成图像生成，25个几何失败；其中233个实际FOV过于重复而隔离，未重新裁切补数。原生图、标签、失败和旧版本保留。

地面描述子可用**{summary['R_local_descriptor_pairs']}组**（≥4点），比旧1264组增加{summary['R_local_descriptor_pairs']-1264}组。充分结构{roles['structural_positive']}组；有限结构{roles['limited_structure']}组中{summary['R_local_descriptor_pairs']-roles['structural_positive']}组也能用于描述子，其余仅提供少量检测点。不能把全部limited都算描述子正例。

新增描述子关系的可核验归因：**512组**来自同图、同角点筛选下的训练准入调整；**2组**来自同图区域证据复查；**1073组**来自实际新增ROI。不是全部靠重加权。严格全局合格从1264增至2047；复用图像中严格合格{old_reuse_full}组，新ROI贡献严格合格{new_full}组，全局阈值没有放宽。4/8/12/16准入的Train/Val/core/SSDD源数与组件数见`LOCAL_THRESHOLD_SENSITIVITY` CSV。

核心训练地面描述来源**{before}→{after}张（+{after-before}，{100*(after/before-1):.1f}%）**，它们是不同发布文件，不是已证明独立原采集；原始文件总数没有增加。地面来源不能扩到全部海面，677也不应与可选舰船来源混算。每源贡献、ROI家族和重复关系见`PER_SOURCE_CONTRIBUTION`。

{stattable}

各用途存在交集，不能相加当不同来源。natural manifest按每个原始文件取第一个既有可用关系，完全不按结构成功选择；all保留既有难例及全部新增可用ROI。SSDD扩展单列，其原采集身份不完整。已知父/像素组件Train/Val及原Test交集均0，但不能据此声明完全独立物理采集或新未见最终测试。

## 3. 分级监督实际怎样使用

- 充分结构：沿用双方≥16点、≥4个4×4格、凸包≥5%、特征值比≥.03，可进入新合成全幅评价。
- 有限结构：≥1可靠源图角点保留检测标签；≥4才计算局部描述子CE，权重min(1,N/16)。1–3点合法跳过描述损失，绝不补点；不能强迫其进入全幅Success正例。
- 低纹理/无可靠结构：没有角点不等于负例。仅有海面语义/可靠海陆掩膜，同时原图粗纹理std<.010、低角点强度、均值>.005且远离船/结构的区域可做校准；其余有效区明确保存supervision_ignore。
- 无重叠：只接受解析双向重叠为0。单独的负探针只用于已证明无共同FOV的负对应代理，不是海面稳定正点。M2/B可用有界余弦hinge，默认权重.1、margin .8；不能保证仅靠此项就解决实际拒绝问题。

M2检测头冻结，因此低纹理检测校准**不能训练其检测头**；接口明确返回skip，M2采样模式不靠假点填满这类调用。B真正消费检测/描述和有证据校准，unknown不被当成dustbin负例。M2/B适配器是V2.2角色消费版本，仍用原网络结构和既有损失构件；标签准入、校准掩膜与角色权重是本版实验因素，不能与旧B分数混称单因素消融。M2/B局部损失及1/4/8/12/16点均已实际CPU测试，没有优化器更新。

## 4. 独立保存的舰船辅助

R中**{summary['R_optional_ship_descriptor_pairs']}组**有≥4个可用合成舰船辅助点，另保留少量检测点。候选只来自未增强原图的真实分割实例内部，双图均在有效分割支持内、无动态/增强ignore、双方点距≥8像素。BBox-only不冒称船体监督。动态船响应衰减的关系没有舰船辅助对应，QA已检查。辅助点与地面点、掩膜、实例索引分开保存，默认关闭，开启权重.15；绝不并入地面GT或真实跨采集物理同名点。

`ship_aux_masks`是可见分割候选支持；真正用于损失的是NPZ中通过双方共同可见性检查的paired points，不能从整个mask自动填充点。监督位置在原图形成，新增噪声不参与角点发现。原始散斑仍共享，舰船辅助只是同源几何特征学习选项。

## 5. 场景、几何与采样

{scenetable}

海事来源语境占{100*maritime/len(rrows):.2f}%，不把语境当逐像素海陆或独立物理结构真值。离岸多数不提供地面正点，但可参与有证据校准与可选舰船辅助。其他陆地/近岸/港口继续共同提供背景主监督。

{angletable}

此表是**二维旋转，不是真实观测方位差**。新关系使用原R/S外观及0–120°角度/尺度/透视/重叠参数，无新噪声调参；normal .50–.90、low .12–.30、有效区≥90%，尺寸≤512、主要256/384/512，不预放大源图。部分512新ROI受原图大小和防重复约束无法成立，失败保留；旧512、大角度和低纹理关系没有删除。完整场景×角度×尺寸×重叠×角色交叉CSV保存。

B训练参考配额40%充分、25%有限、27%有证据低纹理校准、8%解析无重叠；M2模式55/35/0/10，因其检测头冻结。按角色→已知组件→原始文件→ROI家族→关系轮转，避免旧重复增强关系直接支配抽样。实际在线重新生成几何/外观、重新判断真实角色，参考角色不保证每次仍合格，日志保留实际角色、失败和完整种子；不按标签结果静默重抽。

{simtable}

¹ **这里只是采样器序列模拟，无网络、无新图像，不计入数据集数量。** 10k/50k调用覆盖及最大重复如实保留，重复调用不是新原图或新场景。B的50k覆盖所有3898个有用途来源；M2的819来源包含解析负例，地面描述子来源仍是677。高重复来自稀少组件/角色，不能宣传为独立数据增加；每次online可改变合成关系，也不能据此增加原始独立来源计数。

## 6. 验收与验证边界

全量实际读回{qa['checked_pair_records']}条配对记录/{qa['checked_view_instances']}幅图，H及逆H、双向坐标、像素强度、有效/船/ignore/unknown/校准掩膜、局部点、全局支持和舰船辅助一致性全部通过。{len(interface['tensor_cases'])}个分层tensor实例通过；M2与B各16次CPU角色前向，CUDA未初始化、优化器步数0。32次在线生成完成图像、H、掩膜和点位精确回放，未发生未记日志的替换。CPU损失值仅接口检查，不是模型性能证据。

历史V1标准、V2.1 R/S全部Val及133组共同结构Val原文件保留；受保护的{len(protected)}个旧入口/代码/Val manifest摘要前后相同。新增Val仅原Val组件，四类和所有难度保留；有限结构只报告局部重复性/描述检索/已知点局部误差，不强制算全幅Success。旧133必须用原reader与原协议，不能换成本版标签。所有真实512、人工记录、旧训练和checkpoint保持不变。本版没有真实独立物理GT，不证明真实跨雷达方位泛化。

## 7. 交付与仍然存在的瓶颈

本版具备启动较充分的SuperPoint受控训练条件：背景描述训练关系与来源都已增加，有限结构不再丢弃，舰船辅助可独立开关，有证据校准和无重叠分开消费。推荐先R/core；保留同图准入子集和new_ROI子集用于归因，S与船辅助分开消融，SSDD另报。下一步由算法Codex在新run中接入reader/loss与固定验证，先记录数据/代码/checkpoint/预算及完整在线日志，再开展用户授权的训练。

主要限制仍是源图内容偏海面、未标船/尾迹、粗场景分类、显示DN及共享原始散斑、少数组件重复较高、无真实跨采集物理GT。自动候选不是人工确认静态地物；没有为了把海面数量转成正GT而制造结构或降低全局评价标准。

{images}
'''
    (ROOT/f'DATASET_V22_REPORT_{t}.md').write_text(report,encoding='utf8')
    usage=f'''# V2.2给算法Codex的直接入口

**可训练；默认R/core，船辅助关闭。**

入口：`{entrypath}`。完整分级/冻结说明见同目录报告与协议。环境match，batch1或尺寸分桶，不隐式resize，不修改旧run。

```python
import sys,json
from pathlib import Path
sys.path.insert(0,{str(ROOT/'code')!r})
from data22 import load_pair,tensors22
from loss22 import m2_loss,b_loss
e=json.loads(Path({str(entrypath)!r}).read_text())
data=[json.loads(s) for s in Path(e['recommended_train_local']).read_text().splitlines()]
z=load_pair(data[0],use_ship_aux=False)
x=tensors22(z,device='cpu') # only x['images'] is model input [2,1,H,W]
# model/teacher are the existing M2 or B, loaded in the NEW algorithm run:
# loss,info=m2_loss(model,z,device=device)
# loss,info=b_loss(model,teacher,z,device=device)
```

`points0/1`为地面局部对应，可能1–128点；>=4才描述CE，>=1可供B检测，`global_eligible`独立保存，不能用局部资格改全局评价。`static`是允许的候选支持，精确监督以保存点为准；`calibration_mask`只标有证据抑制，`supervision_ignore`为其余未知区域，`ignore`是动态/遮挡/局部增强损坏。不要从safe_static或所有非角点区域重造正/负GT。

`ship_points0/1`和`ship_aux_masks`独立，NPZ保留实例索引；只有显式use_ship_aux=True才送舰船点给损失。候选mask不能代替已通过双图可见性检查的配对点，更不能变成静态地面GT。船辅助默认权重.15，负对应margin .8/权重.1，地面局部权重min(1,N/16)。源图角点和ROI选择都在增强前、未用任何M2/B匹配结果。

M2冻结检测头，有证据低纹理不应被伪装成描述正例；`skip_descriptor_update=True`时显式跳过该描述更新。M2抽样模式不强占低纹理检测预算，B才训练该分支。无重叠有独立negative_probe_points，解析FOV保证非对应；它们不是稳定地面正锚点。B适配器保留旧网络/损失构件，替换固定256/384标签假设，并按本版角色消费局部点和校准区，unknown不会被强制抑制。适配器不是历史B分数的无变化重现。

## 四类固定与自然来源

`manifests['V2-R/train/core/all']`保留全部角色；`.../natural`每个不同原始文件一个固定关系，无成功选择。角色后缀structural_positive、limited_structure、calibration_evidence、no_overlap；`local_descriptor`>=4、`local_detection_only`1–3；`optional_ship_aux`为独立选项。Val同样路径结构，SSDD改core为ssdd单列。角色来源有交集，不能相加当原采集数。

## 分层抽样和新在线配对

```python
from sampling22 import BalancedSampler,Online22,replay22
from common22 import rows,stamp
reference=e['recommended_train_reference']
sampler=BalancedSampler(rows(reference),mode='B',seed=202622)
row,reference_trace=sampler.get(0) # frozen-pair training option
online=Online22(e['config'],reference,f'/tmp/sar22_worker0_{{stamp()}}.jsonl',
                mode='B',seed=202622,arm='V2-R',use_ship_aux=False)
try:
    z=online.get(step=0,epoch=0,visit=0)
except ValueError:
    # rejection already logged; count it, advance schedule, never retry for an easier positive
    z=None
```

B配额40/25/27/8；M2模式55/35/0/10。每worker独立日志，调用step/epoch/visit在run内保存，不能多worker重复同一个step而隐瞒重放。参考角色与新生成实际角色可能不同，始终按z.loss_role和有效点数计算，不照搬参考标签；失败、参数、source SHA、H、图像/掩膜/点摘要都记日志。Val不传入BalancedSampler/Online22。

重放：读取日志identity核对config/reference/code与release身份，再调用`replay22(event,source_record,cfg)`；source_record来自原`SOURCES_V2_20261015_1258.jsonl`，本版source feature缓存与config语义ROI必须保留。已32次实测精确重放。10k/50k采样覆盖日志仅是抽样模拟，不是模型训练或新视图数量。

## 冻结验证与归因实验

`frozen_validation`指向原V1、V2.1 Val和133共同集，reader路径单列，全部不覆盖。新增`recommended_val_all`包含充分/有限/低纹理/无重叠；按global_eligible分开全幅/局部指标，不能丢掉困难样本或把无结构改负例。SSDD额外报告，不能称全体原始采集绝对独立。

`ablation_subsets`含同像素恢复局部、同像素同旧selector恢复局部、仅新增ROI等子集，配合旧V2.1入口用于说明门槛与裁片贡献。B还改正了未知区校准，所以不是只有门槛一个因素；训练需固定预算、来源调度和新验证，在新目录记录权重/代码/数据身份，不直接套历史结果。

无需再生成数据：{len(rrows)}组R已准备好，core Train描述正例2324组/677源，core Val局部描述322组；全幅仍按原条件。直接交给算法任务，数据任务没有启动训练、改权重或真实数据。
'''
    (ROOT/f'DATASET_USAGE_V22_{t}.md').write_text(usage,encoding='utf8')
    # New scientific artifacts only; no second mass SHA pass over old SAR/checkpoints.
    paths=[entrypath,args.summary,qa_path,ipath,Path(summary['config']),protocol]+list((ROOT/'code').glob('*.py'))+[Path(p) for p in summary['manifests'].values()]+[Path(p) for p in subsets.values()]
    save(ROOT/f'RELEASE_IDENTITY_V22_{t}.json',dict(at=now(),sha256={str(p.resolve()):sha(p) for p in paths},protected_old_files=len(protected),old_files_changed=[],old_imagery_or_weights_mass_rehashed=False,
        appearance_config_unchanged=True,geometry_ranges_unchanged=True,network_training=False,physical_GT_created=0))
    save(ROOT/f'FINAL_COUNTS_V22_{t}.json',dict(at=now(),entrypoint=str(entrypath),R_pairs=len(rrows),R_views=2*len(rrows),R_roles=roles,R_ground_descriptor_pairs=summary['R_local_descriptor_pairs'],
        R_ship_aux_descriptor_pairs=summary['R_optional_ship_descriptor_pairs'],core_train_local_pairs=2324,core_train_ground_sources=after,
        original_source_files=summary['R_sources'],new_independent_raw_acquisitions=0,accepted_new_ROI_pairs=new_count,strict_old_reused=old_reuse_full,strict_new_ROIs=new_full,report=str(ROOT/f'DATASET_V22_REPORT_{t}.md')))
    (ROOT/f'README_{t}.md').write_text(f'# Synthetic SAR V2.2\n\n可训练。权威入口：[{entrypath.name}]({entrypath.name})。\n\n[报告](DATASET_V22_REPORT_{t}.md) · [使用说明](DATASET_USAGE_V22_{t}.md)。\n\nR {len(rrows)}配对/{2*len(rrows)}视图，地面描述2851组；core Train2324组/677源。船辅助默认关闭，旧133集与V1/V2.1全部保持。不要重建数据或重启旧训练。\n',encoding='utf8')
    print(json.dumps(dict(entry=str(entrypath),report=str(ROOT/f'DATASET_V22_REPORT_{t}.md'),figures=figures,strict_new_ROIs=new_full,strict_old_reused=old_reuse_full),ensure_ascii=False))
if __name__=='__main__':main()
