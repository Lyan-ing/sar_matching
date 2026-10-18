"""Freeze one bounded ROI pass and training-only thresholds before new generation."""
from data22 import *
from collections import defaultdict,Counter

def iou(a,b):
    a=np.asarray(a,np.float32);b=np.asarray(b,np.float32)
    inter,_=cv2.intersectConvexConvex(cv2.convexHull(a),cv2.convexHull(b));union=cv2.contourArea(cv2.convexHull(a))+cv2.contourArea(cv2.convexHull(b))-inter
    return float(inter/max(union,1e-6))

def main():
    t=stamp();cfg=copy.deepcopy(CFG21);cfg.update(version='2.2',created_at=now(),arms=['V2-R','V2-S'],semantic_overrides={},max_new_relations=1600,max_new_ROIs_per_source=2,
        supervision22=dict(local_descriptor_min=4,local_detector_min=1,local_weight='min(1,N/16), global positive weight1',
            strict_global_unchanged=True,threshold_sensitivity=[4,8,12,16],calibration='sea evidence AND pre-augmentation local std<.010, eigen<.00002, mean>.005, no ships/structures',
            ship_aux_default=False,ship_aux_weight=.15,ship_aux_segmentation_only=True,ship_aux_pair_spacing_px=8),
        sampling22=dict(weights=dict(structural_positive=.40,limited_structure=.25,no_structural_positive=.27,no_overlap=.08),hierarchy='role->source->ROI->relation; uniform cycling within each level',
            role_is_reference_not_guaranteed_online=True,cover_all_eligible_sources_before_replacement=True),
        roi22=dict(max_old_or_new_native_FOV_IoU=.70,min_center_distance_fraction=.25,selection='pre-augmentation persistent source corners, no model outputs',no_postlabel_recrop=True))
    oldreview=rows(ROOT/'STRATIFIED_REVIEW_INDEX_20261015_1400.jsonl');extra=rows(ROOT/'BACKGROUND_REVIEW_INDEX_20261015_1401.jsonl')
    approved={61:[[450,300,790,790]],62:[[450,400,790,790]],63:[[650,0,799,600]],64:[[250,400,630,790]],65:[[220,200,650,340]],67:[[480,400,790,790]]}
    approved_extra={0:[[590,0,790,340]],1:[[0,0,250,150]],22:[[280,0,790,790]],23:[[300,0,790,790]],24:[[300,0,790,700]],25:[[40,250,500,700]]}
    decisions=[]
    for group,items,selection in [('stratified',oldreview,approved),('background',extra,approved_extra)]:
        for i,r in enumerate(items):
            approved_here=i in selection
            decision=dict(source_path=r['source_path'],review_set=group,review_index=i,approved=approved_here,
                boxes_xyxy=selection.get(i,[]),reviewer='Codex AI provisional source-area evidence, not human GT',
                reason='bounded visible land/industrial boundary/road/breakwater region' if approved_here else 'ship/wake/sea texture or insufficient fixed-ground evidence; leave ground supervision unknown',
                physical_GT=False)
            decisions.append(decision)
            if approved_here:cfg['semantic_overrides'][r['source_path']]=decision
    jsonl(ROOT/f'SEMANTIC_DECISIONS_{t}.jsonl',decisions)
    save(ROOT/f'CONFIG_V22_{t}.json',cfg)
    src=sources();sourcemap={r['image_path']:r for r in src};previous=old_rows();windows=defaultdict(list)
    for r in previous:
        for poly in r['source_window_corners']:windows[r['source_path']].append(np.asarray(poly))
    candidates=[];skip=Counter()
    for r in sorted(src,key=lambda r:stable_seed(22,r['image_path'])):
        f=features(r);p=ground_points(r,f,cfg)
        if len(p)<4:skip['source_ground_below4']+=1;continue
        h,w=f['source_safe'].shape
        accepted=[]
        for roi_no in range(2):
            # Size/difficulty assigned independently BEFORE quality results, never lowered to make a positive.
            size=[256,384,512][stable_seed(22,r['image_path']+f':roi{roi_no}')%3]
            size=min(size,int(min(h,w)/1.48/32)*32)
            margin=.60*size+20
            centers=[]
            for y in np.linspace(min(margin,h/2),max(h-margin,h/2),5):
                for x in np.linspace(min(margin,w/2),max(w-margin,w/2),5):
                    center=np.array([x,y]);count=int(np.all(np.abs(p-center)<.42*size,axis=1).sum())
                    if count>=4:centers.append((count,center))
            for _,center in sorted(centers,key=lambda x:-x[0]):
                half=size/2;poly=center+np.array([[-half,-half],[half,-half],[half,half],[-half,half]])
                similar=False
                for oldpoly in windows[r['image_path']]+[a[1] for a in accepted]:
                    oldside=np.sqrt(max(cv2.contourArea(np.asarray(oldpoly,np.float32)),1.))
                    if iou(poly,oldpoly)>.70 or np.linalg.norm(center-np.mean(oldpoly,axis=0))<.25*min(size,oldside):similar=True;break
                if similar:continue
                accepted.append((center,poly,size));break
            else:skip['no_distinct_ROI_size_'+str(size)]+=1
        for j,(center,poly,size) in enumerate(accepted):candidates.append(dict(source_path=r['image_path'],split=r['split'],center=center.tolist(),requested_size=size,roi_native_bounds=poly.tolist(),ordinal=j))
    # One first ROI per source, then second ROIs; cap is fixed and not expanded after seeing outcomes.
    candidates=sorted(candidates,key=lambda r:(r['ordinal'],stable_seed(22,r['source_path'])))[:cfg['max_new_relations']]
    plan=[]
    for i,item in enumerate(candidates):
        spec=v21.make_spec(1000000+i,stable_seed(202622,item['source_path']+str(item['center'])),cfg,
            forced=dict(roi_center=item['center'],requested_size=item['requested_size'],roi_id=f'roi{i:05d}'))
        plan.append(dict(**item,spec=spec))
    jsonl(ROOT/f'FROZEN_ROI_PLAN_{t}.jsonl',plan)
    save(ROOT/f'PLAN_IDENTITY_{t}.json',dict(at=now(),config=str(ROOT/f'CONFIG_V22_{t}.json'),config_sha=sha(ROOT/f'CONFIG_V22_{t}.json'),plan=str(ROOT/f'FROZEN_ROI_PLAN_{t}.jsonl'),plan_sha=sha(ROOT/f'FROZEN_ROI_PLAN_{t}.jsonl'),
        planned_new_pairs=len(plan),new_source_files=len({r['source_path'] for r in plan}),skip_reasons=dict(skip),source_component_split_unchanged=True,model_results_used=False,
        existing_V21_views_not_regenerated=True,before_new_generation_and_network_training=True))
    protocol='''# V2.2分级监督与新增验证口径

正式算法训练前固定。原V1标准、V2.1 R/S以及133组三臂共同验证全部原样保留，不替换旧分母或分数。

充分结构：沿用双方≥16点、4×4网格≥4格、凸包≥5%、协方差特征值比≥.03。只这类具有全幅Homography评价资格；仍只代表同源二维合成。

有限结构：保留至少1个经原图sigma2/4持久结构检查的双向有效角点，用于局部检测；描述子至少4点，观察4/8/12/16准入敏感性。分级是训练监督设计，不降低最终全局/真实GT评价门槛。有限结构不计入全幅Success正例。

低纹理/无结构：没有接受角点不等于区域是负例。仅海面语义或可靠海陆掩膜并有原图低粗纹理/低角点强度/非纯零证据的区域可做抑制；其余有效区记supervision_ignore。新增噪声不参与角点选择。

无重叠：解析双向覆盖为0才准入；不把有限结构、低纹理或算法失败改成无重叠。负探针仅表示解析无对应，不是稳定结构正点。

舰船辅助：只用原图分割标注内、双图共同可见、未被动态/增强ignore破坏、双侧间距≥8px的图像角点，单独mask/点集，默认关闭。BBox-only、动态区和不充分点数不填补。不会并入静态地面GT，也不是跨真实采集物理同名点。

新增ROI一次冻结，每源最多2个、全局最多1600关系；相对既有视场IoU≤.70且中心距离≥较小边长.25。尺寸/角度/profile事先分配；失败与低监督保留，不降低角度、重新裁切或追加生成凑比例。R/S外观使用已审计1309配置，未作新噪声调参。

新增Val只用原Val组件；报告充分/有限/低纹理/无重叠、场景/角度/尺寸/重叠分层，SSDD单列。局部误差/描述子检索/检测重复性与全幅Success分开。标签均模型无关的合成代理；无真实独立物理GT产生。
'''
    path=ROOT/f'SUPERVISION_PROTOCOL_{t}.md';path.write_text(protocol,encoding='utf8')
    print(json.dumps(dict(config=str(ROOT/f'CONFIG_V22_{t}.json'),plan=str(ROOT/f'FROZEN_ROI_PLAN_{t}.jsonl'),new_pairs=len(plan),skip=dict(skip)),ensure_ascii=False))
if __name__=='__main__':main()
