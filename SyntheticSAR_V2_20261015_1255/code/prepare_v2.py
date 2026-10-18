"""Reuse V1 inventory; bounded scene review and preregister V2 settings before generation."""
from common_v2 import *
from collections import Counter
from PIL import Image
def main():
    t=stamp(); pool=rows(V1/'SYNTHETIC_SOURCE_POOL_20261014_2142.jsonl')
    usable=rows(V1/'SYNTHETIC_train_all_known_components_USABLE_manifest_20261015_0044.jsonl')+rows(V1/'SYNTHETIC_val_all_known_components_USABLE_manifest_20261015_0044.jsonl')
    allow={r['source_path'] for r in usable}; pool=[r for r in pool if r['image_path'] in allow]
    review=rows(ROOT/'SCENE_REVIEW_INDEX_20261015_1255.jsonl')
    # Explicit visual decisions on displayed contact sheets, AI provisional, never human labels.
    ports=set([0,1,2,3,4,5,6,7,8,9,12,13,14,16,17,18,19,20,21,23,25,26,27,28,29,30,31,33,34,38,39,40,41,42,43,44,46,49,50,51,52,53,54,55,56,57,58])
    ports.update(range(60,99)); ports.update(range(100,151))
    ports.update([154,155,157,158,160,161,163,164,165,166,167,168,169,170,171,172,173,174,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,203,204,205,206,207,208,209,211,212,213,214,216,217,218,219,220,221,223,224,225,226,227,228,229,230,231,232,233,235,236,237,238,239])
    ports.update([393,394,395,396,397,398,400,401,402,403,404,534,586,587,589,597])
    land=set([241,242,243,244,245,246,247,251,278,300,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,338,339,340,341,342,343,344,345,346,347,348,349,350,351,353,405,406,407,408,409,410,411,412,413,414,415,416,417,418,420,421,422,423,424,426,427,428,429,431,432,433,434,435,436,437,438,439,440,441,442,443,444,445,446,447,448,449,450,451,452,453,454,455,531,535,536,537,539,540,541,542,543,544,545,546,547,548,550,551,552,553,554,556,561,579,580,581,584,585,588,590,591,592,593,594,595,596,598])
    mapping={r['image_path']:(i, 'port_quay_dense_vessels' if i in ports else 'other_land_structures' if i in land else None) for i,r in enumerate(review)}
    decisions=[]
    for r in pool:
        if r['image_path'] in mapping:
            i,label=mapping[r['image_path']]
            if label:
                r.update(scene_category=label,scene_category_status='AI_visual_provisional_source_context_not_crop_or_human_GT',scene_review_index=i)
                decisions.append(dict(source_path=r['image_path'],review_index=i,scene_category=label,split=r['split'],evidence='fixed coast/docks/industrial blocks or land relief visible in source contact sheet',status=r['scene_category_status']))
        r['data_units']='display_uint8_DN_no_recoverable_per_image_radiometric_transfer'
    cfg=dict(version='2.0',created_at=now(),seed=20261015,arms=['V1-matched','V2-R','V2-S'],
        source_pool=str(ROOT/f'SOURCES_V2_{t}.jsonl'),source_parent_split_unchanged=True,
        original_speckle_fully_removed=False,physical_independent_SAR_observations=False,
        geometry=dict(positive_angle_bins=[0,15,30,60,90,120],positive_angle_weights=[.2,.2,.3,.2,.1],
            normal_overlap=[.5,.9],low_overlap=[.12,.3],low_probability=.10,negative_probability=.04,
            high_angle_scale=[.96,1.02],small_angle_scale=[.92,1.06],perspective_fraction=.002,
            requested_sizes=[256,384,512],no_source_preupsampling=True,min_valid=.90,max_attempts=80),
        appearance=dict(domain='display_DN_only',smooth_residual_blend=[.35,.65],bilateral_sigma_color=.06,bilateral_sigma_space=1.2,
            gamma=[.92,1.08],contrast=[.94,1.06],gain=[.95,1.05],offset=[-.008,.008],
            gamma_field_shape_proxy=[8,24],display_noise_sigma=[.012,.030],correlated_noise_mix=[.0,.5],correlation_sigma=[.35,.65],
            clutter_sigma=.012,anisotropic_blur_sigma=[.15,.50],
            local_scattering_probability=.65,local_blobs=[1,3],blob_sigma=[3,6],blob_gain=[.55,1.35],visibility_dropout_probability=.3,
            local_change_max_fraction=.035,dynamic_ship_probability=.2),
        structure=dict(method='source persistent multiscale corners sigma2/4; absolute coarse contrast; semantic sea veto',
            source_corner_min_distance=12,multiscale_tolerance=4.5,source_ship_guard=10,view_ship_guard=5,
            source_corner_min_eigenvalue=.00008,source_local_std_min=.025,support_radius=8,
            pair_min_points=16,pair_min_cells4x4=4,pair_min_hull=.05,pair_min_eigen_ratio=.03,
            unknown_background_supervision=False,offshore_supervision=False,mask_is_not_physical_GT=True),
        scene_supplement=dict(additional_train_port=600,additional_train_land=500,additional_train_nearshore=300,
            additional_val_port=90,additional_val_land=70,additional_val_nearshore=40,
            counts_are_augmentation_relations_not_new_sources=True),
        acceptance=dict(constant_valid_input='quarantine',valid_std_min=.5,saturated_fraction_max=.30,
            low_supervision_retained_for_calibration=True,common_arm_gate=True,geometry_tolerance_px=.001),
        pixels='OpenCV integer pixel centers, xy, full-original-source->view H; bilinear images; conservative four-neighbor validity',
        frozen_900_used=False,no_network_training=True)
    jsonl(ROOT/f'SOURCES_V2_{t}.jsonl',pool);jsonl(ROOT/f'SCENE_DECISIONS_{t}.jsonl',decisions);save(ROOT/f'CONFIG_V2_{t}.json',cfg)
    # Representative file units inspection, not a repeated large inventory/SHA scan.
    units=[]
    for dataset in ['hrsid','background','ssdd']:
        for r in [q for q in pool if q['dataset']==dataset][:3]:
            with Image.open(r['image_path']) as im:units.append(dict(dataset=dataset,path=r['image_path'],mode=im.mode,metadata=im.info,shape=im.size,calibration_metadata_present=False))
    stats=dict(at=now(),sources=len(pool),scenes=dict(Counter(r['split']+'/'+r['scene_category'] for r in pool)),units=units,
        v1_pair_count=len(usable),v1_profiles=dict(Counter(r['profile'] for r in usable)),v1_angles=dict(Counter(angle_bin(r['image_rotation_2D_deg']) for r in usable if r['profile']!='none')),
        v1_dynamic_requested=sum(r['profile']=='dynamic' for r in usable),v1_dynamic_executed=sum(r['synthetic_dynamic_events']>0 for r in usable),
        v1_note='V1 static matches are random analytic points outside ships, not repeatable structural anchors; raw source speckle shared',
        reviewed_source_contexts=len(decisions),AI_visual_review_not_human=True)
    save(ROOT/f'V1_AUDIT_{t}.json',stats)
    (ROOT/f'V1_ISSUES_{t}.md').write_text('''# V1检查与V2决定

V1可用5948关系、11896视图保持不变。实际读取生成器、配置和最终manifest，不重新扫描全部原始文件。

1. 正例小旋转占主导；V2按20/20/30/20/10目标分层，负例单列；高角度绑定较小尺度与高重叠。
2. V1先warp，再gamma、对比度/偏置、独立lognormal乘噪声、clip及可选Gaussian模糊；独立随机数不消除共享原始散斑。
3. 当前HRSID JPG、背景PNG和SSDD JPG样例均为8-bit RGB发布裁片。无逐图幅度/强度标定、dB参数或可逆显示函数，不能证明物理域。V2只在归一化显示DN域做有界统计代理；Gamma随机场参数不是实测ENL。
4. V1动态profile275个可用关系中实际214个执行移动；其余未移动不能算动态执行。全量舰船（含部分可见）排除修正在V1已存在，V2继续复用annotation_mask并增加保护区。
5. V1解析随机点即使排除舰船仍可落在纯海面散斑。V2强制分离安全静态区、结构支持区与校准区；离岸及未分类背景不给匹配正点。
6. 场景为来源语境，不能冒充裁片语义。新增AI来源预览审查、记录可信度；SSDD海比例只依赖已有SL-SSDD。
7. V1历史验证图像、GT及其旧监督口径冻结。另提供同几何V1-matched/V2-R/V2-S新对照及结构监督，不篡改历史分数。

V2先推荐R。S的局部变化全部进入ignore及双向可见性排除，仅作为受控消融。不声称物理独立观测或真实雷达方位变化。
''',encoding='utf8')
    print(json.dumps(dict(config=str(ROOT/f'CONFIG_V2_{t}.json'),scenes=stats['scenes'],reviewed=len(decisions)),ensure_ascii=False))
if __name__=='__main__':main()
