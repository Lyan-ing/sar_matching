# 代码目录与依赖说明

各时间戳目录保留原有模块边界，便于旧源码中的相对导入继续解析。未打包任何生成的数据目录、源码下载镜像、权重或实验报告；图表生成程序属于代码，生成出的图表不属于发布内容。

|模块|优先查看|
|---|---|
|`DualSAR_DatasetPrep_20261014_2131/`|code/|
|`MaritimeSARRegDataset/`|scripts/|
|`MaritimeSARRegistrationPhase2/`|scripts/|
|`RealSAR_AngleExpansion_20261012_1641/`|code/|
|`RealSAR_AutoLabels_20261011_1149/`|模块根目录|
|`RealSAR_ClearCandidates_20261016_0132/`|code/|
|`RealSAR_FailureDiagnosis_20261014_1305/`|code/|
|`RealSAR_Fixed512_20261014_1743/`|code/|
|`RealSAR_GeoAssist_20261012_1917/`|code/|
|`RealSAR_GeometryGT_20261011_0220/`|模块根目录|
|`RealSAR_IndependentBenchmark_20261011_2015/`|模块根目录|
|`RealSAR_MatchingPilot_20261011_2114/`|模块根目录|
|`RealSAR_MultiSource_512_Kit/`|模块根目录|
|`RealSAR_MultiViewV2_20261012_1259/`|code/|
|`RealSAR_Multiview512_20261015_1534/`|code/|
|`RealSAR_PairingAudit_20261011_2349/`|code/|
|`RealSAR_PhysicalPrep_20261013_2219/`|code/|
|`RealSAR_RepresentationContext_20261011_1903/`|模块根目录|
|`RealSAR_RotationStudy_20261011_1745/`|模块根目录|
|`RealSAR_Sentinel1_DownloadKit/`|模块根目录|
|`RealSAR_SingleExpert_20261015_2332/`|code/|
|`Report_Figures_Ch4_FangSong_20261017_2003/`|code/|
|`Report_Figures_Ch4_Optimized_20261017_1338/`|code/|
|`Report_Figures_Mixed640512_20261016/`|code/|
|`SAR_ORB_SP_Fusion_20261015_2349/`|code/|
|`SAR_SP_JointTransfer_Mixed640512_20261016_2234/`|code/|
|`SAR_SP_KeepAblation_20261017_1343/`|code/|
|`SAR_SP_LG_Acceptance_20261012_1735/`|code/|
|`SAR_SP_LG_M2_Numerics_20261012_1627/`|code/|
|`SAR_SP_LG_M3_Detector_20261012_1932/`|code/|
|`SAR_SP_LG_Matchability_20261014_2031/`|code/|
|`SAR_SP_LG_StructureStable_20261013_2210/`|code/|
|`SAR_SP_LG_Train_20261012_1544/`|code/|
|`SAR_SP_LG_TrainingReview_20261012_1651/`|code/|
|`SAR_SP_LG_V22_20261015_1525/`|code/|
|`SAR_SP_LowTextureKeep_20261017_0908/`|code/|
|`StorageMigration_20261017_0846/`|模块根目录|
|`SyntheticSAR_Mixed640512_Multicondition_20261016_2114/`|code/|
|`SyntheticSAR_V22_20261015_1352/`|code/|
|`SyntheticSAR_V2_20261015_1255/`|code/|
|`real_sar_pilot/`|模块根目录|
|`sar_match_portable/`|scripts/|
|`tools/`|模块根目录|

## 运行依赖

基础算法环境见根目录requirements-match.txt；独立推理见sar_match_portable/requirements.txt。真实数据处理按各工具包requirements选择性安装。所有本机Python开发与检查使用match环境，GPU任务执行前检查占用，不运行未知历史启动脚本。

## 不随Git交付的输入

来源索引、场景划分、固定配对、GT、人工点、图像和checkpoint均需从项目已授权存储单独获取。训练配置和代码中的路径只是入口声明；路径存在不代表仓库包含相应数据。未取得GT的外部数据只支持工程一致性诊断。

## 发布维护

原服务器的.code-publish是独立发布检出，不是训练目录。tools/sync_code_publish.py按白名单复制源代码，不跟随数据软链接，也不提交或推送。发布后的README、本地依赖准备脚本和只含源码的测试适配应保留；再次同步后需复核差异，不能无审查把缓存或实验产物加入Git。
