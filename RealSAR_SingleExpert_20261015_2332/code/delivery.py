"""Short instructions and evidence-linked final entry; all report names timestamped."""
from expert import *

def main():
 t=stamp();d=Dataset();s=Store(d);assert not s.docs,'Do not replace the zero-label preparation report after actual annotation begins'
 prep=json.loads(sorted(ROOT.glob('PREPARATION_*.json'))[-1].read_text());cfg=load_config();exports=sorted(Path(cfg['exports']).glob('export_*/EXPORT_*.json'));last=exports[-1];x=json.loads(last.read_text());assert x['actual_original_correspondences']==0
 checks={p.stem:json.loads(p.read_text()) for pattern in ['SELFCHECK_*.json','BROWSER_CHECK_*.json','BROWSER_PERSISTENCE_*.json'] for p in (ROOT/'evidence').glob(pattern)}
 assert all(v['passed'] for v in checks.values())
 usage=ROOT/f'QUICK_START_{t}.md';report=ROOT/f'SINGLE_EXPERT_DELIVERY_REPORT_{t}.md';entry=ROOT/f'SINGLE_EXPERT_ENTRYPOINTS_{t}.json';protocol=json.loads(Path(cfg['protocol']).read_text())
 with usage.open('x') as f:f.write(f'''# 单人SAR物理点标注：直接开始

地址：**http://127.0.0.1:8774/**。在VS Code内置浏览器打开即可。无需人员登记、A/B包或CSV关联。

1. 点“开始首轮30组”。本轮先做20～30组；40组B候选也可单独浏览。全数据集1033组可以按站点、角度、关系、场景、Train/Val和状态筛选。
2. 左图点同一结构的一侧，右图点另一侧。滚轮缩放，Shift/中键/右键拖动；“本512”回到目标裁片区域，“全景”查看现有较大上下文。亮度、对比度、当前窗口局部拉伸只改变显示。
3. 填结构类型、结构组、简单物理依据、置信度和两侧定位不确定度；如实确认是否同一固定地面结构、风险是否排除、局部二维是否适用。按Enter或“确认对应点”。没有6点也可保存；模糊或有风险的观察保留，但不自动成为可用GT。
4. 点击会自动保存**草稿**；确认对应点后保存**点记录**并自动更新GT。右上角显示“磁盘已保存 v…”。若失败会明确报错并保留浏览器备份，不能把失败当成功。不要清除未落盘时的浏览器存储。
5. 黄色为已保存点或合法裁片复用；粉色为当前草稿。点击已保存点可以修改、删除；撤销保留历史。重开同一任务URL或在总览选择样本，自动读取点和草稿。
6. 选择“较大Native上下文”可以直接标原图。保存的是原始产品像素坐标；只有两侧都在已冻结512内、同一采集与产品网格、有效掩膜及原始像素内容一致时才自动产生局部对应。不会新裁片、旋转、地理投影或补点。
7. “复用本采集已确认的单侧位置”只能引用那一侧；另一侧仍需你定位，最后亲自确认同一物理结构。复用依赖保留，原始点修改后可能需要重新确认。
8. 按“标注完成 / 部分完成 / 无法判断”登记状态；无法判断写简单原因，不勉强加点。“保存状态并下一组”继续当前30组队列。

高价值Val可选“延迟复标”，默认间隔至少24小时。只显示结构文字，隐藏第一次点位/相机/单侧库；重新点击仍属你一人的重复性检查，不称为双人独立核验。两次原始坐标保留，不平均。

每次确认点、修改/删除或改变任务状态后自动导出。右上角“更新GT导出 / 覆盖统计”可查看并下载统一清单、原始点和Train/Val覆盖CSV，不需手动回收文件。导出在`{ROOT/'exports'}`，每版均保留。草稿自动保存不代表草稿点已取得GT资格。

如服务未运行，从项目根目录执行一条命令：

```bash
/home/yl/miniconda3/envs/match/bin/python -B /home/yl/matching/sar_matching/RealSAR_SingleExpert_20261015_2332/code/start.py
```

CPU服务仅用8774。已运行时会返回地址；端口被其他服务占用时拒绝启动，不停止其他进程。不改变8772/8773。
''')
 latestbrowser=sorted((ROOT/'evidence').glob('BROWSER_CHECK_*.json'))[-1];b=json.loads(latestbrowser.read_text());gallery=Path(b['screenshots'][0]);native=Path(b['screenshots'][1])
 with report.open('x') as f:f.write(f'''# 真实SAR单人专家标注工具交付

生成：{now()}。地址：**http://127.0.0.1:8774/**。先点“开始首轮30组”，操作见[{usage.name}]({usage.name})。

## 已完成

沿用1033组冻结原生512：Train747、Val286、137个真实采集关系、20站点，未修改影像、偏移或划分。全部1033组有双图缩略图与筛选入口；40组B候选优先。首轮30组覆盖{len({d.tasks[k]['relation_id'] for k in d.pilot})}关系、{len({d.tasks[k]['component'] for k in d.pilot})}站点，其中11组Val。B与Val固定结构仍是候选，需你亲自判断。

双图同时显示，支持原生坐标点击、滚轮缩放、拖动、当前窗口对比拉伸、修改/撤销/删除、完成/部分/无法判断。草稿和已确认点分别自动落盘，保留所有版本；页面重开可读回。实际写入及读回成功才显示保存成功；磁盘/网络失败保留浏览器草稿并报错。没有身份登记、双人分发或手动CSV关联。

{prep['Native_context_pairs']}组双侧具备大于512的本地Native上下文，涉及{prep['Native_context_relations']}关系。其余只能使用已有本地512或可用单侧上下文，不重新下载。Native操作读取现有缓存/图像，原生细节按不超过1024的窗口显示；两幅大图显示缓存与概览缓存有数量上限，点击只检查小块有效像素。

## Native复用与证据

0基像素中心：`产品坐标 = 当前原生窗口坐标 + 已存产品窗口偏移`；`512局部坐标 = 产品坐标 − 该512原点`。没有地理或算法映射。每侧必须同一真实采集、同产品网格、在有效范围、2×2支持像素值完全一致才转写到对应裁片；两侧任一不满足就不生成该裁片对应。

同一真实关系支持两侧顺序相反的裁片，按真实采集ID交换坐标、不确定度和输入来源，不能只按左右序号复制。共享采集而对手不同，只允许显式引用已存单侧人工结构坐标；另一侧仍由你标记并确认。原始点ID、版本、源Native坐标与自动裁片记录关联保存；一次Native点产生多条裁片记录不增加原始人工点数。源点修改会触发依赖检查。

## 单人GT分级

新协议：`{cfg['protocol']}`。原双人冻结协议SHA `{protocol['old_protocol_sha256']}` 保持不变，新记录始终`GT_qualified=false`、`independent_two_person_GT=false`。

|单人等级|要求与用途|
|---|---|
|点级对应|保留专家实际确认的固定地面点；数量/覆盖不足也保留，不能宣称全图精度|
|局部GT|≥4个合格check、两侧凸包≥5%、双轴≥25%、≥3结构组/每组≤2、点距≥32px、≥3个4×4网格|
|全幅稀疏GT|≥6个合格check、两侧凸包≥10%、双轴≥50%、≥3结构组/每组≤2、点距≥32px、≥4网格/≥3象限|
|无可靠对应|你明确登记无法判断及原因，不把未标任务或纯海面自动设成负例|
|待标注/待补证据|尚未处理，或实际观察的结构/风险/不确定度仍待补全；处理状态另列统计|

合格点要求实际双侧点击/合法单侧引用、固定地面与可见性确认、具体结构/组/物理理由、中高置信、两侧明确不确定度和五类风险排除；局部/全幅还需明确二维适用。自动审核检查这些**专家声明**、坐标、掩膜、内容及空间支持，不能独立证明物体身份。高层屋顶、叠掩、阴影、船和散斑不因图像相似而变成物理GT。

单人不确定度U使用两侧sigma的平方和根；有实际复标时按sigma和观测分歧的最大值增大，不平均。仅U≤阈值/3支持对应3/5/10px成功门槛；没有门槛支持时不能给出该成功率。可选24小时延迟复标隐藏第一次点位和单侧库，保留两次原始坐标及分歧，仍是同一人的重复性证据。

`code/score_single_expert.py`提供独立的专家GT检查点读取和预测评分，接受已由纯图像产生的预测，不拟合/初始化预测。点级/局部/全幅分报，统计按真实关系/来源组件保留依赖，不能把派生裁片当成独立人工标注。最终模型输入仍仅双SAR图，不读取本工具元数据或GT。

## 验证与当前数据量

已完成1033组坐标往返、190次真实原生窗口/裁片像素支持核对、不同采集自动转换拒绝、原冻结空间规则和保护文件检查。实际Chromium验证1033/40B/Val/30队列、缩略图、Native原生高倍窗口、平移、局部对比、CSS缩放坐标以及复标原坐标隐藏。临时空草稿浏览器测试验证自动保存、重开读回、保存失败不报成功及备份恢复；生产SAR未被点击，没有测试物理点或模拟专家GT。

|分区|图像对|单人原始对应|点级/局部/全幅GT|原双人GT|
|---|---:|---:|---|---:|
|Train|747|0|0 / 0 / 0|0|
|Val|286|0|0 / 0 / 0|0|

当前覆盖：`{last}`。标注后每次变更自动生成新的专家manifest、原始点、复标、Train/Val清单、覆盖CSV和简短MD；历史版不覆盖。实际人工确认及真实通过率仍须由你完成首轮20～30组后才能知道。未模拟旧40辅助版本中的精确对应，旧人工记录和原始C001/C002/C003冻结材料均保持不变；本轮只读引用其图像和来源信息。

![单人工具实际浏览器截图（只读验收，未点击物理点）]({native.relative_to(ROOT)})

图：同屏原生上下文、高倍窗口及单人结构字段。截图为自动只读检查，不是人工标注证据。

## 边界与下一步

未下载、重裁、旋转预对齐、运行匹配器或训练；未修改合成V2.2、旧GT协议/标签/评价、原始SAR与Train/Val。新服务只用CPU和8774。

现在由你打开页面，优先标首轮20～30组：能确定几个点就保存几个，找不到则写原因。工具自动整理结果；真人标注后再检查实际字段、原生/裁片坐标、复用条数、误拒原因和单人GT通过率。没有真人点之前不宣称已经取得真实精度GT。
''')
 entrydata=dict(at=now(),url='http://127.0.0.1:8774/',root=str(ROOT),quick_start=str(usage),report_md=str(report),report_pdf=str(report.with_suffix('.pdf')),task_index=cfg['task_index'],single_expert_protocol=cfg['protocol'],old_two_person_protocol_unchanged=True,actual_original_correspondences=0,single_expert_GT=0,old_two_person_GT_qualified=0,initial_coverage=str(last),exports=str(ROOT/'exports'),observations=str(ROOT/'observations'),start_command=f'/home/yl/miniconda3/envs/match/bin/python -B {ROOT}/code/start.py',current_status_endpoint='http://127.0.0.1:8774/api/health',latest_export_endpoint='http://127.0.0.1:8774/api/export',automatic_GT_download='http://127.0.0.1:8774/api/export_file?kind=manifest',separate_scoring_code=str(ROOT/'code/score_single_expert.py'),read_only_checks=[str(p) for p in sorted((ROOT/'evidence').glob('*.json'))],source_entrypoint=cfg['data_entrypoint'],input_population=dict(Train=747,Val=286,total=1033,relations=137),native_context_pairs=prep['Native_context_pairs'],native_context_relations=prep['Native_context_relations'],pilot30=d.pilot,actual_human_required='user only; no second person or identity registration',GPU=False,old_services_modified=False)
 save(entry,entrydata)
 print('REPORT',report,'ENTRY',entry,'QUICK_START',usage,flush=True)

if __name__=='__main__':main()
