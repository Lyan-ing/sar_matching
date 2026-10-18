"""Evidence-linked final report and Kingston visual audit. No annotation inputs."""
import sys
sys.dont_write_bytecode=True
import csv, hashlib, html, json, platform, re
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import rasterio, cv2
from PIL import Image
import audit as a

ORIGINAL_CLASS={
 'S1SF_EXTENSION_000__chip_002':(3,'地理链路正确；同向大结构对照，但原512主要水面，精确核验仅考虑岸边局部。'),
 'Capella_0003__chip_001':(5,'B明确：原生地面尺度不同；C有异向及外观变化证据，机制尚未分离；没有A制作错误证据。固定高程GEC和散射位移仍需独立证据。'),
 'Umbra_0061__chip_001':(3,'地理链路正确、相近方向；局部边界候选，重复箱区/船舶/吊机不能直接当固定对应。'),
 'Umbra_0071__chip_000':(4,'地理链路正确；小地面范围、散斑/水面主导，当前512难以建立固定物理对应。'),
 'Umbra_0014__chip_002':(4,'地理链路正确、异向；矿区重复台阶、地形遮蔽和较长时间间隔，不足以确认稳定散射点。')}

def kingston(records, identities):
    sid='Capella_0003__chip_001';r=records[sid];folder=a.ROOT/'kingston';folder.mkdir(exist_ok=True)
    sources=[]
    for z in r['checks']:
        m=json.loads(Path(z['provider_metadata_file']).read_text());p=np.array(m['geometry']['coordinates'][0])[:,:2]
        xx,yy=a.transform('EPSG:4326','EPSG:32755',p[:,0].tolist(),p[:,1].tolist());sources.append(np.c_[xx,yy])
    origin=np.array(r['metric_polygons'][0]).mean(axis=0)
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained')
    for i,p in enumerate(sources):axes[0].add_patch(Polygon(p-origin,fill=False,edgecolor=['#0072B2','#D55E00'][i],linestyle=['-','--'][i],lw=2,label=['R provider acquisition footprint','M provider acquisition footprint'][i]))
    for i,z in enumerate(r['checks']):
        c=z['context'];info=z['native_info'];cp=Path(c['path'])
        with rasterio.open(cp) as ds:ci={'crs':str(ds.crs),'transform':list(ds.transform)[:6],'width':ds.width,'height':ds.height}
        p=a.metric_polygon(ci,'EPSG:32755')-origin
        for ax in axes:ax.add_patch(Polygon(p,fill=False,edgecolor=['#0072B2','#D55E00'][i],ls=['-','--'][i],lw=1.5,label=f'{["R","M"][i]} actual 1024 context'))
        p=np.array(r['metric_polygons'][i])-origin
        axes[1].add_patch(Polygon(p,fill=False,edgecolor=['#0072B2','#D55E00'][i],lw=2,ls=['-','--'][i],label=f'{["R","M"][i]} native512 chip'))
    for ax in axes:
        ax.autoscale();ax.set_aspect('equal');ax.set_xlabel('Easting offset (m)');ax.set_ylabel('Northing offset (m)');ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.02,1))
    axes[0].set_title('Original acquisitions + retained real1024 windows')
    axes[1].set_title('Real1024 contexts + native512 chip coverage')
    fig.suptitle('Kingston | EPSG:32755 | provider footprints only, NOT independent SAR GT',fontsize=12)
    fig.savefig(folder/'acquisition_context_map.png',dpi=180,bbox_inches='tight');plt.close(fig)
    pair=a.ROOT/'pairs'/sid
    # A broad boundary-search ROI, not a corresponding point/control/check.
    left=np.array(Image.open(pair/'robust_independent0.png'))
    right=np.array(Image.open(pair/'robust_independent1.png'))
    info=[z['native_info'] for z in r['checks']];mp=a.geo.PixelMap(a.Affine(*info[0]['transform']),info[0]['crs'],a.Affine(*info[1]['transform']),info[1]['crs'])
    xy=mp(a.geo.pixel_grid((512,512))).reshape(512,512,2).astype(np.float32)
    warped=cv2.remap(right,xy[:,:,0],xy[:,:,1],cv2.INTER_LINEAR)
    a.panel([left[256:512,96:416],warped[256:512,96:416]],['native R broad path/edge search region','M geography-remapped same ground search region'],sid+' | possible fixed path/edge outlines; NOT confirmed correspondence',folder/'boundary_search.png',960)
    imgs=[('original_pair.png','原生512左/右图'),('robust_independent_pair.png','来自TIF的固定独立对比度'),('footprints.png','两张512各自覆盖和共同地理区域'),('geo_compare.png','右图按地理坐标映射到左图，非GT'),('context_footprints.png','真实1024中两张裁片的地理位置'),('context_geo_compare.png','真实1024场景的地理辅助对照')]
    panels=''.join(f'<h2>{text}</h2><a target="_blank" href="../pairs/{sid}/{path}"><img src="../pairs/{sid}/{path}" alt="{text}"></a>' for path,text in imgs)
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kingston专项核查</title><style>body{{background:#111820;color:#edf1f7;font:18px system-ui;margin:25px}}a{{color:#69c5ff}}img{{max-width:100%;height:auto}}td,th{{padding:8px;border:1px solid #617385}}table{{border-collapse:collapse}}</style><body><a href="../index.html">返回图集</a><h1>Capella Kingston 配对专项核查</h1><p>不是左右图索引错配的现有证据。两图原生512像素表示163.3m与212.6m地面范围，中心地理距离0.284m。有效地理覆盖为100%/58.918%，共同面积比例100%/59.007%。这些数字都来自产品地理编码，不能保证物理散射中心对齐。</p><p>来源：2025-03-06T14:46:36.184635418Z（descending/left，azimuth208.8°，incidence39.6°）与2025-03-07T20:04:13.176981002Z（ascending/right，azimuth97.6°，incidence24.9°），不同collect_id，HH。原窗口为R(13293,10958)、M(10476,8430)。真实1024窗口分别从(13037,10702)、(10220,8174)开始，中央512的有效像素和GeoTransform均一致。</p><p>分类：B（原生地面尺度差异）明确；C（异向与明显散射外观差异）有支持。没有发现A裁片制作错误。D仍适用于像素级共同结构和绝对地理精度。原产品的GEC重投影使用约100.2434m与98.5768m固定参考高程，不能等价于精确地形/建筑正射纠正；地形、建筑高度、叠掩/阴影和辐射差异可能共同影响外观，尚未独立量化。</p><p>1024对照中可以搜索大尺度弯曲路径/地块边界。下方区域只是候选结构搜索范围；不能自动认定两图同一地物，也不能当人工control/check。未引入外部地图/光学数据；本轮证据足以检查制作链路，但不能替代绝对定位及物理对应的独立证据。</p><h2>原采集范围和1024位置</h2><img src="acquisition_context_map.png" alt="原采集与上下文地图">{panels}<h2>可能固定路径/边界的局部检查（地理辅助、待确认）</h2><img src="boundary_search.png" alt="可能路径和地块边界，待人工确认"><p><a href="../pairs/{sid}/audit.json">完整原点、CRS、GeoTransform、来源和双向覆盖审计</a></p></body></html>'''
    (folder/'index.html').write_text(text)

def classifications(results, candidates):
    selected={r['sample_id']:r for r in candidates}
    obviously_unhelpful={'Capella_0005__chip_000','Capella_0005__chip_002','Capella_0006__chip_000','Capella_0006__chip_002','Capella_0007__chip_002','Umbra_0052__chip_000','Umbra_0052__chip_001','Umbra_0052__chip_002','Umbra_0053__chip_000','Umbra_0053__chip_001','Umbra_0054__chip_000','Umbra_0057__chip_001','Umbra_0058__chip_000','Umbra_0059__chip_000','Umbra_0061__chip_002','Umbra_0071__chip_002'}
    output=[]
    for r in results:
        sid=r['sample_id']
        if sid in selected:c=selected[sid]['provisional_category'];reason=selected[sid]['selection_reason']
        elif sid in ORIGINAL_CLASS:c,reason=ORIGINAL_CLASS[sid]
        elif sid in obviously_unhelpful:c=4;reason='本轮原生联系图中水面/均匀区或孤立舰船主导；未确认可用固定地物。'
        else:c=5;reason='未在本轮证据中确认非重复共同静态结构；需更大原生场景或进一步独立人工判读。'
        if not r['engineering_metadata_checks_passed']:c=1;reason='裁片/来源元数据检查存在不一致，需单独定位。'
        output.append({'sample_id':sid,'pair_id':r['pair_id'],'site_name':r['site_name'],'category':c,'category_is_provisional':True,'engineering_metadata_checks_passed':r['engineering_metadata_checks_passed'],'reason':reason,'physical_correspondence_status':'pending_human_confirmation','GT_qualified':False})
    a.dump(a.ROOT/'audit'/'classification191.json',output)
    with (a.ROOT/'audit'/'classification191.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    return Counter(r['category'] for r in output)

def main():
    a.require_unsealed();stamp=datetime.now().strftime('%Y%m%d_%H%M')
    results=json.loads((a.ROOT/'audit/all191.json').read_text());records={r['sample_id']:r for r in results};candidates=json.loads((a.ROOT/'candidates.json').read_text());identities=json.loads((a.ROOT/'audit/identity_ledger.json').read_text())
    kingston(records,identities);counts=classifications(results,candidates)
    stats=json.loads((a.ROOT/'audit/candidate_statistics.json').read_text());summary=json.loads((a.ROOT/'audit/scan_summary.json').read_text())
    table=['|原5对|有效地理覆盖 R/M|面积交集 R/M|地面边长 R/M（m）|分类/限制|','|---|---:|---:|---:|---|']
    for sid in a.FIVE:
        r=records[sid];v=r['effective_overlap'];p=r['polygon_overlap_fractions'];s=r['metric_geometry'];c,reason=ORIGINAL_CLASS[sid]
        table.append(f"|{sid}|{v[0]['fraction_of_valid_pixels']:.3%}/{v[1]['fraction_of_valid_pixels']:.3%}|{p[0]:.3%}/{p[1]:.3%}|{s[0]['coverage_edge_lengths_m'][0]:.2f}/{s[1]['coverage_edge_lengths_m'][0]:.2f}|类别{c}，{reason}|")
    ct=['|候选|站点/方向|有效覆盖 R/M|地面边长 R/M（m）|判读状态|','|---|---|---:|---:|---|']
    for q in candidates:
        c=q['provisional_category'];ct.append(f"|{q['sample_id']}|{q['site_name']} / {q['look_pair_category']}|{q['effective_overlap'][0]:.2%}/{q['effective_overlap'][1]:.2%}|{q['ground_edges_m'][0][0]:.1f}/{q['ground_edges_m'][1][0]:.1f}|类别{c}候选；{q['selection_reason']}；物理对应待确认|")
    tests=[p for p in (a.ROOT/'logs').glob('tests_*.log')];test_ok=bool(tests and '\nOK\n' in tests[-1].read_text())
    browser=list((a.ROOT/'browser').glob('*/acceptance.json'));browser_ok=bool(browser and json.loads(browser[-1].read_text())['passed'])
    report=f'''# 真实SAR配对正确性核查与人工核验候选筛选

完成时间：{datetime.now().astimezone().isoformat()}。输出目录：`{a.ROOT}`。使用现有 Conda `match`；本轮没有模型推理、训练、参数搜索或来源下载。

## 结论与证据边界

原5对未发现采集索引错配、CRS/轴方向颠倒、裁片窗口偏移或1024上下文错位。5/5的两侧中央512有效像素、掩膜和变换均与本地保留真实1024一致。191对的索引、裁片元数据与58个本地保留原产品QA profile的窗口变换/范围检查通过。这是制作链路核查，不能证明供应商绝对地理定位误差为零，也不能证明物理散射中心可匹配。

Kingston有明确的原生地面尺度差异：163.30m与212.58m，而不是同地面范围的两张512。异向采集及GEC重投影高程近似也需要考虑。仍可在真实1024上下文中搜索路径/地块轮廓，但本轮没有独立人眼点击或外部精确对应证据，因此不承诺其可形成精确GT。

最终保留 **{stats['crop_pairs']}对候选，{stats['distinct_acquisition_pairs']}个原采集对，{stats['distinct_acquisitions']}个不同采集，{stats['sites']}个站点/来源组件**；3对异向、5对相近/同向。4对是精确核验的优先候选，4对只建议先做局部结构核验；全部待人工确认，新增GT为0。最初10对高清复看后剔除2对含疑似舰船/模糊孤立边界的候选，见`selection_initial_10.json`及`selection.json`，没有按匹配器结果筛选。

## 原5对逐对核查

{chr(10).join(table)}

来源身份和完整采集时间、升降轨、左右观测、方位/入射角、许可见`audit/identity_ledger.json`；逐侧原始URL、原图尺寸、窗口原点、CRS/GeoTransform、中心经纬度和pixel-center坐标约定见`audit/<sample_id>.json`及`pairs/<sample_id>/audit.json`。索引中实际collect_id与STAC item_id分别保留，不混作同一种ID。5对的物理collect_id均与已有原GeoTIFF嵌入采集身份记录相符，日期也与提供方元数据相符。

有效地理覆盖：逐原生像素中心映射到对侧，以双图有限值、发布有效掩膜、GeoTIFF有效掩膜及双线性所有支持邻居均有效计算；R/M分别除以该侧有效像素数。面积比例：密集边界转本地UTM后凸多边形相交，分别除以两侧几何面积；这是足迹几何比例，不含散射双图可见性。本地坐标减去世界原点再进行float32交集，避免大Northing精度损失。面积比例与有效像素比例因边界和插值有效支持不同，略有差异，不能替代独立匹配精度。

## Kingston专项

`kingston/index.html`组织原生双图、各自足迹/共同区域、地理映射对照、真实1024中的裁片位置、原采集valid footprint与局部候选路径/边界检查。R/M原窗口为(13293,10958)/(10476,8430)；1024窗口为(13037,10702)/(10220,8174)，两侧中央偏移均(256,256)。中心经纬度约(147.29178235,-42.97838730)/(147.29178064,-42.97838508)，地理中心距离0.284m。CRS均EPSG:32755，像素格距0.31893924m/0.41519840m，不能与物理分辨率混用。

真实采集：2025-03-06T14:46:36.184635418Z与2025-03-07T20:04:13.176981002Z；Capella-13 HH，descending/left→ascending/right，方位208.8°→97.6°，入射39.6°→24.9°。原GeoTIFF embedded collect_id分别eb19b018-9fe6-4d4b-9291-2d2315d7d729与b3f51c9f-3bdc-4bae-a5e4-6306fd64d02f。

- **A：没有发现配对/制作错误证据**；不生成所谓修正数据，不覆盖191对。
- **B：尺度差异明确**。面积交集覆盖reference全部，但只占moving59.007%；两PNG并排时不同缩尺和取景范围会加重差异。moving中与reference地面范围相近的区域约393.3px边长。
- **C：异向及散射外观差异有证据**。并不能把全部不一致都归因于散斑。建筑方向响应、叠掩/阴影、地形位移、辐射/成像处理都可能影响；未做独立因果分离。
- **D：绝对地理精度和可精确点击的同一物理结构证据仍不足**。嵌入影像说明显示GEC重投影使用`ExplicitInflatedWGS84[100.24340057373047]`/`[98.57683563232422]`，不是逐建筑/局部地形的精确正射。地理正确不等价于SAR散射中心正确。1024上可查看大尺度路径/边界候选，没有把它们标成已确认同一地物。未获取额外外部地图/光学数据，若人工仍无法确认，应保留不可评价而不是指定点位。

## 新候选与来源依赖

{chr(10).join(ct)}

三组Sunny Isles采集对属于同一来源组件并共享采集，不能算三个独立站点。长滩和鹿特丹各自的采集也可能共享；`candidates.json`保留采集ID/component/split，原划分未改变。新候选是已有数据的判读子集，不是新未见来源独立测试集。`audit/geography191.csv`提供191对共同有效覆盖及尺度，`audit/classification191.csv`给出五类暂定分流。

五类统计：类别1（确认工程问题/待修复）{counts.get(1,0)}；类别2（正确地理、精确核验候选）{counts.get(2,0)}；类别3（局部核验候选）{counts.get(3,0)}；类别4（当前困难）{counts.get(4,0)}；类别5（更大原生场景或额外独立证据/进一步判读）{counts.get(5,0)}。这些不是人工GT评级。未对191对做逐点物理对应审定，未明确确认的样本默认类别5；类别4只用于本轮联系图中明显水面/均匀区/疑似舰船主导的裁片及原困难样本。困难图像全部保留。

## 图集与显示协议

打开`index.html`，点击样本进入原生双图；支持50–400%最近邻显示放大、分别滚动或可选同步滚动、四种显示切换。`candidate_atlas.png`为2048×5265像素无损总览，单图原生512另存，可逐对放大。图库保留原始冻结PNG、原生TIF固定1%/99%线性显示、共同DN dB范围和各图独立DN dB范围显示。S1用10log10(power)，幅度DN用20log10(DN)；线性显示直接使用native power/DN。valid zero按原掩膜保留，dB显示映射到该图最小正值，规则写入显示JSON；NoData始终排除。没有滤波、锐化、超分辨或恢复已丢失信息，独立对比度不表示跨采集辐射可比。

Umbra 8bit原产品量化及散斑严重，对数显示会放大低DN背景；图集因此默认显示固定线性版本并提供原始和对数版本对照，不做择优滤波。局部ROI只是地理辅助搜索区域，最近邻放大不增加原生分辨率。地图重映射与叠加使用双线性插值，单独折叠且明确标注NOT GT。没有算法预测点或自动地理对应点，也没有修改现有独立标注网页/记录。

所有原5对都有本地真实1024上下文；新8对只有Umbra_0056__chip_002有本位置本地保留1024。其余7对的`context_status.json`明确不可用。本轮没有重下载大图，也没有用512插值/填充伪造更大场景。其他位置的1024不能冒充该裁片上下文。

图库是辅助判读入口，**不是独立标注入口**。地图辅助区默认关闭；打开会记录本浏览器localStorage暴露，记录键`sar_candidate_preview_exposure`，不冒充原标注系统审计链。在辅助图集看过地理信息的人员不应再被当作完全未暴露标注者；后续必须在原工具如实记录辅助暴露，最好安排未看辅助图的人员独立重新点击。当前真实标注工具仍保持原5任务与独立第二人规则，未自动新增候选任务或填写点。

## 推荐先人工判读

1. Capella_0006__chip_001：两图道路/固定设施边界清楚，同向、约212m场景；先确认地面交叉/边界转角，避免重复灯点/屋顶。
2. S1SF_EXTENSION_000__chip_000：城市/岸线更充分，优于原chip_002主要水面；20m格距的点位不确定度必须按实际可见尺度填写，不能声明亚米真值。
3. Capella_0005__chip_001：弯曲岸边和城市结构；先区分岸线变化与固定道路。
4. Capella_0007__chip_001：另一采集组合与道路/边界；与Sunny Isles前两对相关，不能重复统计独立站点。
5. Umbra_0061__chip_000：仅作局部固定地面边界尝试，不用重复箱区或移动船舶凑点。真实异向先看Umbra_0056__chip_002的1024和Umbra_0068__chip_001局部，无法确认则停止赋予精确GT。

## 512裁片方式建议

原实现保留两侧原生512，制作链路未发现应修复的错误；但固定像素数不能保证共同地面范围或静态结构充分。后续新版本应先用较大实际原生上下文判读固定道路/设施，再以共同地面FOV和双侧有效覆盖确定ROI，保留各自原生格距及可不同像素数裁片。需要统一512输入时，应单独提供标明重采样的common-FOV显示/算法版本及native→input坐标映射，同时保留原生裁片；不能把重采样对齐当GT。对于大格距比/旋转采集（例如Kingston及Melbourne），检查双向覆盖，不只看reference100%。选择地面结构而不是强散射亮点或站点标签，并保留1024以上真实上下文、建筑/地形高度相关信息及独立核验范围。旧191对继续冻结。

## 验收与复现

当前13项针对性测试通过：{test_ok}；真实Chromium图集测试通过：{browser_ok}，见`logs/tests_*.log`与`browser/*/acceptance.json`。测试覆盖native幅度/功率、valid zero/NoData、pixel中心/窗口原点、跨CRS、双向有效支持、UTM面积与世界坐标精度、5对真实上下文一致性、身份与时间、无GT升级、数据依赖隔离及冻结输出拒绝覆盖。图库浏览器测试是程序操作验收，不是人工物理结构确认。

```bash
conda run --no-capture-output -n match python {a.ROOT}/code/serve_gallery.py --port 8770
# 本机访问 http://127.0.0.1:8770/；远程访问沿用SSH端口转发
```

完整新目录复现用`REPRODUCE.sh`：先独占新目录复制代码/固定selection，再扫描、生成预览、测图集和出报告。正式交付后`DELIVERY_COMPLETE.json`门控阻止scan/render/finish/report覆盖本轮输出。实际代码和使用库版本、依赖文件、命令/退出状态、引用冻结文件SHA核对写入`execution_*.log`、`environment.json`与`verification.json`。

本轮程序曾遇到3个实现兼容问题：Python3.10不支持此处starred np.c_表达式、初次只加载主QA目录导致缺少split专用profile、Python3.10 ISO parser不接受9位小数秒；均定位并修复，失败日志保留。缺少QA元数据不能当作发现原数据错误。另有固定边长/视图方向近似与原生量化限制，未通过新参数搜索消除。报告只支持工程核查和候选判读，不报告真实Success@3/5/10px。
'''
    file=a.ROOT/f'pairing_audit_report_{stamp}.md';file.write_text(report)
    a.dump(a.ROOT/'report_identity.json',{'report':file.name,'created':datetime.now().astimezone().isoformat(),'candidates':stats,'classification_counts':dict(counts),'independent_GT_added':0})
    print('REPORT',file,flush=True)

if __name__=='__main__':main()
