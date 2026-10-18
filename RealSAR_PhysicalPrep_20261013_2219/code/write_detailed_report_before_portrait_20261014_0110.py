"""Detailed local report from frozen preparation evidence; no data/GT mutation."""
import sys,json,csv,os,hashlib,re,importlib.util
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def exclusive(p,text):
 with p.open('x',encoding='utf-8') as f:f.write(text)
def make_report():
 at=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=at.strftime('%Y%m%d_%H%M');out=ROOT/'reports';out.mkdir(exist_ok=True)
 name='physical_correspondence_preparation_report_'+stamp
 md_path=out/(name+'.md');html_path=out/(name+'.html');pdf_path=out/(name+'.pdf')
 audit=read(ROOT/'custodian/four_pair_preparation_20261013_2227.json')
 source=lines(ROOT/'custodian/source_audit_20261013_2227.jsonl')
 regions=lines(ROOT/'assisted_research/candidate_ground_regions_20261013_2227.jsonl')
 mapping=read(ROOT/'custodian/blind_to_source_map_20261013_2227.json')
 deps=read(ROOT/'custodian/source_dependencies_20261013_2334.json')
 refs=list(csv.DictReader((ROOT/'custodian/old_navigation_centers_20261013_2227.csv').open(encoding='utf-8-sig')))
 states=list(csv.DictReader((ROOT/'GT_annotation_status_20261013_2334.csv').open(encoding='utf-8-sig')))
 exposures=lines(ROOT/'custodian/exposure_events_snapshot_20261013_2227.jsonl')
 propagated=lines(ROOT/'custodian/exposure_propagation_20261013_2227.jsonl')
 summary=read(ROOT/'preparation_summary_20261013_2227.json')
 verification=read(ROOT/'material_verification_20261013_2343.json')
 browser=read(ROOT/'blind_browser_check_20261013_2328.json')
 policy=read(ROOT/'custodian/quality_policy_20261013_2227.json')['policy']
 evidence={
 'E01':BASE/'RealSAR_GeoAssist_20261012_1917/documentation/GeoAssist_Operations_20261013_212121/GeoAssist_Operations_Report.md',
 'E02':ROOT/'custodian/source_audit_20261013_2227.jsonl',
 'E03':ROOT/'custodian/four_pair_preparation_20261013_2227.json',
 'E04':ROOT/'custodian/old_navigation_centers_20261013_2227.csv',
 'E05':ROOT/'custodian/exposure_events_snapshot_20261013_2227.jsonl',
 'E06':ROOT/'custodian/exposure_propagation_20261013_2227.jsonl',
 'E07':ROOT/'custodian/source_dependencies_20261013_2334.json',
 'E08':ROOT/'assisted_research/candidate_ground_regions_20261013_2227.jsonl',
 'E09':ROOT/'custodian/blind_to_source_map_20261013_2227.json',
 'E10':ROOT/'GT_annotation_status_20261013_2334.csv',
 'E11':ROOT/'custodian/quality_policy_20261013_2227.json',
 'E12':ROOT/'blind_browser_check_20261013_2328.json',
 'E13':ROOT/'material_verification_20261013_2343.json',
 'E14':ROOT/'blind_browser_initial_check_failure_20261013_2239.json',
 'E15':ROOT/'custodian/CUSTODIAN_INSTRUCTIONS_20261013_2334.md',
 'E16':ROOT/'image_only_pairs_20261013_2334.jsonl',
 'E17':ROOT/'delivery_manifest_20261013_2343.json',
 }
 text=[]
 def rel(p):return os.path.relpath(p,out)
 def p(s):text.append(s+'\n')
 def h(s,n=2):p('#'*n+' '+s)
 def link(label,path):return '['+label+']('+rel(path)+')'
 def e(ids):return '依据：'+ '、'.join(link(i,evidence[i]) for i in ids)+'。'
 def table(head,rows):
  clean=lambda v:str(v).replace('|','\\|').replace('\n','；')
  p('|'+ '|'.join(map(clean,head))+'|\n|'+ '|'.join('---' for _ in head)+'|\n'+'\n'.join('|'+ '|'.join(map(clean,row))+'|' for row in rows))
 def fig(label,path,caption):p('!['+label+']('+rel(path)+')\n\n'+caption)
 site={'Port of Rotterdam, Netherlands':'鹿特丹','Port of Jebel Ali, United Arab Emirates':'杰贝阿里','Port of Antwerp, Belgium':'安特卫普'}
 labels={a['sample_id']:site[a['site']]+f" {a['azimuth_delta_deg']:.3f}°" for a in audit}
 selected_ids=set(labels)
 assert len(audit)==4 and len(source)==8 and len(regions)==8 and len(refs)==8
 assert len({x['acquisition_id'] for x in source})==7 and len({x['component'] for x in source})==3
 assert all(s['qualified_GT_point_count']=='0' for s in states)
 h('真实多视角SAR：首批四组物理对应材料准备详细报告',1)
 p('生成时间：'+at.isoformat()+'（Asia/Shanghai）。项目根目录：`'+str(BASE)+'`；独立成果目录：`'+str(ROOT)+'`。报告依据2026-10-13批次交付记录及23:43材料核验编写，未替代原始数据或版本。')
 p('阅读范围：本报告面向研究人员与材料管理员，包含候选窗口、日志中心、观测元数据和盲码映射。拟承担独立标注或复核的人员只接收独立盲包及包内人员说明；阅读本报告也属于辅助信息暴露。报告由Codex整理，人工筛选原文来自真实操作者字段“1”，候选区域解释来自AI辅助检查；无真人独立物理标注或第二人复核新增。')
 h('摘要与当前结论')
 p('本轮完成鹿特丹18.882°、鹿特丹45.984°、杰贝阿里36.369°和安特卫普88.394°四组真实SAR采集关系的来源核查、Native局部搜索材料和独立人工输入材料。四组对应8个显示侧、7个唯一采集身份、3个站点/来源组件；两组鹿特丹共享同一左侧采集，不能当作两个完全独立场景。')
 table(['指标','交付快照结果','含义'],[
  ['原队列/已人工辅助筛选','18组 / 13组','原库存与既有人工结果；本轮只准备4组'],
  ['原历史/本批选中历史','40版 / 16版','选中四组各4版，原JSON字节保持'],
  ['原最新判断','A=7、B=2、C=4','本批次为A=3、B=1'],
  ['旧浏览中心','原26个；本批取8个','每侧一个未确认参考中心，未与保存版本绑定'],
  ['候选地面搜索窗','8个','4组×2侧；Native范围、掩膜和有限值核验通过'],
  ['盲包显示图','7份唯一视图','完整现有Native显示图，按内容去重'],
  ['辅助物理点 / 独立点 / 合格GT','0 / 0 / 0','人工筛选文字、浏览中心和AI搜索框不计为点'],
  ['新下载 / 模型推理 / GPU / 新服务','0 / 未运行 / 未使用 / 未启动','没有扩建来源或改动训练路线'],
 ])
 p('鹿特丹18.882°是首选人工检查对象；45.984°需注意右侧窗口边界，杰贝阿里需解决模糊与结构歧义，安特卫普应转向罐区周边地面道路/围界。四组均可交给真人开展盲查，但没有一组已具备可报告真实配准精度的独立GT。'+e(['E01','E03','E10','E13']))
 h('1. 研究目的与执行范围')
 p('项目最终目标保持为仅输入两幅SAR图像、无经纬度或初始地理变换先验的真实多视角海事SAR匹配与配准。本轮把真实辅助筛选成果转换为可直接交给人工的材料，准备未来M0/M2/Model-A/Model-B在真实多视角数据上的对应评价依据。')
 p('执行范围是四组已有真实采集，不重新下载Native影像、不增加站点或来源、不重跑基线或训练。Geo Assist 8773代码和旧GT数据库不改；8772此前已授权停止，本轮未重启，也未启动其他端口。数据处理、材料生成和浏览器核验使用CPU与match环境。')
 p('本轮“可标注”指材料齐全并值得实际人员检查，不等于保证能找到足够点。独立合格GT仍需未暴露的真实人员标注、另一位真实独立人员重新定位、共同地面结构与定位不确定度核验。缺人或缺同结构证据时，停留在辅助开发/待核验状态。')
 h('2. 起点核查：既有记录与真实性')
 p('实际阅读的Geo Assist操作报告位于其documentation/GeoAssist_Operations_20261013_212121/子目录，而不是Geo Assist根目录。进一步读取了18项Native任务索引、13项最新辅助筛选、40版历史、暴露事件、固定20:10旧日志参考，以及prepare_gt.py、serve_gt.py、gt_bridge.py与独立GT协议。没有因路径变化重新构建数据。'+e(['E01','E03','E04','E05']))
 p('40版旧记录没有准确点击点或窗口坐标；本批四组也只有位置文字和判断理由。新工具支持点击持久化，不会补回过去没有写盘的坐标。本批保留16份原版本副本及8个旧中心的原始日志行号/时间/来源，未将它们加入人工点字段。')
 p('原报告和旧协议中的计数必须按时间解释。原初始协议曾记录真人A/B=0；后续用户已形成13组实际辅助筛选。本文使用后续实际版本，仍保留独立合格GT=0，不删除旧记录来匹配早期状态。')
 h('3. 首批四组的采集身份与来源核查')
 table(['真实关系','方位差°','左/右UTC开始时间','开始时差','组件'],[
  [labels[a['sample_id']],f"{a['azimuth_delta_deg']:.6f}",next(x['utc_start'] for x in source if x['sample_id']==a['sample_id'] and x['side']==0)+' / '+next(x['utc_start'] for x in source if x['sample_id']==a['sample_id'] and x['side']==1),str(next(x['time_gap_seconds'] for x in source if x['sample_id']==a['sample_id']))+' s',a['component']] for a in audit])
 p('时差依据采集开始时间计算。STAC的datetime通常为采集中间时间，source_audit另行保留，不能混用两种时间定义。方位差为元数据方位的最小圆周差，不是完整三维观测角度。源产品为本地已取得的Umbra GEC；来源记录保存原asset_url、metadata_url、STAC ID、collect ID和task ID。')
 table(['关系 / 侧','真实采集ID','采集任务ID','Native宽×高','原产品窗口[x,y,w,h]'],[
  [labels[x['sample_id']]+' / '+('左' if x['side']==0 else '右'),x['acquisition_id'],x['task_id'],f"{x['native_width']}×{x['native_height']}",x['original_window_xywh']] for x in source])
 p('核查比较了实际Native TIFF头与冻结尺寸、Affine、CRS，Native文件SHA256与来源索引，原始STAC文件SHA256、umbra:collect_id、开始时间、方位，以及asset侧车的原始窗口偏移。PNG尺寸与Native完全一致。旧inventory字段METADATA_VERIFIED_RASTER_NOT_READ是历史状态；本批实际进行了本地头和窗口读取，不能继续把该旧字段理解为从未读过。'+e(['E02','E13']))
 h('4. 来源依赖、组件隔离与暴露')
 p('两组鹿特丹共享左采集d1f1f1ba-56a5-4d97-9cf1-aaf3ff14688c；另外两个右采集54e07361…和0048ea3f…来自同一采集任务44fad5f7…。其中既有27.102°关系也连接这些采集。在18项库存内，v2_component_011关联8项鹿特丹任务，因此按四条关系或裁片数量计算独立样本量会夸大独立性。')
 table(['本批关系','共享采集的其他现有任务','同组件任务数','拆分/统计边界'],[
  [labels[d['sample_id']],', '.join(d['shared_acquisition_tasks']) or '在18项库存中未发现',len(d['same_component_tasks']),'按完整采集、站点和来源组件隔离'] for d in deps])
 p('这些图像此前已用于开发/诊断，保留diagnostic_development_seen属性。独立人工标签可以提高评价证据可信度，但不会把已观察影像变为封存未见最终测试集。未来不得把同采集或同组件随机拆到训练与测试，也不得将同地物的多个点当作多个独立场景。'+e(['E07']))
 counts=Counter(x['sample_id'] for x in propagated)
 table(['关系','直接/共享采集传播记录数','已知暴露身份','同组件暴露身份'],[
  [labels[a['sample_id']],counts[a['sample_id']],', '.join(a['direct_or_shared_acquisition_exposed_actors']),', '.join(a['component_exposed_actors'])] for a in audit])
 p(f'暴露快照共{len(exposures)}条原事件；针对本批产生{len(propagated)}条关系传播记录，其中同一原事件可传播到多个关系，不能解释为{len(propagated)}名人员。已知身份均为“1”，这是稳定记录字段，不是经过认证的实名。必须核对真人身份与所有历史别名；未记录的其他暴露仍需如实申报。'+e(['E05','E06']))
 p('关闭地理提示、换页面或换ID不能恢复该人员在同一关系上的盲标资格。共享采集暴露需传播；同组件记录也保留用于管理员审查，不能据空白列表推断新人一定未暴露。')
 h('5. 用户实际辅助筛选：原文与版本')
 table(['关系 / 盲码','最新版本 / 时间(+08)','判断','位置原文','理由与风险原文'],[
  [labels[a['sample_id']]+' / '+a['blind_code'],'v'+str(a['latest_assisted_record']['revision'])+' / '+a['latest_assisted_record']['at'],a['latest_assisted_record']['verdict'],a['latest_assisted_record']['location_description'],a['latest_assisted_record']['reason']] for a in audit])
 p('四组均为操作者“1”真实保存，每组4版，共16版。所有最新版本viewed_geographic_hint=true、blind_eligible_for_this_sample=false、independent_GT=false，推荐独立标注字段仍false。本文保留用户A/B，不用AI检查覆盖或更改。A/B说明值得检查或局部可辨认，并不是实际物理点已经核实。'+e(['E03']))
 h('6. 坐标语义与可定位材料')
 p('每侧Native坐标以x为列、y为行，整数(0,0)为左上像素中心。原生图被浏览器缩放只影响显示，不改变坐标。候选局部窗bbox=[x0,y0,w,h]是整数像素边界；局部中心(u,v)返回Native中心为(x0+u,y0+v)，再返回原产品时只加一次既有整数窗口偏移。')
 p('例如鹿特丹18.882°左搜索窗为[7050,2120,320,320]，对应原产品窗[14389,9462,320,320]，因为已有Native裁片原点为[7339,7342]。该公式只是像素索引溯源，不是地理配准或对应标签。')
 table(['关系 / 侧','旧浏览中心Native(x,y)','旧跨度px','原日志时间（未写时区）'],[
  [labels[x['sample_id']]+' / '+('左' if x['side']=='0' else '右'),f"({float(x['x']):.6f}, {float(x['y']):.6f})",x['span'],x['log_timestamp']] for x in refs])
 p('表中左右中心来自各侧末次viewport请求，可能是点击、平移或地理预测目标。日志没有操作者/session、动作来源或保存版本关联，不能将同一行左右中心配成物理同点。小数位是计算/请求保存精度，不是已测定位准确度。'+e(['E04']))
 table(['关系 / 侧','Native粗搜索窗[x,y,w,h]','原产品粗搜索窗','掩膜与有限值比例'],[
  [labels[x['sample_id']]+' / '+('左' if x['side']==0 else '右'),x['bbox_native_xywh'],x['source_product_bbox_xywh'],f"{x['crop_valid_fraction']:.3f}"] for x in regions])
 p('8个搜索窗均在各侧Native有效范围内，像素与现有显示PNG的直接整数裁片逐像素一致；没有旋转、地理重采样或人为补边。有效比例1.000只代表栅格有效性，不能证明共同地面散射体。预测中心另外保存reference_only=true、GT=false、confirmed_human_point=false、exact_correspondence=false。数学往返误差也不等于实际SAR同点误差。'+e(['E08','E13']))
 h('7. 四组实际结构检查与候选局部图')
 p('本节展示AI在真实Native局部上完成的辅助检查。规则暗区、线状设施或圆形结构的物理类别/高度没有外部实测证据；描述中保留“候选”。每侧搜索窗独立保存，左右局部并列展示不表示已经建立对应。')
 for number,a in enumerate(audit,1):
  sid=a['sample_id'];folder=ROOT/'assisted_research'/sid
  h(f"7.{number} {labels[sid]}：{a['ai_search_decision']}",3)
  p('任务ID：`'+sid+'`；人工最新'+a['latest_assisted_record']['verdict']+'，原文：“'+a['latest_assisted_record']['reason']+'”。')
  p('辅助图像观察：'+a['visual_search_description']+'。风险：'+a['risk'])
  if number==1:p('下一步先请真人确认岸缘/设施边界是否为双方可见的同一低矮固定地面转角，再判断能否精确重点击。邻近最亮块不自动入选，亮度相似不能代替结构依据。')
  elif number==2:p('右旧中心x约75像素，粗搜索窗从x=0开始；靠边内容确有截断可能。不能向图外外推拐角或用边界夹紧坐标凑点。可保留线段/区域级候选，精确点需真人确认。')
  elif number==3:p('采集开始时差84571秒（约23.49小时），并非秒级同时观测。地面边缘转折可继续搜索，但两侧模糊、亮散射带与位置差异要求更严格物理识别；无法确认就保留局部区域级。')
  else:p('采集开始时差133441秒（约37.07小时）。圆形储罐形态提供场区搜索线索，罐顶/罐体亮边/阴影不能直接作地面同点。候选转向周边道路/围界，当前图像检查没有证明同一精确地面交汇点。')
  fig(f'{labels[sid]} 2048邻域辅助预览',folder/'comparison_2048_20261013_2219.png',f'图{number*2-1}：两侧旧日志邻域。实际窗口靠边时截断，拼图为显示缩略，原生独立局部文件保留；无物理对应线。')
  imgs=['![左Native候选区域]('+rel(folder/'ground_search_side0_20261013_2227.png')+')','![右Native候选区域]('+rel(folder/'ground_search_side1_20261013_2227.png')+')']
  table(['左Native整数裁片','右Native整数裁片'],[imgs])
  p(f'图{number*2}：各侧候选地面搜索区域，完整坐标见第6节。该并列仅供辅助研究，不能作为已确认匹配对或独立标签。')
 h('8. 两套材料的组织与分发')
 table(['材料','位置/入口','内容与使用者'],[
  ['辅助研究','assisted_research/assisted_materials_20261013_2227.html','256/1024/2048邻域、旧中心、地理参考、AI搜索框、用户原文；给已暴露研究人员'],
  ['独立输入','blind_native_materials_20261013_2334.zip','完整Native灰度视图、匿名任务码、离线标注、身份/暴露声明、空CSV；给审核后的真实人员'],
  ['管理员溯源','custodian/','真实采集、组件、盲码、传播暴露、原记录、政策与人员登记；不进入盲包'],
  ['状态与模型入口','GT_annotation_status_20261013_2334.csv；image_only_pairs_20261013_2334.jsonl','GT状态与两个已有原生显示路径；图像模型仅读两图数组'],
 ])
 p('盲包7个唯一视图按PNG内容去重，保持完整Native像素网格和现有dB显示强度；没有为了隐藏地理信息再裁成辅助ROI。原PNG没有附带地理文本元数据，分发文件使用I001–I007别名且不共享原资产的可写inode。ZIP包含实际PNG字节，解压即可使用，不需链接回原服务器。')
 table(['盲码','管理员真实关系','左图别名','右图别名'],[[m['blind_code'],labels[m['sample_id']],m['images'][0],m['images'][1]] for m in mapping])
 p('此映射只用于管理员溯源，必须与盲人员隔离。I001复用于B001和B003，保留真实共享来源而不把重复文件计成新采集。盲包没有地图、观测方位、GeoTransform、自动匹配或辅助候选点；真实人员自行在原生双图中寻找共同结构。'+e(['E09','E13']))
 h('9. 标注人员可以直接执行的流程')
 p('管理员先确认两名不同的真实人员及历史别名，核查既往地理、算法、候选区域/AI提示和他人坐标暴露。已知身份“1”不能在这四组上转为独立盲标；新人未出现在记录中也不等于资格已核实。人员登记模板保持空白，没有填入模拟人名。')
 p('将盲包解压，打开blind_native_materials/blind_annotation_20261013_2227.html。使用浏览器离线文件页面，无需启动8772、8773或新服务。本人输入真实身份、别名、四项暴露声明并确认属实；未知声明不能被默认为未暴露。')
 p('通过滚轮和按钮缩放，Shift或中键拖动；左右分别寻找同一固定地面结构并点击。高级/逐点结构编辑记录物理结构描述、静态性、双图可见性、各侧不确定度、风险备注、置信度与check/control职责。未知项保持未知，不预填定位误差，不按数量目标补点。')
 p('点击保存生成下载JSON，包含该人的真实输入和新版本号，文件名带北京时间分钟。需要确认下载真正落盘；这是离线导出，不是写入项目GT数据库。页面内存刷新会清空，每个下载版本都应保留。显示的独立合格数始终为0，真实人员声明不会自动授予资格。')
 p('第二人使用自己的干净会话独立重新点击，不能读取第一人坐标或带点截图。管理员可提供第一人实际填写的物理结构文字与点ID，但不能给区域位置/地理提示。第二人不确定或拒绝的情况要保留；同点ID不证明同一物理结构。两人分别回交本人全部JSON/CSV，由管理员合并和核验。')
 p('看过辅助提示的真实人员仍可进行辅助探索，但导出必须如实标识assisted_human_observations；全部未声明暴露的输入仍是human_blind_capture_pending_custodian_verification，不是合格GT。'+e(['E15']))
 fig('离线盲包实际浏览器截图',ROOT/'blind_materials_readonly_browser_20261013_2328.png','图9：真实完整Native盲视图的只读浏览器检查，未输入真人身份、未产生点击记录或下载标注。截图和本报告不能提供给尚未审核暴露的盲标人员。')
 h('10. 真实人工点的核验与GT阶段')
 table(['阶段','实际定义','本批次状态'],[
  ['准备完成、未标注','只有筛选文字与搜索材料，没有人工对应坐标','4组均处于prepared_unannotated'],
  ['辅助标注','真实人员在辅助暴露后填写对应观察，仅供开发','0个对应点；不能用AI框替代'],
  ['独立标注、待复核','实际未暴露人员标点，管理员核实身份和来源','0；未收到实际人员数据'],
  ['独立复核通过','第二名真实独立人员重点击，结构、有效性和不确定度通过','0；仍须应用原GT质量门限'],
 ])
 p('每个实际点必须核验六项：双侧Native范围与有效像素；同一地面结构的文字/图像依据；高度、阴影、叠掩与散射中心风险；两人真实独立性及同结构重点击；原始像素坐标与采集/显示来源；点级还是仅区域级支持。区域级线索不能硬填一个中心升级点级标签。')
 table(['原质量门限','要求','本轮处理'],[
  ['局部支持',f"至少{policy['local_min_checks']}个合格check，双方凸包面积比例≥{policy['local_min_hull_fraction']}",'未放宽；没有点就保持未核验'],
  ['全幅稀疏支持',f"至少{policy['global_min_checks']}个合格check，凸包比例≥{policy['global_min_hull_fraction']}，双轴跨度≥{policy['global_min_span_fraction']}",'局部结构不能外推全幅'],
  ['Success阈值',f"至少{policy['success_min_checks']}个check，合成定位不确定度≤阈值的1/3",'未计算真实Success@3/5/10'],
  ['独立复核','真实不同人员，坐标隐藏后重点击，暴露如实传播','没有合格第二人则保留待核验'],
 ])
 p('上述门限是原项目证据政策，不能因为找不到点而降低，也不是人为凑点要求。既有quality.py还检查重复坐标、静态/共同可见确认、置信度、独立身份、真实重新点击和未解决的较大人员定位分歧；不确定度由真实人员填写，不是程序自动假设的统计置信区间。'+e(['E10','E11']))
 h('11. 既有GT流程复用与模型评价边界')
 p('实际读取了V2 prepare_gt.py与serve_gt.py，以及Geo Assist gt_bridge.py，盲包复用了既有Native标注组件的坐标、画布、缩放/平移、放大和结构编辑，只替换离线图像加载与下载记录接口。沒有启动服务或真实GT批次。')
 p('当前四组recommend_independent_annotation=false。现有桥接prepare会返回无合格推荐，或在有推荐时要求两名真实独立人员并核查暴露。本轮不改变旧字段或虚构身份越过条件；本次材料准备不代表桥接流程已经完成。真实数据回收后，应在新的四组工作目录保留实际辅助证据与暴露，复用正式存储/质量规则，不把辅助记录改名成盲判读。')
 p('后续M0/M2/Model-A/Model-B预测仅输入两幅SAR图像数组，不读取本报告、位置提示、GT、有效掩膜、CRS或采集字段进行初始化或接受判断。先冻结预测与接受结果，再用同一组合格check评价。control只能按预声明职责用于后验诊断，不能把物理GT提供给模型推理。')
 p('如果最终只确认局部点，报告局部稀疏精度及支持范围；如果只有区域级对应，不计算像素级真实误差。如果没有合格GT，记录accepted_unassessable/未可评价，不能用自动内点、投影残差或已知合成几何替代真实精度。'+e(['E11','E15','E16']))
 h('12. 实际核验、失败记录与可复现范围')
 table(['核验项目','实际结果','证明范围'],[
  ['来源记录',verification['native_metadata_records_verified'],'8侧真实采集元数据与本地Native头/文件身份'],
  ['候选局部像素',verification['candidate_crops_native_pixel_equality_passed'],'8个整数裁片与原PNG逐像素一致'],
  ['原文件保护',verification['protected_existing_files_sha_unchanged'],'76份列入保护清单的文件SHA未变；不是全项目重新哈希'],
  ['原历史记录',verification['original_human_versions_unchanged'],'40个原辅助版本未改'],
  ['盲包实际浏览器',', '.join(browser['loaded_blind_tasks']),'四组完整Native显示、隐藏提示、无默认身份、坐标约定与刷新空状态'],
  ['真实人工点 / GT',0,'所有核验只读，不模拟真人、不保存GT'],
 ])
 p('实际Chromium版本'+browser['browser']+'；GPU=false、ports_started=[]、页面错误为空，网络请求仅本地file资源。未知暴露声明进入流程被拒绝，初始记录与重载后记录均为空，未默认给人独立身份。测试覆盖显示和只读流程，没有在该专项测试中创建真人身份、完整下载一份真实人工标注或证明人员资格。')
 p('首轮只读检查使用inner_text读取折叠的metadata元素，导致断言失败；改用text_content后四组检查通过。该失败已保存，属于核验脚本读隐藏文本方式，不是影像匹配成功/失败或物理GT问题。原生分发图也已与旧资产隔离成独立只读副本，避免人员编辑副本连带修改原文件。'+e(['E12','E13','E14']))
 p('本报告整理直接使用现有交付证据，不重复下载、模型训练或全量匹配评价。PDF导出只验证文档转换与图像加载；不增加独立SAR物理精度结论。')
 h('13. 下一步执行与仍缺少的证据')
 table(['优先顺序','关系','真人需要确认的关键问题'],[
  ['1','鹿特丹18.882°','规则边界转角是否同一低矮固定地面位置；排除邻近亮块'],
  ['2，条件性','鹿特丹45.984°','右边界附近是否保留完整同一拐点；不可外推到图外'],
  ['条件性局部','杰贝阿里36.369°','模糊/强散射下是否仍可双方重定位同一地面转折'],
  ['地面结构专项','安特卫普88.394°','罐区周边道路/围界能否确认同地面位置，排除罐顶和阴影'],
 ])
 p('可以立即开展的是实际人员在盲包中的独立搜索与记录。仍缺少的是两名符合条件的真人身份、实际点击坐标、同结构描述、定位不确定度和第二人复核证据。人员安排、分发登记和数据回收都由真实负责人实施；AI不能填入人名或代替第二人。')
 p('找不到可靠结构时保留疑义、局部区域或不可评价原因。四组候选没有被承诺为最终正例，没有任何自动投影点加入真实GT；当前可交付的是可直接标注的准备材料。')
 h('14. 文件入口、操作与证据索引')
 p('成果总目录：`'+str(ROOT)+'`。完整盲包：'+link('blind_native_materials_20261013_2334.zip',ROOT/'blind_native_materials_20261013_2334.zip')+'；辅助入口：'+link('assisted_materials_20261013_2227.html',ROOT/'assisted_research/assisted_materials_20261013_2227.html')+'。')
 p('离线入口：解压盲包后双击blind_native_materials/blind_annotation_20261013_2227.html；人员只接收该包。项目管理员可以直接查看本地blind_release中的同名文件；不要把整个成果目录发给盲人员。')
 p('状态表：'+link('GT_annotation_status_20261013_2334.csv',ROOT/'GT_annotation_status_20261013_2334.csv')+'；人员说明、空CSV和第二人纯结构文字模板位于盲包；人员分发登记、盲码映射与来源身份位于custodian。')
 p('本轮执行命令记录如下，build_materials会新建输出且拒绝覆盖，已完成批次无需重跑。浏览器检查只用于材料可用性，不写人类点。')
 p("```bash\ncd /home/yl/matching/sar_matching\nCUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\\n/home/yl/miniconda3/envs/match/bin/python -B \\\nRealSAR_PhysicalPrep_20261013_2219/code/build_materials.py\n\nCUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\\nPLAYWRIGHT_BROWSERS_PATH=/tmp/sar_gt_browsers_verified \\\n/home/yl/miniconda3/envs/match/bin/python -B \\\nRealSAR_PhysicalPrep_20261013_2219/code/check_blind_materials.py\n```")
 table(['证据ID','原文件','说明'],[[i,link(path.name,path),{
 'E01':'既有操作、人工筛选与40版/26中心记录','E02':'8侧真实来源、SHA、UTC、Native窗口','E03':'4组原判读与AI搜索判断','E04':'本批8个未确认浏览中心','E05':'17条原暴露快照','E06':'14条关系传播记录','E07':'共享采集、组件与开发属性','E08':'8个候选窗口及投影参考标志','E09':'管理员盲码/真实来源映射','E10':'当前全部未标注状态','E11':'既有质量政策快照','E12':'只读实际浏览器通过结果','E13':'76文件与8裁片最终核验','E14':'初次检查失败保留','E15':'真实人员与管理员操作要求','E16':'纯双图模型入口','E17':'本批文件身份清单'}[i]] for i,path in evidence.items()])
 h('附录A. 7个唯一Native资产的文件身份')
 unique={x['acquisition_id']:x for x in source}
 table(['采集ID','Native文件SHA256','本地文件'],[[k,x['native_sha256'],link('native.tif',Path(x['native_tif']))] for k,x in unique.items()])
 p('8侧来源记录中鹿特丹左采集重复，附录按7个唯一采集去重；原产品asset_url和STAC元数据URL在E02保留。本报告不重新访问网络下载这些资产，也不把分发副本计为新增来源。')
 h('附录B. 报告版本与证据边界')
 p('MD为报告正文版本，PDF从同一MD渲染生成；保留同名HTML用于转换和浏览。文件名以北京时间精确到分钟的时间戳结尾，历史报告不覆盖。报告输入文件与代码SHA256另存于report_identity_'+stamp+'.json，PDF转换记录带同一分钟尾缀。')
 p('本报告中的“通过”指材料、坐标索引、来源及只读显示核验；物理对应、真实像素精度、人员独立资格和合格GT仍未取得实际证据。任何未来新增人工数据须新版本保存、重新核查来源和暴露，再按原质量政策判断。')
 md='\n'.join(text)
 exclusive(md_path,md)
 # Reuse the existing local Markdown renderer without running its old report workflow.
 sys.path.insert(0,'/tmp/sar_gt_browser_deps')
 spec=importlib.util.spec_from_file_location('existing_pdf_renderer',BASE/'SAR_SP_LG_TrainingReview_20261012_1651/code/export_pdf.py')
 renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
 css='''@page{size:A4 landscape;margin:13mm 12mm 15mm}body{font-family:"Noto Sans CJK SC","DejaVu Sans",sans-serif;font-size:10pt;line-height:1.6;color:#182532;margin:0}h1{font-size:21pt;line-height:1.35}h2{font-size:15pt;margin-top:6mm;border-bottom:.3mm solid #b8c4d0;padding-bottom:2mm;break-before:page;break-after:avoid}h3{font-size:12pt;break-before:page;break-after:avoid}p{margin:3mm 0;overflow-wrap:anywhere;orphans:3;widows:3}table{border-collapse:collapse;table-layout:fixed;width:100%;font-size:8pt;line-height:1.45;margin:3mm 0}th,td{border:.2mm solid #bbc6d0;padding:2mm;vertical-align:top;overflow-wrap:anywhere;word-break:break-word}th{background:#edf2f7}thead{display:table-header-group}tr{break-inside:avoid}img{display:block;max-width:100%;width:100%;height:auto;max-height:108mm;object-fit:contain;margin:3mm 0;break-inside:avoid}td img{max-height:65mm}pre{background:#eef2f5;padding:3mm;white-space:pre-wrap;overflow-wrap:anywhere;font-size:8pt}code{font-family:"Noto Sans Mono CJK SC",monospace;font-size:.9em;overflow-wrap:anywhere}a{color:#175680;text-decoration:none}'''
 html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实多视角SAR物理对应准备详细报告</title><style>'+css+'</style><body>'+renderer.render(md)+'</body></html>'
 exclusive(html_path,html)
 for target in re.findall(r'\]\(([^)]+)\)',md):
  if not target.startswith(('http://','https://')):assert (out/target).exists(),target
 protected={str(path):sha(path) for path in evidence.values()}
 identity=dict(created_at=at.isoformat(),markdown=str(md_path),html=str(html_path),planned_pdf=str(pdf_path),
               markdown_sha256=sha(md_path),html_sha256=sha(html_path),inputs=[dict(evidence_id=i,path=str(path),sha256=sha(path)) for i,path in evidence.items()],
               code=[dict(path=str(p),sha256=sha(p)) for p in sorted((ROOT/'code').glob('*')) if p.is_file()],
               totals=dict(selected_pairs=4,unique_acquisitions=7,components=3,selected_histories=16,all_prior_histories=40,selected_navigation_refs=8,all_prior_navigation_refs=26,
                           candidate_windows=8,propagated_exposure_rows=len(propagated),original_exposure_events=len(exposures),independent_GT=0),
               report_is_assisted_information_not_for_blind_people=True,original_data_modified=False,document_links_passed=True)
 identity_path=out/('report_identity_'+stamp+'.json');exclusive(identity_path,json.dumps(identity,ensure_ascii=False,indent=2))
 assert all(sha(Path(p))==value for p,value in protected.items())
 print(json.dumps(dict(markdown=str(md_path),html=str(html_path),planned_pdf=str(pdf_path),identity=str(identity_path),figures=md.count('![')),ensure_ascii=False,indent=2))
 return identity_path
def export_pdf(identity_path):
 from playwright.sync_api import sync_playwright
 identity=read(identity_path);pdf=Path(identity['planned_pdf']);html=Path(identity['html']);md=Path(identity['markdown'])
 if pdf.exists():raise FileExistsError(pdf)
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
  page=browser.new_page(viewport={'width':1700,'height':1200})
  page.goto(html.as_uri(),wait_until='load',timeout=60000);page.emulate_media(media='print');page.evaluate('document.fonts.ready')
  page.wait_for_function('Array.from(document.images).every(im=>im.complete&&im.naturalWidth>0)',timeout=30000)
  images=page.locator('img').count()
  page.pdf(path=str(pdf),prefer_css_page_size=True,print_background=True,display_header_footer=True,header_template='<span></span>',footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#555">真实多视角SAR物理对应准备 · <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
  version=browser.version;browser.close()
 assert sha(md)==identity['markdown_sha256'] and sha(html)==identity['html_sha256']
 stamp=md.stem.rsplit('_',2)[-2]+'_'+md.stem.rsplit('_',2)[-1]
 exclusive(pdf.parent/('pdf_conversion_'+stamp+'.json'),json.dumps(dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),source_md=str(md),source_html=str(html),pdf=str(pdf),pdf_sha256=sha(pdf),
             images_loaded=images,browser=version,browser_sandbox=True,environment='match',GPU=False,external_requests_required=False,scope='document rendering only; not SAR GT validation'),ensure_ascii=False,indent=2))
 print(str(pdf))
if __name__=='__main__':
 if len(sys.argv)==3 and sys.argv[1]=='--pdf':
  sys.path.insert(0,'/tmp/sar_gt_browser_deps');export_pdf(Path(sys.argv[2]))
 else:make_report()
