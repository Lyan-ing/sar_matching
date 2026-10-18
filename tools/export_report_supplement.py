"""Seven report figures from frozen SAR predictions. No training or new inference."""
import base64,csv,hashlib,html,io,json,math,os,re,shutil,sys,zipfile,stat
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
BASE=Path('/home/yl/matching/sar_matching');OLD=BASE/'SAR_Report_Ch5Ch6_Assets_20261017_1821';OT='20261017_1821'
NOW=datetime.now(ZoneInfo('Asia/Shanghai'));TAG=NOW.strftime('%Y%m%d_%H%M');NAME='SAR_Report_FigureSupplement_'+TAG
OUT=BASE/NAME;PHYSICAL=Path('/data1/libo/yl/ckpt/sar_matching')/NAME
FONT="'Times New Roman','FangSong','仿宋','FandolFang',serif";INK='#17252b';TEAL='#65b7ad';WARM='#d5a36b'
METHODS=['m0','m2','b','keep'];NAMES={'m0':'预训练SuperPoint','m2':'SAR描述子迁移基础模型','b':'多尺度结构增强模型','keep':'本项目方法'}
FIGURES=[];INPUTS={};SELECTION=[];DISPLAY=[]
MANIFEST_SOURCE=BASE/'SyntheticSAR_Mixed640512_Multicondition_20261016_2114/manifests/val_CANONICAL_20261016_2147.jsonl'

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def track(p):
 p=Path(p);INPUTS[str(p)]=digest(p);return p

def readcsv(name):
 p=track(OLD/'csv'/f'{name}_{OT}.csv');return list(csv.DictReader(p.open()))

def prediction(row):
 p=row.get('frozen_prediction_path') or str(BASE/'SAR_SP_JointTransfer_Mixed640512_20261016_2234/predictions/final'/f'{row["pair_id"]}_{row["method"]}_natural_20261016_2234.npz')
 p=track(p)
 with np.load(p,allow_pickle=False) as z:pred={k:z[k].copy() for k in z.files}
 assert len(pred['matches_A'])==len(pred['matches_B'])==int(row['match_count']);assert int(pred['inliers'].sum())==int(row['inliers'])
 assert all(np.isfinite(pred[k]).all() for k in ['matches_A','matches_B','keypoints_A','keypoints_B'])
 return pred,p

class Figure:
 def __init__(self,height,title,filename,pair_id,section,number):
  self.h=height;self.title=title;self.filename=filename;self.pair_id=pair_id;self.section=section;self.number=number;self.notes=[]
  self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="160mm" height="{height/10:.1f}mm" viewBox="0 0 1600 {height}"><title>{html.escape(title)}</title><rect width="1600" height="{height}" fill="#ffffff"/>']
 def text(self,x,y,s,size=44):
  assert size>=44;self.parts.append(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{INK}" text-anchor="middle">{html.escape(str(s))}</text>')
 def image(self,path,x,y,w):
  b64=base64.b64encode(Path(path).read_bytes()).decode();self.parts.append(f'<image x="{x}" y="{y}" width="{w}" height="{w}" preserveAspectRatio="none" xlink:href="data:image/png;base64,{b64}"/>')
 def line(self,x1,y1,x2,y2,color):
  self.parts.append(f'<line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" stroke="#152021" stroke-width="4.6"/><line x1="{x1:.4f}" y1="{y1:.4f}" x2="{x2:.4f}" y2="{y2:.4f}" stroke="{color}" stroke-width="3.0"/>')
 def point(self,x,y,color):
  self.parts.append(f'<circle cx="{x:.4f}" cy="{y:.4f}" r="3.8" fill="none" stroke="#152021" stroke-width="4.6"/><circle cx="{x:.4f}" cy="{y:.4f}" r="3.8" fill="none" stroke="{color}" stroke-width="3.0"/>')
 def legend(self,y):
  self.line(390,y-13,470,y-13,TEAL);self.text(620,y,'几何内点');self.line(900,y-13,980,y-13,WARM);self.text(1130,y,'其他匹配')
 def save(self):
  path=OUT/'selected_figures'/f'{self.filename}_final_{TAG}.svg';path.write_text(''.join(self.parts)+'</svg>');FIGURES.append(dict(stem=self.filename,svg=path,png=path.with_suffix('.png'),title=self.title,pair_id=self.pair_id,section=self.section,number=self.number,width_mm=160,height_mm=self.h/10,viewbox_height=self.h,min_font_pt=44*.1/25.4*72))

def draw_pair(fig,row,pred,raw,x,y,width,gap,cap):
 native=int(row['size']);scale=width/native
 for L,xx in [('A',x),('B',x+width+gap)]:fig.image(raw[row['pair_id'],L],xx,y,width)
 a,b=pred['matches_A'],pred['matches_B'];inliers=pred['inliers'].astype(bool).reshape(-1);ids=np.linspace(0,len(a)-1,min(cap,len(a)),dtype=int) if len(a) else np.array([],int)
 for j in ids:
  color=TEAL if inliers[j] else WARM;x1,y1=x+scale*a[j,0],y+scale*a[j,1];x2,y2=x+width+gap+scale*b[j,0],y+scale*b[j,1];fig.line(x1,y1,x2,y2,color);fig.point(x1,y1,color);fig.point(x2,y2,color)
 DISPLAY.append(dict(figure=fig.filename,pair_id=row['pair_id'],method=row['method'],display_match_indices=ids.tolist(),candidate_count=len(a),inlier_count=int(inliers.sum()),native_scale_to_svg=scale,point_origin_A=[x,y],point_origin_B=[x+width+gap,y],GT_used_for_line_selection=False))

def comparison(pid,rows,records,raw,title,filename,number):
 f=Figure(1300,title,filename,pid,'实验',number)
 for j,m in enumerate(METHODS):
  row=rows[pid,m];pred,p=prediction(row);x=25+(j%2)*800;y=110+(j//2)*560;center=x+360
  f.text(center,y-50,NAMES[m]);draw_pair(f,row,pred,raw,x,y,350,20,20);f.text(center,y+415,f'匹配{len(pred["matches_A"])} / 内点{int(pred["inliers"].sum())}')
 r=records[pid];f.text(800,1185,f'重叠{r["overlap_ratio"]*100:.1f}%　二维旋转{r["rotation_actual_deg"]:.1f}°');f.legend(1250);f.save()

def standalone(pid,rows,records,raw,title,filename,number):
 f=Figure(1940,title,filename,pid,'实验',number);row=rows[pid,'keep'];pred,p=prediction(row)
 f.text(410,60,'参考图像');f.text(1190,60,'待配准图像')
 for L,x in [('A',40),('B',820)]:f.image(raw[pid,L],x,100,740)
 f.text(800,925,'本项目方法');draw_pair(f,row,pred,raw,40,970,740,40,24)
 r=records[pid];f.text(800,1785,f'匹配{len(pred["matches_A"])} / 内点{int(pred["inliers"].sum())}　重叠{r["overlap_ratio"]*100:.1f}%　二维旋转{r["rotation_actual_deg"]:.1f}°');f.legend(1860);f.save()

def render():
 sys.path.insert(0,'/tmp/sar_gt_browser_deps');os.environ['PLAYWRIGHT_BROWSERS_PATH']='/tmp/sar_gt_browsers_verified'
 from playwright.sync_api import sync_playwright
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
  for m in FIGURES:
   w=3200;h=m['viewbox_height']*2;page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
   page.set_content('<html><meta charset="utf-8"><style>html,body{margin:0}svg{display:block;width:100vw;height:100vh}</style>'+m['svg'].read_text()+'</html>',wait_until='load');page.evaluate('document.fonts.ready')
   bboxes=page.evaluate('''()=>Array.from(document.querySelectorAll('text')).map(e=>{let b=e.getBBox();return {s:e.textContent,x:b.x,y:b.y,w:b.width,h:b.height,font:+e.getAttribute('font-size')}})''')
   assert all(b['x']>=0 and b['x']+b['w']<=1600.1 and b['y']>=0 and b['y']+b['h']<=m['viewbox_height'] and b['font']>=44 for b in bboxes),bboxes
   session=page.context.new_cdp_session(page);session.send('DOM.enable');session.send('CSS.enable');doc=session.send('DOM.getDocument')['root']['nodeId'];nodes=session.send('DOM.querySelectorAll',{'nodeId':doc,'selector':'text'})['nodeIds'];families=set()
   for node in nodes:
    families.update(f['familyName'] for f in session.send('CSS.getPlatformFontsForNode',{'nodeId':node})['fonts'] if f['glyphCount'])
   assert any('FandolFang' in v for v in families) and any('Times New Roman' in v for v in families),families
   page.screenshot(path=str(m['png']));page.close()
   with Image.open(m['png']) as im:im.save(m['png'],dpi=(508,508))
   m.update(pixels=[w,h],effective_dpi_at_160mm=508,font_families=sorted(families),text_clipping_checked=True);print('EXPORTED',m['stem'],flush=True)
  browser.close()

def audit_previous():
 # Review only directly relevant original SVGs. Do not modify frozen report figures.
 previous=[]
 selected=BASE/'SAR_Review_Selected_Assets_20261017/figures'
 ch4=BASE/'Report_Figures_Ch4_FangSong_20261017_2003/figures'
 for directory,patterns in [(selected,['A_network_connections_*.svg','B_transfer_training_*.svg']),(ch4,['FIG_4_1_GENERATION_FLOW_*.svg','FIG_4_2_OVERLAP_ROTATION_*.svg','FIG_4_3_CONTROLLED_APPEARANCE_*.svg','FIG_D_14POINTS_SUPERVISION_*.svg','FIG_E1_MARITIME_SCENES_*.svg'])]:
  for pattern in patterns:
   path=track(next(directory.glob(pattern)));xml=ET.parse(path).getroot();visible=[n for n in xml.iter() if n.tag.endswith('text')];data=path.read_text();assert visible and 'FandolFang' in data and 'Times New Roman' in data
   if directory==selected:
    edges=[n for n in xml.iter() if 'data-source' in n.attrib]
    if path.name.startswith('A_'):
     assert any(n.attrib['data-source']=='encoder' and n.attrib['data-target']=='detector' for n in edges);assert not any(n.attrib['data-source']=='structure' and n.attrib['data-target']=='detector' for n in edges)
   previous.append(dict(path=str(path),sha256=digest(path),text_nodes=len(visible),font_family_declarations_verified=True,visual_review='fonts, spacing, captions, arrow direction and palette reviewed from rendered originals',action='保留原图，无需重复导出',revision_required=False))
 return previous

def main_table():
 rr=readcsv('MAIN_LEARNED_BASELINES');complete=readcsv('PER_PAIR_CANONICAL');out=[]
 for m in METHODS:
  for mode in ['natural','equal']:
   r=next(r for r in rr if r['method_key']==m and r['budget']==mode);per=[p for p in complete if p['method']==m and p['protocol']==mode and p['geometry_eligible']=='True'];assert len(per)==int(r['positive_denominator'])==1140
   for t in [3,5,10]:assert sum(p[f'Success@{t}px']=='True' for p in per)==int(r[f'success_{t}px_count'])
   out.append({'方法':NAMES[m],'关键点预算':'自然点数' if mode=='natural' else '统一点数','几何正例数':1140,'Success@3px(%)':f'{float(r["success_3px_pct"]):.2f}','Success@5px(%)':f'{float(r["success_5px_pct"]):.2f}','Success@10px(%)':f'{float(r["success_10px_pct"]):.2f}','5像素成功数':int(r['success_5px_count'])})
 path=OUT/f'main_results_table_{TAG}.csv'
 with path.open('x',newline='',encoding='utf-8-sig') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 return path

def naming():
 text='''# 方法名称与技术关系

**共同评价设置。** 四种名称均表示特征网络接入同一冻结LightGlue匹配器后的完整配准流程，使用相同图像、原生坐标和几何验收规则。它们是本项目技术路线中的基线及阶段性模型，不应表述为四种来自不同论文的独立算法。

**预训练SuperPoint。** 采用公开预训练SuperPoint提取关键点与256维描述子，特征网络参数保持冻结。它是SAR域适配前的学习式基线，与冻结LightGlue共同构成对照流程。

**SAR描述子迁移基础模型。** 这是本项目第一阶段的SAR域适配模型，以预训练SuperPoint为起点，利用SAR合成几何对应微调描述子头。共享编码器、检测头和LightGlue保持冻结，网络中尚未加入多尺度结构分支；它用于衡量描述子迁移本身的作用。

**多尺度结构增强模型。** 这是本项目继SAR描述子迁移后形成的阶段性模型，增加三个尺度的高斯平滑与Scharr结构响应，经轻量卷积编码后以残差方式融入描述子路径。该阶段学习结构分支和检测头，共享编码器与基础描述子头保持冻结；检测头仍直接接收共享编码特征，结构残差不注入检测路径。它为最终特征保持迁移提供初始化。

**本项目方法。** 以多尺度结构增强模型为基础，在可靠结构区域学习描述子对应，同时在低纹理或结构不足区域保持冻结教师的同输入描述特征。该阶段仅更新描述子头，共享编码器、检测头、多尺度结构分支、教师网络与LightGlue均冻结；两项损失采用已锁定权重λ=0.328844。其作用是平衡结构描述能力与低纹理特征保持，而不是新增检测器或更换匹配框架。
'''
 path=OUT/f'method_name_notes_{TAG}.md';path.write_text(text)
 return path

def manifest(table,notes,records,rows,previous):
 lines=['# 第二研究内容：定向补图素材','',f'生成时间：{NOW.strftime("%Y-%m-%d %H:%M")}（北京时间）。交付7张结果图，每张提供可编辑SVG与3200px宽PNG；图宽160mm，最小文字12.47pt。图内中文实际采用Fandol仿宋，英文与数字采用Times New Roman。','',
 '|建议编号|推荐图题|用途|文件|','|---|---|---|---|']
 for m in FIGURES:
  lines.append(f'|{m["number"]}|{m["title"]}|{m["section"]}|[PNG]({m["png"].relative_to(OUT)}) / [SVG]({m["svg"].relative_to(OUT)})|')
 lines+=['',f'|建议编号|推荐表题或素材|用途|文件|','|---|---|---|---|',f'|表6-1|主要学习式方法的配准性能比较|实验|[{table.name}]({table.name})|',f'|—|四类方法名称及关系|方法|[{notes.name}]({notes.name})|','',
 '## 图件与数据说明','',
 '图6-1至图6-5采用统一的2×2排布：预训练SuperPoint、SAR描述子迁移基础模型、多尺度结构增强模型、本项目方法。每个子图左侧为参考图像、右侧为待配准图像；各方法从冻结候选的原顺序等距展示最多20条，线色分别表示RANSAC内点与其他匹配。图6-6和图6-7的上行为原生输入，下行为本项目方法的24条冻结候选连线。所有匹配线均来自预测文件，未按GT筛线或改写坐标。',
 '所有图像来自同一已观察的合成SAR固定Val；母图为真实SAR来源，旋转是二维几何变换。完整数据为1200条Val关系，其中几何正例1140、无重叠60；主表各阈值均以全部1140正例为分母，保留拒绝与无输出。Success@t为接受且双向check均误差不超过t；Natural每图最多1024点，Equal沿用冻结共同预算min(128,旧比较各方法实际点数)。四种方法均使用相同冻结LightGlue及Homography/RANSAC门控。',
 '主表同时保留自然与统一点数；结果分别为固定协议的完整统计，不由下述说明性样本替代。方法命名中的“基础模型”对应历史描述子适配，不是前轮结构模型初始化后的另一项描述迁移。',
 '', '## 入选样本及冻结结果','',
 '|图号|样本ID|来源场景|尺寸|重叠|二维旋转|','|---|---|---|---:|---:|---:|']
 scene_names={'offshore_vessels_low_texture_sea':'离岸弱纹理','nearshore_vessels':'近岸舰船','port_quay_dense_vessels':'港口码头','other_land_structures':'其他陆地结构'}
 for s in SELECTION:
  r=records[s['pair_id']];lines.append(f'|{s["number"]}|{s["pair_id"]}|{scene_names[r["source_scene"]]}|{r["width"]}×{r["height"]}|{r["overlap_ratio"]*100:.2f}%|{r["rotation_actual_deg"]:.2f}°|')
 lines+=['','|样本ID|方法|匹配数|RANSAC内点|静态正确匹配|S@5|双向误差(px)|','|---|---|---:|---:|---:|---|---:|']
 for pid in dict.fromkeys(s['pair_id'] for s in SELECTION):
  for m in METHODS:
   r=rows[pid,m];err=f'{float(r["bidirectional_error_px"]):.3f}' if r['bidirectional_error_px'] else '—';lines.append(f'|{pid}|{NAMES[m]}|{r["match_count"]}|{r["inliers"]}|{r["correct_structure_matches"]}|{"通过" if r["Success@5px"]=="True" else "未通过"}|{err}|')
 lines+=['',
 '场景名称按来源语境保留。离岸样本没有合格静态结构锚点，其S@5只表示同源合成几何通过。两组困难样本用于说明冻结预测的行为，按既有结果事后选择，不是独立随机测试，也不支持真实跨采集精度结论。自然点数下最终模型仍有10个正例错误接受；离岸S@5为45.25%，低于结构模型47.25%，统一预算为33.75%对39.50%。这些完整评价中的不足不因选图而改变。',
 '', '## 既有图件复核','',
 '已核查现有网络连接图、迁移训练图，以及第4章流程、重叠/旋转拼图、外观拼图、结构监督拼图和场景拼图。它们已经统一使用仿宋与Times New Roman；文字/边距未见截断，网络箭头与描述残差接入方向正确，蓝绿色配色一致，底部保留的是必要参数或图例。本轮无需重复导出这些图件。',
 '本地索引使用4-1、4-2、4-3、D、E1编号，未发现2-2、2-3、2-5、2-6的明确对应；复核按可定位的相同图件类型进行，不改写未知正文编号。本次新结果图已采用简洁方法标签、参数与图例，移除了旧结果展示中的重复长说明，文件名统一标记final。',
 '', '## 追溯与使用','',
 f'权威Val索引：`{MANIFEST_SOURCE.relative_to(BASE)}`。源预测主要位于`SAR_SP_JointTransfer_Mixed640512_20261016_2234/predictions/final/`与`SAR_SP_LowTextureKeep_20261017_0908/predictions/formal_step6000/`；具体文件、SHA、原生图像、显示匹配索引及逐对分数见`evidence/selection_and_checks_{TAG}.json`。',
 '入选图像A/B原文件逐字节保存在raw_pairs；未裁剪、调灰度或重着色。SVG只将这些灰度PNG嵌入并叠加预测线条。3200px为排版输出分辨率，不增加512/640原生SAR细节。字体为已安装开源Fandol仿宋与Times New Roman，SVG文本可编辑；Windows原版仿宋未冒称已安装。',
 f'复现导出：`conda run -n match python tools/export_report_supplement.py`。源码与命令日志保留在项目tools/logs；本轮无训练、无补推理、无匹配阈值修改。清单仅链接图件，不嵌入图片。']
 path=OUT/f'selected_assets_manifest_{TAG}.md';path.write_text('\n'.join(lines)+'\n');return path

def main():
 assert not OUT.exists() and not PHYSICAL.exists();PHYSICAL.mkdir();OUT.symlink_to(PHYSICAL,target_is_directory=True)
 for d in ['selected_figures','raw_pairs','evidence']:(OUT/d).mkdir()
 records={r['pair_id']:r for r in map(json.loads,track(MANIFEST_SOURCE).read_text().splitlines())}
 csvrows=readcsv('PER_PAIR_CANONICAL');rows={(r['pair_id'],r['method']):r for r in csvrows if r['protocol']=='natural' and r['method'] in METHODS}
 spec=[('MIX_val_00023','不同方法在离岸弱纹理场景下的匹配结果对比','A1_offshore_comparison','图6-1'),('MIX_val_00793','不同方法在近岸复杂场景下的匹配结果对比','A2_nearshore_comparison','图6-2'),('MIX_val_00486','不同方法在港口场景下的匹配结果对比','A3_port_comparison','图6-3'),('MIX_val_00054','不同方法在低重叠条件下的匹配结果对比','B1_low_overlap_comparison','图6-4'),('MIX_val_00174','不同方法在大旋转条件下的匹配结果对比','B2_large_rotation_comparison','图6-5')]
 raw={}
 for pid in dict.fromkeys(x[0] for x in spec):
  for L in 'AB':
   src=track(records[pid]['image_'+L]);dst=OUT/'raw_pairs'/f'{pid}_{L}_{TAG}.png';shutil.copyfile(src,dst);assert digest(src)==digest(dst);raw[pid,L]=dst
 for pid,title,filename,number in spec:
  assert rows[pid,'keep']['Success@5px']=='True';SELECTION.append(dict(pair_id=pid,number=number,title=title));comparison(pid,rows,records,raw,title,filename,number)
 standalone('MIX_val_00793',rows,records,raw,'本项目方法在近岸场景中的匹配结果示例','C1_method_nearshore','图6-6')
 standalone('MIX_val_00486',rows,records,raw,'本项目方法在港口场景中的匹配结果示例','C2_method_port','图6-7')
 table=main_table();notes=naming();render();previous=audit_previous();index=manifest(table,notes,records,rows,previous)
 evidence=dict(created=NOW.isoformat(),configuration=dict(methods=NAMES,mode='natural',comparison_line_cap=20,single_line_cap=24,sampling='linspace in frozen match order; no GT/confidence filtering',input_grayscale_changed=False,seed=42,training=False,new_match_inference=False),figures=[{k:str(v.relative_to(OUT)) if isinstance(v,Path) else v for k,v in m.items()} for m in FIGURES],selection=[dict(**s,record=records[s['pair_id']],method_scores=[rows[s['pair_id'],m] for m in METHODS]) for s in SELECTION],display_indices=DISPLAY,previous_figure_reviews=previous,input_sha256=INPUTS,checkpoint_identity=readcsv('CHECKPOINT_IDENTITY'),export_code_sha256=digest(BASE/'tools/export_report_supplement.py'),verification=dict(complete_Val_table_recomputed_from_frozen_per_pair=True,all_predictions_match_saved_counts=True,all_raw_images_byte_preserved=True,all_text_boxes_in_canvas=True,font_CDP_checked=True,figure_count=7,no_old_outputs_modified=True))
 # Readcsv above only adds a direct identity; freeze the final source dictionary after all accesses.
 evidence['input_sha256']=INPUTS
 (OUT/'evidence'/f'selection_and_checks_{TAG}.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
 assert len(list((OUT/'selected_figures').glob('*.png')))==len(list((OUT/'selected_figures').glob('*.svg')))==7
 for p in OUT.rglob('*.png'):
  with Image.open(p) as im:
   if p.parent.name=='selected_figures':assert im.width==3200 and im.width/160*25.4>=300
 for p,expected in INPUTS.items():assert digest(p)==expected
 for p in OUT.rglob('*'):
  if p.is_file():p.chmod(0o444)
 for p in sorted((p for p in OUT.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):p.chmod(0o555)
 OUT.chmod(0o555)
 archive=PHYSICAL.parent/f'{NAME}.zip';assert not archive.exists()
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(p for p in OUT.rglob('*') if p.is_file()):z.write(p,arcname=f'{NAME}/{p.relative_to(OUT)}')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for p in OUT.rglob('*'):
   if p.is_file():assert hashlib.sha256(z.read(f'{NAME}/{p.relative_to(OUT)}')).hexdigest()==digest(p)
 archive.chmod(0o444);(BASE/archive.name).symlink_to(archive)
 print(json.dumps(dict(completed=True,output=str(OUT),manifest=str(index),table=str(table),notes=str(notes),zip=str(BASE/archive.name),zip_bytes=archive.stat().st_size,zip_sha256=digest(archive),figures=7,frozen_inputs_unchanged=True,CRC_and_per_file_SHA_verified=True),ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
