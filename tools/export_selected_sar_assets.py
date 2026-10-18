"""Selected publication assets from frozen evidence. No model loading or inference."""
import argparse,base64,csv,hashlib,html,io,json,math,os,shutil,sys,zipfile,stat
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np,cv2
from PIL import Image
B=Path('/home/yl/matching/sar_matching');OLD=B/'SAR_Report_Ch5Ch6_Assets_20261017_1821';OT='20261017_1821'
START=datetime.now(ZoneInfo('Asia/Shanghai'));T=START.strftime('%Y%m%d_%H%M');NAME='SAR_Review_Selected_Assets_20261017';ROOT=B/NAME;PHYSICAL=Path('/data1/libo/yl/ckpt/sar_matching')/NAME
FONT="'Times New Roman','FangSong','仿宋','FandolFang',serif"
BLUE='#dfe8ec';TEAL='#dfebe7';WARM='#f0dfc5';INK='#17252b';LINE='#405d68';MATCH='#69b9b0';ACCENT='#d5a36b'
META=[];INPUTS=[];MIN_FONT=44

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readcsv(name):return list(csv.DictReader((OLD/'csv'/f'{name}_{OT}.csv').open()))
def noteinput(p):
 p=Path(p);r=dict(path=str(p),sha256=sha(p));INPUTS.append(r);return r

def csvwrite(name,rows):
 path=ROOT/'csv'/f'{name}_{T}.csv'
 with path.open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return path

def typed(rows):
 for r in rows:
  for k,v in r.items():
   if v=='':r[k]=None
   elif v in ['True','False']:r[k]=v=='True'
   else:
    try:r[k]=json.loads(v)
    except (ValueError,TypeError):pass
 return rows

class SVG:
 def __init__(self,h,description):
  self.w=1600;self.h=h;self.description=description;self.texts=[];self.edges=[];self.nodes={};self.b=[f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="160mm" height="{h/10}mm" viewBox="0 0 1600 {h}"><title>{html.escape(description)}</title><defs><marker id="flow" markerWidth="15" markerHeight="15" refX="13" refY="7.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L15 7.5 L0 15Z" fill="{LINE}"/></marker><marker id="optim" markerWidth="15" markerHeight="15" refX="13" refY="7.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L15 7.5 L0 15Z" fill="{ACCENT}"/></marker></defs><rect width="1600" height="{h}" fill="#ffffff"/>']
 def text(self,x,y,s,fs=44,anchor='middle',id=None):
  assert fs>=44;tid=id or f'text_{len(self.texts)}';self.texts.append((tid,fs));self.b.append(f'<text id="{tid}" x="{x}" y="{y}" font-family="{FONT}" font-size="{fs}" fill="{INK}" text-anchor="{anchor}">{html.escape(str(s))}</text>')
 def lines(self,x,y,ss,fs=46,dy=62):
  for i,s in enumerate(ss):self.text(x,y+i*dy,s,fs)
 def box(self,id,x,y,w,h,ss,color=BLUE,fs=46):
  self.nodes[id]=(x,y,w,h);self.b.append(f'<rect id="{id}" x="{x}" y="{y}" width="{w}" height="{h}" rx="15" fill="{color}" stroke="{LINE}" stroke-width="3.4"/>');self.lines(x+w/2,y+h/2-(len(ss)-1)*31+16,ss,fs)
 def arrow(self,id,pts,source,target,optim=False):
  self.edges.append(dict(id=id,source=source,target=target,optimization=optim,points=pts));self.b.append(f'<polyline id="{id}" data-source="{source}" data-target="{target}" points="'+ ' '.join(f'{x},{y}' for x,y in pts)+f'" fill="none" stroke="{ACCENT if optim else LINE}" stroke-width="3.4" stroke-linejoin="round"'+(' stroke-dasharray="16 11"' if optim else '')+f' marker-end="url(#{"optim" if optim else "flow"})"/>')
 def image(self,path,x,y,w,h):
  data=base64.b64encode(Path(path).read_bytes()).decode();self.b.append(f'<image x="{x}" y="{y}" width="{w}" height="{h}" preserveAspectRatio="none" xlink:href="data:image/png;base64,{data}"/>')
 def point(self,x,y,color=MATCH):self.b.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="6" fill="none" stroke="#17252b" stroke-width="6.4"/><circle cx="{x:.3f}" cy="{y:.3f}" r="6" fill="none" stroke="{color}" stroke-width="3.4"/>')
 def line(self,x1,y1,x2,y2,color=MATCH):
  self.b.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="#17252b" stroke-width="6.4"/><line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{color}" stroke-width="3.4"/>')
 def save(self,name):
  path=ROOT/'figures'/f'{name}_{T}.svg';path.write_text(''.join(self.b)+'</svg>');META.append(dict(stem=name,description=self.description,svg=path,png=path.with_suffix('.png'),height=self.h,edges=self.edges,nodes=self.nodes,min_font_pt=44/1600*160/25.4*72,min_line_pt=3.4/1600*160/25.4*72))

def network():
 s=SVG(2240,'实际网络连接；结构残差只进入描述子路径')
 s.box('input',570,40,460,150,['SAR灰度输入','像素值除以255'],TEAL)
 s.box('encoder',70,300,760,260,['共享编码器','八层卷积，三次降采样','128维特征，步长8'])
 s.box('gaussian',1000,300,540,210,['对数灰度','高斯尺度：1、2、4'],TEAL)
 s.arrow('input_encoder',[(670,190),(670,245),(450,245),(450,300)],'input','encoder')
 s.arrow('input_gaussian',[(930,190),(1270,190),(1270,300)],'input','gaussian')
 s.box('scharr',1000,600,540,150,['Scharr梯度幅值','三尺度响应'],TEAL);s.arrow('gauss_scharr',[(1270,510),(1270,600)],'gaussian','scharr')
 s.box('pool',1000,850,540,150,['平均池化','下采样8倍'],TEAL);s.arrow('scharr_pool',[(1270,750),(1270,850)],'scharr','pool')
 s.box('structure',1000,1100,540,150,['结构卷积映射','3→16→32→256']);s.arrow('pool_structure',[(1270,1000),(1270,1100)],'pool','structure')
 s.box('detector',70,820,360,170,['关键点检测头','128→256→65'])
 s.box('convDa',520,820,380,170,['描述卷积','128→256'],WARM)
 s.arrow('encoder_detector',[(250,560),(250,820)],'encoder','detector');s.arrow('encoder_descriptor',[(710,560),(710,820)],'encoder','convDa')
 s.nodes['sum']=(690,1380,40,40);s.b.append(f'<circle id="sum" cx="710" cy="1400" r="20" fill="#ffffff" stroke="{LINE}" stroke-width="3.4"/>');s.text(710,1416,'+',48)
 s.arrow('convDa_sum',[(710,990),(710,1380)],'convDa','sum');s.arrow('structure_sum',[(1270,1250),(1270,1400),(730,1400)],'structure','sum');s.text(1050,1345,'残差相加',44)
 s.box('nms',70,1150,360,250,['响应归一化','非极大抑制','关键点筛选'],TEAL);s.arrow('detector_nms',[(250,990),(250,1150)],'detector','nms')
 s.box('keypoints',70,1640,360,140,['关键点坐标'],TEAL);s.arrow('nms_points',[(250,1400),(250,1640)],'nms','keypoints')
 s.box('convDb',520,1550,500,220,['激活与描述卷积','逐位置归一化','256维稠密描述'],WARM);s.arrow('sum_convDb',[(710,1420),(710,1550)],'sum','convDb')
 s.box('sample',480,1940,600,160,['关键点处双线性采样','256维归一化描述子'],TEAL);s.arrow('points_sample',[(250,1780),(250,2020),(480,2020)],'keypoints','sample');s.arrow('dense_sample',[(770,1770),(770,1940)],'convDb','sample')
 s.text(800,2180,'蓝灰：冻结　青绿：固定运算　浅橙：可训练',44)
 assert not any(e['source']=='structure' and e['target']=='detector' for e in s.edges)
 s.save('A_network_connections')

def training():
 s=SVG(1900,'迁移训练：可靠结构描述学习与同输入低纹理特征保持')
 s.box('desc_input',50,40,650,170,['可靠结构图像对','每步4组'],TEAL)
 s.box('keep_input',900,40,650,170,['低纹理保持图像对','每步4组'],TEAL)
 s.box('student',50,390,650,430,[],BLUE);s.text(375,455,'学生网络',48)
 s.text(375,510,'编码、检测与结构分支冻结',44);s.text(375,575,'两条数据流使用同一学生',44)
 s.box('train_head',120,650,510,120,['描述子头可训练'],WARM)
 s.box('teacher',900,390,650,380,['教师网络','全部冻结，停止梯度','同一输入、相同采样位置','仅用于训练'])
 s.arrow('desc_student',[(375,210),(375,390)],'desc_input','student')
 s.arrow('keep_student',[(1070,210),(1070,295),(555,295),(555,390)],'keep_input','student')
 s.arrow('keep_teacher',[(1380,210),(1380,390)],'keep_input','teacher')
 s.box('desc_loss',50,990,650,160,['可靠结构描述损失','双向匹配与合法负例'],TEAL)
 s.box('keep_loss',900,990,650,160,['低纹理特征保持损失','同位置单位描述子余弦差'],TEAL)
 s.arrow('student_desc_loss',[(375,770),(375,990)],'train_head','desc_loss')
 s.arrow('student_keep_loss',[(630,710),(790,710),(790,880),(1070,880),(1070,990)],'train_head','keep_loss')
 s.arrow('teacher_keep_loss',[(1380,770),(1380,990)],'teacher','keep_loss')
 s.box('total_loss',460,1270,680,170,['描述损失＋λ×保持损失','λ＝0.328844'],TEAL)
 s.arrow('desc_total',[(375,1150),(375,1210),(610,1210),(610,1270)],'desc_loss','total_loss')
 s.arrow('keep_total',[(1225,1150),(1225,1210),(990,1210),(990,1270)],'keep_loss','total_loss')
 s.box('update',50,1570,650,150,['仅更新描述子头','教师不进入推理'],WARM)
 s.box('lightglue',900,1570,650,150,['LightGlue匹配器','保持冻结'])
 s.arrow('total_update',[(800,1440),(800,1500),(375,1500),(375,1570)],'total_loss','update',True)
 s.arrow('update_head',[(50,1645),(25,1645),(25,710),(120,710)],'update','train_head',True)
 s.text(800,1820,'实线：特征与损失计算　虚线：描述子优化',44)
 s.save('B_transfer_training')

def prediction(pid):
 p=OLD/'evidence/predictions'/f'{pid}_keep_natural_{OT}.npz';noteinput(p)
 with np.load(p) as z:return {k:z[k].copy() for k in z.files},p

def pairbase(row,height):
 s=SVG(height,'冻结纯图像匹配结果；原始灰度与稀疏连线');n=int(row['native_width']);scale=740/n
 for L,x in [('A',40),('B',820)]:s.image(ROOT/'raw'/f'{row["pair_id"]}_{L}_{T}.png',x,110,740,740)
 s.text(410,65,f'参考图像（{n}×{n}）',44);s.text(1190,65,f'待配准图像（{n}×{n}）',44)
 return s,scale

def matchfigure(row,name):
 p,src=prediction(row['pair_id']);s,scale=pairbase(row,1230);a=p['matches_A'];b=p['matches_B'];ins=p['inliers'].astype(bool).reshape(-1)
 ids=np.linspace(0,len(a)-1,min(24,len(a)),dtype=int)
 for i in ids:
  color=MATCH if ins[i] else ACCENT
  x1,y1=40+scale*a[i,0],110+scale*a[i,1];x2,y2=820+scale*b[i,0],110+scale*b[i,1]
  s.line(x1,y1,x2,y2,color);s.point(x1,y1,color);s.point(x2,y2,color)
 s.text(800,920,f'重叠{float(row["overlap_fraction"])*100:.1f}%　二维旋转{float(row["synthetic_rotation_deg"]):.1f}°',44)
 s.text(800,985,f'候选{len(a)}组　内点{int(ins.sum())}组　显示{len(ids)}组',44)
 s.text(800,1050,'5像素判据：成功（合成几何）',44)
 s.text(800,1115,'青绿：几何内点　浅橙：其余候选',44)
 s.text(800,1180,'仅图像预测；连线不等于物理正确对应',44)
 s.save(name);META[-1].update(pair_id=row['pair_id'],display_match_indices=ids.tolist(),prediction=str(src),prediction_sha256=sha(src),GT_used_for_match_selection=False)

def failure(row,diag):
 p,src=prediction(row['pair_id']);s,scale=pairbase(row,1390);a=p['matches_A'];b=p['matches_B'];ins=p['inliers'].astype(bool).reshape(-1);ids=np.flatnonzero(ins)
 for L,points,x in [('A',a[ins],40),('B',b[ins],820)]:
  hull=cv2.convexHull(points.astype(np.float32)).reshape(-1,2);coords=' '.join(f'{x+scale*u:.2f},{110+scale*v:.2f}' for u,v in hull);s.b.append(f'<polygon points="{coords}" fill="{MATCH}" fill-opacity="0.18" stroke="{MATCH}" stroke-width="3.4"/>')
 for i in ids:
  x1,y1=40+scale*a[i,0],110+scale*a[i,1];x2,y2=820+scale*b[i,0],110+scale*b[i,1];s.line(x1,y1,x2,y2);s.point(x1,y1);s.point(x2,y2)
 # These check points were already used in the frozen post-hoc diagnostic. They never become matching lines.
 gtpath=OLD/'evidence/evaluation_gt'/f'{row["pair_id"]}_GT_{OT}.npz';noteinput(gtpath)
 with np.load(gtpath) as gt:checks=[gt['check_points_'+L].copy() for L in 'AB']
 show=np.linspace(0,len(checks[0])-1,min(12,len(checks[0])),dtype=int)
 for points,x in [(checks[0],40),(checks[1],820)]:
  for q in points[show]:
   u,v=x+scale*q[0],110+scale*q[1]
   for dx,dy in [(8,8),(8,-8)]:s.line(u-dx,v-dy,u+dx,v+dy,ACCENT)
 s.text(800,920,f'内点{int(ins.sum())}组　双图凸包覆盖约3.4%',44)
 s.text(800,985,f'内点拟合残差{float(diag["candidate_inlier_rmse"]):.2f}像素',44)
 s.text(800,1050,f'凸包内／外检查：{float(diag["inside_check_error"]):.2f}／{float(diag["outside_check_error"]):.2f}像素',44)
 s.text(800,1115,f'全幅检查均误差{float(diag["global_error"]):.2f}像素：5像素失败',44)
 s.text(800,1180,'青绿：内点支持　浅橙叉：事后检查点',44)
 s.text(800,1245,f'{100*float(diag["check_outside_inlier_hull_fraction"]):.1f}%检查点在双图共同凸包支持外',44)
 s.text(800,1310,'检查点仅用于事后诊断，不参与预测',44)
 s.save('E_failure_spatial_support');META[-1].update(pair_id=row['pair_id'],display_match_indices=ids.tolist(),prediction=str(src),prediction_sha256=sha(src),posthoc_check_indices=show.tolist(),frozen_failure_facts=diag)

def tables():
 main=readcsv('MAIN_LEARNED_BASELINES');controls=readcsv('STRICT_KEEP_ABLATION');scenes=readcsv('SCENE_RESULTS');sel=readcsv('QUALITATIVE_SELECTION');allrows=typed(readcsv('PER_PAIR_CANONICAL'));fixed=typed(readcsv('FIXED6000_PER_PAIR'))
 labels={'m0':'原始SuperPoint+LightGlue','m2':'SAR迁移基础网络+LightGlue','b':'多尺度结构增强网络+LightGlue','keep':'结构增强与低纹理保持（最终方法）','lambda0':'同采样取消保持梯度'}
 mainfields=['method_key','method','budget','pair_count','positive_denominator','success_3px_count','success_3px_pct','success_5px_count','success_5px_pct','success_10px_count','success_10px_pct']
 for r in main:r['method']=labels[r['method_key']]
 mainout=[{k:r[k] for k in mainfields} for r in main];csvwrite('main_learned',mainout)
 cf=['comparison','method_key','method','budget','step','lambda_keep','positive_denominator','success_5px_count','success_5px_pct','offshore_denominator','offshore_success_5px_count','offshore_success_5px_pct','structure_denominator','structure_success_5px_count','structure_success_5px_pct','positive_wrong_accept_5px_count','positive_wrong_accept_5px_pct','no_overlap_denominator','no_overlap_false_accept_count','no_overlap_false_accept_pct']
 ctrlout=[]
 for r in controls:
  r['method']=labels[r['method_key']];out={k:r[k] for k in cf};m=r['method_key'];mode=r['budget'];pool=allrows if r['comparison']=='selected_checkpoints' else fixed
  aa={x['pair_id']:x for x in pool if x['method']=='lambda0' and x['protocol']==mode};kk={x['pair_id']:x for x in pool if x['method']=='keep' and x['protocol']==mode}
  for group,accept in [('overall',lambda x:x['geometry_eligible']),('offshore',lambda x:x['geometry_eligible'] and x['scene']=='offshore_vessels_low_texture_sea'),('structure',lambda x:x['structure_global_eligible'])]:
   ids=[i for i in aa if accept(aa[i])];out[f'keep_restored_vs_lambda0_{group}']=sum(not aa[i]['Success@5px'] and bool(kk[i]['Success@5px']) for i in ids);out[f'keep_lost_vs_lambda0_{group}']=sum(bool(aa[i]['Success@5px']) and not kk[i]['Success@5px'] for i in ids)
  ctrlout.append(out)
 csvwrite('keep_ablation',ctrlout)
 sf=['method_key','method','budget','stratum_value','source_scene_label','pair_count','positive_denominator','structure_denominator','success_5px_count','success_5px_pct','positive_wrong_accept_5px_count','no_overlap_denominator','no_overlap_false_accept_count']
 sceneout=[]
 for r in scenes:
  if r['method_key'] not in ['m0','m2','b','keep']:continue
  r['method']=labels[r['method_key']];out={k:r[k] for k in sf};ref=next(x for x in scenes if x['method_key']=='b' and x['budget']==r['budget'] and x['stratum_value']==r['stratum_value']);out['S5_difference_vs_structure_base_pp']=float(r['success_5px_pct'])-float(ref['success_5px_pct']);sceneout.append(out)
 csvwrite('scene_results',sceneout)
 # Validate each aggregate against complete frozen per-pair rows, without re-running evaluation.
 for out in mainout+sceneout:
  rr=[r for r in allrows if r['method']==out['method_key'] and r['protocol']==out['budget'] and ('stratum_value' not in out or r['scene']==out['stratum_value'])];pp=[r for r in rr if r['geometry_eligible']]
  assert len(pp)==int(out['positive_denominator']);assert sum(bool(r['Success@5px']) for r in pp)==int(out['success_5px_count']);assert math.isclose(float(out['success_5px_pct']),100*sum(bool(r['Success@5px']) for r in pp)/len(pp),abs_tol=1e-12)
 chosen=[next(r for r in sel if r['pair_id']==pid) for pid in ['MIX_val_00023','MIX_val_00793','MIX_val_00910']]
 for r in chosen:
  for L in 'AB':
   src=OLD/r['raw_'+L+'_path'];dest=ROOT/'raw'/f'{r["pair_id"]}_{L}_{T}.png';shutil.copyfile(src,dest);assert sha(src)==sha(dest)==r['raw_'+L+'_sha'];noteinput(src)
 return chosen,allrows

def render():
 sys.path.insert(0,'/tmp/sar_gt_browser_deps');os.environ['PLAYWRIGHT_BROWSERS_PATH']='/tmp/sar_gt_browsers_verified'
 from playwright.sync_api import sync_playwright
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
  for m in META:
   w=3200;h=m['height']*2;page=browser.new_page(viewport={'width':w,'height':h},device_scale_factor=1)
   page.set_content('<html><meta charset="utf-8"><style>html,body{margin:0}svg{display:block;width:100vw;height:100vh}</style>'+m['svg'].read_text()+'</html>',wait_until='load');page.evaluate('document.fonts.ready')
   # Check every text is editable and legible at 160mm; detect overflow, not just file dimensions.
   bb=page.evaluate('''()=>Array.from(document.querySelectorAll('text')).filter(e=>e.textContent.trim()).map(e=>{let b=e.getBBox();return {id:e.id,text:e.textContent,x:b.x,y:b.y,w:b.width,h:b.height,font:e.getAttribute('font-family'),size:+e.getAttribute('font-size')}})''')
   assert all(t['x']>=0 and t['y']>=0 and t['x']+t['w']<=1600.1 and t['y']+t['h']<=m['height']+.1 and t['size']>=44 for t in bb),[t for t in bb if t['x']<0 or t['x']+t['w']>1600.1]
   # Chromium tells us the real glyph fonts, rather than trusting CSS family declarations.
   session=page.context.new_cdp_session(page);session.send('DOM.enable');session.send('CSS.enable');doc=session.send('DOM.getDocument')['root']['nodeId'];ns=session.send('DOM.querySelectorAll',{'nodeId':doc,'selector':'text'})['nodeIds'];fonts=[]
   for node in ns:
    fonts+=session.send('CSS.getPlatformFontsForNode',{'nodeId':node})['fonts']
   families=sorted({f['familyName'] for f in fonts if f['glyphCount']});assert any('FandolFang' in f for f in families),families;assert any('Times New Roman' in f for f in families),families
   page.screenshot(path=str(m['png']));page.close()
   with Image.open(m['png']) as im:im.save(m['png'],dpi=(508,508))
   m.update(png_width=w,png_height=h,actual_rendered_families=families,font_glyph_counts={k:sum(f['glyphCount'] for f in fonts if f['familyName']==k) for k in families},text_overflow_checked=True)
   print('RENDERED',m['stem'],families,flush=True)
  browser.close()

def manifest(chosen,diag):
 noteinput(OLD/'csv'/f'MAIN_LEARNED_BASELINES_{OT}.csv');noteinput(OLD/'csv'/f'STRICT_KEEP_ABLATION_{OT}.csv');noteinput(OLD/'csv'/f'SCENE_RESULTS_{OT}.csv');noteinput(OLD/'csv'/f'PER_PAIR_CANONICAL_{OT}.csv');noteinput(OLD/'csv'/f'FIXED6000_PER_PAIR_{OT}.csv')
 rows=[]
 for r in chosen:
  pid=r['pair_id'];src=OLD/'evidence/predictions'/f'{pid}_keep_natural_{OT}.npz';row={k:r[k] for k in ['pair_id','source_scene','source_scene_label','native_width','native_height','overlap_fraction','synthetic_rotation_deg','structure_global_eligible','structural_anchor_count','accepted','success_5px','check_bidirectional_error_px','matches','inliers','correct_static_matches','selection_reason']};row.update(raw_A_path=f'raw/{pid}_A_{T}.png',raw_B_path=f'raw/{pid}_B_{T}.png',original_image_A=str(B/'SyntheticSAR_Mixed640512_Multicondition_20261016_2114/pairs/val'/pid/f'{pid}_A.png'),original_image_B=str(B/'SyntheticSAR_Mixed640512_Multicondition_20261016_2114/pairs/val'/pid/f'{pid}_B.png'),frozen_prediction_source=str(src),frozen_prediction_sha256=sha(src),correctness_basis='固定合成GT的接受与双向check误差；不代表真实跨采集物理对应',selection_posthoc=True)
  if pid==diag['pair_id']:
   for k in ['inlier_hull_fraction','moving_inlier_hull_fraction','inside_check_error','outside_check_error','check_outside_inlier_hull_fraction','candidate_inlier_rmse']:row[k]=diag[k]
  rows.append(row)
 # Fixed union fields so the failure row doesn't force extra tables.
 fields=list(dict.fromkeys(k for r in rows for k in r));csvwrite('qualitative_ids',[{k:r.get(k,'') for k in fields} for r in rows])
 wid=readcsv('CHECKPOINT_IDENTITY');fontrecord=json.loads(Path('/tmp/sar_font_setup_20261017_1958/FONT_INSTALLATION.json').read_text())
 lines=['# 精选图件与冻结数字证据','',f'生成：{START.isoformat()}。本包仅含5张主图（SVG+PNG）、4个CSV、3组未注释原图及本清单。不提供章节正文、Word或PDF；本MD仅链接图片，不嵌入图像。',
 '', '## 图件（Word按160mm宽插入）','', '|文件|内容与事实依据|尺寸|','|---|---|---|']
 for m in META:
  stem=m['stem'];sv=str(m['svg'].relative_to(ROOT));pn=str(m['png'].relative_to(ROOT));lines.append(f'|[SVG]({sv}) / [PNG]({pn})|{m["description"]}|160×{m["height"]/10:.0f}mm；{m["png_width"]}×{m["png_height"]}px；508dpi|')
 lines+=['','网络图：结构响应→平均池化→卷积映射只接`convDa(X)`输出的加法节点，然后激活/convDb。检测头直接接共享编码X，无结构分支入边。依据`SAR_SP_LowTextureKeep_20261017_0908/code/runtime.py:68–80`、`SAR_SP_LG_StructureStable_20261013_2210/code/network_corrected.py:10–26,38–44`；可编辑模块/箭头保留id及data-source/target。',
 '训练图：仅convDa/convDb四个参数张量（360960参数）更新；教师、共享编码、检测、结构分支及LightGlue冻结。结构/保持各4组，损失各按本流均值；实线是特征/损失计算，暖色虚线仅返回描述子头。依据`keep_objective.py:6–50`、`keep_data.py:10–40,73–86`、`keep_train.py:27–37,70–74`。',
 '匹配图：沿冻结候选原顺序等距抽24组，显示其关键点端点；保留非内点。图像灰度、匹配坐标、内点标记和预测H均未改写；不按GT筛选连线，无GT预对齐。未重复导出叠合近似版本。',
 '失败图：MIX_val_00910属于已冻结10个Natural错误接受之一；显示全部19内点及其凸包，12组事后check点仅用叉号显示，不连接成预测匹配。凸包内/外误差0.45/9.57px，96.2%check在共同支持外，全幅9.23px；均来自`FROZEN_FAILURE_DETAILS_20261017_1621.csv`的keep/natural行，未重新拟合或改门控。',
 '', '## 数据与表格','',
 f'- `main_learned_{T}.csv`：4类模型×Natural/Equal=8行，S3/5/10保留成功数与分母；SAR迁移基础指历史m2（无结构分支），不与前轮结构初始化描述迁移t2混淆。',
 f'- `keep_ablation_{T}.csv`：严格同4+4采样λ=0/0.328844的选中checkpoint与固定6000步对照，各2预算；包含各分母、错误接受与逐对恢复/损失计数。λ=0选中4000步、保持选中6000步，训练均6000步。',
 f'- `scene_results_{T}.csv`：4模型×2预算×4来源场景=32行，场景标签是来源语境，非逐裁片人工语义；相对结构基础的差值列未省略退化。',
 f'- `qualitative_ids_{T}.csv`：两组成功和一组失败的尺寸、重叠、二维旋转、来源、评判及原图/预测路径。',
 '权威数据：`SyntheticSAR_Mixed640512_Multicondition_20261016_2114/manifests/val_CANONICAL_20261016_2147.jsonl`；1200关系=1140几何正例+60负例，离岸400正例、结构合格348。Val已被开发观察；原900关系未重跑。',
 'Natural冻结阈值/NMS下每图≤1024；Equal沿用冻结共同上限min(128,旧比较各方法实际点数)，少于预算不补点。S@t=门控接受且原生双向check均误差≤t，失败/拒绝/无输出仍在正例分母。正例错接受按5px，无重叠误接受用60负例，不能混合。百分数列单位0–100；空单元为不适用。',
 '最终方法离岸S@5=45.25%（Natural）、33.75%（Equal），历史结构基础47.25%/39.50%，仍下降2.00/5.75个百分点。无效静态证据的低纹理样例仅为合成几何成功，不声称真实物理对应。ORB保留在上一素材包附录，本包不重复，也不声称所有预算全面领先。',
 '', '## 字体与保真','',
 '已安装用户级Times New Roman 2.82（四字形）及FandolFang开源仿宋，未更改Conda/系统依赖。SVG每个text保留`font-family="Times New Roman,FangSong,仿宋,FandolFang,serif"`且未转曲；英文数字优先Times New Roman，中文实际是Fandol仿宋，**不是微软FangSong/simfang.ttf**。浏览器CDP逐文字节点确认实际字形来自Times New Roman/FandolFang，无Noto代替；仅μ/符号可能由记录的字体提供。',
 '字体来源：https://downloads.sourceforge.net/corefonts/times32.exe；https://ctan.org/tex-archive/fonts/fandol（GPL+font exception）。不随本包分发字体二进制。最小44 SVG单位在160mm宽等效12.47pt，线宽3.4单位约0.96pt；PNG3200px宽，等效508dpi；放大素材不增加原SAR原生分辨率。所有文字边界已测量，未溢出画布。',
 '', '## 冻结权重及追溯','', '|模型|源文件|SHA-256|','|---|---|---|']
 for w in wid:
  if w['method_key'] not in ['m0','m2','b','keep','lambda0','lightglue']:continue
  lines.append(f'|{w["method_key"]}|{w["source"]}|`{w["sha256"]}`|')
 lines+=['', '导出脚本：`tools/export_selected_sar_assets.py`（完整确定性抽样、筛表、校验及绘图实现）；运行环境match，CPU导出，无训练/新匹配/协议修改。旧素材包与全部冻结数据保持不变。原图按字节复制、SVG嵌入同一原图，清单最后列出交付有效载荷SHA。','', '|文件|字节数|SHA-256|','|---|---:|---|']
 for p in sorted(p for p in ROOT.rglob('*') if p.is_file()):lines.append(f'|{p.relative_to(ROOT)}|{p.stat().st_size}|`{sha(p)}`|')
 lines+=['',f'本清单档案`MANIFEST_{T}.md`，固定名`MANIFEST.md`为同一正文入口；ZIP展开为普通文件。本清单不递归计算自身SHA。',
 '关键直接输入（源文件未改写）：']
 for i in {r['path']:r for r in INPUTS}.values():lines.append(f'- `{i["path"]}` — `{i["sha256"]}`')
 for f in fontrecord['fonts']:
  if f['file'].endswith(('Times.TTF','FandolFang-Regular.otf')):lines.append(f'- 实际字体`{Path(f["file"]).name}`：`{f["sha256"]}`。')
 lines.append(f'- 导出脚本SHA：`{sha(B/"tools/export_selected_sar_assets.py")}`。')
 path=ROOT/f'MANIFEST_{T}.md';path.write_text('\n'.join(lines)+'\n');(ROOT/'MANIFEST.md').symlink_to(path.name)
 assert len(list((ROOT/'csv').glob('*.csv')))==4;assert len(list((ROOT/'figures').glob('*.svg')))==5;assert len(list((ROOT/'figures').glob('*.png')))==5;assert len(list((ROOT/'raw').glob('*.png')))==6
 for i in INPUTS:assert sha(i['path'])==i['sha256']
 for m in META:
  with Image.open(m['png']) as im:assert im.size[0]/160*25.4>=300 and im.size==(m['png_width'],m['png_height'])
  for edge in m['edges']:assert edge['source'] in m['nodes'] and edge['target'] in m['nodes']
 for p in ROOT.rglob('*'):
  if p.is_file() and not p.is_symlink():p.chmod(0o444)
 for p in sorted((p for p in ROOT.rglob('*') if p.is_dir()),key=lambda x:len(x.parts),reverse=True):p.chmod(0o555)
 ROOT.chmod(0o555)
 zip_path=PHYSICAL.parent/f'{NAME}_{T}.zip';assert not zip_path.exists()
 with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(p for p in ROOT.rglob('*') if p.is_file()):z.write(p,arcname=f'{NAME}/{p.relative_to(ROOT)}')
 with zipfile.ZipFile(zip_path) as z:
  assert z.testzip() is None
  for p in ROOT.rglob('*'):
   if p.is_file():assert hashlib.sha256(z.read(f'{NAME}/{p.relative_to(ROOT)}')).hexdigest()==sha(p)
 zip_path.chmod(0o444);(B/zip_path.name).symlink_to(zip_path)
 print(json.dumps(dict(completed=True,directory=str(ROOT),physical=str(PHYSICAL),zip=str(B/zip_path.name),zip_bytes=zip_path.stat().st_size,zip_sha256=sha(zip_path),main_figures=5,csv_files=4,native_original_images=6,min_font_pt=12.47244094488189,actual_fonts=sorted(set(f for m in META for f in m['actual_rendered_families'])),no_new_training_or_matching=True,ZIP_CRC_and_payload_SHA_passed=True),ensure_ascii=False,indent=2),flush=True)

def main():
 assert not ROOT.exists() and not PHYSICAL.exists(),'Never overwrite an existing selected packet'
 PHYSICAL.mkdir();ROOT.symlink_to(PHYSICAL,target_is_directory=True)
 for d in ['figures','csv','raw']:(ROOT/d).mkdir()
 chosen,rawrows=tables()
 diagpath=B/'SAR_SP_KeepAblation_20261017_1343/FROZEN_FAILURE_DETAILS_20261017_1621.csv';noteinput(diagpath);diag=next(r for r in csv.DictReader(diagpath.open()) if r['pair_id']=='MIX_val_00910' and r['method']=='keep' and r['protocol']=='natural')
 network();training();matchfigure(chosen[0],'C_low_texture_match');matchfigure(chosen[1],'D_nearshore_match');failure(chosen[2],diag);render();manifest(chosen,diag)
if __name__=='__main__':main()
