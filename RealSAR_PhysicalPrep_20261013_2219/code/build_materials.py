"""Build four-pair research contexts and a separate offline blind release.
No downloaded sources, training, server startup, or manufactured human points.
"""
import sys,os,json,csv,hashlib,shutil,re,zipfile,math
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import numpy as np,rasterio
from rasterio.windows import Window
from PIL import Image,ImageDraw
BASE=Path('/home/yl/matching/sar_matching');ROOT=Path(__file__).resolve().parents[1]
STAMP=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');CONTEXT_STAMP='20261013_2219'
G=BASE/'RealSAR_GeoAssist_20261012_1917';A=BASE/'RealSAR_AngleExpansion_20261012_1641'
sys.path.insert(0,str(G/'code'))
from records import Records
from geometry import sha,View,point_result
from gt_bridge import exposures_for
SOURCE_UI=BASE/'RealSAR_IndependentBenchmark_20261011_2015/annotation_adapter'
SELECTED=['ANGLE_e76cc299fa7b__native_000','ANGLE_e3bfffe1139e__native_000','ANGLE_75c261e08eae__native_000','ANGLE_5da40401a853__native_000']
# Independent side search boxes, inferred from visual inspection, NOT point pairs.
FOCUS={
 SELECTED[0]:([[7050,2120,320,320],[7170,2960,320,320]],'优先盲查','规则暗区的岸缘/设施边界转角，两侧轮廓可追踪；低矮地面属性仍待真人确认','邻近亮块可能是船舶或抬高设施，不能按最亮位置标点。'),
 SELECTED[1]:([[320,6050,400,450],[0,2840,450,420]],'条件性局部盲查','可追踪暗区边缘及下方设施带；右侧接近Native左边界，拐角可能截断','裁片边缘限制、亮边可能受散射影响；不能将外推边界当同点。'),
 SELECTED[2]:([[1000,3180,420,420],[590,4460,420,420]],'条件性局部盲查','规则暗区边缘转折及邻近地面设施交汇候选','模糊、强散射和约一天时差；先辨别地面轮廓，暂不支持点级精度。'),
 SELECTED[3]:([[4750,2570,560,450],[2250,2990,620,700]],'优先地面结构盲查','圆形储罐旁地面道路/围界转折候选；罐顶不列为锚点','罐体高度、叠掩、阴影、约37小时变化，原浏览中心可能落在罐区而非地面同点。'),
}
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def textfile(path,text):
 with path.open('x',encoding='utf-8') as f:f.write(text)
def jsonl(path,rows):
 with path.open('x',encoding='utf-8') as f:
  for d in rows:f.write(json.dumps(d,ensure_ascii=False,allow_nan=False)+'\n')
def table(path,heads,rows):
 with path.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=heads);w.writeheader();w.writerows(rows)

def build():
 tasks={d['sample_id']:d for d in map(json.loads,(G/'tasks.jsonl').read_text().splitlines())}
 queue={d['sample_id']:d for d in map(json.loads,(A/'review/candidate_regions.jsonl').read_text().splitlines())}
 store=Records(G/'assisted_triage',tasks)
 histories={sid:store.history(sid) for sid in tasks}
 refs=list(csv.DictReader((G/'exports/assisted_review_snapshot_20261013_201035/last_view_positions.csv').open(encoding='utf-8-sig')))
 protected={str(p):sha(p) for p in (G/'assisted_triage').rglob('*') if p.is_file()}
 for p in [G/'tasks.jsonl',G/'code/server.py',G/'code/app.js',A/'review/candidate_regions.jsonl',A/'source_isolation.json',A/'acquisition_inventory.jsonl']:
  protected[str(p)]=sha(p)
 audit=[];source_rows=[];regions=[];exposures=[];blind_tasks=[];image_aliases={};image_identities={};mapping=[];unique_acq=set()
 # Deliberately different code order from angle/site priorities. No cues in release.
 blind_codes={SELECTED[0]:'B003',SELECTED[1]:'B001',SELECTED[2]:'B004',SELECTED[3]:'B002'}
 for sid in SELECTED:
  r=queue[sid];doc=histories[sid][-1];views=[View(r,i) for i in [0,1]];images=[]
  bbox,decision,description,risk=FOCUS[sid]
  for side,v in enumerate(r['views']):
   meta=r['acquisition_metadata']['views'][side];native=Path(v['native_tif']);png=A/'review/regions'/sid/f'native_db{side}.png'
   unique_acq.add(v['acquisition_id'])
   for p in [native,png,Path(meta['metadata_path']),native.parent/'metadata.json']:
    protected[str(p)]=sha(p)
   assert sha(native)==v['native_sha256']
   assert sha(Path(meta['metadata_path']))==meta['metadata_sha256']
   stac=json.loads(Path(meta['metadata_path']).read_text());props=stac['properties']
   assert props['umbra:collect_id']==v['acquisition_id']
   assert props.get('start_datetime',props['datetime'])==meta['utc']
   assert abs(props['view:azimuth']-meta['azimuth_degrees'])<1e-9
   assetmeta=json.loads((native.parent/'metadata.json').read_text())
   assert assetmeta['original_window_xywh']==v['original_native_window_xywh']
   x0,y0,w,h=bbox[side]
   with rasterio.open(native) as d:
    assert (d.width,d.height)==(v['native_info']['width'],v['native_info']['height'])
    assert list(d.transform)[:6]==v['native_info']['transform']
    assert x0>=0 and y0>=0 and x0+w<=d.width and y0+h<=d.height
    values=d.read(1,window=Window(x0,y0,w,h));mask=d.read_masks(1,window=Window(x0,y0,w,h))>0
    valid_fraction=float((mask&np.isfinite(values)).mean())
   with Image.open(png) as im:
    assert im.size==(v['native_info']['width'],v['native_info']['height']) and im.mode=='L' and not im.info
    crop=im.crop((x0,y0,x0+w,y0+h));filename=f'ground_search_side{side}_{STAMP}.png';crop.save(ROOT/'assisted_research'/sid/filename)
   origin=v['original_native_window_xywh'][:2]
   projection=point_result(r,views,side,[x0+(w-1)/2,y0+(h-1)/2])
   region=dict(sample_id=sid,side=side,bbox_native_xywh=bbox[side],source_product_bbox_xywh=[origin[0]+x0,origin[1]+y0,w,h],
    coordinate_rule='crop pixel center (u,v) -> Native (x0+u,y0+v); original product adds stored integer parent offset once',
    source_native_tif=str(native),native_sha256=sha(native),display_png=str(png),display_sha256=sha(png),
    acquisition_id=v['acquisition_id'],crop_file=str(ROOT/'assisted_research'/sid/filename),crop_valid_fraction=valid_fraction,
    candidate_description=description,risk=risk,candidate_provenance='AI visual search after actual assisted review and unconfirmed navigation context',
    projected_search_center_hint=projection,confirmed_human_point=False,exact_correspondence=False,independent_GT=False)
   regions.append(region)
   source_rows.append(dict(sample_id=sid,side=side,site=r['site_name'],component=r['component'],acquisition_id=v['acquisition_id'],task_id=meta['task_id'],
    utc_start=meta['utc'],stac_midpoint_datetime=props['datetime'],azimuth_deg=meta['azimuth_degrees'],azimuth_pair_delta_deg=r['azimuth_delta_deg'],time_gap_seconds=r['days']*86400,
    native_tif=str(native),native_sha256=sha(native),native_width=v['native_info']['width'],native_height=v['native_info']['height'],original_window_xywh=v['original_native_window_xywh'],
    source_metadata_path=meta['metadata_path'],source_metadata_sha256=meta['metadata_sha256'],source_asset_url=meta['asset_url'],source_metadata_url=meta['metadata_url'],
    stac_id=stac['id'],product='GEC',local_header_verified=True,prior_inventory_raster_status=meta.get('raster_status'),
    licensed_source='Umbra Open Data CC-BY-4.0',split='diagnostic_development_seen'))
   display_hash=sha(png)
   if display_hash not in image_aliases:
    alias='I'+str(len(image_aliases)+1).zfill(3);name=alias+'_'+STAMP+'.png'
    # Distribution copy, not a new acquisition. Never share a writable inode with old assets.
    shutil.copyfile(png,ROOT/'blind_release'/name)
    (ROOT/'blind_release'/name).chmod(0o444)
    image_aliases[display_hash]=name
    image_identities[name]=dict(sha256=display_hash,width=v['native_info']['width'],height=v['native_info']['height'],pixel_convention='zero-based Native pixel centers')
   images.append(image_aliases[display_hash])
  sidcode=blind_codes[sid]
  blind_tasks.append(dict(sample_id=sidcode,width0=r['views'][0]['native_info']['width'],height0=r['views'][0]['native_info']['height'],
                          width1=r['views'][1]['native_info']['width'],height1=r['views'][1]['native_info']['height'],image0=images[0],image1=images[1],
                          batch_id='blind_native',batch_label='原生双图 · 独立输入',site_name=''))
  mapping.append(dict(blind_code=sidcode,sample_id=sid,pair_id=r['pair_id'],component=r['component'],images=images,stage='prepared_only',GT=False))
  for e in exposures_for([r]):exposures.append(e)
  # Same-component exposure retained in addition to bridge's shared-acquisition rule.
  original_events=[json.loads(z) for z in (G/'assisted_triage/exposure_events.jsonl').read_text().splitlines()]
  samecomponent=[e for e in original_events if queue.get(e.get('sample_id'),{}).get('component')==r['component']]
  audit.append(dict(sample_id=sid,blind_code=sidcode,site=r['site_name'],azimuth_delta_deg=r['azimuth_delta_deg'],component=r['component'],
                   latest_assisted_record=doc,history_versions=len(histories[sid]),ai_search_decision=decision,visual_search_description=description,risk=risk,
                   direct_or_shared_acquisition_exposed_actors=sorted({e['actor'] for e in exposures_for([r])}),
                   component_exposed_actors=sorted({e['actor'] for e in samecomponent}),
                   recommendation_field=doc['recommend_independent_annotation'],no_gt_batch_launched=True,physical_points=[]))
 jsonl(ROOT/'custodian'/f'source_audit_{STAMP}.jsonl',source_rows)
 jsonl(ROOT/'assisted_research'/f'candidate_ground_regions_{STAMP}.jsonl',regions)
 jsonl(ROOT/'custodian'/f'exposure_propagation_{STAMP}.jsonl',exposures)
 write(ROOT/'custodian'/f'blind_to_source_map_{STAMP}.json',mapping)
 write(ROOT/'custodian'/f'four_pair_preparation_{STAMP}.json',audit)
 write(ROOT/'custodian'/f'quality_policy_{STAMP}.json',json.loads((BASE/'RealSAR_AngleExpansion_20261012_1641/gt_policy_snapshot.json').read_text()))
 # Preserve actual prior evidence with timestamped names, without changing semantics.
 for sid in SELECTED:
  for number,doc in enumerate(histories[sid],1):
   src=G/'assisted_triage'/sid/f'v{number:06d}.json';dst=ROOT/'custodian'/f'{sid}_v{number:06d}_{STAMP}.json'
   shutil.copyfile(src,dst)
 shutil.copyfile(G/'assisted_triage/exposure_events.jsonl',ROOT/'custodian'/f'exposure_events_snapshot_{STAMP}.jsonl')
 table(ROOT/'custodian'/f'old_navigation_centers_{STAMP}.csv',list(refs[0]),[d for d in refs if d['sample_id'] in SELECTED])
 statuses=[dict(sample_id=d['sample_id'],blind_code=d['blind_code'],component=d['component'],assisted_verdict=d['latest_assisted_record']['verdict'],
                assisted_triage_saved=True,assisted_correspondence_points=0,independent_annotation_status='not_annotated',
                independent_annotator_identity='',independent_reviewer_identity='',independent_point_count=0,independent_review_status='not_started',
                qualified_GT_point_count=0,GT_stage='prepared_unannotated',evaluation_scope='none_pending_humans',next_step=d['ai_search_decision']) for d in audit]
 table(ROOT/f'GT_annotation_status_{STAMP}.csv',list(statuses[0]),statuses)
 write(ROOT/'custodian'/f'protected_inputs_before_{STAMP}.json',protected)
 create_blind(blind_tasks,image_identities)
 # Auxiliary material index has coordinates and reasons. NEVER distribute to blind people.
 body=['<!doctype html><meta charset="utf-8"><title>辅助研究材料，非GT</title><style>body{font:16px system-ui;max-width:1300px;margin:auto;line-height:1.6}img{max-width:100%}pre{white-space:pre-wrap}section{border-top:1px solid #bbb;margin-top:24px}</style><h1>首批4组辅助搜索材料 · 不是独立GT</h1><p>本文件只供已暴露研究人员。不得交给拟承担独立标注/复核的人员。</p>']
 for d in audit:
  sid=d['sample_id'];folder=ROOT/'assisted_research'/sid
  note=f"{sid} | {d['azimuth_delta_deg']:.3f}° | {d['ai_search_decision']}\n用户{d['latest_assisted_record']['verdict']}：{d['latest_assisted_record']['reason']}\nAI辅助观察：{d['visual_search_description']}\n风险：{d['risk']}"
  textfile(folder/f'candidate_note_{STAMP}.md','# 辅助候选搜索，非GT\n\n'+note+'\n\n日志中心未认证为人工点，不能绑定保存版本。精确窗口和来源见candidate_ground_regions清单；裁片坐标加窗口整数偏移返回Native，无重新采样。\n')
  body.append('<section><h2>'+sid+'</h2><pre>'+__import__('html').escape(note)+'</pre>')
  body.append(f'<img src="{sid}/comparison_2048_{CONTEXT_STAMP}.png"><p>2048邻域预览：裁片靠边时按实际尺寸截断，拼图仅作显示。</p>')
  body.append(f'<a href="{sid}/candidate_note_{STAMP}.md">结构说明</a> · <a href="{sid}/candidate_contexts_{CONTEXT_STAMP}.json">旧浏览中心及原生窗口</a> · ')
  for side in [0,1]:body.append(f'<a href="{sid}/ground_search_side{side}_{STAMP}.png">侧{side}地面候选原生局部</a> · ')
  body.append('</section>')
 textfile(ROOT/'assisted_research'/f'assisted_materials_{STAMP}.html',''.join(body))
 # No image duplication beyond distributing the SAME existing display material.
 archive=ROOT/f'blind_native_materials_{STAMP}.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_STORED) as z:
  for p in sorted((ROOT/'blind_release').iterdir()):z.write(p,'blind_native_materials/'+p.name)
 assert len(unique_acq)==7 and len(image_aliases)==7
 summary=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),tasks_checked=18,assisted_reviewed=len([h for h in histories.values() if h]),history_versions=sum(map(len,histories.values())),
              selected_pairs=4,unique_acquisitions=7,source_components=3,blind_unique_images=7,blind_package=str(archive),
              independent_GT=0,human_annotations_created=0,prior_clicks_in_selected=sum(len(h[-1].get('manual_clicks',[])) for sid,h in histories.items() if sid in SELECTED),
              new_downloads=0,GPU=False,ports_started=[],split='diagnostic_development_seen',blind_does_not_make_unseen_final_test=True)
 write(ROOT/f'preparation_summary_{STAMP}.json',summary)
 print(json.dumps(summary,ensure_ascii=False,indent=2))
 return summary

def create_blind(tasks,identities):
 b=ROOT/'blind_release';tasks.sort(key=lambda t:t['sample_id'])
 html=(SOURCE_UI/'index.html').read_text();app=(SOURCE_UI/'app.js').read_text()
 html=html.replace('aspect-ratio:1;touch-action','aspect-ratio:auto;touch-action').replace('type="checkbox" checked>联动','type="checkbox">联动')
 html=html.replace('<title>SAR 对应点标注</title>','<title>Native盲标离线采集</title>')
 html=html.replace('逐点正式确认','逐点填写结构依据').replace('确认本点并保存 / 下一个','导出本点观察 / 下一个').replace('确认并保存候选 · Enter','导出人工观察 · Enter')
 html=html.replace('采集信息与质量详情','原生像素尺寸与未核验状态')
 html=html.replace('Umbra / Copyright Capella Space CC BY4.0. Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026.','Umbra Open Data — CC-BY-4.0。')
 html=html.replace('<details><summary>辅助层','<details hidden><summary>辅助层')
 html=html.replace('id="reviewLink"','id="reviewLink" hidden').replace('class="review-only" href="/"','class="review-only" hidden href="#"')
 html=html.replace('仅输入自己的真实身份；第二人不得换ID模拟自审。','仅输入自己的真实身份及既往别名；不能通过换ID清除暴露。该包只采集原始观察，不自动产生GT。')
 extra='''<label>曾看地理投影/地图提示？<select id="priorGeo"><option value="unknown">未填写</option><option value="no">否</option><option value="yes">是</option></select></label><label>曾看算法匹配？<select id="priorAlgorithm"><option value="unknown">未填写</option><option value="no">否</option><option value="yes">是</option></select></label><label>曾看候选区域/AI同点提示？<select id="priorHints"><option value="unknown">未填写</option><option value="no">否</option><option value="yes">是</option></select></label><label>曾看他人坐标/标注？<select id="priorOtherPoints"><option value="unknown">未填写</option><option value="no">否</option><option value="yes">是</option></select></label><label>此前使用的身份ID/别名<input id="aliases" type="text"></label><label><input id="actualHuman" type="checkbox">我是实际操作者，声明属实</label>'''
 html=html.replace('<button id="begin"',extra+'<button id="begin"')
 html=html.replace('<p class="hint">原坐标和原定位误差不传入此页面。','<p class="hint">原坐标和原定位误差不传入此页面。')
 html=html.replace('<footer>','<footer>离线采集：点击保存只生成下载，请确认JSON已落盘；刷新清空内存。第二人单独使用干净页面，不接收第一人坐标。不得将本人声明直接判为独立合格GT。<br>')
 # Reuse coordinates, canvas controls, loupe, structure editor; replace only persistence/image loading.
 start=app.index('async function api(');end=app.index('\nfunction actor()',start)
 app=app[:start]+"async function api(path,data){return OfflineAPI(path,data);}"+app[end:]
 app=app.replace("const reviewing=new URLSearchParams(location.search).get('review')==='1';","const reviewing=false; // each real person supplies their own independent inputs")
 app=app.replace("im.src='/api/image?sample_id='+encodeURIComponent(next.sample_id)+'&side='+i;","im.src=next['image'+i];")
 start=app.index("  $('metadata').textContent=");end=app.index('\n  clearPoint();',start)
 app=app[:start]+"  $('metadata').textContent=`${task.sample_id} | Native左${task.width0}×${task.height0}，右${task.width1}×${task.height1}；像素中心零基。只采集真人观察，非GT。`;"+app[end:]
 app=app.replace("$('reviewLink').href='/?review=1&task='+encodeURIComponent(task.sample_id);","$('reviewLink').removeAttribute('href');")
 app=app.replace("for(const id of ['geo','pseudo'])$(id).disabled=disabled;","for(const id of ['geo','pseudo'])$(id).disabled=true;")
 app=app.replace("const name=$('actor').value.trim();if(!name)","const name=$('actor').value.trim();declarations();if(!name)")
 app=app.replace("先选一个已保存点","先选一个已导出观察").replace('已保存','已生成下载').replace('确认已保存','观察已生成下载')
 # Display-only metadata (no pair/acquisition/site/angle/CRS/projected point).
 script='window.BLIND_TASKS='+json.dumps(tasks,ensure_ascii=False)+';window.BLIND_PACKAGE_ID="physical_prep_'+STAMP+'";window.BLIND_IMAGE_IDENTITIES='+json.dumps(identities)+';'
 jsfile=f'blind_app_{STAMP}.js';coords=f'coordinates_{STAMP}.js';adapter=f'offline_capture_{STAMP}.js'
 html=html.replace('<script>window.SAR_TOKEN="__TOKEN__";</script><script src="/coordinates.js"></script><script src="/app.js"></script>',
                   '<script>'+script+'</script><script src="'+coords+'"></script><script src="'+adapter+'"></script><script src="'+jsfile+'"></script>')
 textfile(b/f'blind_annotation_{STAMP}.html',html)
 textfile(b/jsfile,app);shutil.copyfile(SOURCE_UI/'coordinates.js',b/coords);shutil.copyfile(ROOT/'code/offline_capture.js',b/adapter)
 table(b/f'physical_observation_template_{STAMP}.csv',
       ['blind_code','person_id','person_aliases','annotated_at','prior_geographic_hint','prior_algorithm_matches','prior_candidate_or_AI_hints','prior_other_person_points','point_id','x_left_native','y_left_native','x_right_native','y_right_native','physical_structure_description','height_shadow_layover_risk','static_confirmed','visible_both_confirmed','sigma_left_px','sigma_right_px','confidence','role_control_or_check','point_or_region_level','conclusion','reason'],[])
 textfile(b/f'PERSONNEL_INSTRUCTIONS_{STAMP}.md',f'''# Native独立输入材料（尚无GT）\n\n仅给此前未看过这些样本的地理/算法/候选提示和其他人坐标的真实人员。材料管理员须先审核真实身份及全部既往别名；本人声明不能代替历史暴露核查。已暴露者可以做辅助探索，必须如实填“是”，不能获得盲标资格。\n\n1. 解压后在浏览器打开blind_annotation_{STAMP}.html。不需要服务器或端口。只显示原生SAR，不含投影、方位、GeoTransform或候选位置。\n2. 本人填写真实身份、既往别名和四项暴露声明。独立第二人使用自己的干净页面/浏览器会话，不接收第一人的坐标或画点截图；不要在同一会话换名字模拟第二人。\n3. 滚轮放大；Shift或中键拖动；左右分别寻找同一固定地面结构。鼠标位置/点击均用原图零基像素中心x=列、y=行。显示缩放不改变坐标；不得估算屏幕截图坐标。\n4. 左右各点击后进入“高级模式”或“逐点填写结构依据”，填写明确物理结构、静态/双图可辨认判断、各侧定位不确定度、风险备注、check或control职责。罐顶、亮船、起重机、集装箱、阴影和疑似抬高散射体不作为地面锚点。\n5. 点击“导出人工观察”：会下载新版本JSON，文件名含操作者、版本和北京时间分钟。确认下载文件实际落盘；页面内存不是项目数据库，刷新会清空内存。保留所有文件，不覆盖以前版本。\n6. 不能确认时，可用“结束任务”填写不可评价/局部区域级理由，不必凑点。区域级线索可在空CSV填写，精确坐标不可确认时保持空白，不能伪填中心当对应点。\n7. 两名实际人员分别将自己下载的全部JSON（或填写CSV）交给材料管理员。管理员核对独立性、原图来源、有效掩膜、结构风险和独立重点击差异，之后才导入已有GT流程。此离线包始终显示合格数0，不自动授予GT。\n\n若两人不能定位到同一物理结构，保留疑义/拒绝，不按地理预测、模型内点或点数要求修正。查找困难本身不等于没有重叠。材料提供者：Umbra Open Data，CC-BY-4.0；不得为补来源说明向标注者发送地理或采集提示。\n''')

if __name__=='__main__':build()
