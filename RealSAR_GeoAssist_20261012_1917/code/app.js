'use strict';
const $=id=>document.getElementById(id);
let tasks=[],current=null,ready=false,busy=false,dirty=false,revision=0,last=null,seq=0;
let overviewImages=[null,null],cropImages=[null,null],crops=[null,null],positions=[null,null],focusPositions=[null,null];
let regions=[],manualClicks=[],clickBaselineCount=0,lockedActor='',historyRecords=[],historySelected=null;
let navigationReferences=[],recoveredActive=false;
const YELLOW='#ffd36b',PINK='#ff62d2',RECOVERED='#66cfff';
async function api(path,data){
 const r=await fetch(path,{method:data?'POST':'GET',headers:{'X-SAR-Token':TOKEN,...(data?{'Content-Type':'application/json'}:{})},...(data?{body:JSON.stringify(data)}:{})});
 const v=await r.json();if(!r.ok)throw Error(v.error||`HTTP ${r.status}`);return v;
}
function message(s){$('status').textContent=s;}
function handle(fn){return async(...args)=>{try{await fn(...args);}catch(e){message(e.message);}};}
function setReady(v){ready=v;$('save').disabled=!v||READONLY||busy;$('addRegion').disabled=!v||!crops.some(c=>c&&!c.outside)||busy;}
function image(src){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(Error('显示图加载失败'));im.src=src;});}
async function fetchedImage(path){const r=await fetch(path,{headers:{'X-SAR-Token':TOKEN}});if(!r.ok)throw Error((await r.json()).error);const url=URL.createObjectURL(await r.blob());try{return await image(url);}finally{URL.revokeObjectURL(url);}}
function mark(ctx,x,y,color){ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.moveTo(x-12,y);ctx.lineTo(x+12,y);ctx.moveTo(x,y-12);ctx.lineTo(x,y+12);ctx.stroke();}
function drawClicks(ctx,side,clicks,label,sx,sy,ox=0,oy=0){
 ctx.save();ctx.fillStyle=YELLOW;ctx.font='12px system-ui';
 clicks.forEach((p,i)=>{if(p.side!==side)return;const [x,y]=p.native_xy,dx=(x+.5-ox)*sx,dy=(y+.5-oy)*sy;mark(ctx,dx,dy,YELLOW);ctx.fillText(`${label} 点击${i+1} · 非GT`,dx+10,dy-10);});ctx.restore();
}
function drawHistory(ctx,side,sx,sy,ox=0,oy=0){
 if(recoveredActive){ctx.save();ctx.fillStyle=RECOVERED;ctx.font='12px system-ui';navigationReferences.filter(p=>p.side===side).forEach(p=>{const x=(p.native_xy[0]+.5-ox)*sx,y=(p.native_xy[1]+.5-oy)*sy;mark(ctx,x,y,RECOVERED);ctx.fillText('旧浏览参考 · 日志恢复 · 非GT',x+10,y-10);});ctx.restore();}
 drawClicks(ctx,side,manualClicks.slice(clickBaselineCount),'待保存',sx,sy,ox,oy);
 if(!historySelected)return;
 drawClicks(ctx,side,historySelected.manual_clicks||[],`v${historySelected.revision}`,sx,sy,ox,oy);
 ctx.save();ctx.strokeStyle=YELLOW;ctx.fillStyle=YELLOW;ctx.lineWidth=2;ctx.font='12px system-ui';
 (historySelected.approx_regions||[]).forEach((r,i)=>{if(r.side!==side)return;const [x,y,w,h]=r.bbox_xywh;
  // bbox boundaries are pixel edges; unlike pixel centres they need no +0.5.
  ctx.strokeRect((x-ox)*sx,(y-oy)*sy,w*sx,h*sy);
  ctx.fillText(`v${historySelected.revision} 区域${i+1} · 辅助筛选，不是GT`,Math.max(2,(x-ox)*sx),Math.max(14,(y-oy)*sy+14));
 });ctx.restore();
}
function draw(side){
 if(!current)return;const v=current.views[side],a=$('overview'+side),ctx=a.getContext('2d'),im=overviewImages[side];ctx.clearRect(0,0,a.width,a.height);ctx.imageSmoothingEnabled=false;
 if(im){ctx.filter=`brightness(${$('brightness').value}) contrast(${$('contrast').value})`;ctx.drawImage(im,0,0,a.width,a.height);ctx.filter='none';}
 const sx=a.width/v.width,sy=a.height/v.height,poly=current.coverage.native_polygons[side];
 if($('coverage').checked&&poly.length){ctx.strokeStyle='#3aff93';ctx.lineWidth=2;ctx.beginPath();poly.forEach((p,i)=>ctx[i?'lineTo':'moveTo']((p[0]+.5)*sx,(p[1]+.5)*sy));ctx.closePath();ctx.stroke();}
 drawHistory(ctx,side,sx,sy);
 if(positions[side]&&last)mark(ctx,(positions[side][0]+.5)*sx,(positions[side][1]+.5)*sy,last.source_side===side&&last.input_event?YELLOW:PINK);
 const d=$('detail'+side),dc=d.getContext('2d'),crop=crops[side];dc.clearRect(0,0,d.width,d.height);dc.imageSmoothingEnabled=false;
 if(crop&&!crop.outside&&cropImages[side]){const [x,y,w,h]=crop.bbox_xywh;dc.filter=`brightness(${$('brightness').value}) contrast(${$('contrast').value})`;dc.drawImage(cropImages[side],0,0,d.width,d.height);dc.filter='none';drawHistory(dc,side,d.width/w,d.height/h,x,y);if(positions[side]&&last)mark(dc,(positions[side][0]+.5-x)*d.width/w,(positions[side][1]+.5-y)*d.height/h,last.source_side===side&&last.input_event?YELLOW:PINK);}
 if(crop&&crop.outside){dc.fillStyle=YELLOW;dc.font='18px system-ui';dc.fillText('地理预测在此Native图之外',20,40);}
}
function regionList(){$('regionList').textContent=regions.map((r,i)=>`${i+1}. ${r.side===0?'左':'右'} Native粗略窗口 [${r.bbox_xywh.map(x=>Math.round(x)).join(', ')}]（辅助筛选，不是对应点）`).join('\n');}
function clickList(){$('clickList').textContent=manualClicks.length?`随下一次保存写入的人工观察点击：${manualClicks.length} 个（不是GT，不自动配对）\n`+manualClicks.map((p,i)=>`${i+1}. ${p.side===0?'左':'右'} Native (${p.native_xy.map(z=>z.toFixed(3)).join(', ')})`).join('\n'):'尚无直接点击坐标；直接点击图像后会随辅助筛选一起保存。';}
function applyForm(d){document.querySelectorAll('[name=verdict]').forEach(el=>el.checked=el.value===d.verdict);$('location').value=d.location_description||'';$('reason').value=d.reason||'';$('recommend').checked=!!d.recommend_independent_annotation;regions=(d.approx_regions||[]).map(r=>({side:r.side,bbox_xywh:r.bbox_xywh.slice()}));manualClicks=(d.manual_clicks||[]).map(p=>({side:p.side,native_xy:p.native_xy.slice(),input_event:p.input_event}));clickBaselineCount=manualClicks.length;regionList();clickList();dirty=false;}
function renderHistory(d){
 $('historyStatus').textContent=d.record_count?`已从磁盘读取 ${d.record_count} 条辅助筛选版本，最新 v${d.latest_revision}；点击版本查看历史窗口。`:
  d.exposure_only?'仅登记暴露，尚未成功保存辅助筛选。点击图像或加入窗口还不足以保存；请填写理由并点击保存。':'磁盘没有辅助筛选版本；尚未完成保存。';
 const list=$('historyList');list.replaceChildren();
 for(const record of [...historyRecords].reverse()){
  const item=document.createElement('div');item.style.cssText='border:1px solid #6982a0;padding:8px;margin:8px 0;white-space:pre-wrap';
  const button=document.createElement('button');button.dataset.revision=record.revision;
  button.textContent=`查看 v${record.revision} · ${record.verdict} · ${record.actor} · ${record.at}`;
  button.onclick=handle(()=>{if(!busy)return locateHistory(record.revision);});item.append(button);
  const text=document.createElement('p');text.textContent=`判断理由与风险：${record.reason}\n位置描述：${record.location_description||'使用保存的观察点击或窗口'}\n直接观察点击：${(record.manual_clicks||[]).length} 个；Native粗略窗口：${(record.approx_regions||[]).length} 个 · 辅助筛选，不是GT`;item.append(text);
  (record.approx_regions||[]).forEach((r,i)=>{const b=document.createElement('button');b.textContent=`定位${r.side===0?'左':'右'}区域${i+1} [${r.bbox_xywh.join(', ')}]`;b.dataset.region=i;b.dataset.version=record.revision;b.onclick=handle(()=>{if(!busy)return locateHistory(record.revision,i);});item.append(b);});
  (record.manual_clicks||[]).forEach((p,i)=>{const b=document.createElement('button');b.textContent=`定位${p.side===0?'左':'右'}点击${i+1} (${p.native_xy.map(z=>z.toFixed(3)).join(', ')})`;b.dataset.click=i;b.dataset.version=record.revision;b.onclick=handle(()=>{if(!busy)return locateHistory(record.revision,null,true,i);});item.append(b);});
  list.append(item);
 }
}
async function locateHistory(version,regionIndex=null,notify=true,clickIndex=null){
 const record=historyRecords.find(d=>d.revision===version);if(!record||!current)return;
 const sid=current.sample_id,ticket=++seq;historySelected=record;recoveredActive=false;last=null;positions=[null,null];focusPositions=[null,null];crops=[null,null];cropImages=[null,null];
 const saved=record.approx_regions||[],clicks=record.manual_clicks||[];let chosen=[0,1].map(side=>clicks.filter(p=>p.side===side).at(-1)||saved.find(r=>r.side===side)||null);
 if(regionIndex!==null&&saved[regionIndex])chosen[saved[regionIndex].side]=saved[regionIndex];
 if(clickIndex!==null&&clicks[clickIndex])chosen[clicks[clickIndex].side]=clicks[clickIndex];
 const missing=chosen.map((r,i)=>r?null:(i===0?'左':'右')).filter(Boolean);
 $('historyFocus').textContent=saved.length||clicks.length?`显示 v${version} 黄色历史观察位置；${missing.length?missing.join('、')+'图没有已保存观察坐标，不把预测位置补成人工点击。':'两侧观察位置分别定位，不表示精确对应。'} 查看历史不会改写表单或覆盖旧版本。`:
  `v${version} 只有位置文字，没有Native坐标或窗口，无法自动定位：${record.location_description||'未提供'}。`;
 if(!saved.length&&!clicks.length&&navigationReferences.length){
  $('historyFocus').textContent=`v${version} 当时没有写入坐标；已显示同一样本末次浏览参考（青色）。日志没有版本关联，不能称为此版本的人工点击。`;
  await locateRecovered(notify);return;
 }
 [0,1].forEach(side=>{$('coord'+side).textContent=`${current.views[side].width} × ${current.views[side].height} Native像素 · ${chosen[side]?'正在读取历史窗口':'未保存此侧位置'}`;draw(side);});
 await Promise.all(chosen.map(async(r,side)=>{if(!r)return;const bbox=r.bbox_xywh;
  const query=r.native_xy?`x=${r.native_xy[0]}&y=${r.native_xy[1]}&span=${$('span').value}`:`bbox=${encodeURIComponent(JSON.stringify(bbox))}`;
  const crop=await api(`/api/crop?sid=${encodeURIComponent(sid)}&side=${side}&${query}`),im=await image(crop.png);
  if(ticket!==seq||!current||current.sample_id!==sid)return;
  focusPositions[side]=crop.requested_xy;crops[side]=crop;cropImages[side]=im;
  $('coord'+side).textContent=`Native历史${r.native_xy?'点击':'窗口'} v${version} [${(r.native_xy||bbox).join(', ')}] · 辅助筛选，不是GT`;draw(side);
 }));
 if(ticket!==seq)return;setReady(ready);if(notify)message(`已定位历史 v${version}，黄色表示已保存的观察位置；不是GT。`);
}
async function locateRecovered(notify=true){
 if(!current||!navigationReferences.length)return;
 const sid=current.sample_id,ticket=++seq;recoveredActive=true;last=null;positions=[null,null];focusPositions=[null,null];crops=[null,null];cropImages=[null,null];
 const spans=[...new Set(navigationReferences.map(p=>String(p.span_native)))];if(spans.length===1)$('span').value=spans[0];
 [0,1].forEach(draw);
 $('recoveredStatus').textContent='正在恢复旧浏览视图…';
 await Promise.all(navigationReferences.map(async p=>{const side=p.side,xy=p.native_xy;
  const crop=await api(`/api/crop?sid=${encodeURIComponent(sid)}&side=${side}&x=${xy[0]}&y=${xy[1]}&span=${p.span_native}`),im=await image(crop.png);
  if(ticket!==seq||!current||current.sample_id!==sid)return;
  focusPositions[side]=xy.slice();crops[side]=crop;cropImages[side]=im;
  $('coord'+side).textContent=`旧浏览参考 Native (${xy.map(x=>x.toFixed(3)).join(', ')}) · 日志 ${p.log_timestamp} · 未确认、非GT`;draw(side);
 }));
 if(ticket!==seq)return;setReady(ready);
 $('recoveredStatus').textContent='已恢复原生放大视图：青色十字是日志里的末次浏览中心，可能包含地理预测；没有写入人工点击或GT。';
 if(notify)message('已显示旧浏览位置。青色为日志恢复参考；黄色只用于实际保存的人工观察点击，粉色为当前地理预测。');
}
async function loadRecord({locate=true,form=true}={}){
 const sid=current.sample_id,d=await api('/api/history?sid='+encodeURIComponent(sid));if(!current||current.sample_id!==sid)return;
 historyRecords=d.records;revision=d.latest_revision;navigationReferences=d.navigation_references||[];renderHistory(d);
 $('recoveredPanel').hidden=!navigationReferences.length&&!d.navigation_reference_error;$('locateRecovered').disabled=!navigationReferences.length;
 $('recoveredStatus').textContent=d.navigation_reference_error?'旧浏览位置读取失败：'+d.navigation_reference_error:
  navigationReferences.map(p=>`${p.side===0?'左':'右'} Native (${p.native_xy.map(x=>x.toFixed(3)).join(', ')}) · 日志 ${p.log_timestamp}`).join('\n');
 if(form)applyForm(historyRecords.at(-1)||{});
 if(locate&&historyRecords.length)await locateHistory(revision,null,false);
 else if(!historyRecords.length){historySelected=null;$('historyFocus').textContent='没有已保存坐标，无法恢复过去的点击位置。';[0,1].forEach(draw);}
 return d;
}
async function load(sid){
 if(busy)return;if(dirty&&!confirm('当前辅助判断未保存，放弃并切换？')){$('task').value=current.sample_id;return;}
 const actor=$('actor').value.trim();if(!READONLY&&!actor)throw Error('请填写本人的真实身份');setReady(false);busy=true;++seq;
 current=null;historyRecords=[];historySelected=null;last=null;positions=[null,null];focusPositions=[null,null];crops=[null,null];cropImages=[null,null];overviewImages=[null,null];regions=[];manualClicks=[];clickBaselineCount=0;navigationReferences=[];recoveredActive=false;$('recoveredPanel').hidden=true;
 $('historyList').replaceChildren();$('historyStatus').textContent='正在读取磁盘历史…';$('saveStatus').textContent='尚未保存本次辅助筛选。';
 try{
  if(!READONLY)await api('/api/exposure',{sample_id:sid,actor});lockedActor=actor;
  current=await api('/api/task?sid='+encodeURIComponent(sid));
  await Promise.all([0,1].map(async side=>{overviewImages[side]=await fetchedImage('/api/overview?sid='+encodeURIComponent(sid)+'&side='+side);const v=current.views[side],c=$('overview'+side);c.width=overviewImages[side].width;c.height=overviewImages[side].height;$('coord'+side).textContent=`${v.width} × ${v.height} Native像素 · ${v.crs}`;draw(side);}));
  const d=await loadRecord();$('meta').textContent=JSON.stringify({pair_id:current.pair_id,site:current.site_name,views:current.views.map(v=>({acquisition:v.acquisition_id,dimensions:[v.width,v.height],CRS:v.crs,transform_method:v.method})),coverage_m2:current.coverage.area_m2,note:current.coverage.note},null,2);
  message(recoveredActive?'已恢复旧浏览视图：青色标记来自日志，供继续判读，不是原版本里保存的人工点击或GT。':d.record_count?'已读取最新历史；有保存的点击或窗口时自动定位，旧文字记录不会补造坐标。':'可点击任一侧图像。仅登记暴露尚未保存筛选；理由必填；直接点击位置、位置文字或查看区域至少一项。');
 }catch(e){current=null;$('historyStatus').textContent='历史读取失败：'+e.message;throw e;}
 finally{busy=false;setReady(!!current);}
}
async function refreshCrops(ticket=seq){
 const sid=current.sample_id,span=$('span').value;
 await Promise.all([0,1].map(async side=>{const p=focusPositions[side];if(!p)return;const crop=await api(`/api/crop?sid=${encodeURIComponent(sid)}&side=${side}&x=${p[0]}&y=${p[1]}&span=${span}`);const im=crop.outside?null:await image(crop.png);if(ticket!==seq||!current||current.sample_id!==sid)return;crops[side]=crop;cropImages[side]=im;draw(side);}));setReady(ready);
}
async function mapPoint(side,xy,inputEvent=null){
 if(!ready||busy)return;const ticket=++seq,sid=current.sample_id;
 recoveredActive=false;
 if(inputEvent&&!READONLY){
  if(manualClicks.length>=500)throw Error('待保存点击达到500个，请先保存或清理点击');
  manualClicks.push({side,native_xy:xy.slice(),input_event:inputEvent});dirty=true;clickList();draw(side);
  $('saveStatus').textContent=`已记录直接观察点击，共 ${manualClicks.length} 个；点击“保存辅助筛选”后写入磁盘，不需要再加入窗口。自动预测点不会作为人工点击保存。`;
 }
 const result=await api(`/api/map?sid=${encodeURIComponent(sid)}&side=${side}&x=${xy[0]}&y=${xy[1]}`);if(ticket!==seq||!current||current.sample_id!==sid)return;
 result.request_sequence=ticket;result.input_event=inputEvent;last=result;positions[side]=result.source_xy;positions[1-side]=result.target_xy;focusPositions=positions.map(p=>p.slice());
 await refreshCrops(ticket);if(ticket!==seq)return;
 const direction=side===0?'左→右':'右→左';message(`${direction} Native (${result.source_xy.map(x=>x.toFixed(2)).join(', ')}) → (${result.target_xy.map(x=>x.toFixed(2)).join(', ')})\n地理参考 lon/lat ${result.lon_lat.map(x=>x.toFixed(7)).join(', ')}；数学往返 ${result.roundtrip_px.toExponential(2)}px（不是物理误差）；${result.target_in_extent?(result.target_valid?'目标有效':'目标掩膜/有限值需检查'):'预测在目标Native范围外'}；映射 ${result.mapping_ms.toFixed(1)}ms`);
 [0,1].forEach(i=>{$('coord'+i).textContent=`${current.views[i].width} × ${current.views[i].height} · Native参考 (${positions[i].map(x=>x.toFixed(2)).join(', ')}) · 辅助筛选，不是GT`;draw(i);});setReady(true);
}
for(let side=0;side<2;side++){
 $('overview'+side).onclick=handle(async e=>{if(!ready)return;const box=e.currentTarget.getBoundingClientRect(),v=current.views[side];await mapPoint(side,[Math.min(v.width-1,Math.max(0,(e.clientX-box.left)/box.width*v.width-.5)),Math.min(v.height-1,Math.max(0,(e.clientY-box.top)/box.height*v.height-.5))],'overview_click');});
 $('detail'+side).onclick=handle(async e=>{if(!ready||!crops[side]||crops[side].outside)return;const box=e.currentTarget.getBoundingClientRect(),[x,y,w,h]=crops[side].bbox_xywh,v=current.views[side];await mapPoint(side,[Math.max(0,Math.min(v.width-1,x+(e.clientX-box.left)/box.width*w-.5)),Math.max(0,Math.min(v.height-1,y+(e.clientY-box.top)/box.height*h-.5))],'detail_click');});
}
$('enter').onclick=handle(()=>load($('task').value));$('task').onchange=handle(()=>{if(ready)return load($('task').value);});
$('locateRecovered').onclick=handle(()=>{if(ready&&!busy)return locateRecovered();});
$('span').onchange=handle(async()=>{if(focusPositions.some(Boolean))await refreshCrops(++seq);});
for(const id of ['brightness','contrast','coverage'])$(id).oninput=()=>{draw(0);draw(1);};
document.querySelectorAll('[data-side]').forEach(b=>b.onclick=handle(()=>{const side=Number(b.dataset.side);if(!ready||busy||!focusPositions[side])return;const v=current.views[side],step=Number($('span').value)/4,p=focusPositions[side];return mapPoint(side,[Math.max(0,Math.min(v.width-1,p[0]+Number(b.dataset.dx)*step)),Math.max(0,Math.min(v.height-1,p[1]+Number(b.dataset.dy)*step))]);}));
$('addRegion').onclick=()=>{for(let side=0;side<2;side++)if(crops[side]&&!crops[side].outside)regions.push({side,bbox_xywh:crops[side].bbox_xywh.slice()});regionList();dirty=true;$('saveStatus').textContent='查看窗口已加入待保存区域，尚未写入磁盘；请填写理由后点击保存。';};
$('clearRegions').onclick=()=>{regions=[];regionList();dirty=true;$('saveStatus').textContent='待保存区域已清空，磁盘历史未改动。';};
$('clearClicks').onclick=()=>{if(last)last.input_event=null;manualClicks=[];clickBaselineCount=0;clickList();dirty=true;[0,1].forEach(draw);$('saveStatus').textContent='待保存点击已清空，磁盘历史未改动。';};
$('reload').onclick=handle(async()=>{if(!ready||busy)return;if(dirty&&!confirm('放弃未保存的修改并重新读取磁盘？'))return;busy=true;setReady(true);try{await loadRecord();message('已重新读取磁盘历史；旧版本未改动。');}finally{busy=false;setReady(true);}});
document.querySelector('fieldset:last-of-type').oninput=()=>{dirty=true;$('saveStatus').textContent='表单已修改，尚未保存到磁盘。';};
$('save').onclick=async()=>{
 let submitted=false;
 try{
  if(READONLY)throw Error('自动只读检查不能保存真人记录');if(!ready||busy)return;
  const actor=$('actor').value.trim();if(actor!==lockedActor)throw Error('身份变更后请重新登记辅助暴露');
  const chosen=document.querySelector('[name=verdict]:checked');if(!chosen)throw Error('请明确选择A/B/C/D');
  if(!$('reason').value.trim()){$('reason').focus();throw Error('请填写判断理由与风险（必填）');}
  if(!manualClicks.length&&!regions.length&&!$('location').value.trim()){$('location').focus();throw Error('请直接点击图像、填写位置文字，或加入当前查看区域');}
  busy=true;setReady(true);$('saveStatus').textContent='正在写入磁盘并核验版本…';
  submitted=true;const d=await api('/api/triage',{sample_id:current.sample_id,actor,verdict:chosen.value,reason:$('reason').value,location_description:$('location').value,approx_regions:regions,manual_clicks:manualClicks,recommend_independent_annotation:$('recommend').checked,expected_revision:revision});
  if(d.disk_confirmed!==true)throw Error('服务没有确认磁盘写入，请重新读取历史核实');
  revision=d.revision;dirty=false;
  $('saveStatus').textContent=`实际写入成功：v${d.revision} · ${d.verdict} · ${d.at}；磁盘记录 ${d.history_count} 条；本版本直接观察点击 ${(d.manual_clicks||[]).length} 个，窗口 ${(d.approx_regions||[]).length} 个。辅助筛选，不是GT。`;
  try{await loadRecord();message('已刷新历史列表并定位新版本的黄色观察位置。');}
  catch(e){message('版本已写入，但历史显示刷新失败：'+e.message+'；请重新读取磁盘记录。');}
 }catch(e){$('saveStatus').textContent=`${submitted?'保存请求失败，写入状态请重新读取历史核实':'未保存，未发出写入请求'}：${e.message}`;message($('saveStatus').textContent);}
 finally{busy=false;setReady(ready);}
};
window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue='';}};
(async()=>{tasks=await api('/api/tasks');for(const t of tasks){const o=document.createElement('option');o.value=t.sample_id;o.textContent=`${t.priority<=6?'优先 '+t.priority+' · ':''}${t.site_name} · ${t.azimuth_delta_deg===null?'原V2':t.azimuth_delta_deg.toFixed(3)+'°'} · ${t.pair_id}`;$('task').append(o);}if(READONLY){$('verification').hidden=false;$('actor').disabled=true;$('enter').textContent='自动只读检查';await load(tasks[0].sample_id);}})().catch(e=>message(e.message));
