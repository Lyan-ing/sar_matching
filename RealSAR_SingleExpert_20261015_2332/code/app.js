'use strict';
// Reuses the Native viewer's pixel-center, CSS scaling and zoom conventions.
const $=id=>document.getElementById(id),C=Coordinates,deep=o=>JSON.parse(JSON.stringify(o)),uid=()=>crypto.randomUUID();
let tasks=[],filtered=[],page=0,scene=null,doc=null,mode='patch',selected=null,xy=[null,null],inputs=[null,null],undo=[],readonly=false,loading=false,saving=false,dirtyVersion=0,ackVersion=0,timer=null,loadToken=0;
const views=[0,1].map(i=>({canvas:$('c'+i),camera:{scale:1,tx:0,ty:0},overview:null,tile:null,drag:null,fetchToken:0,tileTimer:null}));
const fields=['structureType','group','reason','confidence','sigma0','sigma1','physical','risk','planar','role','riskNote','repeatTarget'];
async function api(path,body){const r=await fetch(path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(d.error||'请求失败');return d;}
function status(s){$('status').textContent=s;}
function error(e){$('saved').textContent='未保存：'+e.message;status(e.message+'。当前草稿保留在本浏览器；不要清除浏览器存储。');}
function handle(fn){return async(...args)=>{try{await fn(...args);}catch(e){error(e);}};}
function cacheKey(){return 'sar_single_expert:'+doc?.scope;}
function currentDraft(){return {selected,xy:deep(xy),inputs:deep(inputs),frame_ids:scene.frames.map(f=>f.frame_id),form:Object.fromEntries(fields.map(k=>[k,k==='physical'?$(k).checked:$(k).value])),cameras:views.map(v=>({...v.camera}))};}
function snapshot(){return {sample_id:scene.task.sample_id,mode,frame_ids:scene.frames.map(f=>f.frame_id),expected_revision:doc.revision,client_save_id:uid(),points:deep(doc.points),draft:currentDraft(),task_status:$('taskStatus').value,decision_reason:$('decision').value};}
function dirty(){if(!doc||loading||readonly)return;dirtyVersion++;const p=snapshot();localStorage.setItem(cacheKey(),JSON.stringify(p));$('saved').textContent='草稿待落盘…';clearTimeout(timer);timer=setTimeout(()=>flush().catch(error),500);}
async function flush(){
 if(readonly||!doc||dirtyVersion===ackVersion)return;
 if(saving){await saving;return flush();}
 const version=dirtyVersion,p=snapshot(),key=cacheKey();
 saving=(async()=>{const r=await api('/api/save',p);if(!r.disk_confirmed)throw Error('服务器未确认实际落盘');doc.revision=r.revision;
  if(version===dirtyVersion){doc.points=r.document.points;doc.task_status=r.document.task_status;ackVersion=version;localStorage.removeItem(key);$('saved').textContent=`磁盘已保存 v${r.revision} · ${r.history_count}个版本`;}else{$('saved').textContent=`已保存 v${r.revision}，继续保存新草稿…`;}
  if(r.quality)quality(r.quality);if(r.export_error)status('人工记录已落盘，但GT导出失败：'+r.export_error);else if(r.export)status('已自动更新单人专家GT与Train/Val覆盖；原双人GT标志不变。');
 })();
 try{await saving;}finally{saving=false;}
 if(version!==dirtyVersion)return flush();
}
function formLoad(d){
 selected=d?.selected||null;xy=deep(d?.xy||[null,null]);inputs=deep(d?.inputs||[null,null]);
 for(const id of fields){if(id==='physical')$(id).checked=d?.form?.physical===true;else $(id).value=d?.form?.[id]??(id==='confidence'||id==='risk'||id==='planar'?'unknown':id==='role'?'check':'');}
 render();
}
function clearPoint(keepGroup=true){const group=keepGroup?$('group').value:'',type=keepGroup?$('structureType').value:'';formLoad(null);$('group').value=group;$('structureType').value=type;render();}
function localPoint(p){
 if(p.native_xy&&p.acquisition_ids)return scene.frames.map(f=>{const side=p.acquisition_ids.indexOf(f.acquisition_id);return side>=0?[p.native_xy[side][0]-f.origin_xy[0],p.native_xy[side][1]-f.origin_xy[1]]:null;});
 return p.frame_ids?.join()===scene.frames.map(f=>f.frame_id).join()?p.xy:[null,null];
}
function selectedPoint(p){
 const order=p.acquisition_ids?scene.frames.map(f=>p.acquisition_ids.indexOf(f.acquisition_id)):[0,1];
 const loc=localPoint(p);if(loc.some((x,i)=>!x||x[0]<0||x[1]<0||x[0]>scene.frames[i].width-1||x[1]>scene.frames[i].height-1)){
  status('此点在另一Native窗口，正在回到原标注窗口。');return openTask(scene.task.sample_id,mode,order.map(i=>p.frame_ids[i])).then(()=>selectedPoint(doc.points.find(q=>q.point_id===p.point_id)));
 }
 selected=p.point_id;xy=deep(loc);inputs=deep(order.map(i=>p.inputs[i]));$('structureType').value=p.structure_type;$('group').value=p.structure_group;$('reason').value=p.reason;$('confidence').value=p.confidence;$('physical').checked=p.physical_confirmed;$('risk').value=p.risk_status;$('planar').value=p.planar;$('role').value=p.role;$('riskNote').value=p.risk_note||'';[0,1].forEach(i=>$('sigma'+i).value=p.sigma[order[i]]??'');if(mode==='repeat')$('repeatTarget').value=p.repeat_target||'';render();dirty();
}
function render(){
 if(!scene)return;$('pointTitle').textContent=selected?'编辑 '+selected.slice(0,8):'新对应点';
 $('coords').textContent=xy.map((p,i)=>(i?'右':'左')+(p?` x=${p[0].toFixed(3)} y=${p[1].toFixed(3)}\n产品 x=${(p[0]+scene.frames[i].origin_xy[0]).toFixed(3)} y=${(p[1]+scene.frames[i].origin_xy[1]).toFixed(3)}`:'图未选')).join('\n');
 $('savePoint').disabled=readonly||loading||!xy[0]||!xy[1];$('delete').disabled=readonly||!selected;$('undo').disabled=readonly||!undo.length;
 $('pointList').replaceChildren();
 for(const p of doc.points){const b=document.createElement('button');b.textContent=(p.point_id===selected?'▶ ':'')+(p.structure_type||'未判断结构')+' · '+p.point_id.slice(0,8);b.onclick=handle(()=>selectedPoint(p));$('pointList').append(b);}
 for(const p of scene.inherited_points){const b=document.createElement('button');b.textContent='复用位置 · '+p.structure_type+'（点击编辑原始记录）';b.style.color='#ffe18a';b.onclick=handle(async()=>{await openTask(p.source_sample_id,p.source_mode);const q=doc.points.find(q=>q.point_id===p.point_id);if(q)await selectedPoint(q);});$('pointList').append(b);}
 drawAll();
}
function quality(q){const names={pending:'待标注/待补证据',point:'点级对应',local:'局部GT',full:'全幅稀疏GT',no_reliable:'用户判断无可靠对应'};
 const label=r=>{const exact={insufficient_independent_checks:'检查点不足',insufficient_independent_structure_groups:'独立结构组不足',too_many_checks_on_one_structure_group:'同组超过2个检查点'};return exact[r]||r.replace(/side([01])_(hull|span|grid|quadrants|separation)/,(_,side,k)=>(side==='0'?'左图':'右图')+({hull:'凸包覆盖不足',span:'跨度不足',grid:'网格分散不足',quadrants:'象限不足',separation:'检查点距离不足32px'})[k]);};
 $('quality').textContent=`单人专家：${names[q.expert_gt_level]} · 可用物理点 ${q.eligible_points} · 几何阈值支持 ${q.supported_success_thresholds_px.join('/')||'尚无'} px\n全幅：${q.primary.passed?'满足':q.primary.reasons.map(label).join('、')}\n局部：${q.local.passed?'满足':q.local.reasons.map(label).join('、')}`+(q.point_checks||[]).map(p=>'\n点 '+p.point_id.slice(0,8)+'：'+(p.failures.length?'待补 '+p.failures.join('；'):'物理字段通过')+(p.duplicate_not_counted?' · 重复观察不重复计数':'' )).join('')+(q.warnings.length?'\n'+q.warnings.join('\n'):'');}
function image(url){return new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(Error('图像读取失败'));im.src=url;});}
function fit(i){const f=scene.frames[i],v=views[i],scale=Math.min(v.canvas.width/f.width,v.canvas.height/f.height);v.camera={scale,tx:(512-f.width*scale)/2,ty:(512-f.height*scale)/2};}
function focusCrop(i){const f=scene.frames[i],crop=scene.task.frame_ids[i];if(mode!=='native'){fit(i);return;}
 const p=scene.frame_options[i].find(x=>x.frame_id===crop);if(!p){fit(i);return;}const x=p.origin_xy[0]-f.origin_xy[0]+256,y=p.origin_xy[1]-f.origin_xy[1]+256;views[i].camera={scale:1,tx:256-x,ty:256-y};}
async function openTask(sid,newMode=null,frameIDs=null){
 newMode=newMode||(tasks.find(t=>t.sample_id===sid)?.native_context_available?'native':'patch');
 if(tasks.find(t=>t.sample_id===sid)?.native_only)newMode='native';
 await flush();clearTimeout(timer);loading=true;const token=++loadToken;$('saved').textContent='读取图像与已保存版本…';
 try{
  const s=await api('/api/scene?sid='+encodeURIComponent(sid)+'&mode='+newMode+(frameIDs?'&frames='+encodeURIComponent(JSON.stringify(frameIDs)):''));const ims=await Promise.all(s.frames.map(f=>image('/api/image?frame='+f.frame_id)));if(token!==loadToken)return;
  scene=s;doc=s.document;mode=newMode;readonly=s.readonly;dirtyVersion=ackVersion=0;undo=[];$('gallery').hidden=true;$('editor').hidden=false;$('viewMode').value=mode;$('sampleTitle').textContent=`${s.task.site} · ${s.task.angle===null?'角度未知':s.task.angle.toFixed(3)+'°'} · ${s.task.split}`;
  for(const o of $('viewMode').options)o.disabled=!!s.task.native_only&&o.value!=='native';
  $('taskStatus').value=doc.task_status;$('decision').value=doc.decision_reason;$('repeatLabel').hidden=mode!=='repeat';$('repeatTarget').replaceChildren();
  for(const p of s.repeat_targets){const o=document.createElement('option');o.value=p.repeat_token;o.textContent=p.structure_type+' · '+p.structure_group+' · '+p.reason;$('repeatTarget').append(o);}
  s.frames.forEach((f,i)=>{
   const v=views[i];v.overview=ims[i];v.tile=null;v.fetchToken++;fit(i);$('frame'+i).replaceChildren();
   for(const a of s.frame_options[i]){const o=document.createElement('option');o.value=a.frame_id;o.textContent=`${a.width}×${a.height} 原生窗口 · 起点 ${a.origin_xy.join(',')}`;$('frame'+i).append(o);}
   if(!$('frame'+i).querySelector(`option[value="${f.frame_id}"]`)){const o=document.createElement('option');o.value=f.frame_id;o.textContent=f.width+'×'+f.height+' 当前窗口';$('frame'+i).append(o);}
   $('frame'+i).value=f.frame_id;$('frame'+i).disabled=mode!=='native';$('bank'+i).replaceChildren();
   if(mode!=='repeat')for(const a of s.bank[i]){const b=document.createElement('button');b.textContent=a.structure_type+' · '+a.structure_group+' · '+a.xy.map(x=>x.toFixed(1)).join(',');b.onclick=()=>{xy[i]=deep(a.xy);inputs[i]={kind:'reused_single_side',input_id:a.input_id,reference:a.reference};$('physical').checked=false;$('sigma'+i).value='';render();dirty();status('仅引用了这一侧的真实旧坐标；另一侧仍由你寻找并确认同一物理点。');};$('bank'+i).append(b);}
  });
  formLoad(doc.draft);
  if(doc.draft?.cameras?.length===2&&!frameIDs)views.forEach((v,i)=>v.camera=doc.draft.cameras[i]);
  const cached=localStorage.getItem(cacheKey());if(cached&&!readonly){const p=JSON.parse(cached);if(p.expected_revision===doc.revision){doc.points=p.points;$('taskStatus').value=p.task_status;$('decision').value=p.decision_reason;formLoad(p.draft);dirtyVersion=1;status('已恢复上次尚未落盘的浏览器草稿，正在重试保存。');}else status('发现另一版本浏览器备份，磁盘记录未覆盖。可先完成当前已保存版本；备份仍保留在浏览器。');}
  $('taskMeta').textContent=JSON.stringify({sample:s.task.sample_id,relation:s.task.relation_id,acquisitions:s.task.acquisition_ids,scene:s.task.scene,AI_class:s.task.AI_class,AI_hint:s.task.AI_hint,Native_context_available:s.task.native_context_available,component:s.task.component,mode,delayed_repeat_minimum_hours:24},null,2);
  if(mode==='repeat'){$('quality').textContent='同一专家延迟复标：第一次点位、草稿、相机位置及单侧库均隐藏。按结构文字重新定位；24小时以内的记录仅作早期重复，不冒充独立验证。';status(s.repeat_targets.length?'选择结构文字，再独立重新点两侧，保留两次原记录，不自动平均。':'本512尚无可复标的已确认物理点，请先正常标注。');}
  else{quality(s.quality);status(mode==='native'?`当前读取原生大图/已下载窗口：左 ${s.frames[0].width}×${s.frames[0].height}，右 ${s.frames[1].width}×${s.frames[1].height}。先全景找结构，再滚轮放大或点“原生1:1”。保存原始产品坐标；不受512边界限制，不确定的点可保留待核验。`:(s.task.native_context_available?'':'本关系当前本地没有双侧大图，仅有512；不要勉强标注。\n')+'左图点击 → 右图点击 → 填简短物理依据、不确定度与风险判断 → Enter确认。点击草稿会自动保存。');}
  const h=await api('/api/history?sid='+encodeURIComponent(sid)+'&mode='+mode);$('history').textContent=h.length? h.map(v=>`v${v.revision} · ${v.recorded_at} · ${v.points}点 · ${v.task_status}`).join('； '):'尚无人工记录';
  $('saved').textContent=readonly?'只读自动验收 · 禁止保存':`磁盘版本 v${doc.revision}`;history.replaceState(null,'','/?task='+encodeURIComponent(sid)+'&mode='+mode);render();views.forEach((v,i)=>requestTile(i));
 }finally{loading=false;if(dirtyVersion!==ackVersion&&!readonly)await flush();}
}
function stretch(im,sharedHist=null){const a=document.createElement('canvas');a.width=im.naturalWidth||im.width;a.height=im.naturalHeight||im.height;const c=a.getContext('2d',{willReadFrequently:true});c.drawImage(im,0,0);const d=c.getImageData(0,0,a.width,a.height),hist=sharedHist||new Uint32Array(256);if(!sharedHist)for(let i=0;i<d.data.length;i+=4)hist[d.data[i]]++;const total=sharedHist?hist.reduce((sum,n)=>sum+n,0):a.width*a.height;let acc=0,lo=0,hi=255;for(let i=0;i<256;i++){acc+=hist[i];if(acc<total*.01)lo=i;if(acc<total*.99)hi=i;}for(let i=0;i<d.data.length;i+=4){const v=Math.max(0,Math.min(255,(d.data[i]-lo)/Math.max(1,hi-lo)*255));d.data[i]=d.data[i+1]=d.data[i+2]=v;}c.putImageData(d,0,0);return a;}
function draw(i){
 if(!scene)return;const v=views[i],ctx=v.canvas.getContext('2d'),f=scene.frames[i],c=v.camera;ctx.clearRect(0,0,512,512);ctx.save();ctx.imageSmoothingEnabled=false;ctx.filter=`contrast(${$('contrast'+i).value}) brightness(${$('bright'+i).value})`;if(v.overview)ctx.drawImage(v.overview,c.tx,c.ty,f.width*c.scale,f.height*c.scale);
 if(v.tile)for(const t of v.tile.patches||[v.tile]){const b=t.box;ctx.drawImage($('local'+i).checked&&v.tile.complete!==false?t.stretched:t.im,c.tx+b[0]*c.scale,c.ty+b[1]*c.scale,b[2]*c.scale,b[3]*c.scale);}ctx.restore();
 const mark=(p,label,color)=>{if(!p)return;const s=C.pixelToScreen(...p,c);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(s[0],s[1],5,0,Math.PI*2);ctx.stroke();ctx.fillText(label,s[0]+7,s[1]-7);};
 for(const p of doc.points)mark(localPoint(p)[i],p.point_id.slice(0,4),'#ffe18a');for(const p of scene.inherited_points)mark(p.xy[i],'复用','#ffe18a');if(mode!=='repeat')for(const a of scene.bank[i])mark(a.xy,'单侧','#71d9e9');mark(xy[i],selected?'编辑':'草稿','#ff99cf');
}
function drawAll(){views.forEach((v,i)=>draw(i));}
function requestTile(i){
 const v=views[i];clearTimeout(v.tileTimer);v.tileTimer=setTimeout(async()=>{if(!scene)return;const f=scene.frames[i],c={...v.camera};const x=Math.max(0,Math.floor(-c.tx/c.scale)),y=Math.max(0,Math.floor(-c.ty/c.scale)),right=Math.min(f.width,Math.ceil((512-c.tx)/c.scale)),bottom=Math.min(f.height,Math.ceil((512-c.ty)/c.scale)),box=[x,y,right-x,bottom-y],token=++v.fetchToken;
  v.tile=null;
  if(Math.min(box[2],box[3])<=0){$('detail'+i).textContent='视野在影像范围外';draw(i);return;}
  const overviewScale=Math.min(v.overview.naturalWidth/f.width,v.overview.naturalHeight/f.height);
  if(Math.max(box[2],box[3])>1024&&c.scale<=overviewScale){$('detail'+i).textContent='缩小总览；放大后自动读取原生细节';draw(i);return;}
  // Keep each existing image endpoint request bounded; load the visible window in pieces.
  const boxes=[],patches=[];for(let yy=y;yy<bottom;yy+=1024)for(let xx=x;xx<right;xx+=1024)boxes.push([xx,yy,Math.min(1024,right-xx),Math.min(1024,bottom-yy)]);
  $('detail'+i).textContent='正在读取原生细节 0/'+boxes.length;
  try{for(let n=0;n<boxes.length;n+=4){
   if(token!==v.fetchToken||scene.frames[i].frame_id!==f.frame_id)return;
   const batch=await Promise.all(boxes.slice(n,n+4).map(async b=>{const im=await image('/api/image?frame='+f.frame_id+'&box='+encodeURIComponent(JSON.stringify(b)));return {im,box:b,stretched:stretch(im)};}));
   if(token!==v.fetchToken||scene.frames[i].frame_id!==f.frame_id)return;
   patches.push(...batch);
   if(patches.length===boxes.length&&patches.length>1){
    // One contrast range for the entire visible window, avoiding artificial tile seams.
    const histogram=new Uint32Array(256),a=document.createElement('canvas'),ctx=a.getContext('2d',{willReadFrequently:true});
    for(const p of patches){a.width=p.im.naturalWidth;a.height=p.im.naturalHeight;ctx.drawImage(p.im,0,0);const pixels=ctx.getImageData(0,0,a.width,a.height).data;for(let j=0;j<pixels.length;j+=4)histogram[pixels[j]]++;}
    for(const p of patches)p.stretched=stretch(p.im,histogram);
   }
   v.tile={...patches[0],patches,requested_box:box,complete:patches.length===boxes.length};draw(i);
   $('detail'+i).textContent=v.tile.complete?'原生细节已加载（未使用总览代替当前窗口）':'正在读取原生细节 '+patches.length+'/'+boxes.length;
  }}catch(e){if(token===v.fetchToken){$('detail'+i).textContent='原生细节读取失败；当前总览不宜精确标点';status(e.message);}}
 },90);
}
function cameraChanged(i){views[i].tile=null;views[i].fetchToken++;draw(i);requestTile(i);dirty();}
views.forEach((v,i)=>{
 const local=e=>C.clientToCanvas(e.clientX,e.clientY,v.canvas.getBoundingClientRect(),512,512);
 v.canvas.oncontextmenu=e=>e.preventDefault();v.canvas.onwheel=e=>{e.preventDefault();if(!scene)return;v.camera=C.zoomAt(v.camera,...local(e),e.deltaY<0?1.2:1/1.2);cameraChanged(i);};
 v.canvas.onpointerdown=e=>{if(!scene||loading)return;const p=local(e);if(e.shiftKey||e.button===1||e.button===2){v.drag={p,c:{...v.camera}};v.canvas.setPointerCapture(e.pointerId);return;}if(readonly){status('只读验收不会登记真人点击。');return;}const q=C.screenToPixel(...p,v.camera),f=scene.frames[i];if(q[0]<0||q[1]<0||q[0]>f.width-1||q[1]>f.height-1)return;xy[i]=q.map(x=>Number(x.toFixed(4)));inputs[i]={kind:'direct_click',input_id:uid()};$('physical').checked=false;$('sigma'+i).value='';render();dirty();};
 v.canvas.onpointermove=e=>{if(!scene)return;const p=local(e);if(v.drag){v.camera={...v.drag.c,tx:v.drag.c.tx+p[0]-v.drag.p[0],ty:v.drag.c.ty+p[1]-v.drag.p[1]};cameraChanged(i);}const q=C.screenToPixel(...p,v.camera);$('hover'+i).textContent=`原生 x=${q[0].toFixed(2)} y=${q[1].toFixed(2)} · 显示${v.camera.scale.toFixed(2)}×`;};v.canvas.onpointerup=v.canvas.onpointercancel=()=>v.drag=null;
 for(const id of ['contrast','bright','local'])$(id+i).oninput=()=>draw(i);
 $('frame'+i).onchange=handle(async()=>{const ids=scene.frames.map(f=>f.frame_id);ids[i]=$('frame'+i).value;await openTask(scene.task.sample_id,mode,ids);dirty();});
});
document.querySelectorAll('[data-zoom]').forEach(b=>b.onclick=()=>{if(!scene)return;const [i,k]=b.dataset.zoom.split(',').map(Number);views[i].camera=C.zoomAt(views[i].camera,256,256,k);cameraChanged(i);});
document.querySelectorAll('[data-native-pixel]').forEach(b=>b.onclick=()=>{if(!scene)return;const i=Number(b.dataset.nativePixel);views[i].camera=C.zoomAt(views[i].camera,256,256,1/views[i].camera.scale);cameraChanged(i);});
document.querySelectorAll('[data-fit]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset.fit);fit(i);cameraChanged(i);});document.querySelectorAll('[data-crop]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset.crop);focusCrop(i);cameraChanged(i);});
function collectPoint(){return {point_id:selected||uid(),frame_ids:scene.frames.map(f=>f.frame_id),xy:deep(xy),inputs:deep(inputs),structure_type:$('structureType').value,structure_group:$('group').value.trim(),reason:$('reason').value.trim(),confidence:$('confidence').value,sigma:[0,1].map(i=>$('sigma'+i).value===''?null:Number($('sigma'+i).value)),physical_confirmed:$('physical').checked,risk_status:$('risk').value,risk_note:$('riskNote').value.trim(),planar:$('planar').value,role:$('role').value,repeat_target:mode==='repeat'?$('repeatTarget').value:null};}
$('savePoint').onclick=handle(async()=>{if(!xy[0]||!xy[1])throw Error('请亲自在两幅图中定位同一物理结构');if(mode==='repeat'&&!$('repeatTarget').value)throw Error('先选择需要重复定位的结构文字');undo.push(deep(doc.points));const p=collectPoint(),idx=doc.points.findIndex(x=>x.point_id===p.point_id);if(idx>=0)doc.points[idx]=p;else doc.points.push(p);if($('taskStatus').value==='pending')$('taskStatus').value='partial';clearPoint();dirty();await flush();render();status('对应点已实际落盘。黄色显示已保存位置，返回此任务或重开页面会自动恢复。');});
$('newPoint').onclick=()=>{clearPoint();dirty();};$('delete').onclick=handle(async()=>{if(!selected)return;undo.push(deep(doc.points));doc.points=doc.points.filter(p=>p.point_id!==selected);clearPoint();dirty();await flush();render();});$('undo').onclick=handle(async()=>{if(!undo.length)return;doc.points=undo.pop();clearPoint();dirty();await flush();render();});
fields.forEach(id=>$(id).addEventListener('input',dirty));$('taskStatus').onchange=dirty;$('decision').oninput=dirty;
$('repeatTarget').onchange=()=>{const p=scene.repeat_targets.find(x=>x.repeat_token===$('repeatTarget').value);if(p){$('structureType').value=p.structure_type;$('group').value=p.structure_group;$('reason').value=p.reason;}dirty();};
function options(id,values){for(const v of [...new Set(values)].sort()){const o=document.createElement('option');o.value=v;o.textContent=v;$(id).append(o);}}
function filterTasks(){const term=$('search').value.toLowerCase(),focus=$('focus').value,seen=new Set();return tasks.filter(t=>(focus!=='clear'||t.clear_candidate)&&(focus!=='native'||t.native_context_available)&&(focus!=='pilot'||t.pilot30)&&(focus!=='B'||t.AI_class==='B')&&(focus!=='Val'||t.split==='Val')&&(!$('split').value||t.split===$('split').value)&&(!$('site').value||t.site===$('site').value)&&(!$('angle').value||t.angle_bin===$('angle').value)&&(!$('relation').value||t.relation_id===$('relation').value)&&(!$('scene').value||t.scene===$('scene').value)&&(!$('state').value||t.task_status===$('state').value)&&(!term||(t.sample_id+' '+t.site+' '+t.relation_id).toLowerCase().includes(term))).filter(t=>{if(focus!=='native')return true;if(seen.has(t.relation_id))return false;seen.add(t.relation_id);return true;});}
function renderGallery(){filtered=filterTasks();if($('focus').value==='clear')filtered.sort((a,b)=>(a.clear_priority??99)-(b.clear_priority??99));const pages=Math.max(1,Math.ceil(filtered.length/24));page=Math.min(page,pages-1);$('cards').replaceChildren();$('count').textContent=$('focus').value==='native'?`${filtered.length}组原图采集关系（同关系仅展示一次）`:`${filtered.length}项 / ${tasks.filter(t=>!t.native_only).length}对原512 + ${tasks.filter(t=>t.native_only).length}组新原图观察`;$('pageNumber').textContent=`${page+1} / ${pages}`;
 for(const t of filtered.slice(page*24,page*24+24)){const b=document.createElement('button');b.className='card';const title=document.createElement('strong');title.textContent=t.site+' · '+(t.angle===null?'未知':t.angle.toFixed(3)+'°')+' · '+t.split;const im=document.createElement('img');im.loading='lazy';im.src='/api/thumbnail?sid='+encodeURIComponent(t.sample_id);im.alt=t.sample_id+'左右原生SAR缩略图';const desc=document.createElement('div');desc.className='meta';desc.textContent=`${t.same_direction_control?'同向0°对照 · ':t.native_only?'新原图观察 · ':''}${t.AI_class==='B'?'★ B结构候选 · ':''}${t.scene} · ${t.task_status}\n${t.sample_id}\n${t.relation_id}`;b.append(title,im,desc);b.onclick=handle(()=>openTask(t.sample_id));$('cards').append(b);}
}
for(const id of ['focus','split','site','angle','relation','scene','state','search'])$(id).addEventListener(id==='search'?'input':'change',()=>{page=0;renderGallery();});$('prevPage').onclick=()=>{page=Math.max(0,page-1);renderGallery();};$('nextPage').onclick=()=>{page=Math.min(Math.ceil(filtered.length/24)-1,page+1);renderGallery();};
async function refreshTasks(){const d=await api('/api/tasks');tasks=d.tasks;readonly=d.readonly;return d;}
$('browse').onclick=handle(async()=>{await flush();await refreshTasks();$('editor').hidden=true;$('gallery').hidden=false;history.replaceState(null,'','/');renderGallery();});$('pilotStart').onclick=handle(async()=>{$('focus').value=tasks.some(t=>t.clear_candidate)?'clear':'pilot';renderGallery();if(!filtered.length)throw Error('当前队列为空');await openTask(filtered.find(t=>t.task_status!=='complete'&&t.task_status!=='unjudgeable')?.sample_id||filtered[0].sample_id);});
async function nextTask(step){await flush();const list=filtered.length?filtered:tasks;const i=list.findIndex(t=>t.sample_id===scene.task.sample_id),n=i+step;if(n<0||n>=list.length){status('当前筛选队列已到末尾。可回缩略图选择其他任务，不要求继续标全数据集。');return;}await openTask(list[n].sample_id);}
$('next').onclick=handle(()=>nextTask(1));$('previous').onclick=handle(()=>nextTask(-1));$('finish').onclick=handle(async()=>{dirty();await flush();await nextTask(1);});$('reload').onclick=handle(()=>openTask(scene.task.sample_id,mode));$('viewMode').onchange=handle(()=>openTask(scene.task.sample_id,$('viewMode').value));
$('export').onclick=handle(async()=>{await flush();const d=readonly?(await api('/api/export')).latest_export:await api('/api/export',{});if(!d)throw Error('尚无导出');$('exportInfo').textContent=d.split.map(s=>`${s.split}: ${s.total_pairs}组 · 待标${s.pending} · 点级${s.point_level} · 局部${s.local_GT} · 全幅${s.full_GT} · 无可靠${s.no_reliable}`).join('\n')+`\n原始人工对应 ${d.actual_original_correspondences}，派生裁片对应 ${d.derived_crop_correspondence_records}\n${d.manifest}`;$('exportDialog').showModal();});$('closeExport').onclick=()=>$('exportDialog').close();
document.addEventListener('keydown',e=>{const typing=['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName);if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();flush().catch(error);}else if(!typing&&e.key==='Enter'&&scene){e.preventDefault();$('savePoint').click();}else if(!typing&&e.key==='Escape'&&scene)$('newPoint').click();else if(!typing&&(e.ctrlKey||e.metaKey)&&e.key==='z'){e.preventDefault();$('undo').click();}else if(!typing&&e.key==='Delete')$('delete').click();});
window.addEventListener('beforeunload',e=>{if(doc&&dirtyVersion!==ackVersion){localStorage.setItem(cacheKey(),JSON.stringify(snapshot()));e.preventDefault();e.returnValue='草稿仍在浏览器备份，磁盘保存尚未确认';}});
refreshTasks().then(async()=>{options('site',tasks.map(t=>t.site));options('angle',tasks.map(t=>t.angle_bin));options('relation',tasks.map(t=>t.relation_id));options('scene',tasks.map(t=>t.scene));renderGallery();$('saved').textContent=readonly?'只读自动验收':'单人使用 · 自动保存已启用';const p=new URLSearchParams(location.search);if(p.get('task'))await openTask(p.get('task'),p.get('mode'));}).catch(error);
