'use strict';
const $=id=>document.getElementById(id), C=Coordinates;
const deep=o=>JSON.parse(JSON.stringify(o));
let tasks=[],task,doc,points=[],undo=[],selected=null,dirty=false,assistance=[],loadSequence=0;
const views=[0,1].map(i=>({canvas:$('c'+i),image:new Image(),camera:{scale:1,tx:0,ty:0},drag:null,moved:false}));
function message(s){$('message').textContent=s;}
async function api(path,data){const r=await fetch(path,data?{method:'POST',headers:{'Content-Type':'application/json','X-SAR-Token':window.SAR_TOKEN},body:JSON.stringify(data)}:{});const v=await r.json();if(!r.ok)throw Error(v.error);return v;}
function actor(){const a=$('actor').value.trim();if(!a)throw Error('请填写真实标注/复核人员身份');return a;}
function values(side){const x=$('x'+side).value,y=$('y'+side).value;return x!==''&&y!==''?[Number(x),Number(y)]:null;}
function coordinateInputs(p0,p1){[p0,p1].forEach((p,i)=>{ $('x'+i).value=p?p[0]:'';$('y'+i).value=p?p[1]:'';});}
function clearPoint(){selected=null;coordinateInputs(null,null);$('pointId').value='P'+String(Math.max(0,...points.map(p=>Number(p.point_id.replace(/^P/,''))||0))+1).padStart(4,'0');$('landmark').value='';$('note').value='';$('static').checked=false;$('visible').checked=false;$('sigma0').value='';$('sigma1').value='';drawAll();renderList();}
function history(){undo.push(deep(points));if(undo.length>100)undo.shift();dirty=true;}
function draw(i){
 const v=views[i],ctx=v.canvas.getContext('2d'),c=v.camera;ctx.clearRect(0,0,v.canvas.width,v.canvas.height);ctx.save();ctx.filter=`contrast(${$('contrast'+i).value}) brightness(${$('bright'+i).value})`;ctx.imageSmoothingEnabled=false;ctx.drawImage(v.image,c.tx,c.ty,v.image.width*c.scale,v.image.height*c.scale);ctx.restore();
 const mark=(p,label,color,r=4)=>{if(!p)return;const s=C.pixelToScreen(...p,c);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(s[0],s[1],r,0,Math.PI*2);ctx.stroke();ctx.fillText(label,s[0]+6,s[1]-5);};
 if($('mode').value!=='review')points.forEach(p=>mark(p['xy'+i],p.point_id,p.point_id===selected?'#ffe087':'#00e5ff'));
 if($('mode').value==='assisted')assistance.forEach((p,j)=>mark(p['xy'+i],'A'+j,'#ffbe55',3));
 mark(values(i),$('mode').value==='review'?'R':'new','#ee9aff',5);
}
function drawAll(){views.forEach((v,i)=>{if(v.image.complete&&v.image.naturalWidth)draw(i);});}
function loupe(i,p){const v=views[i],ctx=$('l'+i).getContext('2d');ctx.clearRect(0,0,128,128);ctx.save();ctx.imageSmoothingEnabled=false;ctx.filter=`contrast(${$('contrast'+i).value}) brightness(${$('bright'+i).value})`;ctx.drawImage(v.image,p[0]+.5-8,p[1]+.5-8,16,16,0,0,128,128);ctx.restore();ctx.strokeStyle='#ffdd73';ctx.beginPath();ctx.moveTo(0,64);ctx.lineTo(128,64);ctx.moveTo(64,0);ctx.lineTo(64,128);ctx.stroke();}
function renderList(){const list=$('points');list.replaceChildren();points.forEach(p=>{const b=document.createElement('button');b.className='point'+(selected===p.point_id?' selected':'');b.textContent=p.point_id+' '+(p.review?p.review.conclusion:'未复核');b.onclick=()=>{selected=p.point_id;$('pointId').value=p.point_id;$('role').value=p.role;$('landmark').value=p.landmark_type||'';$('note').value=p.note||'';if($('mode').value==='review'){coordinateInputs(null,null);$('static').checked=false;$('visible').checked=false;$('sigma0').value='';$('sigma1').value='';}else{coordinateInputs(p.xy0,p.xy1);$('static').checked=p.is_static;$('visible').checked=p.visible_both;$('sigma0').value=p.sigma0_px;$('sigma1').value=p.sigma1_px;$('confidence').value=p.confidence;}drawAll();renderList();};list.append(b);});const q=doc?.gt_quality;$('quality').textContent=`${points.length} 个草稿点 · 已保存版本 ${doc?.revision||0} · GT状态 ${q?.status||'unverified'}\n合格检查点 ${q?.eligible_checks||0} · 合格控制点 ${q?.eligible_controls||0} · 评价范围 ${q?.scope||'local'}\n联合定位不确定性上限 ${q?.max_combined_uncertainty_px??'未知'} px · 可报告稀疏 Success 阈值 ${(q?.supported_success_thresholds_px||[]).join(',')||'无'}\n`+(q?.warnings||[]).join('\n');}
async function load(index){
 if(dirty&&!confirm('当前草稿未保存，切换将放弃。继续？'))return;
 const sequence=++loadSequence;task=tasks[(index+tasks.length)%tasks.length];$('task').value=task.sample_id;assistance=[];undo=[];dirty=false;
 doc=await api('/api/annotation?sample_id='+encodeURIComponent(task.sample_id));if(sequence!==loadSequence)return;points=deep(doc.points);
 $('disposition').value=doc.disposition;$('dispositionNote').value=doc.disposition_note||'';$('scope').value=doc.evaluation_scope;
 $('metadata').textContent=`${task.sample_id} | ${task.sensor} | ${task.site_name} | ${task.visual_scene}\n图0 ${task.reference_utc}  spacing ${task.native_reference_spacing_m_x.toFixed(3)} × ${task.native_reference_spacing_m_y.toFixed(3)} m  orbit ${task.reference_orbit_direction||'unknown'}  look ${task.reference_observation_direction||'unknown'}  azimuth ${task.reference_azimuth_degrees??'unknown'}° incidence ${task.reference_incidence_degrees??'unknown'}°\n图1 ${task.moving_utc}  spacing ${task.native_moving_spacing_m_x.toFixed(3)} × ${task.native_moving_spacing_m_y.toFixed(3)} m  orbit ${task.moving_orbit_direction||'unknown'}  look ${task.moving_observation_direction||'unknown'}  azimuth ${task.moving_azimuth_degrees??'unknown'}° incidence ${task.moving_incidence_degrees??'unknown'}°\n采集身份 ${task.reference_acquisition_id} → ${task.moving_acquisition_id}\n准备状态：${task.annotation_readiness} · ${task.visual_note}（不是人工 GT 判断）`;
 views.forEach((v,i)=>{v.camera={scale:1,tx:0,ty:0};v.canvas.width=task['width'+i];v.canvas.height=task['height'+i];v.image.onload=()=>draw(i);v.image.src='/api/image?sample_id='+encodeURIComponent(task.sample_id)+'&side='+i;});clearPoint();message('已加载原生图像。默认无辅助点；保存坐标为原生像素中心坐标。');
}
function handle(fn){return async()=>{try{await fn();}catch(e){message(e.message);}};}
function acceptSaved(value){doc=value;points=deep(doc.points);dirty=false;undo=[];renderList();drawAll();message('保存成功，版本 '+doc.revision+'；GT '+doc.gt_quality.status);}
$('upsert').onclick=handle(()=>{if($('mode').value==='review')throw Error('复核模式请使用保存独立复核');const xy0=values(0),xy1=values(1);if(!xy0||!xy1)throw Error('请分别在两图选择点');if($('sigma0').value===''||$('sigma1').value==='')throw Error('需明确判断两图定位不确定性');actor();history();const p={sample_id:task.sample_id,point_id:$('pointId').value.trim(),xy0,xy1,role:$('role').value,landmark_type:$('landmark').value.trim(),is_static:$('static').checked,visible_both:$('visible').checked,confidence:$('confidence').value,sigma0_px:Number($('sigma0').value),sigma1_px:Number($('sigma1').value),note:$('note').value.trim()};const idx=points.findIndex(x=>x.point_id===p.point_id);if(idx>=0)points[idx]=p;else points.push(p);selected=p.point_id;renderList();drawAll();message('草稿已更新；请保存图像对。');});
$('save').onclick=handle(async()=>acceptSaved(await api('/api/save',{sample_id:task.sample_id,expected_revision:doc.revision,actor:actor(),points,disposition:$('disposition').value,disposition_note:$('dispositionNote').value,evaluation_scope:$('scope').value})));
$('review').onclick=handle(async()=>{if($('mode').value!=='review')throw Error('请切换到独立复核模式');if(dirty)throw Error('请先保存或重新加载草稿');if(!selected||!values(0)||!values(1))throw Error('选择一个点ID并在两图中独立重新点击');if($('sigma0').value===''||$('sigma1').value==='')throw Error('复核定位不确定性必须填写');acceptSaved(await api('/api/review',{sample_id:task.sample_id,expected_revision:doc.revision,actor:actor(),point_id:selected,xy0:values(0),xy1:values(1),sigma0_px:Number($('sigma0').value),sigma1_px:Number($('sigma1').value),conclusion:$('conclusion').value,static_confirmed:$('static').checked,visible_confirmed:$('visible').checked,note:$('note').value}));coordinateInputs(null,null);drawAll();});
$('delete').onclick=handle(()=>{if(!selected)throw Error('先选点 ID');history();points=points.filter(p=>p.point_id!==selected);clearPoint();});
$('undo').onclick=()=>{if(undo.length){points=undo.pop();dirty=true;clearPoint();message('撤销草稿操作；保存后产生新版本，旧版本保留。');}};
$('newPoint').onclick=clearPoint;$('reload').onclick=handle(()=>load(tasks.findIndex(t=>t.sample_id===task.sample_id)));
$('task').onchange=handle(()=>load(tasks.findIndex(t=>t.sample_id===$('task').value)));$('prev').onclick=handle(()=>load(tasks.findIndex(t=>t.sample_id===task.sample_id)-1));$('next').onclick=handle(()=>load(tasks.findIndex(t=>t.sample_id===task.sample_id)+1));
$('mode').onchange=()=>{assistance=[];$('geo').disabled=$('sift').disabled=$('mode').value!=='assisted';coordinateInputs(null,null);selected=null;drawAll();renderList();message($('mode').value==='review'?'复核者必须为另一人；隐藏原坐标，先选点ID，再重新判断物理结构和位置。':'模式已切换。已经打开过的辅助信息记录不会删除。');};
for(const [id,source] of [['geo','geographic'],['sift','sift_candidates']])$(id).onclick=handle(async()=>{if($('mode').value!=='assisted')throw Error('先切换辅助模式');const r=await api('/api/assistance',{sample_id:task.sample_id,actor:actor(),source});assistance=r.points;drawAll();message(r.warning+' 图层 '+r.event.source+' 生成器 '+r.event.generator+' 候选数 '+r.points.length);});
$('hide').onclick=()=>{assistance=[];drawAll();};
views.forEach((v,i)=>{
 const local=e=>{const r=v.canvas.getBoundingClientRect();return C.clientToCanvas(e.clientX,e.clientY,r,v.canvas.width,v.canvas.height);};
 v.canvas.oncontextmenu=e=>e.preventDefault();v.canvas.onwheel=e=>{e.preventDefault();const p=local(e);v.camera=C.zoomAt(v.camera,...p,e.deltaY<0?1.2:1/1.2);draw(i);};
 v.canvas.onpointerdown=e=>{if(e.shiftKey||e.button===1||e.button===2){v.drag={position:local(e),camera:{...v.camera}};v.canvas.setPointerCapture(e.pointerId);}else{const p=C.screenToPixel(...local(e),v.camera);if(p[0]>=0&&p[0]<=task['width'+i]-1&&p[1]>=0&&p[1]<=task['height'+i]-1){$('x'+i).value=p[0].toFixed(3);$('y'+i).value=p[1].toFixed(3);draw(i);}}};
 v.canvas.onpointermove=e=>{const s=local(e);if(v.drag){v.camera.tx=v.drag.camera.tx+s[0]-v.drag.position[0];v.camera.ty=v.drag.camera.ty+s[1]-v.drag.position[1];draw(i);}const p=C.screenToPixel(...s,v.camera);$('hover'+i).textContent=`原生 x=${p[0].toFixed(2)}, y=${p[1].toFixed(2)}; zoom=${v.camera.scale.toFixed(2)}`;if(v.image.complete)loupe(i,p);};
 v.canvas.onpointerup=v.canvas.onpointercancel=()=>v.drag=null;
 for(const id of ['contrast','bright'])$(id+i).oninput=()=>draw(i);
 for(const id of ['x','y'])$(id+i).oninput=()=>draw(i);
});
document.querySelectorAll('[data-zoom]').forEach(b=>b.onclick=()=>{const [i,f]=b.dataset.zoom.split(',').map(Number);views[i].camera=C.zoomAt(views[i].camera,views[i].canvas.width/2,views[i].canvas.height/2,f);draw(i);});
document.querySelectorAll('[data-reset]').forEach(b=>b.onclick=()=>{const i=Number(b.dataset.reset);views[i].camera={scale:1,tx:0,ty:0};draw(i);});
for(const id of ['disposition','scope','dispositionNote'])$(id).onchange=()=>dirty=true;
window.onbeforeunload=e=>{if(dirty){e.preventDefault();e.returnValue='';}};
api('/api/tasks').then(async t=>{tasks=t;for(const r of t){const o=document.createElement('option');o.value=r.sample_id;o.textContent=(r.demo_priority?'FIRST '+r.demo_priority+' · ':'')+r.sample_id+' · '+r.site_name;$('task').append(o);}await load(0);}).catch(e=>message(e.message));
