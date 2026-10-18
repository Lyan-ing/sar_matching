'use strict';
const $=id=>document.getElementById(id),C=Coordinates,deep=o=>JSON.parse(JSON.stringify(o));
const reviewing=new URLSearchParams(location.search).get('review')==='1';
let tasks=[],task,doc,points=[],selected=null,operator='',busy=false,loading=false,undo=[],assistance=[],sequence=0;
const views=[0,1].map(i=>({canvas:$('c'+i),image:new Image(),camera:{scale:1,tx:0,ty:0},drag:null}));
if(reviewing){document.body.classList.add('review');$('title').textContent='第二人独立复核';$('editor').hidden=false;$('editorTitle').textContent='独立重新点击与判断';$('landmark').readOnly=true;$('help').textContent='先选择点的地物说明，在双图重新点击，然后独立判断并保存。原坐标隐藏；不要换身份自审。';$('identityTitle').textContent='真实第二复核人员';}
function message(s){$('message').textContent=s;}
async function api(path,data){const r=await fetch(path,{method:data?'POST':'GET',headers:{'X-SAR-Token':window.SAR_TOKEN,...(data?{'Content-Type':'application/json'}:{})},...(data?{body:JSON.stringify(data)}:{})});const v=await r.json();if(!r.ok)throw Error(v.error);return v;}
function actor(){if(!operator)throw Error('请先输入本人的真实身份');return operator;}
function values(i){const x=$('x'+i).value,y=$('y'+i).value;return x!==''&&y!==''?[Number(x),Number(y)]:null;}
function coordinateInputs(a,b){[a,b].forEach((p,i)=>{$('x'+i).value=p?p[0]:'';$('y'+i).value=p?p[1]:'';});}
function unknownFields(){$('static').checked=false;$('visible').checked=false;$('sigma0').value='';$('sigma1').value='';$('confidence').value='unknown';}
function clearPoint(){selected=null;coordinateInputs(null,null);unknownFields();$('landmark').value='';$('note').value='';$('role').value='check';$('conclusion').value='uncertain';render();drawAll();}
function firstConfirmed(p){return p.is_static===true&&p.visible_both===true&&p.sigma0_px!=null&&p.sigma1_px!=null&&!!p.landmark_type&&['high','medium'].includes(p.confidence);}
function selectPoint(p){selected=p.point_id;$('landmark').value=p.landmark_type||'';$('role').value=p.role||'check';$('note').value=reviewing?'':p.note||'';unknownFields();$('conclusion').value='uncertain';if(reviewing){coordinateInputs(null,null);}else{coordinateInputs(p.xy0,p.xy1);$('static').checked=p.is_static===true;$('visible').checked=p.visible_both===true;$('sigma0').value=p.sigma0_px??'';$('sigma1').value=p.sigma1_px??'';$('confidence').value=p.confidence||'unknown';}render();drawAll();}
function render(){
 const n=doc?.points.length||0,q=doc?.gt_quality,z=q?.eligible_point_ids?.length||0;
 $('progress').textContent=`已保存${n}组 · 待核验${n-z}组 · 合格${z}组`;
 $('identity').textContent=operator?operator+' · v'+(doc?.revision||0):'未进入';
 $('pointLabel').textContent=selected?'当前点 '+selected:(reviewing?'请选择要独立复核的地物':'下一点 '+(doc?.next_point_id||'P0001'));
 $('clickState').textContent=(selected?'编辑 '+selected+' · ':'')+[0,1].map(i=>(i?'右图':'左图')+(values(i)?'已选':'未选')).join(' · ');
 const disabled=busy||loading||!operator||!doc;
 $('quickSave').disabled=disabled||!values(0)||!values(1);$('formalSave').disabled=disabled||!values(0)||!values(1);
 $('reviewSave').disabled=disabled||!selected||!values(0)||!values(1);$('undo').disabled=disabled||!undo.length;$('delete').disabled=disabled||!selected;
 for(const id of ['batch','task','reload','newPoint','finish','finishSave','confirmMode','mode','changeActor'])$(id).disabled=busy||loading;
 for(const id of ['geo','pseudo'])$(id).disabled=disabled;
 for(const id of ['pointStrip','queue']){
  const list=$(id);list.replaceChildren();
  for(const p of points){const b=document.createElement('button');b.className='point'+(p.point_id===selected?' selected':'');
   b.textContent=p.point_id+' · '+(p.landmark_type||'未说明地物')+' · '+(q?.eligible_point_ids?.includes(p.point_id)?'合格':(p.review?'已复核，见质量规则':(!reviewing&&firstConfirmed(p)?'待第二人复核':'待核验')));
   b.disabled=busy||loading;b.onclick=()=>{if(hasPending()&&!confirm('当前点尚未保存，放弃当前点击？'))return;selectPoint(p);};list.append(b);}
 }
 $('quality').textContent=reviewing?'原坐标与原不确定度已在服务器响应中隐藏。合格数仍由同一证据规则计算。':JSON.stringify(q,null,2);
}
function hasPending(){if(reviewing)return !!values(0)||!!values(1);if(!selected)return !!values(0)||!!values(1);const p=points.find(p=>p.point_id===selected);if(!p)return false;
 if(JSON.stringify([values(0),values(1)])!==JSON.stringify([p.xy0,p.xy1]))return true;
 return $('landmark').value!== (p.landmark_type||'')||$('note').value!==(p.note||'')||$('role').value!==p.role||$('static').checked!==(p.is_static===true)||$('visible').checked!==(p.visible_both===true)||$('sigma0').value!==String(p.sigma0_px??'')||$('sigma1').value!==String(p.sigma1_px??'')||$('confidence').value!==p.confidence;
}
function draw(i){const v=views[i],ctx=v.canvas.getContext('2d'),c=v.camera;ctx.clearRect(0,0,v.canvas.width,v.canvas.height);ctx.save();ctx.filter=`contrast(${$('contrast'+i).value}) brightness(${$('bright'+i).value})`;ctx.imageSmoothingEnabled=false;ctx.drawImage(v.image,c.tx,c.ty,v.image.width*c.scale,v.image.height*c.scale);ctx.restore();
 const mark=(p,label,color)=>{if(!p)return;const s=C.pixelToScreen(...p,c);ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineWidth=1.5;ctx.beginPath();ctx.arc(s[0],s[1],5,0,Math.PI*2);ctx.stroke();ctx.fillText(label,s[0]+7,s[1]-6);};
 if(!reviewing)points.forEach(p=>mark(p['xy'+i],p.point_id,p.point_id===selected?'#ffe087':'#00d5ff'));
 assistance.forEach((p,j)=>mark(p['xy'+i],'辅助'+j,'#ffbe55'));
 mark(values(i),reviewing?'复核点击':(selected||doc?.next_point_id||'新点'),'#ee9aff');
}
function drawAll(){views.forEach((v,i)=>{if(v.image.complete&&v.image.naturalWidth)draw(i);});}
function loupe(i,p){const v=views[i],ctx=$('l'+i).getContext('2d');ctx.clearRect(0,0,128,128);ctx.save();ctx.imageSmoothingEnabled=false;ctx.filter=`contrast(${$('contrast'+i).value}) brightness(${$('bright'+i).value})`;ctx.drawImage(v.image,p[0]+.5-8,p[1]+.5-8,16,16,0,0,128,128);ctx.restore();ctx.strokeStyle='#ffdd73';ctx.beginPath();ctx.moveTo(0,64);ctx.lineTo(128,64);ctx.moveTo(64,0);ctx.lineTo(64,128);ctx.stroke();}
function handle(fn){return async(...args)=>{try{await fn(...args);}catch(e){message(e.message);$('finishError').textContent=e.message;}};}
async function load(index){
 if(busy||loading)return;if(hasPending()&&!confirm('当前点未保存，切换任务将放弃当前点击。继续？')){taskOptions(task.batch_id||'legacy');$('task').value=task.sample_id;return;}
 const next=tasks[(index+tasks.length)%tasks.length],seq=++sequence;loading=true;render();
 try{const saved=await api('/api/annotation?sample_id='+encodeURIComponent(next.sample_id));
  await Promise.all(views.map((v,i)=>new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>{if(seq===sequence){v.image=im;v.camera={scale:1,tx:0,ty:0};v.canvas.width=next['width'+i];v.canvas.height=next['height'+i];}resolve();};im.onerror=()=>reject(Error('原生图像读取失败'));im.src='/api/image?sample_id='+encodeURIComponent(next.sample_id)+'&side='+i;})));
  if(seq!==sequence)return;task=next;doc=saved;points=deep(doc.points);undo=[];assistance=[];taskOptions(task.batch_id||'legacy');$('task').value=task.sample_id;
  $('reviewLink').href='/?review=1&task='+encodeURIComponent(task.sample_id);
  $('metadata').textContent=`${task.sample_id} | ${task.sensor} | ${task.site_name}\n${task.reference_acquisition_id} → ${task.moving_acquisition_id}\n${task.reference_utc} → ${task.moving_utc}\n观测方向 ${task.reference_observation_direction||'unknown'} → ${task.moving_observation_direction||'unknown'}\n${task.visual_note||''}\n原生512坐标；新候选仍待独立确认。显示亮度/对比度不改变坐标。`;
  clearPoint();if(reviewing&&points.length)selectPoint(points.find(p=>!p.review)||points[0]);
  message(reviewing?'原点位不显示也不传入本页面。选择地物后独立重新点击。':'左图点击 → 右图点击 → Enter 保存候选。');
 }finally{loading=false;render();}
}
async function saveDocument(newPoints,taskFields={}){
 if(busy||loading)throw Error('正在保存或加载，请稍候');actor();const prior=deep(doc.points);busy=true;render();
 try{const result=await api('/api/save',{sample_id:task.sample_id,expected_revision:doc.revision,actor:operator,points:newPoints,disposition:doc.disposition,disposition_note:doc.disposition_note,evaluation_scope:doc.evaluation_scope,...taskFields});doc=result;points=deep(doc.points);return prior;}finally{busy=false;render();}
}
async function savePoint(formal=false){
 if(reviewing)throw Error('请使用独立复核保存');const xy0=values(0),xy1=values(1);if(!xy0||!xy1)throw Error('请在左、右图各点击一次');
 const nullable=id=>$(id).value===''?null:Number($(id).value);
 if(formal&&(!$('static').checked||!$('visible').checked||nullable('sigma0')===null||nullable('sigma1')===null||!['high','medium'].includes($('confidence').value)||!$('landmark').value.trim()))throw Error('本点正式确认需要具体地物、静态/可见确认、两图实际不确定度及中/高置信；不能默认合格。');
 const old=selected?points.find(p=>p.point_id===selected):null;
 const same=old&&JSON.stringify([old.xy0,old.xy1])===JSON.stringify([xy0,xy1]);
 const fields=($('mode').value==='advanced'||formal)?{is_static:$('static').checked,visible_both:$('visible').checked,sigma0_px:nullable('sigma0'),sigma1_px:nullable('sigma1'),confidence:$('confidence').value}:(same?{is_static:old.is_static,visible_both:old.visible_both,sigma0_px:old.sigma0_px,sigma1_px:old.sigma1_px,confidence:old.confidence}:{is_static:false,visible_both:false,sigma0_px:null,sigma1_px:null,confidence:'unknown'});
 const point={sample_id:task.sample_id,point_id:selected||doc.next_point_id,xy0,xy1,role:$('role').value,landmark_type:$('landmark').value.trim(),note:$('note').value.trim(),...fields};
 const list=deep(points),idx=list.findIndex(p=>p.point_id===point.point_id);if(idx<0)list.push(point);else list[idx]=point;
 const prior=await saveDocument(list);undo.push(prior);if(undo.length>100)undo.shift();clearPoint();
 if(formal){const next=points.find(p=>!firstConfirmed(p));if(next)selectPoint(next);message(`本点确认已保存 · v${doc.revision}。${next?'请逐点确认下一点。':'第一人确认已完成；仍需真实第二人独立复核。'}`);}else message(`已保存 ${point.point_id} · v${doc.revision}；下一点 ${doc.next_point_id}。候选不自动获得GT资格。`);
 render();
}
$('quickSave').onclick=handle(()=>savePoint());$('formalSave').onclick=handle(()=>savePoint(true));
$('reviewSave').onclick=handle(async()=>{
 if(!reviewing||!selected||!values(0)||!values(1))throw Error('先选地物描述，在两图独立重新点击');
 if($('sigma0').value===''||$('sigma1').value==='')throw Error('必须填写自己判断的两图定位不确定度');
 if($('conclusion').value==='confirmed'&&(!$('static').checked||!$('visible').checked))throw Error('确认对应必须逐点确认静态性和双图可见性');
 if(busy||loading)return;actor();busy=true;render();
 try{doc=await api('/api/review',{sample_id:task.sample_id,expected_revision:doc.revision,actor:operator,point_id:selected,xy0:values(0),xy1:values(1),sigma0_px:Number($('sigma0').value),sigma1_px:Number($('sigma1').value),conclusion:$('conclusion').value,static_confirmed:$('static').checked,visible_confirmed:$('visible').checked,note:$('note').value});points=deep(doc.points);clearPoint();const next=points.find(p=>!p.review);if(next)selectPoint(next);message('独立复核已保存 · v'+doc.revision+'；'+(next?'继续重新点击下一点。':'当前队列已复核；GT资格仍由原规则判断。'));}finally{busy=false;render();drawAll();}
});
$('undo').onclick=handle(async()=>{if(!undo.length)return;const prior=undo[undo.length-1];await saveDocument(prior);undo.pop();clearPoint();message('撤销已保存为新版本 v'+doc.revision+'；历史版本保留，恢复/修改点的旧复核不会伪造恢复。');render();});
$('delete').onclick=handle(async()=>{if(!selected)throw Error('先选一个已保存点');if(!confirm('删除 '+selected+'？旧版本保留，可撤销。'))return;const prior=await saveDocument(points.filter(p=>p.point_id!==selected));undo.push(prior);clearPoint();message('删除已保存 · v'+doc.revision);});
$('newPoint').onclick=()=>{clearPoint();message('当前未保存点击已清除，已保存点不变。');};
$('reload').onclick=handle(()=>load(tasks.findIndex(t=>t.sample_id===task.sample_id)));
$('task').onchange=handle(()=>load(tasks.findIndex(t=>t.sample_id===$('task').value)));
function taskOptions(batch){$('batch').value=batch;$('task').replaceChildren();for(const r of tasks.filter(t=>(t.batch_id||'legacy')===batch)){const o=document.createElement('option');o.value=r.sample_id;o.textContent=r.sample_id+' · '+r.site_name;$('task').append(o);}}
$('batch').onchange=handle(()=>load(tasks.findIndex(t=>(t.batch_id||'legacy')===$('batch').value)));
function setMode(mode){document.body.dataset.mode=mode;$('editor').hidden=mode==='minimal';$('editorTitle').textContent=mode==='confirmation'?'逐点正式确认':'高级标注（仍可保存未核验候选）';drawAll();}
$('mode').onchange=()=>setMode($('mode').value);
$('confirmMode').onclick=()=>{if(hasPending()&&!confirm('进入核验队列将放弃当前未保存输入，继续？'))return;setMode('confirmation');const next=points.find(p=>!firstConfirmed(p));if(next)selectPoint(next);};
$('closeConfirm').onclick=()=>{$('mode').value='minimal';setMode('minimal');clearPoint();};
$('finish').onclick=()=>{$('disposition').value=doc.disposition;$('scope').value=doc.evaluation_scope;$('dispositionNote').value=doc.disposition_note||'';$('finishError').textContent='';$('finishDialog').showModal();};
$('finishCancel').onclick=()=>$('finishDialog').close();
$('finishSave').onclick=handle(async()=>{await saveDocument(points,{disposition:$('disposition').value,evaluation_scope:$('scope').value,disposition_note:$('dispositionNote').value.trim()});$('finishDialog').close();message('任务判断已保存 · v'+doc.revision+'；GT '+doc.gt_quality.status);});
$('changeActor').onclick=()=>{$('actor').value=operator;$('identityDialog').showModal();};
$('begin').onclick=handle(()=>{const name=$('actor').value.trim();if(!name){$('actor').focus();throw Error('请输入本人的真实身份');}operator=name;$('identityDialog').close();render();});
$('actor').addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();$('begin').click();}});
$('identityDialog').addEventListener('cancel',e=>{if(!operator)e.preventDefault();});
for(const [id,source] of [['geo','geographic'],['pseudo','sift_candidates']])$(id).onclick=handle(async()=>{if(busy||loading)return;const r=await api('/api/assistance',{sample_id:task.sample_id,actor:actor(),source});assistance=r.points;drawAll();message(r.warning);});
$('hide').onclick=()=>{assistance=[];drawAll();message('辅助层已隐藏，已发生的暴露记录不会删除。');};
function applyCamera(i,c,e){views[i].camera=c;if($('sync').checked&&!e?.altKey)views[1-i].camera={...c};drawAll();}
views.forEach((v,i)=>{
 const local=e=>{const r=v.canvas.getBoundingClientRect();return C.clientToCanvas(e.clientX,e.clientY,r,v.canvas.width,v.canvas.height);};
 v.canvas.oncontextmenu=e=>e.preventDefault();v.canvas.onwheel=e=>{e.preventDefault();const p=local(e);applyCamera(i,C.zoomAt(v.camera,...p,e.deltaY<0?1.2:1/1.2),e);};
 v.canvas.onpointerdown=e=>{if(!task||busy||loading||!operator)return;const s=local(e);if(e.shiftKey||e.button===1||e.button===2){v.drag={position:s,camera:{...v.camera}};v.canvas.setPointerCapture(e.pointerId);}else{const p=C.screenToPixel(...s,v.camera);if(p[0]>=0&&p[0]<=task['width'+i]-1&&p[1]>=0&&p[1]<=task['height'+i]-1){$('x'+i).value=p[0].toFixed(3);$('y'+i).value=p[1].toFixed(3);if(!reviewing&&selected)unknownFields();render();drawAll();}}};
 v.canvas.onpointermove=e=>{const s=local(e);if(v.drag)applyCamera(i,{scale:v.drag.camera.scale,tx:v.drag.camera.tx+s[0]-v.drag.position[0],ty:v.drag.camera.ty+s[1]-v.drag.position[1]},e);const p=C.screenToPixel(...s,v.camera);$('hover'+i).textContent=`x${p[0].toFixed(1)} y${p[1].toFixed(1)} · ${v.camera.scale.toFixed(1)}×`;if(v.image.complete&&v.image.naturalWidth)loupe(i,p);};
 v.canvas.onpointerup=v.canvas.onpointercancel=()=>v.drag=null;
 for(const id of ['contrast','bright'])$(id+i).oninput=()=>draw(i);
 for(const id of ['x','y'])$(id+i).oninput=()=>{if(!reviewing)unknownFields();render();drawAll();};
});
document.querySelectorAll('[data-zoom]').forEach(b=>b.onclick=e=>{const [i,f]=b.dataset.zoom.split(',').map(Number);applyCamera(i,C.zoomAt(views[i].camera,views[i].canvas.width/2,views[i].canvas.height/2,f),e);});
document.querySelectorAll('[data-reset]').forEach(b=>b.onclick=e=>applyCamera(Number(b.dataset.reset),{scale:1,tx:0,ty:0},e));
$('resetAll').onclick=()=>{views.forEach(v=>v.camera={scale:1,tx:0,ty:0});drawAll();};
document.addEventListener('keydown',e=>{const typing=['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName);if(e.key==='Escape'&&!$('finishDialog').open&&!$('identityDialog').open){clearPoint();return;}if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();if(reviewing)$('reviewSave').click();else if(document.body.dataset.mode==='confirmation')$('formalSave').click();else $('quickSave').click();}else if(!typing&&e.key==='Enter'){e.preventDefault();if(reviewing)$('reviewSave').click();else if(document.body.dataset.mode==='confirmation')$('formalSave').click();else $('quickSave').click();}else if(!typing&&!reviewing&&(e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();$('undo').click();}else if(!typing&&!reviewing&&e.key==='Delete'){$('delete').click();}});
window.onbeforeunload=e=>{if(busy||hasPending()){e.preventDefault();e.returnValue='';}};
api('/api/tasks').then(async t=>{tasks=t.sort((a,b)=>(b.batch_default===true)-(a.batch_default===true)||(a.batch_priority??999)-(b.batch_priority??999));const batches=new Map();for(const r of tasks)batches.set(r.batch_id||'legacy',r.batch_label||'原有5对独立核验');for(const [value,label] of batches){const o=document.createElement('option');o.value=value;o.textContent=label;$('batch').append(o);}const sid=new URLSearchParams(location.search).get('task');await load(Math.max(0,tasks.findIndex(t=>t.sample_id===sid)));$('identityDialog').showModal();}).catch(e=>message(e.message));
