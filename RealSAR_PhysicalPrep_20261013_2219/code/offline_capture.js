/* Thin offline adapter for the existing Native annotation UI.
 * Captures actual human inputs only; never grants GT qualification.
 */
const localDocuments=new Map();
function minuteStamp(){return new Intl.DateTimeFormat('sv-SE',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}).format(new Date()).replace(/[- :]/g,'').replace(/^(\d{8})(\d{4})$/,'$1_$2');}
function declarations(){
 const ids=['priorGeo','priorAlgorithm','priorHints','priorOtherPoints'];
 const values=Object.fromEntries(ids.map(id=>[id,document.getElementById(id).value]));
 if(Object.values(values).includes('unknown'))throw Error('请逐项如实填写此前暴露；未知不能声明未暴露。');
 if(!document.getElementById('actualHuman').checked)throw Error('仅由实际操作者本人填写与点击。');
 return {...values,aliases:document.getElementById('aliases').value.trim(),declared_at:new Date().toISOString(),identity_verified_by_custodian:false};
}
function documentKey(sid){return operator+'|'+sid;}
function emptyDocument(sid){return {sample_id:sid,revision:0,points:[],next_point_id:'P0001',disposition:'pending',disposition_note:'',evaluation_scope:'local'};}
function visible(doc){return {...deep(doc),gt_quality:{status:'unverified',eligible_point_ids:[],eligible_checks:0,eligible_controls:0,formal_errors_allowed:false,note:'离线人工采集；身份、既往暴露及第二人尚未核验，非合格GT。'}};}
function download(doc){
 const blob=new Blob([JSON.stringify({...doc,GT_qualified:false,independent_review_passed:false,projected_points_copied_to_GT:false},null,2)],{type:'application/json;charset=utf-8'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;
 const safe=operator.replace(/[^\p{L}\p{N}_-]/gu,'_').slice(0,60);
 a.download=`human_observations_${doc.sample_id}_${safe}_v${String(doc.revision).padStart(6,'0')}_${minuteStamp()}.json`;
 a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function OfflineAPI(path,data){
 const u=new URL(path,'http://offline.invalid');
 if(u.pathname==='/api/tasks')return deep(window.BLIND_TASKS);
 if(u.pathname==='/api/annotation')return visible(localDocuments.get(documentKey(u.searchParams.get('sample_id')))||emptyDocument(u.searchParams.get('sample_id')));
 if(u.pathname!=='/api/save')throw Error('离线包不提供地理投影、算法提示或自审。第二人须独立打开自己的包。');
 const t=window.BLIND_TASKS.find(t=>t.sample_id===data.sample_id);if(!t)throw Error('未知任务');
 const exposure=declarations();
 if(data.actor!==operator||!operator.trim())throw Error('必须使用本人真实身份');
 const old=localDocuments.get(documentKey(data.sample_id))||emptyDocument(data.sample_id);
 if(old.revision!==data.expected_revision)throw Error('内存版本冲突，请重新加载');
 const assisted=['priorGeo','priorAlgorithm','priorHints','priorOtherPoints'].some(k=>exposure[k]==='yes');
 const allIDs=new Set();
 const points=data.points.map(p=>{
  if(allIDs.has(p.point_id))throw Error('重复点ID');allIDs.add(p.point_id);
  if(!['check','control'].includes(p.role))throw Error('未知点角色');
  for(const i of [0,1])if(!Array.isArray(p['xy'+i])||p['xy'+i].length!==2||!p['xy'+i].every(Number.isFinite)||p['xy'+i][0]<0||p['xy'+i][0]>t['width'+i]-1||p['xy'+i][1]<0||p['xy'+i][1]>t['height'+i]-1)throw Error('点击超出Native像素中心范围');
  const prior=old.points.find(z=>z.point_id===p.point_id);
  const same=prior&&JSON.stringify(prior.xy0)===JSON.stringify(p.xy0)&&JSON.stringify(prior.xy1)===JSON.stringify(p.xy1);
  return {...deep(p),annotator:operator,annotated_at:same?prior.annotated_at:new Date().toISOString(),used_assistance:assisted,exposure_declaration:exposure,review:null};
 });
 const n=Math.max(0,...points.map(p=>Number(p.point_id.replace(/^P/,''))||0));
 const saved={...old,...deep(data),points,revision:old.revision+1,next_point_id:'P'+String(n+1).padStart(4,'0'),actor:operator,recorded_at:new Date().toISOString(),exposure_declaration:exposure,
  stage:assisted?'assisted_human_observations':'human_blind_capture_pending_custodian_verification',
  package_id:window.BLIND_PACKAGE_ID,display_image_identities:window.BLIND_IMAGE_IDENTITIES,qualification:'unverified',independent_GT:false};
 // Download happens on actual user save. Nothing is sent to project servers.
 download(saved);localDocuments.set(documentKey(data.sample_id),saved);return visible(saved);
}
window.addEventListener('DOMContentLoaded',()=>{
 document.getElementById('identityDialog').addEventListener('close',()=>{
  if(typeof operator==='string'&&operator&&task){doc=visible(localDocuments.get(documentKey(task.sample_id))||emptyDocument(task.sample_id));points=deep(doc.points);undo=[];clearPoint();}
 });
 window.addEventListener('beforeunload',e=>{if(localDocuments.size){e.preventDefault();e.returnValue='请确认下载文件已保存；刷新会清除本页面内存。';}});
});
