"""Single-expert evidence, append-only persistence and exact native-offset reuse."""
import sys,json,os,hashlib,threading,copy,math,uuid,io
from pathlib import Path
from collections import defaultdict,Counter,OrderedDict
from functools import lru_cache
from datetime import datetime
import numpy as np,rasterio
from rasterio.windows import Window
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;DATA=BASE/'RealSAR_Multiview512_20261015_1534'
sys.path.insert(0,str(DATA/'code'))
from common import now,stamp,save,jsonl,table,rows,sha,display
from gt_qa import support as frozen_spatial_support

def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def load_config():return json.loads(sorted(ROOT.glob('CONFIG_*.json'))[-1].read_text())
def finite_xy(xy,w,h):
 if not isinstance(xy,list) or len(xy)!=2 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in xy):raise ValueError('点坐标必须是两个有限数字')
 if not(0<=xy[0]<=w-1 and 0<=xy[1]<=h-1):raise ValueError('点超出原生像素中心范围')
 return [float(x) for x in xy]
def local_to_native(frame,xy):return (np.asarray(xy,float)+frame['origin_xy']).tolist()
def native_to_local(frame,xy):return (np.asarray(xy,float)-frame['origin_xy']).tolist()
def point_hash(p):
 d={k:v for k,v in p.items() if k not in ['updated_at','created_at']}
 if p.get('acquisition_ids'):
  order=sorted(range(2),key=lambda i:p['acquisition_ids'][i])
  for key in ['acquisition_ids','xy','native_xy','sigma','inputs','frame_ids']:d[key]=[p[key][i] for i in order]
 return digest(d)

class Dataset:
 def __init__(self,config=None):
  self.config=config or load_config();d=json.loads(Path(self.config['task_index']).read_text());self.tasks={p['sample_id']:p for p in d['tasks']};self.order=list(self.tasks);self.frames=d['frames'];self.pilot=d['pilot30'];self.protocol=json.loads(Path(self.config['protocol']).read_text());self.relations=defaultdict(list)
  if self.config.get('extra_native_index'):
   extra=json.loads(Path(self.config['extra_native_index']).read_text())
   for fid,f in extra['frames'].items():
    if fid in self.frames:raise ValueError('新增原图frame ID冲突，拒绝覆盖旧数据')
    self.frames[fid]=f
   for task in extra['tasks']:
    if task['sample_id'] in self.tasks:raise ValueError('新增原图任务ID冲突，拒绝覆盖旧任务')
    if not task.get('native_only') or task['split']!='research_unassigned':raise ValueError('新增来源只能是未分区原图观察')
    self.tasks[task['sample_id']]=task
   for sid in extra.get('existing_clear_controls',[]):self.tasks[sid]['clear_candidate']=True;self.tasks[sid]['same_direction_control']=True
   self.order=list(self.tasks)
  for p in self.tasks.values():self.relations[p['relation_id']].append(p)
 def scope(self,sid,mode):
  p=self.tasks[sid]
  if p.get('native_only') and mode!='native':raise ValueError('此新增来源只开放原图观察，尚未生成512或复标任务')
  if mode=='patch':return sid
  if mode=='native':return 'native:'+p['relation_id']
  if mode=='repeat':return 'repeat:'+sid
  raise ValueError('未知显示模式')
 def choose_frames(self,sid,mode,ids=None):
  p=self.tasks[sid];ids=ids or (p['native_frame_ids'] if mode=='native' else p['frame_ids'])
  if len(ids)!=2:raise ValueError('必须有两侧原生影像')
  for side,fid in enumerate(ids):
   f=self.frames[fid];crop=self.frames[p['frame_ids'][side]]
   if f['acquisition_id']!=p['acquisition_ids'][side] or f['grid_id']!=crop['grid_id']:raise ValueError('采集或原生产品网格不一致，禁止地理投影替代')
   if mode!='native' and fid!=p['frame_ids'][side]:raise ValueError('512模式只使用已冻结裁片')
  return list(ids)
 @lru_cache(maxsize=12000)
 def pixel_support(self,fid,x,y):
  f=self.frames[fid];xy=finite_xy([x,y],f['width'],f['height']);lo=np.floor(xy).astype(int);hi=np.ceil(xy).astype(int);w=Window(lo[0],lo[1],hi[0]-lo[0]+1,hi[1]-lo[1]+1)
  with rasterio.open(f['raw_path']) as ds:a=ds.read(1,window=w);valid=ds.read_masks(1,window=w)>0
  if f.get('valid_mask'):
   with Image.open(f['valid_mask']) as im:valid&=np.asarray(im.crop((lo[0],lo[1],hi[0]+1,hi[1]+1)))>0
  return bool(valid.all() and np.isfinite(a).all()),a
 def exact_reuse(self,source_frame,target_frame,native_xy):
  a,b=[self.frames[f] for f in [source_frame,target_frame]]
  if (a['acquisition_id'],a['grid_id'])!=(b['acquisition_id'],b['grid_id']):return None,'different_acquisition_or_product_grid'
  xy=native_to_local(b,native_xy);sx=native_to_local(a,native_xy)
  try:
   va,pa=self.pixel_support(source_frame,*sx);vb,pb=self.pixel_support(target_frame,*xy)
  except ValueError:return None,'outside_original_or_crop'
  if not va or not vb:return None,'invalid_native_mask_support'
  if pa.shape!=pb.shape or not np.array_equal(pa,pb,equal_nan=True):return None,'native_pixel_content_mismatch'
  return xy,None

class Store:
 def __init__(self,dataset,root=None):
  self.data=dataset;self.root=Path(root or dataset.config['observations']);self.root.mkdir(exist_ok=True,parents=True);self.lock=threading.RLock();self.docs={}
  for folder in self.root.iterdir():
   if folder.is_dir():
    files=sorted(folder.glob('v*.json'))
    if files:
     d=json.loads(files[-1].read_text());h=d.pop('content_sha256');assert digest(d)==h;d['content_sha256']=h;self.docs[d['scope']]=d
 def empty(self,sid,mode):return dict(scope=self.data.scope(sid,mode),sample_id=sid,mode=mode,relation_id=self.data.tasks[sid]['relation_id'],revision=0,points=[],draft={},task_status='pending',decision_reason='',frame_ids=self.data.choose_frames(sid,mode),GT_qualified=False,operator='user_single_expert',evidence_type='single_expert_not_two_person_GT',assistance=dict(metadata=True,AI_candidates=True,geographic_predictions=False))
 def get(self,sid,mode):return copy.deepcopy(self.docs.get(self.data.scope(sid,mode),self.empty(sid,mode)))
 def orient(self,p,targets):
  order=[];xy=[]
  for target in targets:
   cid=self.data.frames[target]['acquisition_id'];side=next((i for i,f in enumerate(p['frame_ids']) if self.data.frames[f]['acquisition_id']==cid),None)
   if side is None:return None,'different_acquisition'
   loc,error=self.data.exact_reuse(p['frame_ids'][side],target,p['native_xy'][side])
   if error:return None,error
   order.append(side);xy.append(loc)
  out=dict(p,xy=xy,source_side_order=order)
  for k in ['native_xy','sigma','inputs','frame_ids','acquisition_ids']:
   if k in p:out[k]=[p[k][i] for i in order]
  return out,None
 def original_points(self):
  for doc in list(self.docs.values()):
   if doc['mode']!='repeat':
    for p in doc['points']:yield doc,p
 def anchor(self,ref):
  doc=self.docs[ref['scope']];p=next(p for p in doc['points'] if p['point_id']==ref['point_id'])
  if doc['mode']=='repeat' or point_hash(p)!=ref['point_hash']:raise ValueError('原单侧点已修改，请重新选择或人工点击')
  side=next((i for i,a in enumerate(p['inputs']) if a['input_id']==ref.get('input_id')),None) if ref.get('input_id') else int(ref['side'])
  if side is None:raise ValueError('原单侧点击已被修改，请重新选择')
  return doc,p,side
 def physical_reasons(self,p):
  fail=[]
  if p.get('physical_confirmed') is not True:fail.append('未明确确认同一固定地面结构/双侧可见')
  for k in ['structure_type','structure_group','reason']:
   if not p.get(k,'').strip():fail.append('缺少'+k)
  if p.get('confidence') not in ['medium','high']:fail.append('置信度不足或未知')
  if p.get('risk_status')!='clear':fail.append('高度/阴影/叠掩/动态/散射风险未排除')
  if p.get('structure_type')=='有高度/动态风险的观察':fail.append('风险目标类型不能作为已确认固定地面GT')
  sig=p.get('sigma',[])
  if len(sig)!=2 or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) or x<0 for x in sig):fail.append('缺少实际两侧定位不确定度')
  for a in p.get('inputs',[]):
   if a.get('kind')=='reused_single_side':
    try:self.anchor(a['reference'])
    except (KeyError,StopIteration,ValueError):fail.append('复用单侧原始记录已变化，需重新确认')
  return fail
 def normalize_point(self,p,old,sid,mode):
  p=copy.deepcopy(p)
  if p.get('frame_ids') and len(p['frame_ids'])==2:
   ids=[self.data.frames[f]['acquisition_id'] for f in p['frame_ids']];wanted=self.data.tasks[sid]['acquisition_ids']
   if ids==list(reversed(wanted)):
    for k in ['frame_ids','xy','sigma','inputs']:p[k]=list(reversed(p[k]))
  out={k:copy.deepcopy(p.get(k)) for k in ['point_id','structure_type','structure_group','reason','confidence','sigma','physical_confirmed','risk_status','risk_note','planar','role','inputs','repeat_target']}
  if not isinstance(out['point_id'],str) or not 1<=len(out['point_id'])<=100:raise ValueError('无效点ID')
  for k in ['structure_type','structure_group','reason','risk_note']:
   out[k]=out.get(k) or ''
   if not isinstance(out[k],str) or len(out[k])>2000:raise ValueError('结构说明应为简短文字')
  if out['role'] not in ['check','control']:raise ValueError('未知点角色')
  out['frame_ids']=self.data.choose_frames(sid,mode,p.get('frame_ids'));out['acquisition_ids']=[self.data.frames[f]['acquisition_id'] for f in out['frame_ids']];out['xy']=[];out['native_xy']=[]
  if not isinstance(p.get('xy'),list) or len(p['xy'])!=2:raise ValueError('对应必须由你确认两侧位置')
  for side,fid in enumerate(out['frame_ids']):
   f=self.data.frames[fid];xy=finite_xy(p['xy'][side],f['width'],f['height']);valid,_=self.data.pixel_support(fid,*xy)
   if not valid:raise ValueError('点击位于原生无效区域；草稿保留，请重新定位')
   out['xy'].append(xy);out['native_xy'].append(local_to_native(f,xy))
  if not isinstance(out['inputs'],list) or len(out['inputs'])!=2:raise ValueError('需要保留两侧点击或单侧复用来源')
  for side,a in enumerate(out['inputs']):
   if a.get('kind') not in ['direct_click','reused_single_side'] or not isinstance(a.get('input_id'),str):raise ValueError('缺少实际输入来源')
   if a['kind']=='reused_single_side':
    _,origin,oside=self.anchor(a['reference']);of=self.data.frames[origin['frame_ids'][oside]];f=self.data.frames[out['frame_ids'][side]]
    if (of['acquisition_id'],of['grid_id'])!=(f['acquisition_id'],f['grid_id']) or not np.allclose(origin['native_xy'][oside],out['native_xy'][side],atol=1e-8,rtol=0) or a['input_id']!=origin['inputs'][oside]['input_id']:raise ValueError('复用坐标或采集身份不一致')
  if out['physical_confirmed'] not in [True,False] or out['planar'] not in ['yes','no','unknown'] or out['risk_status'] not in ['clear','risk','unknown']:raise ValueError('物理判断必须明确记录')
  if out['confidence'] not in ['unknown','low','medium','high']:raise ValueError('未知置信度')
  if not isinstance(out['sigma'],list) or len(out['sigma'])!=2:raise ValueError('不确定度格式无效')
  if any(x is not None and (isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0) for x in out['sigma']):raise ValueError('不确定度必须非负有限值或保留未知')
  out.update(operator='user_single_expert',GT_qualified=False,independent_review=False,created_at=old.get('created_at',now()) if old else now(),updated_at=now())
  return out
 def commit(self,p):
  with self.lock:
   sid=p['sample_id'];mode=p['mode'];doc=self.get(sid,mode)
   if not isinstance(p.get('client_save_id'),str) or not p['client_save_id']:raise ValueError('缺少保存事务ID')
   if p['client_save_id']==doc.get('client_save_id'):return doc
   if p.get('expected_revision')!=doc['revision']:raise ValueError('版本冲突：当前草稿已保存在浏览器，请重新加载后核对，未覆盖磁盘记录')
   status=p.get('task_status','pending')
   if status not in ['pending','partial','complete','unjudgeable']:raise ValueError('未知任务状态')
   reason=p.get('decision_reason','')
   if not isinstance(reason,str) or len(reason)>2000:raise ValueError('任务判断理由格式无效')
   if status=='unjudgeable' and not reason.strip():raise ValueError('无法判断时请简述原因，避免把未处理误记为无对应')
   original={q['point_id']:q for q in doc['points']};incoming=p.get('points',[])
   if not isinstance(incoming,list) or len(incoming)>500:raise ValueError('每任务最多500点')
   normalized=[self.normalize_point(q,original.get(q['point_id']),sid,mode) for q in incoming]
   if len({q['point_id'] for q in normalized})!=len(normalized):raise ValueError('重复点ID')
   draft=p.get('draft',{})
   if not isinstance(draft,dict) or len(json.dumps(draft))>50000:raise ValueError('草稿格式无效')
   record=dict(doc,sample_id=sid,points=normalized,draft=draft,frame_ids=self.data.choose_frames(sid,mode,p.get('frame_ids')),task_status=status,decision_reason=reason.strip(),revision=doc['revision']+1,recorded_at=now(),client_save_id=p['client_save_id'],parent_sha256=doc.get('content_sha256'))
   record.pop('content_sha256',None)
   if mode=='repeat':
    # Snapshot targets server-side; first coordinates are never returned by the repeat API.
    targets={x['repeat_token']:x for x in self.repeat_baselines(sid)};prior=doc.get('repeat_baselines',{})
    for q in normalized:
     token=q.get('repeat_target')
     if token not in prior:
      if token not in targets:raise ValueError('原点已更新或缺少复标目标，请重新进入延迟复标')
      prior[token]=targets[token]
    record['repeat_baselines']=prior
   record['content_sha256']=digest(record);folder=self.root/digest(record['scope'])[:24];folder.mkdir(exist_ok=True)
   dest=folder/f'v{record["revision"]:08d}_{stamp()}.json';tmp=folder/('.pending_'+uuid.uuid4().hex)
   with tmp.open('x') as f:json.dump(record,f,ensure_ascii=False,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())
   try:os.link(tmp,dest)
   finally:tmp.unlink()
   fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
   try:os.fsync(fd)
   finally:os.close(fd)
   # Immediate readback: never report success on write failure.
   written=json.loads(dest.read_text());assert written==record;self.docs[record['scope']]=record
   return copy.deepcopy(record)
 def projected(self,task):
  result=[];failures=[]
  for doc,p in self.original_points():
   if doc['relation_id']!=task['relation_id']:continue
   oriented,error=self.orient(p,task['frame_ids'])
   if error:
    if error!='outside_original_or_crop':failures.append(dict(source_scope=doc['scope'],point_id=p['point_id'],reasons=[error]))
    continue
   d=dict(oriented,xy0=oriented['xy'][0],xy1=oriented['xy'][1],physical_structure_group=p['structure_group'],source_scope=doc['scope'],source_revision=doc['revision'],origin_observation_id=doc['scope']+'/'+p['point_id'],source_point_hash=point_hash(p),physical_failures=self.physical_reasons(p),projection='exact_same_acquisition_product_grid_offset_plus_raw_pixel_support_check')
   result.append(d)
  return result,failures
 def repeat_baselines(self,sid):
  ps,_=self.projected(self.data.tasks[sid]);out=[]
  for p in ps:
   if p['physical_failures']:continue
   token=digest([p['origin_observation_id'],p['source_point_hash']])[:24]
   out.append(dict(p,repeat_token=token,baseline_point_created_at=p['created_at']))
  return out
 def repeat_evidence(self,sid,point):
  result=[];d=self.docs.get('repeat:'+sid)
  if not d:return result
  for p in d['points']:
   b=d.get('repeat_baselines',{}).get(p.get('repeat_target'))
   if not b or b['origin_observation_id']!=point['origin_observation_id']:continue
   fail=self.physical_reasons(p)
   if b['source_point_hash']!=point['source_point_hash']:fail.append('original_point_changed_after_repeat')
   delta=[float(np.linalg.norm(np.array(b['native_xy'][i])-p['native_xy'][i])) for i in [0,1]]
   hours=(datetime.fromisoformat(p['created_at'])-datetime.fromisoformat(b['baseline_point_created_at'])).total_seconds()/3600
   result.append(dict(repeat_scope=d['scope'],repeat_point_id=p['point_id'],repeat_revision=d['revision'],native_xy=p['native_xy'],sigma=p['sigma'],disagreement_px=delta,elapsed_hours=hours,delayed=hours>=self.data.protocol['delayed_repeat']['minimum_hours'],first_coordinates_hidden=True,same_expert=True,independent_two_person=False,failures=fail))
  return result
 def assessment(self,task):
  projected,failures=self.projected(task);eligible=[];duplicates=[]
  for p in projected:
   p['repeatability']=self.repeat_evidence(task['sample_id'],p)
   sig=list(p['sigma'])
   for rep in p['repeatability']:
    if not rep['failures'] and all(x is not None for x in sig):sig=[max(sig[i],rep['sigma'][i],rep['disagreement_px'][i]) for i in [0,1]]
   p['combined_uncertainty_px']=math.hypot(*sig) if all(x is not None for x in sig) else None
   if p['physical_failures']:continue
   if any(min(np.linalg.norm(np.array(p['xy0'])-q['xy0']),np.linalg.norm(np.array(p['xy1'])-q['xy1']))<.5 for q in eligible):duplicates.append(p['origin_observation_id']);continue
   eligible.append(p)
  planar=[p for p in eligible if p['planar']=='yes'];views=[dict(native_info=dict(width=512,height=512))]*2
  if task.get('native_only'):primary=local=dict(passed=False,reasons=['尚无冻结512窗口；原图观察不授予512几何评价资格'])
  else:primary=frozen_spatial_support(planar,views,'primary_patch_sparse');local=frozen_spatial_support(planar,views,'secondary_local_sparse')
  doc=self.get(task['sample_id'],'native' if task.get('native_only') else 'patch')
  level='full' if primary['passed'] else 'local' if local['passed'] else 'point' if eligible else 'no_reliable' if doc['task_status']=='unjudgeable' else 'pending'
  warnings=[]
  for p in projected:
   for rep in p['repeatability']:
    if not rep['failures'] and max(rep['disagreement_px'])>3:warnings.append(p['point_id']+'：本人复标位置差异>3px，请核查结构；不确定度已按真实差异增大')
  if doc['task_status']=='unjudgeable' and eligible:warnings.append('无法判断与已确认点冲突；点保留，但不授予局部/全幅资格');level='point'
  checks=[p for p in planar if p['role']=='check'];u=max((p['combined_uncertainty_px'] for p in checks),default=None);thresholds=[v for v in [3,5,10] if u is not None and u<=v/3 and level in ['local','full']]
  crop_views=[{k:f[k] for k in ['frame_id','acquisition_id','grid_id','origin_xy','width','height','raw_path','display','native_info']} for f in (self.data.frames[fid] for fid in task['frame_ids'])]
  point_checks=[dict(point_id=p['point_id'],origin_observation_id=p['origin_observation_id'],failures=p['physical_failures'],combined_uncertainty_px=p['combined_uncertainty_px'],planar=p['planar'],duplicate_not_counted=p['origin_observation_id'] in duplicates) for p in projected]
  return dict(sample_id=task['sample_id'],relation_id=task['relation_id'],component=task['component'],split=task['split'],acquisition_ids=task['acquisition_ids'],crop_views=crop_views,source_task_index=self.data.config['task_index'],task_status=doc['task_status'],processed=bool(doc['revision'] or projected),decision_reason=doc['decision_reason'],expert_gt_level=level,points=projected,point_checks=point_checks,eligible_point_ids=[p['origin_observation_id'] for p in eligible],eligible_points=len(eligible),primary=primary,local=local,supported_success_thresholds_px=thresholds,single_expert_geometry_qualified=level in ['local','full'],GT_qualified=False,independent_two_person_GT=False,source_projection_failures=failures,duplicate_observations_not_counted=duplicates,warnings=warnings,evidence_type='single_expert_user_physical_confirmation',full_image_dense_GT=False)
 def bank(self,frame_ids):
  bank=[[],[]];seen=[set(),set()]
  for d,p in self.original_points():
   if self.physical_reasons(p):continue
   for side,target in enumerate(frame_ids):
    for s,source in enumerate(p['frame_ids']):
     xy,error=self.data.exact_reuse(source,target,p['native_xy'][s])
     if error:continue
     key=p['inputs'][s]['input_id']
     if key in seen[side]:continue
     seen[side].add(key);bank[side].append(dict(xy=xy,native_xy=p['native_xy'][s],structure_type=p['structure_type'],structure_group=p['structure_group'],reason=p['reason'],input_id=key,reference=dict(scope=d['scope'],point_id=p['point_id'],side=s,input_id=key,point_hash=point_hash(p))))
  return bank
 def scene(self,sid,mode,frame_ids=None):
  doc=self.get(sid,mode);task=self.data.tasks[sid];derived=[];targets=[]
  requested=frame_ids or doc.get('frame_ids')
  if requested and [self.data.frames[f]['acquisition_id'] for f in requested]!=task['acquisition_ids']:requested=None
  ids=self.data.choose_frames(sid,mode,requested);doc['frame_ids']=ids
  draft=doc.get('draft',{});oldframes=draft.get('frame_ids')
  if oldframes and oldframes!=ids:
   coords=draft.get('xy',[None,None]);converted=[];converted_inputs=[]
   for i,target in enumerate(ids):
    cid=self.data.frames[target]['acquisition_id'];side=next((j for j,fid in enumerate(oldframes) if self.data.frames[fid]['acquisition_id']==cid),None);loc=None
    if side is not None and coords[side] is not None:
     native=local_to_native(self.data.frames[oldframes[side]],coords[side]);loc,_=self.data.exact_reuse(oldframes[side],target,native)
    converted.append(loc);converted_inputs.append(draft.get('inputs',[None,None])[side] if side is not None and loc else None)
   draft.update(xy=converted,inputs=converted_inputs,frame_ids=ids);draft.pop('cameras',None)
  if mode=='repeat':
   baseline=self.repeat_baselines(sid)
   # No original coordinates, uncertainties, cameras or bank in the response.
   targets=[dict(repeat_token=p['repeat_token'],structure_type=p['structure_type'],structure_group=p['structure_group'],reason=p['reason']) for p in baseline]
   doc.pop('repeat_baselines',None);bank=[[],[]]
  else:
   bank=self.bank(ids)
   for source,p in self.original_points():
    if source['relation_id']!=task['relation_id'] or source['scope']==doc['scope']:continue
    oriented,error=self.orient(p,ids)
    if not error:derived.append(dict(point_id=p['point_id'],xy=oriented['xy'],source_scope=source['scope'],source_sample_id=source['sample_id'],source_mode=source['mode'],structure_type=p['structure_type'],origin_observation_id=source['scope']+'/'+p['point_id']))
  # Existing own points may originate in another context of the same acquisition.
  visible=[]
  for p in doc['points']:
   oriented,error=self.orient(p,ids)
   if not error:visible.append(dict(oriented,display_frame_ids=ids))
  options=[]
  for side,fid in enumerate(ids):
   f=self.data.frames[fid];options.append([dict(frame_id=z['frame_id'],width=z['width'],height=z['height'],origin_xy=z['origin_xy']) for z in self.data.frames.values() if z['acquisition_id']==f['acquisition_id'] and z['grid_id']==f['grid_id'] and (z['large_context'] or z['frame_id']==task['frame_ids'][side])])
  qa=self.assessment(task) if mode!='repeat' else None
  if qa:qa={k:v for k,v in qa.items() if k not in ['points']}
  return dict(task=task,document=doc,visible_points=visible,inherited_points=derived,frames=[self.data.frames[f] for f in ids],frame_options=options,bank=bank,repeat_targets=targets,quality=qa)
 def export(self):
  with self.lock:
   export_root=Path(self.data.config['exports']);export_root.mkdir(exist_ok=True);serial=len(list(export_root.glob('export_*')))+1;t=stamp();folder=export_root/f'export_{serial:06d}_{t}';folder.mkdir();suffix=f'v{serial:06d}_{t}'
   # Native-only observations await point review and a separate crop preparation step.
   manifests=[self.assessment(p) for p in self.data.tasks.values() if not p.get('native_only')];csvrows=[]
   for split in ['Train','Val']:
    z=[r for r in manifests if r['split']==split];levels=Counter(r['expert_gt_level'] for r in z)
    good=[r for r in z if r['eligible_points']];csvrows.append(dict(split=split,total_pairs=len(z),pending=levels['pending'],point_level=levels['point'],local_GT=levels['local'],full_GT=levels['full'],no_reliable=levels['no_reliable'],reviewed_or_with_reused_points=sum(r['processed'] for r in z),pending_quality_after_review=sum(r['processed'] and r['expert_gt_level']=='pending' for r in z),annotation_complete=sum(r['task_status']=='complete' for r in z),annotation_partial=sum(r['task_status']=='partial' for r in z),annotation_unjudgeable=sum(r['task_status']=='unjudgeable' for r in z),physical_points_unique=len({p['origin_observation_id'] for r in z for p in r['points']}),derived_crop_point_records=sum(len(r['points']) for r in z),expert_eligible_points=sum(r['eligible_points'] for r in z),expert_point_relations=len({r['relation_id'] for r in good}),expert_point_source_components=len({r['component'] for r in good}),old_two_person_GT_qualified=0,relations=len({r['relation_id'] for r in z}),source_components=len({r['component'] for r in z})))
   manifest=folder/f'SINGLE_EXPERT_GT_MANIFEST_{suffix}.jsonl';jsonl(manifest,manifests)
   for split in ['Train','Val']:jsonl(folder/f'SINGLE_EXPERT_{split}_{suffix}.jsonl',[r for r in manifests if r['split']==split])
   originals=[dict(source_scope=d['scope'],source_revision=d['revision'],relation_id=d['relation_id'],source_frames=[self.data.frames[f] for f in p['frame_ids']],point=p) for d,p in self.original_points()]
   jsonl(folder/f'ORIGINAL_EXPERT_POINTS_{suffix}.jsonl',originals)
   jsonl(folder/f'ORIGINAL_DELAYED_REPEATS_{suffix}.jsonl',[d for d in self.docs.values() if d['mode']=='repeat'])
   coverage=folder/f'GT_COVERAGE_{suffix}.csv';table(coverage,csvrows,list(csvrows[0]))
   stats=dict(at=now(),single_expert=True,actual_original_correspondences=len(originals),actual_distinct_side_inputs=len({a['input_id'] for o in originals for a in o['point']['inputs']}),derived_crop_correspondence_records=sum(len(r['points']) for r in manifests),GT_qualified_two_person=0,split=csvrows,manifest=str(manifest),coverage=str(coverage),original_points=str(folder/f'ORIGINAL_EXPERT_POINTS_{suffix}.jsonl'),protocol=self.data.config['protocol'],coordinate_reuse_not_new_clicks=True)
   metadata=folder/f'EXPORT_{suffix}.json';save(metadata,stats)
   report=folder/f'GT_COVERAGE_{suffix}.md'
   with report.open('x') as f:
    f.write('# 单人专家GT覆盖\n\n'+now()+'\n\n此统计只描述单人专家核验，原双人GT_qualified始终为false。自动审核检查坐标/字段/风险声明与空间支持，不能独立证明物理身份。\n\n|分区|总数|待标|点级|局部|全幅|无可靠|\n|---|---:|---:|---:|---:|---:|---:|\n')
    for z in csvrows:f.write('|'+ '|'.join(str(z[k]) for k in ['split','total_pairs','pending','point_level','local_GT','full_GT','no_reliable'])+'|\n')
    f.write(f'\n原始人工对应数{len(originals)}，派生裁片对应记录{stats["derived_crop_correspondence_records"]}。一次Native点投到多个裁片不增加原始人工标注数。完整清单：{manifest.name}。\n')
   return dict(stats,metadata=str(metadata),report=str(report))

class Renderer:
 def __init__(self,data):
  self.data=data;self.lock=threading.RLock();self.arrays=OrderedDict();self.overviews=OrderedDict();self.limits={};self.cache={}
  cache=Path(data.config['cache_source'])
  if cache.exists():self.cache={v['source_png']:v['array'] for v in json.loads(cache.read_text())['assets'].values()}
 def convert(self,f,a,valid):
  if f['frame_id'] not in self.limits:
   im,settings=display(a,valid,units=f['units']);self.limits[f['frame_id']]=settings;return im
  z=self.limits[f['frame_id']];scale=20 if f['units']=='linear_amplitude' else 10
  val=scale*np.log10(np.maximum(np.asarray(a,dtype=np.float32),z.get('zero_display_floor',1e-8)));lo,hi=z['limits'];im=np.clip((val-lo)/max(1e-8,hi-lo)*255,0,255).astype(np.uint8);im[~valid]=0;return im
 def array(self,f):
  key=f['frame_id']
  if key in self.arrays:self.arrays.move_to_end(key);return self.arrays[key]
  if not f.get('display'):return None
  path=f['display']
  if path in self.cache:a=np.load(self.cache[path],mmap_mode='r',allow_pickle=False)
  else:
   with Image.open(path) as im:a=np.asarray(im.convert('L'))
  assert a.shape==(f['height'],f['width'])
  self.arrays[key]=a
  while len(self.arrays)>2:self.arrays.popitem(last=False)
  return a
 def png(self,fid,box=None):
  with self.lock:
   f=self.data.frames[fid];a=self.array(f)
   if box is None and fid in self.overviews:self.overviews.move_to_end(fid);return self.overviews[fid]
   if box is None:
    if a is not None:im=Image.fromarray(a);im.thumbnail((1024,1024),Image.Resampling.NEAREST)
    else:
     scale=min(1,1024/max(f['width'],f['height']));shape=(max(1,round(f['height']*scale)),max(1,round(f['width']*scale)))
     with rasterio.open(f['raw_path']) as ds:ar=ds.read(1,out_shape=shape,resampling=rasterio.enums.Resampling.nearest);valid=ds.read_masks(1,out_shape=shape)>0
     im=Image.fromarray(self.convert(f,ar,valid&np.isfinite(ar)))
   else:
    x,y,w,h=box
    if min(x,y)<0 or min(w,h)<=0 or max(w,h)>1024 or x+w>f['width'] or y+h>f['height']:raise ValueError('只允许有效范围内不超过1024的原生显示窗口')
    if a is not None:im=Image.fromarray(np.asarray(a[y:y+h,x:x+w]))
    else:
     if fid not in self.limits:self.png(fid)
     with rasterio.open(f['raw_path']) as ds:ar=ds.read(1,window=Window(*box));valid=ds.read_masks(1,window=Window(*box))>0
     im=Image.fromarray(self.convert(f,ar,valid&np.isfinite(ar)))
   b=io.BytesIO();im.save(b,format='PNG',compress_level=1);result=b.getvalue()
   if box is None:
    self.overviews[fid]=result
    while len(self.overviews)>64:self.overviews.popitem(last=False)
   return result
