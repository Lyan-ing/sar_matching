"""Reuse the offline HTML; immutable packages and actual-human return collection."""
from common import *
import argparse,shutil,zipfile

def packages(manifest,batch_size=25):
 t=stamp();rr=rows(manifest);rr=[r for r in rr if r['sample_id'] not in ['C001','C002','C003'] and r.get('qa_state')=='passed']
 # Human review priorities cover sites and angles; no algorithm results or GT used.
 groups={}
 for r in rr:groups.setdefault((r['site'],r['angle_bin']),[]).append(r)
 for g in groups.values():g.sort(key=lambda r:(r.get('annotatability') not in ['A','B'],r.get('annotatability')=='C',r['sample_id']))
 ordered=[]
 for tier in [{'A','B'},{'E'},{'C'},{'D'}]:
  pool={k:[r for r in g if r.get('annotatability','E') in tier] for k,g in groups.items()}
  while any(pool.values()):
   for k in sorted(pool):
    if pool[k]:ordered.append(pool[k].pop(0))
 source=FIXED/'blind_release';htmlsource=next(source.glob('blind_annotation*.html')).read_text();out=ROOT/f'annotation_{t}';out.mkdir()
 index=[];assignments=[]
 for batch_ix,start in enumerate(range(0,len(ordered),batch_size),1):
  batch=ordered[start:start+batch_size];bid=f'B{batch_ix:03d}_{t}';tasks=[];identities={};mapping=[]
  for ix,r in enumerate(batch,1):
   anon=f'T{batch_ix:03d}_{ix:03d}';task=dict(sample_id=anon,width0=512,height0=512,width1=512,height1=512,batch_id=bid,batch_label='原生SAR双图 · 独立采集',site_name='')
   for side in [0,1]:
    p=Path(r[f'image{side}']);assert p.suffix.lower()=='.png';name=f'I{ix:03d}_{side}_{t}.png';task[f'image{side}']=name
    identities[name]=dict(sha256=sha(p),width=512,height=512,pixel_convention='zero based native pixel centers')
   tasks.append(task);mapping.append(dict(task_id=anon,sample_id=r['sample_id'],relation_id=r['relation_id'],component=r['component'],image_names=[task[f'image{s}'] for s in [0,1]],image_sha256=[identities[task[f'image{s}']]['sha256'] for s in [0,1]]))
  for slot in ['A','B']:
   package_id=bid+'_'+slot;folder=out/package_id;folder.mkdir()
   for task,r in zip(tasks,batch):
    for side in [0,1]:shutil.copyfile(r[f'image{side}'],folder/task[f'image{side}'])
   for name in ['coordinates_20261013_2227.js','offline_capture_20261013_2227.js','blind_app_20261013_2227.js']:shutil.copyfile(source/name,folder/name)
   orderedtasks=list(tasks)
   if slot=='B':np.random.default_rng(20261015+batch_ix).shuffle(orderedtasks)
   payload='<script>window.BLIND_TASKS='+json.dumps(orderedtasks,ensure_ascii=False)+';window.BLIND_PACKAGE_ID='+json.dumps(package_id)+';window.BLIND_IMAGE_IDENTITIES='+json.dumps(identities)+';</script>'
   html=re.sub(r'<script>window.BLIND_TASKS=.*?</script>',lambda _:payload,htmlsource,count=1,flags=re.S)
   html=html.replace('Umbra Open Data — CC-BY-4.0。','来源授权已由管理员存档：Umbra/Capella CC-BY-4.0；既有Sentinel RTC沿用原Indigo Atlas许可，不推断为CC-BY。')
   p=folder/f'blind_annotation_{t}.html';p.write_text(html)
   instruction=f'''# 独立原生SAR标注 {package_id}

只把此目录交给本批实际操作者；不要接收管理员地理图、算法结果或另一人的坐标。A、B是分发槽位，不是假定的人员身份。每人填写自己的真实稳定ID、别名及全部暴露声明。曾看辅助提示者仍可提交辅助开发观察，但不能取得独立GT资格。

双击blind_annotation_{t}.html。左图点击→右图点击→“逐点填写结构依据”，填写具体低矮固定地面结构、静态/双图可见确认、两侧实际定位不确定度、中/高置信，并导出。每次导出JSON必须实际保存；刷新会清空页面内存。找不到可直接结束并写原因，不凑点。

同时填写POINT_EVIDENCE_TEMPLATE_{t}.csv：每个点的结构组、物理类型/理由、地面和二维平面适用判断，逐项记录高度、叠掩、阴影、动态及散射歧义。结构组不能用不同角点ID伪装成不同物理设施。仅收集“check”的主全幅资格仍要求>=6点、>=3结构组、各组<=2点、双方点距>=32px、凸包>=10%、两轴跨度>=50%、4网格/3象限；局部另按原协议，未达门槛保留少量真实点。

B在自己的干净包独立定位，不看A的坐标。可以独立寻找自己的点；返回后由真实管理员用物理文字依据关联两人的同一结构。不能仅按坐标接近自动认定同点。

回收：所有human_observations_*.json原版本、实际填写的POINT_EVIDENCE CSV、真实人员身份/暴露核查。只按Enter快存但未确认物理字段的点仍是待核验。此包不会自动授予GT资格。
'''
   (folder/f'ANNOTATOR_INSTRUCTIONS_{t}.md').write_text(instruction)
   fields=['package_id','task_id','point_id','structure_group','structure_type','physical_reason','ground_fixed','low_height_or_ground','planar_2d_applicable','height_risk','layover_risk','shadow_risk','dynamic_risk','scattering_ambiguity']
   table(folder/f'POINT_EVIDENCE_TEMPLATE_{t}.csv',[],fields)
   zip_path=out/f'{package_id}.zip'
   with zipfile.ZipFile(zip_path,'x',compression=zipfile.ZIP_DEFLATED) as z:
    for f in folder.iterdir():z.write(f,f.name)
   index.append(dict(package_id=package_id,batch_id=bid,slot=slot,task_count=len(batch),zip=str(zip_path),zip_sha256=sha(zip_path),html=str(p),tasks=mapping,person_id=None,status='awaiting_actual_people'))
   assignments.append(dict(package_id=package_id,slot=slot,person_id='',aliases='',identity_verified='',exposure_audited='',independent_work_verified='',verified_by='',verified_at=''))
 jsonl(out/f'PACKAGE_INDEX_{t}.jsonl',index);table(out/f'PERSONNEL_ASSIGNMENTS_TEMPLATE_{t}.csv',assignments,['package_id','slot','person_id','aliases','identity_verified','exposure_audited','independent_work_verified','verified_by','verified_at'])
 table(out/f'PHYSICAL_POINT_LINKS_TEMPLATE_{t}.csv',[],['sample_id','A_package_id','A_point_id','B_package_id','B_point_id','physical_identity_confirmed','structure_group','conflict_resolved','linked_by','linked_at'])
 save(out/f'PACKAGE_SUMMARY_{t}.json',dict(at=now(),image_pairs=len(ordered),batches=(len(ordered)+batch_size-1)//batch_size,packages=len(index),actual_people_assigned=0,real_submissions=0,GT=0,old_C001_C002_C003_packages_rebuilt=False))
 print('ANNOTATION_PACKAGES',len(index),'INDEX',out/f'PACKAGE_INDEX_{t}.jsonl',flush=True)

def collect(inputs,out):
 """Append-only preservation; never trusts a returned GT_qualified flag."""
 out=Path(out);out.mkdir(parents=True,exist_ok=False);report=[]
 for folder in inputs:
  for p in sorted(Path(folder).glob('*.json')):
   try:
    d=json.loads(p.read_text());assert isinstance(d.get('points'),list) and d.get('actor') and d.get('package_id')
    h=sha(p);dest=out/(p.stem+'_'+h[:12]+'_'+stamp()+'.json')
    with dest.open('xb') as f:f.write(p.read_bytes())
    report.append(dict(file=str(dest),sha256=h,actor=d['actor'],package_id=d['package_id'],task_id=d['sample_id'],revision=d.get('revision'),point_count=len(d['points']),status='actual_return_pending_identity_exposure_physical_review',GT_qualified=False))
   except Exception as e:report.append(dict(file=str(p),error=repr(e),status='rejected_input',GT_qualified=False))
 jsonl(out/f'RECEIPTS_{stamp()}.jsonl',report);print('RECEIVED',len(report),'No qualification without independent evidence',flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
 p=sub.add_parser('packages');p.add_argument('--manifest',required=True);p.add_argument('--batch-size',type=int,default=25)
 p=sub.add_parser('collect');p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--out',required=True)
 a=ap.parse_args()
 if a.command=='packages':packages(a.manifest,a.batch_size)
 else:collect(a.inputs,a.out)
