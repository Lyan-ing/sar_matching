"""Read-only real-pixel checks; isolated empty-draft persistence, never human GT."""
from expert import *
import tempfile,subprocess
from unittest.mock import patch

def main():
 t=stamp();data=Dataset();store=Store(data);assert len(data.tasks)==1033 and not store.docs
 checks={};maxerror=0.;reuse_checks=0;invalid=0;different_acq=0
 for task in data.tasks.values():
  assert data.choose_frames(task['sample_id'],'patch')==task['frame_ids']
  assert len(set(task['acquisition_ids']))==2
  for fid in task['frame_ids']:
   f=data.frames[fid]
   for xy in [[0.,0.],[511.,511.],[127.25,311.75]]:
    err=float(np.max(np.abs(np.asarray(native_to_local(f,local_to_native(f,xy)))-xy)));maxerror=max(maxerror,err)
 assert maxerror<1e-8
 checks['all1033_crop_product_coordinate_roundtrips']=True
 sampled=list(data.tasks.values())[::16]+[data.tasks[s] for s in data.pilot]
 for task in sampled:
  for side,crop in enumerate(task['frame_ids']):
   f=data.frames[crop];q=[255.25,255.75];native=local_to_native(f,q);parent=task['native_frame_ids'][side]
   xy,reason=data.exact_reuse(parent,crop,native)
   if reason=='invalid_native_mask_support':invalid+=1;continue
   assert reason is None,(task['sample_id'],reason)
   assert np.allclose(xy,q,rtol=0,atol=1e-8);reuse_checks+=1
  a,b=task['frame_ids'];_,error=data.exact_reuse(a,b,local_to_native(data.frames[a],[255.,255.]));assert error=='different_acquisition_or_product_grid';different_acq+=1
 checks['real_native_parent_to_crop_mask_and_exact_raw_support']=True
 checks['different_acquisition_cannot_be_auto_mapped']=True
 # Order invariance is a hash/math check, not an annotation or physical GT.
 a=dict(point_id='math_only',acquisition_ids=['math_a','math_b'],xy=[[1.25,2.5],[3.5,4.75]],native_xy=[[101.25,202.5],[303.5,404.75]],sigma=[1,2],inputs=[dict(input_id='math_side_a'),dict(input_id='math_side_b')],frame_ids=['math_fa','math_fb'])
 b=copy.deepcopy(a)
 for key in ['acquisition_ids','xy','native_xy','sigma','inputs','frame_ids']:b[key]=list(reversed(b[key]))
 assert point_hash(a)==point_hash(b);checks['reversed_view_order_preserves_original_point_identity']=True
 from score_single_expert import point_errors
 assert np.allclose(point_errors(np.eye(3),[[1,2],[4,6]],[[1,2],[4,6]]),0)
 checks['separate_expert_scorer_uses_given_prediction_only_math_arrays']=True
 old=json.loads(Path(data.protocol['old_protocol_path']).read_text())
 for k in ['primary_patch_sparse','secondary_local_sparse']:assert data.protocol[k]==old[k]
 # Unit arrays are mathematical support probes, not stored annotation points.
 q=[[40,40],[470,40],[470,470],[40,470],[256,128],[128,256]]
 points=[dict(xy0=p,xy1=p,role='check',physical_structure_group=str(i//2)) for i,p in enumerate(q)]
 views=[dict(native_info=dict(width=512,height=512))]*2
 assert frozen_spatial_support(points,views,'primary_patch_sparse')['passed']
 assert not frozen_spatial_support(points[:5],views,'primary_patch_sparse')['passed']
 assert not frozen_spatial_support(points[:4],views,'secondary_local_sparse')['passed']
 checks['frozen_spatial_requirements_unchanged']=True
 sid=data.pilot[0]
 with tempfile.TemporaryDirectory(prefix='sar_expert_empty_draft_') as temporary:
  temp=Store(data,Path(temporary)/'records');p=dict(sample_id=sid,mode='patch',expected_revision=0,client_save_id=uuid.uuid4().hex,points=[],frame_ids=data.tasks[sid]['frame_ids'],draft=dict(test_only='automated empty draft; no pixel clicks or physical labels'),task_status='pending',decision_reason='')
  d=temp.commit(p);assert d['revision']==1 and d['points']==[]
  assert temp.commit(p)['revision']==1
  reloaded=Store(data,Path(temporary)/'records');assert reloaded.get(sid,'patch')['draft']==p['draft']
  try:reloaded.commit(dict(p,client_save_id=uuid.uuid4().hex));raise AssertionError('stale version accepted')
  except ValueError:pass
  with patch('os.link',side_effect=OSError('deliberate temporary empty-draft write-failure probe')):
   try:reloaded.commit(dict(p,expected_revision=1,client_save_id=uuid.uuid4().hex));raise AssertionError('failed disk write reported success')
   except OSError:pass
  assert reloaded.get(sid,'patch')['revision']==1 and Store(data,Path(temporary)/'records').get(sid,'patch')['revision']==1
 checks['isolated_empty_draft_fsync_reload_idempotency_conflict_and_write_failure']=True
 js=ROOT/'code/coordinates_20261013_2227.js'
 code="const C=require(process.argv[1]);let max=0;for(const c of [{scale:.0625,tx:-17,ty:83},{scale:1,tx:0,ty:0},{scale:9.7,tx:-731,ty:49}]){for(const p of [[0,0],[511,511],[127.25,311.75]]){const s=C.pixelToScreen(...p,c),q=C.screenToPixel(...s,c);max=Math.max(max,Math.abs(q[0]-p[0]),Math.abs(q[1]-p[1]));}const z=C.zoomAt(c,173.4,251.9,1.2),a=C.screenToPixel(173.4,251.9,c),b=C.screenToPixel(173.4,251.9,z);if(Math.hypot(a[0]-b[0],a[1]-b[1])>1e-8)throw Error('zoom drift');}if(max>1e-8)throw Error('coordinate drift');console.log(JSON.stringify({max_error:max}));"
 result=subprocess.check_output(['node','-e',code,str(js)],text=True);checks['reused_JS_pixel_centers_CSS_zoom_no_coordinate_drift']=True
 for path,h in json.loads(sorted(ROOT.glob('PROTECTED_INPUTS_*.json'))[-1].read_text()).items():assert sha(path)==h,path
 frozen=json.loads((BASE/'RealSAR_Fixed512_20261014_1743/release_20261014_1756/FREEZE_20261014_1810.json').read_text())
 for name,z in frozen['files'].items():assert sha(BASE/'RealSAR_Fixed512_20261014_1743/release_20261014_1756'/name)==z['sha256']
 checks['protected_source_manifests_protocol_synthV22_and36_fixed_files_unchanged']=True
 assert not store.docs and sum(1 for _ in store.original_points())==0
 out=ROOT/'evidence'/f'SELFCHECK_{t}.json';save(out,dict(at=now(),passed=True,checks=checks,roundtrip_max_px=maxerror,real_raw_support_checks=reuse_checks,invalid_mask_probes=invalid,different_acquisition_rejection_checks=different_acq,JS=json.loads(result),temporary_empty_draft_only=True,actual_human_points_created=0,production_annotation_files=0,GPU=False))
 print('SELF_CHECK_PASSED',out,'real raw checks',reuse_checks,flush=True)

if __name__=='__main__':main()
