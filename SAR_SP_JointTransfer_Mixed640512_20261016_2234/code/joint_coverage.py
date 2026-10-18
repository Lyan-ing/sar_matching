"""Audit actual updates against the locked sampler; no images or GT are regenerated."""
from mixed_io import *
from collections import Counter

def main():
 rows=records('train');val=records('val');trainvis={v for r in rows for v in r['visible_source_members']};valvis={v for r in val for v in r['visible_source_members']};assert not trainvis&valvis
 allruns={};comparison=[]
 for name in ['t1','t2','t3','repeat']:
  folder=ROOT/'runs'/name;status=json.loads((folder/f'status_{TAG}.json').read_text());assert status['state']=='completed'
  config=json.loads(sorted(folder.glob('config_*.json'))[0].read_text());plan=[json.loads(s) for s in Path(config['sampler']).read_text().splitlines()];actual=[];metrics=[]
  for p in sorted(folder.glob('sampling_actual_*.jsonl')):actual.extend(json.loads(s) for s in p.read_text().splitlines() if s)
  for p in sorted(folder.glob('metrics_*.jsonl')):metrics.extend(json.loads(s) for s in p.read_text().splitlines() if s)
  assert len(actual)==len(metrics)==6000 and [r['step'] for r in actual]==list(range(1,6001))
  calls=Counter();updates=Counter();pointcounts=Counter();supervision=Counter();by_scene=Counter();by_size=Counter();by_angle=Counter();by_overlap=Counter();by_profile=Counter();positiveonly=0;raw_total=0;calibration_nonempty=0;prior_nonempty=0;calibration_cells=0;prior_cells=0
  for r,p in zip(actual,plan):
   assert r['active']==p['active'] and r['consumed']==p['consumed'];assert len(r['active'])==8
   calls.update(r['consumed']);updates.update(r['active'])
   for i,info in zip(r['active'],r['supervision']):
    calibration_nonempty+=int(info['calibration_cells']>0);prior_nonempty+=int(info['prior_cells']>0);calibration_cells+=info['calibration_cells'];prior_cells+=info['prior_cells']
    row=rows[i];assert info['active'] and info['points']>0;pointcounts[i]+=info['points'];raw_total+=info['raw_points'];positiveonly+=info['positive_only'];supervision[row['loss_role']]+=1;by_scene[row['source_scene']]+=1;by_size[row['width']]+=1;by_angle[row['angle_bin']]+=1;by_overlap[row['overlap_bin']]+=1;by_profile[row['enhancement_profile']]+=1
  assert len(calls)==4800 and len(updates)==2384 and sum(updates.values())==48000
  assert all(np.isfinite(r['gradient_norm']) and r['gradient_norm']>0 for r in metrics)
  report=dict(optimization_steps=6000,effective_pairs=48000,consumed_relationship_calls=sum(calls.values()),different_consumed_relationships=len(calls),different_gradient_relationships=len(updates),different_frozen_pair_PNG_files_with_gradient=2*len(updates),skipped_before_image_forward=sum(calls.values())-sum(updates.values()),center_source_files_with_gradient=len({rows[i]['source_path'] for i in updates}),visible_member_source_files_with_gradient=len({v for i in updates for v in rows[i]['visible_source_members']}),known_gradient_components=len({rows[i]['source_component_id'] for i in updates}),empty_gradient_update_steps=0,empty_positive_relationships_skipped=sum(calls.values())-sum(updates.values()),raw_anchor_instances=raw_total,anchor_instances_used=sum(pointcounts.values()),positive_only_point_instances=positiveonly,calibration_nonempty_pairs=calibration_nonempty,detector_prior_nonempty_pairs=prior_nonempty,calibration_cell_instances=calibration_cells,detector_prior_cell_instances=prior_cells,role_updates=dict(supervision),scene_updates=dict(by_scene),size_updates=dict(by_size),angle_updates=dict(by_angle),overlap_updates=dict(by_overlap),profile_updates=dict(by_profile),seed=config['seed'],sampler_sha=config['sampler_sha'],initialization_sha=config['initialization_sha'],code_sha=config['code_sha'],elapsed_seconds=status['elapsed_seconds'],peak_gpu_MiB=max(r['peak_memory_MiB'] for r in metrics),mean_optimizer_step_seconds=float(np.mean([r['seconds'] for r in metrics])),mean_preclip_gradient_norm=float(np.mean([r['gradient_norm'] for r in metrics])),gradient_clipping_fraction=float(np.mean([r['gradient_norm']>5 for r in metrics])),mean_gradient_scale=float(np.mean([min(1.,5/(r['gradient_norm']+1e-6)) for r in metrics])),last_100_descriptor_loss=float(np.mean([r['descriptor'] for r in metrics[-100:]])),first_100_descriptor_loss=float(np.mean([r['descriptor'] for r in metrics[:100]])))
  allruns[name]=report
  for i,row in enumerate(rows):comparison.append(dict(run=name,pair_id=row['pair_id'],center_source_file=row['source_path'],visible_member_source_files=row['visible_source_members'],component=row['source_component_id'],size=row['width'],scene=row['source_scene'],role=row['loss_role'],saved_anchors=row['structural_anchor_count'],consumed=calls[i],gradient_calls=updates[i],anchors_used=pointcounts[i]))
 assert all(allruns[x]['sampler_sha']==allruns['t1']['sampler_sha'] for x in ['t2','t3'])
 t=stamp();write(ROOT/f'ACTUAL_TRAINING_COVERAGE_{t}.json',dict(runs=allruns,visible_source_member_split_overlap=0,train_known_components=len({r['source_component_id'] for r in rows}),val_known_components=len({r['source_component_id'] for r in val}),training_and_validation_are_development_data=True));csvwrite(ROOT/f'ACTUAL_SOURCE_SAMPLING_{t}.csv',comparison);print('Actual6000step coverage and split audit passed for all four runs.')
if __name__=='__main__':main()
