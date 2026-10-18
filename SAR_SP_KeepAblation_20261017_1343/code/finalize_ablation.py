"""One completed control run versus frozen reference; no re-inference or threshold selection."""
from keep_train import *
from old_statistics import detailed
from collections import Counter,defaultdict
import csv
REF=FROZEN_ROOT
OFF='offshore_vessels_low_texture_sea'

def loadjsonl(p):return [json.loads(s) for s in p.read_text().splitlines()]
def main():
 t=stamp();folder=ROOT/'runs/formal';status=json.loads((folder/f'status_{TAG}.json').read_text());assert status['state']=='completed' and status['step']==6000
 weight=json.loads((ROOT/f'FORMAL_PROTOCOL_{TAG}.json').read_text());reference=json.loads((REF/f'FORMAL_PROTOCOL_{FROZEN_TAG}.json').read_text())
 differences={k:[reference.get(k),weight.get(k)] for k in reference if reference.get(k)!=weight.get(k)}
 assert set(differences)<= {'lambda_keep','code_sha'} and weight['lambda_keep']==0.,differences
 assert weight['sampler_sha']==reference['sampler_sha'] and weight['sampler']==reference['sampler']
 cand=folder/f'best_trained_{TAG}.ckpt';qualified=folder/f'best_qualified_{TAG}.ckpt';chosen=qualified if status.get('qualified') else cand
 ck=torch.load(chosen,map_location='cpu',weights_only=False);step=ck['global_step'];oldck=torch.load(REF/f'runs/formal/best_trained_{FROZEN_TAG}.ckpt',map_location='cpu',weights_only=False)
 initial=Model('b');changed=[k for k,v in ck['model_state'].items() if not torch.equal(v,initial.state_dict()[k])];assert set(changed)==set(weight['trainable_names'])
 original=json.loads((REF/f'final_scores_{FROZEN_TAG}.json').read_text());assert len(original)==9600
 for r in original:r['comparison_role']='frozen_reference'
 control=loadjsonl(ROOT/f'predictions/formal_step{step}/scores_{TAG}.jsonl');assert len(control)==2400
 for r in control:
  r.update(method='lambda0',comparison_role='new_control',frozen_prediction_path=str(ROOT/f'predictions/formal_step{step}/{r["pair_id"]}_keep_{r["protocol"]}_{TAG}.npz'))
 allrows=original+control;look={(r['pair_id'],r['method'],r['protocol']):r for r in allrows};assert len(look)==12000
 for r in control:
  a=look[r['pair_id'],'keep',r['protocol']];assert r['common_budget']==a['common_budget'] and [r['keypoints0'],r['keypoints1']]==[a['keypoints0'],a['keypoints1']]
 table=[];strata=[];paired=[]
 for method in ['m0','b','t2','lambda0','keep']:
  for mode in ['natural','equal']:
   rr=[r for r in allrows if r['method']==method and r['protocol']==mode];table.append(dict(method=method,protocol=mode,**detailed(rr)))
   for key in ['scene','overlap_bin','angle_bin','size']:
    for value in sorted({r[key] for r in rr},key=str):strata.append(dict(method=method,protocol=mode,axis=key,stratum=value,**detailed([r for r in rr if r[key]==value])))
 for mode in ['natural','equal']:
  for a in control:
   if a['protocol']!=mode:continue
   pid=a['pair_id'];b,d,k=[look[pid,m,mode] for m in ['b','t2','keep']]
   paired.append(dict(pair_id=pid,protocol=mode,component=a['component'],scene=a['scene'],geometry_eligible=a['geometry_eligible'],structure_eligible=a['structure_global_eligible'],baseline_success=b['Success@5px'],old_descriptor_success=d['Success@5px'],lambda0_success=a['Success@5px'],keep_success=k['Success@5px'],restored_by_keep=bool(k['Success@5px'] and not a['Success@5px']),lost_by_keep=bool(a['Success@5px'] and not k['Success@5px']),lambda0_wrong=bool(a['geometry_eligible'] and a['accepted'] and not a['Success@5px']),keep_wrong=bool(k['geometry_eligible'] and k['accepted'] and not k['Success@5px']),lambda0_static_matches=a['correct_structure_matches'],keep_static_matches=k['correct_structure_matches'],lambda0_error=a['bidirectional_error_px'],keep_error=k['bidirectional_error_px'],lambda0_reason=a['reason'],keep_reason=k['reason']))
 summaries=[];cis=[]
 for mode in ['natural','equal']:
  for label,predicate in [('all',lambda r:r['geometry_eligible']),('offshore',lambda r:r['geometry_eligible'] and r['scene']==OFF),('structure',lambda r:r['structure_eligible'])]:
   subset=[p for p in paired if p['protocol']==mode and predicate(p)];n=len(subset);k=sum(bool(r['keep_success']) for r in subset);a=sum(bool(r['lambda0_success']) for r in subset);b=sum(bool(r['baseline_success']) for r in subset)
   summaries.append(dict(protocol=mode,subset=label,n=n,baseline_success=b,lambda0_success=a,keep_success=k,delta_keep_vs_lambda0=(k-a)/n,delta_control_vs_initial=(a-b)/n,total_delta_keep_vs_initial=(k-b)/n,restored_by_keep=sum(r['restored_by_keep'] for r in subset),lost_by_keep=sum(r['lost_by_keep'] for r in subset),net_additional_success=k-a,keep_old_descriptor_loss_restored=sum(r['baseline_success'] and not r['old_descriptor_success'] and r['keep_success'] for r in subset),control_old_descriptor_loss_restored=sum(r['baseline_success'] and not r['old_descriptor_success'] and r['lambda0_success'] for r in subset)))
   groups=defaultdict(list)
   for r in subset:groups[r['component']].append(int(bool(r['keep_success']))-int(bool(r['lambda0_success'])))
   keys=sorted(groups);ss=np.array([sum(groups[g]) for g in keys]);nn=np.array([len(groups[g]) for g in keys]);ix=np.random.default_rng(42).integers(0,len(keys),(10000,len(keys)));bs=ss[ix].sum(1)/nn[ix].sum(1)
   cis.append(dict(protocol=mode,subset=label,known_components=len(keys),delta=(k-a)/n,lower=float(np.quantile(bs,.025)),upper=float(np.quantile(bs,.975)),exploratory_observed_dev_val_single_training_seed=True))
 # Same-step endpoint table supplements identical model-selection rule.
 endpoints=[]
 for st in [2000,4000,6000]:
  for meth,root,tag in [('lambda0',ROOT,TAG),('keep',REF,FROZEN_TAG)]:
   rr=loadjsonl(root/f'predictions/formal_step{st}/scores_{tag}.jsonl')
   for mode in ['natural','equal']:endpoints.append(dict(method=meth,step=st,protocol=mode,**detailed([r for r in rr if r['protocol']==mode])))
 # Verify every sampled relation/role/position count, not only plan hash.
 metrics=loadjsonl(folder/f'metrics_{TAG}.jsonl');usage=loadjsonl(folder/f'sampling_actual_{TAG}.jsonl');oldmet=loadjsonl(REF/f'runs/formal/metrics_{FROZEN_TAG}.jsonl');oldusage=loadjsonl(REF/f'runs/formal/sampling_actual_{FROZEN_TAG}.jsonl');assert len(metrics)==len(usage)==len(oldmet)==len(oldusage)==6000
 for a,b,m,n in zip(usage,oldusage,metrics,oldmet):
  assert a['step']==b['step']==m['step']==n['step'];assert a['ids']==b['ids'] and a['roles']==b['roles'] and a['pairs']==b['pairs'];assert m['points']==n['points'] and m['keep_positions']==n['keep_positions'];assert m['structure_pairs']==m['keep_pairs']==4 and m['gradient_keep']==m['weighted_keep_gradient']==0.
 coverage=dict(steps=6000,structure_pair_calls=24000,keep_stream_forward_calls=24000,keep_stream_gradient_calls=0,all_6000_actual_samples_roles_and_supervision_counts_identical_to_reference=True,first_step_descriptor_gradient_difference=metrics[0]['gradient_desc']-oldmet[0]['gradient_desc'],first_step_descriptor_loss_difference=metrics[0]['desc']-oldmet[0]['desc'],zero_data_gradient_steps=sum(m['preclip_gradient']<=1e-12 for m in metrics),nonfinite_steps=sum(not np.isfinite(m['preclip_gradient']) for m in metrics),first100_desc=float(np.mean([m['desc'] for m in metrics[:100]])),last100_desc=float(np.mean([m['desc'] for m in metrics[-100:]])),training_seconds=status['elapsed_seconds'],physical_gpu=status['physical_gpu'],config_differences=differences)
 # Dense detector heads stay fixed; inspect same top128 coordinates and descriptors only.
 driftrows=[];oldfolder=OLD/'predictions/final'
 for row in records('val'):
  pid=row['pair_id'];paths=[oldfolder/f'{pid}_b_top128_{OLDTAG}.npz',ROOT/f'predictions/formal_step{step}/{pid}_keep_top128_{TAG}.npz',REF/f'predictions/formal_step6000/{pid}_keep_top128_{FROZEN_TAG}.npz'];fs=[]
  for p in paths:
   with np.load(p) as z:fs.append({k:z[k] for k in z.files})
  vals=[[],[]]
  for side in [0,1]:
   key='keypoints'+str(side);assert all(np.array_equal(fs[0][key],f[key]) for f in fs[1:]);dkey='descriptors'+str(side);td=fs[0][dkey][0]
   for j in [0,1]:vals[j].extend(np.maximum(0,1-(td*fs[j+1][dkey][0]).sum(1)).tolist())
  driftrows.append(dict(pair_id=pid,scene=row['source_scene'],positions=len(vals[0]),lambda0_detected_drift=float(np.mean(vals[0])) if vals[0] else None,keep_detected_drift=float(np.mean(vals[1])) if vals[1] else None))
 selected=dict(checkpoint=str(chosen),sha=sha(chosen),step=step,qualified=bool(status.get('qualified')),lambda_keep=0.,trainable_changes=changed,frozen_reference_checkpoint_sha=sha(REF/f'runs/formal/best_trained_{FROZEN_TAG}.ckpt'),reference_selected_step=oldck['global_step'],config_differences=differences)
 safety=[]
 for mode in ['natural','equal']:
  pp=[r for r in paired if r['protocol']==mode and r['geometry_eligible']]
  safety.append(dict(protocol=mode,positive_n=len(pp),lambda0_wrong=sum(r['lambda0_wrong'] for r in pp),keep_wrong=sum(r['keep_wrong'] for r in pp),wrong_to_success=sum(r['lambda0_wrong'] and r['keep_success'] for r in pp),wrong_to_reject=sum(r['lambda0_wrong'] and not r['keep_wrong'] and not r['keep_success'] for r in pp),success_to_wrong=sum(r['lambda0_success'] and r['keep_wrong'] for r in pp),new_wrong_from_other_reject=sum(not r['lambda0_wrong'] and not r['lambda0_success'] and r['keep_wrong'] for r in pp),wrong_persisted=sum(r['lambda0_wrong'] and r['keep_wrong'] for r in pp)))
 csvwrite(ROOT/f'WRONG_ACCEPT_TRANSITIONS_{t}.csv',safety);write(ROOT/f'WRONG_ACCEPT_TRANSITIONS_{t}.json',safety)
 for name,data in [('UNIFIED_COMPARISON',table),('STRATIFIED',strata),('PAIRED_EFFECTS',paired),('CONTROLLED_ATTRIBUTION',summaries),('SAME_STEP_ENDPOINTS',endpoints),('DETECTED_DESCRIPTOR_DRIFT',driftrows)]:csvwrite(ROOT/f'{name}_{t}.csv',data);write(ROOT/f'{name}_{t}.json',data)
 csvwrite(ROOT/f'FINAL_PER_PAIR_{t}.csv',allrows);write(ROOT/f'ALL_SCORES_{t}.json',allrows);write(ROOT/f'SELECTED_CONTROL_{t}.json',selected);write(ROOT/f'ACTUAL_TRAINING_CONTROL_{t}.json',coverage);write(ROOT/f'EXPLORATORY_COMPONENT_INTERVALS_{t}.json',cis)
 print(json.dumps(dict(selected=selected,attribution=summaries,coverage=coverage),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
