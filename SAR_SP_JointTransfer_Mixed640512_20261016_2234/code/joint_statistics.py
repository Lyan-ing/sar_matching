"""No predictions altered: grouped metrics, paired changes and component uncertainty."""
from mixed_io import *
from collections import Counter,defaultdict

def scores():return [json.loads(s) for s in (ROOT/'predictions/final'/f'scores_{TAG}.jsonl').read_text().splitlines() if s]
def detailed(rr):
 a=scorer.aggregate(rr);p=[r for r in rr if r['geometry_eligible']];accepted=[r for r in p if r['accepted']]
 def avg(key,items):v=[r[key] for r in items if r.get(key) is not None];return float(np.mean(v)) if v else None
 err=[r['bidirectional_error_px'] for r in p if r['bidirectional_error_px'] is not None];ae=[r['bidirectional_error_px'] for r in accepted if r['bidirectional_error_px'] is not None]
 a.update(positive_wrong_accept_count=sum(r['accepted'] and not r['Success@5px'] for r in p),positive_wrong_accept_rate=sum(r['accepted'] and not r['Success@5px'] for r in p)/len(p) if p else None,no_overlap_false_accept_count=sum(bool(r['false_accept']) for r in rr if r['no_overlap']),mean_finite_candidate_error=float(np.mean(err)) if err else None,median_finite_candidate_error=float(np.median(err)) if err else None,finite_candidate_count=len(err),mean_accepted_error=float(np.mean(ae)) if ae else None,median_accepted_error=float(np.median(ae)) if ae else None,keypoints0=avg('keypoints0',rr),keypoints1=avg('keypoints1',rr),matches=avg('match_count',rr),inliers=avg('inliers',rr),correct_static_matches=avg('correct_structure_matches',rr),static_precision_micro=sum(r['correct_structure_matches'] for r in rr)/sum(r['supervisable_match_count'] for r in rr) if sum(r['supervisable_match_count'] for r in rr) else None,anchor_recall_micro=sum(r['anchor_recalled'] for r in rr)/sum(r['structure_anchor_count'] for r in rr) if sum(r['structure_anchor_count'] for r in rr) else None,no_output_count=sum(r['state']=='no_output' for r in rr),rejected_candidate_count=sum(r['state']=='rejected' for r in rr),accepted_count=sum(r['state']=='accepted' for r in rr),known_components=len({r['component'] for r in rr}))
 return a

def main():
 rr=scores();methods=list(dict.fromkeys(r['method'] for r in rr));lookup={(r['pair_id'],r['method'],r['protocol']):r for r in rr};table=[];paired=[];summ=[]
 for m in methods:
  for mode in ['natural','equal']:
   subset=[r for r in rr if r['method']==m and r['protocol']==mode];table.append(dict(method=m,protocol=mode,**detailed(subset)))
   for axis in ['size','scene','overlap_bin','angle_bin','profile']:
    for v in sorted({r[axis] for r in subset},key=str):summ.append(dict(method=m,protocol=mode,stratification=axis,stratum=v,**detailed([r for r in subset if r[axis]==v])))
   if m=='b':continue
   changes=[]
   for r in subset:
    base=lookup[(r['pair_id'],'b',mode)];d=dict(pair_id=r['pair_id'],method=m,protocol=mode,component=r['component'],scene=r['scene'],size=r['size'],geometry_eligible=r['geometry_eligible'],baseline_success=base['Success@5px'],method_success=r['Success@5px'],recovered=bool(r['geometry_eligible'] and r['Success@5px'] and not base['Success@5px']),lost=bool(r['geometry_eligible'] and not r['Success@5px'] and base['Success@5px']),delta_correct_static=r['correct_structure_matches']-base['correct_structure_matches']);paired.append(d)
    if r['geometry_eligible']:changes.append((r['component'],int(bool(r['Success@5px']))-int(bool(base['Success@5px']))))
   groups=defaultdict(list)
   for c,d in changes:groups[c].append(d)
   keys=sorted(groups);sums=np.array([sum(groups[k]) for k in keys]);counts=np.array([len(groups[k]) for k in keys]);rng=np.random.default_rng(42);indices=rng.integers(0,len(keys),(10000,len(keys)));boots=sums[indices].sum(1)/counts[indices].sum(1)
   table[-1].update(recovered=sum(x['recovered'] for x in paired if x['method']==m and x['protocol']==mode),lost=sum(x['lost'] for x in paired if x['method']==m and x['protocol']==mode),paired_delta_S5=float(np.mean([d for c,d in changes])),component_bootstrap_lower=float(np.quantile(boots,.025)),component_bootstrap_upper=float(np.quantile(boots,.975)),bootstrap_components=len(keys),bootstrap_caveat='Known source components, not proven independent original acquisitions; development Val; exploratory interval')
 t=stamp();csvwrite(ROOT/f'UNIFIED_COMPARISON_{t}.csv',table);csvwrite(ROOT/f'DETAILED_STRATIFIED_{t}.csv',summ);csvwrite(ROOT/f'PAIRED_CHANGES_{t}.csv',paired);write(ROOT/f'UNIFIED_COMPARISON_{t}.json',table)
 # Strict separation of all-geometry/structure-qualified/no-evidence subsets.
 csvwrite(ROOT/f'ROLE_SPECIFIC_{t}.csv',[dict(method=m,protocol=mode,role=role,**detailed([r for r in rr if r['method']==m and r['protocol']==mode and r['role']==role])) for m in methods for mode in ['natural','equal'] for role in sorted({r['role'] for r in rr})])
 decomposition=[]
 for m in methods:
  for mode in ['natural','equal']:
   subset=[r for r in rr if r['method']==m and r['protocol']==mode];ref=[r for r in rr if r['method']=='b' and r['protocol']==mode]
   count=lambda a,key:sum(bool(r.get(key)) for r in a)
   total=count(subset,'Success@5px');qualified=count(subset,'StructureSuccess@5px');bt=count(ref,'Success@5px');bq=count(ref,'StructureSuccess@5px')
   decomposition.append(dict(method=m,protocol=mode,all_geometry_positive_n=1140,structure_qualified_n=348,other_geometric_positive_n=792,all_success=total,qualified_success=qualified,other_success=total-qualified,delta_all=total-bt,delta_qualified=qualified-bq,delta_other=(total-qualified)-(bt-bq),other_subset_is_not_negative_and_not_independent_physical_GT=True))
 csvwrite(ROOT/f'SUPERVISION_EVIDENCE_DECOMPOSITION_{t}.csv',decomposition)
 components=[]
 for m in methods:
  for mode in ['natural','equal']:
   subset=[r for r in rr if r['method']==m and r['protocol']==mode]
   for component in sorted({r['component'] for r in subset}):components.append(dict(method=m,protocol=mode,component=component,**detailed([r for r in subset if r['component']==component])))
 csvwrite(ROOT/f'COMPONENT_RESULTS_{t}.csv',components)
 import importlib.util
 spec=importlib.util.spec_from_file_location('joint_failure_stages',ROOT/'code/analyze.py');stages=importlib.util.module_from_spec(spec);spec.loader.exec_module(stages)
 csvwrite(ROOT/f'FINAL_FAILURE_STAGES_{t}.csv',stages.diagnose(rr))
 print('Final grouped, paired and component-level statistics completed.')
if __name__=='__main__':main()
