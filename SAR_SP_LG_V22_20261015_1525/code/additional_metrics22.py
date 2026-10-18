"""Cached-result diagnostics only: fixed common133, component uncertainty and failure stages."""
from support22 import *
from evaluate22 import summarize
from collections import Counter,defaultdict
def main():
 a={v:json.loads(out('main_'+v).read_text()) for v in ['v1','v21','v22']};tables=[];stages=[];comparisons=[]
 for version,data in a.items():
  rr=data['rows']
  for method in data['summaries']['natural']:
   for mode in ['natural','equal']:
    s=[r for r in rr if r['method']==method and r['protocol']==mode];pos=[r for r in s if r['global_eligible']];counter=Counter()
    for r in pos:
     if r['success5']:stage='success'
     elif r['visible_detected0']<8:stage='fewer8_visible_static_detections'
     elif r['repeatable']<8:stage='fewer8_repeated_static_detections'
     elif r['correct_static_matches']<8:stage='fewer8_correct_static_LG_matches'
     elif r['correct_hull']<.01 or r['correct_grid']<.0625:stage='correct_matches_spatially_concentrated'
     elif not r['accepted']:stage='enough_static_matches_but_rejected'
     else:stage='accepted_incorrect_geometry'
     counter[stage]+=1
    for stage,n in counter.items():stages.append(dict(version=version,method=method,protocol=mode,stage=stage,n=n,denominator=len(pos)))
    errors=[r['local_bidir_error'] for r in pos if r['local_bidir_error'] is not None];tables.append(dict(version=version,method=method,protocol=mode,global_n=len(pos),finite_candidate_error_n=len(errors),candidate_bidir_error_mean=float(np.mean(errors)) if errors else None,candidate_bidir_error_median=float(np.median(errors)) if errors else None,no_output=sum(r['state']=='no_output' for r in s),rejected=sum(r['state']=='rejected' for r in s),accepted=sum(r['accepted'] for r in s)))
  for mode in ['natural','equal']:
   for new,base in [('M2_V22','M2'),('B_V22','B'),('B_V22','M2')]:
    left={r['sample_id']:r for r in rr if r['method']==new and r['protocol']==mode and r['global_eligible']};right={r['sample_id']:r for r in rr if r['method']==base and r['protocol']==mode and r['global_eligible']};assert left.keys()==right.keys();cluster=defaultdict(list)
    for k,r in left.items():cluster[r['component']].append(int(r['success5'])-int(right[k]['success5']))
    vals=np.array([np.mean(v) for v in cluster.values()]);rg=np.random.default_rng(202622);boot=rg.choice(vals,(2000,len(vals)),replace=True).mean(1);comparisons.append(dict(version=version,protocol=mode,new=new,baseline=base,pair_n=len(left),known_components=len(vals),pair_weighted_S5_delta=float(np.mean([x for v in cluster.values() for x in v])),component_equal_S5_delta=float(vals.mean()),component_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),note='Observed development sources; known parents, not proven independent raw acquisitions'))
 common=[]
 for method in a['v21']['summaries']['natural']:
  for mode in ['natural','equal']:common.append(dict(method=method,protocol=mode,**summarize([r for r in a['v21']['rows'] if r['method']==method and r['protocol']==mode and r['common133']])))
 tag=stamp();csvwrite(ROOT/f'failure_stages_{tag}.csv',stages);csvwrite(ROOT/f'candidate_error_and_decisions_{tag}.csv',tables);csvwrite(ROOT/f'V21_common133_{tag}.csv',common);write(ROOT/f'component_comparisons_{tag}.json',comparisons)
 print(json.dumps(comparisons,indent=2))
if __name__=='__main__':main()
