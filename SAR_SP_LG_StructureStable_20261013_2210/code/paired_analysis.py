"""Known-component paired exploratory uncertainty; no claim of independent raw acquisition."""
from ss_common import *

def main():
 files=list(ROOT.glob('validation_results_*.json'));rr=json.loads(max(files,key=lambda p:p.stat().st_mtime_ns).read_text())['rows'];base={r['id']:r for r in rr if r['method']=='m2'};result=[];rng=np.random.default_rng(26101322)
 for split in ['select','holdout','all']:
  for method in ['a','b','control','orb']:
   sub=[r for r in rr if r['method']==method and not r['no_overlap'] and (split=='all' or r['split']==split)];cs=sorted({r['source_component'] for r in sub});diff=np.array([np.mean([int(r['success_5px'])-int(base[r['id']]['success_5px']) for r in sub if r['source_component']==c]) for c in cs]);boot=diff[rng.integers(0,len(cs),(2000,len(cs)))].mean(1)*100
   result.append(dict(split=split,method=method,n_pairs=len(sub),n_known_components=len(cs),gained=sum(r['success_5px'] and not base[r['id']]['success_5px'] for r in sub),lost=sum(base[r['id']]['success_5px'] and not r['success_5px'] for r in sub),paired_component_mean_delta_pp=float(diff.mean()*100),component_bootstrap_95lo=float(np.percentile(boot,2.5)),component_bootstrap_95hi=float(np.percentile(boot,97.5)),caveat='known component bootstrap; same proxy source, previously observed; not independent SAR acquisition uncertainty or confirmatory significance'))
 event('paired_analysis',dict(seed=26101322,bootstrap=2000,rows=result));export_csv(ROOT/f'paired_component_comparison_{stamp()}.csv',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
