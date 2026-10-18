"""Phase 1/2: all cached pairs first; no GPU, fitting or fusion selection."""
from fusion_io import *
from collections import Counter,defaultdict
def main():
 tag=stamp();result=[];summary=[];audit=[];hashes=hashlib.sha256()
 for version in ['v1','v21','v22']:
  data=read(OLD/f'main_{version}_20261015_1525.json');records=rows(data['manifest']);idx={(x['sample_id'],x['method'],x['protocol']):x for x in data['rows']}
  assert len(idx)==len(records)*12
  for method in ['M0','M2','B','M2_V22','ORB']:
   audit.append(dict(version=version,method=method,pairs=len(records),global_n=data['summaries']['natural'][method]['global_n'],no_overlap_n=data['summaries']['natural'][method]['no_overlap_n'],manifest=data['manifest'],manifest_sha=sha(data['manifest']),cache='main_'+version,protocol='same_native/RANSAC3/full_gate',state='verified_existing',weights=data.get('weights',{}).get(method)))
  for n,row in enumerate(records):
   sid=row['sample_id'];shape=(row['height'],row['width'])
   for mode in ['natural','equal']:
    orb=idx[(sid,'ORB',mode)];p=cache(version,sid,'ORB',mode)
    with np.load(p) as f:o={k:f[k] for k in ['keypoints0','keypoints1','points0','points1','correct_static']}
    hashes.update(p.read_bytes())
    for method in ['M2','B','M2_V22']:
     sp=idx[(sid,method,mode)];p=cache(version,sid,method,mode)
     with np.load(p) as f:s={k:f[k] for k in ['keypoints0','keypoints1','points0','points1','correct_static']}
     hashes.update(p.read_bytes());assert len(s['keypoints0'])==sp['n_keypoints0'] and len(o['keypoints0'])==orb['n_keypoints0']
     od=[nearest(o[f'keypoints{i}'],s[f'keypoints{i}']) for i in [0,1]];sd=[nearest(s[f'keypoints{i}'],o[f'keypoints{i}']) for i in [0,1]]
     om=[nearest(o[f'points{i}'],s[f'keypoints{i}']) for i in [0,1]];sm=[nearest(s[f'points{i}'],o[f'keypoints{i}']) for i in [0,1]]
     oc=o['correct_static'].astype(bool);sc=s['correct_static'].astype(bool);on=~overlap_pairs(o,s);sn=~overlap_pairs(s,o)
     co,cs=cells(o['points0'][oc],shape),cells(s['points0'][sc],shape)
     cell='not_global_evaluable'
     if sp['global_eligible']:cell='both_success' if sp['success5'] and orb['success5'] else 'sp_only' if sp['success5'] else 'orb_only' if orb['success5'] else 'both_fail'
     result.append(dict(version=version,sample_id=sid,source_path=sp['source_path'],component=sp['component'],scene=sp['scene'],role=sp['role'],global_eligible=sp['global_eligible'],no_overlap=sp['no_overlap'],profile=sp['profile'],angle_bin=sp['angle_bin'],sp_method=method,protocol=mode,success_cell=cell,
      sp_success5=sp['success5'],orb_success5=orb['success5'],sp_keypoints0=sp['n_keypoints0'],sp_keypoints1=sp['n_keypoints1'],orb_keypoints0=orb['n_keypoints0'],orb_keypoints1=orb['n_keypoints1'],sp_matches=sp['n_matches'],orb_matches=orb['n_matches'],sp_correct=int(sc.sum()),orb_correct=int(oc.sum()),sp_reason=sp['reason'],orb_reason=orb['reason'],sp_state=sp['state'],orb_state=orb['state'],sp_repeatable=sp['repeatable'],orb_repeatable=orb['repeatable'],
      orb_unique_kp0=int((od[0]>3).sum()),orb_unique_kp1=int((od[1]>3).sum()),sp_unique_kp0=int((sd[0]>3).sum()),sp_unique_kp1=int((sd[1]>3).sum()),orb_kp_overlap0=float((od[0]<=3).mean()) if len(od[0]) else None,sp_kp_overlap0=float((sd[0]<=3).mean()) if len(sd[0]) else None,
      orb_correct_at_sp_omitted_both=int((oc&(om[0]>3)&(om[1]>3)).sum()),orb_correct_at_sp_omitted_any=int((oc&((om[0]>3)|(om[1]>3))).sum()),sp_correct_at_orb_omitted_both=int((sc&(sm[0]>3)&(sm[1]>3)).sum()),orb_distinct_correct_matches=int((oc&on).sum()),sp_distinct_correct_matches=int((sc&sn).sum()),orb_new_correct_grid_cells=len(co-cs),sp_new_correct_grid_cells=len(cs-co),joint_correct_grid_cells=len(co|cs),independent_physical_GT=False,structure_evidence='legacy_static_proxy_includes_unverified_sea' if version=='v1' else 'frozen_structure_mask_only_unknown_excluded'))
   if n%100==0:print(tag,version,n+1,len(records),flush=True)
  for mode in ['natural','equal']:
   for method in ['M2','B','M2_V22']:
    allr=[x for x in result if x['version']==version and x['protocol']==mode and x['sp_method']==method and x['global_eligible']]
    for scene in ['ALL',*sorted(set(x['scene'] for x in allr))]:
     rr=[x for x in allr if scene=='ALL' or x['scene']==scene];c=Counter(x['success_cell'] for x in rr)
     summary.append(dict(version=version,protocol=mode,method=method,scene=scene,n=len(rr),**{k:c[k] for k in ['both_success','sp_only','orb_only','both_fail']},sp_S5=(c['both_success']+c['sp_only'])/len(rr),orb_S5=(c['both_success']+c['orb_only'])/len(rr),oracle_union_S5=1-c['both_fail']/len(rr),orb_distinct_correct_mean=np.mean([x['orb_distinct_correct_matches'] for x in rr]),orb_correct_at_sp_omitted_both_mean=np.mean([x['orb_correct_at_sp_omitted_both'] for x in rr]),orb_new_correct_cells_mean=np.mean([x['orb_new_correct_grid_cells'] for x in rr])))
 csvwrite(ROOT/f'INPUT_CHECK_TABLE_{tag}.csv',audit);csvwrite(ROOT/f'COMPLEMENTARITY_PAIRWISE_{tag}.csv',result);csvwrite(ROOT/f'COMPLEMENTARITY_SUMMARY_{tag}.csv',summary)
 primary=[x for x in summary if x['version']=='v1' and 'offshore' in x['scene'] and x['method'] in ['M2','B']]
 # Presence of nonzero distinctive synthetic correspondences permits bounded feasibility, not a physical-GT claim.
 decision=dict(at=tag,all_cached_samples_completed=True,pairs=sum(len(rows(read(OLD/f'main_{v}_20261015_1525.json')['manifest'])) for v in ['v1','v21','v22']),pairwise_rows=len(result),cache_aggregate_sha=hashes.hexdigest(),primary=primary,permit_bounded_fusion=any(x['protocol']=='equal' and x['orb_only']>=1 and x['orb_correct_at_sp_omitted_both_mean']>0 for x in primary),rationale='Small but nonzero equal-budget ORB-exclusive synthetic correspondences and successes; no evidence yet of cross-acquisition physical stability. Only fixed 64/64 and96/32 feasibility, no threshold search.')
 write(ROOT/f'STAGE2_DECISION_{tag}.json',decision);print(json.dumps(decision,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
