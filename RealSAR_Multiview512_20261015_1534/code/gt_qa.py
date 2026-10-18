"""Actual-person return QA. No generated correspondences, identities, or averages."""
from common import *
import argparse,importlib.util
from collections import defaultdict,Counter
spec=importlib.util.spec_from_file_location('historical_gt_quality',BASE/'RealSAR_GeometryGT_20261011_0220/gt_quality/quality.py')
quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)
PROTOCOL=json.loads((FIXED/'EVALUATION_PROTOCOL_20261014_1754.json').read_text())

def csvrows(p):
 with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def yes(v):return str(v).strip().casefold() in ['yes','true','1','是']
def explicit_no(v):return str(v).strip().casefold() in ['no','false','0','否']
def aliases(person):return {person['person_id'].strip(),*filter(None,re.split(r'[;,|，；]',person.get('aliases','')))}

def support(points,views,kind):
 rule=PROTOCOL[kind];checks=[p for p in points if p['role']=='check'];reasons=[];stats=[]
 if len(checks)<rule['min_checks']:reasons.append('insufficient_independent_checks')
 groups=Counter(p['physical_structure_group'] for p in checks)
 if len(groups)<rule['min_distinct_physical_structure_groups']:reasons.append('insufficient_independent_structure_groups')
 if max(groups.values(),default=0)>rule['max_checks_per_physical_structure_group']:reasons.append('too_many_checks_on_one_structure_group')
 for side,v in enumerate(views):
  z=v['native_info'];p=np.array([p[f'xy{side}'] for p in checks],float).reshape(-1,2);s=quality.spatial(p,(z['height'],z['width']));span=s['span']
  quadrants=len({(int(x>=z['width']/2),int(y>=z['height']/2)) for x,y in p})
  distance=min((float(np.linalg.norm(p[i]-p[j])) for i in range(len(p)) for j in range(i)),default=float('inf'))
  stats.append(dict(s,quadrants=quadrants,min_pairwise_separation_px=distance if np.isfinite(distance) else None))
  if s['hull_fraction']<rule['min_hull_fraction_each_view']:reasons.append(f'side{side}_hull')
  if span[0]<rule['min_x_span_fraction_each_view'] or span[1]<rule['min_y_span_fraction_each_view']:reasons.append(f'side{side}_span')
  if s['grid_cells']<rule['min_4x4_cells_each_view']:reasons.append(f'side{side}_grid')
  if quadrants<rule.get('min_quadrants_each_view',0):reasons.append(f'side{side}_quadrants')
  if distance<rule['min_pairwise_check_separation_each_view_px']:reasons.append(f'side{side}_separation')
 return dict(passed=not reasons,reasons=reasons,spatial=stats,groups=dict(groups),checks=len(checks))

def exposure_denied(actor_ids,component_id):
 # A new crop never resets an old sample/site exposure.
 source=ANGLE/'review/candidate_regions.jsonl';sites={r['sample_id']:r['site_name'] for r in rows(source)}
 reasons=[];g=BASE/'RealSAR_GeoAssist_20261012_1917/assisted_triage'
 for path in g.glob('*/v*.json'):
  d=json.loads(path.read_text());site=sites.get(d.get('sample_id'))
  if str(d.get('actor')) in actor_ids and site and component(site)==component_id:reasons.append(str(path))
 events=g/'exposure_events.jsonl'
 if events.exists():
  for d in rows(events):
   site=sites.get(d.get('sample_id'))
   if str(d.get('actor')) in actor_ids and site and component(site)==component_id:reasons.append(str(events))
 return sorted(set(reasons))

def qualified_documents(pkgA,pkgB,docA,docB,assignA,assignB,task_map):
 reasons=[]
 for p,d,person in [(pkgA,docA,assignA),(pkgB,docB,assignB)]:
  if not person or not person.get('person_id'):reasons.append('unassigned_real_person');continue
  if str(d.get('actor'))!=person['person_id']:reasons.append('returned_actor_not_assigned_person')
  for key in ['identity_verified','exposure_audited','independent_work_verified']:
   if not yes(person.get(key)):reasons.append('custodian_'+key+'_missing')
  if not person.get('verified_by') or not person.get('verified_at'):reasons.append('missing_custodian_identity_provenance')
  declarations=d.get('exposure_declaration',{})
  if any(declarations.get(k)!='no' for k in ['priorGeo','priorAlgorithm','priorHints','priorOtherPoints']):reasons.append('assisted_or_unknown_exposure')
  if d.get('package_id')!=p['package_id']:reasons.append('package_identity_mismatch')
  for name,h in zip(task_map['image_names'],task_map['image_sha256']):
   if d.get('display_image_identities',{}).get(name,{}).get('sha256')!=h:reasons.append('returned_image_identity_mismatch')
 if assignA and assignB and aliases(assignA)&aliases(assignB):reasons.append('same_person_or_alias_cannot_independently_review')
 if assignA and exposure_denied(aliases(assignA),task_map['component']):reasons.append('historical_component_exposure_A')
 if assignB and exposure_denied(aliases(assignB),task_map['component']):reasons.append('historical_component_exposure_B')
 return sorted(set(reasons))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pairs',required=True);ap.add_argument('--package-index',required=True);ap.add_argument('--received',nargs='*',default=[]);ap.add_argument('--registry');ap.add_argument('--evidence',nargs='*',default=[]);ap.add_argument('--links');ap.add_argument('--out',required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t=stamp();pairmap={r['sample_id']:r for r in rows(a.pairs)};packages={p['package_id']:p for p in rows(a.package_index)}
 registry={r['package_id']:r for r in csvrows(a.registry)} if a.registry else {};evidence={}
 for p in a.evidence:
  for r in csvrows(p):evidence[(r.get('package_id'),r.get('task_id'),r.get('point_id'))]=r
 documents={};raw_records=[];raw_points=[];invalid=[]
 for folder in a.received:
  for p in Path(folder).glob('*.json'):
   try:
    d=json.loads(p.read_text());key=(d['package_id'],d['sample_id']);assert key[0] in packages
    assert isinstance(d.get('points'),list);raw_records.append(dict(file=str(p),sha256=sha(p),package_id=key[0],task_id=key[1],actor=d.get('actor'),points=len(d['points'])))
    raw_points.append(dict(file=str(p),sha256=sha(p),package_id=key[0],task_id=key[1],actor=d.get('actor'),revision=d.get('revision'),points=d['points'],exposure_declaration=d.get('exposure_declaration'),GT_qualified=False,status='original_actual_return_not_recertified'))
    if key not in documents or int(d.get('revision',0))>int(documents[key][0].get('revision',0)):documents[key]=(d,p)
   except Exception as e:invalid.append(dict(file=str(p),error=repr(e)))
 links=csvrows(a.links) if a.links else [];by_sample=defaultdict(list)
 for r in links:by_sample[r['sample_id']].append(r)
 gt=[];statuses=[];point_failures=[]
 for sid,pair in pairmap.items():
  if pair.get('qa_state')!='passed':
   statuses.append(dict(sample_id=sid,relation_id=pair['relation_id'],component=pair['component'],split=pair['split'],state='quarantined_input_not_formal',independent_reviewed_points=0,GT_qualified=False));continue
  records=by_sample.get(sid,[]);confirmed=[];notes=[];all_input_points=0;used_clicks=set()
  for link in records:
   try:
    if not yes(link.get('physical_identity_confirmed')) or not link.get('linked_by') or not link.get('linked_at'):raise ValueError('actual physical-identity linkage by custodian required')
    if not link.get('structure_group','').strip():raise ValueError('actual custodian structure group required')
    pa,pb=[packages[link[k]] for k in ['A_package_id','B_package_id']];assert pa['slot']=='A' and pb['slot']=='B'
    keys=[(link['A_package_id'],link['A_point_id']),(link['B_package_id'],link['B_point_id'])]
    if any(k in used_clicks for k in keys):raise ValueError('one actual click cannot become multiple independent checks')
    ta=next(x for x in pa['tasks'] if x['sample_id']==sid);tb=next(x for x in pb['tasks'] if x['sample_id']==sid)
    da,patha=documents[(pa['package_id'],ta['task_id'])];db,pathb=documents[(pb['package_id'],tb['task_id'])]
    reasons=qualified_documents(pa,pb,da,db,registry.get(pa['package_id']),registry.get(pb['package_id']),ta)
    if reasons:raise ValueError(';'.join(reasons))
    p=next(p for p in da['points'] if p['point_id']==link['A_point_id']);q=next(p for p in db['points'] if p['point_id']==link['B_point_id']);all_input_points+=1
    for doc,point,pkg,task in [(da,p,pa,ta),(db,q,pb,tb)]:
     ev=evidence.get((pkg['package_id'],task['task_id'],point['point_id']))
     if not ev:raise ValueError('missing actual-person point physical evidence CSV')
     for k in ['structure_type','physical_reason','structure_group']:
      if not ev.get(k):raise ValueError('missing point evidence '+k)
     if not yes(ev.get('ground_fixed')) or not yes(ev.get('low_height_or_ground')):raise ValueError('ground physical identity unconfirmed')
     if any(not explicit_no(ev.get(k)) for k in ['height_risk','layover_risk','shadow_risk','dynamic_risk','scattering_ambiguity']):raise ValueError('unresolved physical risk/unknown')
     if point.get('is_static') is not True or point.get('visible_both') is not True or point.get('confidence') not in ['medium','high']:raise ValueError('missing independent physical confirmations')
    pp=dict(p,sample_id=sid,annotator=da['actor'],annotated_at=p.get('annotated_at'),used_assistance=False,physical_structure_group=link['structure_group'],original_annotation=str(patha),original_review=str(pathb),original_annotation_sha256=sha(patha),original_review_sha256=sha(pathb),point_link=link,
      review=dict(reviewer=db['actor'],reviewed_at=q.get('annotated_at'),xy0=q['xy0'],xy1=q['xy1'],sigma0_px=q['sigma0_px'],sigma1_px=q['sigma1_px'],conclusion='confirmed',static_confirmed=True,visible_confirmed=True,used_assistance=False),
      planar_ground_claim=all(yes(evidence[(pkg['package_id'],task['task_id'],point['point_id'])].get('planar_2d_applicable')) for pkg,task,point in [(pa,ta,p),(pb,tb,q)]))
    task=dict(sample_id=sid,height0=512,width0=512,height1=512,width1=512);quality.validate_points([pp],task)
    for side,v in enumerate(pair['views']):
     coords=[pp[f'xy{side}'],pp['review'][f'xy{side}']]
     with rasterio.open(v['path']) as ds:mask=ds.read_masks(1)>0
     if v.get('valid_mask'):mask&=np.asarray(Image.open(v['valid_mask']))>0
     for xy in coords:
      # Conservative all four neighbors for subpixel annotations.
      lo=np.floor(xy).astype(int);hi=np.ceil(xy).astype(int)
      if not all(mask[y,x] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]]):raise ValueError('click outside native valid support')
    disagreements=[float(np.linalg.norm(np.array(p[f'xy{s}'])-q[f'xy{s}'])) for s in [0,1]]
    if max(disagreements)>3 and not yes(link.get('conflict_resolved')):raise ValueError('independent reclick conflict requires actual adjudication')
    confirmed.append(pp);used_clicks.update(keys)
   except Exception as e:point_failures.append(dict(sample_id=sid,link=link,error=repr(e)))
  state='unannotated';qa=None
  if confirmed:
   task=dict(sample_id=sid,height0=512,width0=512,height1=512,width1=512);doc=dict(points=confirmed,evaluation_scope='local',disposition='reliable_correspondences')
   qa=quality.quality(doc,task);ids=set(qa['eligible_point_ids']);good=[p for p in confirmed if p['point_id'] in ids]
   primary=support(good,pair['views'],'primary_patch_sparse');local=support(good,pair['views'],'secondary_local_sparse')
   planar=all(p['planar_ground_claim'] for p in good if p['role']=='check')
   if not planar:primary['passed']=local['passed']=False;primary['reasons'].append('2D_ground_plane_not_confirmed');local['reasons'].append('2D_ground_plane_not_confirmed')
   threshold=qa['supported_success_thresholds_px'];qual=(primary['passed'] or local['passed']) and bool(threshold)
   state='independent_review_passed_GT' if qual else 'independent_reviewed_sparse_pending_protocol_support'
   gt.append(dict(sample_id=sid,relation_id=pair['relation_id'],component=pair['component'],split=pair['split'],points=confirmed,original_coordinates_never_averaged=True,primary=primary,local=local,quality=qa,GT_qualified=qual,scope='full_sparse' if primary['passed'] else 'local_sparse' if local['passed'] else 'sparse_correspondence_only',actual_two_people=True))
  else:
   found=[]
   for pkg in packages.values():
    for x in pkg['tasks']:
     key=(pkg['package_id'],x['task_id'])
     if x['sample_id']==sid and key in documents:found.append((pkg,documents[key][0]))
   if found:
    assisted=False;verified_first=False
    for pkg,doc in found:
     exp=doc.get('exposure_declaration',{});reg=registry.get(pkg['package_id'])
     assisted|=any(exp.get(k)=='yes' for k in ['priorGeo','priorAlgorithm','priorHints','priorOtherPoints'])
     if reg and exposure_denied(aliases(reg),pair['component']):assisted=True
     if reg and all(yes(reg.get(k)) for k in ['identity_verified','exposure_audited','independent_work_verified']) and str(doc.get('actor'))==reg.get('person_id') and all(exp.get(k)=='no' for k in ['priorGeo','priorAlgorithm','priorHints','priorOtherPoints']):verified_first=True
    state='assisted_human_annotation' if assisted else 'independent_annotation_pending_second_person_or_linkage' if verified_first else 'actual_human_return_identity_or_exposure_pending'
  statuses.append(dict(sample_id=sid,relation_id=pair['relation_id'],component=pair['component'],split=pair['split'],state=state,independent_reviewed_points=len(confirmed),GT_qualified=state=='independent_review_passed_GT'))
 jsonl(out/f'REAL_GT_MANIFEST_{t}.jsonl',gt);jsonl(out/f'GT_STATUS_{t}.jsonl',statuses);jsonl(out/f'RAW_HUMAN_RETURN_INDEX_{t}.jsonl',raw_records)
 jsonl(out/f'ACTUAL_RAW_HUMAN_OBSERVATIONS_{t}.jsonl',raw_points)
 save(out/f'GT_QA_RESULTS_{t}.json',dict(at=now(),actual_return_versions=len(raw_records),actual_return_people=sorted({str(r['actor']) for r in raw_records}),invalid_inputs=invalid,point_failures=point_failures,qualified_pairs=sum(r['GT_qualified'] for r in gt),status_counts=dict(Counter(r['state'] for r in statuses)),frozen_protocol=str(FIXED/'EVALUATION_PROTOCOL_20261014_1754.json'),frozen_protocol_sha256=sha(FIXED/'EVALUATION_PROTOCOL_20261014_1754.json'),automatic_physical_points_created=0))
 print('GT_QA',len(raw_records),'real submitted versions,',sum(r['GT_qualified'] for r in gt),'qualified pairs',flush=True)

if __name__=='__main__':main()
