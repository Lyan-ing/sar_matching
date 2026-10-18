"""Bounded offline preparation. Reuses UI; never creates human points or GT."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json,csv,shutil,hashlib,re,zipfile,sys
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image,ImageDraw

BASE=Path('/home/yl/matching/sar_matching')
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
STAMP=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
OLD=BASE/'RealSAR_PhysicalPrep_20261013_2219/blind_release'
G=BASE/'RealSAR_GeoAssist_20261012_1917'
A=BASE/'RealSAR_AngleExpansion_20261012_1641'
sys.path.insert(0,str(G/'code'))
from geometry import View,mapped,overlap

# Chosen from visual structure evidence before any human annotation/model run.
# These are rectangle origins, not clicked or matched physical points.
SELECT=[
 ('C003','ANGLE_e76cc299fa7b__native_000',[[6350,2380],[6391,2979]],'priority_ground_structure_confirmation'),
 ('C002','ANGLE_e3bfffe1139e__native_000',[[6350,2380],[6634,4641]],'priority_ground_structure_confirmation'),
 ('C001','ANGLE_75c261e08eae__native_000',[[4800,6250],[5510,4679]],'conditional_blur_and_ground_structure_confirmation'),
]

def jsave(p,data):
 with p.open('x',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)
def text(p,s):
 with p.open('x',encoding='utf-8') as f:f.write(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(p,rows,fields=None):
 with p.open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def build():
 if len(sys.argv)>1:ROOT.mkdir(exist_ok=False)
 # Check our declared centre-based selection rule before writing any material.
 prequeue={r['sample_id']:r for r in lines(A/'review/candidate_regions.jsonl')}
 for code,sid,origins,_ in SELECT:
  vs=[View(prequeue[sid],i) for i in [0,1]]
  predicted=mapped(*vs,[np.asarray(origins[0])+255.5])[0]
  assert np.max(np.abs(predicted-(np.asarray(origins[1])+255.5)))<=.51,(code,predicted,origins)
 for folder in ['custodian','raw_crops','image_only_inputs','blind_release','previews']:(ROOT/folder).mkdir()
 queue={r['sample_id']:r for r in lines(A/'review/candidate_regions.jsonl')}
 legacy=lines(BASE/'real_sar_pilot/multisource512_20261010_2352/dataset_index.jsonl')
 assert len(legacy)==191 and len(queue)==18
 protocol=dict(protocol_id='REAL512_PHYSICAL_V1',frozen_before_human_annotation=True,
  frozen_before_algorithm_evaluation=True,created_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
  task='two real native SAR crops only, no geographic/pose/GT input or initialization',shape=[512,512],
  coordinate_convention='zero based pixel centers x=column y=row, range 0..511; half-open integer crop windows',
  human_requirements=dict(real_annotator_and_distinct_real_reviewer=True,identity_alias_and_exposure_audit=True,
   independent_reclick_required=True,static_and_visible_both_explicit=True,structure_description_required=True,
   confidence=['medium','high'],explicit_uncertainty_each_view=True,assisted_formal_eligible=False),
  primary_patch_sparse=dict(min_checks=6,min_hull_fraction_each_view=.1,min_x_span_fraction_each_view=.5,
   min_y_span_fraction_each_view=.5,min_4x4_cells_each_view=4,min_quadrants_each_view=3,
   min_distinct_physical_structure_groups=3,max_checks_per_physical_structure_group=2,
   min_pairwise_check_separation_each_view_px=32),
  secondary_local_sparse=dict(min_checks=4,min_hull_fraction_each_view=.05,min_x_span_fraction_each_view=.25,
   min_y_span_fraction_each_view=.25,min_4x4_cells_each_view=3,min_distinct_physical_structure_groups=3,
   max_checks_per_physical_structure_group=2,min_pairwise_check_separation_each_view_px=32),
  uncertainty=dict(thresholds_px=[3,5,10],max_combined_uncertainty_fraction=1/3,
   rule='u0=max(annotator sigma0, reviewer sigma0, actual reclick disagreement0); u1 analog; U=hypot(u0,u1)',
   no_filled_or_default_sigma=True,unresolved_disagreement_requires_review=True),
  prediction_evaluation=dict(checks_never_fit_or_initialize_prediction=True,
   primary_error='moving native pixel Euclidean residual, mean/median/p90/p95 at independent checks',
   sparse_success_rule='accepted pure-image prediction AND mean check error <= threshold; all required checks U <= threshold/3',
   primary_sparse_population='only independent primary_patch_sparse qualified; denominator and unsupported thresholds reported explicitly',
   secondary_local_separate=True,no_dense_or_absolute_geographic_accuracy_claim=True,
   no_GT_based_method_selection_or_prediction_acceptance=True),
  geometry=dict(local_affine_ground_plane_hypothesis='unverified until independent point evidence',
   applicability_diagnostic='if >=3 separately designated controls and >=4 checks: fixed affine fit controls only; p90 held-out check residual >10px flags nonuniformity, does not delete physical points or prove GT',
   homography_not_guaranteed=True,models_and_predictive_transform_choice_locked_before_GT_opening=True),
  freeze=dict(no_recrop_after_annotation=True,no_geometry_resampling=True,no_upsampling=True,no_rotation_prealignment=True,
   no_model_success_selection=True,all_candidate_failures_and_unknowns_retained=True),
  splits='existing components/site/acquisition dependency retained; development seen; no new independent final-test claim',
  original_policy=str(BASE/'RealSAR_GeometryGT_20261011_0220/gt_quality/quality.py'),
  original_policy_unmodified=True,GT_qualified_count=0)
 pp=ROOT/f'EVALUATION_PROTOCOL_{STAMP}.json';jsave(pp,protocol)
 rows=[];tasks=[];identities={};image_alias={};verification=[];status=[];exposure_docs=[]
 for code,sid,origins,priority in SELECT:
  r=queue[sid];vs=[View(r,i) for i in [0,1]]
  histories=sorted((G/'assisted_triage'/sid).glob('v*.json'));history=[json.loads(p.read_text()) for p in histories]
  exposure_docs.append(dict(blind_code=code,source_sample_id=sid,history_files=[str(p) for p in histories],
    actual_historical_versions=history,propagation='same real acquisition/site and prior aliases remain exposed after cropping; operator 1 not blind eligible',
    direct_exact_human_clicks_in_historical_records=sum(len(d.get('manual_clicks',[])) for d in history),
    prior_navigation_centers='unconfirmed browsing centers; never paired as correspondences',
    prepared_by='Codex assisted visual inspection; NOT human physical confirmation'))
  row=dict(candidate_id=code,source_sample_id=sid,pair_id=r['pair_id'],site=r['site_name'],sensor=r['sensor'],
    component=r['component'],component_alias='real_component_011' if 'Rotterdam' in r['site_name'] else None,
    split='diagnostic_development_seen',azimuth_delta_deg=r['azimuth_delta_deg'],incidence_delta_deg=r['incidence_delta_deg'],
    time_gap_seconds=r['days']*86400,priority=priority,physical_correspondence_status='frozen_unannotated_candidate',
    independent_GT=False,GT_qualified=False,selection_depends_on_models=False,
    selection_uses_product_geography_and_prior_assisted_context=True,geometry_transform_applied_to_pixels=False,
    selection_not_same_as_human_point_confirmation=True,views=[])
  task=dict(sample_id=code,width0=512,height0=512,width1=512,height1=512,batch_id='fixed512',
    batch_label='固定原生双图',site_name='')
  ps=ROOT/'raw_crops'/code;ps.mkdir()
  for i,(v,origin) in enumerate(zip(r['views'],origins)):
   x,y=origin;win=Window(x,y,512,512);acq=r['acquisition_metadata']['views'][i]
   tif=ROOT/'raw_crops'/code/f'raw_side{i}_{STAMP}.tif'
   maskfile=ROOT/'raw_crops'/code/f'valid_side{i}_{STAMP}.png'
   inputfile=ROOT/'image_only_inputs'/f'{code}_side{i}_DN_{STAMP}.png'
   with rasterio.open(v['native_tif']) as src:
    assert vs[i].method=='Affine' and src.rpcs is None and not src.gcps[0]
    assert 0<=x<=src.width-512 and 0<=y<=src.height-512
    arr=src.read(1,window=win);mask=(src.read_masks(1,window=win)>0)&np.isfinite(arr)
    assert arr.shape==(512,512) and arr.dtype==np.uint8
    profile=src.profile.copy();profile.update(width=512,height=512,count=1,transform=src.window_transform(win),compress='deflate')
    with rasterio.open(tif,'w',**profile) as dst:dst.write(arr,1);dst.write_mask(mask.astype('uint8')*255)
    Image.fromarray(arr).save(inputfile);Image.fromarray(mask.astype('uint8')*255).save(maskfile)
    with rasterio.open(tif) as dst:
     assert np.array_equal(dst.read(1),arr) and np.array_equal(dst.read_masks(1)>0,mask)
     assert dst.transform==src.window_transform(win) and dst.crs==src.crs
    assert np.array_equal(np.asarray(Image.open(inputfile)),arr)
   key=(v['acquisition_id'],x,y)
   if key not in image_alias:
    name=f'I{len(image_alias)+1:03d}_{STAMP}.png'
    with Image.open(vs[i].png['db']) as im:crop=im.crop((x,y,x+512,y+512));crop.save(ROOT/'blind_release'/name)
    assert np.array_equal(np.asarray(Image.open(ROOT/'blind_release'/name)),np.asarray(crop))
    image_alias[key]=name;identities[name]=dict(sha256=sha(ROOT/'blind_release'/name),width=512,height=512,pixel_convention='zero-based native crop pixel centers')
   name=image_alias[key];task[f'image{i}']=name
   native_parent=v['original_native_window_xywh'];product_offset=[native_parent[0]+x,native_parent[1]+y]
   display_settings=json.loads((Path(v['native_tif']).parent/'display_db.json').read_text())
   row['views'].append(dict(side=i,acquisition=acq,native_metadata=v,source_native_tif=v['native_tif'],
    original_product_profile_metadata=str(Path(v['native_tif']).parent/'metadata.json'),
    crop_offset_in_retained_native_xy=origin,crop_offset_in_original_product_xy=product_offset,
    original_native_parent_window_xywh=native_parent,crop_shape=[512,512],
    crop_transform=list(profile['transform'])[:6],crop_crs=str(profile['crs']),
    raw_crop_tif=str(tif),raw_crop_sha256=sha(tif),raw_DN_png=str(inputfile),raw_DN_png_sha256=sha(inputfile),
    valid_mask_png=str(maskfile),valid_fraction=float(mask.mean()),display_image=name,display_identity=identities[name],
    display_source_existing_png=str(vs[i].png['db']),display_settings=display_settings,
    physical_resolution_metadata=acq.get('resolution'),metric_pixel_spacing=v.get('metric_geometry'),
    coordinate_rule='crop xy -> retained Native xy + crop offset -> original product xy + parent offset, each integer offset once'))
   verification.append(dict(candidate_id=code,side=i,shape=[512,512],exact_raw_pixel_equality=True,
    exact_raw_DN_png_equality=True,exact_existing_display_crop_equality=True,valid_fraction=float(mask.mean()),
    transform_and_CRS_consistent=True,source_header_method=vs[i].method,resampled=False,rotated=False,upsampled=False))
  # Geographical overlap is confined to custodian provenance, never displayed in blind material.
  from affine import Affine
  cv=[]
  for i in [0,1]:
   vv=object.__new__(View);vv.width=vv.height=512;vv.affine=vs[i].affine*Affine.translation(*origins[i]);vv.inverse=~vv.affine
   vv.crs=vs[i].crs;vv.gcp=None;cv.append(vv)
  row['geographic_extent_support_NOT_GT']=overlap(cv)
  rows.append(row);tasks.append(task)
  latest=history[-1]
  status.append(dict(candidate_id=code,source_sample_id=sid,site=r['site_name'],azimuth_delta_deg=round(r['azimuth_delta_deg'],6),
    component=r['component'],shared_left_crop='C002/C003 identical left crop' if code in ['C002','C003'] else 'none',
    left_native_origin=str(origins[0]),right_native_origin=str(origins[1]),latest_assisted_class=latest['verdict'],
    human_exact_pair_count=0,independent_reviewed_count=0,GT_qualified=False,status='frozen_unannotated_candidate',priority=priority))
  preview=Image.new('RGB',(1040,555),(22,22,22));d=ImageDraw.Draw(preview)
  d.text((8,6),f'{code} | native 512x512 | crop frozen | no physical GT',fill='white')
  for i in [0,1]:preview.paste(Image.open(ROOT/'blind_release'/task[f'image{i}']).convert('RGB'),(i*528,30))
  preview.save(ROOT/'previews'/f'{code}_preview_{STAMP}.png')
 # Reuse all behaviour code byte-for-byte. Only task/image data change.
 for name in ['coordinates_20261013_2227.js','offline_capture_20261013_2227.js','blind_app_20261013_2227.js']:
  shutil.copyfile(OLD/name,ROOT/'blind_release'/name)
 html=(OLD/'blind_annotation_20261013_2227.html').read_text()
 payload='<script>window.BLIND_TASKS='+json.dumps(sorted(tasks,key=lambda t:t['sample_id']),ensure_ascii=False)+';window.BLIND_PACKAGE_ID="fixed512_'+STAMP+'";window.BLIND_IMAGE_IDENTITIES='+json.dumps(identities)+';</script>'
 html=re.sub(r'<script>window.BLIND_TASKS=.*?</script>',lambda m:payload,html,count=1,flags=re.S)
 html=html.replace('Native盲标离线采集','固定512盲标离线采集')
 text(ROOT/'blind_release'/f'blind_annotation_{STAMP}.html',html)
 for name in ['physical_observation_template_20261013_2227.csv','reviewer_structure_descriptions_template_20261013_2334.csv']:
  dest=name.split('_template_')[0]+'_template_'+STAMP+'.csv';shutil.copyfile(OLD/name,ROOT/'blind_release'/dest)
 shutil.copyfile(BASE/'RealSAR_PhysicalPrep_20261013_2219/custodian/personnel_distribution_registry_template_20261013_2334.csv',
  ROOT/'custodian'/f'personnel_distribution_registry_template_{STAMP}.csv')
 jsave(ROOT/'custodian'/f'candidate_sources_{STAMP}.json',rows)
 jsave(ROOT/'custodian'/f'exposure_history_{STAMP}.json',exposure_docs)
 events=lines(G/'assisted_triage/exposure_events.jsonl')
 jsave(ROOT/'custodian'/f'prior_exposure_events_{STAMP}.json',events)
 table(ROOT/f'FIXED_CANDIDATES_{STAMP}.csv',status)
 for file in ['/tmp/candidate512_drafts_20261014_1711.json','/tmp/candidate512_drafts_20261014_1718.json']:
  shutil.copyfile(file,ROOT/'custodian'/Path(file).name)
 draft_count=sum(len(json.loads(Path(file).read_text())) for file in ['/tmp/candidate512_drafts_20261014_1711.json','/tmp/candidate512_drafts_20261014_1718.json'])
 jsave(ROOT/f'preparation_verification_{STAMP}.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
  old512_count=len(legacy),old512_acquisition_relations=len(set(x['pair_id'] for x in legacy)),
  old512_direction_counts={k:sum(x['look_pair_category']==k for x in legacy) for k in sorted(set(x['look_pair_category'] for x in legacy))},
  native_queue_count=len(queue),bounded_native_draft_windows=draft_count,selected_count=len(rows),unique_acquisitions=len(image_alias),
  selected_source_components=len(set(x['component'] for x in rows)),selected_GT_count=0,actual_human_annotations_created=0,
  services_started=[],training_started=False,GPU_used=False,network_downloads=0,large_original_SHA_audit=False,
  per_crop_checks=verification,UI_javascript_byte_identical=True,protocol_sha256=sha(pp)))
 print(ROOT,flush=True)

if __name__=='__main__':build()
