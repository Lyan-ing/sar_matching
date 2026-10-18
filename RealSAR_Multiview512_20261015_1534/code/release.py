"""Actual native readback, source/component accounting, frozen image-only manifests."""
from common import *
from collections import Counter,defaultdict
import argparse

def normalize_angles(a):
 a=dict(a);raw=a.get('azimuth_degrees');a['vendor_native_azimuth_degrees']=raw
 a['platform_heading_degrees']=None
 a['radar_los_azimuth_degrees']=None
 a['azimuth_definition_verified_source']='https://docs.canopy.umbra.space/docs/collect-metadata' if a['provider']=='Umbra' else 'https://support.capellaspace.com/what-are-capellas-tasking-parameters' if a['provider']=='Capella' else None
 if a['provider']=='Umbra':
  a['vendor_azimuth_definition']='clockwise from true north at sub-satellite ground point toward scene center; angleAzimuthDegrees / view:azimuth; NOT platform heading and NOT target azimuth'
  a['azimuth_not_same_frame_as_capella']=True
  p=a.get('metadata_path')
  if p and Path(p).exists():
   d=json.loads(Path(p).read_text());v=d.get('properties',{}).get('umbra:target_azimuth_angle_degrees')
   if v is not None:a['radar_los_azimuth_degrees']=v;a['radar_los_source']='explicit umbra:target_azimuth_angle_degrees'
 elif a['provider']=='Capella':
  a['vendor_azimuth_definition']='clockwise from north in topocentric ENZ from target to satellite; view:azimuth, not platform heading'
  a['radar_los_azimuth_degrees']=raw;a['radar_los_source']='vendor target-to-satellite view:azimuth'
 else:a['vendor_azimuth_definition']='unknown; same orbit is not a verified zero azimuth difference'
 a['no_unverified_180deg_LOS_conversion']=True
 return a

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--tag',default='release');ap.add_argument('--reuse-verified');args=ap.parse_args();t=stamp()
 pairs=rows(latest('EXISTING512_REFERENCES_*.jsonl'))
 for p in sorted(ROOT.glob('BATCH_PAIRS_*.jsonl')):pairs+=rows(p)
 if len({p['sample_id'] for p in pairs})!=len(pairs):raise ValueError('Duplicate sample ID')
 metadata={a['acquisition_id']:normalize_angles(a) for a in rows(latest('MERGED_SOURCE_INVENTORY_*.jsonl'))}
 cached={};cached_pixels=defaultdict(list);reused=0
 if args.reuse_verified:
  previous=Path(args.reuse_verified);cached={r['sample_id']:r for r in rows(previous)}
  prior_t=re.search(r'(\d{8}_\d{4})\.jsonl$',str(previous))[1]
  for r in rows(ROOT/f'INPUT_PIXEL_QA_{prior_t}.jsonl'):cached_pixels[r['sample_id']].append(r)
 output=[];failures=[];pixel_rows=[];geom=Counter();duplicate_groups=defaultdict(list)
 tif_inputs=ROOT/'legacy_pointwise_inputs';tif_inputs.mkdir(exist_ok=True)
 for i,row in enumerate(pairs):
  prior=cached.get(row['sample_id'])
  if prior and prior['qa_state']=='passed':
   paths=[Path(prior['image0']),Path(prior['image1'])]
   for v in prior['views']:
    paths.extend(Path(v[k]) for k in ['path','valid_mask','source_native_path'] if v.get(k))
   if all(p.stat().st_mtime_ns<=previous.stat().st_mtime_ns for p in paths) and [v['path'] for v in row['views']]==[v['path'] for v in prior['views']]:
    prior=dict(prior,qa_reused_from=str(previous));output.append(prior);pixel_rows.extend(cached_pixels[row['sample_id']]);reused+=1;continue
  r=dict(row);reasons=[];views=[]
  if len(set(r['acquisition_ids']))!=2:reasons.append('not_two_different_actual_collects')
  for side,v in enumerate(r['views']):
   v=dict(v);path=Path(v['path'])
   with rasterio.open(path) as ds:
    check_geo(ds);arr=ds.read(1);valid=(ds.read_masks(1)>0)&np.isfinite(arr);actual=info(ds)
    if actual['width']!=512 or actual['height']!=512:reasons.append('unexpected_shape')
    if actual['crs']!=v['native_info']['crs'] or not np.allclose(actual['transform'],v['native_info']['transform'],rtol=0,atol=1e-10):reasons.append('source_geocoding_disagreement')
   if v.get('valid_mask'):
    saved=np.asarray(Image.open(v['valid_mask']))>0
    if saved.shape!=arr.shape:reasons.append('mask_shape')
    else:valid&=saved
   if float(valid.mean())<.9:reasons.append('valid_fraction_below_.9')
   image=Path(r[f'image{side}'])
   if image.suffix.casefold()!='.png':
    # Older pilot exposes raw RTC TIFFs; make only a pointwise display, no geometry.
    im,settings=display(arr,valid,units=metadata[r['acquisition_ids'][side]].get('input_units','linear_power'))
    image=tif_inputs/f'{r["sample_id"]}_{side}_{t}.png';Image.fromarray(im).save(image);v['display_settings']=settings;r[f'image{side}']=str(image)
   im=np.asarray(Image.open(image))
   if im.shape!=(512,512) or im.dtype!=np.uint8:reasons.append('algorithm_input_not_native_single_channel_uint8')
   vals=im[valid];rawhash=hashlib.sha256(arr.tobytes()).hexdigest()
   if v.get('raw_pixel_sha256') and rawhash!=v['raw_pixel_sha256']:reasons.append('new_raw_pixels_changed_after_freeze')
   if v.get('source_native_path'):
    with rasterio.open(v['source_native_path']) as src:parent=src.read(1,window=Window(*v['source_local_window_xywh']));parentmask=src.read_masks(1,window=Window(*v['source_local_window_xywh']))>0
    if not np.array_equal(arr,parent,equal_nan=True):reasons.append('native_values_not_exact_source_window')
    if not np.array_equal(valid,parentmask&np.isfinite(parent)):reasons.append('native_validity_not_exact_source')
   pixel_rows.append(dict(sample_id=r['sample_id'],side=side,native_tif=str(path),algorithm_image=str(image),valid_fraction=float(valid.mean()),raw_pixel_sha256=rawhash,input_sha256=sha(image),input_std=float(vals.std()),display_black_fraction=float((vals==0).mean()),display_saturated_fraction=float((vals==255).mean()),constant_input=bool(np.ptp(vals)==0),native_shape=list(arr.shape),raw_dtype=str(arr.dtype),exact_parent_pixels=v.get('source_native_path') is not None))
   v.update(image=str(image),valid_fraction=float(valid.mean()),raw_pixel_sha256=rawhash,algorithm_input_sha256=sha(image),constant_input=bool(np.ptp(vals)==0));views.append(v)
  r['views']=views;r['image_only_input']=[r['image0'],r['image1']]
  a,b=[metadata[cid] for cid in r['acquisition_ids']];angle=delta(a.get('vendor_native_azimuth_degrees'),b.get('vendor_native_azimuth_degrees'))
  if a['provider']!=b['provider']:
   angle=None;r['angle_warning']='different vendor direction definitions; no unverified mixed-frame difference'
  r['azimuth_delta_deg']=angle;r['angle_bin']=binof(angle)
  r['angle_definition']='minimum circular difference of same-vendor actual observation azimuth fields; not platform heading, not 2D image rotation; target-LOS frame may differ by vendor'
  r['radar_target_LOS_delta_deg']=delta(a.get('radar_los_azimuth_degrees'),b.get('radar_los_azimuth_degrees'))
  r['incidence_delta_deg']=delta(a.get('incidence_degrees'),b.get('incidence_degrees'))
  r['geometry_resampling_applied']=False;r['GT_qualified']=False;r['allow_supervised_training']=False
  r['component']=component(r['site']);r['split']='Val' if a.get('split')=='Val' and b.get('split')=='Val' else 'Train'
  r['maritime_source_context']=r['site'].startswith('Port of ') or r['site'] in ['Panama Canal, Panama','Sunny_Isles','San_Francisco','Kingston_Australia','Melbourne, Australia']
  r['scene_class']='maritime_context_content_unverified' if r['maritime_source_context'] else 'other_land_structures_context'
  r['visible_vessels']={'side0':'unreviewed','side1':'unreviewed','human_confirmed':False}
  r.setdefault('annotatability','E');r.setdefault('gt_state','unannotated')
  r['qa_state']='failed' if reasons else 'passed';r['qa_failures']=reasons
  if reasons:failures.append(dict(sample_id=r['sample_id'],reasons=reasons))
  output.append(r)
  if (i+1)%100==0:print('REAL_NATIVE_READBACK',i+1,'/',len(pairs),flush=True)
 relationmap=defaultdict(list)
 for r in output:
  relationmap[r['relation_id']].append(r)
  if all(v.get('product_origin_xy') is not None for v in r['views']):
   k=tuple(sorted((cid,*v['product_origin_xy'],v['native_info']['width'],v['native_info']['height']) for cid,v in zip(r['acquisition_ids'],r['views'])));duplicate_groups[k].append(r['sample_id'])
 spatial=[]
 for rid,rs in relationmap.items():
  for aidx,a in enumerate(rs):
   for b in rs[:aidx]:
    bm={cid:v for cid,v in zip(b['acquisition_ids'],b['views'])};ovs=[];cross_grid=False
    for cid,av in zip(a['acquisition_ids'],a['views']):
     bv=bm[cid];ao,bo=av.get('product_origin_xy'),bv.get('product_origin_xy');za,zb=av['native_info'],bv['native_info']
     if ao is None or bo is None:ovs.append(None);continue
     ta=Affine(*za['transform'])*Affine.translation(-ao[0],-ao[1]);tb=Affine(*zb['transform'])*Affine.translation(-bo[0],-bo[1])
     same=za['crs']==zb['crs'] and np.allclose(tuple(ta),tuple(tb),atol=1e-8,rtol=0)
     if same:ovs.append(roi_overlap([*ao,512,512],[*bo,512,512]))
     else:
      cross_grid=True;crs=metric_crs(za);pa,pb=polygon(za,crs),polygon(zb,crs);inter=poly_area(poly_intersection(pa,pb));ovs.append(inter/max(1e-12,poly_area(pa)+poly_area(pb)-inter))
    if any(v is not None and v>.2 for v in ovs):spatial.append(dict(relation_id=rid,sample_a=a['sample_id'],sample_b=b['sample_id'],side_iou=ovs,cross_product_grid=cross_grid,state='historical_correlated_crops_retained' if a['source_kind']!='new_real_native_crop' and b['source_kind']!='new_real_native_crop' else 'new_spatial_correlation_requires_quarantine'))
 for r in spatial:
  if r['state']=='new_spatial_correlation_requires_quarantine':
   # Keep files and freeze records; do not move windows to replace a failed candidate.
   sid=r['sample_a'] if next(p for p in output if p['sample_id']==r['sample_a'])['source_kind']=='new_real_native_crop' else r['sample_b']
   p=next(p for p in output if p['sample_id']==sid);p['qa_state']='quarantined_spatial_duplicate';p['qa_failures'].append('same relation high spatial overlap in source/grid audit')
 used_ids={cid for r in output for cid in r['acquisition_ids']};actual_downloaded={json.loads(p.read_text())['acquisition_id'] for p in (ROOT/'acquisitions').glob('*/ASSET_*.json')}
 downloaded=used_ids|actual_downloaded|{r['acquisition_id'] for r in rows(latest('LOCAL_NATIVE_ASSETS_*.jsonl'))}
 for cid,a in metadata.items():a['local_pixel_status']='locally_available' if cid in downloaded else 'metadata_only';a['used_in_512_pair']=cid in used_ids
 acqpath=ROOT/f'REAL_ACQUISITION_MANIFEST_{t}.jsonl';jsonl(acqpath,sorted(metadata.values(),key=lambda a:a['acquisition_id']))
 allpath=ROOT/f'REAL_PAIR_MANIFEST_{t}.jsonl';jsonl(allpath,output)
 jsonl(ROOT/f'INPUT_PIXEL_QA_{t}.jsonl',pixel_rows);jsonl(ROOT/f'SPATIAL_DUPLICATE_AUDIT_{t}.jsonl',spatial)
 save(ROOT/f'REAL_PAIR_QA_{t}.json',dict(at=now(),passed=not failures,actual_pairs_read=len(output)-reused,views_read=2*(len(output)-reused),earlier_actual_readback_reused=reused,reused_QA_manifest=args.reuse_verified,technical_failures=failures,spatial_quarantined=sum(r['qa_state'].startswith('quarantined') for r in output),exact_coordinate_duplicates=[v for v in duplicate_groups.values() if len(v)>1],historical_correlated_crop_comparisons=len(spatial),new_images_are_exact_native_windows=True,old_whole_sar_rehashed=False,physical_GT_generated=0))
 for split in ['Train','Val']:
  keep=[r for r in output if r['split']==split and r['qa_state']=='passed'];jsonl(ROOT/f'REAL_{split}_UNLABELED_{t}.jsonl',keep)
  jsonl(ROOT/f'IMAGE_ONLY_{split}_{t}.jsonl',[dict(sample_id=r['sample_id'],image0=r['image0'],image1=r['image1']) for r in keep])
 train=[r for r in output if r['split']=='Train'];val=[r for r in output if r['split']=='Val']
 assert not ({r['component'] for r in train}&{r['component'] for r in val})
 assert not ({cid for r in train for cid in r['acquisition_ids']}&{cid for r in val for cid in r['acquisition_ids']})
 summary=dict(at=now(),actual_pairs=len(output),usable_pairs=sum(r['qa_state']=='passed' for r in output),new_pairs=sum(r['source_kind']=='new_real_native_crop' for r in output),legacy_pairs=197,relations=len(relationmap),acquisition_ids_in_pairs=len(used_ids),site_components=len({r['component'] for r in output}),newly_downloaded_acquisitions=len(actual_downloaded),metadata_inventory=len(metadata),metadata_only=sum(a['local_pixel_status']=='metadata_only' for a in metadata.values()),maritime_source_context_pairs=sum(r['maritime_source_context'] for r in output),angle_bins=dict(Counter(r['angle_bin'] for r in output)),split={s:dict(pairs=len(rr),relations=len({r['relation_id'] for r in rr}),acquisitions=len({cid for r in rr for cid in r['acquisition_ids']}),components=len({r['component'] for r in rr})) for s,rr in [('Train',train),('Val',val)]},qualified_GT=0,formal_precision_evaluable=0,source_acquisition_train_val_intersection=0,station_component_train_val_intersection=0,pair_manifest=str(allpath),acquisition_manifest=str(acqpath),source_angles_are_vendor_native_differences=True,ship_presence_not_inferred_from_port_names=True)
 save(ROOT/f'REAL_DATASET_COUNTS_{t}.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
