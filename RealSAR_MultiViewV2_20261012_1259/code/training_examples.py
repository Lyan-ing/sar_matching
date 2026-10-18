"""Four-layer indexes and tiny demonstrations; NO learning or invented real GT."""
import sys
sys.dont_write_bytecode=True
import json,argparse
from pathlib import Path
import numpy as np
from PIL import Image
from geo_tools import *
def export_real_local(region,doc,task,out,seed=26101201):
    # Called only by independent data construction, never by a predictor.
    from evaluation import qualified_quality
    q=qualified_quality(doc,task)
    if q['status']!='gt_qualified':raise ValueError('No qualified real static corresponding training positives')
    if task['sample_id']!=region['sample_id']:raise ValueError('Not this mother-region native GT')
    if task.get('split') not in ['train','diagnostic_development_seen']:raise ValueError('Never construct training positives from val/final-test check evidence')
    for i in [0,1]:
        if (task['width'+str(i)],task['height'+str(i)])!=(region['views'][i]['native_info']['width'],region['views'][i]['native_info']['height']):raise ValueError('GT native frame mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=False);eligible=set(q['eligible_point_ids']);rng=np.random.default_rng(seed);rows=[]
    for p in doc['points']:
        if p['point_id'] not in eligible or p['role']!='control':continue
        for k in range(2):
            origins=[];valid=True
            for i in [0,1]:
                a,v,z=read_tif(ROOT/'regions'/region['sample_id']/f'native{i}.tif');center=np.array(p['xy'+str(i)]);origin=np.floor(center-64+rng.integers(-8,9,2)).astype(int);x,y=origin
                if x<0 or y<0 or x+128>a.shape[1] or y+128>a.shape[0] or not v[y:y+128,x:x+128].all():valid=False;break
                origins.append(origin)
            if not valid:continue
            for i,origin in enumerate(origins):
                a,v,z=read_tif(ROOT/'regions'/region['sample_id']/f'native{i}.tif');x,y=origin;zz=dict(z,width=128,height=128,transform=list(Affine(*z['transform'])*Affine.translation(int(x),int(y)))[:6]);write_tif(out/f"{p['point_id']}_{k}_{i}.tif",a[y:y+128,x:x+128],v[y:y+128,x:x+128],zz)
            rows.append(dict(sample_id=region['sample_id'],point_id=p['point_id'],role=p['role'],patch_origins_xy=origins,point_in_patch=[np.array(p['xy'+str(i)])-origins[i] for i in [0,1]],declared_uncertainty=[p['sigma0_px'],p['sigma1_px']],component=region['component'],acquisitions=[v['acquisition_id'] for v in region['views']],count_as_new_acquisition=False,seed=seed))
    jsonl(out/'index.jsonl',rows);return rows
def run(resume=False):
    out=ROOT/'training_examples';out.mkdir(exist_ok=resume);regions=lines(ROOT/'regions.jsonl');rows=[]
    # Use an existing sufficiently large native SAR mother image, never upscale.
    source=next(r for r in regions if min(r['views'][0]['native_info'][k] for k in ['height','width'])>=800 and read_tif(ROOT/'regions'/r['sample_id']/'native0.tif')[1].all())
    sys.path.insert(0,str(BASE/'MaritimeSARRegDataset'));from sarreg.synthesis import synthesize,load_cfg
    cfg=load_cfg(BASE/'MaritimeSARRegDataset/configs/easy.yaml');cfg.update(output_size=512,allow_source_upsampling=False,output_size_multiple=32,augmentation={'enabled':False});r=dict(dataset='RealSAR_MultiViewV2',source_id=source['views'][0]['acquisition_id'],source_group=source['component'],image_path=str(ROOT/'regions'/source['sample_id']/'native_db0.png'),split='diagnostic_development_seen',annotations=[],scene_type=source['site_name'])
    synthetic=out/'single_image_known_geometry'
    if not synthetic.exists():synthetic.mkdir();synthesize(r,cfg,synthetic,seed=26101201)
    elif not resume:raise FileExistsError('Existing synthetic demonstration refused')
    meta=json.loads((synthetic/'meta.json').read_text());assert meta['seed']==26101201 and meta['source_image_path']==r['image_path']
    assert meta['source_resize_factors_xy']==[1.,1.] and meta['output_size']==512
    rows.append(dict(layer='single_SAR_known_2D_transform',sample_id=source['sample_id'],component=source['component'],source_acquisition=r['source_id'],path=str(synthetic),supervision='exact synthetic geometry ONLY; not real cross-view',allow_training_now=False,split='diagnostic_development_seen'))
    for region in regions:
        zs=[v['native_info'] for v in region['views']];p=geo.regular_grid((zs[0]['height'],zs[0]['width']),8);q=PixelMap(Affine(*zs[0]['transform']),zs[0]['crs'],Affine(*zs[1]['transform']),zs[1]['crs'])(p);v=[read_tif(ROOT/'regions'/region['sample_id']/f'native{i}.tif')[1] for i in [0,1]];keep=sample_valid(v[0],p)&sample_valid(v[1],q)
        weak=out/('weak_'+region['sample_id']+'.json');save(weak,dict(points0_geographic_seed=p[keep],predicted_points1_geographic_only=q[keep],search_radius_moving_native_px=64,not_strong_positive=True,no_physical_correspondence_evidence=True))
        rows.append(dict(layer='geographic_weak_search',sample_id=region['sample_id'],component=region['component'],path=str(weak),positive_label=None,split='diagnostic_development_seen'))
        rows.append(dict(layer='difficult_unknown',sample_id=region['sample_id'],component=region['component'],label='unknown_not_negative',allow_strong_positive=False,allow_confirmed_negative=False))
    # True disjoint geographical footprints, not "unknown is a negative".
    a=regions[0];b=next(r for r in regions if r['site_name']!=a['site_name']);crs=a['metric_crs'];p0=polygon(a['views'][0]['native_info'],crs);p1=polygon(b['views'][0]['native_info'],crs);assert poly_area(poly_intersection(p0,p1))==0
    rows.append(dict(layer='geographic_no_overlap_negative',samples=[a['sample_id'],b['sample_id']],components=[a['component'],b['component']],native_paths=[str(ROOT/'regions'/r['sample_id']/'native0.tif') for r in [a,b]],geographic_intersection_area_m2=0,physical_local_negative_claim=False,split='diagnostic_development_seen'))
    jsonl(out/'index.jsonl',rows);save(out/'real_cross_acquisition_positives.json',dict(count=0,reason='No V2 independent qualified human geometry; readability pending',future_export_function='training_examples.export_real_local',training_started=False))
    save(out/'schema.json',dict(required_fields=['layer','real_acquisitions','source_component','site','split_role','supervision_kind','uncertainty','transform_or_search_region','parent_native_window','license'],unknown_not_negative=True,augmentations_not_independent=True,all_current_examples_diagnostic=True,real_positive_gate='gt_qualified + same V2 native frame + independent exposure screening'))
    print('TRAINING_INTERFACE_DEMOS',len(rows),'real positives 0; no training',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-export',action='store_true');a=p.parse_args();run(a.resume_export)
