"""Controlled real-SAR diagnostic; immutable old inputs, no independent GT."""
import sys
sys.dont_write_bytecode=True
import os,json,time,csv,argparse,hashlib,shutil,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import defaultdict,Counter
import numpy as np,cv2,rasterio
from rasterio.windows import Window
from PIL import Image
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent
OLD=BASE/'RealSAR_AutoLabels_20261011_1149';ROT=BASE/'RealSAR_RotationStudy_20261011_1745'
SOURCE=BASE/'real_sar_pilot/multisource512_20261010_2352';MANUAL=BASE/'RealSAR_GeometryGT_20261011_0220'
sys.path.insert(0,str(ROT))
import study
from study import save,sha,clean,load_pair,sample_valid,regular_grid,PixelMap,Prediction,estimate,perspective,geometry_diagnostics,gate,spatial,grade,hull_mask
sys.path.insert(0,str(SOURCE/'code_snapshot_final'))
from build_real_pairs import display,to_db
METHODS=['sift','orb','lightglue'];REPRESENTATIONS=['linear','asinh','clahe']
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def selected():return lines(ROOT/'samples.jsonl')
def config():return read(ROOT/'config.json')
def jsonlines(p,rows):
    with Path(p).open('x') as f:
        for r in rows:f.write(json.dumps(clean(r),ensure_ascii=False,allow_nan=False)+'\n')
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def csvwrite(p,rows):
    if not rows:Path(p).write_text('');return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)

def normalize(raw,valid,kind,units):
    """Per-image fixed 1/99 percentile, never outcome-dependent. Valid zero retained."""
    if units not in ['linear_amplitude','linear_power','db']:raise ValueError('unknown radiometry')
    a=np.asarray(raw,np.float64);v=np.asarray(valid,bool)&np.isfinite(a)
    if not v.any():raise ValueError('no valid pixels')
    if units!='db' and np.any(a[v]<0):raise ValueError('negative linear SAR values: inspect units')
    values=a[v];scale=1.
    if kind=='asinh':
        if units!='db':
            positive=values[values>0];scale=float(np.median(positive)) if len(positive) else 1.
        else:scale=max(float(np.median(np.abs(values))),1.)
        a=np.arcsinh(a/scale)
    lo,hi=np.percentile(a[v],[1,99]);b=np.zeros(a.shape,np.uint8)
    if hi>lo:b[v]=np.rint(np.clip((a[v]-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    meta={'kind':kind,'percentiles':[1,99],'lo':float(lo),'hi':float(hi),'asinh_scale':scale,'units':units,'degenerate':bool(hi<=lo),'valid_zero_preserved':True,'independent_GT':False}
    return b,meta
def fill_invalid(image,valid):
    """Nearest valid pixel extension for preprocessing, declared validity unchanged."""
    if valid.all():return image.copy()
    if not valid.any():raise ValueError('no valid support')
    _,labels=cv2.distanceTransformWithLabels((~valid).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    lookup=np.zeros(int(labels.max())+1,np.uint8);lookup[labels[valid]]=image[valid]
    return np.where(valid,image,lookup[labels]).astype(np.uint8)
def representation(raw,valid,old,kind,units):
    if kind=='clahe':
        z=fill_invalid(old,valid);im=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(z)
        meta={'kind':'clahe','input':'frozen old PNG','clipLimit':2.,'tileGridSize':[8,8]}
    else:im,meta=normalize(raw,valid,kind,units)
    infer=fill_invalid(im,valid);published=np.where(valid,im,0).astype(np.uint8)
    return published,infer,meta

def prepare():
    rows=study.rows();seed=26101103
    sites=['Bingham Copper Mine','Hetch Hetchy Reservoir','Johnston, Iowa Farm','Melbourne, Australia','Port of Long Beach, California, United States','Port of Rotterdam, Netherlands','Kingston_Australia','San_Francisco']
    def overlap(r):return min(x['fraction_of_all_pixels'] for x in r['effective_geographic_overlap'])
    def tie(r):return hashlib.sha256((str(seed)+r['sample_id']).encode()).hexdigest()
    chosen=[]
    for site in sites:
        candidates=[r for r in rows if r['site_name']==site and r['look_pair_category']!='orientation_unknown']
        r=min(candidates,key=lambda r:(-overlap(r),tie(r)))
        chosen.append(dict(r,selection_reason='pre-fixed site; maximum native valid geographic overlap; seeded identity tie break; no algorithm outcomes',selection_scope='primary_metadata_only'))
    for site,target in [('Bingham Copper Mine',.6),('Hetch Hetchy Reservoir',.4),('Johnston, Iowa Farm',.5),('Port of Long Beach, California, United States',.7)]:
        candidates=[r for r in rows if r['site_name']==site and r['look_pair_category']=='different_look_verified' and r['pair_id'] not in {x['pair_id'] for x in chosen}]
        r=min(candidates,key=lambda r:(abs(overlap(r)-target),tie(r)))
        chosen.append(dict(r,selection_reason=f'pre-fixed second acquisition pair at site; overlap nearest {target}; seeded ID tie; no algorithm outcomes',selection_scope='primary_metadata_only'))
    assert len(chosen)==12 and len({r['pair_id'] for r in chosen})==12
    cfg={'fixed_at':now(),'seed':seed,'inference_seed':42,'representations':{'original':'frozen PNG/cache unmodified','linear':'per image native valid values, p1/p99, linear, round uint8','asinh':'asinh(raw / median positive valid raw), per image p1/p99, signed db compatible','clahe':'old PNG only, clipLimit2,8x8; invalid nearest-valid extension before processing'},'invalid_policy':'preserve declared finite/read-mask native validity; valid zeros stay valid; derived inference PNG fills invalid by nearest valid extension; invalid matches fail original post-QA','feature_geometry':'unchanged Phase2.5; Affine/full, max2048, seed42; original model and B threshold hashes','B_frozen':read(OLD/'config.json'),'context_size':1024,'context_source':'existing exact product URLs/local paths only; bounded read; no new discovery/full downloads','context_radiometry':'exact original display_db_range and original to_db; central512 equality required; inferred zero validity follows original source allow_valid_zero policy','context_direct':'1024 native full-resolution, max2048; frozen Affine/full thresholds in input pixels; outputs converted to original512; separate direct condition','context_fine':'only image-only accepted context Affine initializes warp of moving native512 into reference512; frozen local detector+Affine/full fitted AFTER match coordinates restored; no map/mask in coarse gate or H prediction','context_fine_invalid':'warp out-of-support pixels nearest extended; native validity only post-QA','geo_assisted_control':'reuse existing frozen geo_assisted512 records; no new geographic assisted tuning','local_geometry':'4x4 reference tiles >=8 matches, original basic affine; self-fit residual not GT; no global promotion','expansion_rule':'if a non-baseline factor produces at least1 new original512 general/high B eligible candidate with spatial support and finite accepted geometry, fix it for component-disjoint one-shot confirmation; otherwise stop at12; no full191 sweep','holdout_components':sorted({r['component'] for r in rows}-{r['component'] for r in chosen}),'limitations':'all191 previously examined; no sealed real GT; no heldout verified-different-look sites remain; confirmation can only be current-stage component-disjoint, not unbiased generalization','no_training':True,'no_independent_GT':True,'Success_metrics_computed':False}
    save(ROOT/'config.json',cfg);jsonlines(ROOT/'samples.jsonl',chosen)
    save(ROOT/'preregistration.json',{'at':now(),'samples_sha256':sha(ROOT/'samples.jsonl'),'config_sha256':sha(ROOT/'config.json'),'frozen':study.frozen(),'previous_index_sha256':sha(OLD/'dataset_index.jsonl'),'radiometry_results_seen':bool(os.environ.get('SAR_STUDY_REPLAY')),'context_results_seen':bool(os.environ.get('SAR_STUDY_REPLAY')),'execution_role':'replay_existing_locked_study' if os.environ.get('SAR_STUDY_REPLAY') else 'first_current_stage_run','algorithms_unchanged':True})
    # Necessary-input protection only, no repeated historical full-source audit.
    paths={OLD/'ENGINEERING_ACCEPTANCE.json',ROT/'ENGINEERING_ACCEPTANCE.json',MANUAL/'ENGINEERING_ACCEPTANCE.json',SOURCE/'DELIVERY_COMPLETE.json',OLD/'dataset_index.jsonl',OLD/'config.json',ROT/'config.json',OLD/'quality/model_grades.jsonl',OLD/'predictions/all_frozen/records.jsonl'}
    for r in chosen:
        paths.update(Path(r[k]) for k in ['reference_native_png','moving_native_png','reference_native_geotiff','moving_native_geotiff','provider_metadata_reference','provider_metadata_moving','geographic_quality_path','geographic_correspondences_path'])
        paths.update(Path(r['folder'])/n for n in ['metadata.json','reference_valid.png','moving_native_valid.png'])
    baseline=[r for r in lines(OLD/'quality/model_grades.jsonl') if r['sample_id'] in {x['sample_id'] for x in chosen}]
    paths.update(Path(r['cache_path']) for r in baseline)
    for r in rows:
        if r['has_B_pseudo']:paths.update(Path(r[k]) for k in ['pseudo_correspondences_path','pseudo_transform_path','pseudo_support_path','quality_report'])
    save(ROOT/'protected_inputs.json',{str(p):sha(p) for p in sorted(paths)})
    print('fixed samples',[(r['sample_id'],r['site_name']) for r in chosen],flush=True)

def stats(raw,valid,image):
    vals=raw[valid];im=image[valid];g=cv2.Sobel(image.astype(np.float32)/255,cv2.CV_32F,1,0);gy=cv2.Sobel(image.astype(np.float32)/255,cv2.CV_32F,0,1)
    means=[];cvs=[]
    for y in range(0,image.shape[0],64):
        for x in range(0,image.shape[1],64):
            m=valid[y:y+64,x:x+64];a=image[y:y+64,x:x+64][m].astype(float)/255;b=raw[y:y+64,x:x+64][m]
            if len(a):means.append(float(a.std()));cvs.append(float(b.std()/max(abs(float(b.mean())),1e-12)))
    return {'valid_fraction':float(valid.mean()),'raw_quantiles':np.quantile(vals,[0,.01,.5,.99,1]).tolist(),'raw_positive_fraction':float(np.mean(vals>0)),'raw_negative_fraction':float(np.mean(vals<0)),'raw_zero_fraction':float(np.mean(vals==0)),'raw_unique':len(np.unique(vals)),'png_unique':len(np.unique(im)),'png_zero_fraction':float(np.mean(im==0)),'png_255_fraction':float(np.mean(im==255)),'display_gradient_mean':float(np.hypot(g,gy)[valid].mean()),'local_display_std_median':float(np.median(means)),'raw_local_CV_median_heterogeneous_NOT_ENL':float(np.median(cvs))}
def data():
    folder=ROOT/'data';folder.mkdir(exist_ok=False);quality=[]
    for row in selected():
        sid=row['sample_id'];raw,valid,images,infos,mapper=load_pair(row);f=folder/sid;f.mkdir();meta=read(Path(row['folder'])/'metadata.json');q=dict(sample_id=sid,sensor=row['sensor'],pair_id=row['pair_id'],site_name=row['site_name'],component=row['component'],scene_class=row['scene_class'],look_pair_category=row['look_pair_category'],effective_overlap=row['effective_geographic_overlap'],scale_geometry=row['native_scale_geometry'],selection_reason=row['selection_reason'],metadata_angle_evidence={k:row[k] for k in row if any(s in k for s in ['azimuth_degrees','orbit_direction','observation_direction','incidence_degrees'])},distinct_acquisition=row['reference_acquisition_id']!=row['moving_acquisition_id'],different_dates=row['reference_utc']!=row['moving_utc'],static_structure_confirmed=False,physical_correspondence='needs independent review',views=[])
        for i,prefix in enumerate(['reference','moving']):
            units=row[prefix+'_input_units'];replayed=display(raw[i],valid[i],*meta['display_db_range'],units)
            assert np.array_equal(replayed,images[i]),'old PNG replay mismatch'
            db=to_db(raw[i],units);lo,hi=meta['display_db_range'];details=dict(stats(raw[i],valid[i],images[i]),info=infos[i],old_display_range_db=[lo,hi],old_mapping='round(255*clip((to_db(raw)-lo)/(hi-lo),0,1)); invalid0',PNG_byte_replay_equal=True,db_below_clip_fraction=float(np.mean(db[valid[i]]<lo)),db_above_clip_fraction=float(np.mean(db[valid[i]]>hi)),units=units)
            q['views'].append(details)
            for kind in REPRESENTATIONS:
                a,b,m=representation(raw[i],valid[i],images[i],kind,units);Image.fromarray(a).save(f/f'{kind}_{i}.png');Image.fromarray(b).save(f/f'{kind}_{i}_inference.png');save(f/f'{kind}_{i}.json',dict(m,statistics=stats(raw[i],valid[i],a)))
        save(f/'quality.json',q);quality.append(q)
    save(folder/'quality.json',quality);csvwrite(folder/'quality.csv',[dict({k:r[k] for k in ['sample_id','sensor','site_name','component','look_pair_category']},side=i,**v) for r in quality for i,v in enumerate(r['views'])]);print('validity and oldPNG replay',len(quality),flush=True)

def window_read(ds,xy,size=1024):
    x,y=map(int,xy);left=x-(size-512)//2;top=y-(size-512)//2
    if left<0 or top<0 or left+size>ds.width or top+size>ds.height:raise ValueError('full context window outside native product')
    win=Window(left,top,size,size);raw=ds.read(1,window=win);valid=(ds.read_masks(1,window=win)>0)&np.isfinite(raw)
    return raw,valid,ds.window_transform(win),[left,top]
def acquire_context(retry=False,attempt=1):
    folder=ROOT/(f'contexts_retry{attempt}' if retry else 'contexts');folder.mkdir(exist_ok=False);records=[]
    chosen=selected()
    if retry:
        old=read(ROOT/'contexts/SUMMARY.json')['records'];expected="KeyError('allow_declared_valid_zero')" if attempt==1 else "AssertionError('context center original pixels mismatch')";ids={r['sample_id'] for r in old if not r['available'] and expected==r.get('error')};chosen=[r for r in chosen if r['sample_id'] in ids]
    for row in chosen:
        start=time.perf_counter();sid=row['sample_id'];f=folder/sid;f.mkdir();meta=read(Path(row['folder'])/'metadata.json');r={'sample_id':sid,'available':False,'views':[],'new_acquisition_downloaded':False,'full_product_downloaded':False}
        try:
            for i,prefix in enumerate(['reference','moving']):
                origin=meta['reference_window_xy'] if i==0 else meta['moving_native_window_xy'];source=meta[prefix+'_path'];nativefile=row[prefix+'_native_geotiff']
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='20',GDAL_HTTP_CONNECTTIMEOUT='8',GDAL_HTTP_MAX_RETRY='0',AWS_NO_SIGN_REQUEST='YES',CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',GDAL_CACHEMAX=32):
                    with rasterio.open(source) as ds:
                        a,v,t,offset=window_read(ds,origin);crs=ds.crs;sourceinfo={'width':ds.width,'height':ds.height,'dtype':ds.dtypes[0],'block_shapes':ds.block_shapes,'nodata':ds.nodata,'crs':str(crs),'source_transform':list(ds.transform)[:6]}
                if row[prefix+'_input_units']!='db':v&=(a>=0 if row.get('allow_declared_valid_zero',False) else a>0)
                with rasterio.open(nativefile) as native:
                    central=a[256:768,256:768];cv=v[256:768,256:768];nv=native.read_masks(1)>0;na=native.read(1)
                    assert np.array_equal(cv,nv),'context center validity differs from frozen native'
                    assert np.array_equal(central[cv].astype(np.float32),na[nv],equal_nan=True),'valid original native pixels changed'
                    invalid_storage_differences=int(np.sum(central[~cv]!=na[~nv]))
                    assert np.allclose(list(t*rasterio.Affine.translation(256,256))[:6],list(native.transform)[:6],rtol=0,atol=1e-8),'context center geotransform mismatch'
                im=display(a,v,*meta['display_db_range'],row[prefix+'_input_units'])
                oldimage=np.array(Image.open(row[prefix+'_native_png']));assert np.array_equal(im[256:768,256:768],oldimage),'context old display512 mismatch'
                target=f/f'view{i}_float.tif'
                with rasterio.open(target,'w',driver='GTiff',height=1024,width=1024,count=1,dtype=a.dtype,crs=crs,transform=t,compress='deflate') as d:d.write(a,1);d.write_mask(v.astype(np.uint8)*255)
                Image.fromarray(im).save(f/f'view{i}.png');Image.fromarray(v.astype(np.uint8)*255).save(f/f'valid{i}.png')
                r['views'].append({'source':source,'source_metadata':sourceinfo,'window_xy':offset,'original512_offset':[256,256],'valid_native_pixels_equal':True,'native_valid_mask_equal':True,'invalid_storage_differences':invalid_storage_differences,'display_center_equal':True,'source_grid_unresampled':True,'sha256':sha(target),'valid_fraction':float(v.mean()),'size':[1024,1024]})
            r['available']=True
        except Exception as e:r['error']=repr(e)
        r['bounded_fetch_ms']=1000*(time.perf_counter()-start);save(f/'metadata.json',r);records.append(r);print('context',sid,'available',r['available'],r.get('error',''),flush=True)
    save(folder/'SUMMARY.json',{'records':records,'available':sum(r['available'] for r in records),'failed':sum(not r['available'] for r in records)})
def resolve_context():
    records=[dict(read(context_path(row['sample_id'])/'metadata.json'),context_folder=str(context_path(row['sample_id']))) for row in selected()]
    save(ROOT/'contexts_resolved.json',{'records':records,'available':sum(r['available'] for r in records),'failed':sum(not r['available'] for r in records),'correction':'optional zero policy defaultsFalse; compare valid raw pixels and validity independently of old -9999 storage encoding; initial/retry failures retained; summary reconstruction does not re-fetch data'})
def context_path(sid):
    for name in ['contexts_retry2','contexts_retry1','contexts']:
        p=ROOT/name/sid
        if (p/'metadata.json').exists():return p

def sync(torch):
    if torch.cuda.is_available():torch.cuda.synchronize()
def model_estimate(pred,shape=(512,512)):
    e=estimate(pred,shape,model='affine',seed=42);d=geometry_diagnostics(e,pred,shape,shape,42) if e.candidate_H is not None else {};return gate(e,d,'full'),d
def restore_prediction(pred,offset0,offset1):
    return Prediction(pred.method,pred.points0-np.asarray(offset0),pred.points1-np.asarray(offset1),reason=pred.reason,n_keypoints0=pred.n_keypoints0,n_keypoints1=pred.n_keypoints1,keypoints0=pred.keypoints0-np.asarray(offset0),keypoints1=pred.keypoints1-np.asarray(offset1))
def nativewarp(image,H):
    xy=study.pixel_grid((512,512)) if hasattr(study,'pixel_grid') else np.stack(np.meshgrid(np.arange(512),np.arange(512)),axis=-1).reshape(-1,2)
    q=perspective(H,xy).reshape(512,512,2).astype(np.float32);support=np.isfinite(q).all(-1)&(q[...,0]>=0)&(q[...,0]<=511)&(q[...,1]>=0)&(q[...,1]<=511)
    if not support.any():raise ValueError('coarse projection has no native moving support')
    warp=cv2.remap(image,q[...,0],q[...,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return fill_invalid(warp,support),support
def empty(row,method,condition,reason,elapsed=0,**extras):
    return dict({k:row[k] for k in ['sample_id','pair_id','sensor','site_name','component','scene_class','look_pair_category']},method=method,condition=condition,accepted=False,reason=reason,output_available=False,n_matches=0,n_inliers=0,inlier_ratio=0,pseudo_quality='unreliable_candidate',end_to_end_ms=elapsed,independent_GT=False,**extras)
def inferred_record(row,method,condition,pred,elapsed,torch,cache,valid,mapper):
    e,d=model_estimate(pred);r=study.diagnose(row,method,0,pred,e,d,elapsed,valid,mapper,cache)
    r.update(condition=condition,protocol='native',original_inference_protocol='image_only',peak_cuda_allocated_MB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0.)
    return r
def run(which):
    models,frozen,torch=study.initialize();folder=ROOT/'runs'/which;folder.mkdir(parents=True,exist_ok=False);(folder/'matches').mkdir();records=[]
    save(folder/'provenance.json',{'at':now(),'frozen':frozen,'config_sha256':sha(ROOT/'config.json'),'samples_sha256':sha(ROOT/'samples.jsonl'),'runner_sha256':sha(__file__),'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu','torch':torch.__version__,'initialization_excluded':True,'allocator_peak_scope':'includes loaded models and matching peak; not total system VRAM','GT_read_by_predictor':False})
    lookup={(r['sample_id'],r['method'],r['protocol']):r for r in lines(OLD/'quality/model_grades.jsonl')}
    for row in selected():
        sid=row['sample_id'];_,valid,old,_,mapper=load_pair(row)
        if which=='radiometry':
            for method in METHODS:
                for protocol,condition in [('native','original'),('geo_assisted','geo_assisted512_control')]:
                    r=dict(lookup[sid,method,protocol],condition=condition,baseline_reused=True,independent_GT=False);assert sha(r['cache_path'])==r['cache_sha256'];records.append(r)
            for kind in REPRESENTATIONS:
                for method in METHODS:
                    sync(torch);start=time.perf_counter()
                    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
                    raw,_,_,_,_=load_pair(row);a,b=[representation(raw[i],valid[i],old[i],kind,row[p+'_input_units'])[1] for i,p in enumerate(['reference','moving'])]
                    pred=models[method].predict(a,b);e,d=model_estimate(pred);sync(torch);elapsed=1000*(time.perf_counter()-start)
                    cache=folder/'matches'/f'{sid}_{kind}_{method}.npz';r=study.diagnose(row,method,0,pred,e,d,elapsed,valid,mapper,cache);r.update(condition=kind,peak_cuda_allocated_MB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0.,runtime_scope='includes local raw read, representation, features, affine/full; excludes H diagnostic and geographic QA')
                    records.append(r)
        else:
            contextfolder=context_path(sid);info=read(contextfolder/'metadata.json')
            for method in METHODS:
                if not info['available']:
                    records.extend(empty(row,method,c,'source_context_unavailable',error=info.get('error')) for c in ['context_direct','context_fine']);continue
                sync(torch);start=time.perf_counter()
                if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
                a,b=[np.array(Image.open(contextfolder/f'view{i}.png')) for i in [0,1]]
                pred=models[method].predict(a,b);coarse,cd=model_estimate(pred,(1024,1024));sync(torch);coarse_ms=1000*(time.perf_counter()-start)
                cache=folder/'matches'/f'{sid}_context_raw_{method}.npz';np.savez_compressed(cache,points0=pred.points0,points1=pred.points1,keypoints0=pred.keypoints0,keypoints1=pred.keypoints1,inlier_mask=coarse.inlier_mask if coarse.inlier_mask is not None else np.zeros(len(pred.points0),bool))
                pp=restore_prediction(pred,[256,256],[256,256]);inside=(pp.points0.min(1)>=0)&(pp.points0.max(1)<=511)&(pp.points1.min(1)>=0)&(pp.points1.max(1)<=511) if len(pp.points0) else np.zeros(0,bool)
                local=Prediction(method,pp.points0[inside],pp.points1[inside],n_keypoints0=pred.n_keypoints0,n_keypoints1=pred.n_keypoints1,keypoints0=pp.keypoints0[(pp.keypoints0.min(1)>=0)&(pp.keypoints0.max(1)<=511)],keypoints1=pp.keypoints1[(pp.keypoints1.min(1)>=0)&(pp.keypoints1.max(1)<=511)])
                cr=folder/'matches'/f'{sid}_context_direct_{method}.npz';r=inferred_record(row,method,'context_direct',local,coarse_ms,torch,cr,valid,mapper);r.update(coarse_accepted=coarse.accepted,coarse_reason=coarse.reason,coarse_matches=len(pred.points0),coarse_inliers=coarse.n_inliers,coarse_geometry=cd,coarse_candidate_H=coarse.candidate_H,coarse_cache=str(cache),coarse_end_to_end_ms=coarse_ms,context_model_restoration='localH=T(-256)@contextH@T(+256); local direct model re-fit on same raw context matches inside both512',n_keypoints0=len(local.keypoints0),n_keypoints1=len(local.keypoints1));records.append(clean(r))
                if not coarse.accepted:
                    records.append(empty(row,method,'context_fine','coarse_frozen_gate_rejected:'+coarse.reason,coarse_ms,coarse_accepted=False,coarse_matches=len(pred.points0),coarse_inliers=coarse.n_inliers,coarse_candidate_H=coarse.candidate_H,coarse_cache=str(cache)));continue
                T=np.array([[1,0,256],[0,1,256],[0,0,1.] ]);H=np.linalg.inv(T)@coarse.H@T
                sync(torch);fine_start=time.perf_counter()
                try:
                    warped,support=nativewarp(old[1],H);Image.fromarray(warped).save(folder/'matches'/f'{sid}_{method}_fine_input.png')
                    fine=models[method].predict(old[0],warped);points1=perspective(H,fine.points1);key1=perspective(H,fine.keypoints1)
                    native=Prediction(method,fine.points0,points1,n_keypoints0=fine.n_keypoints0,n_keypoints1=fine.n_keypoints1,keypoints0=fine.keypoints0,keypoints1=key1)
                    e,d=model_estimate(native);sync(torch);fine_ms=1000*(time.perf_counter()-fine_start)
                    cache=folder/'matches'/f'{sid}_context_fine_{method}.npz';r=study.diagnose(row,method,0,native,e,d,coarse_ms+fine_ms,valid,mapper,cache);r.update(condition='context_fine',coarse_accepted=True,coarse_matches=len(pred.points0),coarse_inliers=coarse.n_inliers,coarse_native_H=H,coarse_candidate_H=coarse.candidate_H,coarse_cache=str(folder/'matches'/f'{sid}_context_raw_{method}.npz'),coarse_end_to_end_ms=coarse_ms,fine_end_to_end_ms=fine_ms,fine_projection_support=float(support.mean()),peak_cuda_allocated_MB=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0.,runtime_scope='actual sequential coarse + warp/native512 fine; excludes diagnostic and QA',protocol='native',original_inference_protocol='image_only_context_coarse_to_fine');records.append(clean(r))
                except ValueError as err:records.append(empty(row,method,'context_fine','no_coarse_projection_support',coarse_ms,error=repr(err)))
        print(which,sid,'done',flush=True)
    jsonlines(folder/'records.jsonl',records);save(folder/'SUMMARY.json',{'records':len(records),'runtime_errors':sum(r.get('reason')=='runtime_error' for r in records),'C':0,'Success_metrics_computed':False})

def geometric():
    folder=ROOT/'geometry';folder.mkdir(exist_ok=False);records=[];locals=[]
    allrecords=lines(ROOT/'runs/radiometry/records.jsonl')+lines(ROOT/'runs/context/records.jsonl')
    byid={r['sample_id']:r for r in selected()}
    for r in allrecords:
        if not r.get('cache_path') or r['condition']=='geo_assisted512_control':continue
        row=byid[r['sample_id']];_,valid,_,_,mapper=load_pair(row);cache=np.load(r['cache_path']);p0=cache['points0'];p1=cache['points1_native'];pred=Prediction(r['method'],p0,p1)
        for model in ['affine','homography']:
            start=time.perf_counter();est=estimate(pred,(512,512),model=model,seed=42);diag=geometry_diagnostics(est,pred,(512,512),(512,512),42) if est.candidate_H is not None else {};g=gate(est,diag,'full');iv=est.inlier_mask if est.inlier_mask is not None else np.zeros(len(p0),bool);nv=iv&sample_valid(valid[0],p0)&sample_valid(valid[1],p1)
            diff=np.linalg.norm(mapper(p0[nv])-p1[nv],axis=1);fit=np.linalg.norm(perspective(est.candidate_H,p0[nv])-p1[nv],axis=1) if est.candidate_H is not None else np.array([])
            records.append(clean(dict(sample_id=r['sample_id'],condition=r['condition'],method=r['method'],model=model,n_matches=len(p0),n_inliers=est.n_inliers,inlier_ratio=est.inlier_ratio,accepted=g.accepted,reason=g.reason,candidate_H=est.candidate_H,native_valid_inliers=int(nv.sum()),geographic_difference_median_px=float(np.median(diff)) if len(diff) else None,geographic_difference_p90_px=float(np.quantile(diff,.9)) if len(diff) else None,fit_resubstitution_p90_NOT_GT=float(np.quantile(fit,.9)) if len(fit) else None,geometry=diag,extra_ms=1000*(time.perf_counter()-start),independent_GT=False)))
        for gy in range(4):
            for gx in range(4):
                m=(p0[:,0]>=gx*128)&(p0[:,0]<(gx+1)*128)&(p0[:,1]>=gy*128)&(p0[:,1]<(gy+1)*128)&sample_valid(valid[0],p0)&sample_valid(valid[1],p1)
                if m.sum()<8:continue
                e=estimate(Prediction(r['method'],p0[m],p1[m]),(512,512),model='affine',seed=42);inliers=e.inlier_mask if e.inlier_mask is not None else np.zeros(m.sum(),bool);dd=np.linalg.norm(mapper(p0[m][inliers])-p1[m][inliers],axis=1)
                locals.append(clean(dict(sample_id=r['sample_id'],condition=r['condition'],method=r['method'],tile_x=gx,tile_y=gy,model='local_affine_NOT_global_GT',n_matches=int(m.sum()),n_inliers=e.n_inliers,accepted_basic=e.accepted,reason=e.reason,candidate_H=e.candidate_H,geographic_difference_median_px=float(np.median(dd)) if len(dd) else None,independent_GT=False)))
    jsonlines(folder/'models.jsonl',records);csvwrite(folder/'models.csv',records);jsonlines(folder/'local.jsonl',locals);csvwrite(folder/'local.csv',locals)
    # Contradiction measured on support centers, never global accuracy.
    contradictions=[]
    groups=defaultdict(list)
    for r in locals:
        if r['accepted_basic']:groups[r['sample_id'],r['condition'],r['method']].append(r)
    for key,group in groups.items():
        for i,a in enumerate(group):
            for b in group[i+1:]:
                p=np.array([[a['tile_x']*128+63.5,a['tile_y']*128+63.5],[b['tile_x']*128+63.5,b['tile_y']*128+63.5]])
                diff=np.linalg.norm(perspective(a['candidate_H'],p)-perspective(b['candidate_H'],p),axis=1)
                contradictions.append(dict(sample_id=key[0],condition=key[1],method=key[2],tileA=[a['tile_x'],a['tile_y']],tileB=[b['tile_x'],b['tile_y']],cross_support_extrapolation_difference_px=diff.tolist(),NOT_GT=True))
    save(folder/'local_contradictions.json',contradictions);print('model records',len(records),'local tiles',len(locals),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','data','acquire_context','retry_context','retry_context2','resolve_context','radiometry','context','geometry']);a=p.parse_args()
    if a.stage in ['radiometry','context']:run(a.stage)
    elif a.stage=='geometry':geometric()
    elif a.stage=='retry_context':acquire_context(True)
    elif a.stage=='retry_context2':acquire_context(True,2)
    else:globals()[a.stage]()
if __name__=='__main__':main()
