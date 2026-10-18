"""Acquisition-first V2 pilot. No matcher, pseudo-label or human-point reads."""
import sys
sys.dont_write_bytecode=True
import argparse,json,time,math,re
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np,cv2,rasterio
from rasterio.windows import Window
from rasterio.warp import transform
from affine import Affine
from PIL import Image
from geo_tools import *
DATA=BASE/'real_sar_pilot/multisource512_20261010_2352'
CONTEXT=BASE/'RealSAR_RepresentationContext_20261011_1903'
def angular(a,b):return None if a is None or b is None else abs((a-b+180)%360-180)
def utc(s):
    # Python 3.10 accepts microseconds; retain nanosecond original strings in metadata.
    return datetime.fromisoformat(re.sub(r'(\.\d{6})\d+',r'\1',s.replace('Z','+00:00')))
def gap(a,b):return abs((utc(a)-utc(b)).total_seconds())/86400
def profiles():
    result={}
    for folder in ['quality/source_rasters','quality_retry1/source_rasters','ports_recovery/quality/source_rasters','melbourne_zero_aware/quality/source_rasters','melbourne_zero_aware/quality_retry1/source_rasters','s1_sf_extension/quality_retry1/source_rasters']:
        for p in (DATA/folder).glob('*.json'):
            z=json.loads(p.read_text());result[z['url']]=dict(z['profile'],profile_record=str(p))
    return result
def catalog():
    rows=lines(DATA/'dataset_index.jsonl');pairs={};contexts=json.loads((CONTEXT/'contexts_resolved.json').read_text())['records'];sources=profiles();local=defaultdict(list)
    for cx in contexts:
        if not cx['available']:continue
        for i,z in enumerate(cx['views']):
            local[z['source']].append(dict(path=str(Path(cx['context_folder'])/f'view{i}_float.tif'),origin=z['window_xy'],sample_id=cx['sample_id'],source_metadata=z['source_metadata']))
    for r in rows:pairs.setdefault(r['pair_id'],r)
    inventory=[]
    for pid,r in pairs.items():
        z=dict(pair_id=pid,sensor=r['sensor'],site_name=r['site_name'],component=r['component'],old_split=r['split'],scene_class=r['scene_class'],look=r['look_pair_category'],days=gap(r['reference_utc'],r['moving_utc']),azimuth_delta_deg=angular(r['reference_azimuth_degrees'],r['moving_azimuth_degrees']),distinct=r['reference_acquisition_id']!=r['moving_acquisition_id'],views=[])
        for side in ['reference','moving']:
            url=r[side+'_asset_url'];p=sources.get(url);m={k:r.get(side+'_'+k) for k in ['acquisition_id','utc','orbit_direction','observation_direction','azimuth_degrees','incidence_degrees','polarizations','input_units','license','license_url','physical_resolution_metadata']}
            m.update(asset_url=url,profile=p,local_contexts=local.get(url,[]),processing='RTC gamma0' if r['sensor']=='Sentinel-1' else 'GEC detected',elevation_model='not established from retained metadata; needs DEM/product documentation',original_local_product=Path(url).is_file() if not url.startswith(('http','s3:')) else False)
            z['views'].append(m)
        if all(v['profile'] for v in z['views']):
            crs=metric_crs(z['views'][0]['profile']);p0,p1=[polygon(v['profile'],crs) for v in z['views']];inter=poly_intersection(p0,p1)
            z.update(metric_crs=crs,product_overlap_fractions=[poly_area(inter)/max(poly_area(p0),1e-12),poly_area(inter)/max(poly_area(p1),1e-12)],product_intersection=mapping(inter) if len(inter) else None)
        inventory.append(z)
    jsonl(ROOT/'acquisition_pair_inventory.jsonl',inventory)
    jsonl(ROOT/'acquisition_inventory.jsonl',lines(DATA/'downloaded_acquisitions.jsonl'))
    return rows,pairs,local,{r['pair_id']:r for r in inventory}
def context_from_local(record,row):
    arrays=[];valid=[];zs=[];provenance=[]
    for side,c in enumerate(record):
        a,v,z=read_tif(c['path']);arrays.append(a);valid.append(v);zs.append(z)
        provenance.append(dict(actual_read='existing_unresampled_context',path=c['path'],sha256=sha(c['path']),parent_original_asset=c.get('url'),original_window_xywh=c['origin']+[z['width'],z['height']],native_no_resampling=True))
    return arrays,valid,zs,provenance
def native_context(row,locals_,inv,cfg):
    # First prefer both views from the same prior large-scene acquisition pair.
    urls=[row[x+'_asset_url'] for x in ['reference','moving']]
    choices=[locals_.get(u,[]) for u in urls];compatible=[]
    for c0 in choices[0]:
        for c1 in choices[1]:
            with rasterio.open(c0['path']) as d0,rasterio.open(c1['path']) as d1:z0,z1=info(d0),info(d1)
            crs=metric_crs(z0);p0,p1=polygon(z0,crs),polygon(z1,crs);q=poly_intersection(p0,p1);area=poly_area(q)
            if area>0 and min(z0['width'],z0['height'],z1['width'],z1['height'])>512:
                compatible.append((min(area/max(poly_area(p0),1),area/max(poly_area(p1),1)),c0,c1))
    if compatible:
        overlap,c0,c1=max(compatible,key=lambda z:z[0])
        if overlap>=.4:
            return context_from_local([dict(c0,url=urls[0]),dict(c1,url=urls[1])],row)
    if not cfg['allow_bounded_existing_product_reads']:raise ValueError('No adequate overlapping local native context; remote reads disabled')
    if not inv.get('product_intersection'):raise ValueError('No original-profile common geography')
    p=np.array(inv['product_intersection']['coordinates'][0][:-1]);center=(p.min(0)+p.max(0))/2;crs=inv['metric_crs']
    arrays=[];valid=[];zs=[];provenance=[]
    for i,(side,url) in enumerate(zip(['reference','moving'],urls)):
        timeout=str(cfg['per_asset_remote_timeout_seconds'])
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT=timeout,GDAL_HTTP_CONNECTTIMEOUT='8',GDAL_HTTP_MAX_RETRY='0',AWS_NO_SIGN_REQUEST='YES',GDAL_CACHEMAX=32,CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif'):
            with rasterio.open(url) as ds:
                if ds.crs is None:raise ValueError('Original CRS missing')
                x,y=transform(crs,ds.crs.to_string(),[center[0]],[center[1]]);q=(~ds.transform)*(x[0],y[0]);size=min(cfg['native_context_max_side'],ds.width,ds.height)
                if size<=512:raise ValueError('Original source provides no larger native context')
                left=int(np.clip(round(q[0]-size/2),0,ds.width-size));top=int(np.clip(round(q[1]-size/2),0,ds.height-size))
                a,v,z=read_window(ds,Window(left,top,size,size),row[side+'_input_units'],row.get('allow_declared_valid_zero',False))
                actual_profile=info(ds);saved=inv['views'][i]['profile']
                if saved and (saved['width']!=ds.width or saved['height']!=ds.height or not np.allclose(saved['transform'],z['transform'] if left==top==0 else list(ds.transform)[:6],atol=1e-8,rtol=0)):raise ValueError('Original raster profile changed since release')
        arrays.append(a);valid.append(v);zs.append(z);provenance.append(dict(actual_read='bounded_NEW_window_from_existing_asset_not_full_download',parent_original_asset=url,original_window_xywh=[left,top,size,size],native_no_resampling=True,original_profile=actual_profile))
    return arrays,valid,zs,provenance
def choose_roi(arrays,valid,zs,units,cfg):
    crs=metric_crs(zs[0]);polys=[polygon(z,crs) for z in zs];joint=poly_intersection(*polys)
    if len(joint)<3:raise ValueError('Native contexts have no joint geography')
    span=np.min(joint.max(0)-joint.min(0));side=span*.6
    if side<=0:raise ValueError('Degenerate ground intersection')
    # Fixed nine ROI positions and deterministic descending sizes. This selects
    # image structure worth reviewing, not descriptor correspondences or GT.
    options=[];center=(joint.min(0)+joint.max(0))/2
    for scale in [1.,.75,.5]:
        s=side*scale
        for fy in [-.15,0,.15]:
            for fx in [-.15,0,.15]:
                c=center+np.array([fx,fy])*span;roi=box(c[0]-s/2,c[1]-s/2,c[0]+s/2,c[1]+s/2)
                if not poly_inside(roi,joint):continue
                dest=dict(crs=crs,transform=[s/256,0,roi[0,0],0,-s/256,roi[2,1]],width=256,height=256)
                scores=[];fractions=[]
                for a,v,z,u in zip(arrays,valid,zs,units):
                    r,m=sample_grid(a,v,z,dest);im,_=display(r,m,'db',u);blur=cv2.GaussianBlur(im.astype(np.float32)/255,(0,0),2);edges=cv2.Canny(np.rint(blur*255).astype(np.uint8),30,90)>0
                    fractions.append(float(m.mean()));scores.append(float(edges[m].mean())+float(blur[m].std())*.1 if m.any() else 0.)
                if min(fractions)<cfg['min_joint_valid_fraction']:continue
                windows=[roi_window(z,roi,crs) for z in zs]
                if min(min(w.width,w.height) for w in windows)<cfg['min_native_ROI_side_pixels']:continue
                options.append((min(scores),s,roi,dest,fractions,scores))
        if options:break
    if not options:raise ValueError('No sufficiently large native ROI with declared joint-valid coverage')
    score,s,roi,_,fractions,scores=max(options,key=lambda z:(z[0],z[1],-z[2][0,0]))
    if score<.003:raise ValueError('Low static-structure screening proxy; not padding candidate count with uniform scene')
    n=cfg['common_fov_size'];dest=dict(crs=crs,transform=[s/n,0,roi[0,0],0,-s/n,roi[2,1]],width=n,height=n)
    return roi,crs,dest,dict(score=score,worst_view_score=score,view_scores=scores,valid_preview_fractions=fractions,structure_not_human_evidence=True,considered_positions=len(options))
def build_pair(row,inv,local,cfg):
    arrays,valid,zs,provenance=native_context(row,local,inv,cfg)
    units=[row[x+'_input_units'] for x in ['reference','moving']];roi,crs,dest,ranking=choose_roi(arrays,valid,zs,units,cfg)
    sid=row['pair_id']+'__mother_000';folder=ROOT/'regions'/sid;folder.mkdir(parents=True,exist_ok=False)
    native=[];native_valid=[];native_info=[];views=[]
    for i,(a,v,z,u,prov) in enumerate(zip(arrays,valid,zs,units,provenance)):
        write_tif(folder/f'context{i}.tif',a,v,z)
        win=roi_window(z,roi,crs);x,y,w,h=map(int,[win.col_off,win.row_off,win.width,win.height]);b=a[y:y+h,x:x+w].copy();m=v[y:y+h,x:x+w].copy();t=Affine(*z['transform'])*Affine.translation(x,y);nz=dict(z,width=w,height=h,transform=list(t)[:6]);write_tif(folder/f'native{i}.tif',b,m,nz)
        # ROI mask is geographic coverage, separate from technical native validity.
        yy,xx=np.indices((h,w));world=center_metric(nz,np.c_[xx.ravel(),yy.ravel()],crs);roi_m=((world[:,0]>=roi[0,0])&(world[:,0]<=roi[2,0])&(world[:,1]>=roi[0,1])&(world[:,1]<=roi[2,1])).reshape(h,w)
        Image.fromarray((roi_m&m).astype(np.uint8)*255).save(folder/f'roi_valid{i}.png')
        common,cm=sample_grid(a,v,z,dest);write_tif(folder/f'common{i}.tif',common,cm,dest)
        displays={}
        for scope,raw,mask in [('context',a,v),('native',b,m),('common',common,cm)]:
            for kind in ['linear','db']:
                im,disp=display(raw,mask,kind,u);Image.fromarray(im).save(folder/f'{scope}_{kind}{i}.png');displays[scope+'_'+kind]=disp
        Image.fromarray(m.astype(np.uint8)*255).save(folder/f'native_valid{i}.png');Image.fromarray(cm.astype(np.uint8)*255).save(folder/f'common_valid{i}.png')
        local_origin=np.array([x,y]);orig=np.array(prov['original_window_xywh'][:2]);original_xy=(orig+local_origin).astype(int)
        mapper=PixelMap(Affine(*nz['transform']),nz['crs'],Affine(*dest['transform']),crs)
        points=geo.regular_grid((h,w),8);roundtrip=geo.roundtrip(mapper,points)
        if not roundtrip['finite'] or roundtrip['max_px']>1e-4:raise ValueError('Native/input coordinate roundtrip failed')
        common_spacing=dest['transform'][0];metric=geo.metric_geometry(nz)
        views.append(dict(side=i,acquisition_id=row[['reference','moving'][i]+'_acquisition_id'],context_info=z,native_info=nz,common_info=dest,context_native_window_xywh=[x,y,w,h],original_native_window_xywh=original_xy.tolist()+[w,h],context_provenance=prov,metric_geometry=metric,
            native_to_common_affine=mapper.affine_matrix(),native_to_common_crs_mapping=dict(A=nz,B=dest,pixel_centers=True,nonlinear_CRS=mapper.affine_matrix() is None),roundtrip_numeric_not_GT=roundtrip,
            common_spacing_m=common_spacing,common_to_native_spacing_ratio=[common_spacing/metric['spacing_m_x'],common_spacing/metric['spacing_m_y']],resampling='linear DN/power bilinear; mask all bilinear support valid; no scatterer GT',display=displays))
        native.append(b);native_valid.append(m);native_info.append(nz)
    overlaps=overlap_masks(native_valid[0],native_info[0],native_valid[1],native_info[1])
    for i,m in enumerate(overlaps):Image.fromarray(m.astype(np.uint8)*255).save(folder/f'geographic_overlap{i}.png')
    counts=[dict(fraction_all=float(m.mean()),fraction_valid=float(m.sum()/max(v.sum(),1))) for m,v in zip(overlaps,native_valid)]
    common_masks=[np.array(Image.open(folder/f'common_valid{i}.png'))>0 for i in [0,1]];jointcommon=common_masks[0]&common_masks[1]
    if jointcommon.mean()<cfg['min_joint_valid_fraction']:raise ValueError('Final Common-FOV joint validity below preregistered threshold')
    Image.fromarray(jointcommon.astype(np.uint8)*255).save(folder/'common_joint_geographic_valid.png')
    roi_geo=transform(crs,'EPSG:4326',roi[:,0].tolist(),roi[:,1].tolist());feature=dict(type='FeatureCollection',features=[dict(type='Feature',geometry=mapping(np.column_stack(roi_geo)),properties=dict(kind='common_ground_ROI',geographic_NOT_scatterer_GT=True))]);save(folder/'roi.geojson',feature)
    metadata=dict(sample_id=sid,pair_id=row['pair_id'],sensor=row['sensor'],site_name=row['site_name'],component=inv['v2_component'],old_component=row['component'],split='diagnostic_development_seen',look_pair_category=row['look_pair_category'],acquisition_metadata=inv,views=views,ground_ROI_metric=mapping(roi),metric_crs=crs,ground_edge_m=dest['transform'][0]*dest['width'],native_geographic_overlap=counts,common_joint_geographic_valid_fraction=float(jointcommon.mean()),ranking=ranking,
        static_structure_candidates='roads/intersections, grounded quay/breakwater edges or permanent facilities: human visibility NOT confirmed; avoid ships, container stacks, roofs and shadows',
        risks=['layover/shadow and height-dependent scatterer displacements unknown','product-geocoding ground ROI is not 3D visibility or dense correspondence','multi-month time gap may include construction, moving containers and vessels','DN quantization/calibration limits retained; no detail recovered by stretch'],human_readability='pending',independent_GT=False,gt_qualified=False,external_height_or_reference_needed='not independently constrained; recommended for elevated structures',files={str(p.relative_to(folder)):sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    save(folder/'metadata.json',metadata)
    return metadata
def components(inventory):
    # Exact dependence graph: shared acquisition, station, or legacy component.
    parent={r['pair_id']:r['pair_id'] for r in inventory}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    keys={}
    for r in inventory:
        deps=['site:'+r['site_name'].casefold(),'old_component:'+r['component']]+['acquisition:'+r['sensor']+':'+v['acquisition_id'] for v in r['views']]
        for k in deps:
            if k in keys:parent[find(r['pair_id'])]=find(keys[k])
            else:keys[k]=r['pair_id']
    roots=sorted({find(p) for p in parent});ids={p:'v2_component_'+str(roots.index(find(p))).zfill(3) for p in parent}
    for r in inventory:r['v2_component']=ids[r['pair_id']]
    return ids
def run():
    cfg=json.loads((ROOT/'protocol.json').read_text());pre=json.loads((ROOT/'preregistration.json').read_text());assert sha(ROOT/'protocol.json')==pre['config_sha256']
    if (ROOT/'regions').exists():raise FileExistsError('Output exists; copy code/config to a new independent V2 directory')
    rows,pairs,local,inventory=catalog();isolation=components(list(inventory.values()));save(ROOT/'source_isolation.json',dict(pair_component=isolation,all_seen=True,final_unseen_test=0,random_chip_split_forbidden=True,components=len(set(isolation.values()))))
    results=[];failures=[]
    for pid in cfg['selection']['pair_priority']:
        start=time.perf_counter();row=pairs[pid];inv=inventory[pid]
        try:
            if not inv['distinct']:raise ValueError('Same original acquisition')
            if inv['look']=='different_look_verified':
                delta=inv['azimuth_delta_deg'];changed=any(str(inv['views'][0][k]).lower()!=str(inv['views'][1][k]).lower() for k in ['orbit_direction','observation_direction'])
                if (delta is None or delta<30) and not changed:raise ValueError('Direction identity not sufficiently explicit for different-view mother')
            result=build_pair(row,inv,local,cfg['selection']);results.append(result);print('BUILT',pid,'look',inv['look'],'native',[v['native_info']['width'] for v in result['views']],'joint',result['common_joint_geographic_valid_fraction'],flush=True)
        except Exception as e:
            fail=dict(pair_id=pid,site=inv['site_name'],look=inv['look'],error=repr(e),elapsed_seconds=time.perf_counter()-start,not_replaced_by_easier_pair=True);failures.append(fail);print('FAILED',json.dumps(fail),flush=True)
    jsonl(ROOT/'regions.jsonl',results);save(ROOT/'failed_candidates.json',failures)
    save(ROOT/'construction_summary.json',dict(regions=len(results),different=sum(r['look_pair_category']=='different_look_verified' for r in results),same=sum(r['look_pair_category']=='repeat_pass_same_look' for r in results),distinct_pairs=len({r['pair_id'] for r in results}),sites=sorted({r['site_name'] for r in results}),components=sorted({r['component'] for r in results}),acquisitions=len({(r['sensor'],v['acquisition_id']) for r in results for v in r['views']}),human_verified=0,qualified_GT=0,geography_only=len(results),failures=len(failures),no_training=True))
if __name__=='__main__':run()
