"""Verify original collection sidecars and build a locked real-pair manifest.

No image matcher, estimated transform, or synthetic ground truth is used here.
"""
import argparse,csv,hashlib,itertools,json,math,re
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import requests,rasterio
from rasterio.warp import transform,transform_bounds
from rasterio.windows import Window
from discover_stac import intersection_ratio

S1_SITES={'14TPN':('South_Dakota',-97.70,43.29,'farmland'),
 '10SEG':('San_Francisco',-122.40,37.78,'urban_coast'),
 '10TET':('Seattle',-122.33,47.60,'urban_port'),
 '11SLT':('Long_Beach',-118.25,33.73,'port_coast'),
 '13SDD':('Denver_Mountains',-105.45,39.65,'mountains'),
 '10SFH':('Sacramento',-121.60,38.60,'farmland')}

def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False))
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def parsed_utc(value):
    # Python 3.10 accepts microseconds, while Capella records nanosecond timestamps.
    return datetime.fromisoformat(re.sub(r'(\.\d{6})\d+',r'\1',value).replace('Z','+00:00'))
def raster_info(ds):
    return {'width':ds.width,'height':ds.height,'dtype':ds.dtypes[0],'count':ds.count,
       'crs':str(ds.crs),'transform':list(ds.transform)[:6],'nodata':ds.nodata,
       'bounds_wgs84':list(transform_bounds(ds.crs,'EPSG:4326',*ds.bounds)),
       'block_shapes':ds.block_shapes,'rpc':ds.tags(ns='RPC'),'tags':ds.tags()}

def s1(run):
    out=run/'s1_verified';out.mkdir(exist_ok=False);all_sources=[];failures=[]
    for tile,(site,lon,lat,scene) in S1_SITES.items():
        catalog=json.loads((run/'s1_discovery'/f'{tile}_scenes.json').read_text())
        candidates=[]
        for direction in ['ascending','descending']:
            group=[x for x in catalog if x['orbit_direction']==direction and '20200501'<=x['date']<='20200715']
            # Source/date choices precede image-only quality checks. Do not tune registration parameters.
            candidates.extend(group[::max(1,len(group)//5)][:5])
        def verify(rec):
            folder=out/rec['scene_name'];folder.mkdir();product_ids=[];datatakes=[];safes=[]
            try:
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='40',GDAL_HTTP_MAX_RETRY='2',AWS_NO_SIGN_REQUEST='YES'):
                    with rasterio.open(rec['url']) as ds:
                        info=raster_info(ds);tags=ds.tags()
                        if int(tags['NUMBER_SCENES'])!=1:raise ValueError('RTC mosaic: cannot represent one atomic acquisition; retained but excluded')
                        xx,yy=transform('EPSG:4326',ds.crs,[lon],[lat]);row,col=ds.index(xx[0],yy[0]);x=col-256;y=row-256
                        if x<0 or y<0 or x+512>ds.width or y+512>ds.height:raise ValueError('Target outside source support')
                        a=ds.read(1,window=Window(x,y,512,512));fraction=float((np.isfinite(a)&(a>0)).mean())
                        if fraction<.98:raise ValueError(f'Target 512 valid fraction {fraction:.4f}<.98')
                        roi=[max(0,col-1024),max(0,row-1024),min(ds.width,col+1024),min(ds.height,row+1024)]
                m=json.loads(tags['SCENE_1_METADATA']);pid=m['title'];product_ids=[pid];datatakes=[m['mission_id']+':'+m['mission_data_take_id']]
                base=rec['url'].rsplit('/',1)[0]+'/'+pid+'/'
                for name in ['manifest.safe','productInfo.json']:
                    rr=requests.get(base+name,timeout=40);rr.raise_for_status();(folder/name).write_bytes(rr.content)
                    if name=='manifest.safe':
                        root=ET.fromstring(rr.content)
                        get=lambda k:[{'value':n.text,'attributes':n.attrib} for n in root.iter() if n.tag.split('}')[-1]==k]
                        safe={k:get(k) for k in ['startTime','stopTime','relativeOrbitNumber','orbitNumber','pass']};safes.append(safe)
                start=safe['startTime'][0]['value'].rstrip('Z')+'Z';end=safe['stopTime'][0]['value'].rstrip('Z')+'Z'
                assert start[:10].replace('-','')==rec['date']
                relative=next(int(v['value']) for v in safe['relativeOrbitNumber'] if v['attributes'].get('type')=='start')
                direction=safe['pass'][0]['value'].lower();assert direction==rec['orbit_direction']
                r={'provider':'Sentinel-1','site_name':site,'scene_class':scene,'tile':tile,
                   'acquisition_id':pid,'original_products':product_ids,'datatake_ids':datatakes,
                   'datetime':start,'end_datetime':end,'platform':rec['satellite'],'orbit_direction':direction,
                   'relative_orbit':relative,'absolute_orbit':int(m['absolute_orbit_number']),
                   'observation_direction':'right','azimuth_degrees':None,'incidence_degrees':None,
                   'polarizations':['VV'],'asset_url':rec['url'],'input_units':'linear_power',
                   'radiometry':'RTC Gamma0 linear power; terrain corrected; 20m grid spacing',
                   'license':'Indigo Atlas Sentinel License','license_url':'https://www.indigoag.com/forms/atlas-sentinel-license',
                   'geometry':json.loads(m['footprint']),'bbox':info['bounds_wgs84'],
                   'roi_xyxy':roi,'metadata_path':str((folder/'source.json').resolve()),'raster':info,
                   'resolution':{'grid_spacing_m':20,'physical_resolution_note':'RTC processing and Lee Sigma filtering; grid spacing is not measured resolving power'},'safe':safes}
                write(folder/'source.json',r);return r,None
            except Exception as e:
                error={'scene_name':rec['scene_name'],'url':rec['url'],'failure':repr(e)};write(folder/'failure.json',error);return None,error
        with ThreadPoolExecutor(max_workers=4) as pool:
            checked=list(pool.map(verify,candidates))
        accepted=[r for r,e in checked if r];failures.extend(e for r,e in checked if e)
        ascending=[r for r in accepted if r['orbit_direction']=='ascending'];descending=[r for r in accepted if r['orbit_direction']=='descending']
        chosen=(ascending[:2]+descending[:1]) if ascending and descending else accepted[:3]
        if len(chosen)<3:chosen=accepted[:3]
        all_sources.extend(chosen);print('S1',tile,'checked',len(checked),'good',len(accepted),'selected',len(chosen),flush=True)
    (out/'inventory.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in all_sources));write(out/'summary.json',{'sources':len(all_sources),'failures':failures})

def capella(run):
    out=run/'capella_verified';out.mkdir(exist_ok=False);items=lines(run/'capella_discovery/inventory.jsonl');selected=[];failures=[]
    for label,site,scene in [('mount-etna','Etna','mountains_volcano'),('sunny-isles','Sunny_Isles','urban_coast'),('kingston,-australia','Kingston_Australia','urban')]:
        group=sorted([r for r in items if label in r['collection']],key=lambda x:x['datetime'])
        for ix in np.linspace(0,len(group)-1,min(3,len(group))).astype(int):
            rec=group[int(ix)];folder=out/rec['id'];folder.mkdir()
            try:
                u=rec['assets']['metadata']['href'];rr=requests.get(u,timeout=40);rr.raise_for_status();(folder/'extended.json').write_bytes(rr.content)
                d=rr.json();c=d['collect'];props=rec['properties'];assert c['collect_id']==props['capella:collect_id'];assert d['product_type']=='GEC'
                with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_TIMEOUT='40',GDAL_HTTP_MAX_RETRY='2'):
                    with rasterio.open(rec['asset_url']) as ds:info=raster_info(ds)
                scale=c['image']['scale_factor'];assert scale>0
                r={'provider':'Capella','site_name':site,'scene_class':scene,'acquisition_id':c['collect_id'],
                   'original_products':[rec['id']],'datetime':c['start_timestamp'],'end_datetime':c['stop_timestamp'],
                   'platform':props['platform'],'orbit_direction':props['sat:orbit_state'],
                   'relative_orbit':props.get('sat:relative_orbit'),'absolute_orbit':props.get('sat:absolute_orbit'),
                   'observation_direction':props.get('sar:observation_direction'),'azimuth_degrees':props.get('view:azimuth'),
                   'incidence_degrees':props.get('view:incidence_angle'),'polarizations':props['sar:polarizations'],
                   'asset_url':rec['asset_url'],'metadata_url':u,'metadata_path':str((folder/'source.json').resolve()),
                   'input_units':'linear_amplitude','radiometry':'Raw DN amplitude preserved; sigma0_dB=20log10(scale_factor*DN)',
                   'scale_factor':scale,'geometry':rec['geometry'],'bbox':rec['bbox'],'raster':info,
                   'resolution':{'range_m':c['image'].get('range_resolution'),'ground_range_m':c['image'].get('ground_range_resolution'),'azimuth_m':c['image'].get('azimuth_resolution')},
                   'license':'CC-BY-4.0','license_url':'https://www.capellaspace.com/legal/data-licensing','original_stac_url':rec['source_stac_url']}
                write(folder/'source.json',r);selected.append(r);print('Capella',site,rec['id'],flush=True)
            except Exception as e:failures.append({'id':rec['id'],'error':repr(e)})
    (out/'inventory.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in selected));write(out/'summary.json',{'sources':len(selected),'failures':failures})

def manifest(run):
    sources=lines(run/'umbra_entities_v2/inventory.jsonl')+lines(run/'s1_verified/inventory.jsonl')+lines(run/'capella_verified/inventory.jsonl')
    sites=defaultdict(list)
    for r in sources:sites[(r['provider'],r['site_name'])].append(r)
    parents={k:k for k in sites}
    def find(k):
        while parents[k]!=k:k=parents[k]
        return k
    def join(a,b):parents[find(b)]=find(a)
    edges=[]
    for a,b in itertools.combinations(sites,2):
        aa=sites[a];bb=sites[b];idsa=set(x for r in aa for x in [r['acquisition_id']]+r.get('original_products',[])+r.get('datatake_ids',[]));idsb=set(x for r in bb for x in [r['acquisition_id']]+r.get('original_products',[])+r.get('datatake_ids',[]))
        spatial=any(intersection_ratio(x['bbox'],y['bbox'])>0 for x in aa for y in bb)
        nearby=any(abs((x['bbox'][0]+x['bbox'][2]-y['bbox'][0]-y['bbox'][2])/2)<.25 and abs((x['bbox'][1]+x['bbox'][3]-y['bbox'][1]-y['bbox'][3])/2)<.25 for x in aa for y in bb)
        if idsa&idsb or spatial or nearby:join(a,b);edges.append({'a':a,'b':b,'identity_intersection':sorted(idsa&idsb),'spatial_bbox_overlap':spatial,'nearby':nearby})
    groups=defaultdict(list)
    for k in sites:groups[find(k)].append(k)
    assignment={};components=[]
    for n,keys in enumerate(sorted(groups.values())):
        names=' '.join(k[1] for k in keys)
        split='train' if any(k[0]=='Sentinel-1' for k in keys) else ('dev' if 'Bingham' in names or 'Melbourne' in names else 'val' if 'Johnston' in names or 'Sunny' in names else 'confirmation')
        component=f'real_component_{n:03d}';components.append({'component':component,'sites':keys,'split':split})
        for k in keys:assignment[k]=(component,split)
    pairs=[]
    for (provider,site),group in sorted(sites.items()):
        group=sorted(group,key=lambda x:x['datetime']);component,split=assignment[(provider,site)]
        proposals=[]
        for a,b in itertools.combinations(group,2):
            if a['polarizations']!=b['polarizations'] or a['acquisition_id']==b['acquisition_id']:continue
            delta=(parsed_utc(b['datetime'])-parsed_utc(a['datetime'])).total_seconds()/86400
            if delta<1 or intersection_ratio(a['bbox'],b['bbox'])<.55:continue
            da,db=a.get('azimuth_degrees'),b.get('azimuth_degrees');ad=abs((float(da)-float(db)+180)%360-180) if da is not None and db is not None else None
            opposite=any(a.get(k) and b.get(k) and a[k].lower()!=b[k].lower() for k in ['orbit_direction','observation_direction'])
            different=opposite or ad is not None and ad>=15
            if different:look='different_look_verified'
            elif provider=='Sentinel-1':look='repeat_pass_same_look' if a.get('relative_orbit')==b.get('relative_orbit') else 'orientation_unknown'
            elif ad is not None and ad<=2 and abs(float(a.get('incidence_degrees') or 0)-float(b.get('incidence_degrees') or 0))<=2:look='repeat_pass_same_look'
            else:look='orientation_unknown'
            proposals.append((look,delta,a,b,ad))
        # Include same-look and different-look subsets with deterministic temporal coverage.
        proposals.sort(key=lambda p:(p[0],-p[1],p[2]['acquisition_id']))
        chosen=proposals if provider!='Umbra' else proposals[:10]
        same=[p for p in proposals if p[0]=='repeat_pass_same_look']
        if provider=='Umbra' and same and not any(p[0]=='repeat_pass_same_look' for p in chosen):chosen[-1]=same[0]
        for look,days,a,b,ad in chosen:
            pid=f'{provider.replace("-","")}_{len(pairs):04d}';scene=a.get('scene_class') or {'Bingham Copper Mine':'mountain_mine','Johnston, Iowa Farm':'farmland','Melbourne, Australia':'urban','Hetch Hetchy Reservoir':'mountain_water'}.get(site,'port_coast')
            r={'pair_id':pid,'site_id':component,'split':split,'sensor':provider,
               'reference_acquisition_id':a['acquisition_id'],'moving_acquisition_id':b['acquisition_id'],
               'reference_path':a['asset_url'],'moving_path':b['asset_url'],'reference_utc':a['datetime'],'moving_utc':b['datetime'],
               'input_units':a['input_units'],'product_level':'RTC_Gamma0_VV' if provider=='Sentinel-1' else 'GEC',
               'site_name':site,'scene_class':scene,'look_pair_category':look,'azimuth_difference_degrees':ad,
               'time_separation_days':days,'source_metadata_status':'ORIGINAL_SIDECAR_VERIFIED__RASTER_CROP_QA_REQUIRED',
               'license':a['license'],'license_url':a['license_url'],
               'reference_metadata':a['metadata_path'],'moving_metadata':b['metadata_path'],
               'reference_platform':a['platform'],'moving_platform':b['platform'],
               'reference_orbit_direction':a['orbit_direction'],'moving_orbit_direction':b['orbit_direction'],
               'reference_observation_direction':a.get('observation_direction'),'moving_observation_direction':b.get('observation_direction'),
               'reference_incidence_degrees':a.get('incidence_degrees'),'moving_incidence_degrees':b.get('incidence_degrees'),
               'reference_resolution_json':json.dumps(a.get('resolution')),'moving_resolution_json':json.dumps(b.get('resolution')),
               'polarization':','.join(a['polarizations']),'display_mode':'fixed_db' if provider=='Sentinel-1' else 'joint_native_percentiles',
               'reference_roi_xyxy':json.dumps(a.get('roi_xyxy')),'scene_label_status':'SITE_CONTEXT__CHIP_VISUAL_REVIEW_REQUIRED',
               'angular_class_rule':'pass/left-right change OR circular azimuth difference>=15deg; same look only same track or <=2deg azimuth AND incidence; others unknown'}
            pairs.append(r)
    fields=list(dict.fromkeys(k for r in pairs for k in r))
    with (run/'verified_sources.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(pairs)
    write(run/'source_components.json',{'components':components,'edges':edges,'source_count':len(sources),'pairs':len(pairs),'known_prior_14TPN_component':'train; linked to earlier real pilot by same MGRS site','limits':'BBox neighborhoods and published IDs; unknown acquisition associations remain possible; all confirmation candidates are inspected during engineering and are not a sealed benchmark'})
    (run/'sources.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in sources));print('Locked pairs',len(pairs),'sources',len(sources),'components',len(components),'by_provider',dict(Counter(x['sensor'] for x in pairs)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['s1','capella','manifest']);p.add_argument('--run',required=True);a=p.parse_args();globals()[a.mode](Path(a.run).resolve())
