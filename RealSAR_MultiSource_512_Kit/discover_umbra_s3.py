"""Resolve real public Umbra task folders when the static STAC assets have empty hrefs.

S3 listing and vendor sidecars are retained verbatim. This does not make GT or splits.
"""
import argparse,hashlib,json,re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
import xml.etree.ElementTree as ET
import requests
import numpy as np

ROOT='https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/'
NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}

def discover(out,sites,per_site=8,seed=260512):
    out=Path(out);out.mkdir(exist_ok=False,parents=True);raw=out/'raw';raw.mkdir()
    errors=[]; inventory=[];rng=np.random.default_rng(seed)
    def listing(prefix,delimiter=None):
        keys=[];prefixes=[];token=None
        for page in range(100):
            params={'list-type':'2','prefix':prefix,'max-keys':'1000'}
            if delimiter:params['delimiter']=delimiter
            if token:params['continuation-token']=token
            r=requests.get(ROOT,params=params,timeout=40);r.raise_for_status()
            (raw/(hashlib.sha256(r.url.encode()).hexdigest()+'.xml')).write_bytes(r.content)
            d=ET.fromstring(r.content);keys.extend([x.text for x in d.findall('s:Contents/s:Key',NS)])
            prefixes.extend([x.text for x in d.findall('s:CommonPrefixes/s:Prefix',NS)])
            if d.findtext('s:IsTruncated','false',NS)!='true':return keys,prefixes
            token=d.findtext('s:NextContinuationToken',None,NS)
            if not token:raise ValueError('Truncated listing without token')
        raise RuntimeError('Listing page limit')
    _,available=listing('sar-data/tasks/','/')
    chosen=[];counts={}
    for site in sites:
        prefix='sar-data/tasks/'+site+'/'
        if prefix not in available:raise ValueError('Site absent from real listing: '+site)
        _,folders=listing(prefix,'/');counts[site]=len(folders)
        for ix in sorted(rng.choice(len(folders),size=min(per_site,len(folders)),replace=False)):
            chosen.append((site,folders[int(ix)]))
    def task(arg):
        site,prefix=arg
        try:
            keys,_=listing(prefix)
            gec=[k for k in keys if k.endswith('_GEC.tif') or k.endswith('_GEC_MM.tif')]
            metas=[k for k in keys if k.endswith('_METADATA.json') or k.endswith('_MM.json') or k.endswith('.stac.v2.json')]
            result=[]
            for key in gec:
                directory=key.rsplit('/',1)[0]
                metadata=next((k for k in metas if k.rsplit('/',1)[0]==directory),None)
                if metadata is None:raise ValueError('No original metadata sidecar for '+key)
                u=ROOT+quote(metadata,safe='/');r=requests.get(u,timeout=40);r.raise_for_status()
                mf=raw/(hashlib.sha256(u.encode()).hexdigest()+'.json');mf.write_bytes(r.content)
                d=r.json()
                if d.get('type')=='Feature':
                    props=d['properties'];ids=props.get('umbra:collect_ids') or [props.get('umbra:collect_id')]
                    if len(ids)!=1 or not ids[0]:raise ValueError('Unknown/multiple original collect IDs')
                    c={'id':ids[0],'taskId':props['umbra:task_id'],
                       'startAtUTC':props.get('start_datetime') or props['datetime'],
                       'endAtUTC':props.get('end_datetime'),'satelliteTrack':props.get('sat:orbit_state'),
                       'observationDirection':props.get('sar:observation_direction'),
                       'angleAzimuthDegrees':props.get('view:azimuth'),'angleIncidenceDegrees':props.get('view:incidence_angle'),
                       'polarizations':props.get('sar:polarizations'),'footprintPolygonLla':d['geometry']}
                    d={'collects':[c],'umbraSatelliteName':props['platform'],
                       'derivedProducts':{'GEC':[{'groundResolution':{'rangeMeters':props.get('sar:resolution_range'),'azimuthMeters':props.get('sar:resolution_azimuth')}}]}}
                collects=d.get('collects',[])
                if len(collects)!=1:raise ValueError('Multiple/unknown collects require separate review')
                c=collects[0];polygon=c['footprintPolygonLla'];pts=polygon['coordinates'][0]
                result.append({'provider':'Umbra','site_name':site,'acquisition_id':c['id'],
                    'task_id':c['taskId'],'platform':d['umbraSatelliteName'],
                    'datetime':c['startAtUTC'],'end_datetime':c['endAtUTC'],
                    'orbit_direction':c.get('satelliteTrack'),'observation_direction':c.get('observationDirection'),
                    'azimuth_degrees':c.get('angleAzimuthDegrees'),'incidence_degrees':c.get('angleIncidenceDegrees'),
                    'polarizations':c.get('polarizations'),'geometry':polygon,
                    'bbox':[min(x[0] for x in pts),min(x[1] for x in pts),max(x[0] for x in pts),max(x[1] for x in pts)],
                    'resolution':d.get('derivedProducts',{}).get('GEC',[]),'base_ipr_m':d.get('baseIpr'),
                    'asset_url':ROOT+quote(key,safe='/'),'metadata_url':u,'metadata_path':str(mf.resolve()),
                    'metadata_sha256':hashlib.sha256(r.content).hexdigest(),'input_units':'linear_amplitude',
                    'radiometry':'Detected DN amplitude; no absolute sigma0 calibration assumed',
                    'license':'CC-BY-4.0','license_url':'https://registry.opendata.aws/umbra-open-data/'})
            if not result:raise ValueError('No GEC GeoTIFF in sampled actual collect folder')
            return result,None
        except Exception as e:return [],{'site':site,'prefix':prefix,'error':repr(e)}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for n,(results,error) in enumerate(pool.map(task,chosen)):
            inventory.extend(results)
            if error:errors.append(error)
            if (n+1)%8==0:print('Umbra tasks',n+1,'verified sidecars',len(inventory),'failures',len(errors),flush=True)
    (out/'inventory.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in inventory))
    (out/'discovery.json').write_text(json.dumps({'actual_site_folders':counts,'sampled_task_folders':len(chosen),
        'seed':seed,'inventory_acquisitions':len(inventory),'failures':errors,'status':'SIDECAR_VERIFIED__RASTER_QA_PENDING'},indent=2))
    print('Inventory',len(inventory),'errors',len(errors),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--sites',nargs='+',required=True);p.add_argument('--per-site',type=int,default=8);p.add_argument('--seed',type=int,default=260512)
    a=p.parse_args();discover(a.out,a.sites,a.per_site,a.seed)
