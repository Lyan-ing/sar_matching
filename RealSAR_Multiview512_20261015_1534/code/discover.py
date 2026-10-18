"""Targeted new maritime sites, bounded metadata only; anonymous official S3."""
from common import *
import requests,xml.etree.ElementTree as ET
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
HOST='https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/'
NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
SITES=['Port of Hamburg, Germany','Port of Hong Kong, Hong Kong','Port of Savannah, Georgia, United States']

def main():
 t=stamp();folder=ROOT/'discovery'/t;folder.mkdir(parents=True);raw=folder/'raw';raw.mkdir();errors=[];records=[];requests_log=[]
 def get(url,params=None):
  req=requests.Request('GET',url,params=params).prepare();p=raw/(hashlib.sha256(req.url.encode()).hexdigest()+('.xml' if params else '.json'))
  r=requests.get(req.url,timeout=(10,25));r.raise_for_status();p.write_bytes(r.content);requests_log.append(dict(url=req.url,path=str(p),bytes=len(r.content)));return r.content,p,req.url
 def listing(prefix,delimiter=None):
  params={'list-type':'2','prefix':prefix,'max-keys':'1000'}
  if delimiter:params['delimiter']=delimiter
  content,_,_=get(HOST,params);d=ET.fromstring(content)
  if d.findtext('s:IsTruncated','false',NS)=='true':raise ValueError('Truncated listing, not silently complete')
  return [n.text for n in d.findall('s:Contents/s:Key',NS)],[n.text for n in d.findall('s:CommonPrefixes/s:Prefix',NS)]
 tasks=[];site_counts={}
 for site in SITES:
  _,pp=listing('sar-data/tasks/'+site+'/','/');site_counts[site]=len(pp)
  # Fixed deterministic bounded metadata sample, no pixels/model scores.
  ids=np.random.default_rng(20261015).choice(len(pp),min(18,len(pp)),replace=False) if pp else []
  tasks.extend((site,pp[int(i)]) for i in sorted(ids));print('DISCOVER_SITE',site,len(pp),len(ids),flush=True)
 def task(item):
  site,prefix=item
  try:
   kk,_=listing(prefix);products=sorted(k for k in kk if k.endswith('_GEC.tif'))
   out=[]
   for k in products:
    directory=k.rsplit('/',1)[0];mm=[m for m in kk if m.rsplit('/',1)[0]==directory and (m.endswith('.stac.v2.json') or m.endswith('_METADATA.json'))]
    if not mm:continue
    mk=sorted(mm,key=lambda m:0 if m.endswith('.stac.v2.json') else 1)[0]
    content,path,url=get(HOST+quote(mk,safe='/'));d=json.loads(content)
    if d.get('type')=='Feature':
     p=d['properties'];ids=p.get('umbra:collect_ids') or [p.get('umbra:collect_id')]
     if len(ids)!=1 or not ids[0]:raise ValueError('Not a single atomic collect')
     a=dict(acquisition_id=ids[0],task_id=p.get('umbra:task_id'),platform=p.get('platform'),datetime=p.get('start_datetime') or p.get('datetime'),end_datetime=p.get('end_datetime'),orbit_direction=p.get('sat:orbit_state'),observation_direction=p.get('sar:observation_direction'),azimuth_degrees=p.get('view:azimuth'),incidence_degrees=p.get('view:incidence_angle'),polarizations=p.get('sar:polarizations'),geometry=d['geometry'],resolution=[{'groundResolution':{'rangeMeters':p.get('sar:resolution_range'),'azimuthMeters':p.get('sar:resolution_azimuth')}}],azimuth_source_field='view:azimuth')
    else:
     cc=d['collects']
     if len(cc)!=1:raise ValueError('Not a single atomic collect')
     c=cc[0];a=dict(acquisition_id=c['id'],task_id=c['taskId'],platform=d.get('umbraSatelliteName'),datetime=c['startAtUTC'],end_datetime=c.get('endAtUTC'),orbit_direction=c.get('satelliteTrack'),observation_direction=c.get('observationDirection'),azimuth_degrees=c.get('angleAzimuthDegrees'),incidence_degrees=c.get('angleIncidenceDegrees'),polarizations=c.get('polarizations'),geometry=c['footprintPolygonLla'],resolution=d.get('derivedProducts',{}).get('GEC',[]),azimuth_source_field='angleAzimuthDegrees')
    if utc(a['datetime'])>datetime.now(ZoneInfo('Asia/Shanghai')):raise ValueError('Future timestamp')
    a.update(provider='Umbra',site_name=site,asset_url=HOST+quote(k,safe='/'),metadata_url=url,metadata_path=str(path),metadata_sha256=sha(path),input_units='linear_amplitude',license='CC-BY-4.0',license_url='https://registry.opendata.aws/umbra-open-data/',split='Val',component=component(site),platform_heading_degrees=None,radar_los_azimuth_degrees=a.get('azimuth_degrees'),heading_not_inferred_from_look_or_orbit=True,split_basis='new station actual images; no previous algorithm diagnosis')
    out.append(a)
   return out
  except Exception as e:errors.append(dict(site=site,prefix=prefix,error=repr(e)));return []
 with ThreadPoolExecutor(max_workers=4) as pool:
  for rs in pool.map(task,tasks):records.extend(rs)
 records=list({a['acquisition_id']:a for a in records}.values());jsonl(ROOT/f'NEW_METADATA_INVENTORY_{t}.jsonl',records)
 jsonl(folder/f'REQUESTS_{t}.jsonl',requests_log);save(folder/f'SUMMARY_{t}.json',dict(at=now(),site_task_counts=site_counts,tasks_sampled=len(tasks),acquisitions=len(records),failures=errors,pixels_read=False,GT=0))
 print('METADATA_ACQUISITIONS',len(records),'FAILURES',len(errors),flush=True)

if __name__=='__main__':main()
