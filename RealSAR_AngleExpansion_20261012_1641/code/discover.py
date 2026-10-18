"""Bounded public metadata discovery; cache reuse, no image download or GT."""
import sys
sys.dont_write_bytecode = True
import hashlib, json, re, time, requests, xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
OLD = BASE / 'real_sar_pilot/multisource512_20261010_2352'
HOST = 'https://umbra-open-data-catalog.s3.us-west-2.amazonaws.com/'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
SITES = ['Port of Long Beach, California, United States', 'Port of Rotterdam, Netherlands',
         'Port of Antwerp, Belgium', 'Port of Singapore, Singapore',
         'Port of Busan, South Korea', 'Port of Jebel Ali, United Arab Emirates', 'Panama Canal, Panama']

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def discover():
    out = ROOT / 'discovery'; out.mkdir(exist_ok=False); raw = out / 'raw'; raw.mkdir()
    retained = {}
    for folder in ['umbra_entities', 'umbra_entities_v2']:
        for p in (OLD / folder / 'raw').glob('*'):
            retained.setdefault(p.name, p)
    def fetch(url, params=None):
        req = requests.Request('GET', url, params=params).prepare()
        suffix = '.xml' if params else '.json'
        key = hashlib.sha256(req.url.encode()).hexdigest() + suffix
        cached = retained.get(key)
        if cached:
            return cached.read_bytes(), str(cached), req.url, True
        p = raw / key
        if p.exists():
            return p.read_bytes(), str(p), req.url, True
        response = requests.get(req.url, timeout=(8, 25)); response.raise_for_status()
        with p.open('xb') as f: f.write(response.content)
        return response.content, str(p), req.url, False
    listings = []; tasks = []
    for site in SITES:
        prefix = 'sar-data/tasks/' + site + '/'
        body, p, url, reuse = fetch(HOST, {'list-type':'2','prefix':prefix,'max-keys':'1000','delimiter':'/'})
        d = ET.fromstring(body)
        if d.findtext('s:IsTruncated', 'false', NS) == 'true':
            raise ValueError('Site listing truncated: ' + site)
        folders = [x.text for x in d.findall('s:CommonPrefixes/s:Prefix', NS)]
        listings.append(dict(site=site, folders=len(folders), path=p, url=url, reused=reuse))
        # Complete each listed site up to 80 tasks; explicitly record any cap.
        for folder in folders[:80]: tasks.append((site, folder))
        print('SITE',site,'folders',len(folders),'selected',min(80,len(folders)),flush=True)
    def resolve(arg):
        site, prefix = arg
        try:
            body, listing_path, _, _ = fetch(HOST, {'list-type':'2','prefix':prefix,'max-keys':'1000'})
            d = ET.fromstring(body)
            if d.findtext('s:IsTruncated','false',NS)=='true': raise ValueError('Task listing truncated')
            keys = [x.text for x in d.findall('s:Contents/s:Key',NS)]
            products = [k for k in keys if k.endswith(('_GEC.tif','_GEC_MM.tif'))]
            metas = [k for k in keys if k.endswith(('_METADATA.json','_MM.json','.stac.v2.json'))]
            result = []
            for k in products:
                directory = k.rsplit('/',1)[0]
                mk = next((m for m in metas if m.rsplit('/',1)[0]==directory),None)
                if not mk: raise ValueError('Missing product metadata')
                content, path, url, reuse = fetch(HOST+quote(mk,safe='/'))
                doc = json.loads(content)
                if doc.get('type')=='Feature':
                    props=doc['properties']; ids=props.get('umbra:collect_ids') or [props.get('umbra:collect_id')]
                    if len(ids)!=1 or not ids[0]: raise ValueError('Non-single collect')
                    rec=dict(acquisition_id=ids[0],task_id=props.get('umbra:task_id'),platform=props.get('platform'),
                        datetime=props.get('start_datetime') or props.get('datetime'),end_datetime=props.get('end_datetime'),
                        orbit_direction=props.get('sat:orbit_state'),observation_direction=props.get('sar:observation_direction'),
                        azimuth_degrees=props.get('view:azimuth'),incidence_degrees=props.get('view:incidence_angle'),
                        polarizations=props.get('sar:polarizations'),geometry=doc['geometry'],
                        resolution=[{'groundResolution':{'rangeMeters':props.get('sar:resolution_range'),'azimuthMeters':props.get('sar:resolution_azimuth')}}],
                        azimuth_source_field='view:azimuth')
                else:
                    cs=doc.get('collects',[])
                    if len(cs)!=1: raise ValueError('Non-single collect')
                    c=cs[0]
                    rec=dict(acquisition_id=c['id'],task_id=c['taskId'],platform=doc.get('umbraSatelliteName'),
                        datetime=c['startAtUTC'],end_datetime=c.get('endAtUTC'),orbit_direction=c.get('satelliteTrack'),
                        observation_direction=c.get('observationDirection'),azimuth_degrees=c.get('angleAzimuthDegrees'),
                        incidence_degrees=c.get('angleIncidenceDegrees'),polarizations=c.get('polarizations'),geometry=c['footprintPolygonLla'],
                        resolution=doc.get('derivedProducts',{}).get('GEC',[]),azimuth_source_field='angleAzimuthDegrees')
                points=rec['geometry']['coordinates'][0]
                rec.update(provider='Umbra',site_name=site,asset_url=HOST+quote(k,safe='/'),metadata_url=url,metadata_path=path,
                    metadata_sha256=hashlib.sha256(content).hexdigest(),metadata_reused=reuse,listing_path=listing_path,
                    bbox=[min(x[0] for x in points),min(x[1] for x in points),max(x[0] for x in points),max(x[1] for x in points)],
                    input_units='linear_amplitude',license='CC-BY-4.0',license_url='https://registry.opendata.aws/umbra-open-data/',
                    raster_status='METADATA_VERIFIED_RASTER_NOT_READ',radiometry='Detected amplitude DN; no absolute calibration assumed')
                result.append(rec)
            return result, None
        except Exception as e:
            return [],dict(site=site,prefix=prefix,error=repr(e),at=now())
    inventory=[]; failures=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for n,(rs,error) in enumerate(pool.map(resolve,tasks),1):
            inventory.extend(rs)
            if error: failures.append(error)
            if n%10==0: print('TASKS',n,'of',len(tasks),'acquisitions',len(inventory),'failures',len(failures),flush=True)
    with (out/'inventory.jsonl').open('x') as f:
        for r in inventory: f.write(json.dumps(r,ensure_ascii=False)+'\n')
    with (out/'summary.json').open('x') as f:
        json.dump(dict(at=now(),listings=listings,tasks=len(tasks),acquisitions=len(inventory),failures=failures,
            full_rasters_downloaded=0,GT=0,site_cap=80,network_workers=4),f,ensure_ascii=False,indent=2)

if __name__=='__main__': discover()
