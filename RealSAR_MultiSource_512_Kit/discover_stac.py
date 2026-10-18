#!/usr/bin/env python3
"""Read PUBLIC static STAC catalogs, find real SAR GeoTIFF assets and propose temporal image pairs.

Outputs inventory and REVIEW-ONLY manifest. Does not download full imagery, guess pixel units,
assume angular diversity, or create pixel-level GT. Supports Umbra and similar public STAC roots.
"""
from __future__ import annotations
import argparse,csv,json,re,hashlib
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin,urlparse
import requests
from concurrent.futures import ThreadPoolExecutor

DEFAULT_UMBRA='https://s3.us-west-2.amazonaws.com/umbra-open-data-catalog/stac/catalog.json'
CATALOGS={'umbra':DEFAULT_UMBRA,
          'capella':'https://capella-open-data.s3.us-west-2.amazonaws.com/stac/catalog.json'}
CSV_FIELDS=['pair_id','site_id','split','sensor','reference_acquisition_id','moving_acquisition_id',
 'reference_path','moving_path','reference_utc','moving_utc','input_units','product_level',
 'look_pair_category','source_metadata_status','license']


def http_s3(s):
    """Convert anonymous s3:// URLs only; provider may still require account for assets."""
    if s.startswith('s3://'):
        u=s[5:].split('/',1)
        if len(u)==2:return 'https://'+u[0]+'.s3.us-west-2.amazonaws.com/'+u[1]
    return s


def node_json(session,url,timeout):
    r=session.get(url,timeout=timeout)
    r.raise_for_status()
    return r.json()


def asset_best(item):
    assets=item.get('assets',{})
    opts=[]
    for k,v in assets.items():
        uri=str(v.get('href',''))
        if not ('.tif' in uri.lower() or '.tiff' in uri.lower()):continue
        if any(x in (k+' '+uri).lower() for x in ('thumbnail','browse','quicklook','mask','incidence','angle','alpha')):
            continue
        rank=0
        for word,sc in [('geocoded',30),('gec',30),('rtc',20),('intensity',12),('amplitude',10),('grd',8)]:
            if word in (k+' '+uri+' '+str(v.get('title',''))).lower():rank+=sc
        if '.tif' in uri.lower():rank+=1
        opts.append((rank,k,uri,v))
    return max(opts,default=None,key=lambda x:x[0])


def traverse(root,max_documents=800,timeout=35,cache_dir=None):
    session=requests.Session()
    session.headers.update({'User-Agent':'RealSAR-multisource-512/0.2 STAC-discovery'})
    queue=[root];visited=set();entries=[];errors=[]
    buffer={}
    def fetch(url):
        cached=Path(cache_dir)/(hashlib.sha256(url.encode()).hexdigest()+'.json') if cache_dir else None
        try:
            if cached and cached.exists():return json.loads(cached.read_text())['document']
            return node_json(session,url,timeout)
        except Exception as exc:return exc
    while queue and len(visited)<max_documents:
        link=queue.pop(0)
        url=http_s3(link)
        if url in visited:continue
        visited.add(url)
        try:
            if url not in buffer:
                batch=list(dict.fromkeys([url]+[http_s3(x) for x in queue[:7] if http_s3(x) not in visited]))
                with ThreadPoolExecutor(max_workers=8) as pool:
                    buffer.update(zip(batch,pool.map(fetch,batch)))
            data=buffer.pop(url)
            if isinstance(data,Exception):raise data
            if cache_dir:
                Path(cache_dir).mkdir(parents=True,exist_ok=True)
                (Path(cache_dir)/(hashlib.sha256(url.encode()).hexdigest()+'.json')).write_text(json.dumps({'url':url,'document':data},indent=2))
            typ=data.get('type','')
            if typ=='Feature':
                choice=asset_best(data)
                if choice:
                    rank,key,path,v=choice
                    if not path.startswith(('http://','https://','s3://')):path=urljoin(url,path)
                    props=data.get('properties',{})
                    bbox=data.get('bbox',None)
                    dt=props.get('datetime') or props.get('start_datetime')
                    entries.append({'id':str(data.get('id','')), 'source_stac_url':url,
                        'sensor':props.get('platform') or props.get('constellation') or 'UNKNOWN',
                        'datetime':dt,'collection':data.get('collection',''),
                        'orbit_direction':props.get('sat:orbit_state') or props.get('orbit_state','unknown'),
                        'incidence_angle':props.get('view:incidence_angle',props.get('sar:incidence_angle')),
                        'properties':props,'geometry':data.get('geometry'),'assets':assets if (assets:=data.get('assets')) else {},
                        'bbox':bbox,'asset_key':key,'asset_url':http_s3(path),
                        'asset_rank':rank,'georeferencing_status':'VERIFY_OPEN_AND_CRS',
                        'data_units':'UNKNOWN__VERIFY_FROM_PROVIDER'})
            for child in data.get('features',[]):
                if isinstance(child,dict) and child.get('type')=='Feature':
                    # Embedded STAC Features may need special treatment, but do not silently omit.
                    errors.append({'url':url,'note':'embedded FeatureCollection item; use item link or extend parser'})
            for rel in data.get('links',[]):
                if rel.get('rel') not in ('child','item','items','next'):continue
                dest=rel.get('href','')
                if dest:
                    if dest.startswith('s3://'):next_url=http_s3(dest)
                    else:next_url=urljoin(url,dest)
                    if next_url.startswith('https://') and next_url not in visited and next_url not in queue:
                        queue.append(next_url)
        except Exception as e:
            errors.append({'url':url,'error':type(e).__name__+': '+str(e)[:250]})
        if len(visited)%25==0:print('STAC documents',len(visited),'items',len(entries),'pending',len(queue),flush=True)
    return {'items':entries,'errors':errors,'documents_visited':len(visited),
            'queue_remaining':len(queue),'truncated':bool(queue)}


def intersection_ratio(b1,b2):
    if not b1 or not b2 or len(b1)!=4 or len(b2)!=4:return 0.
    dx=max(0.,min(b1[2],b2[2])-max(b1[0],b2[0]))
    dy=max(0.,min(b1[3],b2[3])-max(b1[1],b2[1]))
    a1=max(0.,b1[2]-b1[0])*max(0.,b1[3]-b1[1]);a2=max(0.,b2[2]-b2[0])*max(0.,b2[3]-b2[1])
    return dx*dy/max(1e-12,min(a1,a2))


def proposed_pairs(items,provider,max_pairs=60,min_overlap=.55,min_days=1):
    pairs=[]; use={}
    parsed=[]
    for i,rec in enumerate(items):
        if not rec.get('datetime') or not rec.get('bbox'):continue
        try:dt=datetime.fromisoformat(rec['datetime'].replace('Z','+00:00'))
        except (ValueError,TypeError):continue
        parsed.append((i,rec,dt))
    for a in range(len(parsed)):
        i,ref,da=parsed[a]
        for b in range(a+1,len(parsed)):
            j,mov,db=parsed[b]
            if ref['sensor']!=mov['sensor'] or ref['id']==mov['id']:continue
            days=abs((da-db).total_seconds())/86400
            if days<min_days:continue
            overlap=intersection_ratio(ref['bbox'],mov['bbox'])
            if overlap<min_overlap:continue
            pairs.append((overlap,days,ref,mov))
    pairs.sort(key=lambda x:(-x[0],-x[1]))
    result=[]
    for overlap,days,ref,mov in pairs:
        if len(result)>=max_pairs:break
        if use.get(ref['id'],0)>=2 or use.get(mov['id'],0)>=2:continue
        use[ref['id']]=use.get(ref['id'],0)+1;use[mov['id']]=use.get(mov['id'],0)+1
        # Coordinate grids are only a candidate grouping. Do not assign a train/val split automatically.
        center=((ref['bbox'][0]+ref['bbox'][2])/2,(ref['bbox'][1]+ref['bbox'][3])/2)
        grid=(int(center[0]*5),int(center[1]*5))
        site=f'{provider}_cell_{grid[0]}_{grid[1]}'.replace('-','n')
        site=re.sub('[^A-Za-z0-9_.-]','_',site)
        result.append(dict(pair_id=f'{provider}_candidate_{len(result):04d}',
          site_id=site,split='REVIEW_REQUIRED',sensor=ref['sensor'],
          reference_acquisition_id=ref['id'],moving_acquisition_id=mov['id'],
          reference_path=ref['asset_url'],moving_path=mov['asset_url'],
          reference_utc=ref['datetime'],moving_utc=mov['datetime'],
          input_units='',product_level='GEO_PRODUCT_VERIFY',
          look_pair_category='UNKNOWN__VERIFY_ORBIT_LOOK_ANGLES',
          source_metadata_status=f'UNVERIFIED_STAC_SPATIAL_OVERLAP_{overlap:.2f}',
          license='CHECK_PROVIDER_LICENSE_AND_ATTRIBUTION'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--provider',choices=['umbra','capella','custom'],default='umbra')
    p.add_argument('--root',help='Public STAC root, mandatory if provider=custom')
    p.add_argument('--max-documents',type=int,default=800)
    p.add_argument('--max-pairs',type=int,default=60)
    p.add_argument('--min-overlap',type=float,default=.55)
    p.add_argument('--min-days',type=int,default=1)
    p.add_argument('--out',required=True)
    p.add_argument('--cache-dir',help='Reuse raw STAC cache from an interrupted discovery')
    x=p.parse_args()
    root=x.root or CATALOGS.get(x.provider)
    if not root:p.error('Need --root for custom provider')
    base=Path(x.out).resolve();base.mkdir(parents=True,exist_ok=False)
    obj=traverse(root,max_documents=x.max_documents,cache_dir=x.cache_dir or base/'raw_stac')
    with (base/'inventory.jsonl').open('w') as f:
        for item in obj['items']:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    proposed=proposed_pairs(obj['items'],x.provider,x.max_pairs,x.min_overlap,x.min_days)
    with (base/'pairs_PROPOSALS_REVIEW_REQUIRED.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=CSV_FIELDS);w.writeheader();w.writerows(proposed)
    (base/'discovery.json').write_text(json.dumps({'root':root,'documents_visited':obj['documents_visited'],
        'item_count':len(obj['items']),'pair_proposals':len(proposed),'truncated':obj['truncated'],
        'errors':obj['errors'],'status':'UNVERIFIED_SOURCE_CANDIDATES__DO_NOT_TRAIN_OR_REPORT_GT'},
        ensure_ascii=False,indent=2))
    print('Items',len(obj['items']),'proposals',len(proposed),'errors',len(obj['errors']))

if __name__=='__main__':main()
