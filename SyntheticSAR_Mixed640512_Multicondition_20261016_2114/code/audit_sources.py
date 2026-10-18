from common import *
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import argparse

def inspect(r):
    im=read_gray(r['image_path']);valid=source_valid(im);c=coords(r)
    return dict(source_path=r['image_path'],source_id=r['source_id'],dataset=r['dataset'],split=r['split'],
        source_component_id=r['known_component'],source_scene=r['scene_category'],scene_status=r['scene_category_status'],
        width=im.shape[1],height=im.shape[0],source_hash=sha(r['image_path']),pixel_hash=ahash(im),
        valid_fraction=float(valid.mean()),valid_hash=ahash(valid),std=float(im[valid].std()),
        coordinate_bounds_xyxy=c,annotation_count=len(r['annotations']),v22_feature_cache=str(cachepath(r)),
        independent_raw_acquisition_identity_known=r.get('independent_raw_acquisition_identity_known',False),
        single_source_640_dimension_support=min(im.shape)>=640,single_source_512_dimension_support=min(im.shape)>=512)

@lru_cache(maxsize=64)
def image(p):return read_gray(p)
def edge(job):
    a,b=job;xa,ya,ra,ba=a['coordinate_bounds_xyxy'];xb,yb,rb,bb=b['coordinate_bounds_xyxy']
    x,y,r,bottom=max(xa,xb),max(ya,yb),min(ra,rb),min(ba,bb)
    aa=image(a['source_path'])[y-ya:bottom-ya,x-xa:r-xa];bb=image(b['source_path'])[y-yb:bottom-yb,x-xb:r-xb]
    delta=np.abs(aa.astype('int16')-bb.astype('int16'));exact=float((delta==0).mean());mae=float(delta.mean())
    return dict(a=a['source_path'],b=b['source_path'],bounds_xyxy=[x,y,r,bottom],overlap_pixels=int(aa.size),
        exact_fraction=exact,mae_DN=mae,accepted=exact>=.995 and mae<=.05,
        method='published row/column offsets + direct pixel equality; no learned matcher or geographic projection')

def main():
    t=stamp();ss=raw_sources();start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as ex:aa=list(ex.map(inspect,ss,chunksize=8))
    jsonl(ROOT/f'meta/SOURCE_AUDIT_{t}.jsonl',aa)
    groups=defaultdict(list)
    for a in aa:
        if a['coordinate_bounds_xyxy']:groups[(a['dataset'],a['source_id'].split('_')[0])].append(a)
    jobs=[]
    for gg in groups.values():
        for i,a in enumerate(gg):
            x,y,r,b=a['coordinate_bounds_xyxy']
            for q in gg[i+1:]:
                X,Y,R,B=q['coordinate_bounds_xyxy']
                if min(r,R)-max(x,X)>=128 and min(b,B)-max(y,Y)>=128:jobs.append((a,q))
    ee=[]
    with ProcessPoolExecutor(max_workers=4) as ex:
        for k,e in enumerate(ex.map(edge,jobs,chunksize=16),1):
            ee.append(e)
            if k%2000==0:print(json.dumps(dict(stage='source-coordinate-verification',done=k,total=len(jobs))),flush=True)
    jsonl(ROOT/f'meta/COORDINATE_EDGE_AUDIT_{t}.jsonl',ee)
    # Exclude identities whose decoded pixels or bytes would cross splits, rather than
    # moving historical assignments. Parent components retain existing isolation.
    conflict=set()
    for key in ['source_component_id','source_hash','pixel_hash']:
        ids=defaultdict(list)
        for a in aa:ids[a[key]].append(a)
        for gg in ids.values():
            if len({a['split'] for a in gg})>1:conflict.update(a['source_path'] for a in gg)
    links=defaultdict(set)
    for e in ee:
        if e['accepted']:links[e['a']].add(e['b']);links[e['b']].add(e['a'])
    bypath={a['source_path']:a for a in aa};domains=[]
    for a in aa:
        if a['source_path'] in conflict or min(a['width'],a['height'])<512 or a['std']<.5:continue
        members={a['source_path']};front=[a['source_path']]
        c=a['coordinate_bounds_xyxy']
        # At most a 3x3 native tile neighbourhood, no unobserved gap filling.
        if c:
            while front:
                p=front.pop()
                for q in sorted(links[p]):
                    cq=bypath[q]['coordinate_bounds_xyxy']
                    if q not in members and q not in conflict and abs(cq[0]-c[0])<=600 and abs(cq[1]-c[1])<=600:
                        members.add(q);front.append(q)
        mm=[bypath[p] for p in sorted(members)]
        if c:
            bounds=[min(q['coordinate_bounds_xyxy'][0] for q in mm),min(q['coordinate_bounds_xyxy'][1] for q in mm),max(q['coordinate_bounds_xyxy'][2] for q in mm),max(q['coordinate_bounds_xyxy'][3] for q in mm)]
        else:bounds=[0,0,a['width'],a['height']]
        domains.append(dict(anchor=a['source_path'],source_id=a['source_id'],source_component_id=a['source_component_id'],
            split=a['split'],source_scene=a['source_scene'],members=sorted(members),bounds_xyxy=bounds,
            width=bounds[2]-bounds[0],height=bounds[3]-bounds[1],source_hash=hashlib.sha256(json.dumps([(q['source_hash'],q['coordinate_bounds_xyxy']) for q in mm]).encode()).hexdigest(),
            source_kind='verified_adjacent_native_tiles' if len(mm)>1 else 'single_published_image',
            scene_status=a['scene_status'],raw_acquisition_identity_known=a['independent_raw_acquisition_identity_known']))
    jsonl(ROOT/f'meta/SOURCE_DOMAINS_{t}.jsonl',domains)
    counts=Counter((a['split'],a['dataset'],a['width']>=640 and a['height']>=640,a['width']>=512 and a['height']>=512) for a in aa)
    summary=dict(created=now(),original_published_files=len(aa),datasets=dict(Counter(a['dataset'] for a in aa)),
        dimension_counts=[dict(split=k[0],dataset=k[1],supports640=k[2],supports512=k[3],count=v) for k,v in counts.items()],
        domain_count=len(domains),domain_kinds=dict(Counter(d['source_kind'] for d in domains)),
        coordinate_edges_checked=len(ee),coordinate_edges_accepted=sum(e['accepted'] for e in ee),
        excluded_cross_split_files=sorted(conflict),eligible_scene_domains=dict(Counter(d['split']+'/'+d['source_scene'] for d in domains)),
        geometry_feasibility_not_yet_inferred_from_dimensions=True,elapsed_seconds=time.monotonic()-start)
    save(ROOT/f'meta/SOURCE_AUDIT_SUMMARY_{t}.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
