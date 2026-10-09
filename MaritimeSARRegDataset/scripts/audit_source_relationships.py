"""Conservative pixel-supported near-duplicate and overlapping-source audit.

Perceptual similarity retrieves candidates only. Scene joins require decoded
equality or extensive aligned non-ship pixel correlation. All settings/evidence
are exported; absence of candidates is not proof of acquisition independence.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from sarreg.io import read_gray, jsonl_read, jsonl_write, save_json
from scripts.make_leakage_safe_index import derive


def ncc(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    a=a-a.mean();b=b-b.mean()
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/norm) if norm>1e-8 else 0.


@lru_cache(maxsize=96)
def image(path):
    return read_gray(path)


def nonship(row,shape):
    mask=np.full(shape,255,np.uint8)
    for a in row['annotations']:
        x,y,w,h=a['bbox_xywh']
        mask[max(0,int(y)-5):min(shape[0],int(np.ceil(y+h))+5),
             max(0,int(x)-5):min(shape[1],int(np.ceil(x+w))+5)]=0
    return mask


def score_alignment(a,b,ra,rb,H):
    h,w=a.shape
    wb=cv2.warpPerspective(b,H,(w,h))
    vb=cv2.warpPerspective(nonship(rb,b.shape),H,(w,h),flags=cv2.INTER_NEAREST)
    inside=(nonship(ra,a.shape)>0)&(vb>0)
    inside=cv2.erode(inside.astype(np.uint8),np.ones((5,5),np.uint8))>0
    # Low texture is never accepted by visual similarity or NCC alone.
    if inside.sum()<1024 or inside.mean()<.25: return None
    av,bv=a[inside],wb[inside]
    if min(float(av.std()),float(bv.std()))<4: return None
    blur_a=cv2.GaussianBlur(a,(5,5),1.0)
    blur_b=cv2.warpPerspective(cv2.GaussianBlur(b,(5,5),1.0),H,(w,h))
    raw=ncc(av,bv);blur=ncc(blur_a[inside],blur_b[inside])
    grad_a=cv2.Sobel(blur_a,cv2.CV_32F,1,0)+cv2.Sobel(blur_a,cv2.CV_32F,0,1)
    grad_b=cv2.Sobel(blur_b,cv2.CV_32F,1,0)+cv2.Sobel(blur_b,cv2.CV_32F,0,1)
    gradient=ncc(grad_a[inside],grad_b[inside])
    # Both images must contain substantial common background, not one ship.
    back=cv2.warpPerspective(np.ones(a.shape,np.uint8),np.linalg.inv(H),(b.shape[1],b.shape[0]),flags=cv2.INTER_NEAREST)
    if back.mean()<.25: return None
    if raw>=.96 and blur>=.99 and gradient>=.95:
        return {'raw_nonship_ncc':raw,'blur_nonship_ncc':blur,'gradient_ncc':gradient,
                'common_nonship_fraction_a':float(inside.mean()),'common_fraction_b':float(back.mean()),
                'H_b_to_a':H.tolist()}
    return None


def verify_pair(ra,rb):
    a,b=image(ra['image_path']),image(rb['image_path'])
    if a.shape==b.shape:
        result=score_alignment(a,b,ra,rb,np.eye(3))
        if result: return dict(result,method='aligned_pixels')
    # Geometric verification is an audit of shared source pixels, not a Phase 2 benchmark.
    detector=cv2.SIFT_create(nfeatures=600,contrastThreshold=.035)
    ka,da=detector.detectAndCompute(a,None);kb,db=detector.detectAndCompute(b,None)
    if da is None or db is None or len(db)<2:return None
    good=[m for m,n in cv2.BFMatcher().knnMatch(da,db,k=2) if m.distance<.65*n.distance]
    if len(good)<12:return None
    pa=np.float32([ka[m.queryIdx].pt for m in good]);pb=np.float32([kb[m.trainIdx].pt for m in good])
    H,inliers=cv2.findHomography(pb,pa,cv2.RANSAC,1.5)
    if H is None or inliers.sum()<10 or inliers.mean()<.6:return None
    if not np.isfinite(H).all() or np.linalg.cond(H)>1e8:return None
    result=score_alignment(a,b,ra,rb,H)
    return dict(result,method='geometric_pixels',inliers=int(inliers.sum()),matches=len(good)) if result else None


def audit(rows):
    cv2.setNumThreads(1);cv2.setRNGSeed(20261009)
    thumb=[];phash=[];hashes=defaultdict(list);coords=defaultdict(list)
    descriptors=[];owners=[]
    orb=cv2.ORB_create(nfeatures=120,fastThreshold=20)
    for i,r in enumerate(rows):
        a=read_gray(r['image_path'])
        digest=hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest();hashes[digest].append(i)
        t=cv2.resize(a,(64,64),interpolation=cv2.INTER_AREA).astype(np.float32)
        thumb.append(t)
        d=cv2.dct(cv2.resize(t,(32,32)))[:8,:8].ravel();phash.append(np.packbits(d>np.median(d[1:])))
        if r['dataset'] in ('hrsid','background'):
            coords[tuple(r['source_id'].split('_')[1:])].append(i)
        scale=min(1.,384/max(a.shape));small=cv2.resize(a,(round(a.shape[1]*scale),round(a.shape[0]*scale)))
        _,ds=orb.detectAndCompute(small,None)
        if ds is not None:
            descriptors.append(ds);owners.extend([i]*len(ds))
        if i%500==0:print(f'Fingerprinted {i}/{len(rows)}',flush=True)
    thumb=np.asarray(thumb);phash=np.asarray(phash);pairs=set();routes=Counter()
    exact=[]
    for group in hashes.values():
        if len(group)>1:
            exact.append(group)
            for j in group[1:]:pairs.add((group[0],j))
    bitcounts=np.array([x.bit_count() for x in range(256)],np.uint8)
    # Exhaustive descriptor distances: no visual-similarity scene merge.
    for i in range(len(rows)):
        distances=bitcounts[np.bitwise_xor(phash[i],phash[i+1:])].sum(axis=1)
        for j in np.where(distances<=10)[0]+i+1:
            if rows[i]['source_group']==rows[j]['source_group'] and rows[i]['dataset']==rows[j]['dataset']:continue
            if ncc(thumb[i].ravel(),thumb[j].ravel())>=.85:
                pairs.add((i,int(j)));routes['phash_thumbnail']+=1
    for group in coords.values():
        for ix,i in enumerate(group):
            for j in group[ix+1:]:
                if rows[i]['source_group']==rows[j]['source_group']:continue
                if ncc(thumb[i].ravel(),thumb[j].ravel())>=.90:
                    pairs.add((min(i,j),max(i,j)));routes['shared_crop_coordinates']+=1
    print('Pixel/coordinate candidates',len(pairs),flush=True)
    # Local-feature retrieval also covers shifted crops and cross-dataset copies.
    desc=np.vstack(descriptors);owners=np.asarray(owners)
    matcher=cv2.FlannBasedMatcher(dict(algorithm=6,table_number=8,key_size=16,multi_probe_level=1),dict(checks=32))
    matcher.add([desc]);matcher.train();votes=Counter()
    for start in range(0,len(desc),5000):
        batch=matcher.knnMatch(desc[start:start+5000],k=8)
        for q,near in enumerate(batch):
            i=int(owners[start+q]);seen=set()
            for match in near:
                j=int(owners[match.trainIdx])
                if i==j or match.distance>32 or j in seen:continue
                if rows[i]['source_group']==rows[j]['source_group'] and rows[i]['dataset']==rows[j]['dataset']:continue
                seen.add(j);votes[min(i,j),max(i,j)]+=1
        if start%100000==0:print(f'Local descriptor retrieval {start}/{len(desc)}',flush=True)
    feature_candidates=[k for k,v in votes.items() if v>=10]
    pairs.update(feature_candidates);routes['local_feature_votes']=len(feature_candidates)
    print('Total candidates',len(pairs),'verify background pixels',flush=True)
    accepted=[];rejected=[]
    exact_pairs={(min(g[0],j),max(g[0],j)) for g in exact for j in g[1:]}
    for k,(i,j) in enumerate(sorted(pairs)):
        evidence={'method':'decoded_pixel_equality'} if (i,j) in exact_pairs else verify_pair(rows[i],rows[j])
        edge={'a':rows[i]['image_path'],'b':rows[j]['image_path'],
              'input_splits':[rows[i]['split'],rows[j]['split']],
              'input_groups':[rows[i]['source_group'],rows[j]['source_group']]}
        if evidence:accepted.append(dict(edge,**evidence))
        else:rejected.append(edge)
        if k%100==0:print(f'Verified {k}/{len(pairs)}, accepted={len(accepted)}',flush=True)
    return {'decoded_duplicate_groups':[[rows[i]['image_path'] for i in g] for g in exact],
            'retrieval':dict(routes),'candidate_pairs':len(pairs),'accepted':accepted,'rejected':rejected,
            'source_inventory_sha256':hashlib.sha256(''.join(sorted(hashes)).encode()).hexdigest(),
            'thresholds':{'phash_hamming':10,'thumbnail_ncc':.85,'nonship_raw_ncc':.96,'nonship_blur_ncc':.99,
                          'gradient_ncc':.95,'minimum_common_fraction':.25,'minimum_nonship_std':4},
            'limitations':['Candidate retrieval is not exhaustive for every transform/crop; absence is not proof of independent acquisitions.',
                           'Scene joins require shared pixel texture outside annotated ships; visual similarity of pure sea is rejected.',
                           'HRSID filename parent groups are retained and merged by accepted edges, including background crops.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True)
    p.add_argument('--out-index',required=True);p.add_argument('--report',required=True)
    a=p.parse_args();out=Path(a.out_index);reportpath=Path(a.report)
    if out.exists() or reportpath.exists():raise FileExistsError('Audit destination exists; reuse its evidence')
    rows=jsonl_read(a.index);report=audit(rows)
    edges=[[{'path':e['a']},{'path':e['b']}] for e in report['accepted']]
    result,changes=derive(rows,{'byte_identical_sources':edges})
    for new,old in zip(result,rows):
        new['source_group_original']=old.get('source_group_original',old['source_group'])
        new['original_experiment_split']=old.get('original_experiment_split',old['split'])
        new['previous_experiment_split']=old['split']
        new['split_protocol']='phase1_pixel_supported_source_components_v2'
    jsonl_write(out,result)
    report.update(input_index=str(Path(a.index).resolve()),input_sha256=hashlib.sha256(Path(a.index).read_bytes()).hexdigest(),
                  output_index=str(out.resolve()),output_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
                  changed_split_rows=len(changes),changes=changes,
                  accepted_cross_split_edges=sum(e['input_splits'][0]!=e['input_splits'][1] for e in report['accepted']))
    lookup={r['image_path']:r for r in result}
    report['remaining_accepted_cross_split_edges']=sum(lookup[e['a']]['split']!=lookup[e['b']]['split'] for e in report['accepted'])
    save_json(reportpath,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('accepted','rejected','changes','decoded_duplicate_groups')},indent=2))


if __name__=='__main__':main()
