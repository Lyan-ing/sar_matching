"""One source feature pass, caching old and proposed evidence; no model predictions."""
from common22 import *
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from PIL import Image,ImageDraw
import time

def worker(r):
    im,union,instances,sea,pold,_=v21.source_material(r,CFG21)
    x=im.astype('float32')/255.;a=cv2.GaussianBlur(x,(0,0),2);b=cv2.GaussianBlur(x,(0,0),4)
    std=np.sqrt(np.maximum(cv2.GaussianBlur(b*b,(0,0),7)-cv2.GaussianBlur(b,(0,0),7)**2,0));eig=cv2.cornerMinEigenVal(b,7,3)
    strong=(std>=.025)&(eig>=.00008)&erode(np.ones_like(im,bool),10)
    raw=persistent(a,b,strong)
    safe=~dilate(union,10)&erode(np.ones_like(im,bool),10)
    nobg=safe.copy()
    if sea is not None:nobg &= ~dilate(sea,2)
    proposal=persistent(a,b,strong&nobg)
    # Existing sea masks can expose small land regions despite an offshore source label.
    active=r['scene_category'] in ['nearshore_vessels','port_quay_dense_vessels','other_land_structures'] or sea is not None
    points=proposal if active else np.empty((0,2),np.float32)
    # Synthetic ship-only points, from original pixels, not polygon vertices or added noise.
    sa=cv2.GaussianBlur(x,(0,0),.6);sb=cv2.GaussianBlur(x,(0,0),1.)
    sm=erode(union>0,1)&(cv2.cornerMinEigenVal(sb,3,3)>=.00002)
    ship=persistent(sa,sb,sm,dist=4,tolerance=2.)
    # Require a real segmentation instance. BBox-only regions stay excluded candidates.
    ids=np.full(len(ship),-1,np.int32);segmented=np.zeros_like(union)
    for k,(_,m,method) in enumerate(instances):
        if method=='bbox_proxy':continue
        segmented |= m
        if len(ship):
            ij=np.rint(ship).astype(int);inside=m[ij[:,1],ij[:,0]]>0;ids[(ids<0)&inside]=k
    ship=ship[ids>=0];ids=ids[ids>=0]
    # Calibration is supported by sea semantics AND homogeneous pre-augmentation texture.
    sea_evidence=sea if sea is not None else np.ones_like(im,bool) if r['scene_category']=='offshore_vessels_low_texture_sea' else np.zeros_like(im,bool)
    calib=sea_evidence&safe&(std<.010)&(eig<.00002)&(cv2.GaussianBlur(x,(0,0),4)>.005)
    if len(proposal):
        guard=np.zeros_like(im)
        for xy in proposal:cv2.circle(guard,tuple(np.rint(xy).astype(int)),16,255,-1)
        calib &= guard==0
    p=cachepath(r);assert not p.exists()
    np.savez_compressed(p,old=pold,proposal=proposal,ground=points,ship=ship,ship_instance_index=ids,
        source_safe=nobg,calibration=calib,ship_union=union>0,segmented_ship=segmented>0)
    return dict(source_path=r['image_path'],source_id=r['source_id'],dataset=r['dataset'],split=r['split'],component=r['known_component'],scene=r['scene_category'],
        raw_persistent=len(raw),after_ship_and_sea=len(proposal),old_source_points=len(pold),proposed_source_points=len(points),
        semantic_review_needed=not active and len(proposal)>=4,aux_ship_points=len(ship),segmented_ship_pixels=int((segmented>0).sum()),
        calibration_fraction=float(calib.mean()),source_std=float(im.std()),cache=str(p),raw_candidate_pool='independent extractor before mask, not a strict same-point causal subtraction')

def main():
    (ROOT/'features').mkdir(exist_ok=False);ss=sources();results=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,r in enumerate(ex.map(worker,ss,chunksize=2),1):
            results.append(r)
            if i%500==0 or i==len(ss):print(json.dumps(dict(done=i,total=len(ss),seconds=round(time.monotonic()-start))),flush=True)
    t=stamp();jsonl(ROOT/f'SOURCE_FEATURE_AUDIT_{t}.jsonl',results)
    summary=dict(at=now(),sources=len(ss),old_ge4=sum(r['old_source_points']>=4 for r in results),proposed_ge4=sum(r['proposed_source_points']>=4 for r in results),
        old_train_core_ge4=sum(r['split']=='train' and r['dataset']!='ssdd' and r['old_source_points']>=4 for r in results),
        review_needed=sum(r['semantic_review_needed'] for r in results),ship_ge4=sum(r['aux_ship_points']>=4 for r in results),old_v21_roles=dict(Counter(r['loss_role'] for r in old_rows())))
    save(ROOT/f'SOURCE_FEATURE_SUMMARY_{t}.json',summary)
    review=sorted([r for r in results if r['semantic_review_needed']],key=lambda r:(r['split'],r['scene'], -r['after_ship_and_sea']))[:96]
    jsonl(ROOT/f'SEMANTIC_REVIEW_INDEX_{t}.jsonl',review)
    for page in range((len(review)+47)//48):
        canvas=Image.new('RGB',(1440,1920),'#171717');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(review[page*48:page*48+48]):
            x0=j%6*240;y0=j//6*240;im=Image.open(r['source_path']).convert('RGB');im.thumbnail((236,205));canvas.paste(im,(x0,y0+35));draw.text((x0+2,y0+2),f"{j+page*48:02d} {r['dataset']} {r['split']} pts {r['after_ship_and_sea']}",fill='white');draw.text((x0+2,y0+17),r['source_id'][:26],fill='white')
        canvas.save(ROOT/f'SEMANTIC_REVIEW_{page:02d}_{t}.jpg')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
