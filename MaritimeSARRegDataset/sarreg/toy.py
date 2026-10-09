"""Generated only for software smoke tests; it is not SAR data."""
from pathlib import Path
import cv2
import numpy as np
from .io import write_png,save_json


def make_toy(root, count=24, seed=17):
    root=Path(root)
    images=root/'JPEGImages';anns=root/'annotations'
    images.mkdir(parents=True,exist_ok=True)
    anns.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed)
    coco={'images':[],'annotations':[],'categories':[{'id':1,'name':'ship'}]}
    aid=1
    for i in range(count):
        size=800
        img=np.clip(rng.normal(35,8,(size,size)),0,255).astype('uint8')
        # add weak waves, coastline corner and static scattering to exercise transforms
        for k in range(7):
            px=int(rng.integers(0,size));py=int(rng.integers(0,size))
            cv2.line(img,(max(0,px-80),py),(min(size-1,px+80),min(size-1,py+int(rng.integers(-20,21)))),int(rng.integers(35,65)),1)
        if i%2==0:
            cv2.rectangle(img,(0,0),(120,800),70,-1)
            for _ in range(20):
                c=(int(rng.integers(0,118)),int(rng.integers(0,800)))
                cv2.rectangle(img,c,(min(119,c[0]+12),min(799,c[1]+20)),int(rng.integers(90,200)),-1)
        fname=f'DEMO{i//4:03d}_{i:04d}.png'
        coco['images'].append({'id':i+1,'file_name':fname,'width':size,'height':size})
        for k in range(int(rng.integers(2,7))):
            cx=int(rng.integers(145,740));cy=int(rng.integers(55,740))
            ww=int(rng.integers(15,42)); hh=int(rng.integers(5,15))
            poly=[cx-ww//2,cy-hh//2,cx+ww//2,cy-hh//2,cx+ww//2,cy+hh//2,cx-ww//2,cy+hh//2]
            pts=np.array(poly).reshape(-1,2).astype('int32')
            cv2.fillPoly(img,[pts],int(rng.integers(185,255)))
            coco['annotations'].append({'id':aid,'image_id':i+1,'category_id':1,
                'bbox':[cx-ww//2,cy-hh//2,ww,hh],'segmentation':[poly],'area':ww*hh,'iscrowd':0})
            aid+=1
        write_png(images/fname,img)
    save_json(anns/'train_test2017.json',coco)
    return {'images':count,'instances':aid-1,'root':str(root.resolve()),
            'warning':'Synthetic software demo, not genuine SAR imagery'}
