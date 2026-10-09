"""Focused native-scale, no-overlap, dynamic and conservative audit checks."""
import json
import cv2
import numpy as np
from sarreg.io import write_png, read_gray
from sarreg.synthesis import synthesize, load_cfg, _dynamic_move
from sarreg.qa import validate_pair
from scripts.audit_source_relationships import verify_pair
from scripts.summarize_phase1 import check_sample


def test_native_scale_no_overlap_gt(tmp_path):
    p=tmp_path/'source.png';write_png(p,np.random.default_rng(3).integers(0,200,(300,400),dtype=np.uint8))
    row={'dataset':'generic','source_id':'a','source_group':'a','split':'test','image_path':str(p),'annotations':[]}
    cfg=load_cfg('configs/easy.yaml');cfg.update(output_size=96,allow_source_upsampling=False,min_source_side_factor=3.,min_overlap=0.,max_overlap=0.)
    folder=tmp_path/'pair';synthesize(row,cfg,folder,44)
    m=json.loads((folder/'meta.json').read_text())
    assert m['source_resize_factors_xy']==[1.,1.] and m['output_size']==96
    assert m['pairs'][0]['match_count']==0
    assert all(np.count_nonzero(read_gray(folder/name))==0 for name in ('overlap_0_1_in_0.png','overlap_0_1_in_1.png'))
    assert validate_pair(folder)['passed']
    assert check_sample(folder,row)[0]['passed']


def test_dynamic_mask_cannot_enter_invalid_footprint():
    image=np.zeros((64,64),np.uint8);mask=image.copy();mask[20:40,20:40]=255;image[mask>0]=200
    valid=np.full_like(image,255);valid[:,:15]=0
    entry={'instance_id':'1','visible_fraction_estimate':1.0}
    events=[]
    _,entries,_=_dynamic_move(image,[(entry,mask)],{'enabled':True,'move_probability':1,'max_shift_pixels':24},np.random.default_rng(12),valid,events)
    assert events
    assert all(np.count_nonzero(m[valid==0])==0 for _,m in entries)


def test_visual_sea_similarity_is_not_source_identity(tmp_path):
    a=np.clip(np.random.default_rng(1).normal(40,8,(256,256)),0,255).astype(np.uint8)
    b=np.clip(np.random.default_rng(2).normal(40,8,(256,256)),0,255).astype(np.uint8)
    pa,pb=tmp_path/'a.png',tmp_path/'b.png';write_png(pa,a);write_png(pb,b)
    row=lambda p:{'image_path':str(p),'annotations':[]}
    assert verify_pair(row(pa),row(pb)) is None
    write_png(tmp_path/'copy.png',np.clip(a.astype(float)+3,0,255).astype(np.uint8))
    assert verify_pair(row(pa),row(tmp_path/'copy.png')) is not None
