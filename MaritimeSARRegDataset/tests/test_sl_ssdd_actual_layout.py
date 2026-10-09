"""Regression checks based on the user's SL-SSDD/train, SL-SSDD/test layout.
Synthetic miniature fixtures: not a claim that private server data was executed.
"""
import json
import numpy as np
import pytest
from pathlib import Path
from sarreg.io import jsonl_read, write_png, read_gray
from sarreg.indexer import prepare_index
from sarreg.synthesis import generate
from sarreg.qa import validate_generated


def _create_fixture(tmp_path):
    base=tmp_path/'SSDD'/'Official-SSDD-OPEN'/'BBox_RBox_PSeg_SSDD'/'coco_style'
    source=base/'JPEGImages'
    annotations=base/'annotations'
    masks=tmp_path/'SL-SSDD'
    for folder in (source,annotations,masks/'train',masks/'test'):
        folder.mkdir(parents=True,exist_ok=True)
    img=np.full((512,512),52,np.uint8)
    img[90:200,250:300]=194
    for sp,name in [('train','000002'),('train','000003'),('test','000001')]:
        write_png(source/(name+'.jpg'),img)
        mask=np.full_like(img,255)
        mask[:90]=0
        write_png(masks/sp/(name+'.png'),mask)
    def coco(names):
        return {'images':[{'id':i+1,'file_name':name+'.jpg','width':512,'height':512} for i,name in enumerate(names)],
                'categories':[{'id':0,'name':'ship'}],
                'annotations':[{'id':i+1,'image_id':i+1,'category_id':0,'bbox':[250,90,50,110],
                                'segmentation':[[250,90,300,90,300,200,250,200]]} for i,_ in enumerate(names)]}
    (annotations/'train.json').write_text(json.dumps(coco(['000002','000003'])))
    (annotations/'test.json').write_text(json.dumps(coco(['000001'])))
    return tmp_path/'SSDD',masks


def test_real_style_sl_ssdd_split_mapping_and_generated_masks(tmp_path):
    ssdd,masks=_create_fixture(tmp_path)
    idx=tmp_path/'ssdd.jsonl'
    result=prepare_index('ssdd',ssdd,idx,sea_mask_root=masks)
    assert result['total_images']==3
    assert result['sea_masks']==3 and result['sea_masks_missing']==0
    assert result['sea_masks_by_official_split']=={'train':2,'test':1}
    rows=jsonl_read(idx)
    assert len(rows)==3
    assert all(Path(r['sea_mask_path']).parent.name==r['official_split'] for r in rows)
    out=tmp_path/'out'
    report=generate(idx,'configs/easy.yaml',out,split='all',count=4,seed=23)
    assert report['generated']==4,report
    qa=validate_generated(out)
    assert qa['failed']==0,qa
    for p in out.glob('*/pair_*/meta.json'):
        meta=json.loads(p.read_text())
        assert meta['sea_mask_provenance']['source_encoding']=={'land':0,'sea':255}
        for v in range(meta['num_views']):
            sea=read_gray(p.parent/f'sea_{v}.png')
            valid=read_gray(p.parent/f'valid_{v}.png')
            assert set(np.unique(sea)).issubset({0,255})
            assert np.count_nonzero((sea>0)&(valid==0))==0


def test_missing_wrong_split_mask_fails_instead_of_silent_match(tmp_path):
    ssdd,masks=_create_fixture(tmp_path)
    (masks/'train'/'000002.png').unlink()
    # Fake same stem under TEST; must not incorrectly attach it to train image.
    write_png(masks/'test'/'000002.png',np.full((512,512),255,np.uint8))
    ix=tmp_path/'bad.jsonl'
    with pytest.raises(ValueError,match='lack same-split SL-SSDD masks'):
        prepare_index('ssdd',ssdd,ix,sea_mask_root=masks)
    assert not ix.exists()
    r=prepare_index('ssdd',ssdd,ix,sea_mask_root=masks,allow_partial_sea_masks=True)
    assert r['sea_masks']==2 and r['sea_masks_missing']==1


def test_mismatched_mask_size_detected_by_generator(tmp_path):
    ssdd,masks=_create_fixture(tmp_path)
    write_png(masks/'test'/'000001.png',np.zeros((300,200),np.uint8))
    ix=tmp_path/'idx.jsonl'
    prepare_index('ssdd',ssdd,ix,sea_mask_root=masks)
    row=next(r for r in jsonl_read(ix) if r['official_split']=='test')
    from sarreg.synthesis import synthesize,load_cfg
    with pytest.raises(ValueError,match='Sea mask shape mismatch'):
        synthesize(row,load_cfg('configs/easy.yaml'),tmp_path/'bad_pair',seed=4)


def test_non_binary_mask_rejected(tmp_path):
    ssdd,masks=_create_fixture(tmp_path)
    write_png(masks/'test'/'000001.png',np.full((512,512),127,np.uint8))
    ix=tmp_path/'idx.jsonl'
    prepare_index('ssdd',ssdd,ix,sea_mask_root=masks)
    row=next(r for r in jsonl_read(ix) if r['official_split']=='test')
    from sarreg.synthesis import synthesize,load_cfg
    with pytest.raises(ValueError,match='not binary 0/255'):
        synthesize(row,load_cfg('configs/easy.yaml'),tmp_path/'bad_pair',seed=4)
