"""Fixtures based on the uploaded 2026-10-09 directory-structure report.
They test the *layout*, not the full 6,764 real files absent from this runtime.
"""
import json
from pathlib import Path
import numpy as np
from sarreg.io import write_png, jsonl_read
from sarreg.indexer import prepare_index


def image(p, value=80):
    write_png(p, np.full((192, 192), value, dtype=np.uint8))


def coco(p, records, *, cid=1):
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'images': [{'id': i + 1, 'file_name': n, 'width': 192, 'height': 192} for i,n in enumerate(records)],
        'categories': [{'id': cid, 'name': 'ship'}],
        'annotations': [{'id': i+1, 'image_id': i+1, 'category_id': cid, 'bbox': [60,60,15,12],
                         'segmentation': [[60,60,75,60,75,72,60,72]]} for i,_ in enumerate(records)]
    }
    p.write_text(json.dumps(data))


def test_hrsid_real_layout_preserve_scenes_and_deduplicate(tmp_path):
    root = tmp_path/'HRSID'
    base = root/'HRSID_JPG'
    paths = [f'P0001_{i}_800_10190_10990.jpg' for i in (0,1200,2400)]
    paths += ['P0002_0_800_9000_9800.jpg']
    for p in paths: image(base/'JPEGImages'/p)
    coco(base/'annotations'/'train2017.json', paths[:2])
    coco(base/'annotations'/'test2017.json', paths[2:])
    coco(base/'annotations'/'train_test2017.json', paths)
    coco(base/'inshore_offshore'/'inshore.json', paths[:1])
    coco(base/'inshore_offshore'/'offshore.json', paths[1:])
    image(root/'pure background'/'P0001_4000_4800_0_800.png')
    ix = tmp_path/'h.jsonl'
    result = prepare_index('hrsid', root, ix)
    rows = jsonl_read(ix)
    assert result['total_images'] == 4 and result['ships'] == 4
    assert result['official_splits'] == {'train': 2, 'test': 2}
    assert result['scene_types'] == {'inshore': 1, 'offshore': 3}
    assert result['split_policy'] == 'group'
    assert result['cross_split_groups'] == 0
    assert len({r['image_path'] for r in rows}) == 4
    by_p = [r for r in rows if r['source_group'] == 'P0001']
    assert len({r['split'] for r in by_p}) == 1
    ixbg = tmp_path/'bg.jsonl'
    bg = prepare_index('background',root/'pure background',ixbg)
    assert bg['total_images'] == 1 and bg['ships'] == 0
    assert jsonl_read(ixbg)[0]['source_group'] == 'P0001'
    assert bg['split_policy'] == 'group'
    official_idx = tmp_path/'h_official.jsonl'
    r = prepare_index('hrsid',root,official_idx,split_policy='official',val_fraction=0)
    official_rows = jsonl_read(official_idx)
    assert sorted(x['split'] for x in official_rows).count('test') == 2
    assert r['cross_split_groups'] == 1  # expected same P0001 scene in both official splits


def test_ssdd_real_duplicate_layout_selects_single_coco_branch(tmp_path):
    root = tmp_path/'SSDD'/'Official-SSDD-OPEN'
    filenames_train = ['000002.jpg','000003.jpg']
    filenames_test = ['000001.jpg']
    for branch,value in [('BBox_RBox_PSeg_SSDD',100),('BBox_SSDD',180)]:
        base = root/branch/'coco_style'
        for name in filenames_train+filenames_test:
            image(base/'JPEGImages'/name,value)
        coco(base/'annotations'/'train.json',filenames_train,cid=0)
        coco(base/'annotations'/'test.json',filenames_test,cid=0)
        coco(base/'annotations'/'test_inshore.json',['000001.jpg'],cid=0)
        coco(base/'annotations'/'test_offshore.json',[],cid=0)
    # Add duplicate names inside a ship mask directory: should not be mistaken for SAR scene imagery.
    image(root/'PSeg_SSDD'/'voc_style'/'JPEGImages_PSeg_GT_Mask'/'000001.jpg',255)
    ix = tmp_path/'ssdd.jsonl'
    result = prepare_index('ssdd',root.parent,ix)
    rows = jsonl_read(ix)
    assert result['total_images'] == 3 and result['ships'] == 3
    assert result['official_splits'] == {'train':2,'test':1}
    assert len({r['source_id'] for r in rows}) == 3
    assert all('BBox_RBox_PSeg_SSDD/coco_style/JPEGImages/' in r['image_path'] for r in rows)
    assert all('PSeg_GT_Mask' not in r['image_path'] for r in rows)
    assert result['scene_types']['inshore'] == 1
    assert result['split_policy'] == 'official'


def test_do_not_import_ssdd_ship_instance_mask_as_sea_land(tmp_path):
    root=tmp_path/'SSDD'/'Official-SSDD-OPEN'/'BBox_SSDD'/'coco_style'
    image(root/'JPEGImages'/'000001.jpg')
    coco(root/'annotations'/'train.json',['000001.jpg'],cid=0)
    bad=tmp_path/'SSDD'/'Official-SSDD-OPEN'/'PSeg_SSDD'/'voc_style'/'JPEGImages_PSeg_GT_Mask'
    image(bad/'000001.jpg',255)
    import pytest
    with pytest.raises(ValueError,match='SEA-LAND'):
        prepare_index('ssdd',tmp_path/'SSDD',tmp_path/'bad.jsonl',sea_mask_root=bad)
