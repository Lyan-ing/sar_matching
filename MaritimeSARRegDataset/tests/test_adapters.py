from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from sarreg.io import write_png,jsonl_read
from sarreg.indexer import prepare_index
from sarreg.synthesis import generate
from sarreg.qa import validate_generated


def test_ssdd_voc_with_sl_ssdd_masks_and_dynamic(tmp_path):
    root=tmp_path/'SSDD'/'BBox_SSDD'/'voc_style'
    images=root/'JPEGImages'; xmls=root/'Annotations'; sets=root/'ImageSets'/'Main'
    masks=tmp_path/'masks'
    for p in (images,xmls,sets,masks):p.mkdir(parents=True,exist_ok=True)
    img=np.full((560,650),35,dtype=np.uint8)
    img[200:218,300:352]=220
    write_png(images/'000001.png',img)
    sea=np.full(img.shape,255,dtype=np.uint8)
    sea[:60]=0
    write_png(masks/'000001.png',sea)
    xml='''<annotation><filename>000001.png</filename><object><name>ship</name><bndbox><xmin>300</xmin><ymin>200</ymin><xmax>352</xmax><ymax>218</ymax></bndbox></object></annotation>'''
    (xmls/'000001.xml').write_text(xml)
    (sets/'train.txt').write_text('000001\n')
    ix=tmp_path/'index.jsonl'
    result=prepare_index('ssdd',tmp_path/'SSDD',ix,sea_mask_root=masks)
    assert result['total_images']==1 and result['ships']==1 and result['sea_masks']==1
    c=tmp_path/'cfg.yaml'
    c.write_text('''output_size: 256
views: 3
rotation_degrees: 9
scale_min: 0.9
scale_max: 1.1
shear_degrees: 1.5
perspective_fraction: 0.004
min_overlap: 0.50
max_overlap: 0.85
min_valid_fraction: 0.8
max_sampling_attempts: 300
min_source_side_factor: 1.35
min_ship_visible_fraction: 0.1
min_ship_pixels: 5
augmentation:
  enabled: true
  speckle_sigma: 0.1
dynamic_ships:
  enabled: true
  move_probability: 1.0
  max_shift_pixels: 8
  min_area: 5
correspondences_per_pair: 64
''')
    out=tmp_path/'output'
    rep=generate(ix,c,out,split='all',count=2)
    assert rep['generated']==2,rep
    report=validate_generated(out)
    assert report['failed']==0,report
    pair=next(out.glob('*/pair_*'))
    assert (pair/'sea_2.png').exists()
    assert (pair/'matches_1_2.npz').exists()
