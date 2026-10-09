"""Manifest reader for image-only algorithms; load evaluation GT separately."""
import json
from pathlib import Path
import numpy as np
from .io import jsonl_read, read_gray


def iter_pairs(root):
    """Yield 900 pair relations from 860 fixed groups (including triple-view groups)."""
    root=Path(root)
    for row in jsonl_read(root/'manifest.jsonl'):
        folder=root/row['folder']
        meta=json.loads((folder/'meta.json').read_text())
        for p in meta['pairs']:
            i,j=p['views']
            yield {'pair_id':f'{row["sample_id"]}:{i}-{j}', 'folder':folder,
                   'image0':folder/f'view_{i}.png','image1':folder/f'view_{j}.png',
                   'views':(i,j),'profile':row['profile'],'stratum':row['stratum'],
                   'dataset':row['dataset'],'source_group':row['source_group'],
                   'source_id':row['source_id']}


def load_images(pair):
    """Native-size uint8 gray images. No Homography or annotation supplied to matcher."""
    return read_gray(pair['image0']),read_gray(pair['image1'])


def load_groundtruth(pair):
    folder=pair['folder'];i,j=pair['views']
    meta=json.loads((folder/'meta.json').read_text())
    p=next(p for p in meta['pairs'] if p['views']==[i,j])
    with np.load(folder/p['matches']) as m:
        gt={'points0':m['points_i'].copy(),'points1':m['points_j'].copy()}
    gt.update(H_0_to_1=np.asarray(p['H_i_to_j'],np.float64),H_1_to_0=np.asarray(p['H_j_to_i'],np.float64),
              overlap0=read_gray(folder/p['overlap_mask_i']),overlap1=read_gray(folder/p['overlap_mask_j']),
              overlap_fraction0=p['overlap_fraction_i'],overlap_fraction1=p['overlap_fraction_j'])
    for key,prefix in [('valid','valid'),('ships','ships'),('dynamic','dynamic')]:
        gt[key+'0']=read_gray(folder/f'{prefix}_{i}.png');gt[key+'1']=read_gray(folder/f'{prefix}_{j}.png')
    gt['sea0']=read_gray(folder/f'sea_{i}.png') if (folder/f'sea_{i}.png').exists() else None
    gt['sea1']=read_gray(folder/f'sea_{j}.png') if (folder/f'sea_{j}.png').exists() else None
    gt['ship_annotations0']=meta['ship_annotations'][i];gt['ship_annotations1']=meta['ship_annotations'][j]
    return gt
