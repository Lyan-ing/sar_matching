"""Fixed offline data, V2.2-shaped explicit supervision; only images enter a model."""
from common import *
import argparse

def load_pair(row,use_ship_aux=False):
    if use_ship_aux:raise ValueError('This release does not provide synthetic ship positive labels; boxes remain exclusion proxies.')
    with np.load(row['gt_path'],allow_pickle=False) as f:z={k:f[k] for k in f.files}
    masks=lambda k:[z[k+'_A'],z[k+'_B']]
    negative=[]
    for L in 'AB':
        yy,xx=np.mgrid[12:row['height']-12:24,12:row['width']-12:24];xy=np.c_[xx.ravel(),yy.ravel()].astype('float32')
        k=bilinear_at(z['valid_'+L]&~z['ignore_'+L],xy)
        negative.append(xy[k][:64] if z['no_overlap'] else np.empty((0,2),np.float32))
    return dict(images=[read_gray(row['image_A']),read_gray(row['image_B'])],valid=masks('valid'),static=masks('static'),ignore=masks('ignore'),
        visibility=[v&~ig for v,ig in zip(masks('valid'),masks('ignore'))],ship_masks=masks('ship'),safe_static=masks('safe_static'),structure_support=masks('static'),
        calibration_mask=masks('calibration'),supervision_ignore=masks('supervision_ignore'),
        points0=z['structure_points_A'].astype('float32'),points1=z['structure_points_B'].astype('float32'),
        check_points0=z['check_points_A'],check_points1=z['check_points_B'],H_A=z['T_A'],H_B=z['T_B'],H_AB=z['H_AB'],
        source=dict(source_id=row['source_id'],source_group=row['source_component_id'],image_path=row['source_path']),source_origin_xy=z['source_origin_xy'].tolist(),
        overlap=row['overlap_ratio'],regime='none' if z['no_overlap'] else row['overlap_bin'],seed=row['seed'],dynamic_occlusion=False,
        loss_role=row['loss_role'],local_weight=row['local_weight'],local_descriptor_eligible=row['local_descriptor_eligible'],global_eligible=row['global_eligible'],
        negative_probe_points=negative,ship_points0=np.empty((0,2),np.float32),ship_points1=np.empty((0,2),np.float32),
        ship_aux_masks=[np.zeros_like(v) for v in masks('valid')],use_ship_aux=False,relation_id=row['pair_id'])

def tensors(z,device='cpu'):
    out=v21.tensors(z,device)
    import torch
    for k in ['supervision_ignore','ship_aux_masks']:out[k]=torch.as_tensor(np.stack(z[k]),device=device,dtype=torch.bool)[:,None]
    return out

class FixedPairs:
    """One pair per item; bucket by row['width'] or use batch_size=None for mixed sizes."""
    def __init__(self,manifest):self.rows=rows(manifest)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):return load_pair(self.rows[index])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');p.add_argument('--index',type=int,default=0);a=p.parse_args();ds=FixedPairs(a.manifest);row=ds.rows[a.index];z=ds[a.index];t=tensors(z)
    print(json.dumps(dict(pair_id=row['pair_id'],pairs=len(ds),tensor_shape=list(t['images'].shape),ground_points=len(z['points0']),checks=len(z['check_points0']),role=z['loss_role'],image_range=[float(t['images'].min()),float(t['images'].max())])))
