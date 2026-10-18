"""Small NumPy/OpenCV adapter to the existing training pair dictionary.
No model, optimizer, server, database or training process is started.
"""
from pathlib import Path
import json
import numpy as np,cv2

def _gray(p):
 im=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
 if im is None:raise ValueError('Unreadable image: '+str(p))
 return im

def load_inference_images(row):
 """Only the two image arrays leave this function; no GT/geographic input."""
 return [_gray(row['view0']),_gray(row['view1'])]

def _bilinear_valid(mask,xy):
 p=np.asarray(xy);ix=np.floor(p).astype(int);x,y=ix.T
 good=np.isfinite(p).all(1)&(x>=0)&(y>=0)&(x+1<mask.shape[1])&(y+1<mask.shape[0])
 k=np.flatnonzero(good);xx,yy=x[k],y[k]
 good[k]&=mask[yy,xx]&mask[yy+1,xx]&mask[yy,xx+1]&mask[yy+1,xx+1]
 return good

def load_training_pair(row,max_points=128):
 images=load_inference_images(row);valid=[_gray(p)>0 for p in row['valid_masks']]
 static=[_gray(p)>0 for p in row['static_supervision_masks']]
 ships=[_gray(p)>0 for p in row['all_vessel_masks']]
 with np.load(row['static_correspondences']) as z:p=z['points_i'];q=z['points_j']
 keep=_bilinear_valid(static[0],p)&_bilinear_valid(static[1],q);p,q=p[keep],q[keep]
 if max_points is not None and len(p)>max_points:
  selected=np.linspace(0,len(p)-1,max_points).astype(int);p,q=p[selected],q[selected]
 meta=json.loads(Path(row['metadata']).read_text())
 assert all(im.shape==(row['height'],row['width']) for im in images+valid+static+ships)
 assert all(not np.any(m&s) for m,s in zip(static,ships))
 return dict(images=images,valid=valid,static=static,ship_masks=ships,
  H_A=np.asarray(meta['H_source_to_view'][0]),H_B=np.asarray(meta['H_source_to_view'][1]),
  H_AB=np.asarray(row['H_0_to_1']),points0=p,points1=q,overlap=min(row['overlap_fractions']),
  source=dict(dataset=row['source_dataset'],source_id=row['source_id'],source_group=row['known_component'],image_path=row['source_path']),
  source_origin_xy=[0,0],regime=row['profile'],dynamic_occlusion=row['synthetic_dynamic_events']>0,seed=row['seed'])

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args()
 data=[json.loads(s) for s in a.manifest.read_text().splitlines() if s.strip()]
 selected=[];seen=set()
 for r in data:
  key=(r['profile'],r['source_scene_category'],r['construction_pool'])
  if key not in seen:selected.append(r);seen.add(key)
 for r in selected:
  x=load_training_pair(r);p0=x['points0'];p1=x['points1'];z=np.c_[p0,np.ones(len(p0))]@x['H_AB'].T
  if len(z):assert np.linalg.norm(z[:,:2]/z[:,2:]-p1,axis=1).max()<1e-3
  assert len(x['images'])==2 and all(max(im.shape)<=512 for im in x['images'])
 print(json.dumps(dict(passed=True,loaded_representatives=len(selected),profiles=sorted(set(r['profile'] for r in selected)),
  model_or_training_started=False,ground_truth_never_in_inference_images=True)))
