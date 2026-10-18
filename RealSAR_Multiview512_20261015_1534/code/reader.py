"""Prediction API consumes ONLY the pure-image manifest, never geographic targets."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

def image_records(path):
 for line in Path(path).read_text().splitlines():
  if not line.strip():continue
  r=json.loads(line)
  if set(r)!={'sample_id','image0','image1'}:raise ValueError('Prediction manifest must contain image paths and opaque ID only')
  yield r

def load_images(record):
 if set(record)!={'sample_id','image0','image1'}:raise ValueError('Do not pass custodian metadata into prediction')
 ims=[np.asarray(Image.open(record[k])) for k in ['image0','image1']]
 if any(im.ndim!=2 or max(im.shape)>512 or im.dtype!=np.uint8 for im in ims):raise ValueError('Native uint8 SAR image-only inputs <=512 required')
 return tuple(ims)

def evaluation_checks(gt_record):
 if not gt_record.get('GT_qualified') or not gt_record.get('actual_two_people'):raise ValueError('Actual independently verified physical GT is required; candidate imagery is not precision evidence')
 eligible=set(gt_record['quality']['check_point_ids'])
 return [p for p in gt_record['points'] if p['point_id'] in eligible and p['role']=='check']
