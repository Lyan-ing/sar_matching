"""Supplement representative map overlays in both native target frames, NOT GT."""
import sys
sys.dont_write_bytecode=True
import json,argparse
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw
from common import ROOT,source_rows
from geo import load_pair,pixel_grid

def main(out):
    out=Path(out).resolve();file=out/'visualization/manifest.json';manifest=json.loads(file.read_text());lookup={r['sample_id']:r for r in source_rows()}
    for item in manifest:
        sid=item['sample_id'];r=lookup[sid];_,valid,images,_,mapper=load_pair(r);reverse=mapper.reverse()(pixel_grid(valid[1].shape)).reshape(*valid[1].shape,2)
        reverse_warp=cv2.remap(images[0],reverse[:,:,0].astype(np.float32),reverse[:,:,1].astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);mask1=np.array(Image.open(out/'georeference_correspondences'/sid/'geographic_valid_overlap1.png'))>0;reverse_warp[~mask1]=0;forward=np.array(Image.open(out/'georeference_correspondences'/sid/'moving_native_geo_assisted.png'))
        two=Image.new('RGB',(1024,555),'white');d=ImageDraw.Draw(two);d.text((4,2),sid+' GEOGRAPHIC overlays only; NOT GT error',fill='black');d.text((4,20),'left: moving -> reference | right: reference -> moving; red=target, cyan=warped source',fill='black');two.paste(Image.fromarray(np.stack([images[0],forward,forward],-1)),(0,43));two.paste(Image.fromarray(np.stack([images[1],reverse_warp,reverse_warp],-1)),(512,43));path=out/'visualization'/f'{sid}_bidirectional.png'
        if path.exists():raise FileExistsError(path)
        two.save(path);item['bidirectional_geographic_preview']=str(path)
    # Extend this run's unsealed derived preview manifest; original manifests untouched.
    file.write_text(json.dumps(manifest,indent=2)+'\n');print('Bidirectional geographic previews',len(manifest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',default=str(ROOT));a=p.parse_args();main(a.out)
