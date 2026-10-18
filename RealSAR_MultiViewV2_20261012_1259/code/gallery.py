"""Honest native/context/Common-FOV gallery; no matching or GT points."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
import numpy as np,cv2
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geo_tools import *
FONT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
def panel(ims,title,labels,out,width=1024):
    height=max(round(im.height*width/im.width) for im in ims);canvas=Image.new('RGB',(2*width,height+105),'#151a22');d=ImageDraw.Draw(canvas);d.text((12,10),title,fill='white',font=FONT)
    for i,(im,label) in enumerate(zip(ims,labels)):
        h=round(im.height*width/im.width);canvas.paste(im.convert('RGB').resize((width,h),Image.Resampling.NEAREST),(i*width,90));d.text((i*width+10,43),label,fill='white',font=FONT)
    canvas.save(out)
def render():
    regions=lines(ROOT/'regions.jsonl');atlas=Image.new('RGB',(1536,((len(regions)+1)//2)*460+60),'#151a22');draw=ImageDraw.Draw(atlas);draw.text((12,12),'V2 mother regions | geography is NOT scatterer GT | independent human review pending',font=FONT,fill='white')
    for n,r in enumerate(regions):
        folder=ROOT/'regions'/r['sample_id'];views=r['views'];title=r['sample_id']+' | '+r['look_pair_category']+' | human/GT PENDING'
        for scope in ['native','context','common']:
            ims=[Image.open(folder/f'{scope}_db{i}.png') for i in [0,1]];labels=[f"{'reference' if i==0 else 'moving'} {im.width}x{im.height} | fixed per-image dB p1/p99" for i,im in enumerate(ims)];panel(ims,title+' | '+scope,labels,folder/f'{scope}_comparison.png')
        detail=[]
        for i in [0,1]:
            im=np.array(Image.open(folder/f'native_db{i}.png'));v=np.array(Image.open(folder/f'native_valid{i}.png'))>0;h,w=im.shape;size=min(256,h,w);blur=cv2.GaussianBlur(im.astype(np.float32),(0,0),2);edges=cv2.Canny(blur.astype(np.uint8),30,90)/255
            candidates=[]
            for y in np.linspace(0,h-size,5).astype(int):
                for x in np.linspace(0,w-size,5).astype(int):
                    if v[y:y+size,x:x+size].mean()>.9:candidates.append((edges[y:y+size,x:x+size].mean(),x,y))
            _,x,y=max(candidates) if candidates else (0,(w-size)//2,(h-size)//2);patch=Image.fromarray(im[y:y+size,x:x+size]);patch.save(folder/f'detail{i}.png');detail.append(dict(native_xywh=[int(x),int(y),size,size],chosen_independently=True,common_structure_confirmed=False))
        save(folder/'detail_provenance.json',detail);panel([Image.open(folder/f'detail{i}.png') for i in [0,1]],title+' | independently selected structure details; NOT matched ROIs',[str(d['native_xywh'])+' | nearest display enlargement only' for d in detail],folder/'details_comparison.png',512)
        fig,ax=plt.subplots(figsize=(7,6),layout='constrained');crs=r['metric_crs'];polys=[polygon(v['context_info'],crs) for v in views];roi=np.array(r['ground_ROI_metric']['coordinates'][0][:-1]);origin=roi.mean(0)
        for p,color,style,label in zip(polys,['#0072B2','#D55E00'],['-','--'],['reference native context','moving native context']):
            q=np.vstack([p,p[0]])-origin;ax.plot(q[:,0],q[:,1],style,color=color,label=label)
        q=np.vstack([roi,roi[0]])-origin;ax.plot(q[:,0],q[:,1],':',color='#009E73',linewidth=2,label='common ground ROI (NOT GT)');ax.set(xlabel='Local UTM east offset (m)',ylabel='Local UTM north offset (m)',title=r['sample_id']+'\nGeographic coverage only');ax.set_aspect('equal');ax.legend(fontsize=8);fig.savefig(folder/'coverage.png',dpi=180);plt.close(fig)
        gx=(n%2)*768;gy=60+(n//2)*460;ims=[Image.open(folder/f'native_db{i}.png') for i in [0,1]]
        draw.text((gx+8,gy+5),r['sample_id']+' | '+('DIFFERENT' if r['look_pair_category']=='different_look_verified' else 'SAME'),fill='white',font=FONT)
        draw.text((gx+8,gy+30),f"FOV {r['ground_edge_m']:.1f}m | shared technical valid {100*r['common_joint_geographic_valid_fraction']:.1f}%",fill='white',font=FONT)
        for i,im in enumerate(ims):
            h=round(im.height*384/im.width);atlas.paste(im.convert('RGB').resize((384,min(h,384)),Image.Resampling.NEAREST),(gx+i*384,gy+70))
    atlas.save(ROOT/'atlas.png')
    print('RENDERED',len(regions),'mother regions; individual raw-size PNGs retained',flush=True)
def candidate_atlas():
    regions=lines(ROOT/'candidate_regions.jsonl');canvas=Image.new('RGB',(1536,((len(regions)+1)//2)*460+60),'#151a22');d=ImageDraw.Draw(canvas);d.text((12,12),'V2 proposed candidates | independent static correspondence PENDING | native aspect ratio preserved',font=FONT,fill='white')
    for n,r in enumerate(regions):
        x=(n%2)*768;y=60+(n//2)*460;d.text((x+8,y+5),r['sample_id']+' | '+('DIFFERENT' if r['look_pair_category']=='different_look_verified' else 'SAME'),fill='white',font=FONT)
        d.text((x+8,y+30),f"FOV {r['ground_edge_m']:.1f}m | common geographic valid {100*r['common_joint_geographic_valid_fraction']:.1f}%",fill='white',font=FONT)
        for i in [0,1]:
            im=Image.open(ROOT/'regions'/r['sample_id']/f'native_db{i}.png').convert('RGB');im.thumbnail((384,360),Image.Resampling.NEAREST);canvas.paste(im,(x+i*384+(384-im.width)//2,y+70))
    canvas.save(ROOT/'candidate_atlas.png');print('CANDIDATE_ATLAS',len(regions),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--candidate-atlas',action='store_true');a=p.parse_args();candidate_atlas() if a.candidate_atlas else render()
