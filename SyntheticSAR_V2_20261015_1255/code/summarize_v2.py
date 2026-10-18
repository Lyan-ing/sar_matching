"""Evidence tables and bounded visual QA, with explicitly labelled proxy metrics."""
from dataset_v2 import *
from collections import Counter,defaultdict
from PIL import Image,ImageDraw
import argparse,csv

def table(path,records):
    with Path(path).open('x',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]) if records else []);w.writeheader();w.writerows(records)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('summary',type=Path);a=parser.parse_args();summary=json.loads(a.summary.read_text());allrows=rows(summary['all_generated']);rr=[r for r in allrows if r['qa_state']=='usable'];t=stamp()
    counts=[]
    for arm in ['V1-matched','V2-R','V2-S']:
        for split in ['train','val']:
            for pool in ['core','ssdd','all']:
                selected=[r for r in rr if r['arm']==arm and r['split']==split and (pool=='all' or (r['source_dataset']=='ssdd')==(pool=='ssdd'))]
                counts.append(dict(arm=arm,split=split,pool=pool,pairs=len(selected),views=2*len(selected),original_images=len({r['source_path'] for r in selected}),
                    components=len({r['known_component'] for r in selected}),match_positive=sum(r['loss_role']=='structural_positive' for r in selected),
                    limited_structure=sum(r['loss_role']=='limited_structure' for r in selected),no_structure=sum(r['loss_role']=='no_structural_positive' for r in selected),
                    no_overlap=sum(r['profile']=='none' for r in selected),actual_dynamic=sum(bool(r['dynamic_events']) for r in selected),local_scattering=sum(bool(r['local_scattering_events']) for r in selected)))
    table(ROOT/f'COUNTS_{t}.csv',counts)
    groups=Counter((r['arm'],r['split'],r['source_scene_category'],r['angle_bin_2D'],
                    'negative' if r['profile']=='none' else 'low_.12_.30' if r['profile']=='low' else 'normal_.50_.90',r['loss_role']) for r in rr)
    table(ROOT/f'SCENE_ANGLE_OVERLAP_SUPPORT_{t}.csv',[dict(arm=k[0],split=k[1],scene=k[2],angle_2D=k[3],overlap=k[4],support=k[5],pairs=v) for k,v in sorted(groups.items())])
    scenes=Counter((r['arm'],r['split'],r['source_scene_category'],r['loss_role']) for r in rr)
    table(ROOT/f'SCENES_{t}.csv',[dict(arm=k[0],split=k[1],scene=k[2],support=k[3],pairs=v) for k,v in sorted(scenes.items())])
    sizes=Counter((r['arm'],r['split'],r['width']) for r in rr)
    table(ROOT/f'SIZES_{t}.csv',[dict(arm=k[0],split=k[1],size=k[2],pairs=v) for k,v in sorted(sizes.items())])
    angles=Counter((r['arm'],r['split'],r['angle_bin_2D'],r['loss_role']) for r in rr)
    table(ROOT/f'ANGLES_2D_{t}.csv',[dict(arm=k[0],split=k[1],angle_bin=k[2],support=k[3],pairs=v) for k,v in sorted(angles.items())])
    # Select without model results: all source scenes, bins, sizes, negatives and actual dynamic/ignore.
    ref=[r for r in rr if r['arm']=='V2-R'];selected=[];criteria=[lambda r:r['source_scene_category']=='port_quay_dense_vessels' and r['loss_role']=='structural_positive' and r['split']=='val',
        lambda r:r['source_scene_category']=='other_land_structures' and r['loss_role']=='structural_positive',
        lambda r:r['source_scene_category']=='nearshore_vessels' and r['loss_role']=='structural_positive',
        lambda r:r['source_scene_category']=='offshore_vessels_low_texture_sea',lambda r:r['source_dataset']=='ssdd',
        lambda r:r['profile']=='none',lambda r:r['profile']=='low' and r['loss_role']=='structural_positive',
        lambda r:r['angle_bin_2D']=='90_120' and r['loss_role']=='structural_positive',lambda r:bool(r['dynamic_events']) and r['loss_role']=='structural_positive',
        lambda r:r['loss_role']=='limited_structure',lambda r:r['source_scene_category']=='unclassified_background',lambda r:r['width']==512 and r['loss_role']=='structural_positive']
    for criterion in criteria:
        found=next((r for r in ref if criterion(r) and r['sample_id'] not in {s['sample_id'] for s in selected}),None)
        if found:selected.append(found)
    byid=defaultdict(dict)
    for r in rr:byid[r['sample_id']][r['arm']]=r
    visualpaths=[];metrics=[]
    for page in range((len(selected)+3)//4):
        canvas=Image.new('RGB',(1536,4*320+44),'white');draw=ImageDraw.Draw(canvas)
        for col,arm in enumerate(['V1-matched','V2-R','V2-S']):draw.text((col*512+12,10),arm+' left | right (native display)',fill='black')
        for j,base in enumerate(selected[page*4:page*4+4]):
            y=44+j*320
            for col,arm in enumerate(['V1-matched','V2-R','V2-S']):
                r=byid[base['sample_id']][arm];z=load_training_pair(r)
                for i in range(2):
                    im=cv2.cvtColor(z['images'][i],cv2.COLOR_GRAY2RGB)
                    # Orange outlines = ignore. Green = structure anchors. No display contrast stretch.
                    contours,_=cv2.findContours(z['ignore'][i].astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(im,contours,-1,(255,150,0),1)
                    for p in z['points0' if i==0 else 'points1']:cv2.circle(im,tuple(np.round(p).astype(int)),2,(30,230,90),1)
                    tile=Image.fromarray(im);tile.thumbnail((254,254));canvas.paste(tile,(col*512+i*256,y))
                draw.text((col*512+3,y+258),f"{r['sample_id']} {r['split']} {r['source_dataset']} {r['width']}px {r['image_rotation_2D_deg']:.1f}deg",fill='black')
                draw.text((col*512+3,y+274),r['source_scene_category'][:45],fill='black')
                draw.text((col*512+3,y+290),f"anchors {r['structural_anchor_count']} {r['loss_role']} over {min(r['overlap_fractions']):.2f}",fill='black')
        path=ROOT/f'APPEARANCE_COMPARE_{page:02d}_{t}.jpg';canvas.save(path,quality=94);visualpaths.append(str(path))
    # Native aligned high-frequency correlation is a diagnostic, NOT independent speckle evidence.
    # Avoid interpolation-dependent conclusions: compare each new arm with the same fixed baseline.
    for base in selected:
        for arm in ['V1-matched','V2-R','V2-S']:
            r=byid[base['sample_id']][arm];z=load_training_pair(r);H=z['H_AB'];size=r['width']
            y0=z['images'][0].astype('float32')/255.;y1=z['images'][1].astype('float32')/255.
            aligned=warp(y1,np.linalg.inv(H),size);joint=erode(z['safe_static'][0]&(warp(z['safe_static'][1].astype('uint8')*255,np.linalg.inv(H),size,mask=True)>0),6)
            for i in range(2):joint &= ~z['ignore'][0] if i==0 else ~(warp(z['ignore'][1].astype('uint8')*255,np.linalg.inv(H),size,mask=True)>0)
            a0=y0-cv2.GaussianBlur(y0,(0,0),1.5);a1=aligned-cv2.GaussianBlur(aligned,(0,0),1.5)
            corr=float(np.corrcoef(a0[joint],a1[joint])[0,1]) if joint.sum()>100 and min(float(a0[joint].std()),float(a1[joint].std()))>1e-6 else None
            metrics.append(dict(sample_id=r['sample_id'],arm=arm,scene=r['source_scene_category'],pixels=int(joint.sum()),aligned_high_frequency_correlation=corr,
                note='display-space residual after interpolation; not physical speckle independence or estimated ENL'))
    table(ROOT/f'SHARED_TEXTURE_DIAGNOSTIC_{t}.csv',metrics)
    # Bounded native input patch statistics to constrain display noise, not estimate physical looks.
    sources=rows(json.loads(Path(summary['config']).read_text())['source_pool']);patchstats=[]
    for dataset in ['hrsid','background','ssdd']:
        candidates=[r for r in sources if r['dataset']==dataset]
        for scene in sorted(set(r['scene_category'] for r in candidates)):
            for r in [q for q in candidates if q['scene_category']==scene][:6]:
                im=read_gray(r['image_path']).astype('float32')/255.;h,w=im.shape
                for k in range(3):
                    x=int((w-64)*(k+1)/4);y=int((h-64)*(3-k)/4);tile=im[y:y+64,x:x+64];blur=cv2.GaussianBlur(tile,(0,0),1.5)
                    patchstats.append(dict(dataset=dataset,scene=scene,source_id=r['source_id'],x=x,y=y,width=64,height=64,mean_DN01=float(tile.mean()),std_DN01=float(tile.std()),high_frequency_std=float((tile-blur).std()),physical_ENL=None))
    table(ROOT/f'NATIVE_DISPLAY_PATCH_STATS_{t}.csv',patchstats)
    summary2=dict(at=now(),counts=counts,examples=visualpaths,example_ids=[r['sample_id'] for r in selected],visual_review_status='pending_AI_inspection_not_human_GT',
        qa_before_generated_pair_records=len(allrows),qa_after_usable_pair_records=len(rr),quarantined_pair_records=len(allrows)-len(rr),
        unique_geometry_relations=len({r['sample_id'] for r in rr}),distinct_original_images=len({r['source_path'] for r in rr}),
        native_patch_statistics=str(ROOT/f'NATIVE_DISPLAY_PATCH_STATS_{t}.csv'),texture_diagnostic=str(ROOT/f'SHARED_TEXTURE_DIAGNOSTIC_{t}.csv'))
    save(ROOT/f'DISTRIBUTION_SUMMARY_{t}.json',summary2);print(json.dumps({k:v for k,v in summary2.items() if k!='counts'},ensure_ascii=False))
if __name__=='__main__':main()
