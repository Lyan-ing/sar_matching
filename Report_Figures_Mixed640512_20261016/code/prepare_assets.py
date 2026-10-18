"""Freeze figure selection, copy exact PNGs; never generate/augment dataset samples."""
from source_usage import *
import shutil

SCENES=['offshore_vessels_low_texture_sea','nearshore_vessels','port_quay_dense_vessels','other_land_structures']
def gray(p):return cv2.imread(str(p),cv2.IMREAD_GRAYSCALE)
def quality(r):
    aa=[gray(r['image_'+L]) for L in 'AB']
    # Figure readability only. Never read model predictions, scores or matches.
    mean=min(float(a.mean()) for a in aa);std=min(float(a.std()) for a in aa)
    return mean*.65+min(std,70)*.35
def choose(rr,scene,overlaps=None,angles=None,profile=None,need_ops=None,need_structure=False):
    pool=[r for r in rr if r['source_scene']==scene and r['width']==640 and (overlaps is None or r['overlap_bin'] in overlaps) and
        (angles is None or r['angle_bin'] in angles) and (profile is None or r['enhancement_profile'] in profile)]
    if need_ops:pool=[r for r in pool if any(op in r['enhancement_ops_A']+r['enhancement_ops_B'] for op in need_ops)]
    if need_structure:pool=[r for r in pool if r['structural_anchor_count']>=28 and r['overlap_ratio']>=.4]
    assert pool,(scene,overlaps,angles,profile,need_ops)
    pool=sorted(pool,key=lambda r:(-min(v['std'] for v in r['intensity']),r['pair_id']))[:60]
    return sorted(pool,key=lambda r:(-quality(r),r['pair_id']))[0]
def run():
    e,rr,audit,ss=inputs();t=stamp();OUT.mkdir(exist_ok=True);(OUT/'raw_pairs').mkdir(exist_ok=True);(OUT/'raw_sources').mkdir(exist_ok=True)
    rule=dict(created=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        B_slots=[dict(overlap='zero',scene=SCENES[1],angle_bins=[4]),dict(overlap='10_30',scene=SCENES[2],angle_bins=[2]),
                 dict(overlap='30_50',scene=SCENES[1],angle_bins=[5]),dict(overlap='50_70',scene=SCENES[2],angle_bins=[3]),
                 dict(overlap='70_90',scene=SCENES[0],angle_bins=[1]),dict(overlap='90_100',scene=SCENES[2],angle_bins=[0])],
        size=640,preferred_profiles=['P0','P1'],readability='within target slot: top 60 by minimum side std; rank min-side mean*.65 + min(min-side std,70)*.35; pair-ID tie-break',
        E='one per source context; 50-90% overlap; 0-90deg; P0/P1; same deterministic brightness/contrast ranking',
        D='port source context; >=28 already-saved structure points for sparse display; no algorithm success used',
        C='existing five same-source/same-geometry controlled files only; no regeneration',
        C_sea='separate formal offshore examples; different sources/geometries explicitly retained; NOT a controlled before-after comparison',
        no_model_outputs_read=True,no_training_or_resampling=True,display_mapping='same fixed 0..255 grayscale; no contrast stretch/denoise/sharpen')
    save(OUT/f'meta/FIGURE_SELECTION_RULE_{t}.json',rule)
    B=[choose(rr,x['scene'],[x['overlap']],x['angle_bins'],['P0','P1']) for x in rule['B_slots']]
    E=[choose(rr,s,['50_70','70_90'],[0,1,2,3],['P0','P1']) for s in SCENES]
    D=choose(rr,SCENES[2],['50_70','70_90'],[0,1,2,3],['P0','P1','P2'],need_structure=True)
    controls=[r for r in rows(DATA/'smoke_20261016_2122/SMOKE_MANIFEST_20261016_2122.jsonl') if r['pair_id'].startswith('PROFILE_CONTROL_')]
    assert len(controls)==5
    with np.load(controls[0]['gt_path']) as z:baseT=[z['T_A'],z['T_B']]
    for r in controls:
        assert r['source_members']==controls[0]['source_members']
        with np.load(r['gt_path']) as z:assert np.array_equal(z['T_A'],baseT[0]) and np.array_equal(z['T_B'],baseT[1])
    sea=[]
    for i in range(5):
        op=None if i<2 else ['fine_texture'] if i==2 else ['local_scatter'] if i==3 else ['directional_blur','multiplicative_speckle']
        sea.append(choose(rr,SCENES[0],profile=['P'+str(i)],need_ops=op))
    chosen=dict(A=[D],B=B,C=controls,C_SEA=sea,D=[D],E=E);copied={};index=[]
    for fig,group in chosen.items():
        for slot,r in enumerate(group):
            for L in 'AB':
                src=Path(r['image_'+L]);dst=OUT/'raw_pairs'/f"{r['pair_id']}_{L}_{t}.png"
                if src not in copied:
                    assert not dst.exists();shutil.copyfile(src,dst);assert sha(src)==sha(dst);copied[src]=str(dst)
                index.append(dict(figure=fig,slot=slot+1,pair_id=r['pair_id'],formal_release=not r['pair_id'].startswith('PROFILE_CONTROL'),side=L,
                    native_png=copied[src],source_image=str(src),width=r['width'],height=r['height'],overlap_A=r['overlap_A'],overlap_B=r['overlap_B'],overlap_min=r['overlap_ratio'],
                    rotation_actual_2D_deg=r['rotation_actual_deg'],profile=r['enhancement_profile'],actual_ops=r['enhancement_ops_'+L],source_context=r['source_scene'],
                    verified_visible_scene=r['verified_visible_scene'],anchor_path=r['source_path'],source_members=r['source_members'],source_component=r['source_component_id'],
                    original_png_copied_byte_identical=True,gt_path=r['gt_path']))
    A_sources=[]
    for i,p in enumerate(D['visible_source_members'][:2]):
        src=Path(p);dst=OUT/'raw_sources'/f'A_SOURCE_{i+1}_{src.stem}_{t}{src.suffix.lower()}';assert not dst.exists();shutil.copyfile(src,dst);assert sha(src)==sha(dst);A_sources.append(dict(original=p,copied=str(dst),published_id=ss[p]['source_id'],bounds=audit[p]['coordinate_bounds_xyxy']))
    assert len(A_sources)==2
    write_csv(OUT/f'RAW_PAIR_PNG_INDEX_{t}.csv',index)
    plan=dict(timestamp=t,selection_rule=str(OUT/f'meta/FIGURE_SELECTION_RULE_{t}.json'),groups=chosen,A_sources=A_sources,raw_index=str(OUT/f'RAW_PAIR_PNG_INDEX_{t}.csv'),native_copies=copied and {str(k):v for k,v in copied.items()})
    save(OUT/f'meta/FIGURE_PLAN_{t}.json',plan)
    # Small selection proof sheet, no modifications to the originals.
    from PIL import ImageDraw
    sample=B+E+[D];canvas=Image.new('RGB',(1000,len(sample)*230),'white');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(sample):
        draw.text((4,i*230+2),f"{i} {r['pair_id']} {r['source_scene']} overlap={r['overlap_ratio']:.3f} angle={r['rotation_actual_deg']:.1f}",fill='black')
        for j,L in enumerate('AB'):
            im=Image.open(r['image_'+L]);im=im.convert('RGB');im.thumbnail((220,205),Image.Resampling.NEAREST);canvas.paste(im,(j*235,i*230+22))
    canvas.save(OUT/f'meta/SELECTION_PREVIEW_{t}.jpg',quality=94)
    print(json.dumps(dict(plan=str(OUT/f'meta/FIGURE_PLAN_{t}.json'),B=[r['pair_id'] for r in B],E=[r['pair_id'] for r in E],D=D['pair_id'],pure_original_pngs=len(copied)),ensure_ascii=False),flush=True)
if __name__=='__main__':run()
