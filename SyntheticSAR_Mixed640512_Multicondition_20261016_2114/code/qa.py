from generator import *
from reader import load_pair,tensors
from score import assess,summaries
from concurrent.futures import ProcessPoolExecutor
from PIL import Image,ImageDraw
import argparse,csv,itertools

DOMAINS=None
def verify(row):
    try:
        with np.load(row['gt_path'],allow_pickle=False) as f:z={k:f[k] for k in f.files}
        n=row['width'];H=z['H_AB'];assert n in [512,640] and row['height']==n
        assert np.allclose(H@z['H_BA'],np.eye(3),atol=1e-9)
        relative=z['T_B']@np.linalg.inv(z['T_A']);relative/=relative[2,2];assert np.allclose(relative,H,atol=1e-8)
        source=domain_arrays(DOMAINS[row['source_path']]);origin=z['source_origin_xy'];O=np.array([[1,0,origin[0]],[0,1,origin[1]],[0,0,1.]])
        valid=[z['valid_'+L] for L in 'AB'];ignored=[z['ignore_'+L] for L in 'AB'];images=[read_gray(row['image_'+L]) for L in 'AB'];maxerr=0.;contrast_ratios=[]
        for i,L in enumerate('AB'):
            A=z['T_'+L]@O;v=valid[i];ig=ignored[i];im=images[i]
            assert im.shape==(n,n) and im.dtype==np.uint8
            rebuilt=erode(cv2.warpPerspective(source['valid'].astype('float32'),A,(n,n),flags=cv2.INTER_LINEAR)>1-1e-6,1)
            assert np.array_equal(rebuilt,v) and v.mean()>=.98
            assert not im[~v].any();assert im[v].std()>=.5 and (im[v]==255).mean()<=.3 and (im[v]==0).mean()<=.5
            raw=warp(source['image'],A,n);raw[~v]=0
            pts=z['structure_points_'+L]
            if len(pts):
                # Measure preservation of coarse local contrast at existing source
                # anchors. This diagnostic never creates or relocates a point.
                xy=np.rint(pts).astype(int);values=[]
                for aa in [raw,im]:
                    coarse=cv2.GaussianBlur(aa.astype('float32'),(0,0),2.)
                    mean=cv2.boxFilter(coarse,-1,(17,17));variance=cv2.boxFilter(coarse*coarse,-1,(17,17))-mean*mean
                    values.append(np.sqrt(np.maximum(variance,0))[xy[:,1],xy[:,0]])
                ratios=values[1]/np.maximum(values[0],1e-4);contrast_ratios.extend(ratios[values[0]>=1.].tolist())
            if row['enhancement_profile']=='P0' or row['enhancement_params_'+L]['effective_profile']=='P0':
                assert np.array_equal(im,raw),'P0 must exactly preserve geometric image'
            st=z['static_'+L];ca=z['calibration_'+L]
            assert not (ig&~v).any()
            assert not (st&(~v|ig|dilate(z['ship_'+L],5))).any()
            assert not (ca&(~v|ig|st)).any()
            assert np.array_equal(z['supervision_ignore_'+L],v&~st&~ca)
            assert bilinear_at(st,z['structure_points_'+L]).all()
            assert bilinear_at(v&z['overlap_'+L]&~ig,z['check_points_'+L]).all()
            changed=z['appearance_changed_'+L];assert not (changed&~ig).any()
            params=row['enhancement_params_'+L];ops=params['ordered_ops']
            if params['effective_profile']=='P0':assert not ops and not changed.any()
            if params['effective_profile']=='P1':assert ops==['radiometry']
        overlaps=[m>0 for m in overlap_mask(*(v.astype('uint8')*255 for v in valid),H)]
        for L,o,v in zip('AB',overlaps,valid):
            assert np.array_equal(o,z['overlap_'+L]);assert abs(ratio(o,v)-row['overlap_'+L])<1e-12
        assert overlap_bin(min(row['overlap_A'],row['overlap_B']))==row['overlap_bin']
        assert angle_bin(angle_of(H,n))==row['angle_bin'];assert abs(angle_of(H,n)-row['rotation_actual_deg'])<1e-9
        for kind in ['structure','check']:
            p,q=z[kind+'_points_A'],z[kind+'_points_B']
            if len(p):
                err=max(float(np.linalg.norm(transform_points(H,p)-q,axis=1).max()),float(np.linalg.norm(transform_points(z['H_BA'],q)-p,axis=1).max()));maxerr=max(maxerr,err);assert err<.001
        assert len(z['structure_points_A'])==row['structural_anchor_count'] and len(z['check_points_A'])==row['check_point_count']
        assert global_ok(z['structure_points_A'],z['structure_points_B'],n)==row['global_eligible']
        if z['no_overlap']:
            assert max(row['overlap_A'],row['overlap_B'])==0 and not len(z['check_points_A']) and not len(z['structure_points_A'])
            polys=[footprint(z['T_'+L]@O,n) for L in 'AB'];assert cv2.intersectConvexConvex(*polys)[0]<=1e-4
        assert row['output_pixel_hashes']==[ahash(im) for im in images]
        assert row['image_file_hashes']==[sha(row['image_'+L]) for L in 'AB'] and sha(row['gt_path'])==row['gt_file_hash']
        return dict(pair_id=row['pair_id'],passed=True,size=n,geometry_max_error_px=maxerr,
            ignore_fractions=[float(x.mean()) for x in ignored],checks=len(z['check_points_A']),ground=len(z['structure_points_A']),
            pixel_hashes=row['output_pixel_hashes'],check_spatial_support=[spatial_support(z['check_points_'+L],n) for L in 'AB'],
            coarse_anchor_contrast_ratio_min=min(contrast_ratios) if contrast_ratios else None,
            coarse_anchor_contrast_ratio_median=float(np.median(contrast_ratios)) if contrast_ratios else None)
    except Exception as e:
        import traceback
        return dict(pair_id=row['pair_id'],passed=False,error=repr(e),trace=traceback.format_exc())

def contact(rr,path):
    # Representative artifact only; source training PNGs are never drawn on.
    canvas=Image.new('RGB',(1440,280*len(rr)),(25,25,25));draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(rr):
        with np.load(r['gt_path']) as z:
            for side,L in enumerate('AB'):
                im=Image.open(r['image_'+L]).convert('RGB');im.thumbnail((260,260));canvas.paste(im,(side*280,i*280+20))
                for point in z['structure_points_'+L]:
                    x=side*280+point[0]*im.width/r['width'];y=i*280+20+point[1]*im.height/r['height'];draw.ellipse((x-2,y-2,x+2,y+2),fill='#44ff77')
            panels=[]
            for L in 'AB':
                mask=np.zeros((r['width'],r['width'],3),np.uint8);mask[z['valid_'+L]]=[120,120,120];mask[z['overlap_'+L]]=[40,160,190];mask[z['ignore_'+L]]=[230,130,30];panels.append(mask)
            v=Image.fromarray(np.concatenate(panels,axis=1));v.thumbnail((520,260));canvas.paste(v,(600,i*280+20))
        label=f"{r['pair_id']} size={r['width']} overlap={r['overlap_ratio']:.3f} angle={r['rotation_actual_deg']:.1f} {r['enhancement_profile']} ground={r['structural_anchor_count']}"
        draw.text((5,i*280+2),label,fill='white');draw.text((1140,i*280+30),r['source_scene'].replace('_','\n'),fill='white')
    canvas.save(path,'JPEG',quality=92)

def scorer_checks(rr,folder):
    outcomes=[];true=[];wrong=[]
    for r in rr:
        with np.load(r['gt_path']) as z:
            pred=dict(accepted=not bool(z['no_overlap']),H_AB=z['H_AB'].tolist(),matches_A=z['structure_points_A'].tolist(),matches_B=z['structure_points_B'].tolist(),keypoints_A=z['structure_points_A'].tolist(),keypoints_B=z['structure_points_B'].tolist())
            a=assess(r,pred);true.append(a)
            if a['geometry_eligible']:assert a['Success@3px'] and a['bidirectional_error_px']<.001
            if a['structure_anchor_count']:assert a['structure_anchor_recall']==1 and a['structure_precision']==1 and a['structure_repeatability']==1
            bad=dict(pred,H_AB=(np.array([[1,0,60],[0,1,60],[0,0,1.]])@z['H_AB']).tolist());b=assess(r,bad);wrong.append(b)
            if b['geometry_eligible']:assert not b['Success@10px']
            if z['no_overlap']:
                falsely=assess(r,dict(accepted=True,H_AB=np.eye(3).tolist()));assert falsely['false_accept']
            missing=assess(r,{})
            if missing['geometry_eligible']:assert not missing['Success@10px']
    result=dict(passed=True,analytic_GT=summaries(true),wrong_60px_H=summaries(wrong),
        no_overlap_accept_counted_as_FPR=True,missing_predictions_kept_as_failure=True,synthetic_scorer_test_not_algorithm_performance=True)
    save(folder/f'SCORER_SELFTEST_{stamp()}.json',result);return result

def tables(rr,folder,t):
    keys=['split','width','overlap_bin','angle_bin','source_scene','enhancement_profile'];counts=Counter(tuple(r[k] for k in keys) for r in rr)
    with (folder/f'SIZE_OVERLAP_ANGLE_SCENE_PROFILE_COUNTS_{t}.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(keys+['count']);w.writerows([list(k)+[counts[k]] for k in itertools.product(['train','val'],[512,640],OVERLAPS,range(6),SCENES,['P0','P1','P2','P3','P4'])])
    for key in [('width','overlap_bin'),('angle_bin','overlap_bin'),('source_scene','overlap_bin'),('source_scene','enhancement_profile'),('split','width')]:
        c=Counter(tuple(r[k] for k in key) for r in rr)
        with (folder/('_'.join(key).upper()+f'_COUNTS_{t}.csv')).open('x',newline='') as f:
            w=csv.writer(f);w.writerow(list(key)+['count']);w.writerows([list(k)+[v] for k,v in sorted(c.items())])
    anchors=Counter(r['source_path'] for r in rr);raw=Counter(p for r in rr for p in r['visible_source_members'])
    with (folder/f'SOURCE_REUSE_{t}.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['source_path','anchor_pairs','actual_visible_pairs']);w.writerows([p,anchors[p],raw[p]] for p in sorted(anchors.keys()|raw.keys()))
    params=defaultdict(list)
    def walk(obj,key):
        if isinstance(obj,dict):
            for k,v in obj.items():
                if k not in ['seed','pixels','center_xy']:walk(v,key+'.'+k)
        elif isinstance(obj,list):
            for v in obj:walk(v,key)
        elif isinstance(obj,(float,int)) and not isinstance(obj,bool):params[key].append(float(obj))
    for r in rr:
        for L in 'AB':
            p=r['enhancement_params_'+L]
            for op in r['enhancement_ops_'+L]:walk(p[op],op)
    with (folder/f'ACTUAL_ENHANCEMENT_PARAMETER_STATS_{t}.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['parameter','count','min','median','max']);w.writerows([k,len(v),min(v),float(np.median(v)),max(v)] for k,v in sorted(params.items()))

def qa(manifests,smoke=False):
    global DOMAINS
    rr=[r for p in manifests for r in rows(p)];cfg=json.loads(latest('meta/GENERATION_CONFIG_*.json').read_text());DOMAINS={d['anchor']:d for d in rows(cfg['source_domains'])};t=stamp();folder=ROOT/('smoke_qa_'+t if smoke else 'qa_'+t);folder.mkdir(exist_ok=False)
    result=[];start=time.monotonic()
    with ProcessPoolExecutor(max_workers=4) as ex:
        for i,x in enumerate(ex.map(verify,rr,chunksize=2),1):
            result.append(x)
            if i%250==0 or i==len(rr):print(json.dumps(dict(stage='QA',checked=i,total=len(rr),failed=sum(not x['passed'] for x in result),seconds=round(time.monotonic()-start))),flush=True)
    jsonl(folder/f'PAIR_QA_{t}.jsonl',result);bad=[x for x in result if not x['passed']]
    if bad:
        save(folder/f'QA_FAILURES_{t}.json',bad);raise RuntimeError(bad[:2])
    pixelhashes=[h for r in rr for h in r['output_pixel_hashes']];splits={s:[r for r in rr if r['split']==s] for s in ['train','val']}
    sets={s:{p for r in x for p in r['source_members']} for s,x in splits.items()};components={s:{r['source_component_id'] for r in x} for s,x in splits.items()}
    assert not sets['train']&sets['val'] and not components['train']&components['val']
    audit={a['source_path']:a for a in rows(cfg['source_audit'])}
    for key in ['source_hash','pixel_hash']:
        a={audit[p][key] for p in sets['train']};b={audit[p][key] for p in sets['val']};assert not a&b
    pre=json.loads(latest('meta/PREREGISTRATION_*.json').read_text());assert all(sha(p)==h for p,h in pre['protected_files'].items())
    # The cached audit is input provenance. Verify actual selected raw file hashes once,
    # rather than rehashing the same raw tile on every generated relation.
    for p in sets['train']|sets['val']:assert sha(p)==audit[p]['source_hash']
    tensors_cases=[];seen=set()
    for r in rr:
        key=(r['width'],r['loss_role'])
        if key not in seen:
            z=load_pair(r);ts=tensors(z);assert tuple(ts['images'].shape)==(2,1,r['width'],r['width']);assert ts['images'].min()>=0 and ts['images'].max()<=1
            tensors_cases.append(dict(size=r['width'],role=r['loss_role'],points=len(z['points0'])));seen.add(key)
    scorer=scorer_checks(rr if smoke else [r for i,r in enumerate(rr) if i%100==0 or r['structural_anchor_count']>0][:100],folder)
    tables(rr,folder,t)
    opcounts=Counter(op for r in rr for L in 'AB' for op in r['enhancement_ops_'+L]);no_ops={}
    for name,ops in [('no_extra_speckle',['correlated_field','multiplicative_speckle']),('no_extra_texture',['fine_texture','correlated_field','smooth_clutter']),('no_local_scatter',['local_scatter']),('no_appearance_ops',['radiometry','fine_texture','correlated_field','multiplicative_speckle','directional_blur','local_scatter','smooth_clutter'])]:
        no_ops[name+'_views']=sum(not (set(r['enhancement_ops_'+L])&set(ops)) for r in rr for L in 'AB')
        no_ops[name+'_pairs_both_sides']=sum(all(not (set(r['enhancement_ops_'+L])&set(ops)) for L in 'AB') for r in rr)
    visible_raw={p for r in rr for p in r['visible_source_members']}
    contrast=[r['coarse_anchor_contrast_ratio_min'] for r in result if r.get('coarse_anchor_contrast_ratio_min') is not None]
    summary=dict(created=now(),passed=True,smoke=smoke,manifests=list(map(str,manifests)),pairs=len(rr),view_instances=len(rr)*2,
        unique_output_pixel_hashes=len(set(pixelhashes)),duplicate_output_pixel_instances=len(pixelhashes)-len(set(pixelhashes)),
        actual_visible_original_files=len({p for r in rr for p in r['visible_source_members']}),anchor_original_files=len({r['source_path'] for r in rr}),
        unique_visible_original_byte_hashes=len({audit[p]['source_hash'] for p in visible_raw}),unique_visible_original_pixel_hashes=len({audit[p]['pixel_hash'] for p in visible_raw}),
        source_components=len(set.union(*components.values())),source_component_overlap=0,raw_file_overlap=0,raw_byte_hash_overlap=0,raw_pixel_hash_overlap=0,
        split={s:dict(pairs=len(x),sizes=dict(Counter(r['width'] for r in x)),components=len(components[s]),anchor_files=len({r['source_path'] for r in x}),visible_original_files=len({p for r in x for p in r['visible_source_members']}),
            scenes=dict(Counter(r['source_scene'] for r in x)),overlaps=dict(Counter(r['overlap_bin'] for r in x)),angles=dict(Counter(r['angle_bin'] for r in x)),profiles=dict(Counter(r['enhancement_profile'] for r in x)),levels=dict(Counter(r['structure_supervision_level'] for r in x))) for s,x in splits.items()},
        geometry_max_error_px=max(x['geometry_max_error_px'] for x in result),geometry_mean_pair_max_error_px=float(np.mean([x['geometry_max_error_px'] for x in result])),
        check_point_counts=dict(min=min(x['checks'] for x in result),median=float(np.median([x['checks'] for x in result])),mean=float(np.mean([x['checks'] for x in result]))),
        positive_geometry_pairs=sum(r['overlap_ratio']>0 for r in rr),geometry_evaluable_pairs=sum(r['overlap_ratio']>0 and r['check_point_count']>=4 for r in rr),
        strict_no_overlap_pairs=sum(r['overlap_ratio']==0 for r in rr),with_structure_points=sum(r['structural_anchor_count']>0 for r in rr),local_descriptor_pairs=sum(r['local_descriptor_eligible'] for r in rr),
        strict_global_structure_pairs=sum(r['global_eligible'] for r in rr),levels=dict(Counter(r['structure_supervision_level'] for r in rr)),profiles=dict(Counter(r['enhancement_profile'] for r in rr)),
        actual_ops_per_view=dict(opcounts),actual_no_extra_operations=no_ops,
        coarse_structure_contrast_min_ratio=min(contrast) if contrast else None,
        source_valid_fraction_min=min(min(r['valid_fraction_A'],r['valid_fraction_B']) for r in rr),
        output_saturation_fraction_max=max(v['saturated_fraction'] for r in rr for v in r['intensity']),
        max_recorded_coarse_RMSE=max(v['coarse_RMSE'] for r in rr for v in r['intensity']),
        max_anchor_source_reuse=max(Counter(r['source_path'] for r in rr).values()),max_visible_raw_source_reuse=max(Counter(p for r in rr for p in r['visible_source_members']).values()),
        tensor_cases=tensors_cases,history_protected_hashes_unchanged=True,source_hashes_verified_once=True,all_visible_scene_labels_explicit_unknown=True,
        geometry_quality_does_not_establish_real_physical_GT=True,optimizer_steps=0,cuda_used=False,elapsed_seconds=time.monotonic()-start)
    save(folder/f'QA_SUMMARY_{t}.json',summary)
    if not smoke:
        frozen=[]
        for r in splits['val']:
            for p in [r['image_A'],r['image_B'],r['gt_path']]:frozen.append(dict(path=p,sha256=sha(p)))
        for p in [*manifests,latest('meta/GENERATION_CONFIG_*.json'),latest('meta/FROZEN_JOB_PLAN_*.jsonl')]:frozen.append(dict(path=str(p),sha256=sha(p)))
        jsonl(folder/f'VAL_FREEZE_HASHES_{t}.jsonl',frozen)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest',nargs='+',type=Path);p.add_argument('--smoke',action='store_true');a=p.parse_args();qa(a.manifest,a.smoke)
