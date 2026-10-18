from build import *
from qa import qa

def matrix():
    cfgpath=latest('meta/GENERATION_CONFIG_*.json');cfg=json.loads(cfgpath.read_text());h=sha(cfgpath);ds=rows(cfg['source_domains']);t=stamp();folder=ROOT/f'matrix_smoke_{t}';folder.mkdir(exist_ok=False)
    out=[];fail=[];probes=[]
    for size in [640,512]:
        for b,over in enumerate(OVERLAPS):
            for profile in range(5):
                sc=SCENES[(b+profile)%4]
                if b==0 and sc==SCENES[3]:sc=SCENES[2]
                ab=(b+profile)%6
                if b==5 and ab in [1,2]:ab=0
                seed=stable_seed(44,f'{size}/{b}/{profile}')
                spec=dict(pair_id=f'MATRIX_{size}_{over}_P{profile}',split='train',source_scene=sc,overlap_bin=over,angle_bin=ab,enhancement_profile='P'+str(profile),geometry_kind=['rigid','similarity','projective'][profile%3],seed=seed,geometry_seed=stable_seed(seed,'geometry'),validation_force_size=size)
                candidates=[d for d in ds if d['split']=='train' and d['source_scene']==sc];candidates.sort(key=lambda d:(-d['width']*d['height'],stable_seed(seed,d['anchor'])))
                for d in candidates[:40]:
                    r=attempt((d,spec,cfg,h,str(folder)))
                    if r['ok']:out.append(r['row']);break
                    if r['trace']:raise RuntimeError(r['trace'])
                    fail.append(r)
                else:raise RuntimeError(('matrix capacity',spec,fail[-1]))
                print(json.dumps(dict(stage='matrix-smoke',done=len(out),size=size,overlap=over,profile=profile)),flush=True)
    manifest=folder/f'MATRIX_MANIFEST_{t}.jsonl';jsonl(manifest,out);jsonl(folder/f'MATRIX_REJECTIONS_{t}.jsonl',fail)
    qa([manifest],smoke=True)
    # Actual bounded proxies on representative source layouts, not a claim that
    # every image of the same dimensions has a full, valid rectangular field.
    chosen=[];counts=Counter()
    for d in sorted(ds,key=lambda d:stable_seed(112,d['anchor'])):
        k=(d['split'],d['source_scene'],d['source_kind'])
        if d['source_scene'] in SCENES and counts[k]<3:chosen.append(d);counts[k]+=1
    pcfg=json.loads(json.dumps(cfg));pcfg['geometry']['attempts_per_size']=20
    for d in chosen:
        for size in [640,512]:
            for b,over in enumerate(OVERLAPS):
                seed=stable_seed(113,d['anchor']+over+str(size));ab=[0,2,3,4,5,0][b]
                spec=dict(geometry_seed=seed,overlap_bin=over,angle_bin=ab,geometry_kind='rigid')
                try:r=sample_geometry(d,spec,pcfg,force_size=size,probe=True);passed=True;reason=None
                except ValueError as e:passed=False;reason=str(e)
                probes.append(dict(source=d['anchor'],split=d['split'],scene=d['source_scene'],size=size,overlap_bin=over,angle_bin=ab,
                    source_kind=d['source_kind'],passed_proxy=passed,reason=reason,not_full_output_QA=True))
    jsonl(folder/f'SOURCE_FEASIBILITY_PROBES_{t}.jsonl',probes)
    # Per-source dimensional upper bounds are explicitly distinguished from actual
    # accepted geometry. Full per-pair support is checked during formal generation.
    capacities=[]
    for split in ['train','val']:
        for scene in SCENES:
            pool=[d for d in ds if d['split']==split and d['source_scene']==scene]
            for size in [640,512]:
                for over in OVERLAPS:
                    def fits(d):
                        if over=='zero':return max(d['width'],d['height'])>=2*size+8 and min(d['width'],d['height'])>=size+8
                        return min(d['width'],d['height'])>=size+8
                    upper=sum(fits(d) for d in pool);sub=[r for r in probes if r['split']==split and r['scene']==scene and r['size']==size and r['overlap_bin']==over]
                    capacities.append(dict(split=split,scene=scene,size=size,overlap_bin=over,source_domain_count=len(pool),dimension_support_upper_estimate=upper,
                        source_reuse_capacity_upper_estimate=upper*cfg['max_anchor_pairs_'+split],actual_proxy_domains_tested=len(sub),proxy_passed=sum(r['passed_proxy'] for r in sub),
                        estimate_not_guaranteed_pair_capacity=True))
    jsonl(ROOT/f'meta/PREGEN_FEASIBILITY_ESTIMATES_{t}.jsonl',capacities)
    save(ROOT/f'meta/PREFLIGHT_GATE_{t}.json',dict(passed=True,created=now(),matrix_manifest=str(manifest),matrix_pairs=len(out),sizes=[640,512],
        overlap_bins=OVERLAPS,profiles=['P0','P1','P2','P3','P4'],all_60_output_pairs_passed_full_QA=True,
        source_proxy_cases=len(probes),generation_config_hash=h,model_selection_used=False,
        implemented_code_hashes={str(p):sha(p) for p in (ROOT/'code').glob('*.py')}))

if __name__=='__main__':matrix()
