"""Canonical, append-only delivery manifests; never edit generated images or GT."""
from common import *

def canonical():
    t=stamp();summary=json.loads(latest('meta/BUILD_SUMMARY_*.json').read_text());cfgpath=Path(summary['config']);cfg=json.loads(cfgpath.read_text());out=[]
    for split in ['train','val']:
        for r in rows(latest(f'manifests/{split}_*.jsonl')):
            r['metadata_version']='canonical_actual_ops_1'
            for L in 'AB':
                p=r['enhancement_params_'+L];selected=list(p['ordered_ops']);actual=[op for op in selected if op!='local_scatter' or len(p.get('local_scatter',{}).get('events',[]))>0]
                r['enhancement_selected_ops_'+L]=selected;r['enhancement_ops_'+L]=actual;p['selected_ops']=selected;p['ordered_ops']=actual
                p['skipped']=list(dict.fromkeys(p.get('skipped',[])+(['local_scatter:no_eligible_bright_region'] if 'local_scatter' in selected and 'local_scatter' not in actual else [])))
                if p['effective_profile']=='P0':p['skipped']=['radiometry','fine_texture','correlated_field','multiplicative_speckle','directional_blur','local_scatter','smooth_clutter']
                p['global_operation_affected_region']='valid_'+L;p['local_scatter_affected_mask']='appearance_changed_'+L
            r['raw_generation_record']=str(ROOT/'pairs'/split/r['pair_id']/f"{r['pair_id']}_RECORD_{TAG}.json")
            out.append(r)
    assert len(out)==6000 and Counter(r['split'] for r in out)==dict(train=4800,val=1200)
    for split in ['train','val']:jsonl(ROOT/f'manifests/{split}_CANONICAL_{t}.jsonl',[r for r in out if r['split']==split])
    save(ROOT/f'meta/RELEASE_METADATA_{t}.json',dict(created=now(),config=str(cfgpath),config_hash=sha(cfgpath),pairs=len(out),
        manifests={s:str(ROOT/f'manifests/{s}_CANONICAL_{t}.jsonl') for s in ['train','val']},
        metadata_only_normalization='selected local-scatter operations with zero realized events are recorded as skip, not actual changes',
        images_GT_seeds_splits_sampling_parameters_unchanged=True,raw_generation_records_preserved=True))
    print(json.dumps(dict(pairs=len(out),created=now(),manifests={s:str(ROOT/f'manifests/{s}_CANONICAL_{t}.jsonl') for s in ['train','val']})),flush=True)
if __name__=='__main__':canonical()
