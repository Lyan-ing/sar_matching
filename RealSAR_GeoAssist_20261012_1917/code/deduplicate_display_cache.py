"""Remove only our redundant display caches; frozen PNGs and raw TIFFs untouched."""
import json
from geometry import ROOT,save,sha

def run():
    p=ROOT/'display_cache.json';doc=json.loads(p.read_text());original=dict(doc)
    alias={};assets={};hash_to_key={};removed=[]
    for key,entry in doc['assets'].items():
        signature=entry['source_sha256']
        if signature not in hash_to_key:
            hash_to_key[signature]=key;assets[key]=entry;alias[key]=key
        else:
            alias[key]=hash_to_key[signature]
            for name in ['array','overview']:
                from pathlib import Path
                file=Path(entry[name]);assert file.parent==ROOT/'cache'
                removed.append(dict(path=str(file),bytes=file.stat().st_size,sha256=sha(file)))
    save(ROOT/'evidence/display_cache_before_dedup.json',original)
    save(ROOT/'evidence/display_cache_dedup_plan.json',dict(removed=removed,frozen_files_removed=0))
    from pathlib import Path
    for e in removed:Path(e['path']).unlink()
    doc.update(assets=assets,task_view_index={sid:alias[key] for sid,key in doc['task_view_index'].items()},
        distinct_display_windows=len(assets),bytes=sum(Path(v['array']).stat().st_size for v in assets.values()))
    p.rename(ROOT/'evidence/display_cache_initial_manifest.json')
    save(p,doc);print('DEDUP',len(assets),'unique displays for',len(doc['task_view_index']),'task views, bytes',doc['bytes'])

if __name__=='__main__':run()
