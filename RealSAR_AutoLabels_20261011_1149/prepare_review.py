"""Diagnostic audit selection, explicitly biased to successes/failures, no GT."""
import json,shutil
from pathlib import Path
from common import ROOT,MANUAL,save,sha

def main(out=ROOT):
    out=Path(out).resolve();rows=[json.loads(s) for s in (out/'dataset_index.jsonl').read_text().splitlines()];lookup={r['sample_id']:r for r in rows};prior=[json.loads(s) for s in (MANUAL/'annotation_tasks/tasks.jsonl').read_text().splitlines()];old={r['sample_id']:r for r in prior}
    chosen=[lookup[s] for s in json.loads((MANUAL/'annotation_tasks/demo5.json').read_text())];used_sites={r['site_name'] for r in chosen};used_pairs={r['pair_id'] for r in chosen}
    # Three extra sites favor B candidates, increasing diagnostic coverage;
    # explicitly no statistical unbiased test claim.
    while len(chosen)<8:
        candidates=[r for r in rows if r['site_name'] not in used_sites and r['pair_id'] not in used_pairs]
        r=min(candidates,key=lambda r:(not r['has_B_pseudo'],r['pseudo_quality']!='high_candidate',r['sensor']=='Sentinel-1',r['sample_id']));chosen.append(r);used_sites.add(r['site_name']);used_pairs.add(r['pair_id'])
    adapter=out/'annotation_adapter'
    for d in ['annotation_tasks','candidate_matches','annotations','gt_quality']:(adapter/d).mkdir(exist_ok=True)
    html=(MANUAL/'annotation_app/index.html').read_text().replace('主动显示 SIFT 候选','主动显示自动伪标签').replace('真实 SAR 原生图像对应点 · 独立 GT 待人工核验','真实 SAR 诊断抽查 · 地理参考 / 自动伪标签 / 独立人工证据分开')
    (adapter/'index.html').write_text(html);js=(MANUAL/'annotation_app/app.js').read_text().replace("message(r.warning+' 候选数 '+r.points.length);","message(r.warning+' 图层 '+r.event.source+' 生成器 '+r.event.generator+' 候选数 '+r.points.length);")
    (adapter/'app.js').write_text(js);shutil.copyfile(MANUAL/'annotation_app/coordinates.js',adapter/'coordinates.js');tasks=[]
    for i,r in enumerate(chosen):
        sid=r['sample_id'];geo=json.loads(Path(r['geographic_correspondences_path']).read_text())['points'];pseudo=json.loads(Path(r['pseudo_correspondences_path']).read_text())['points'] if r['has_B_pseudo'] else []
        payload={'sample_id':sid,'geographic':geo,'sift_candidates':pseudo,'legacy_slot_mapping':{'geographic':'georeference_correspondences','sift_candidates':'pseudo_correspondences; actual generator below'},'pseudo_generator':r['pseudo_generator'],'pseudo_quality':r['pseudo_quality'],'independent_GT':False,'geo_input_sha256':sha(r['geographic_correspondences_path']),'pseudo_input_sha256':sha(r['pseudo_correspondences_path']) if r['has_B_pseudo'] else None}
        f=adapter/'candidate_matches'/f'{sid}.json';save(f,payload)
        previous=old.get(sid,{})
        t=dict(r,image0=r['reference_native_png'],image1=r['moving_native_png'],width0=512,height0=512,width1=512,height1=512,visual_scene=previous.get('visual_scene','unreviewed_site_label:'+r['scene_class']),annotation_readiness='diagnostic_review_only',visual_note=f"诊断抽查，非随机无偏测试；A={r['has_A_georeference']} B={r['has_B_pseudo']} {r['pseudo_quality']}。默认独立模式，不显示自动点。",demo_priority=i+1,candidate_count=len(geo)+len(pseudo),candidate_sha256=sha(f),selection_reason='first five previously frozen representatives + three additional sites prioritizing B/failed diagnostic breadth')
        tasks.append(t)
    manifest=adapter/'annotation_tasks/tasks.jsonl';manifest.write_text(''.join(json.dumps(t,ensure_ascii=False)+'\n' for t in tasks));save(adapter/'annotation_tasks/task_freeze.json',{'task_sha256':sha(manifest),'diagnostic_not_unbiased_test':True,'tasks':len(tasks),'real_acquisition_pairs':len(used_pairs),'sites':len(used_sites),'C_level':0})
    save(adapter/'diagnostic_tasks.json',[{k:r[k] for k in ['sample_id','sensor','site_name','look_pair_category','pseudo_quality','has_B_pseudo','selection_reason']} for r in tasks]);print([t['sample_id'] for t in tasks])

if __name__=='__main__':
    import argparse;p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT));a=p.parse_args();main(a.out)
