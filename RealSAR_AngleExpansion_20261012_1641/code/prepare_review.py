"""Native-only readability queue plus inactive independent-GT staging."""
import sys
sys.dont_write_bytecode=True
import os,json,re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;V2=BASE/'RealSAR_MultiViewV2_20261012_1259'
sys.path.insert(0,str(V2/'code'))
from geo_tools import lines,save,jsonl,sha
sys.path.insert(0,str(BASE/'RealSAR_IndependentBenchmark_20261011_2015'))
from benchmark_quality import POLICY

def prepare():
    target=ROOT/'review'
    if (target/'candidate_regions.jsonl').exists():raise FileExistsError('Review batch exists')
    rows=[];staging=[]
    for source,old in [(ROOT,False),(V2,True)]:
        index=source/('candidate_regions_complete.jsonl' if not old else 'candidate_regions.jsonl')
        for r in lines(index):
            r=json.loads(json.dumps(r));sid=r['sample_id'];folder=target/'regions'/sid;folder.mkdir(parents=True,exist_ok=False)
            for side in [0,1]:
                for kind in ['db','linear']:
                    src=source/'regions'/sid/f'native_{kind}{side}.png';os.link(src,folder/src.name)
                r['views'][side].setdefault('native_tif',str(source/'regions'/sid/f'native{side}.tif'))
                v=r['acquisition_metadata']['views'][side]
                v.setdefault('utc',v.get('datetime'))
            if not old:r['metric_crs']=r['acquisition_metadata']['metric_crs']
            r.update(review_queue_provenance='reused_V2_recommended_not_new_sample' if old else 'new_pair_native_windows',split='diagnostic_development_seen',
                human_readability='pending',gt_qualified=False,AI_screening_not_human_review=True)
            r['source_region_metadata']=str(source/'regions'/sid/'metadata.json')
            save(folder/'metadata.json',r);rows.append(r)
            task=dict(sample_id=sid,pair_id=r['pair_id'],site_name=r['site_name'],component=r['component'],split=r['split'],sensor=r['sensor'],
                candidate_count=0,annotation_readiness='INACTIVE_PENDING_REAL_HUMAN_A_B',automatic_points=0,
                prerequisite='real human A/B, then native clicks and second independent human re-click; no simulated actors',
                reference_acquisition_id=r['views'][0]['acquisition_id'],moving_acquisition_id=r['views'][1]['acquisition_id'])
            for side in [0,1]:
                z=r['views'][side]['native_info'];task.update({f'width{side}':z['width'],f'height{side}':z['height'],
                    f'image{side}':str(folder/f'native_db{side}.png'),f'image_sha256_{side}':sha(folder/f'native_db{side}.png')})
            staging.append(task)
    jsonl(target/'candidate_regions.jsonl',rows);jsonl(ROOT/'gt_staging_manifest.jsonl',staging)
    save(ROOT/'gt_policy_snapshot.json',dict(policy=POLICY,GT=0,points=0,active_GT_tasks=0,
        prerequisite='Human readability is not GT; independent check qualification is unchanged',
        source=str(BASE/'RealSAR_GeometryGT_20261011_0220/gt_quality/quality.py'),
        source_sha256=sha(BASE/'RealSAR_GeometryGT_20261011_0220/gt_quality/quality.py')))
    html=(V2/'quick_review.html').read_text()
    html=re.sub(r'<option value="(?:context|common|detail)">.*?</option>','',html)
    html=html.replace('8对候选高清图集','当前候选图集').replace('V2母区域快速判读','真实SAR角度补充 · 原生可辨识性判读')
    html=html.replace('<button id="showMap">','<button id="showMap" hidden>')
    html=html.replace('...v,profile:undefined,local_contexts:undefined,',
        'acquisition_id:v.acquisition_id,utc:v.utc,platform:v.platform,azimuth_degrees:v.azimuth_degrees,incidence_degrees:v.incidence_degrees,orbit_direction:v.orbit_direction,observation_direction:v.observation_direction,polarizations:v.polarizations,')
    html=html.replace('FOV_m:current.ground_edge_m,','requested_FOV_m:current.requested_ground_edge_m,original_V2_FOV_m:current.ground_edge_m,')
    html=html.replace('RealSAR-MultiView V2 · 快速判读','RealSAR角度补充 · 仅Native真人判读')
    with (target/'quick_review.html').open('x') as f:f.write(html)
    os.link(ROOT/'candidate_atlas_complete.png',target/'candidate_atlas.png')
    save(target/'task_freeze.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        task_sha256=sha(target/'candidate_regions.jsonl'),tasks=len(rows),new_pairs=sum(not r['review_queue_provenance'].startswith('reused') for r in rows),
        reused_V2_controls=sum(r['review_queue_provenance'].startswith('reused') for r in rows),
        active_GT_tasks=0,unqualified_GT=0,native_only=True,manual_reviewed=0))
    print('READABILITY_TASKS',len(rows),'GT=0; staging inactive')

if __name__=='__main__':prepare()
