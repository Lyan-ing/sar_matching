"""Select representatives before matching; read immutable release, write here only."""
import hashlib,json,random,csv,sys
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from PIL import Image,ImageDraw
import rasterio

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'real_sar_pilot/multisource512_20261010_2352'
SEED=261011
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def rank(s):return hashlib.sha256(f'{SEED}:{s}'.encode()).hexdigest()
def main():
    out=ROOT/'selection'; assert not (out/'samples.jsonl').exists()
    receipt=json.loads((SOURCE/'DELIVERY_COMPLETE.json').read_text())
    assert receipt['engineering_accepted'] and receipt['quality']['candidate_crop_pairs']==191
    rows=[json.loads(s) for s in (SOURCE/'dataset_index.jsonl').read_text().splitlines()];assert len(rows)==191
    validation=[]
    for r in rows:
        for k in ['reference_native_png','moving_native_png','reference_native_geotiff','moving_native_geotiff']:
            p=Path(r[k]); assert p.is_file(); assert sha(p)==receipt['file_sha256'][str(p.relative_to(SOURCE))]
        for side in ['reference','moving']:
            with Image.open(r[f'{side}_native_png']) as im:assert im.size==(512,512)
            with rasterio.open(r[f'{side}_native_geotiff']) as src:assert src.width==src.height==512 and src.crs
        validation.append(r['pair_id']+'/'+r['chip_id'])
    write(out/'input_acceptance.json',{'records':len(rows),'native_files_verified_against_receipt':len(rows)*4,'engineering_QA_accepted':True,'independent_GT_pairs':0,'dataset_index_sha256':sha(SOURCE/'dataset_index.jsonl'),'receipt_sha256':sha(SOURCE/'DELIVERY_COMPLETE.json')})
    sites=defaultdict(lambda:defaultdict(list))
    for r in rows:sites[r['site_name']][r['pair_id']].append(r)
    rules={'seed':SEED,'before_candidate_generation':True,'selection_uses_matcher_outputs':False,'quota':'2 distinct acquisition pairs/site; SF and South Dakota only 1 available; +1 Johnston and +1 Long Beach = 24','pair_rule':'greedy categorical diversity in look category and geographic overlap; seeded SHA256 tie break; never registration success','crop_rule':'seeded SHA256 ordering, one native chip per real acquisition pair','scene_labels':'site-level labels retained; crop visual inspection recorded separately','confirmation_is_unbiased_sealed_test':False,'pixel_coordinates':'integer pixel centers (0,0) top left; float subpixels; valid center domain 0<=x<=W-1, 0<=y<=H-1; map conversion uses center +0.5','source':str(SOURCE),'source_index_sha256':sha(SOURCE/'dataset_index.jsonl')}
    write(out/'rules.json',rules)
    selected=[]
    for site,groups in sorted(sites.items()):
        quota=3 if site in ['Johnston, Iowa Farm','Port of Long Beach CA'] else 2
        # Actual catalog names differ in punctuation; identify by substring.
        if 'Johnston' in site or 'LongBeach' in site or 'Long Beach' in site:quota=3
        chosen=[];looks=set();overlaps=set()
        while len(chosen)<min(quota,len(groups)):
            candidates=[(pid,rs) for pid,rs in groups.items() if pid not in chosen]
            pid,rs=min(candidates,key=lambda t:(-(int(t[1][0]['look_pair_category'] not in looks)+int(any(r['geographic_overlap_stratum'] not in overlaps for r in t[1]))),rank(t[0])))
            r=min(rs,key=lambda r:(-int(r['geographic_overlap_stratum'] not in overlaps),rank(pid+'/'+r['chip_id'])))
            chosen.append(pid);looks.add(r['look_pair_category']);overlaps.add(r['geographic_overlap_stratum'])
            s=dict(r,sample_id=pid+'__'+r['chip_id'],original_index_id=pid+'/'+r['chip_id'],image0=r['reference_native_png'],image1=r['moving_native_png'],width0=512,height0=512,width1=512,height1=512,selection_reason='site quota + look/overlap diversity + seeded tie break, no matching outcomes',visual_scene='pending_visual_inspection',annotation_readiness='pending_visual_inspection',gt_status='unverified')
            for angle in ['azimuth','incidence']:
                a,b=r.get('reference_'+angle+'_degrees'),r.get('moving_'+angle+'_degrees'); delta=None if a is None or b is None else abs(a-b);s[angle+'_difference_degrees']=min(delta,360-delta) if angle=='azimuth' and delta is not None else delta
            # All provider strings are UTC; second precision is enough for days;
            # preserve complete nanosecond strings above rather than alter identity.
            a=datetime.fromisoformat(r['reference_utc'][:19]);b=datetime.fromisoformat(r['moving_utc'][:19]);s['time_interval_days']=abs((a-b).total_seconds())/86400
            selected.append(s)
    assert len(selected)==24,(len(selected),Counter(r['site_name'] for r in selected))
    (out/'samples.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected))
    fields=['sample_id','original_index_id','sensor','site_name','scene_class','look_pair_category','azimuth_difference_degrees','incidence_difference_degrees','time_interval_days','geographic_overlap_stratum','image0','image1','reference_acquisition_id','moving_acquisition_id','component','split','selection_reason','gt_status']
    with (out/'samples.csv').open('w',newline='') as f:w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(selected)
    write(out/'selection_freeze.json',{'selected_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'sha256':sha(out/'samples.jsonl'),'rules_sha256':sha(out/'rules.json'),'samples':24,'distinct_real_acquisition_pairs':24,'sites':len(sites),'providers':dict(Counter(s['sensor'] for s in selected)),'inference_not_started':True})
    sheet=Image.new('RGB',(1584,8*310+60),'white');d=ImageDraw.Draw(sheet)
    d.text((5,4),'24 native SAR pairs, 512x512 originals downsized to 256x256 ONLY for overview. NO independent GT.',fill='black')
    d.text((5,20),'Umbra / Copyright Capella Space CC BY 4.0; Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026',fill='black')
    for i,r in enumerate(selected):
        x=(i%3)*528;y=(i//3)*310+60
        d.text((x+3,y),f"{i+1:02d} {r['sample_id']} | {r['sensor']}",fill='black');d.text((x+3,y+15),r['site_name'][:55]+' | '+r['geographic_overlap_stratum'],fill='black')
        for side,k in enumerate(['image0','image1']):
            im=Image.open(r[k]).convert('RGB');im.thumbnail((256,256));sheet.paste(im,(x+side*260,y+35))
    sheet.save(ROOT/'visualization/selection_overview.png')
    for start in [0,12]:sheet.crop((0,60+(start//3)*310,1584,60+((start+12)//3)*310)).save(ROOT/f'visualization/overview_{start+1:02d}_{start+12:02d}.png')
    write(ROOT/'reports/initial_check.json',{'python':sys.executable,'protected_release':str(SOURCE),'index_valid':True,'sample_count':24,'selected_acquisition_pairs':24,'frozen_index_sha256':sha(out/'samples.jsonl')})
    print(ROOT);print(Counter(s['site_name'] for s in selected))
if __name__=='__main__':main()
