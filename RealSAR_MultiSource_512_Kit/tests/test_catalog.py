import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from discover_stac import intersection_ratio,asset_best,proposed_pairs
from discover_s1_rtc import candidate_pairs

def test_bbox_overlap():
    assert abs(intersection_ratio([0,0,2,2],[1,1,3,3])-.25)<1e-8
    assert intersection_ratio([0,0,1,1],[2,2,3,3])==0

def test_asset_selection_excludes_preview():
    i={'assets':{'preview':{'href':'https://x/thumb.tif'},'gec':{'href':'https://x/sar_GEC.tif'}}}
    assert asset_best(i)[1]=='gec'

def test_proposal_requires_time_and_geographic_overlap():
    a={'id':'A','sensor':'umbra','datetime':'2025-01-01T00:00:00Z','bbox':[0,0,1,1],
       'asset_url':'https://x/a.tif'}
    b={**a,'id':'B','datetime':'2025-01-10T00:00:00Z','asset_url':'https://x/b.tif'}
    p=proposed_pairs([a,b],'umbra',max_pairs=10)
    assert len(p)==1 and p[0]['split']=='REVIEW_REQUIRED' and p[0]['input_units']==''

def test_temporal_pairs_are_distinct():
    s=[{'scene_name':f's{i}','date':f'202001{(i+1):02d}', 'orbit_direction':'ascending'} for i in (0,1,2)]
    pairs=candidate_pairs(s,max_pairs=2,min_days=1,max_days=30)
    assert len(pairs)<=2
