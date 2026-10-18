"""Finalize references after explicit recovery; retain the first batch artifacts."""
import sys
sys.dont_write_bytecode=True
import os,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image,ImageDraw
from build_windows import ROOT,SELECTED,SIDE_M
from geo_tools import lines,save,jsonl,sha

def run():
    if (ROOT/'candidate_regions_complete.jsonl').exists():raise FileExistsError('Consolidated manifest exists')
    pairs={r['pair_id']:r for r in lines(ROOT/'pair_candidates.jsonl')}
    initial={r['pair_id']:r for r in lines(ROOT/'regions.jsonl')};rows=[];unavailable=[]
    for pid in SELECTED:
        p=pairs[pid]
        missing=[v['acquisition_id'] for v in p['views'] if not (ROOT/'assets'/v['acquisition_id']/'metadata.json').exists()]
        if missing:
            unavailable.append(dict(pair_id=pid,missing_acquisitions=missing,not_replaced_by_easier_pair=True));continue
        if pid in initial:
            rows.append(dict(initial[pid],requested_ground_edge_m=SIDE_M,ground_edge_m=None));continue
        sid=pid+'__native_000';folder=ROOT/'regions'/sid;folder.mkdir(parents=True,exist_ok=False);views=[]
        for i,v in enumerate(p['views']):
            asset=ROOT/'assets'/v['acquisition_id'];m=json.loads((asset/'metadata.json').read_text())
            for kind in ['db','linear']:os.link(asset/f'native_{kind}.png',folder/f'native_{kind}{i}.png')
            views.append(dict(side=i,acquisition_id=v['acquisition_id'],native_info=m['native_info'],metric_geometry=m['native_spacing'],
                actual_context_smaller_than_requested=m.get('actual_context_smaller_than_requested',False),
                original_native_window_xywh=m['original_window_xywh'],native_tif=m['native_tif'],native_sha256=m['raw_sha256'],window_source=m['source_method']))
        r=dict(sample_id=sid,pair_id=pid,legacy_pair_id=p['legacy_pair_id'],sensor=p['sensor'],site_name=p['site_name'],component=p['component'],
            split='diagnostic_development_seen',look_pair_category='metadata_azimuth_'+p['angle_bin'],angle_bin=p['angle_bin'],days=p['days'],
            azimuth_delta_deg=p['azimuth_delta_deg'],incidence_delta_deg=p['incidence_delta_deg'],
            acquisition_metadata=dict(p,views=[dict(v,utc=v['datetime']) for v in p['views']]),views=views,metric_crs=p['metric_crs'],
            requested_ground_edge_m=SIDE_M,ground_edge_m=None,native_geographic_overlap='geographic screening only, not scatterer GT',human_readability='pending',
            independent_GT=False,gt_qualified=False,recovered_after_disclosed_empty_window_interrupt=True,
            risks=['common ground footprint is not independent scatterer GT','shared site/task/acquisition dependency retained',
                   'roofs, elevated tanks, cranes, shadows and ships cannot establish ground-level GT'])
        save(folder/'metadata.json',r);rows.append(r)
    jsonl(ROOT/'candidate_regions_complete.jsonl',rows)
    assets={v['acquisition_id'] for r in rows for v in r['views']};sites={r['site_name']:r['component'] for r in rows}
    save(ROOT/'delivery_summary.json',dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),new_pairs=len(rows),unique_new_batch_acquisitions=len(assets),
        new_acquisition_identities_vs_old_inventory=len(assets-{v['acquisition_id'] for r in lines(ROOT.parent/'RealSAR_MultiViewV2_20261012_1259/acquisition_pair_inventory.jsonl') for v in r['views']}),
        sites=sites,bins={b:sum(r['angle_bin']==b for r in rows) for b in ['0-20','20-40','40-60','60-90','>90']},
        selected_pairs=len(SELECTED),unavailable_pairs=unavailable,GT=0,human_readability_passed=0,real_accuracy_available=False,final_unseen_test=0,
        data_bytes=sum((ROOT/'assets'/a/'native.tif').stat().st_size for a in assets),
        initial_batch=str(ROOT/'regions.jsonl'),complete_batch=str(ROOT/'candidate_regions_complete.jsonl'),
        sole_recovered_empty_acquisition='54e07361-06f6-4420-b20b-dcb315f8d7b9',old_V2_preserved=True))
    atlas=Image.new('RGB',(2260,650*len(rows)),'#18202a');draw=ImageDraw.Draw(atlas)
    for j,r in enumerate(rows):
        draw.text((15,j*650+8),f"{r['pair_id']} | {r['site_name']} | {r['azimuth_delta_deg']:.2f} deg | {r['days']:.5f} d | human review pending, GT=0",fill='white')
        for i,v in enumerate(r['views']):
            im=Image.open(ROOT/'assets'/v['acquisition_id']/'overview.png');im.thumbnail((1100,600),Image.Resampling.NEAREST);atlas.paste(im,(15+i*1120,j*650+40))
    atlas.save(ROOT/'candidate_atlas_complete.png')
    print('CONSOLIDATED',len(rows),'pairs',len(assets),'acquisitions')

if __name__=='__main__':run()
