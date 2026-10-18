"""Link reviewed native assets into the existing single-expert tool, without 512 cuts."""
import sys,json,hashlib,copy
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from affine import Affine
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
TOOL=ROOT.parent/'RealSAR_SingleExpert_20261015_2332'
sys.path.insert(0,str(TOOL/'code'))
from expert import Dataset,Store,save,jsonl,stamp,now

def main():
 t=stamp();relations=[json.loads(s) for s in sorted(ROOT.glob('RECOVERED_NATIVE_RELATIONS_*.jsonl'))[-1].read_text().splitlines()]
 assert len(relations)==3 and all(len(r['assets'])==2 for r in relations)
 old=Dataset();before=Store(old);old_evidence={scope:d['content_sha256'] for scope,d in before.docs.items()}
 frames={};tasks=[];listing=[]
 for r in sorted(relations,key=lambda r:r['site']!='Paris road intersections'):
  recommend=r['site'] in ['Paris road intersections','Thermal Club']
  structures='道路交汇、低矮地面围界；排除楼顶、建筑强散射及河面船舶' if r['site']=='Paris road intersections' else '赛道内外缘转角及地面路口；排除行驶车辆、树木及设施阴影' if r['site']=='Thermal Club' else '当前窗口复杂设施/重复纹理较多，暂缓优先人工标点'
  fids=[]
  for asset in r['assets']:
   z=asset['native_info'];cid=asset['acquisition_id'];origin=asset['product_origin_xy'];path=str(Path(asset['path']).resolve());parent=tuple(Affine(*z['transform'])*Affine.translation(-origin[0],-origin[1]))[:6]
   fid='F'+hashlib.sha256((cid+path+repr(origin)).encode()).hexdigest()[:18]
   grid='G'+hashlib.sha256((cid+z['crs']+repr(parent)).encode()).hexdigest()[:16]
   frames[fid]=dict(frame_id=fid,acquisition_id=cid,grid_id=grid,raw_path=path,display=asset['display'],valid_mask=None,width=z['width'],height=z['height'],origin_xy=origin,native_info=z,units=asset['acquisition'].get('input_units','linear_amplitude'),large_context=True,metadata=asset['acquisition'].get('metadata_path'),raw_read='native Window only; no reproject')
   fids.append(fid)
  entry=dict(r,priority_recommended=recommend,physical_structure_search=structures,physical_correspondence_state='no actual human point for this new pair; AI regional readability review only',GT_qualified=False,crop_state='native windows only; no 512 crops generated')
  listing.append(entry)
  if not recommend:continue
  sid='CLEAR_'+r['relation_id'];thumb=ROOT/'previews'/f'{sid}_NAVIGATION_{t}.jpg';sheet=Image.new('L',(386,192),0)
  for side,fid in enumerate(fids):
   with Image.open(frames[fid]['display']) as im:
    im=im.convert('L');im.thumbnail((192,192),Image.Resampling.NEAREST);sheet.paste(im,(side*194,0))
  sheet.save(thumb,quality=90)
  tasks.append(dict(sample_id=sid,relation_id=r['relation_id'],site='巴黎城区道路' if r['site']=='Paris road intersections' else 'Thermal Club 赛道与道路',component=r['source_component'],split='research_unassigned',angle=r['azimuth_delta_deg'],angle_bin='0-15' if r['azimuth_delta_deg']<15 else '30-60',scene=r['scene'],AI_class='E',AI_hint=structures+'；只是候选，不是GT',AI_reviewed=True,frame_ids=fids,native_frame_ids=fids,native_context_available=True,native_only=True,clear_candidate=True,clear_priority=len(tasks),thumbnail=str(thumb),source_manifest=str(ROOT),acquisition_ids=r['acquisition_ids'],historical_exposure=True,priority=0,pilot30=False))
 extra=dict(at=now(),tasks=tasks,frames={fid:frames[fid] for task in tasks for fid in task['frame_ids']},existing_clear_controls=['N512_4c4a55a9c29275'],NOT_512_dataset=True,GT_created=0)
 extra_path=ROOT/f'NATIVE_SHORTLIST_INDEX_{t}.json';save(extra_path,extra)
 config=dict(old.config,extra_native_index=str(extra_path))
 new=Dataset(config);after=Store(new)
 assert {scope:d['content_sha256'] for scope,d in after.docs.items()}==old_evidence
 for sid,task in old.tasks.items():
  combined={k:v for k,v in new.tasks[sid].items() if k not in ['clear_candidate','same_direction_control']}
  assert combined==task, sid
 assert sum(not t.get('native_only') for t in new.tasks.values())==1033
 assert len(tasks)==2 and len(new.tasks)==1035
 for task in tasks:
  q=after.assessment(task)
  assert q['expert_gt_level']=='pending' and not q['primary']['passed'] and not q['local']['passed']
  assert not q['single_expert_geometry_qualified'] and not q['GT_qualified']
  for fid in new.choose_frames(task['sample_id'],'native'):
   frame=new.frames[fid];valid,_=new.pixel_support(fid,frame['width']/2+.25,frame['height']/2+.75);assert valid
 config_path=TOOL/f'CONFIG_{t}.json';save(config_path,config)
 jsonl(ROOT/f'CLEAR_CANDIDATE_MANIFEST_{t}.jsonl',listing)
 save(ROOT/f'SHORTLIST_PREPARATION_{t}.json',dict(at=now(),new_real_acquisition_relations_downloaded=3,new_native_image_windows=6,new_priority_cross_angle_relations=2,new_priority_components=2,existing_same_direction_controls=1,new_512_crops=0,original_512_pairs_unchanged=1033,prior_actual_point_count=sum(len(d['points']) for d in before.docs.values() if d['mode']!='repeat'),prior_document_hashes=old_evidence,old_task_fields_unchanged=True,new_source_split='research_unassigned',new_native_geometry_qualification=False,GT_created=0,tool_config=str(config_path),extra_index=str(extra_path),priority_task_ids=[t['sample_id'] for t in tasks]))
 print('SHORTLIST_READY',len(tasks),'new native pairs + 1 existing same-direction control;',config_path,flush=True)

if __name__=='__main__':main()
