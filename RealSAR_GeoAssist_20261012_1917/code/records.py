"""Append-only assisted triage and irreversible assistance exposure, separate from GT."""
import json,os,secrets,threading,hashlib
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

class Records:
    def __init__(self,root,tasks):self.root=Path(root);self.tasks=tasks;self.lock=threading.RLock()
    def actor(self,p):
        actor=p.get('actor')
        if not isinstance(actor,str) or not actor.strip() or len(actor)>100:raise ValueError('请输入真实操作者身份')
        return actor.strip()
    def exposures(self):
        p=self.root/'exposure_events.jsonl'
        return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []
    def expose(self,sid,actor):
        with self.lock:
            if sid not in self.tasks:raise ValueError('unknown task')
            row=self.tasks[sid]
            event=dict(sample_id=sid,pair_id=row['pair_id'],actor=self.actor(dict(actor=actor)),at=now(),
                source='geographic_point_projection_and_coverage',stage='assisted_triage',
                viewed_geographic_hint=True,blind_eligible_for_this_sample=False,independent_GT=False)
            self.root.mkdir(parents=True,exist_ok=True)
            with (self.root/'exposure_events.jsonl').open('a') as f:
                f.write(json.dumps(event,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            return event
    def history(self,sid):
        if sid not in self.tasks:raise ValueError('unknown task')
        with self.lock:
            result=[];previous=None
            for p in sorted((self.root/sid).glob('v*.json')):
                doc=json.loads(p.read_text());copy=dict(doc);content=copy.pop('content_sha256')
                if digest(copy)!=content or doc['parent_sha256']!=previous:raise ValueError('assisted triage history changed')
                previous=content;result.append(doc)
            return result
    def load(self,sid):
        records=self.history(sid)
        return records[-1] if records else dict(revision=0,sample_id=sid,verdict=None)
    def commit(self,p,session_actor):
        with self.lock:
            allowed={'sample_id','actor','verdict','reason','location_description','approx_regions','manual_clicks','recommend_independent_annotation','expected_revision'}
            if set(p)-allowed:raise ValueError('assisted triage cannot set GT, matched points or independence')
            actor=self.actor(p);sid=p['sample_id'];task=self.tasks[sid];old=self.load(sid)
            if actor!=session_actor:raise ValueError('操作者改变后请重新登记辅助暴露')
            if old['revision']!=p.get('expected_revision'):raise ValueError('版本已更新，请重新载入')
            if p.get('verdict') not in ['A','B','C','D']:raise ValueError('请明确选择A/B/C/D')
            if not isinstance(p.get('reason'),str) or not p['reason'].strip():raise ValueError('请填写判断理由')
            if not isinstance(p.get('location_description',''),str):raise ValueError('位置描述必须是文字')
            if len(p['reason'])>4000 or len(p.get('location_description',''))>2000:raise ValueError('文字过长')
            if not isinstance(p.get('recommend_independent_annotation'),bool):raise ValueError('请明确标注建议')
            regions=p.get('approx_regions',[])
            if not isinstance(regions,list) or len(regions)>30:raise ValueError('候选区域格式错误')
            clean=[]
            import math
            for region in regions:
                side=region.get('side');bbox=region.get('bbox_xywh')
                if side not in [0,1] or not isinstance(bbox,list) or len(bbox)!=4:raise ValueError('区域格式错误')
                if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in bbox):raise ValueError('区域坐标必须有限')
                x,y,w,h=bbox;v=task['views'][side]
                if min(x,y)<0 or min(w,h)<=0 or x+w>v['width'] or y+h>v['height']:raise ValueError('区域超出Native范围')
                clean.append(dict(side=side,bbox_xywh=bbox,source='human_selected_assisted_search_region',exact_correspondence=False))
            clicks=p.get('manual_clicks',[])
            if not isinstance(clicks,list) or len(clicks)>500:raise ValueError('人工观察点击最多500个，请先保存或清理待保存点击')
            clean_clicks=[]
            for click in clicks:
                if not isinstance(click,dict) or set(click)-{'side','native_xy','input_event'}:raise ValueError('人工点击格式错误；不得加入预测点、配对或GT字段')
                side=click.get('side');xy=click.get('native_xy');event=click.get('input_event')
                if isinstance(side,bool) or side not in [0,1] or not isinstance(xy,list) or len(xy)!=2 or event not in ['overview_click','detail_click']:raise ValueError('只接受图像直接点击的Native观察坐标')
                if any(isinstance(z,bool) or not isinstance(z,(int,float)) or not math.isfinite(z) for z in xy):raise ValueError('点击坐标必须有限')
                v=task['views'][side]
                if not (0<=xy[0]<=v['width']-1 and 0<=xy[1]<=v['height']-1):raise ValueError('点击超出Native像素中心范围')
                clean_clicks.append(dict(side=side,native_xy=xy,input_event=event,source='human_native_click_assisted',exact_correspondence=False,independent_GT=False))
            if not clean and not clean_clicks and not p.get('location_description','').strip():raise ValueError('请点击图像记录观察位置、描述位置或加入查看区域')
            if not any(e['sample_id']==sid and e['actor']==actor for e in self.exposures()):raise ValueError('辅助暴露尚未登记')
            doc=dict(stage='assisted_triage',sample_id=sid,pair_id=task['pair_id'],actor=actor,at=now(),
                verdict=p['verdict'],reason=p['reason'].strip(),location_description=p.get('location_description','').strip(),
                approx_regions=clean,manual_clicks=clean_clicks,recommend_independent_annotation=p['recommend_independent_annotation'],
                viewed_geographic_hint=True,blind_eligible_for_this_sample=False,independent_GT=False,GT_qualified=False,
                projected_points_copied_to_GT=False,revision=old['revision']+1,parent_sha256=old.get('content_sha256'))
            doc['content_sha256']=digest(doc);folder=self.root/sid;folder.mkdir(parents=True,exist_ok=True)
            tmp=folder/('.pending_'+secrets.token_hex(12))
            with tmp.open('x') as f:json.dump(doc,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
            try:os.link(tmp,folder/f"v{doc['revision']:06d}.json")
            finally:tmp.unlink()
            fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
            try:os.fsync(fd)
            finally:os.close(fd)
            return doc
