"""Preserve append-only assisted reviews and separate unconfirmed navigation logs.
No GT conversion, inferred point pairing, image copies, or production record edits.
"""
import csv,hashlib,json,math,re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlparse,parse_qs
from collections import Counter
from records import digest
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def whole_lines(data):return data[:data.rfind(b'\n')+1] if data else data
def jsonl(path,rows):
    with path.open('x',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def table(path,fields,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
def export():
    at=datetime.now(ZoneInfo('Asia/Shanghai'));files={};task_bytes=(ROOT/'tasks.jsonl').read_bytes()
    tasks={d['sample_id']:d for d in map(json.loads,task_bytes.decode().splitlines()) if d}
    for path in sorted((ROOT/'assisted_triage').rglob('v*.json')):files[path]=path.read_bytes()
    ep=ROOT/'assisted_triage/exposure_events.jsonl';exposure_data=whole_lines(ep.read_bytes()) if ep.exists() else b''
    exposures=[json.loads(line) for line in exposure_data.decode().splitlines()]
    histories={};validated_regions=[];validated_clicks=[]
    for path,data in files.items():histories.setdefault(path.parent.name,[]).append((path,json.loads(data)))
    for sid,history in histories.items():
        assert sid in tasks,'unknown sample in stored review';parent=None
        for number,(path,doc) in enumerate(history,1):
            payload=dict(doc);content=payload.pop('content_sha256')
            assert digest(payload)==content and doc['parent_sha256']==parent,'broken review hash chain'
            assert doc['revision']==number and path.name==f'v{number:06d}.json'
            assert doc['sample_id']==sid and doc['pair_id']==tasks[sid]['pair_id']
            assert doc['stage']=='assisted_triage' and doc['independent_GT'] is False and doc['GT_qualified'] is False and doc['projected_points_copied_to_GT'] is False
            assert doc['verdict'] in ['A','B','C','D'] and doc['reason'].strip()
            assert doc.get('manual_clicks') or doc.get('approx_regions') or doc.get('location_description','').strip()
            assert any(e['sample_id']==sid and e['actor']==doc['actor'] for e in exposures),'review has no preserved exposure event'
            for index,region in enumerate(doc.get('approx_regions',[]),1):
                side=region['side'];x,y,w,h=region['bbox_xywh'];v=tasks[sid]['views'][side]
                assert side in [0,1] and all(math.isfinite(z) for z in [x,y,w,h]) and min(x,y)>=0 and min(w,h)>0 and x+w<=v['width'] and y+h<=v['height']
                validated_regions.append(dict(sample_id=sid,pair_id=doc['pair_id'],revision=number,region_index=index,side=side,acquisition_id=v['acquisition_id'],native_tif=v['native_tif'],x=x,y=y,width=w,height=h,exact_correspondence=False,GT=False))
            for index,click in enumerate(doc.get('manual_clicks',[]),1):
                side=click['side'];x,y=click['native_xy'];v=tasks[sid]['views'][side]
                assert side in [0,1] and all(math.isfinite(z) for z in [x,y]) and 0<=x<=v['width']-1 and 0<=y<=v['height']-1
                assert click['input_event'] in ['overview_click','detail_click'] and click['exact_correspondence'] is False
                validated_clicks.append(dict(sample_id=sid,pair_id=doc['pair_id'],revision=number,click_index=index,side=side,acquisition_id=v['acquisition_id'],native_tif=v['native_tif'],x=x,y=y,input_event=click['input_event'],exact_correspondence=False,GT=False))
            parent=content
    out=ROOT/'exports'/('assisted_review_snapshot_'+at.strftime('%Y%m%d_%H%M%S'));out.mkdir(parents=True,exist_ok=False)
    source_manifest=[]
    def preserve(source,data,target,prefix=False,private=False):
        target=out/target;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(data)
        if private:target.chmod(0o600)
        assert sha(target.read_bytes())==sha(data)
        source_manifest.append(dict(source=str(source),snapshot_file=str(target.relative_to(out)),sha256=sha(data),bytes=len(data),complete_line_prefix=prefix))
    for path,data in files.items():preserve(path,data,Path('raw')/path.relative_to(ROOT))
    preserve(ep,exposure_data,'raw/assisted_triage/exposure_events.jsonl',prefix=True)
    preserve(ROOT/'tasks.jsonl',task_bytes,'raw/tasks.jsonl')
    queue=ROOT.parent/'RealSAR_AngleExpansion_20261012_1641/review/candidate_regions.jsonl'
    preserve(queue,queue.read_bytes(),'raw/candidate_regions.jsonl')
    # Navigation requests lack session/actor and click-versus-pan provenance.
    # Preserve exact requested coordinates, never infer left/right correspondences.
    log_paths=[]
    for record in sorted((ROOT/'maintenance').rglob('service_*.json')):
        launch=json.loads(record.read_text());path=Path(launch['log'])
        if path.is_file() and path not in log_paths:log_paths.append(path)
    navigation=[];rejected=[]
    for log in log_paths:
        data=whole_lines(log.read_bytes());rel=Path('raw/access_logs')/log.name
        preserve(log,data,rel,prefix=True,private=True)
        for number,line in enumerate(data.decode(errors='replace').splitlines(),1):
            match=re.search(r'\[([^]]+)\] "GET (\S+) HTTP/[^\"]+" (\d+)',line)
            if not match:continue
            stamp,url,status=match.groups();parsed=urlparse(url)
            if parsed.path not in ['/api/map','/api/crop'] or status!='200':continue
            q=parse_qs(parsed.query)
            if not all(k in q for k in ['sid','side','x','y']):continue
            try:
                sid=q['sid'][0];side=int(q['side'][0]);x=float(q['x'][0]);y=float(q['y'][0]);v=tasks[sid]['views'][side]
                if side not in [0,1] or not all(math.isfinite(z) for z in [x,y]):raise ValueError('invalid requested coordinate')
                nav=dict(sample_id=sid,pair_id=tasks[sid]['pair_id'],site_name=tasks[sid]['site_name'],side=side,acquisition_id=v['acquisition_id'],native_tif=v['native_tif'],native_width=v['width'],native_height=v['height'],x=x,y=y,
                    span=q.get('span',[''])[0],request_type='map_source_request' if parsed.path=='/api/map' else 'viewport_center_request',
                    log_timestamp=stamp,log_timezone='not present in access log',log_file=str(rel),log_line=number,
                    in_native_center_extent=0<=x<=v['width']-1 and 0<=y<=v['height']-1,
                    operator='not recorded in HTTP log',confirmed_human_point=False,exact_correspondence=False,GT=False)
                navigation.append(nav)
            except (ValueError,IndexError,KeyError) as error:rejected.append(dict(log=str(rel),line=number,error=str(error)))
    latest=[history[-1][1] for history in histories.values()];latest_by_sid={d['sample_id']:d for d in latest}
    rows=[]
    for sid,t in tasks.items():
        doc=latest_by_sid.get(sid,{})
        rows.append(dict(sample_id=sid,pair_id=t['pair_id'],site_name=t['site_name'],component=t['component'],azimuth_delta_deg=t.get('azimuth_delta_deg'),
            reviewed=bool(doc),revision=doc.get('revision',0),operator=doc.get('actor',''),at=doc.get('at',''),verdict=doc.get('verdict',''),reason=doc.get('reason',''),location_description=doc.get('location_description',''),
            saved_region_count=len(doc.get('approx_regions',[])),saved_manual_click_count=len(doc.get('manual_clicks',[])),recommend_independent_annotation=doc.get('recommend_independent_annotation',False),
            left_acquisition=t['views'][0]['acquisition_id'],right_acquisition=t['views'][1]['acquisition_id'],left_native_tif=t['views'][0]['native_tif'],right_native_tif=t['views'][1]['native_tif'],GT=False))
    table(out/'latest_reviews.csv',list(rows[0]),rows);jsonl(out/'latest_assisted_triage.jsonl',latest)
    region_fields=['sample_id','pair_id','revision','region_index','side','acquisition_id','native_tif','x','y','width','height','exact_correspondence','GT']
    table(out/'saved_native_regions_all_versions.csv',region_fields,validated_regions)
    click_fields=['sample_id','pair_id','revision','click_index','side','acquisition_id','native_tif','x','y','input_event','exact_correspondence','GT']
    table(out/'saved_manual_observation_clicks_all_versions.csv',click_fields,validated_clicks)
    fields=list(navigation[0]) if navigation else ['sample_id','side','x','y','request_type','confirmed_human_point','GT']
    table(out/'navigation_request_coordinates.csv',fields,navigation)
    last_views={}
    for row in navigation:
        if row['request_type']=='viewport_center_request':last_views[(row['sample_id'],row['side'])]=row
    table(out/'last_view_positions.csv',fields,list(last_views.values()))
    for path,data in files.items():assert path.read_bytes()==data,'an existing version changed during backup'
    summary=dict(snapshot_at=at.isoformat(),snapshot_directory=str(out),total_queue=len(tasks),reviewed_samples=len(latest),pending_samples=len(tasks)-len(latest),history_versions=len(files),latest_verdict_counts=dict(Counter(d['verdict'] for d in latest)),
        exposure_events=len(exposures),saved_native_regions_all_versions=len(validated_regions),saved_manual_observation_clicks_all_versions=len(validated_clicks),saved_exact_correspondence_points=0,
        map_source_requests=sum(d['request_type']=='map_source_request' for d in navigation),viewport_requests=sum(d['request_type']=='viewport_center_request' for d in navigation),last_view_positions=len(last_views),
        unconfirmed_navigation_only=True,independent_GT=0,all_version_hash_chains_passed=True,all_saved_coordinates_bounds_passed=True,original_versions_unchanged=True,access_log_parse_errors=rejected,
        scope='assisted triage backup; no GT qualification or pairing; no acquisition/download/image copies',source_files=source_manifest)
    (out/'snapshot_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    text=f'''# 人工辅助筛选备份\n\n时间：{summary['snapshot_at']}。13之类的计数以本快照清单为准。\n\n已保存 {len(latest)}/{len(tasks)} 个样本，{len(files)} 个历史版本；最新判断：{summary['latest_verdict_counts']}。\n\n- `raw/assisted_triage/`：原始版本逐字节备份及暴露事件快照，原哈希链全部核验。\n- `latest_reviews.csv`：18项入口及最新操作者、时间、A/B/C/D、理由、位置文字和来源身份。\n- `latest_assisted_triage.jsonl`：每个已保存样本的原始最新版本。\n- `saved_native_regions_all_versions.csv`：版本中明确保存的Native粗略窗口，共 {len(validated_regions)} 个；零行表示没有保存窗口，不是导出失败。\n- `navigation_request_coordinates.csv`：服务日志中地图请求及放大视图请求的原始Native坐标。\n- `last_view_positions.csv`：各样本各侧最后一个放大视图请求中心，仅用于回查。\n- `snapshot_manifest.json`：源文件SHA256、计数、版本与坐标核验。\n\n## 坐标含义\n\n坐标只指各侧Native影像：side=0为左图，side=1为右图；x为列，y为行，像素中心零基坐标。保存窗口的bbox为[x,y,width,height]，x/y是窗口像素边界。\n\n地图请求坐标可能来自点击或方向键；放大视图中心可能是用户点击，也可能是地理预测。HTTP日志没有操作者/session或动作类型，不能认证为人工确认点；不同侧记录不得自动配成物理对应点。日志原始时间没有时区，已原样保留，不强行绑定某个保存版本。原始访问日志可能含只读测试令牌，仅作0600权限的备份。\n\n本快照保留了浏览位置，但不向正式版本补造区域或精确点，不生成GT，不改变A/B/C/D判断或独立标注建议。若版本没有窗口坐标，当前工具不能把过去浏览日志当成已确认的黄色历史区域；今后请点击“加入当前查看区域”再保存。\n'''
    text=text.replace('13之类的计数以本快照清单为准。','计数以本快照清单为准。')
    text=text.replace('- `navigation_request_coordinates.csv`',f'- `saved_manual_observation_clicks_all_versions.csv`：版本中保存的直接人工观察点击，共 {len(validated_clicks)} 个；各侧分别记录，不是GT或物理对应点。\n- `navigation_request_coordinates.csv`')
    (out/'README.md').write_text(text)
    print(json.dumps({k:v for k,v in summary.items() if k!='source_files'},ensure_ascii=False,indent=2))
    return out
if __name__=='__main__':export()
