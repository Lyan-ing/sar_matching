"""Chromium regression check. Synthetic UI fixtures only in a TemporaryDirectory.
No human judgments, GT points or production triage writes are generated.
"""
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
import argparse,hashlib,json,os,threading,tempfile,time
from pathlib import Path
from playwright.sync_api import sync_playwright
from server import make_server
from records import Records,digest
from geometry import ROOT,lines

ACTOR='AUTOMATED_UI_FIXTURE_NOT_A_HUMAN'
def hashes(folder):
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(folder).rglob('*') if p.is_file()}
def run(evidence):
    protected=hashes(ROOT/'assisted_triage')
    tasks=lines(ROOT/'tasks.jsonl');sid=tasks[0]['sample_id'];sid_empty=tasks[1]['sample_id'];errors=[];checks=[]
    with tempfile.TemporaryDirectory(prefix='sar_geo_history_ui_fixture_') as temp:
        store=Path(temp)/'AUTOMATED_FIXTURES_NOT_PRODUCTION';folder=store/sid;folder.mkdir(parents=True)
        parent=None
        for version,boxes in [(1,[[16,24,128,128],[32,48,128,128]]),(2,[[180,220,256,256],[300,320,256,256]])]:
            doc=dict(stage='automated_ui_test_fixture',automated_test_fixture=True,sample_id=sid,pair_id=tasks[0]['pair_id'],
                actor=ACTOR,at=f'2000-01-01T00:00:0{version}+08:00',verdict='D',reason='AUTOMATED UI FIXTURE; no physical structures were judged',
                location_description='synthetic window coordinates for UI regression only',approx_regions=[dict(side=i,bbox_xywh=b,source='automated_ui_fixture',exact_correspondence=False) for i,b in enumerate(boxes)],
                recommend_independent_annotation=False,viewed_geographic_hint=True,blind_eligible_for_this_sample=False,
                independent_GT=False,GT_qualified=False,projected_points_copied_to_GT=False,revision=version,parent_sha256=parent)
            doc['content_sha256']=digest(doc);parent=doc['content_sha256']
            (folder/f'v{version:06d}.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2))
        old_hashes=hashes(folder)
        server=make_server(0,'READONLY_AUTOMATED_UI_FIXTURE',records_root=store)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}/'
        try:
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
                page=browser.new_page(viewport=dict(width=1800,height=1400));page.on('pageerror',lambda e:errors.append(str(e)))
                def enter(sample=sid):
                    page.select_option('#task',sample);page.fill('#actor',ACTOR);page.click('#enter')
                    page.wait_for_function('sid=>ready && !busy && current.sample_id===sid',arg=sample,timeout=30000)
                def wait_history(version):
                    page.wait_for_function('v=>historySelected && historySelected.revision===v && crops.every(c=>c && !c.outside)',arg=version)
                def yellow(canvas):
                    return page.locator(canvas).evaluate("c=>{const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<d.length;i+=4)if(d[i]===255&&d[i+1]===211&&d[i+2]===107&&d[i+3]===255)n++;return n;}")
                page.goto(url);page.wait_for_function('tasks.length===18');enter();wait_history(2)
                assert page.evaluate('revision')==2 and page.locator('#historyList button[data-revision]').count()==2
                assert page.evaluate('crops.map(c=>c.bbox_xywh)')==[[180,220,256,256],[300,320,256,256]]
                assert page.evaluate('last===null && positions.every(p=>p===null)')
                assert all(yellow('#overview'+str(i))>20 and yellow('#detail'+str(i))>20 for i in [0,1])
                checks.append('disk_history_autoload_and_yellow_native_windows')
                page.click('#historyList button[data-revision="1"]');wait_history(1)
                assert page.evaluate('crops.map(c=>c.bbox_xywh)')==[[16,24,128,128],[32,48,128,128]]
                assert page.evaluate('revision')==2 and 'AUTOMATED' in page.locator('#reason').input_value()
                checks.append('old_version_navigation_keeps_latest_edit_revision')
                page.fill('#reason','');page.click('#save');assert '未保存' in page.locator('#saveStatus').inner_text()
                assert len(list(folder.glob('v*.json')))==2
                page.fill('#reason','AUTOMATED UI FAILURE CHECK; no human classification')
                page.click('#clearRegions');page.fill('#location','');page.click('#save')
                assert '未保存' in page.locator('#saveStatus').inner_text() and len(list(folder.glob('v*.json')))==2
                checks.append('missing_reason_and_missing_location_do_not_write')
                # The server also rejects missing required fields and GT/point injection.
                payload=dict(sample_id=sid,actor=ACTOR,verdict='D',reason='',location_description='synthetic fixture',approx_regions=[],recommend_independent_annotation=False,expected_revision=2)
                failure=page.evaluate("async p=>{try{await api('/api/triage',p);return null;}catch(e){return e.message;}}",payload)
                assert '理由' in failure and len(list(folder.glob('v*.json')))==2
                payload['reason']='AUTOMATED reject GT injection';payload['points']=[[1,2]]
                failure=page.evaluate("async p=>{try{await api('/api/triage',p);return null;}catch(e){return e.message;}}",payload)
                assert 'GT' in failure and len(list(folder.glob('v*.json')))==2
                checks.append('server_required_fields_and_GT_injection_rejected')
                # Deliberately simulate an HTTP save failure; no misleading success message.
                page.fill('#reason','AUTOMATED UI NETWORK FAILURE; no human classification');page.fill('#location','synthetic UI location')
                page.route('**/api/triage',lambda route:route.fulfill(status=500,content_type='application/json',body='{"error":"isolated simulated disk failure"}'))
                page.click('#save');page.wait_for_function('document.getElementById("saveStatus").textContent.includes("保存请求失败")')
                assert '写入成功' not in page.locator('#saveStatus').inner_text() and len(list(folder.glob('v*.json')))==2
                page.unroute('**/api/triage');checks.append('failed_write_does_not_show_success')
                # Actual HTTP save is exercised ONLY against the isolated temporary store.
                page.click('#historyList button[data-revision="1"]');wait_history(1);page.click('#addRegion')
                page.fill('#reason','AUTOMATED UI SAVE CHECK; synthetic fixture, NOT human triage');page.fill('#location','')
                page.click('#save');page.wait_for_function('!busy && revision===3 && historyRecords.length===3',timeout=15000)
                assert '实际写入成功' in page.locator('#saveStatus').inner_text() and 'v3' in page.locator('#saveStatus').inner_text() and '3 条' in page.locator('#saveStatus').inner_text()
                record=json.loads((folder/'v000003.json').read_text());assert record['actor']==ACTOR and record['GT_qualified'] is False
                assert 'points' not in record and len(record['approx_regions'])==2 and record['location_description']==''
                assert all(hashes(folder)[p]==sha for p,sha in old_hashes.items())
                checks.append('append_only_save_readback_and_instant_history_refresh')
                page.reload();page.wait_for_function('tasks.length===18');enter();wait_history(3)
                assert page.locator('#historyList button[data-revision]').count()==3
                assert page.evaluate('crops.map(c=>c.bbox_xywh)')==[[16,24,128,128],[32,48,128,128]]
                page.screenshot(path=str(evidence/'history_reopened.png'),full_page=True)
                checks.append('page_reload_restores_disk_version_and_native_regions')
                # A text-only position is retained without manufacturing geometry.
                page.click('#clearRegions');page.fill('#location','AUTOMATED text-only fixture with no pixel coordinates');page.fill('#reason','AUTOMATED TEXT ONLY CHECK; not a human judgment')
                page.click('#save');page.wait_for_function('!busy && revision===4 && historyRecords.length===4')
                assert page.evaluate('crops.every(c=>c===null) && positions.every(p=>p===null)')
                assert '无法自动定位' in page.locator('#historyFocus').inner_text();checks.append('text_only_history_does_not_invent_points')
                # Exposure registration alone is visibly distinguished from triage versions.
                page.select_option('#task',sid_empty);page.wait_for_function('sid=>ready && !busy && current.sample_id===sid',arg=sid_empty,timeout=30000)
                assert '仅登记暴露' in page.locator('#historyStatus').inner_text() and not page.locator('#historyList button').count()
                assert not (store/sid_empty).exists();checks.append('exposure_only_diagnostic')
                # Make sure bidirectional Geo Assist still works on actual SAR previews.
                for side in [0,1]:
                    canvas=page.locator('#overview'+str(side));box=canvas.bounding_box();canvas.click(position=dict(x=box['width']/2,y=box['height']/2))
                    page.wait_for_function('side=>last && last.source_side===side && last.request_sequence===seq && crops.every(c=>c)',arg=side)
                    assert page.evaluate('last.GT') is False
                checks.append('existing_bidirectional_geographic_assist_kept')
                page.screenshot(path=str(evidence/'exposure_only_and_geo_reference.png'),full_page=True)
                # New direct-click persistence: no extra Add Region operation is needed.
                page.click('#clearClicks');page.fill('#location','');page.check('[name=verdict][value="D"]')
                canvas=page.locator('#overview0');box=canvas.bounding_box();canvas.click(position=dict(x=box['width']/2,y=box['height']/2))
                page.wait_for_function('manualClicks.length===1 && last.source_side===0 && last.request_sequence===seq')
                clicked=page.evaluate('manualClicks[0]');assert clicked['input_event']=='overview_click' and clicked['side']==0
                page.click('[data-side="0"][data-dx="1"]');page.wait_for_function('last.request_sequence===seq && crops.every(c=>c)')
                assert page.evaluate('manualClicks')==[clicked],'pan must not create or overwrite a manual click'
                page.fill('#reason','AUTOMATED DIRECT CLICK SAVE CHECK; no human judgment');page.click('#save')
                page.wait_for_function('!busy && revision===1 && historyRecords.length===1')
                click_doc=json.loads((store/sid_empty/'v000001.json').read_text())
                assert len(click_doc['manual_clicks'])==1 and click_doc['manual_clicks'][0]['native_xy']==clicked['native_xy']
                assert click_doc['manual_clicks'][0]['side']==0 and not click_doc['approx_regions'] and not click_doc['location_description']
                assert click_doc['manual_clicks'][0]['exact_correspondence'] is False and click_doc['GT_qualified'] is False
                assert '点击 1 个' in page.locator('#saveStatus').inner_text()
                page.reload();page.wait_for_function('tasks.length===18');enter(sid_empty)
                assert page.evaluate('historySelected.manual_clicks.length===1 && crops[0]!==null && crops[1]===null')
                assert page.evaluate('focusPositions[0]')==clicked['native_xy']
                assert page.locator('#historyList button[data-click]').count()==1 and yellow('#overview0')>20 and yellow('#detail0')>20
                assert '不把预测位置补成人工点击' in page.locator('#historyFocus').inner_text()
                checks.append('source_click_saved_without_window_or_prediction_and_restored_after_reload')
                # Both sides can be clicked separately; they remain observations, not paired GT.
                canvas=page.locator('#overview1');box=canvas.bounding_box();canvas.click(position=dict(x=box['width']/2+20,y=box['height']/2))
                page.wait_for_function('manualClicks.length===2 && last.source_side===1 && last.request_sequence===seq && crops[1]!==null')
                detail=page.locator('#detail1');box=detail.bounding_box();detail.click(position=dict(x=box['width']/2+8,y=box['height']/2+4))
                page.wait_for_function('manualClicks.length===3 && last.request_sequence===seq')
                expected=page.evaluate('manualClicks');assert [c['input_event'] for c in expected]==['overview_click','overview_click','detail_click']
                page.click('#save');page.wait_for_function('!busy && revision===2 && historyRecords.length===2')
                click_doc=json.loads((store/sid_empty/'v000002.json').read_text())
                assert [dict(side=c['side'],native_xy=c['native_xy'],input_event=c['input_event']) for c in click_doc['manual_clicks']]==expected
                assert all(c['exact_correspondence'] is False and c['independent_GT'] is False for c in click_doc['manual_clicks'])
                page.reload();page.wait_for_function('tasks.length===18');enter(sid_empty)
                assert page.locator('#historyList button[data-revision]').count()==2
                assert page.evaluate('crops.every(c=>c!==null) && manualClicks.length===3')
                page.click('#historyList button[data-click="0"][data-version="2"]')
                page.wait_for_function('xy=>focusPositions[0] && focusPositions[0].every((v,i)=>v===xy[i])',arg=clicked['native_xy'])
                assert page.evaluate('revision')==2
                page.screenshot(path=str(evidence/'direct_clicks_reopened.png'),full_page=True)
                checks.append('independent_side_and_detail_clicks_kept_and_history_buttons_locate')
                payload=dict(sample_id=sid_empty,actor=ACTOR,verdict='D',reason='AUTOMATED validation check',location_description='',approx_regions=[],recommend_independent_annotation=False,expected_revision=2)
                payload['manual_clicks']=[dict(side=0,native_xy=[0,0],input_event='geographic_prediction')]
                failure=page.evaluate("async p=>{try{await api('/api/triage',p);return null;}catch(e){return e.message;}}",payload)
                assert failure and len(list((store/sid_empty).glob('v*.json')))==2
                payload['manual_clicks']=[dict(side=0,native_xy=[tasks[1]['views'][0]['width'],0],input_event='overview_click')]
                failure=page.evaluate("async p=>{try{await api('/api/triage',p);return null;}catch(e){return e.message;}}",payload)
                assert 'Native' in failure and len(list((store/sid_empty).glob('v*.json')))==2
                checks.append('prediction_role_and_invalid_native_clicks_rejected')
                # A fresh read-only session can inspect histories but cannot save.
                page.goto(url+'?verify=READONLY_AUTOMATED_UI_FIXTURE');page.wait_for_function('ready && !busy');assert page.locator('#save').is_disabled()
                checks.append('readonly_save_remains_disabled')
                version=browser.version;browser.close()
            assert not errors,errors
            assert hashes(ROOT/'assisted_triage')==protected,'production records changed during isolated tests'
            result=dict(passed=True,checks=checks,browser=version,page_errors=errors,synthetic_fixture_directory_only=True,
                production_assisted_triage_unchanged=True,synthetic_versions_created=6,real_human_judgments_created=0,GT_created=0,GPU=False)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            (evidence/'check_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
        finally:server.shutdown();server.server_close();thread.join(timeout=5)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence-dir',type=Path,required=True);args=parser.parse_args()
    args.evidence_dir.mkdir(parents=True,exist_ok=True);run(args.evidence_dir)
