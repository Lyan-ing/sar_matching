"""Actual read-only five-task UI test; workflow test data exist only in /tmp."""
import sys
sys.dont_write_bytecode=True
import json, tempfile, threading, shutil, hashlib
from pathlib import Path
from serve_review import ROOT, make_server
from export_annotations import export
from playwright.sync_api import sync_playwright

def start(root):
    s=make_server(root,0);t=threading.Thread(target=s.serve_forever,daemon=True);t.start();return s,t
def stop(s,t):s.shutdown();s.server_close();t.join()

def main():
    out=ROOT/'ui_checks';out.mkdir(exist_ok=False)
    evidence={'real_GT_created':False,'simulated_workflow_location':'/tmp only, removed after test','checks':[]}
    errors=[];posts=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,chromium_sandbox=True)
        evidence['browser_version']=browser.version
        page=browser.new_page(viewport={'width':1500,'height':1250},device_scale_factor=1)
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        s,t=start(ROOT/'annotation_adapter')
        try:
            page.goto(f'http://127.0.0.1:{s.server_port}')
            page.wait_for_function('tasks.length===5 && views.every(v=>v.image.naturalWidth===512)')
            for i in range(5):
                sid=page.evaluate(f'tasks[{i}].sample_id');page.select_option('#task',sid)
                page.wait_for_function(f'task.sample_id==={json.dumps(sid)} && views.every(v=>v.image.complete && v.image.naturalWidth===512)')
                assert page.locator('#mode').input_value()=='independent'
                assert page.locator('#geo').is_disabled() and page.locator('#sift').is_disabled()
                assert page.evaluate('points.length===0 && assistance.length===0 && doc.revision===0')
                page.screenshot(path=str(out/(sid+'_independent.png')),full_page=True)
            # Actual pointer -> native mapping after zoom + pan, without saving.
            page.locator('[data-zoom="0,1.5"]').click()
            page.locator('#c0').scroll_into_view_if_needed();box=page.locator('#c0').bounding_box()
            page.keyboard.down('Shift');page.mouse.move(box['x']+220,box['y']+220);page.mouse.down();page.mouse.move(box['x']+245,box['y']+230);page.mouse.up();page.keyboard.up('Shift')
            def click(side,x,y):
                c=page.locator('#c'+str(side));c.scroll_into_view_if_needed();b=c.bounding_box()
                pos=page.evaluate(f'Coordinates.pixelToScreen({x},{y},views[{side}].camera)')
                page.mouse.click(b['x']+pos[0]*b['width']/512,b['y']+pos[1]*b['height']/512)
                assert abs(float(page.locator('#x'+str(side)).input_value())-x)<.6
                assert abs(float(page.locator('#y'+str(side)).input_value())-y)<.6
            click(0,240.25,235.5)
            assert 'zoom=1.50' in page.locator('#hover0').inner_text()
            assert page.evaluate("Array.from(document.getElementById('l0').getContext('2d').getImageData(0,0,128,128).data).some(v=>v!==0)")
            page.screenshot(path=str(out/'zoom_pan_loupe.png'),full_page=True)
            assert not posts
            assert not list((ROOT/'annotation_adapter/annotations').rglob('*'))
            evidence['checks']+=['all_five_real_tasks_render_native_512','default_independent_auxiliary_disabled_empty',
                                 'real_pointer_zoom_pan_native_coordinates','native_8x_loupe_rendered','real_UI_zero_POST_zero_annotations']
        finally:stop(s,t)
        with tempfile.TemporaryDirectory(prefix='SIMULATED_SAR_BENCHMARK_') as tmp:
            tmp=Path(tmp);mock=tmp/'store'
            shutil.copytree(ROOT/'annotation_adapter',mock)
            s,t=start(mock)
            try:
                page.goto(f'http://127.0.0.1:{s.server_port}');page.wait_for_function('tasks.length===5 && views.every(v=>v.image.naturalWidth===512)')
                page.fill('#actor','SIMULATED_OPERATOR_A');click(0,170,220);click(1,172,222)
                page.fill('#landmark','SIMULATED UI TEST ONLY');page.check('#static');page.check('#visible')
                page.fill('#sigma0','1');page.fill('#sigma1','1');page.click('#upsert');page.select_option('#disposition','local_only')
                page.click('#save');page.wait_for_function('doc.revision===1')
                assert page.evaluate('doc.gt_quality.status!=="gt_qualified"')
                page.select_option('#mode','review');page.locator('.point').click()
                assert page.locator('#x0').input_value()=='' and page.locator('#x1').input_value()==''
                click(0,170,220);click(1,172,222);page.fill('#sigma0','1');page.fill('#sigma1','1');page.check('#static');page.check('#visible')
                page.click('#review');page.wait_for_function('document.getElementById("message").textContent.includes("reviewer must differ")')
                assert page.evaluate('doc.revision===1')
                page.fill('#actor','SIMULATED_OPERATOR_B');page.click('#review');page.wait_for_function('doc.revision===2')
                assert page.evaluate('doc.gt_quality.status!=="gt_qualified"')
                page.select_option('#mode','assisted');page.click('#geo');page.wait_for_function('assistance.length>0')
                page.click('#hide');page.select_option('#mode','independent');page.locator('.point').click()
                page.fill('#x0','175');page.click('#upsert');page.click('#save');page.wait_for_function('doc.revision===3')
                assert page.evaluate('doc.points[0].used_assistance && doc.points[0].review===undefined')
                assert len((mock/'annotations/assistance_events.jsonl').read_text().splitlines())==1
                sid=page.evaluate('task.sample_id');doc=s.store.load(sid)
                assert 'annotator_assisted' in doc['gt_quality']['point_evidence'][0]['reasons']
                summary=export(mock,tmp/'export');assert summary['points']==1 and summary['gt_qualified']==0
                assert (tmp/'export/points.csv').exists() and (tmp/'export/history/annotations/assistance_events.jsonl').exists()
                assert len(list((tmp/'export/history/annotations'/sid).glob('v*.json')))==3
                for rel,h in summary['sha256'].items():assert hashlib.sha256((tmp/'export'/rel).read_bytes()).hexdigest()==h
                try:export(mock,tmp/'export')
                except FileExistsError:pass
                else:raise AssertionError('export must refuse overwrite')
                evidence['checks']+=['SIMULATED_only_second_person_reclick_original_coordinates_hidden','SIMULATED_same_person_review_refused',
                    'SIMULATED_assistance_audit_persists_after_hiding','SIMULATED_edit_invalidates_review_assisted_ineligible',
                    'SIMULATED_full_export_coordinates_history_exposure_hashes','SIMULATED_export_overwrite_refused']
            finally:stop(s,t)
        browser.close()
    assert not errors,errors
    evidence.update(passed=True,console_errors=errors)
    (out/'browser_acceptance.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
