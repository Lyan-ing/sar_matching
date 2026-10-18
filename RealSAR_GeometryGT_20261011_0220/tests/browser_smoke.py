"""Actual Chromium interaction, simulated points saved ONLY under /tmp."""
import sys,json,tempfile,threading,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'annotation_app'))
from server import make_server
from playwright.sync_api import sync_playwright

def main():
    evidence={'test_points':'SIMULATED_ONLY under /tmp; real annotations untouched','browser':'Playwright 1.58.0 Chromium headless','checks':[]}
    server=make_server(ROOT,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,chromium_sandbox=True);evidence['browser_version']=browser.version;page=browser.new_page(viewport={'width':1560,'height':1250},device_scale_factor=1)
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        try:
            page.goto(f'http://127.0.0.1:{server.server_port}');page.wait_for_function('tasks.length===24 && views.every(v=>v.image.naturalWidth===512)')
            assert page.locator('#mode').input_value()=='independent';assert page.locator('#geo').is_disabled();assert page.locator('#sift').is_disabled();assert page.evaluate('points.length')==0
            page.screenshot(path=str(ROOT/'visualization/annotation_app_demo.png'),full_page=True);evidence['checks'].append('real24_tasks_load_default_independent_no_GT_no_candidate_overlay')
            for i in range(5):
                sid=page.evaluate(f'tasks[{i}].sample_id');page.select_option('#task',sid);page.wait_for_function(f'task.sample_id==={json.dumps(sid)} && views.every(v=>v.image.naturalWidth===512)')
                assert page.locator('#metadata').inner_text();assert page.evaluate('points.length')==0
                page.screenshot(path=str(ROOT/f'visualization/demo_task_{i+1}.png'),full_page=True)
            evidence['checks'].append('five_real_tasks_navigate_and_native_images_render')
        finally:server.shutdown();server.server_close();thread.join()
        with tempfile.TemporaryDirectory(prefix='SIMULATED_SAR_UI_') as tmp:
            tmp=Path(tmp)
            for d in ['annotation_tasks','gt_quality','candidate_matches']:(tmp/d).mkdir()
            original=[json.loads(s) for s in (ROOT/'annotation_tasks/tasks.jsonl').read_text().splitlines()];mock=[]
            for i,r in enumerate(original[:2]):
                r=dict(r,sample_id=f'SIMULATED_UI_{i}',visual_note='SIMULATED UI TEST, no real correspondence claimed',demo_priority=i+1)
                f=tmp/'candidate_matches'/f"{r['sample_id']}.json";f.write_text(json.dumps({'sample_id':r['sample_id'],'geographic':[{'xy0':[50,50],'xy1':[55,51]}],'sift_candidates':[]}));r['candidate_sha256']=hashlib.sha256(f.read_bytes()).hexdigest();mock.append(r)
            (tmp/'annotation_tasks/tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in mock));server=make_server(tmp,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                page.goto(f'http://127.0.0.1:{server.server_port}');page.wait_for_function('tasks.length===2 && views.every(v=>v.image.naturalWidth===512)');page.fill('#actor','SIMULATED_alice')
                def click_pixel(side,x,y):
                    page.locator('#c'+str(side)).scroll_into_view_if_needed()
                    s=page.evaluate(f'Coordinates.pixelToScreen({x},{y},views[{side}].camera)');box=page.locator('#c'+str(side)).bounding_box();page.mouse.click(box['x']+s[0]*box['width']/512,box['y']+s[1]*box['height']/512)
                    assert abs(float(page.locator('#x'+str(side)).input_value())-x)<.6;assert abs(float(page.locator('#y'+str(side)).input_value())-y)<.6
                click_pixel(0,40.125,60.25);page.locator('[data-zoom="0,1.5"]').click();click_pixel(0,150.25,200.5)
                box=page.locator('#c0').bounding_box();page.keyboard.down('Shift');page.mouse.move(box['x']+200,box['y']+200);page.mouse.down();page.mouse.move(box['x']+240,box['y']+230);page.mouse.up();page.keyboard.up('Shift');click_pixel(0,170.125,240.5);click_pixel(1,172.125,243.5)
                evidence['checks'].append('actual_mouse_coordinate_mapping_CSS_resize_zoom_pan_subpixels')
                def fill():
                    page.fill('#landmark','SIMULATED TEST ONLY');page.check('#static');page.check('#visible');page.fill('#sigma0','.3');page.fill('#sigma1','.3');click_pixel(0,170.125,240.5);click_pixel(1,172.125,243.5)
                fill();page.click('#upsert');assert page.evaluate('points.length')==1;page.click('#undo');assert page.evaluate('points.length')==0
                fill();page.click('#upsert');page.select_option('#disposition','local_only');page.click('#save');page.wait_for_function('doc.revision===1');page.click('#reload');page.wait_for_function('doc.revision===1 && points.length===1');assert page.evaluate('doc.gt_quality.status')!='gt_qualified'
                evidence['checks'].append('add_undo_save_reload_version_and_no_illegal_GT_upgrade')
                page.select_option('#mode','review');page.fill('#actor','SIMULATED_bob');page.locator('.point').click();assert page.locator('#x0').input_value()=='';assert page.evaluate('assistance.length')==0
                click_pixel(0,170.125,240.5);click_pixel(1,172.125,243.5);page.check('#static');page.check('#visible');page.fill('#sigma0','.4');page.fill('#sigma1','.4');page.click('#review');page.wait_for_function('doc.revision===2');assert page.evaluate('doc.points[0].review.reviewer')=='SIMULATED_bob'
                evidence['checks'].append('independent_reviewer_original_coords_hidden_requires_reclick')
                page.select_option('#mode','independent');page.locator('.point').click();page.fill('#x0','172');page.click('#upsert');page.click('#save');page.wait_for_function('doc.revision===3');assert page.evaluate('doc.points[0].review===undefined')
                page.locator('.point').click();page.click('#delete');assert page.evaluate('points.length')==0;page.click('#undo');assert page.evaluate('points.length')==1;page.click('#save');page.wait_for_function('doc.revision===4')
                evidence['checks'].append('edit_invalidates_review_delete_undo_immutable_history')
                page.select_option('#mode','assisted');page.click('#geo');page.wait_for_function('assistance.length===1');assert len((tmp/'annotations/assistance_events.jsonl').read_text().splitlines())==1;assert len(server.store.load('SIMULATED_UI_0')['points'])==1
                evidence['checks'].append('assistance_exposure_audited_never_auto_imported')
                page.select_option('#mode','independent');page.locator('.point').click();page.fill('#x0','174');page.click('#upsert');page.click('#save');page.wait_for_function('doc.revision===5');assert page.evaluate('doc.points[0].used_assistance')
                page.click('#next');page.wait_for_function('task.sample_id==="SIMULATED_UI_1"');page.click('#prev');page.wait_for_function('task.sample_id==="SIMULATED_UI_0" && doc.revision===5')
                evidence['checks'].append('switch_pairs_and_persistent_assistance_provenance')
                assert not errors,errors;evidence['console_errors']=errors;evidence['simulated_versions']=5
            finally:server.shutdown();server.server_close();thread.join()
        browser.close()
    evidence['passed']=True;(ROOT/'tests/browser_acceptance.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
