"""Real browser layers checked on tmp copies; no fake GT or exposure in release."""
import sys
sys.dont_write_bytecode=True
import json,tempfile,threading,shutil,importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT,save,sha
from playwright.sync_api import sync_playwright
spec=importlib.util.spec_from_file_location('annotation_adapter_server',ROOT/'annotation_adapter/serve.py');adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)

def main():
    evidence={'checks':[],'real_annotations_written':0,'real_assistance_exposures_written':0,'simulated_actor_location':'/tmp only','passed':False}
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,chromium_sandbox=True);evidence['browser_version']=browser.version;page=browser.new_page(viewport={'width':1560,'height':1250});errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        real=ROOT/'annotation_adapter';server=adapter.make_server(real,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            page.goto(f'http://127.0.0.1:{server.server_port}');page.wait_for_function('tasks.length===8 && views.every(v=>v.image.naturalWidth===512)')
            assert page.locator('#mode').input_value()=='independent';assert page.locator('#geo').is_disabled() and page.locator('#sift').is_disabled();assert page.evaluate('points.length===0 && assistance.length===0')
            page.screenshot(path=str(ROOT/'visualization/annotation_adapter_default.png'),full_page=True)
            for i in range(8):
                sid=page.evaluate(f'tasks[{i}].sample_id');page.select_option('#task',sid);page.wait_for_function(f'task.sample_id==={json.dumps(sid)} && views.every(v=>v.image.naturalWidth===512)');assert page.evaluate('points.length===0 && assistance.length===0')
            evidence['checks'].append('8_real_diagnostic_tasks_native_images_default_independent_no_points')
        finally:server.shutdown();server.server_close();thread.join()
        with tempfile.TemporaryDirectory(prefix='SIMULATED_AUTO_SAR_BROWSER_') as tmp:
            tmp=Path(tmp)
            for d in ['annotation_tasks','candidate_matches','annotations','gt_quality']:(tmp/d).mkdir()
            for f in ['index.html','app.js','coordinates.js']:shutil.copyfile(real/f,tmp/f)
            tasks=[json.loads(s) for s in (real/'annotation_tasks/tasks.jsonl').read_text().splitlines()]
            task=next(t for t in tasks if t['has_B_pseudo']);original_sid=task['sample_id'];payload=json.loads((real/'candidate_matches'/f'{original_sid}.json').read_text());sid='SIMULATED_BROWSER_LAYER';task=dict(task,sample_id=sid);payload['sample_id']=sid
            f=tmp/'candidate_matches'/f'{sid}.json';f.write_text(json.dumps(payload));task['candidate_sha256']=sha(f);(tmp/'annotation_tasks/tasks.jsonl').write_text(json.dumps(task)+'\n')
            server=adapter.make_server(tmp,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            try:
                page.goto(f'http://127.0.0.1:{server.server_port}');page.wait_for_function('tasks.length===1 && views.every(v=>v.image.naturalWidth===512)');page.fill('#actor','SIMULATED_BROWSER_ACTOR');page.select_option('#mode','assisted')
                for button,key,source in [('geo','geographic','georeference_correspondences'),('sift','sift_candidates','pseudo_correspondences')]:
                    page.click('#'+button);page.wait_for_function(f'assistance.length==={len(payload[key])}');assert source in page.locator('#message').inner_text();assert page.evaluate('points.length===0 && doc.revision===0')
                    actual=page.evaluate('assistance');assert actual==payload[key];event=json.loads((tmp/'annotations/assistance_events.jsonl').read_text().splitlines()[-1]);assert event['source']==source and event['independent_GT'] is False
                    if button=='sift':assert event['generator']==payload['pseudo_generator'] and payload['pseudo_generator'] in page.locator('#message').inner_text()
                evidence['checks'].append('geo_and_pseudo_layers_render_original_native_coordinates_without_import')
                evidence['checks'].append('actual_source_generator_hash_and_person_logged_truthfully_under_tmp')
                page.select_option('#mode','independent');assert page.evaluate('assistance.length===0 && points.length===0');assert len(server.store.assisted(sid,'SIMULATED_BROWSER_ACTOR'))==2
                evidence['checks'].append('switch_to_independent_hides_points_but_preserves_exposure')
                page.screenshot(path=str(ROOT/'visualization/annotation_adapter_simulated_check.png'),full_page=True)
            finally:server.shutdown();server.server_close();thread.join()
        assert not errors,errors;browser.close();evidence.update(passed=True,console_errors=errors)
    assert not list((real/'annotations').rglob('*.json')) and not (real/'annotations/assistance_events.jsonl').exists()
    save(ROOT/'tests/browser_acceptance.json',evidence);print(json.dumps(evidence,indent=2),flush=True)

if __name__=='__main__':main()
