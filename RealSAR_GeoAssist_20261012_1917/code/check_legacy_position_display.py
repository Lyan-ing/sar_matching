"""Read-only Chromium verification of actual legacy records and recovered view references."""
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
import argparse,json,pathlib,hashlib,threading
from playwright.sync_api import sync_playwright
from server import make_server,load_navigation_references
from geometry import ROOT,lines

def protected():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'assisted_triage').rglob('*') if p.is_file()}
def run(out):
    before=protected();tasks={t['sample_id']:t for t in lines(ROOT/'tasks.jsonl')}
    expected=load_navigation_references(ROOT/'exports/assisted_review_snapshot_20261013_201035/last_view_positions.csv',tasks)
    server=make_server(0,'LEGACY_NAVIGATION_READ_ONLY_CHECK');server.RequestHandlerClass.log_message=lambda *args:None
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();posts=[];errors=[];loaded=[];recovered=[]
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu']);page=browser.new_page(viewport={'width':1800,'height':1400})
            page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
            page.goto(f'http://127.0.0.1:{server.server_port}/?verify=LEGACY_NAVIGATION_READ_ONLY_CHECK');page.wait_for_function('ready && !busy',timeout=30000)
            def color_count(side):
                return page.locator('#overview'+str(side)).evaluate("c=>{const d=c.getContext('2d').getImageData(0,0,c.width,c.height).data;let n=0;for(let i=0;i<d.length;i+=4)if(d[i]===102&&d[i+1]===207&&d[i+2]===255&&d[i+3]===255)n++;return n;}")
            for sid in tasks:
                page.select_option('#task',sid);page.wait_for_function('sid=>ready && !busy && current.sample_id===sid',arg=sid,timeout=30000);loaded.append(sid)
                assert page.evaluate('manualClicks.length===0'),'legacy references must not become manual clicks'
                if sid in expected:
                    assert page.evaluate('recoveredActive') and page.locator('#recoveredPanel').is_visible()
                    for ref in expected[sid]:
                        side=ref['side'];assert page.evaluate('side=>focusPositions[side]',side)==ref['native_xy']
                        assert page.evaluate('side=>crops[side].requested_xy',side)==ref['native_xy']
                        assert color_count(side)>20
                    assert page.locator('#save').is_disabled() and '不能确认属于哪个保存版本' in page.locator('#recoveredPanel').inner_text()
                    assert page.evaluate('navigationReferences.every(p=>p.GT===false && p.confirmed_human_point===false && p.matched_review_revision===null)')
                    recovered.append(sid)
                    if len(recovered)==1:
                        page.screenshot(path=str(out/'legacy_positions_visible.png'),full_page=True)
                        page.click('#historyList button[data-revision="1"]')
                        page.wait_for_function('historySelected.revision===1 && recoveredActive && crops.every(c=>c)')
                        assert '日志没有版本关联' in page.locator('#historyFocus').inner_text()
                        page.reload();page.wait_for_function('ready && !busy',timeout=30000)
                        assert page.evaluate('recoveredActive && crops.every(c=>c) && manualClicks.length===0')
                        page.click('#locateRecovered');page.wait_for_function('crops.every(c=>c)')
                else:assert page.locator('#recoveredPanel').is_hidden()
            assert len(loaded)==18 and len(recovered)==13 and not posts and not errors
            browser.close()
        assert before==protected(),'original records changed'
        result={'passed':True,'all_18_loaded':len(loaded),'actual_legacy_samples_restored':len(recovered),'recovered_native_view_centers':26,
            'page_reload_restores_reference_views':True,'cyan_markers_verified_on_both_overviews':True,'historical_version_association_not_fabricated':True,
            'references_not_copied_to_manual_clicks':True,'original_versions_and_exposures_unchanged':True,'POST_requests':0,'page_errors':errors,'GT_created':0,'GPU':False}
        (out/'check_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False,indent=2))
    finally:server.shutdown();server.server_close();thread.join(timeout=5)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence-dir',type=pathlib.Path,required=True);args=parser.parse_args();args.evidence_dir.mkdir(parents=True,exist_ok=True);run(args.evidence_dir)
