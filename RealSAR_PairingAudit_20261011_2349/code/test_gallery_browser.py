"""Real Chromium preview checks; synthetic browser clicks, no annotation saves."""
import sys, os
sys.dont_write_bytecode=True
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
os.environ['PLAYWRIGHT_BROWSERS_PATH']='/tmp/sar_gt_browsers_verified'
import functools, json, threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'browser'/datetime.now().strftime('%Y%m%d_%H%M%S');out.mkdir(parents=True,exist_ok=False)
    errors=[];tested=[]
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(SimpleHTTPRequestHandler,directory=str(ROOT)))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    try:
        with sync_playwright() as p:
            b=p.chromium.launch(headless=True,chromium_sandbox=True)
            page=b.new_page(viewport={'width':1500,'height':1000});page.on('pageerror',lambda e:errors.append(str(e)))
            url=f'http://127.0.0.1:{server.server_port}'
            page.goto(url);assert page.locator('article').count()==8;page.screenshot(path=str(out/'atlas_browser.png'),full_page=False)
            for r in json.loads((ROOT/'candidates.json').read_text()):
                page.goto(url+'/'+r['preview_page']);page.wait_for_function("[...document.querySelectorAll('.scroll img')].every(x=>x.complete&&x.naturalWidth===512)")
                assert page.locator('details[open]').count()==0
                assert page.locator('canvas').count()==0
                page.select_option('#display','original');page.wait_for_function("document.querySelector('#im0').src.endsWith('original0.png')&&document.querySelector('#im0').complete")
                page.locator('#scale').evaluate("e=>{e.value=300;e.dispatchEvent(new Event('input'))}");assert page.locator('#im0').evaluate('e=>e.width')==1536
                page.check('#sync');page.locator('.scroll').nth(0).evaluate('e=>{e.scrollLeft=140;e.scrollTop=180}')
                page.wait_for_function("document.querySelectorAll('.scroll')[1].scrollLeft===140")
                page.uncheck('#sync');page.locator('.scroll').nth(1).evaluate('e=>e.scrollLeft=250');page.wait_for_timeout(100)
                assert page.locator('.scroll').nth(0).evaluate('e=>e.scrollLeft')==140
                page.click('#reset');assert page.locator('#im0').evaluate('e=>e.width')==512
                if r['sample_id']=='Capella_0006__chip_001':page.screenshot(path=str(out/'priority_candidate.png'),full_page=False)
                page.locator('details').filter(has=page.locator('summary',has_text='地图覆盖')).locator('summary').click()
                page.wait_for_function("JSON.parse(localStorage.getItem('sar_candidate_preview_exposure')||'[]').length>=1")
                tested.append(r['sample_id'])
            page.goto(url+'/kingston/index.html');page.screenshot(path=str(out/'kingston_browser.png'),full_page=False)
            assert not errors
            result={'passed':True,'browser':b.version,'candidates_tested':tested,'errors':errors,'checks':['8 cards and all native512 files loaded','geographic diagnostics collapsed by default','native original display selector','300% nearest display zoom','synchronous scrolling','independent scrolling','reset','local geographic exposure recording only','Kingston page loads'],'annotation_store_access':False,'browser_actions':'automated UI test clicks only, not independent human judgement'}
            (out/'acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);b.close()
    finally:server.shutdown();server.server_close()

if __name__=='__main__':main()
