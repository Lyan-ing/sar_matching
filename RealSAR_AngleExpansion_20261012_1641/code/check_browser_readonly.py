"""Real browser load/zoom checks. No actor, judgments, coordinates or POSTs."""
import sys
sys.dont_write_bytecode=True
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]

def run():
    launch=json.loads((ROOT/'readability_service.json').read_text());url=launch['url']
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');errors=[];posts=[];loaded=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
        page=browser.new_page(viewport=dict(width=1600,height=1100))
        page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        page.goto(url);page.wait_for_function('ready && current',timeout=60000)
        tasks=page.evaluate('tasks');assert len(tasks)==18
        samples=[tasks[0],next(t for t in tasks if t['pair_id']=='ANGLE_325c0714a279'),tasks[-1]]
        for task in samples:
            sid=task['sample_id'];page.select_option('#task',sid)
            page.wait_for_function('(sid)=>ready && current.sample_id===sid',arg=sid,timeout=60000)
            actual=page.evaluate('[0,1].map(i=>[$("im"+i).naturalWidth,$("im"+i).naturalHeight])')
            assert actual==[[v['native_info']['width'],v['native_info']['height']] for v in task['views']]
            assert page.locator('#mode option').count()==1 and page.locator('#mode').input_value()=='native'
            assert not page.locator('[name=verdict]:checked').count() and page.locator('#actor').input_value()==''
            assert not page.locator('#showMap').is_visible()
            page.select_option('#zoom','1');assert page.locator('#im0').evaluate('(im)=>Math.abs(im.width-im.naturalWidth)<1')
            page.select_option('#zoom','fit');loaded.append(dict(sample_id=sid,native_dimensions=actual))
        page.screenshot(path=str(ROOT/'reports'/f'native_review_readonly_{stamp}.png'),full_page=True)
        version=browser.version;browser.close()
    assert not errors and not posts and not (ROOT/'review/readability').exists()
    result=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),passed=True,browser=version,loaded=loaded,
        total_tasks=len(tasks),native_only=True,real_human_reviewed=0,POST_requests=posts,errors=errors,
        generated_points=0,GT=0,used_GPU=False,chromium_sandbox=True)
    with (ROOT/'reports'/f'browser_readonly_{stamp}.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
