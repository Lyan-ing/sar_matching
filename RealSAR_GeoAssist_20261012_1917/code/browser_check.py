"""Actual browser interactions with real SAR; read-only session, no simulated human records."""
import sys,json,time,math
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
from playwright.sync_api import sync_playwright
from geometry import ROOT,lines,save,queue,View,mapped

def wait_point(page,side):
    page.wait_for_function('side=>last && last.request_sequence===seq && last.source_side===side && [0,1].every(i=>crops[i] && (crops[i].outside || crops[i].requested_xy.every((v,k)=>Math.abs(v-positions[i][k])<1e-6)))',arg=side,timeout=15000)

def click_native(page,side,point):
    element=page.locator('#overview'+str(side));v=page.evaluate('side=>current.views[side]',side)
    box=element.bounding_box()
    element.click(position=dict(x=(point[0]+.5)*box['width']/v['width'],y=(point[1]+.5)*box['height']/v['height']))
    wait_point(page,side)

def run():
    service=json.loads((ROOT/'service.json').read_text());url=service['url'];errors=[];posts=[];http_errors=[];loaded=[];clicks=[]
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
        page=browser.new_page(viewport=dict(width=1800,height=1400))
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
        page.on('response',lambda r:http_errors.append(dict(status=r.status,path=r.url.split('?')[0])) if r.status>=400 else None)
        # Normal human page refuses anonymous assist locally; no forged actor.
        page.goto(url);page.wait_for_function('tasks.length===18')
        assert not page.evaluate('ready') and page.locator('#actor').input_value()==''
        page.click('#enter');page.wait_for_function('document.getElementById("status").textContent.includes("真实身份")')
        assert not page.evaluate('ready') and not page.locator('[name=verdict]:checked').count()
        # Secret-scoped automated read-only mode. Cannot save human exposure/triage.
        page.goto(url+'?verify='+service['readonly_verification_secret'])
        page.wait_for_function('ready && current',timeout=30000);assert page.locator('#save').is_disabled()
        task_list=page.evaluate('tasks');assert len(task_list)==18
        actual={t['sample_id']:t for t in lines(ROOT/'tasks.jsonl')};rows={r['sample_id']:r for r in queue()}
        for task in task_list:
            sid=task['sample_id'];page.select_option('#task',sid)
            page.wait_for_function('sid=>ready && current && current.sample_id===sid && !busy',arg=sid,timeout=15000)
            dims=page.evaluate('current.views.map(v=>[v.width,v.height])')
            assert dims==[[v['width'],v['height']] for v in actual[sid]['views']]
            # No CSS letterbox distortion: click coordinates follow the full canvas aspect.
            for side in [0,1]:
                size=page.locator('#overview'+str(side)).evaluate('(c)=>{let b=c.getBoundingClientRect();return [b.width,b.height,c.width,c.height];}')
                assert abs(size[0]/size[1]-size[2]/size[3])<.002
            loaded.append(dict(sample_id=sid,dimensions=dims))
            if task['priority']<=6 or actual[sid]['views'][0]['crs']!=actual[sid]['views'][1]['crs']:
                record=json.loads((ROOT/'validation'/sid/'point_tests0.json').read_text())
                candidates=[r for r in record if r['target_valid'] and r['source_valid']]
                expected=candidates[len(candidates)//2];start=time.perf_counter()
                click_native(page,0,expected['source_xy']);left=page.evaluate('last')
                views=[View(rows[sid],i) for i in [0,1]]
                expected_actual=mapped(views[0],views[1],[left['source_xy']])[0]
                assert math.dist(left['target_xy'],expected_actual)<1e-5
                box0=page.locator('#overview0').bounding_box()
                pointer_bound=2*max(dims[0][0]/box0['width'],dims[0][1]/box0['height'])
                assert math.dist(left['source_xy'],expected['source_xy'])<pointer_bound
                elapsed_ms=(time.perf_counter()-start)*1000
                # Genuine DOM right click, reversing the same real geocoded location.
                click_native(page,1,left['target_xy']);right=page.evaluate('last')
                expected_reverse=mapped(views[1],views[0],[right['source_xy']])[0]
                assert math.dist(right['target_xy'],expected_reverse)<1e-5
                box1=page.locator('#overview1').bounding_box()
                assert math.dist(right['source_xy'],left['target_xy'])<2*max(dims[1][0]/box1['width'],dims[1][1]/box1['height'])
                page.select_option('#span','128')
                page.wait_for_function('crops.every(c=>c && !c.outside && c.bbox_xywh[2]<=129 && c.bbox_xywh[3]<=129)')
                before=page.evaluate('positions[1].slice()');box=page.locator('#detail1').bounding_box()
                page.locator('#detail1').click(position=dict(x=box['width']/2+8,y=box['height']/2+4));wait_point(page,1)
                assert math.dist(page.evaluate('positions[1]'),before)>1
                assert page.locator('#save').is_disabled()
                assert page.evaluate('last.GT') is False
                clicks.append(dict(sample_id=sid,left_target=left['target_xy'],reverse_target=right['target_xy'],
                    mapping_ms=left['mapping_ms'],click_to_both_crops_ms=elapsed_ms,zoom_span_native=128,
                    detail_click_works=True,mathematical_reference_only=True,
                    overview_mouse_quantization_px=math.dist(left['source_xy'],expected['source_xy']),
                    actual_source_native_xy=left['source_xy']))
                if task['priority'] in [1,3,6]:
                    page.screenshot(path=str(ROOT/'reports'/f'browser_{task["priority"]}_{stamp}.png'),full_page=True)
                page.select_option('#span','256')
        version=browser.version;browser.close()
    assert not errors and not posts and not http_errors,(errors,posts,http_errors)
    assert not list((ROOT/'assisted_triage').rglob('v*.json'))
    assert not (ROOT/'assisted_triage/exposure_events.jsonl').exists()
    result=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),passed=True,browser=version,
        all_18_loaded=loaded,actual_bidirectional_click_tests=clicks,default_anonymous_gate_passed=True,
        automated_read_only_save_disabled=True,POST_requests=posts,page_errors=errors,http_errors=http_errors,
        real_assisted_triage_records=0,real_GT=0,simulated_human_judgments=0,chromium_sandbox=True,GPU=False)
    save(ROOT/'reports'/f'browser_check_{stamp}.json',result)
    print('PASS browser loaded',len(loaded),'real tasks; bilateral click/detail tests',len(clicks),'no POST or human records')

if __name__=='__main__':run()
