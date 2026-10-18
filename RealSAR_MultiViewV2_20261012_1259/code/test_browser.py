"""Actual Chromium readability + reused varying-size GT UI; SIMULATED fixture only."""
import sys
sys.dont_write_bytecode=True
import os,json,shutil,threading
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
from playwright.sync_api import sync_playwright
from geo_tools import ROOT,BASE,lines,save,sha
from readability_server import make_server
from prepare_gt import prepare
sys.path.insert(0,str(BASE/'RealSAR_IndependentBenchmark_20261011_2015'))
import serve_review
from export_annotations import export

def main():
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M');out=ROOT/('browser_SIMULATED_ONLY_'+stamp);out.mkdir(exist_ok=False);fixture=out/'fixture';fixture.mkdir();shutil.copyfile(ROOT/'quick_review.html',fixture/'quick_review.html');shutil.copyfile(ROOT/'regions.jsonl',fixture/'regions.jsonl');shutil.copyfile(ROOT/'atlas.png',fixture/'atlas.png');shutil.copyfile(ROOT/'candidate_atlas.png',fixture/'candidate_atlas.png');shutil.copyfile(ROOT/'candidate_regions.jsonl',fixture/'candidate_regions.jsonl')
    tasks=lines(ROOT/'candidate_regions.jsonl')
    for r in tasks:
        target=fixture/'regions'/r['sample_id'];target.mkdir(parents=True)
        for p in (ROOT/'regions'/r['sample_id']).glob('*.png'):os.link(p,target/p.name)
    evidence=dict(simulated_only=True,no_real_GT_claimed=True,checks=[],errors=[],passed=False);servers=[]
    def launch(server):
        t=threading.Thread(target=server.serve_forever,daemon=True);t.start();servers.append((server,t));return 'http://127.0.0.1:'+str(server.server_port)
    s=make_server(fixture,0);url=launch(s)
    try:
        with sync_playwright() as p:
            browser=p.chromium.launch(headless=True,chromium_sandbox=True);evidence['browser']=browser.version;page=browser.new_page(viewport=dict(width=1600,height=1100));page.on('pageerror',lambda e:evidence['errors'].append(str(e)));page.on('dialog',lambda d:d.accept());page.goto(url);page.wait_for_function('ready && current')
            assert page.locator('#mode').input_value()=='native'
            assert not (fixture/'readability').exists()
            # Common-FOV cannot be viewed anonymously without an exposure record.
            page.select_option('#mode','common');page.wait_for_function("ready && $('mode').value==='native'");assert not (fixture/'readability').exists()
            page.fill('#actor','SIMULATED_QUICK_A');assert not page.locator('[name=verdict]:checked').count()
            for n,r in enumerate(tasks):
                page.select_option('#task',r['sample_id']);page.wait_for_function('(sid)=>ready && current.sample_id===sid',arg=r['sample_id'])
                actual=page.evaluate('[0,1].map(i=>[$("im"+i).naturalWidth,$("im"+i).naturalHeight])')
                assert actual==[[v['native_info']['width'],v['native_info']['height']] for v in r['views']]
                assert page.locator('#mode').input_value()=='native' and not page.locator('[name=verdict]:checked').count()
                for mode in ['context','common','detail','native']:
                    page.select_option('#mode',mode);page.wait_for_function('ready');assert all(page.locator('#im'+str(i)).evaluate('(im)=>im.complete&&im.naturalWidth>0') for i in [0,1])
                page.select_option('#display','linear');page.wait_for_function('ready');page.select_option('#display','db');page.wait_for_function('ready')
                page.select_option('#zoom','1');assert page.locator('#im0').evaluate('(im)=>Math.abs(im.width-im.naturalWidth)<1')
                page.select_option('#zoom','fit');verdict=['A','B','C','D'][n%4];page.check('[name=verdict][value='+verdict+']');page.fill('#note','SIMULATED browser judgment only');page.click('#save');page.wait_for_function('revision===1 && !busy')
                assert s.reviews.load(r['sample_id'])['GT_qualified'] is False
                page.screenshot(path=str(out/(r['sample_id']+'.png')),full_page=True)
            evidence['checks'].append('all 8 final candidate native/context/Common-FOV/detail pairs loaded; differing native dimensions; fixed displays, true pixel zoom; no automatic points or pass defaults')
            first=tasks[0]['sample_id'];page.select_option('#task',first);page.wait_for_function('(sid)=>ready && current.sample_id===sid',arg=first)
            old=sha(fixture/'readability'/first/'v000001.json');page.check('[name=verdict][value=A]');page.click('#save');page.wait_for_function('revision===2 && !busy');assert old==sha(fixture/'readability'/first/'v000001.json')
            assert len((fixture/'readability/assistance_events.jsonl').read_text().splitlines())>=len(tasks)
            evidence['checks'].append('A/B/C/D explicit save, append-only versions, unchanged previous revision; actual Common-FOV exposure events retained')
            gt=prepare(fixture);assert gt is not None
            gserver=serve_review.make_server(gt,0);gurl=launch(gserver);gpage=browser.new_page(viewport=dict(width=1600,height=1150));gpage.on('pageerror',lambda e:evidence['errors'].append(str(e)));gpage.goto(gurl+'/?task='+first);gpage.wait_for_function('doc && !loading');gpage.fill('#actor','SIMULATED_QUICK_A');gpage.click('#begin')
            assert gpage.locator('#mode').input_value()=='minimal' and gpage.evaluate('assistance.length===0')
            for i in [0,1]:
                assert gpage.evaluate('views['+str(i)+'].canvas.width')==tasks[0]['views'][i]['native_info']['width']
                canvas=gpage.locator('#c'+str(i));box=canvas.bounding_box();w=gpage.evaluate('views['+str(i)+'].canvas.width');h=gpage.evaluate('views['+str(i)+'].canvas.height');assert abs(box['width']/box['height']-w/h)<.01
                gpage.mouse.click(box['x']+80.5*box['width']/w,box['y']+100.5*box['height']/h)
            gpage.keyboard.press('Enter');gpage.wait_for_function('doc.revision===1 && !busy');point=gserver.store.load(first)['points'][0];assert abs(point['xy0'][0]-80)<.02 and abs(point['xy1'][1]-100)<.02
            assert point['used_assistance'] and point['sigma0_px'] is None;assert gpage.evaluate('doc.gt_quality.status')!='gt_qualified'
            review=browser.new_page();review.goto(gurl+'/?review=1&task='+first);review.wait_for_function('doc && !loading');assert review.evaluate('doc.points.every(p=>p.xy0===undefined&&p.xy1===undefined)')
            exported=export(gt,out/'SIMULATED_GEOMETRY_EXPORT');assert exported['gt_qualified']==0
            evidence['checks'].append('only simulated A/B pass tasks emitted; existing minimal UI handles unequal native sizes/aspect ratio; actual clicks/Enter; exposure carried into candidate evidence; review hides coordinates; export no GT')
            assert not evidence['errors'];evidence['passed']=True;browser.close()
    finally:
        for server,t in servers:server.shutdown();server.server_close();t.join(timeout=5)
        save(ROOT/('browser_test_results_'+stamp+'.json'),evidence)
    print(json.dumps(evidence,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
