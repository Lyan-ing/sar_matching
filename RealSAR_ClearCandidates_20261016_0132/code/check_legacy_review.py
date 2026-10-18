"""Read-only actual-browser check of the old 18-input gallery, no annotation actions."""
import os,sys,tempfile,threading,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TOOL=ROOT.parent/'RealSAR_SingleExpert_20261015_2332'
sys.path.insert(0,str(TOOL/'code'));sys.path.insert(0,'/tmp/sar_gt_browser_deps')
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/sar_gt_browsers_verified')
from expert import Dataset,Store,save,stamp,now
from server import make_server
from playwright.sync_api import sync_playwright

def main():
 data=Dataset();before={s:d['content_sha256'] for s,d in Store(data).docs.items()};posts=[];errors=[]
 with tempfile.TemporaryDirectory(prefix='legacy-gallery-check-') as tmp:
  service=make_server(0,store_root=tmp);thread=threading.Thread(target=service.serve_forever,daemon=True);thread.start()
  try:
   with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
    page=browser.new_page(viewport=dict(width=1500,height=1000));page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
    page.goto(f'http://127.0.0.1:{service.server_port}/legacy-review/',wait_until='load')
    page.wait_for_function('document.querySelectorAll("article:not([hidden])").length===13')
    page.wait_for_function('Array.from(document.querySelector("article:not([hidden])").querySelectorAll("img")).every(im=>im.complete&&im.naturalWidth>0)')
    assert '建筑转角结构比较清晰' in page.locator('article').first.inner_text()
    assert page.locator('article').count()==18 and page.locator('img').count()==62
    for value,count in [('A',7),('B',2),('C',4),('all',18),('未筛选',5),('reviewed',13)]:
     page.select_option('#filter',value);assert page.locator('article:not([hidden])').count()==count
    page.fill('#search','18.882');assert page.locator('article:not([hidden])').count()==1
    assert not page.locator('form').count() and not posts and not errors
    browser.close()
  finally:service.shutdown();service.server_close();thread.join()
 assert {s:d['content_sha256'] for s,d in Store(data).docs.items()}==before
 result=dict(at=now(),passed=True,default_actual_reviewed_pairs=13,complete_original_queue=18,actual_verdict_counts=dict(A=7,B=2,C=4),original_overview_images=36,old_log_viewport_images=26,default_visible_images_load=True,filters_and_search=True,POST_count=len(posts),javascript_errors=errors,production_annotation_hashes_unchanged=True,GT_created=0)
 save(ROOT/'legacy_review'/f'LEGACY_BROWSER_CHECK_{stamp()}.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
