"""Read-only actual Chromium check. No invented identity, clicks, saves or GT."""
from common import *
import os
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/sar_gt_browsers_verified')
from playwright.sync_api import sync_playwright

def main():
 t=stamp();idx=rows(latest('annotation_*/PACKAGE_INDEX_*.jsonl'));selected=[idx[0],idx[1],idx[2],idx[3]];checks=[];errors=[];requests=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
  for p in selected:
   page=browser.new_page(viewport=dict(width=1600,height=1050));page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append(dict(url=r.url,method=r.method)))
   page.goto(Path(p['html']).as_uri(),wait_until='load');page.wait_for_function('typeof task!=="undefined" && task && views.every(v=>v.image.complete&&v.image.naturalWidth===512)')
   before=page.evaluate('({tasks:tasks.length,task:task.sample_id,images:views.map(v=>[v.image.naturalWidth,v.image.naturalHeight]),points:points.length,operator,documents:localDocuments.size,identityOpen:document.getElementById("identityDialog").open,package:window.BLIND_PACKAGE_ID})')
   assert before['tasks']==p['task_count'] and before['points']==0 and not before['operator'] and before['documents']==0 and before['identityOpen']
   assert not page.locator('#geo').is_visible() and not page.locator('#pseudo').is_visible()
   # Programmatic task selection under the untouched identity dialog is read-only.
   page.evaluate('load(tasks.length-1)');page.wait_for_function('!loading && views.every(v=>v.image.complete&&v.image.naturalWidth===512)')
   after=page.evaluate('({task:task.sample_id,points:points.length,documents:localDocuments.size,operator})');assert after['points']==0 and after['documents']==0 and not after['operator']
   png=ROOT/'evidence'/f'BROWSER_{p["slot"]}_{p["batch_id"]}_{t}.png';page.screenshot(path=str(png));checks.append(dict(package=p['package_id'],before=before,after=after,screenshot=str(png),file_loading=True,geographic_hints_available=False,actual_annotation_created=False));page.close()
  version=browser.version;browser.close()
 assert not errors and all(r['method']=='GET' and r['url'].startswith('file:') for r in requests)
 save(ROOT/f'BROWSER_QA_{t}.json',dict(at=now(),passed=True,browser=version,sandbox=True,checks=checks,javascript_errors=errors,requests=requests,actual_human_identity_entered=False,point_clicks=0,save_calls=0,GT_created=0))
 print('BROWSER_PASSED',len(checks),'actual local HTML packages; no human inputs',flush=True)

if __name__=='__main__':main()
