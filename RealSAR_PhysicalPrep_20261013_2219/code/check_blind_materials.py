"""Actual browser read-only check; no human identity, click records, downloads or GT."""
import sys,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
def run():
 source=next((ROOT/'blind_release').glob('blind_annotation_*.html'));errors=[];requests=[];loaded=[]
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
  page=browser.new_page(viewport={'width':1700,'height':1200})
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.on('request',lambda r:requests.append(r.url))
  page.goto(source.as_uri(),wait_until='load');page.wait_for_function('task && !loading && doc',timeout=60000)
  page.locator('#begin').click()
  assert '逐项如实填写' in page.locator('#message').inner_text()
  assert page.evaluate('operator')==''
  page.evaluate("document.getElementById('identityDialog').close()")
  for sid in page.evaluate('BLIND_TASKS.map(t=>t.sample_id)'):
   page.select_option('#task',sid);page.wait_for_function('s=>!loading && task.sample_id===s',arg=sid,timeout=60000)
   assert page.evaluate('views.every((v,i)=>v.image.complete && v.image.naturalWidth===task["width"+i] && v.canvas.width===task["width"+i] && v.canvas.height===task["height"+i])')
   assert page.evaluate('points.length===0 && doc.revision===0 && doc.gt_quality.eligible_point_ids.length===0 && assistance.length===0')
   assert not page.locator('#geo').is_visible() and not page.locator('#pseudo').is_visible()
   assert page.locator('#quickSave').is_disabled()
   assert 'Native左' in page.locator('#metadata').text_content()
   assert page.evaluate('views.every((v,i)=>{const xy=[task["width"+i]-1,task["height"+i]-1],s=C.pixelToScreen(...xy,v.camera),q=C.screenToPixel(...s,v.camera);return Math.hypot(q[0]-xy[0],q[1]-xy[1])<1e-9})')
   loaded.append(sid)
  page.reload();page.wait_for_function('task && !loading && doc',timeout=60000)
  assert page.evaluate('points.length===0 && localDocuments.size===0 && operator===""')
  page.evaluate("document.getElementById('identityDialog').close()")
  stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
  page.screenshot(path=str(ROOT/f'blind_materials_readonly_browser_{stamp}.png'))
  version=browser.version;browser.close()
 assert len(loaded)==4 and not errors
 assert all(s.startswith('file:') for s in requests)
 result=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),passed=True,loaded_blind_tasks=loaded,full_native_dimensions_passed=True,
             no_geo_or_algorithm_hints_displayed=True,empty_initial_records=True,no_default_independent_identity=True,
             unknown_exposure_declaration_rejected=True,pixel_center_coordinate_convention_passed=True,
             no_external_requests=True,page_errors=errors,browser=version,real_human_annotations_created=0,GT_created=0,GPU=False,ports_started=[])
 with (ROOT/f'blind_browser_check_{stamp}.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':run()
