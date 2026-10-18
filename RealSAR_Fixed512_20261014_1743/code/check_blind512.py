"""Read-only actual browser verification. No identities or annotation exports."""
import sys,os,json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/sar_gt_browsers_verified')
from playwright.sync_api import sync_playwright
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
source=next((ROOT/'blind_release').glob('blind_annotation_*.html'))
errors=[];requests=[];downloads=[];loaded=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
 page=browser.new_page(viewport={'width':1600,'height':1100})
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.on('request',lambda r:requests.append(r.url))
 page.on('download',lambda d:downloads.append(d.suggested_filename))
 page.goto(source.as_uri(),wait_until='load');page.wait_for_function('task && !loading && doc')
 page.locator('#begin').click()
 assert '逐项如实填写' in page.locator('#message').inner_text()
 assert page.evaluate('operator==="" && localDocuments.size===0')
 page.evaluate("document.getElementById('identityDialog').close()")
 for sid in page.evaluate('BLIND_TASKS.map(t=>t.sample_id)'):
  page.select_option('#task',sid)
  page.wait_for_function('s=>!loading && task.sample_id===s',arg=sid)
  assert page.evaluate('views.every(v=>v.image.complete && v.image.naturalWidth===512 && v.image.naturalHeight===512 && v.canvas.width===512 && v.canvas.height===512)')
  assert page.evaluate('points.length===0 && doc.revision===0 && doc.gt_quality.eligible_checks===0 && assistance.length===0')
  assert not page.locator('#geo').is_visible() and not page.locator('#pseudo').is_visible()
  assert page.locator('#quickSave').is_disabled()
  assert '512×512' in page.locator('#metadata').text_content()
  assert page.evaluate('views.every(v=>{const xy=[511,511],s=C.pixelToScreen(...xy,v.camera),q=C.screenToPixel(...s,v.camera);return Math.hypot(q[0]-511,q[1]-511)<1e-9})')
  loaded.append(sid)
 page.reload();page.wait_for_function('task && !loading && doc')
 assert page.evaluate('operator==="" && localDocuments.size===0 && points.length===0')
 page.evaluate("document.getElementById('identityDialog').close()")
 page.screenshot(path=str(ROOT/f'blind_browser_readonly_{stamp}.png'))
 version=browser.version;browser.close()
assert sorted(loaded)==['C001','C002','C003'] and not errors and not downloads
assert all(url.startswith('file:') for url in requests)
result=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),passed=True,actual_browser=version,
 loaded_512_tasks=loaded,no_external_network=True,page_errors=errors,unknown_exposure_rejected=True,
 native_coordinate_roundtrip_checked=True,reload_empty_records=True,geography_and_algorithm_hints_hidden=True,
 annotation_downloads_created=0,actual_human_annotations_created=0,GT_created=0,port_started=False)
with (ROOT/f'blind_browser_check_{stamp}.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result,ensure_ascii=False,indent=2))
