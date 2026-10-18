"""Actual Chromium, read-only service session, zero SAR annotation clicks/saves."""
from expert import *
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/sar_gt_browsers_verified')
from playwright.sync_api import sync_playwright

def main():
 runtime=json.loads(sorted(ROOT.glob('SERVICE_*.json'),key=lambda p:p.stat().st_mtime_ns)[-1].read_text());url=runtime['url'];token=runtime['readonly_verification_token'];t=stamp();errors=[];posts=[];checks={}
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu']);page=browser.new_page(viewport=dict(width=1600,height=1050));page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:posts.append(urlparse(r.url).path) if r.method=='POST' else None)
  page.goto(url+'?readonly='+token,wait_until='networkidle');page.wait_for_function('tasks.length===1033 && document.querySelectorAll(".card").length===24')
  assert page.locator('#count').inner_text().startswith('30组');page.wait_for_function('Array.from(document.querySelectorAll(".card img")).every(im=>im.complete&&im.naturalWidth>0)')
  assert page.locator('#identityDialog').count()==0
  page.select_option('#focus','all');assert page.locator('#count').inner_text().startswith('1033组')
  page.select_option('#focus','B');assert page.locator('#count').inner_text().startswith('40组')
  page.select_option('#focus','Val');assert page.locator('#count').inner_text().startswith('286组')
  checks['1033_gallery_30pilot_40B_Val_filters_and_actual_thumbnails']=True
  gallery=ROOT/'evidence'/f'BROWSER_GALLERY_{t}.png';page.screenshot(path=str(gallery))
  page.select_option('#focus','pilot');page.click('#pilotStart');page.wait_for_function('scene && !loading && scene.frames.every(f=>f.width===512 && f.height===512)')
  before=page.evaluate('({sid:scene.task.sample_id,revision:doc.revision,points:doc.points.length,xy,readonly,mode})');assert before['points']==0 and before['revision']==0 and before['xy']==[None,None] and before['readonly']
  original=page.url;page.reload(wait_until='networkidle');page.wait_for_function('scene && !loading');assert page.evaluate('doc.revision===0 && doc.points.length===0 && readonly')
  checks['direct_task_URL_reload_and_no_identity_registration']=True
  page.select_option('#viewMode','native');page.wait_for_function('scene && !loading && mode==="native"');native=page.evaluate('({frames:scene.frames.map(f=>[f.width,f.height]),points:doc.points.length,xy,bank:scene.bank.map(a=>a.length)})');assert all(x[0]>512 for x in native['frames']) and native['points']==0
  canvas=page.locator('#c0').bounding_box();page.mouse.move(canvas['x']+canvas['width']/2,canvas['y']+canvas['height']/2)
  for _ in range(11):page.mouse.wheel(0,-120);page.wait_for_timeout(35)
  page.wait_for_function('views[0].tile && Math.max(...views[0].tile.box.slice(2))<=1024')
  assert page.evaluate('views[0].tile.im.naturalWidth===views[0].tile.box[2] && views[0].tile.im.naturalHeight===views[0].tile.box[3]')
  page.mouse.down(button='right');page.mouse.move(canvas['x']+canvas['width']/2+20,canvas['y']+canvas['height']/2+10);page.mouse.up(button='right');page.check('#local0');page.fill('#contrast0','1.7');page.wait_for_timeout(250)
  assert page.evaluate('doc.points.length===0 && xy.every(p=>p===null) && dirtyVersion===0')
  checks['large_Native_zoom_pan_local_contrast_and_exact_pixel_tiles_without_clicks']=True
  # Pure mathematical probes through the actual CSS rectangle, never clicked/stored.
  css=page.evaluate('''(()=>{let error=0;for(const v of views){const r=v.canvas.getBoundingClientRect();for(const p of [[0,0],[127.25,311.75],[511,511]]){const q=Coordinates.pixelToScreen(...p,v.camera),x=r.left+q[0]*r.width/512,y=r.top+q[1]*r.height/512,s=Coordinates.clientToCanvas(x,y,r,512,512),back=Coordinates.screenToPixel(...s,v.camera);error=Math.max(error,Math.hypot(back[0]-p[0],back[1]-p[1]));}}return error;})()''');assert css<1e-8
  checks['actual_CSS_zoom_coordinates_roundtrip']=True
  screenshot=ROOT/'evidence'/f'BROWSER_NATIVE_{t}.png';page.screenshot(path=str(screenshot))
  page.select_option('#viewMode','repeat');page.wait_for_function('scene && !loading && mode==="repeat"');repeat=page.evaluate('({hasOriginal:JSON.stringify(scene).includes("repeat_baselines"),bank:scene.bank,derived:scene.inherited_points,quality:scene.quality,points:doc.points.length})');assert not repeat['hasOriginal'] and repeat['bank']==[[],[]] and repeat['derived']==[] and repeat['quality'] is None and repeat['points']==0
  checks['optional_repeat_hides_original_coordinates_bank_and_quality']=True
  blocked=page.evaluate('''async()=>{const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});return r.status}''');assert blocked==403
  assert posts==['/api/save'];health=page.evaluate('async()=>await (await fetch("/api/health")).json()');assert health['original_correspondences']==0
  checks['readonly_write_rejected_and_production_correspondences_remain0']=True
  assert not errors;version=browser.version;browser.close()
 save(ROOT/'evidence'/f'BROWSER_CHECK_{t}.json',dict(at=now(),passed=True,checks=checks,browser=version,sandbox=True,initial=before,native=native,css_max_roundtrip_px=css,javascript_errors=errors,only_POST='deliberately denied empty payload, no human coordinates',production_human_clicks=0,production_GT_created=0,screenshots=[str(gallery),str(screenshot)],first_task_url=original))
 print('BROWSER_PASSED',len(checks),'groups; no physical clicks or GT',flush=True)

if __name__=='__main__':
 from urllib.parse import urlparse
 main()
