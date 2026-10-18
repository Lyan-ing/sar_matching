"""Read-only browser verification of the original-image annotation entry."""
from expert import *
sys.path.insert(0, '/tmp/sar_gt_browser_deps')
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', '/tmp/sar_gt_browsers_verified')
from playwright.sync_api import sync_playwright


def main():
 runtime = json.loads(max(ROOT.glob('SERVICE_*.json'), key=lambda p: p.stat().st_mtime_ns).read_text())
 errors, posts = [], []
 with sync_playwright() as pw:
  browser = pw.chromium.launch(headless=True, chromium_sandbox=True, args=['--disable-gpu'])
  page = browser.new_page(viewport=dict(width=1700, height=1100))
  page.on('pageerror', lambda e: errors.append(str(e)))
  page.on('request', lambda r: posts.append(r.url) if r.method == 'POST' else None)
  page.goto(runtime['url']+'?readonly='+runtime['readonly_verification_token'], wait_until='networkidle')
  page.wait_for_function('tasks.length===1033 && filtered.length===88')
  assert page.evaluate('readonly && new Set(filtered.map(t=>t.relation_id)).size===88 && filtered.every(t=>t.native_context_available)')
  page.click('.card')
  page.wait_for_function('scene && !loading && mode==="native"')
  assert page.evaluate('scene.frames.every(f=>f.width>512 && f.height>512)')
  page.click('[data-fit="0"]')
  page.click('[data-native-pixel="0"]')
  page.wait_for_function('views[0].camera.scale===1 && views[0].tile!==null')
  assert page.evaluate('views[0].tile.im.naturalWidth===views[0].tile.box[2] && views[0].tile.im.naturalHeight===views[0].tile.box[3]')
  dimensions = page.evaluate('scene.frames.map(f=>({width:f.width,height:f.height,origin:f.origin_xy}))')
  error = page.evaluate('''(()=>{let err=0;for(const v of views){const r=v.canvas.getBoundingClientRect();for(const p of [[0,0],[301.25,703.75],[1023,1023]]){const s=C.pixelToScreen(...p,v.camera),client=[r.left+s[0]*r.width/512,r.top+s[1]*r.height/512],canvas=C.clientToCanvas(...client,r,512,512),back=C.screenToPixel(...canvas,v.camera);err=Math.max(err,Math.hypot(back[0]-p[0],back[1]-p[1]));}}return err;})()''')
  assert error < 1e-8
  page.reload(wait_until='networkidle')
  page.wait_for_function('scene && !loading && mode==="native" && readonly')
  native_url = page.url
  page.click('#browse')
  page.select_option('#focus','all')
  assert page.evaluate('filtered.length===1033')
  page.click('#pilotStart')
  page.wait_for_function('scene && !loading && mode==="native"')
  assert page.evaluate('scene.frames.every(f=>f.width>512 && f.height>512)')
  assert not posts, 'Read-only check must never submit annotation records'
  assert not errors, errors
  browser.close()
 result=dict(at=now(),passed=True,original_image_relations=88,all_existing_pairs=1033,
             default_native=True,actual_native_dimensions=dimensions,native_1to1_tile=True,
             reload_restores_native_mode=True,pilot_opens_native=True,
             maximum_CSS_coordinate_roundtrip_px=error,POST_count=len(posts),
             human_clicks_generated=0,GT_generated=0,javascript_errors=errors,native_URL=native_url)
 path=ROOT/'evidence'/f'NATIVE_ENTRY_CHECK_{stamp()}.json'
 save(path,result)
 print(json.dumps(result,ensure_ascii=False))
 print(path)


if __name__ == '__main__':
 main()
