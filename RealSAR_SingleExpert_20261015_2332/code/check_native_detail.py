"""Read-only browser check: intermediate zoom must not magnify overview pixels."""
from expert import *
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/sar_gt_browsers_verified')
from playwright.sync_api import sync_playwright

def main():
 runtime=json.loads(max(ROOT.glob('SERVICE_*.json'),key=lambda p:p.stat().st_mtime_ns).read_text())
 expected_before='--before' in sys.argv;posts=[];errors=[];checks=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
  page=browser.new_page(viewport=dict(width=1700,height=1100))
  page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
  page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(runtime['url']+'?readonly='+runtime['readonly_verification_token'],wait_until='networkidle')
  page.wait_for_function('tasks.length===1033 && readonly')
  for sid in ['N512_58a771ed441247','N512_ad56da258cc235']:
   page.evaluate('(sid)=>openTask(sid,"native")',sid)
   page.wait_for_function('scene && !loading')
   dimensions=page.evaluate('scene.frames.map(f=>[f.width,f.height])')
   for scale in [.4,1.0]:
    page.evaluate('(scale)=>{views[0].camera={scale,tx:256-scene.frames[0].width*scale/2,ty:256-scene.frames[0].height*scale/2};cameraChanged(0);}',scale)
    if expected_before:
     page.wait_for_timeout(500)
     assert page.evaluate('views[0].tile===null') if scale==.4 else True
     checks.append(dict(sid=sid,scale=scale,detail_loaded=page.evaluate('!!views[0].tile')))
    else:
     page.wait_for_function('views[0].tile && views[0].tile.complete===true',timeout=30000)
     detail=page.evaluate('''(()=>{const t=views[0].tile;return {requested:t.requested_box,boxes:t.patches.map(p=>p.box),exact:t.patches.every(p=>p.im.naturalWidth===p.box[2]&&p.im.naturalHeight===p.box[3]),label:document.getElementById("detail0").textContent};})()''')
     assert detail['exact'];assert all(max(b[2:])<=1024 for b in detail['boxes'])
     assert sum(b[2]*b[3] for b in detail['boxes'])==detail['requested'][2]*detail['requested'][3]
     assert len(detail['boxes'])>1 if scale==.4 else len(detail['boxes'])==1
     consistency=page.evaluate('''(()=>{const t=views[0].tile,b=t.requested_box,a=document.createElement('canvas');a.width=b[2];a.height=b[3];const c=a.getContext('2d');for(const p of t.patches)c.drawImage(p.im,p.box[0]-b[0],p.box[1]-b[1]);const expected=stretch(a).getContext('2d');let error=0;for(const p of t.patches){const ref=expected.getImageData(p.box[0]-b[0],p.box[1]-b[1],p.box[2],p.box[3]).data,actual=p.stretched.getContext('2d').getImageData(0,0,p.box[2],p.box[3]).data;for(let j=0;j<ref.length;j+=4)error=Math.max(error,Math.abs(ref[j]-actual[j]));}return error;})()''')
     assert consistency==0, 'Native pieces must share one window contrast range'
     detail['window_contrast_maximum_difference']=consistency
     checks.append(dict(sid=sid,dimensions=dimensions,scale=scale,detail=detail))
   if expected_before:break
  assert not posts and not errors
  browser.close()
 result=dict(at=now(),phase='before' if expected_before else 'final' if '--final' in sys.argv else 'after',checks=checks,POST_count=len(posts),javascript_errors=errors,human_points_created=0,GT_created=0)
 save(ROOT/'evidence'/f'DISPLAY_DETAIL_{result["phase"]}_{stamp()}.json',result)
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
