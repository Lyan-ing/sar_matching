"""Existing UI/coordinate code, real new SAR images, no production annotation writes."""
import sys,os,json,tempfile,threading,secrets
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];TOOL=ROOT.parent/'RealSAR_SingleExpert_20261015_2332'
sys.path.insert(0,str(TOOL/'code'));sys.path.insert(0,'/tmp/sar_gt_browser_deps')
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/sar_gt_browsers_verified')
from expert import Dataset,Store,stamp,now,save
from server import make_server
from playwright.sync_api import sync_playwright

def main():
 data=Dataset();production=Store(data);evidence={k:d['content_sha256'] for k,d in production.docs.items()}
 errors=[];posts=[];frames=[]
 with tempfile.TemporaryDirectory(prefix='clear-native-check-') as tmp:
  token=secrets.token_hex(32);server=make_server(0,token,store_root=tmp)
  thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
    page=browser.new_page(viewport=dict(width=1700,height=1100));page.on('pageerror',lambda e:errors.append(str(e)))
    page.on('request',lambda r:posts.append(r.url) if r.method=='POST' else None)
    page.goto(f'http://127.0.0.1:{server.server_port}/?readonly='+token,wait_until='networkidle')
    page.wait_for_function('tasks.length===1035 && filtered.length===3 && readonly')
    labels=page.locator('#cards').inner_text();assert '同向0°对照' in labels and '巴黎' in labels and 'Thermal' in labels
    page.click('#pilotStart');page.wait_for_function('scene && !loading && scene.task.native_only')
    assert page.evaluate('mode==="native" && scene.frames.every(f=>f.width===4096) && !scene.quality.primary.passed && !scene.quality.local.passed')
    assert page.locator('#viewMode option[value="patch"]').is_disabled()
    frames.append(page.evaluate('({sample:scene.task.sample_id,frames:scene.frames.map(f=>[f.width,f.height]),split:scene.task.split})'))
    page.click('[data-native-pixel="0"]');page.wait_for_function('views[0].tile && views[0].tile.complete')
    page.reload(wait_until='networkidle');page.wait_for_function('scene && !loading && scene.task.native_only && mode==="native" && readonly')
    page.click('#next');page.wait_for_function('scene && !loading && scene.task.site.includes("Thermal")')
    frames.append(page.evaluate('({sample:scene.task.sample_id,frames:scene.frames.map(f=>[f.width,f.height]),split:scene.task.split})'))
    page.click('#next');page.wait_for_function('scene && !loading && scene.task.same_direction_control')
    assert page.evaluate('scene.task.angle===0 && !scene.task.native_only')
    page.click('#browse');page.select_option('#focus','all');assert page.evaluate('filtered.length===1035 && tasks.filter(t=>!t.native_only).length===1033')
    page.select_option('#split','Train');assert page.evaluate('filtered.length===747')
    page.select_option('#split','Val');assert page.evaluate('filtered.length===286')
    assert not posts and not errors
    browser.close()
   # Storage smoke check uses an empty draft only, outside production, never invented SAR points.
   task=next(t for t in data.tasks.values() if t.get('native_only'))
   empty=server.store.get(task['sample_id'],'native')
   payload=dict(sample_id=task['sample_id'],mode='native',frame_ids=task['frame_ids'],expected_revision=0,client_save_id='automated-empty-storage-check',points=[],draft={},task_status='pending',decision_reason='')
   doc=server.store.commit(payload);loaded=Store(data,tmp).get(task['sample_id'],'native')
   assert loaded['content_sha256']==doc['content_sha256'] and loaded['points']==[]
  finally:server.shutdown();server.server_close();thread.join()
 assert {k:d['content_sha256'] for k,d in Store(data).docs.items()}==evidence
 result=dict(at=now(),passed=True,real_new_pairs_viewed=frames,shortlist=3,new_native_pairs=2,same_direction_controls=1,original_512_pairs=1033,Train=747,Val=286,production_observation_hashes_unchanged=True,empty_draft_reload_in_temporary_store=True,new_native_local_full_qualification_disabled=True,POST_count=len(posts),human_clicks_generated=0,GT_created=0,javascript_errors=errors)
 save(ROOT/f'SHORTLIST_BROWSER_CHECK_{stamp()}.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
