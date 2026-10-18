"""End-to-end autosave/reopen/error recovery on an isolated empty-draft store.

No SAR image clicks, physical coordinates, people or GT are generated.
"""
from expert import *
from server import make_server
import tempfile
sys.path.insert(0,'/tmp/sar_gt_browser_deps');os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/sar_gt_browsers_verified')
from playwright.sync_api import sync_playwright

def main():
 t=stamp();checks={};errors=[];prod=Store(Dataset());assert not prod.docs
 with tempfile.TemporaryDirectory(prefix='sar_expert_browser_emptydraft_') as temporary:
  server=make_server(0,store_root=Path(temporary)/'only_empty_drafts');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url=f'http://127.0.0.1:{server.server_port}/';sid=server.data.pilot[0]
  try:
   with sync_playwright() as pw:
    browser=pw.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu']);context=browser.new_context(viewport=dict(width=1600,height=1050));page=context.new_page();page.on('pageerror',lambda e:errors.append(str(e)))
    page.goto(url+'?task='+sid,wait_until='networkidle');page.wait_for_function('scene && !loading')
    text='自动化空草稿落盘恢复检查；不是人工结构标注，未点击SAR图像。';page.fill('#reason',text);page.wait_for_function('doc.revision>=1 && dirtyVersion===ackVersion')
    assert page.evaluate('doc.points.length===0 && xy.every(p=>p===null)');first=page.evaluate('doc.revision');page.reload(wait_until='networkidle');page.wait_for_function('scene && !loading');assert page.input_value('#reason')==text
    checks['actual_browser_autosave_empty_draft_and_reload_from_disk']=True
    # Fail a write on purpose; UI must retain the draft and never claim success.
    page.route('**/api/save',lambda route:route.fulfill(status=500,content_type='application/json',body=json.dumps(dict(error='deliberate isolated empty-draft save-failure test'))))
    retry_text=text+' 保存失败后的恢复。';page.fill('#reason',retry_text);page.wait_for_function('document.getElementById("saved").textContent.startsWith("未保存")')
    assert page.evaluate('localStorage.getItem(cacheKey())!==null && doc.points.length===0')
    checks['failed_save_not_reported_success_browser_backup_retained']=True
    page.unroute('**/api/save');page.reload(wait_until='networkidle');page.wait_for_function('scene && !loading && doc.revision>=2 && dirtyVersion===ackVersion');assert page.input_value('#reason')==retry_text
    page.close();fresh=context.new_page();fresh.goto(url+'?task='+sid,wait_until='networkidle');fresh.wait_for_function('scene && !loading');assert fresh.input_value('#reason')==retry_text and fresh.evaluate('doc.points.length===0 && xy.every(p=>p===null)')
    checks['reopen_new_page_and_recover_after_failed_save']=True
    # Changes in progress are not observations and trigger no GT export.
    assert len(server.store.docs)==1 and server.store.get(sid,'patch')['revision']>=2 and not server.store.get(sid,'patch')['points'];assert not errors
    version=browser.version;browser.close()
  finally:server.shutdown();server.server_close();thread.join(timeout=2)
 assert not Store(Dataset()).docs
 checks['temporary_service_closed_production_annotation_store_untouched']=True
 out=ROOT/'evidence'/f'BROWSER_PERSISTENCE_{t}.json';save(out,dict(at=now(),passed=True,checks=checks,browser=version,javascript_errors=errors,isolation='temporary directory removed; real SAR images read only',first_revision=first,production_human_clicks=0,production_GT=0,temporary_correspondence_points=0));print('BROWSER_PERSISTENCE_PASSED',out,flush=True)

if __name__=='__main__':main()
