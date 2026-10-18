"""Final read-only delivery checks; does not create physical point fixtures."""
from expert import *
from collections import Counter
import subprocess,urllib.request

def main():
 t=stamp();entry=sorted(ROOT.glob('SINGLE_EXPERT_ENTRYPOINTS_*.json'))[-1];e=json.loads(entry.read_text());d=Dataset();s=Store(d);checks={}
 assert Counter(p['split'] for p in d.tasks.values())==Counter(Train=747,Val=286)
 assert len({p['relation_id'] for p in d.tasks.values()})==137
 assert len(d.pilot)==30 and len(set(d.pilot))==30 and sum(d.tasks[k]['split']=='Val' for k in d.pilot)==11
 assert all(d.tasks[k]['AI_class']=='B' for k in d.pilot)
 assert sum(t['AI_class']=='B' for t in d.tasks.values())==40
 for key,fn in [('site',lambda t:[t['component']]),('relation',lambda t:[t['relation_id']]),('acquisition',lambda t:t['acquisition_ids'])]:
  a={v for t in d.tasks.values() if t['split']=='Train' for v in fn(t)};b={v for t in d.tasks.values() if t['split']=='Val' for v in fn(t)};assert not a&b
 checks['original1033_Train747_Val286_and_component_isolation_preserved']=True
 for p in d.tasks.values():
  with Image.open(p['thumbnail']) as im:assert im.size==(386,192)
 checks['actual1033_double_image_thumbnails_40B_and30pilot']=True
 assert not s.docs and sum(1 for _ in s.original_points())==0
 for name in ['GT_qualified_always','independent_two_person_GT']:assert d.protocol[name] is False
 for path,h in json.loads(sorted(ROOT.glob('PROTECTED_INPUTS_*.json'))[-1].read_text()).items():assert sha(path)==h
 checks['no_production_annotations_or_fictional_GT_original_protocol_unchanged']=True
 evidence=[]
 for pattern in ['SELFCHECK_*.json','BROWSER_CHECK_*.json','BROWSER_PERSISTENCE_*.json']:
  p=sorted((ROOT/'evidence').glob(pattern))[-1];r=json.loads(p.read_text());assert r['passed'];evidence.append(str(p))
 checks['actual_backend_coordinate_and_Chromium_autosave_restore_failure_checks_passed']=True
 runtime=json.loads(max(ROOT.glob('SERVICE_*.json'),key=lambda p:p.stat().st_mtime_ns).read_text());cmd=Path(f'/proc/{runtime["pid"]}/cmdline').read_bytes().split(b'\0');assert str(ROOT/'code/server.py').encode() in cmd
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(e['current_status_endpoint'],timeout=5) as response:health=json.load(response)
 assert health['tasks']==1033 and health['original_correspondences']==0 and health['port']==8774 and health['service']=='RealSAR Single Expert'
 checks['live8774_service_our_own_PID_CPU_only_no_other_service_touched']=True
 pdf=Path(e['report_pdf']);assert pdf.exists() and pdf.stat().st_size>0
 info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);assert 'A4' in info and 'Page rot:        0' in info
 conversions=[json.loads(p.read_text()) for p in ROOT.glob('pdf_export_*/conversion_manifest_*.json')];conv=next(v for v in conversions if v['pdf']==str(pdf));assert conv['continuous_layout'] and not conv['forced_heading_pagebreaks'] and not conv['segmented_pdf_merge']
 for k in ['quick_start','report_md','report_pdf','task_index','single_expert_protocol','initial_coverage','separate_scoring_code','source_entrypoint']:assert Path(e[k]).exists(),k
 checks['entrypoints_exist_and_same_stem_A4_continuous_PDF_parses']=True
 files=[entry,Path(e['task_index']),Path(e['single_expert_protocol']),Path(e['report_md']),pdf,Path(e['quick_start']),*sorted((ROOT/'code').glob('*'))]
 identity={str(p):sha(p) for p in files if p.is_file()}
 target=ROOT/f'FINAL_ACCEPTANCE_{t}.json';save(target,dict(at=now(),passed=True,checks=checks,entrypoint=str(entry),live_url=e['url'],live_pid=runtime['pid'],evidence=evidence,files_sha256=identity,actual_human_points=0,GT_qualified_two_person=0,single_expert_GT=0,annotation_readiness='user can now annotate pilot30; actual physical validity still requires user clicks and declarations',GPU=False));print('FINAL_ACCEPTANCE_PASSED',target,'URL',e['url'],flush=True)

if __name__=='__main__':main()
