"""Convert authoritative Markdown locally with existing sandboxed Chromium."""
import sys,os,re,html,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
sys.dont_write_bytecode=True
sys.path.insert(0,'/tmp/sar_gt_browser_deps')
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'training_detailed_report_20261012_1708.md'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def inline(text):
 text=html.escape(text);text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text);text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
 text=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',r'<a href="\2"><img src="\2" alt="\1"></a>',text)
 return re.sub(r'(?<!!)\[([^]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',text)
def render(md):
 pieces=[];in_code=False;in_table=False;header=False
 for line in md.splitlines():
  if line.startswith('```'):
   if in_table:pieces.append('</tbody></table></div>');in_table=False
   pieces.append('</code></pre>' if in_code else '<pre><code>');in_code=not in_code;continue
  if in_code:pieces.append(html.escape(line)+'\n');continue
  if line.startswith('|'):
   cells=[c.strip() for c in line.strip('|').split('|')]
   if all(re.fullmatch(r':?-+:?',c) for c in cells):continue
   if not in_table:pieces.append('<div class="table"><table>');in_table=True;header=True
   if header:pieces.append('<thead><tr>'+''.join('<th>'+inline(c)+'</th>' for c in cells)+'</tr></thead><tbody>');header=False
   else:pieces.append('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in cells)+'</tr>')
   continue
  if in_table:pieces.append('</tbody></table></div>');in_table=False
  m=re.match(r'^(#{1,6}) (.*)',line)
  if m:pieces.append(f'<h{len(m[1])}>{inline(m[2])}</h{len(m[1])}>')
  elif line.strip():pieces.append('<p>'+inline(line)+'</p>')
 return ''.join(pieces)
def main():
 start=datetime.now(ZoneInfo('Asia/Shanghai'));stamp=start.strftime('%Y%m%d_%H%M');out=ROOT/('pdf_export_'+stamp);out.mkdir(exist_ok=False)
 pdf=ROOT/('training_detailed_report_'+stamp+'.pdf');assert not pdf.exists()
 md=SOURCE.read_text();source_hash=digest(SOURCE);page_html=out/'pdf_layout.html'
 css='''@page {size:A4 landscape;margin:15mm 12mm 16mm}
 body{font-family:"Noto Sans CJK SC","DejaVu Sans",sans-serif;font-size:10pt;line-height:1.55;color:#161616;margin:0}
 h1{font-size:20pt;line-height:1.35}h2{font-size:15pt;margin-top:8mm;border-bottom:.3mm solid #ccc;break-before:page;break-after:avoid}
 h3{font-size:12pt;margin-top:5mm;break-after:avoid}p{margin:3mm 0;orphans:3;widows:3;overflow-wrap:anywhere}
 table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:8pt;line-height:1.45;margin:3mm 0}
 th,td{border:.2mm solid #bbb;padding:2mm;vertical-align:top;overflow-wrap:anywhere;word-break:break-word}th{background:#edf1f5}
 thead{display:table-header-group}tr{break-inside:avoid}img{display:block;width:100%;height:auto;max-height:160mm;object-fit:contain;break-inside:avoid;margin:3mm 0}
 pre{font-family:"Noto Sans Mono CJK SC",monospace;font-size:8pt;line-height:1.5;background:#f3f4f6;padding:3mm;white-space:pre-wrap;overflow-wrap:anywhere;word-break:break-word}
 code{font-family:"Noto Sans Mono CJK SC",monospace;font-size:.9em;overflow-wrap:anywhere}a{color:#174e78;text-decoration:none}strong{font-weight:700}
 '''
 page_html.write_text('<!doctype html><html lang="zh"><head><meta charset="utf-8"><base href="'+ROOT.as_uri()+'/"><title>SAR-SuperPoint训练详细报告</title><style>'+css+'</style></head><body>'+render(md)+'</body></html>')
 with sync_playwright() as p:
  browser=p.chromium.launch(headless=True,chromium_sandbox=True)
  page=browser.new_page(viewport=dict(width=1600,height=1100));page.goto(page_html.as_uri(),wait_until='load',timeout=60000);page.emulate_media(media='print');page.evaluate('document.fonts.ready');page.wait_for_function('Array.from(document.images).every(im=>im.complete&&im.naturalWidth>0)')
  images=page.locator('img').count();assert images==10
  headings=page.locator('h2').all_text_contents();assert len(headings)==18
  assert page.evaluate('(family)=>document.fonts.check("12px " + JSON.stringify(family))','Noto Sans CJK SC')
  page.pdf(path=str(pdf),prefer_css_page_size=True,print_background=True,display_header_footer=True,header_template='<span></span>',footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#555"><span class="pageNumber"></span> / <span class="totalPages"></span></div>')
  version=browser.version;browser.close()
 assert SOURCE.read_text()==md and digest(SOURCE)==source_hash
 info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);textfile=out/'extracted_text.txt';subprocess.run(['pdftotext','-layout',str(pdf),str(textfile)],check=True);extracted=textfile.read_text()
 assert '核心结论与完成边界' in extracted and '报告附件与重生成' in extracted
 assert '89.6159%' in extracted and 'projective_pole_or_near_pole' in extracted
 manifest=dict(source=str(SOURCE),source_sha256=source_hash,pdf=str(pdf),pdf_sha256=digest(pdf),started_at=start.isoformat(),completed_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),browser=version,browser_sandbox=True,environment='match',page_format='A4 landscape',images=images,sections=headings,pdfinfo=info,Chinese_text_extraction_passed=True,original_markdown_unchanged=True,original_html_unchanged=True)
 (out/'conversion_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(ROOT/'logs'/('pdf_conversion_'+stamp+'.log')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 print(json.dumps(manifest,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
