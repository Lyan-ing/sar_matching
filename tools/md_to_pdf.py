"""Continuous A4 portrait PDF, same stem as Markdown; preserve sources and previous PDF."""
import argparse,hashlib,importlib.util,json,re,sys,os,subprocess,shutil
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from PIL import Image
BASE=Path(__file__).resolve().parents[1]
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/sar_gt_browsers_verified')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('markdown',type=Path);a=parser.parse_args();source=a.markdown.resolve();assert source.is_file()
 spec=importlib.util.spec_from_file_location('existing_pdf_renderer',BASE/'SAR_SP_LG_TrainingReview_20261012_1651/code/export_pdf.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 started=datetime.now(ZoneInfo('Asia/Shanghai'));ts=started.strftime('%Y%m%d_%H%M');pdf=source.with_suffix('.pdf');work=source.parent/f'pdf_export_{ts}'
 if work.exists():
  for version in range(1,10000):
   work=source.parent/f'pdf_export_v{version}_{ts}'
   if not work.exists():break
 work.mkdir();previous=None
 if pdf.exists():previous=work/f'previous_pdf_{ts}.pdf';shutil.copy2(pdf,previous)
 raw=source.read_text();source_sha=hashlib.sha256(source.read_bytes()).hexdigest();body=m.render(raw)
 # Only the PDF rendering copies are compressed. Original scientific figures remain untouched.
 figures=[]
 for i,src in enumerate(dict.fromkeys(re.findall(r'<img src="([^"]+)"',body))):
  path=(source.parent/src).resolve();assert path.is_file(),src
  cached=work/f'figure_{i:03d}_{ts}.jpg'
  with Image.open(path) as original:
   rgba=original.convert('RGBA');white=Image.new('RGBA',rgba.size,'white');white.alpha_composite(rgba);im=white.convert('RGB');im.thumbnail((1800,2400),Image.Resampling.LANCZOS);im.save(cached,'JPEG',quality=94,subsampling=0,optimize=True)
  body=body.replace('src="'+src+'"','src="'+cached.as_uri()+'"');figures.append(dict(source=str(path),rendering_copy=str(cached)))
 body=re.sub(r'<p>\s*<a href="[^"]+">(<img[^>]+>)</a>\s*</p>',r'<figure>\1</figure>',body)
 css='''@page{size:A4 portrait;margin:13mm 13mm 15mm}
 body{margin:0;font-family:"Noto Sans CJK SC","DejaVu Sans",sans-serif;font-size:10pt;line-height:1.45;color:#161616}
 h1{font-size:18pt;line-height:1.25;margin:0 0 8pt;break-after:avoid}h2{font-size:13pt;margin:12pt 0 5pt;break-before:auto;break-after:avoid}h3{font-size:11pt;margin:9pt 0 4pt;break-after:avoid}
 p{margin:4pt 0;overflow-wrap:anywhere;orphans:2;widows:2}.table{margin:5pt 0}table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:7pt;line-height:1.25}th,td{border:.2mm solid #bbb;padding:1mm .7mm;overflow-wrap:anywhere;vertical-align:top}thead{display:table-header-group}tr{break-inside:avoid}
 figure{display:block;margin:5pt 0;break-inside:avoid}img{display:block;max-width:100%;max-height:210mm;width:auto;height:auto;margin:0 auto}
 pre{font-size:8pt;white-space:pre-wrap;overflow-wrap:anywhere;margin:5pt 0}a{color:#174e78;text-decoration:none}'''
 layout=work/f'pdf_layout_{ts}.html';layout.write_text('<!doctype html><html lang="zh"><head><meta charset="utf-8"><base href="'+source.parent.as_uri()+'/"><style>'+css+'</style></head><body>'+body+'</body></html>')
 raw_pdf=work/f'continuous_raw_{ts}.pdf'
 with m.sync_playwright() as pw:
  browser=pw.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu']);page=browser.new_page();page.goto(layout.as_uri(),wait_until='load',timeout=120000);page.emulate_media(media='print');page.evaluate('document.fonts.ready');page.wait_for_function('Array.from(document.images).every(im=>im.complete&&im.naturalWidth>0)',timeout=120000)
  images=page.locator('img').count();assert images==len(figures);print(f'Printing continuous A4 portrait document, {images} images',flush=True)
  page.pdf(path=str(raw_pdf),prefer_css_page_size=True,print_background=True);version=browser.version;browser.close()
 rebuilt=work/f'validated_pdf_{ts}.pdf'
 subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=pdfwrite','-dCompatibilityLevel=1.7','-dPDFSETTINGS=/ebook','-dDetectDuplicateImages=true','-dCompressFonts=true','-dSubsetFonts=true','-dDownsampleColorImages=true','-dColorImageResolution=200','-dAutoFilterColorImages=false','-dColorImageFilter=/DCTEncode','-dJPEGQ=94','-dAutoRotatePages=/None','-sOutputFile='+str(rebuilt),str(raw_pdf)],check=True)
 info=subprocess.check_output(['pdfinfo',str(rebuilt)],text=True);match=re.search(r'Page size:\s*([0-9.]+) x ([0-9.]+)',info);assert match and float(match[1])<float(match[2]),info
 subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=nullpage',str(rebuilt)],check=True)
 assert rebuilt.stat().st_size>0 and hashlib.sha256(source.read_bytes()).hexdigest()==source_sha
 os.replace(rebuilt,pdf)
 record=dict(source=str(source),source_sha256=source_sha,pdf=str(pdf),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),started=started.isoformat(),completed=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),environment='match',browser=version,browser_sandbox=True,images=images,figures=figures,source_unchanged=True,previous_pdf=str(previous) if previous else None,same_name_as_markdown=True,paper='A4 portrait',continuous_layout=True,forced_heading_pagebreaks=False,segmented_pdf_merge=False,pdfinfo=info,file_size_bytes=pdf.stat().st_size)
 with (work/f'conversion_manifest_{ts}.json').open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:v for k,v in record.items() if k!='figures'},ensure_ascii=False))
if __name__=='__main__':main()
