"""Compatible local-font PDF export using the project's existing Chromium renderer."""
import os, json, re, subprocess, importlib.util, argparse
from pathlib import Path
import render_ch4 as r
os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH','/tmp/sar_gt_browsers_verified')
spec=importlib.util.spec_from_file_location('existing_renderer',r.BASE/'SAR_SP_LG_TrainingReview_20261012_1651/code/export_pdf.py')
existing=importlib.util.module_from_spec(spec);spec.loader.exec_module(existing)


def font_css():
    styles=[]
    for name,file,weight,style in [('FandolFang','FandolFang-Regular.otf',400,'normal'),
        ('Times New Roman','Times.TTF',400,'normal'),('Times New Roman','Timesbd.TTF',700,'normal'),
        ('Times New Roman','Timesi.TTF',400,'italic'),('Times New Roman','Timesbi.TTF',700,'italic')]:
        styles.append("@font-face{font-family:'"+name+"';src:url('"+(r.FONT_DIR/file).as_uri()+"');font-weight:"+str(weight)+";font-style:"+style+";}")
    return ''.join(styles)


def svg_body(p):
    s=Path(p).read_text();return s[s.index('<svg'):]


def main(markdown=None):
    t=r.stamp();kind='report' if markdown else 'figures';work=r.OUT/'meta'/f'font_pdf_export_{kind}_{t}';work.mkdir()
    records=[]
    with existing.sync_playwright() as pw:
        browser=pw.chromium.launch(headless=True,chromium_sandbox=True,args=['--disable-gpu'])
        page=browser.new_page()
        def emit(html, target, name, expected_pages):
            source=work/f'{name}_{t}.html';source.write_text(html)
            page.goto(source.as_uri(),wait_until='load');page.emulate_media(media='print')
            page.evaluate('document.fonts.ready')
            page.wait_for_function('Array.from(document.images).every(im=>im.complete&&im.naturalWidth>0)')
            fonts=page.evaluate('Array.from(document.fonts).map(f=>({family:f.family,status:f.status,weight:f.weight,style:f.style}))')
            assert any(f['family']=='FandolFang' and f['status']=='loaded' for f in fonts)
            candidate=work/f'validated_{name}_{t}.pdf'
            page.pdf(path=str(candidate),prefer_css_page_size=True,print_background=True)
            info=subprocess.check_output(['pdfinfo',str(candidate)],text=True)
            if expected_pages is not None:assert re.search(r'Pages:\s+'+str(expected_pages)+r'\b',info),info
            font_info=subprocess.run(['pdffonts',str(candidate)],capture_output=True,text=True,check=True)
            assert not font_info.stderr,font_info.stderr
            assert 'TimesNewRomanPSMT' in font_info.stdout
            assert not any(s in font_info.stdout for s in ['DroidSans','Liberation','NotoSerif'])
            subprocess.run(['gs','-q','-dSAFER','-dBATCH','-dNOPAUSE','-sDEVICE=nullpage',str(candidate)],check=True,capture_output=True)
            previous=work/f'previous_{target.name}'
            if target.exists():target.rename(previous)
            candidate.rename(target)
            records.append(dict(pdf=str(target),previous_pdf=str(previous),html=str(source),fonts=fonts,
                final_sha256=r.sha(target),font_table=font_info.stdout,page_info=info))
            print('Exported compatible font PDF:',target.name,flush=True)

        if markdown is None:
            meta=json.loads(sorted((r.OUT/'meta').glob('EXPORTS_*.json'))[-1].read_text())
            for ex in meta['exports']:
                css=font_css()+f"@page{{size:160mm {ex['physical_height_mm']}mm;margin:0}}html,body{{margin:0;padding:0}}svg{{display:block}}"
                html='<!doctype html><html><head><meta charset="utf-8"><style>'+css+'</style></head><body>'+svg_body(ex['svg'])+'</body></html>'
                emit(html,Path(ex['pdf']),ex['key'],1)
            css=font_css()+"@page{size:A4 portrait;margin:0}html,body{margin:0;padding:0}.sheet{width:210mm;height:297mm;position:relative;break-after:page;overflow:hidden}.sheet:last-child{break-after:auto}.sheet svg{display:block}.label{position:absolute;text-align:center;width:210mm;font-family:'Times New Roman','FandolFang';}"
            sheets=[]
            for ex in meta['exports']:
                y=(297-ex['physical_height_mm'])/2
                sheets.append('<section class="sheet"><div class="label" style="top:8mm;font-size:12pt">160 mm 通栏检查 · '+ex['key']+'</div><div style="position:absolute;left:25mm;top:'+str(y)+'mm;width:160mm">'+svg_body(ex['svg'])+'</div><div class="label" style="bottom:9mm;font-size:10pt">160 mm · 按实际大小打印</div></section>')
            emit('<!doctype html><html><head><meta charset="utf-8"><style>'+css+'</style></head><body>'+''.join(sheets)+'</body></html>',Path(meta['proof_pdf']),'all_160mm_A4',len(meta['exports']))
        else:
            # Reuse the exact continuous A4 layout produced by tools/md_to_pdf.py.
            md=Path(markdown).resolve()
            candidates=sorted(md.parent.glob('pdf_export_*/pdf_layout_*.html'))
            assert candidates
            source=candidates[-1].read_text()
            source=source.replace('</head>',"<style>"+font_css()+"body,body *{font-family:'Times New Roman','FandolFang' !important;}</style></head>")
            emit(source,md.with_suffix('.pdf'),'delivery_document',None)
        browser.close()
    r.save(work/f'font_conversion_manifest_{t}.json',dict(timestamp=t,records=records,font_files_packaged=False,
        fonts_directory=str(r.FONT_DIR),CJK_font='FandolFang (provided FangSong OTF)',Latin_font='Times New Roman',
        CJK_PDF_representation='embedded vector glyphs; Unicode map retained',all_pdf_fonts_valid=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--markdown');a=p.parse_args();main(a.markdown)
