"""Build judgement-only/assisted galleries, identity ledger and report data."""
import sys
sys.dont_write_bytecode=True
import csv, hashlib, html, json, platform, shutil, re
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image, ImageDraw
import audit as a

def parse_utc(value):
    # Python 3.10 accepts microseconds; retain original nanosecond strings in ledger.
    return datetime.fromisoformat(re.sub(r'(\.\d{6})\d+',r'\1',value).replace('Z','+00:00'))

def identity(row, side, profile):
    f=Path(row['provider_metadata_'+side]);m=json.loads(f.read_text());tags=profile.get('tags',{})
    indexed=row[side+'_acquisition_id']; embedded=None;terrain=None
    if 'collects' in m:
        x=m['collects'][0];physical=x['id'];dt=x['startAtUTC'];orbit=x['satelliteTrack'];look=x['observationDirection'];az=x['angleAzimuthDegrees'];inc=x['angleIncidenceDegrees'];stac=None
    elif 'properties' in m:
        x=m['properties'];physical=x.get('umbra:collect_id',m['id']);dt=x.get('start_datetime',x['datetime']);orbit=x.get('sat:orbit_state');look=x.get('sar:observation_direction');az=x.get('view:azimuth');inc=x.get('view:incidence_angle');stac=m['id']
    else:
        physical=m['acquisition_id'];dt=m['datetime'];orbit=m['orbit_direction'];look=m['observation_direction'];az=m.get('azimuth_degrees');inc=m.get('incidence_degrees');stac=None
    if row['sensor']=='Capella' and 'TIFFTAG_IMAGEDESCRIPTION' in tags:
        original=json.loads(tags['TIFFTAG_IMAGEDESCRIPTION']);embedded=original['collect']['collect_id'];terrain=original['collect']['image'].get('terrain_models')
    elif row['sensor']=='Umbra':embedded=tags.get('COLLECT_ID')
    elif row['sensor']=='Sentinel-1':embedded=tags.get('SCENES')
    return {'sample_id':row['pair_id']+'__'+row['chip_id'],'side':side,'indexed_acquisition_id':indexed,'physical_collect_id':physical,'stac_item_id':stac,'embedded_collect_id':embedded,'identity_verified_against_embedded':None if embedded is None else embedded==physical==indexed,
        'provider_start_utc':dt,'indexed_utc':row[side+'_utc'],'datetime_agrees':parse_utc(dt)==parse_utc(row[side+'_utc']),'date_comparison_precision':'microsecond; original nanosecond strings retained',
        'orbit':orbit,'observation_direction':look,'azimuth_deg':az,'incidence_deg':inc,'terrain_models':terrain,'license':row[side+'_license'],'license_url':row[side+'_license_url'],
        'asset_url':row[side+'_asset_url'],'metadata_file':str(f),'metadata_sha256':hashlib.sha256(f.read_bytes()).hexdigest()}

def linear_versions(row, folder, record):
    arrays=[];valid=[]
    for s in ['reference','moving']:
        with rasterio.open(row[s+'_native_geotiff']) as d:arrays.append(d.read(1));valid.append(d.read_masks(1)>0)
    joint=np.quantile(np.concatenate([x[v] for x,v in zip(arrays,valid)]),[.01,.99]).tolist()
    ranges=[np.quantile(x[v],[.01,.99]).tolist() for x,v in zip(arrays,valid)]
    ims=[]
    for i,(x,v,r) in enumerate(zip(arrays,valid,ranges)):
        im=a.stretch(x,v,r);Image.fromarray(im).save(folder/f'linear_independent{i}.png');ims.append(im)
        Image.fromarray(a.stretch(x,v,joint)).save(folder/f'linear_joint{i}.png')
    a.panel(ims,['reference native DN/power','moving native DN/power'],record['sample_id']+' | LINEAR independent 1-99% display, not recovered information',folder/'linear_independent_pair.png')
    a.dump(folder/'linear_display.json',{'joint_native_limits':joint,'independent_native_limits':ranges,'quantiles':[.01,.99],'zero':'retained according to original saved masks','denoising':False,'radiometric_calibration':False,'display_only':True})
    if 'context' in record['checks'][0]:
        ims=[]
        for i in range(2):
            c=record['checks'][i]['context']
            with rasterio.open(c['path']) as d:x=d.read(1);v=d.read_masks(1)>0
            r=np.quantile(x[v],[.01,.99]);im=a.stretch(x,v,r);Image.fromarray(im).save(folder/f'context_linear{i}.png');ims.append(im)
        a.panel(ims,['reference native1024','moving native1024'],record['sample_id']+' | TRUE native context; LINEAR independent contrast; NO assistance points',folder/'context_independent_pair.png',1024)

def page(row, rec, sel, folder):
    sid=rec['sample_id'];kind='异向' if row['look_pair_category']=='different_look_verified' else ('相近/同向' if row['look_pair_category']=='repeat_pass_same_look' else '方向未确定')
    caption=f"{row['sensor']} · {row['site_name']} · {kind} · 有效地理重叠 R/M {rec['effective_overlap'][0]['fraction_of_valid_pixels']:.2%}/{rec['effective_overlap'][1]['fraction_of_valid_pixels']:.2%} · 地面边长 R/M {rec['metric_geometry'][0]['coverage_edge_lengths_m'][0]:.1f}/{rec['metric_geometry'][1]['coverage_edge_lengths_m'][0]:.1f}m"
    protocols=[('original','原始冻结灰度 PNG'),('linear_independent','原生TIF：独立线性1–99%显示'),('robust_joint','原生TIF：共同DN dB 1–99%显示'),('robust_independent','原生TIF：独立DN dB 1–99%显示')]
    default='linear_independent' if row['sensor']=='Umbra' else 'robust_independent'
    opts=''.join(f'<option value="{p}" {"selected" if p==default else ""}>{t}</option>' for p,t in protocols)
    views=''.join(f'<div class="view"><span>{"Reference 左" if i==0 else "Moving 右"} — 原生坐标；无对应点</span><div class="scroll" tabindex="0"><img id="im{i}" src="{default}{i}.png" alt="{sid} 原生视图{i}" width="512" height="512"></div></div>' for i in range(2))
    ctx=''
    if (folder/'context_independent_pair.png').exists():
        ctx='<details><summary>更大真实原生1024上下文（不含地理点/范围辅助）</summary><a target="_blank" href="context_independent_pair.png"><img class="panel" src="context_independent_pair.png" alt="真实1024上下文"></a></details>'
    else:ctx='<p>没有该位置本地保留的大于512原生场景；未下载、未填充伪造上下文。建议后续补取实际原生窗口。</p>'
    assisted=''
    for path,text in [('footprints.png','地图覆盖多边形与共同地理区域'),('common_coverage.png','共同有效地理覆盖（并非双图物理可见性）'),('geo_compare.png','右图地理映射到左图，仅诊断'),('geo_overlay.png','地理粗叠加，仅诊断'),('local_zoom.png','地理辅助局部ROI放大：不是人工对应点'),('context_footprints.png','1024中两张512裁片的地理位置'),('context_geo_compare.png','1024地理映射对照')]:
        if (folder/path).exists():assisted+=f'<details><summary>{text}</summary><a href="{path}" target="_blank"><img class="panel" src="{path}" alt="{html.escape(text)}"></a></details>'
    note=sel.get('selection_reason','原5对专项核查；可评价性待人工确认')
    content=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{sid}</title><style>
    body{{background:#111820;color:#edf1f7;margin:20px;font:16px system-ui}}a{{color:#69c5ff}}button,select,input{{font-size:16px;padding:6px}}.bar{{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}}.views{{display:flex;gap:12px}}.view{{width:50%;min-width:0}}.scroll{{height:68vh;overflow:auto;background:#000}}.scroll img{{display:block;image-rendering:pixelated;max-width:none}}.panel{{max-width:100%;height:auto}}details{{border:1px solid #536375;padding:12px;margin:15px 0}}summary{{cursor:pointer}}.warning{{background:#3c2e10;padding:12px}}@media(max-width:850px){{.views{{display:block}}.view{{width:100%}}}}
    </style><body><a href="../../index.html">返回图集</a><h1>{sid}</h1><p>{html.escape(caption)}</p><p>{html.escape(note)}</p><p>工程元数据检查通过；共同静态物理结构仍待人工确认，GT合格：否。</p><p class="warning">此页面只用于候选判读，不是独立标注任务。地理辅助诊断在折叠区；查看后必须在后续独立标注中如实记录暴露。原有标注网页未修改。</p>
    <div class="bar"><label>显示 <select id="display">{opts}</select></label><label>缩放 <input id="scale" type="range" min="50" max="400" step="25" value="100"></label><button id="reset">重置视图</button><label><input id="sync" type="checkbox"> 同步滚动（可取消分别调整）</label><span id="zoom">100%</span></div>
    <div class="views">{views}</div>{ctx}<h2>地理辅助工程核查（点击后记录暴露；不是GT）</h2>{assisted}<p><a href="audit.json">完整审计元数据</a> · <a href="search_regions.json">局部ROI规则</a></p>
    <p>Umbra / Copyright Capella Space: CC BY 4.0. Sentinel-1: Created using Indigo Atlas Analysis-Ready Sentinel-1 Backscatter Imagery https://www.indigoag.com, 2026；沿用原研究/非营利许可条件。</p>
    <script>
    const sc=[...document.querySelectorAll('.scroll')];let busy=false;
    document.querySelector('#display').onchange=e=>{{for(let i=0;i<2;i++)document.querySelector('#im'+i).src=e.target.value+i+'.png'}};
    document.querySelector('#scale').oninput=e=>{{let s=+e.target.value;for(let i=0;i<2;i++){{let im=document.querySelector('#im'+i);im.width=im.height=512*s/100}}document.querySelector('#zoom').textContent=s+'%'}};
    document.querySelector('#reset').onclick=()=>{{document.querySelector('#scale').value=100;document.querySelector('#scale').dispatchEvent(new Event('input'));sc.forEach(x=>{{x.scrollLeft=x.scrollTop=0}})}};
    sc.forEach((x,i)=>x.addEventListener('scroll',()=>{{if(busy||!document.querySelector('#sync').checked)return;busy=true;sc[1-i].scrollLeft=x.scrollLeft;sc[1-i].scrollTop=x.scrollTop;requestAnimationFrame(()=>busy=false)}}));
    document.querySelectorAll('details').forEach(x=>x.addEventListener('toggle',()=>{{if(x.open){{let log=JSON.parse(localStorage.getItem('sar_candidate_preview_exposure')||'[]');log.push({{sample:'{sid}',panel:x.querySelector('summary').textContent,viewed_at:new Date().toISOString(),not_annotation_store:true}});localStorage.setItem('sar_candidate_preview_exposure',JSON.stringify(log))}}}}));
    </script></body></html>'''
    (folder/'index.html').write_text(content)

def main():
    a.require_unsealed()
    selected=json.loads((a.ROOT/'selection.json').read_text())['candidates'];sel={x['sample_id']:x for x in selected}
    rows={r['pair_id']+'__'+r['chip_id']:r for r in [json.loads(l) for l in (a.DATA/'dataset_index.jsonl').read_text().splitlines()]}
    identities=[];links=[];prepared=[]
    for folder in sorted((a.ROOT/'pairs').iterdir()):
        rec=json.loads((folder/'audit.json').read_text());row=rows[folder.name]
        for s,z in zip(['reference','moving'],rec['checks']):identities.append(identity(row,s,z['source_profile']))
        linear_versions(row,folder,rec);page(row,rec,sel.get(folder.name,{}),folder)
        if folder.name in sel:
            q=dict(sel[folder.name],sensor=row['sensor'],site_name=row['site_name'],pair_id=row['pair_id'],component=row['component'],split=row['split'],look_pair_category=row['look_pair_category'],reference_acquisition_id=row['reference_acquisition_id'],moving_acquisition_id=row['moving_acquisition_id'],effective_overlap=[x['fraction_of_valid_pixels'] for x in rec['effective_overlap']],ground_edges_m=[x['coverage_edge_lengths_m'] for x in rec['metric_geometry']],valid_fraction=rec['valid_fraction'],context_available='context' in rec['checks'][0],preview_page=f'pairs/{folder.name}/index.html',source_row=row)
            prepared.append(q)
    a.dump(a.ROOT/'audit'/'identity_ledger.json',identities)
    a.dump(a.ROOT/'candidates.json',prepared)
    # Different pairs sharing acquisitions/site are dependent, explicitly enumerated.
    stats={'crop_pairs':len(prepared),'distinct_acquisition_pairs':len({tuple(sorted([r['reference_acquisition_id'],r['moving_acquisition_id']])) for r in prepared}),
           'distinct_acquisitions':len({r[s+'_acquisition_id'] for r in prepared for s in ['reference','moving']}),'sites':len({r['site_name'] for r in prepared}),'components':len({r['component'] for r in prepared}),
           'look_counts':dict(Counter(r['look_pair_category'] for r in prepared)),'context_available':sum(r['context_available'] for r in prepared),'provisional_categories':dict(Counter(r['provisional_category'] for r in prepared)),'GT_qualified':0}
    a.dump(a.ROOT/'audit'/'candidate_statistics.json',stats)
    with (a.ROOT/'candidates.csv').open('w') as f:
        fields=['sample_id','pair_id','site_name','sensor','component','split','look_pair_category','reference_acquisition_id','moving_acquisition_id','provisional_category','effective_overlap','valid_fraction','context_available','selection_reason','static_physical_correspondence','GT_qualified'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in prepared)
    cards=[]
    for r in prepared:
        sid=r['sample_id'];p='linear_independent_pair.png' if r['sensor']=='Umbra' else 'robust_independent_pair.png'
        cards.append(f'<article><h2><a href="pairs/{sid}/index.html">{sid}</a></h2><p>{html.escape(r["site_name"])} · {r["look_pair_category"]} · 地理覆盖R/M {r["effective_overlap"][0]:.1%}/{r["effective_overlap"][1]:.1%} · 类别{r["provisional_category"]}候选，待人工确认</p><a href="pairs/{sid}/index.html"><img loading="lazy" src="pairs/{sid}/{p}" alt="{sid} 双图"></a><p>{html.escape(r["selection_reason"])}</p></article>')
    old=''.join(f'<li><a href="pairs/{sid}/index.html">{sid}</a></li>' for sid in a.FIVE)
    index=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实SAR配对核查与候选图集</title><style>body{{background:#111820;color:#edf1f7;font:17px system-ui;margin:24px}}a{{color:#69c5ff}}article{{border:1px solid #637387;margin:20px 0;padding:15px}}img{{width:100%;max-width:1600px;height:auto}}.notice{{background:#3c2e10;padding:15px}}</style><body><h1>真实SAR候选判读图集</h1><p class="notice">{stats['crop_pairs']}对候选，{stats['distinct_acquisition_pairs']}个原采集对，{stats['sites']}个站点。{stats['look_counts'].get('different_look_verified',0)}对异向、{stats['look_counts'].get('repeat_pass_same_look',0)}对相近/同向。没有任何新GT资格。候选按地理覆盖与静态场景判读筛选，未读取匹配器结果。点击样本进入原生双图，调整显示与缩放；地图辅助图独立折叠，查看会在本浏览器本地记录暴露（不是原标注系统审计记录）。</p><p>建议先核验 Capella_0006__chip_001、S1SF_EXTENSION_000__chip_000、Capella_0005__chip_001；异向尝试 Umbra_0056__chip_002、Umbra_0068__chip_001 的局部结构。找不到可靠同一地物时标记不可评价，不凑点数。</p><h2>原5对专项核查</h2><ul>{old}</ul><p><a href="candidates.csv">候选来源与覆盖CSV</a> · <a href="audit/geography191.csv">191对覆盖审计CSV</a> · <a href="kingston/index.html">Kingston专项检查</a> · <a href="candidate_atlas.png">高清总览PNG</a></p>{''.join(cards)}</body></html>'''
    (a.ROOT/'index.html').write_text(index)
    atlas=Image.new('RGB',(2048,len(prepared)*650+65),'#171b22');draw=ImageDraw.Draw(atlas);draw.text((15,15),'Native SAR screening candidates | No independent GT | full-resolution pages: index.html',fill='white',font=a.FONT)
    for j,r in enumerate(prepared):
        sid=r['sample_id'];f=a.ROOT/'pairs'/sid;p='linear_independent_pair.png' if r['sensor']=='Umbra' else 'robust_independent_pair.png';im=Image.open(f/p).convert('RGB');im=im.resize((1024,im.height))
        y=65+j*650;atlas.paste(im,(0,y+42));draw.text((10,y+5),f"{sid} | {r['look_pair_category']} | geo overlap {r['effective_overlap'][0]:.1%}/{r['effective_overlap'][1]:.1%}",fill='white',font=a.FONT)
        # Original native PNG next to enhanced panels; never substitute enhanced for original.
        raw=Image.open(f/'original_pair.png').convert('RGB');atlas.paste(raw,(1024,y+42));draw.text((1028,y+5),f"Original PNG | ground edge R/M {r['ground_edges_m'][0][0]:.1f}/{r['ground_edges_m'][1][0]:.1f} m | pending human",fill='white',font=a.FONT)
    atlas.save(a.ROOT/'candidate_atlas.png')
    print('DELIVERY DATA',stats,flush=True)

if __name__=='__main__':main()
