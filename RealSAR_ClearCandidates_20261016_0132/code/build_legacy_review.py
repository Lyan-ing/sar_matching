"""Read-only gallery of the actual old Geo Assist inputs and saved user verdicts."""
import sys,json,csv,io,base64,math,html
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
OLD=BASE/'RealSAR_GeoAssist_20261012_1917';TOOL=BASE/'RealSAR_SingleExpert_20261015_2332'
sys.path.insert(0,str(TOOL/'code'))
from expert import stamp,now,save

def encoded(im):
 out=io.BytesIO();im.save(out,format='PNG',compress_level=6)
 return 'data:image/png;base64,'+base64.b64encode(out.getvalue()).decode()

def main():
 t=stamp();tasks=[json.loads(s) for s in (OLD/'tasks.jsonl').read_text().splitlines()]
 cache=json.loads((OLD/'display_cache.json').read_text());reviews={}
 for folder in (OLD/'assisted_triage').iterdir():
  if folder.is_dir():
   files=sorted(folder.glob('v*.json'))
   if files:reviews[folder.name]=json.loads(files[-1].read_text())
 references=list(csv.DictReader((OLD/'exports/assisted_review_snapshot_20261013_201035/last_view_positions.csv').open(encoding='utf-8-sig')))
 counts={k:sum(r['verdict']==k for r in reviews.values()) for k in ['A','B','C','D']}
 assert len(tasks)==18 and len(reviews)==13 and counts==dict(A=7,B=2,C=4,D=0)
 cards=[];summary=[];overview_count=detail_count=0
 for task in sorted(tasks,key=lambda r:r['priority']):
  sid=task['sample_id'];review=reviews.get(sid);verdict=review['verdict'] if review else '未筛选'
  angle='方位差未确认' if task['azimuth_delta_deg'] is None else f"方位差 {task['azimuth_delta_deg']:.3f}°"
  chunks=[f'<article data-verdict="{verdict}" data-reviewed="{str(bool(review)).lower()}"><h2>{task["priority"]}. {html.escape(task["site_name"])} · {angle} <span class="badge">{verdict}</span></h2>',f'<div class="sid">{html.escape(sid)}</div>']
  if review:
   chunks.append('<p class="reason">你当时的备注：'+html.escape(review['reason'])+'</p>')
   chunks.append('<p class="meta">记录时间 '+html.escape(review['at'])+' · 保存版本 v'+str(review['revision'])+'</p>')
  else:chunks.append('<p>这组在原18项队列内，没有找到你的辅助筛选保存记录。</p>')
  chunks.append('<h3>当时的原生影像总览</h3><div class="images">')
  for side in [0,1]:
   key=cache['task_view_index'][sid+':'+str(side)];asset=cache['assets'][key]
   with Image.open(asset['overview']) as source:
    im=source.convert('L');im.thumbnail((550,550),Image.Resampling.NEAREST)
   chunks.append(f'<figure><figcaption>{"左" if side==0 else "右"}图 · Native {asset["width"]}×{asset["height"]}</figcaption><img loading="lazy" alt="{html.escape(sid)} {side} 原生总览" src="{encoded(im)}"></figure>')
   overview_count+=1
  chunks.append('</div>')
  refs=[r for r in references if r['sample_id']==sid]
  if refs:
   chunks.append('<h3>恢复的旧局部浏览参考</h3><p class="meta">来自固定旧日志，可能含地理预测。只是查看位置，不是你确认的同名点；未写回任何标注。</p><div class="images">')
   for side in [0,1]:
    r=next((r for r in refs if int(r['side'])==side),None)
    if not r:chunks.append('<figure>此侧没有可恢复的旧浏览位置。</figure>');continue
    view=task['views'][side]
    assert r['acquisition_id']==view['acquisition_id'] and Path(r['native_tif']).resolve()==Path(view['native_tif']).resolve()
    x,y=float(r['x']),float(r['y']);span=int(r['span']);key=cache['task_view_index'][sid+':'+str(side)];asset=cache['assets'][key];array=np.load(asset['array'],mmap_mode='r')
    x0,y0=max(0,math.floor(x-span/2)),max(0,math.floor(y-span/2));x1,y1=min(array.shape[1],math.ceil(x+span/2)),min(array.shape[0],math.ceil(y+span/2))
    im=Image.fromarray(np.asarray(array[y0:y1,x0:x1]))
    chunks.append(f'<figure><figcaption>{"左" if side==0 else "右"} · 旧浏览窗口 {x1-x0}×{y1-y0} 原生像素</figcaption><img class="detail" loading="lazy" alt="旧浏览参考，非GT" src="{encoded(im)}"><p class="meta">原生中心 ({x:.2f}, {y:.2f})</p></figure>')
    detail_count+=1
   chunks.append('</div>')
  chunks.append('<details><summary>原始采集身份与来源</summary><pre>'+html.escape(json.dumps(task['views'],ensure_ascii=False,indent=2))+'</pre></details></article>')
  cards.append(''.join(chunks));summary.append(dict(sample_id=sid,verdict=verdict,source_review_revision=review['revision'] if review else None,views=task['views'],restored_viewport_sides=len(refs),GT=False))
 style='''body{margin:0;background:#17202b;color:#edf2f7;font:16px system-ui,sans-serif}header{position:sticky;top:0;background:#223247;padding:12px 20px;z-index:2}header h1{font-size:21px;margin:0 0 7px}a{color:#8bd4ff}select,input,button{font:inherit;padding:6px;margin-right:8px;background:#30435c;color:white;border:1px solid #8b9db3;border-radius:4px}main{max-width:1160px;margin:auto;padding:12px}article{border:1px solid #60768f;border-radius:8px;padding:16px;margin:18px 0;background:#1d2b3b}h2{font-size:20px}h3{font-size:16px}.images{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0;text-align:center}figcaption{margin:5px 0;color:#c3d7eb}img{display:block;width:100%;height:auto;image-rendering:pixelated}.detail{max-width:512px;margin:auto}.badge{background:#496a85;padding:3px 9px;border-radius:5px}.reason{font-size:18px;color:#ffe19a}.meta,.sid{font-size:13px;color:#c1cddd}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:11px}details{margin-top:12px}[hidden]{display:none!important}@media(max-width:700px){.images{grid-template-columns:1fr}}'''
 script='''const select=document.getElementById('filter'),search=document.getElementById('search');function update(){let count=0;for(const card of document.querySelectorAll('article')){const v=select.value,match=(v==='all'||(v==='reviewed'?card.dataset.reviewed==='true':card.dataset.verdict===v))&&card.textContent.toLowerCase().includes(search.value.toLowerCase());card.hidden=!match;count+=match;}document.getElementById('count').textContent=count+' 组';}select.onchange=search.oninput=update;update();'''
 document='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>以前的Geo Assist旧图 · 只读回看</title><style>'+style+'</style><header><h1>以前的Geo Assist旧图 · 只读回看</h1><div>原队列18组；你保存过13组：A=7、B=2、C=4。这里没有巴黎和赛道新图，不保存任何操作。</div><p><select id="filter"><option value="reviewed">你筛过的13组</option><option value="A">A类7组</option><option value="B">B类2组</option><option value="C">C类4组</option><option value="all">完整旧队列18组</option><option value="未筛选">未保存判断的5组</option></select><input id="search" placeholder="搜索站点 / 旧样本ID"><span id="count"></span> · <a href="/">返回当前标点工具</a></p></header><main>'+''.join(cards)+'</main><script>'+script+'</script></html>'
 folder=ROOT/'legacy_review';folder.mkdir(exist_ok=True);path=folder/f'GEOASSIST_OLD_IMAGES_{t}.html'
 assert not path.exists();path.write_text(document)
 config=json.loads(sorted(TOOL.glob('CONFIG_*.json'))[-1].read_text());config.update(legacy_review_html=str(path),legacy_review_created_at=now())
 config_path=TOOL/f'CONFIG_{t}.json';save(config_path,config)
 save(folder/f'LEGACY_GALLERY_INDEX_{t}.json',dict(at=now(),old_queue=18,old_reviewed=13,verdict_counts=counts,overview_images=overview_count,recovered_viewport_images=detail_count,read_only=True,HTML=str(path),source_cache=str(OLD/'display_cache.json'),source_navigation=str(OLD/'exports/assisted_review_snapshot_20261013_201035/last_view_positions.csv'),tasks=summary,old_records_modified=0,GT_created=0))
 print(json.dumps(dict(HTML=str(path),bytes=path.stat().st_size,reviewed=13,all_old_pairs=18,overview_images=overview_count,viewport_images=detail_count,config=str(config_path)),ensure_ascii=False))

if __name__=='__main__':main()
