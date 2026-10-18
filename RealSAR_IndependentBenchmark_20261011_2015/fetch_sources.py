"""Read official repository metadata only. Never install or execute remote code."""
import json, hashlib, urllib.request, concurrent.futures
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent
REPOS=['Parskatt/RoMa','LJY-RS/RIFT2-multimodal-matching-rotation','yeyuanxin110/CFOG','canyagmur/XPoint','cvg/DeepLSD','danini/magsac','XDyaoshi/CroR-OSIR-main']
def read(url):
    req=urllib.request.Request(url,headers={'User-Agent':'SAR-research-metadata-check'})
    with urllib.request.urlopen(req,timeout=30) as r: return r.read()
def inspect(repo):
    record={'repository':repo,'official_URL':'https://github.com/'+repo,'at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'files':[],'errors':[]}
    out=ROOT/'sources'/repo.replace('/','__');out.mkdir(exist_ok=False)
    try:
        meta=json.loads(read('https://api.github.com/repos/'+repo));branch=meta['default_branch']
        commit=json.loads(read('https://api.github.com/repos/'+repo+'/commits/'+branch))['sha']
        record.update(commit=commit,default_branch=branch,license_API=meta.get('license'))
        listing=json.loads(read('https://api.github.com/repos/'+repo+'/contents?ref='+commit))
        names=[r['name'] for r in listing if r['type']=='file' and (r['name'].lower().startswith(('readme','license')) or r['name'] in ['pyproject.toml','requirements.txt','quickstart_install.sh'])]
        record['root_files']=[r['name'] for r in listing]
        for name in names:
            url=f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
            try:
                data=read(url);(out/name).write_bytes(data);record['files'].append(dict(name=name,url=url,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
            except Exception as e:record['errors'].append({'URL':url,'error':str(e)})
    except Exception as e:record['errors'].append({'error':str(e)})
    (out/'metadata.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record
if __name__=='__main__':
    (ROOT/'sources').mkdir(exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: records=list(pool.map(inspect,REPOS))
    (ROOT/'sources/repositories.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{'repo':r['repository'],'commit':r.get('commit'),'files':len(r['files']),'errors':r['errors']} for r in records],indent=2))
