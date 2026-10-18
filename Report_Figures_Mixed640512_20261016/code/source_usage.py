"""Read-only formal-pair source census. No synthesis or sampling functions called."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import json,csv,hashlib,argparse
import numpy as np
import cv2
from PIL import Image
cv2.setNumThreads(1)
BASE=Path('/home/yl/matching/sar_matching')
DATA=BASE/'SyntheticSAR_Mixed640512_Multicondition_20261016_2114'
OUT=BASE/'Report_Figures_Mixed640512_20261016'
ENTRY=DATA/'DATASET_ENTRYPOINTS_MIXED_20261016_2155.json'
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
    with Path(p).open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2)
def write_csv(p,rr):
    keys=list(dict.fromkeys(k for r in rr for k in r))
    with Path(p).open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rr:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()})
def inputs():
    e=json.loads(ENTRY.read_text());rr=[r for p in e['manifests'].values() for r in rows(p)]
    audit={r['source_path']:r for r in rows(e['source_audit'])}
    cfg=json.loads((BASE/e['config']).read_text());ss={r['image_path']:r for r in rows(cfg['source_pool'])}
    return e,rr,audit,ss
def original_format(p):
    with Image.open(p) as im:mode=im.mode;fmt=im.format
    a=cv2.imread(p,cv2.IMREAD_UNCHANGED);assert a is not None
    equal=True if a.ndim==2 else bool(np.array_equal(a[...,0],a[...,1]) and np.array_equal(a[...,1],a[...,2]))
    return dict(path=p,mode=mode,format=fmt,dtype=str(a.dtype),channels=1 if a.ndim==2 else a.shape[-1],equal_RGB_channels=equal,width=a.shape[1],height=a.shape[0])
def inventory():
    return {str(p.relative_to(DATA)):(p.stat().st_size,p.stat().st_mtime_ns) for p in (DATA/'pairs').rglob('*') if p.is_file()}
def run(check_only=False):
    e,rr,audit,ss=inputs();assert len(rr)==6000
    formal_hashes={p:sha(p) for p in e['manifests'].values()};before=inventory()
    details=[];mixed=[];bad=[];actual_unknown=0
    for r in rr:
        original=json.loads(Path(r['raw_generation_record']).read_text())
        for key in ['source_path','source_members','visible_source_members','source_component_id','gt_path','image_A','image_B']:
            assert r[key]==original[key],(r['pair_id'],key)
        mm=r['source_members'];used=r['visible_source_members'];assert used and set(used)<=set(mm) and all(p in audit and p in ss for p in mm)
        cats=sorted({audit[p]['dataset'] for p in used});allcats=sorted({audit[p]['dataset'] for p in mm})
        parents=sorted({(ss[p]['dataset'],str(ss[p]['source_group'])) for p in mm});components=sorted({ss[p]['known_component'] for p in mm});splits={ss[p]['split'] for p in mm}
        if len(allcats)>1:mixed.append(r['pair_id'])
        if len(parents)>1 or components!=[r['source_component_id']] or splits!={r['split']}:bad.append(r['pair_id'])
        # The executed reader consumes ONLY declared members; checking the complete
        # domain also rules out a hidden category in bilinear seam support.
        assert len(allcats)==1 and cats==allcats
        identity_known=all(ss[p].get('independent_raw_acquisition_identity_known',False) for p in mm)
        actual_unknown+=not identity_known
        details.append(dict(pair_id=r['pair_id'],split=r['split'],at_least_categories=cats,exclusive_primary_source=audit[r['source_path']]['dataset'],
            width=r['width'],overlap_bin=r['overlap_bin'],source_component_id=r['source_component_id'],source_parent=parents,
            visible_published_images=len(used),declared_domain_images=len(mm),multiple_SSDD_tiles=sum(audit[p]['dataset']=='ssdd' for p in used)>1,
            all_declared_members_same_parent=True,actual_acquisition_identity_fully_known=identity_known,source_members=mm,visible_source_members=used))
    counts=[]
    for cat in ['hrsid','background','ssdd']:
        for split in ['train','val','all']:
            subset=[r for r,d in zip(rr,details) if cat in d['at_least_categories'] and (split=='all' or r['split']==split)]
            raw={p for r in subset for p in r['visible_source_members'] if audit[p]['dataset']==cat}
            counts.append(dict(source=cat,split=split,pairs=len(subset),views=2*len(subset),pairs_640=sum(r['width']==640 for r in subset),pairs_512=sum(r['width']==512 for r in subset),
                overlap_zero=sum(r['overlap_bin']=='zero' for r in subset),overlap_10_30=sum(r['overlap_bin']=='10_30' for r in subset),overlap_30_50=sum(r['overlap_bin']=='30_50' for r in subset),
                overlap_50_70=sum(r['overlap_bin']=='50_70' for r in subset),overlap_70_90=sum(r['overlap_bin']=='70_90' for r in subset),overlap_90_100=sum(r['overlap_bin']=='90_100' for r in subset),
                distinct_visible_published_files=len(raw),multiple_SSDD_published_tiles=sum(sum(audit[p]['dataset']=='ssdd' for p in r['visible_source_members'])>1 for r in subset),
                counting_rule='at least one visible member; full declared domain and original per-pair record corroborated'))
    if check_only:
        print(json.dumps(dict(formal_pairs=len(rr),source_usage=counts,cross_category_pairs=mixed,cross_parent_or_component_pairs=bad,unknown_actual_acquisition_pairs=actual_unknown),ensure_ascii=False));return
    t=stamp();OUT.mkdir(exist_ok=True);(OUT/'meta').mkdir(exist_ok=True)
    used={p for r in rr for p in r['visible_source_members']};formats=[]
    with ProcessPoolExecutor(max_workers=4) as ex:formats=list(ex.map(original_format,sorted(used),chunksize=8))
    assert before==inventory() and all(sha(p)==h for p,h in formal_hashes.items())
    usage=OUT/f'SOURCE_USAGE_FOR_REPORT_{t}.csv';write_csv(usage,counts)
    write_csv(OUT/f'meta/PAIR_SOURCE_TRACE_{t}.csv',details)
    write_csv(OUT/f'meta/USED_ORIGINAL_FORMATS_{t}.csv',formats)
    cats={c:{split:next(x for x in counts if x['source']==c and x['split']==split) for split in ['train','val','all']} for c in ['hrsid','background','ssdd']}
    fmt=Counter((audit[f['path']]['dataset'],f['mode'],f['format'],f['dtype'],f['channels'],f['equal_RGB_channels']) for f in formats)
    summary=dict(timestamp=t,pairs=6000,source_usage=counts,cross_category_pairs=mixed,cross_parent_or_component_pairs=bad,unknown_actual_acquisition_pairs=actual_unknown,
        raw_format_counts=[dict(dataset=k[0],mode=k[1],format=k[2],dtype=k[3],channels=k[4],equal_RGB_channels=k[5],files=v) for k,v in fmt.items()],
        manifest_hashes=formal_hashes,pair_files_unchanged=len(before),metadata_inventory_before=before)
    save(OUT/f'meta/SOURCE_PROVENANCE_{t}.json',summary)
    examples=[]
    for cat in ['hrsid','background']:
        for r,d in zip(rr,details):
            if d['exclusive_primary_source']==cat and (cat=='background' or r['source_kind']=='verified_adjacent_native_tiles'):
                examples.append(r);break
    examples.append(next(r for r in rr if r['split']=='val' and audit[r['source_path']]['dataset']=='hrsid'))
    def tab(headers,rs):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rs)
    text=f'''# 正式合成SAR数据来源用量

统计对象：冻结正式Train/Val规范清单，共6,000对；非正式方法展示材料不计入。核对每对保存的原始生成记录、完整源成员与可见源成员，不能以源目录存在或候选清点代替实际采用。

**实际采用SSDD为0对，760只是审计过的候选来源。** Train 0、Val 0；同时使用多个SSDD发布裁片0组；实际采用SSDD发布图像0幅，采样范围为空。不得为修正文案新增SSDD数据。

{tab(['来源','Train','Val','总对数','640','512','可见发布文件'],[[c,cats[c]['train']['pairs'],cats[c]['val']['pairs'],cats[c]['all']['pairs'],cats[c]['all']['pairs_640'],cats[c]['all']['pairs_512'],cats[c]['all']['distinct_visible_published_files']] for c in ['hrsid','background','ssdd']])}

主口径是“至少使用该类像素”。本次跨类别混用为0，因此上述计数可加总；互斥主来源与此一致：HRSID 5,282对，背景718对，SSDD 0对。文件数量是不同已发布裁片，不是真实独立雷达采集数量。

{tab(['来源','0%','10–30%','30–50%','50–70%','70–90%','90–100%'],[[c]+[cats[c]['all'][k] for k in ['overlap_zero','overlap_10_30','overlap_30_50','overlap_50_70','overlap_70_90','overlap_90_100']] for c in ['hrsid','background','ssdd']])}

## 来源身份结论

全部6,000对的完整源域均只有一个数据类别、一个已登记父场景及一个已知关联组件，且成员属于本对分区。未发现跨父场景或跨已知组件混合像素；保存可见成员与原始生成记录一致。相邻发布裁片通过文件行列偏移与公共像素核实后在共同源域读取，缺口不填补，冲突像素无效化，拼缝进入监督忽略区。

**实际原始采集身份不充分的关系为{actual_unknown}对**：发布裁片没有完整逐产品采集标识，所以不能把“同父场景、公共像素一致”夸称为已逐产品证明全部物理采集身份。未发现已知不同采集混入，不等于未知身份均已恢复。

## 原始像素及实际处理

{tab(['来源','发布格式','容器模式','类型','通道','灰度通道相同','文件数'],[[k[0],k[2],k[1],k[3],k[4],k[5],v] for k,v in fmt.items()])}

HRSID/背景采用原始发布的8位显示DN，不是可恢复定标的复数回波或物理强度。RGB容器按旧读取函数转灰度；通道是否相同以上表实测为准，不把RGB容器误称单通道文件。随后仅在同源坐标域中进行必要双线性几何采样、有效域处理及已保存的随机显示灰度增强；没有原图预放大。P0没有额外纹理、散斑或辐射操作，但保留原图固有纹理。局部改动单列忽略掩膜，不改变解析几何位置。

## 最小复核命令及代表样本

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/yl/miniconda3/envs/match/bin/python -B {OUT}/code/source_usage.py --check-only
```

{tab(['正式pair ID','分区','主来源','源发布图像','保存可见成员数'],[[r['pair_id'],r['split'],audit[r['source_path']]['dataset'],r['source_id'],len(r['visible_source_members'])] for r in examples])}

逐对链路在`meta/PAIR_SOURCE_TRACE_{t}.csv`；Train/Val、尺寸和重叠分布在`{usage.name}`。本轮只读核查，既有图像、GT、清单、划分均未改动。
'''
    with (OUT/f'SOURCE_USAGE_FOR_REPORT_{t}.md').open('x') as f:f.write(text)
    print(json.dumps(dict(report=str(OUT/f'SOURCE_USAGE_FOR_REPORT_{t}.md'),CSV=str(usage),source_pairs={c:cats[c]['all']['pairs'] for c in cats},formats=summary['raw_format_counts']),ensure_ascii=False),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();run(a.check_only)
