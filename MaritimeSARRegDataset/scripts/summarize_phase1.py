"""Independent source-to-output checks and timestamped Phase 1 statistics."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess

import cv2
import numpy as np


def read(path):
    result = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if result is None: raise ValueError(f'Cannot decode {path}')
    return result


def points(H, xy):
    xy = np.asarray(xy, dtype=np.float64)
    hp = np.column_stack([xy, np.ones(len(xy))]) @ np.asarray(H).T
    return hp[:, :2] / hp[:, 2:]


def nearest_reference(source, H, size):
    """Manual inverse sampling; intentionally does not call sarreg warp/geometry."""
    y, x = np.indices((size, size))
    xy = points(np.linalg.inv(H), np.column_stack([x.ravel(), y.ravel()]))
    q = np.rint(xy).astype(int)
    h, w = source.shape
    inside = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
    result = np.zeros(size*size, np.uint8)
    result[inside] = source[q[inside, 1], q[inside, 0]]
    return result.reshape(size, size)


def source_instance(shape, ann):
    m = np.zeros(shape, np.uint8); seg = ann.get('segmentation')
    if isinstance(seg, list) and seg:
        for poly in seg:
            cv2.fillPoly(m, [np.rint(np.asarray(poly).reshape(-1, 2)).astype(np.int32)], 255)
    elif isinstance(seg, dict):
        raise ValueError('Independent checker requires explicitly implemented RLE decoding')
    else:
        x, y, w, h = ann['bbox_xywh']
        m[max(0,int(np.floor(y))):min(shape[0],int(np.ceil(y+h))),
          max(0,int(np.floor(x))):min(shape[1],int(np.ceil(x+w)))] = 255
    return m


def check_sample(folder, row):
    meta = json.loads((folder/'meta.json').read_text())
    size, n = meta['output_size'], meta['num_views']
    matrices = [np.asarray(H, np.float64) for H in meta['H_source_to_view']]
    valid = [read(folder/f'valid_{i}.png') for i in range(n)]
    dynamic = [read(folder/f'dynamic_{i}.png') for i in range(n)]
    errors = []; visibility_records=[]; forward = backward = inverse = relative = cycle = 0.
    matches = 0; overlap = []; mask_mismatch = sea_mismatch = 0; bbox_count = 0
    src_shape = tuple(meta['source_shape_hw']); resized_shape = tuple(meta['resized_source_shape_hw'])
    for i, H in enumerate(matrices):
        expected = nearest_reference(np.full(resized_shape, 255, np.uint8), H, size)
        mismatch = int(np.count_nonzero(expected != valid[i])); mask_mismatch += mismatch
        if mismatch: errors.append(f'manual valid footprint mismatch view {i}: {mismatch} pixels')
    if row.get('sea_mask_path'):
        sea = read(row['sea_mask_path'])
        if sea.shape != resized_shape:
            sea = cv2.resize(sea, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_NEAREST)
        for i, H in enumerate(matrices):
            expected = nearest_reference(sea, H, size)
            mismatch = int(np.count_nonzero(expected != read(folder/f'sea_{i}.png')))
            sea_mismatch += mismatch
            if mismatch: errors.append(f'manual sea projection mismatch view {i}: {mismatch} pixels')
    source_masks = {str(a['id']): source_instance(src_shape, a) for a in row['annotations']}
    for i, H in enumerate(matrices):
        union = np.zeros((size, size), np.uint8)
        for a in meta['ship_annotations'][i]:
            src = source_masks[a['instance_id']]
            if src.shape != resized_shape:
                src = cv2.resize(src, (resized_shape[1], resized_shape[0]), interpolation=cv2.INTER_NEAREST)
            # OpenCV reference independent of the project's labels/geometry functions.
            m = cv2.warpPerspective(src, H, (size, size), flags=cv2.INTER_NEAREST)
            if 'synthetic_movement_xy' in a:
                dx, dy = a['synthetic_movement_xy']
                m = cv2.warpAffine(m, np.float32([[1,0,dx],[0,1,dy]]), (size,size), flags=cv2.INTER_NEAREST)
                m[valid[i]==0]=0
            sy,sx=np.where(src>0)
            projected=points(H,np.column_stack([sx,sy]))
            boundary_lost=bool(np.any((projected<-.5)|(projected>=size-.5)))
            y, x = np.where(m > 0)
            expected = [int(x.min()), int(y.min()), int(x.max()-x.min()+1), int(y.max()-y.min()+1)] if len(x) else None
            if expected != a['bbox_xywh'] or len(x) != a['area_pixels']:
                errors.append(f'source instance bbox/area mismatch view {i}, ID {a["instance_id"]}')
            visibility_records.append({'folder':str(folder),'view':i,'instance_id':a['instance_id'],
                                       'bbox_xywh':json.dumps(a['bbox_xywh']),
                                       'visible_fraction_estimate':a['visible_fraction_estimate'],
                                       'source_support_outside_view_estimate':boundary_lost,
                                       'mask_source':a['mask_source']})
            union = cv2.bitwise_or(union, m); bbox_count += 1
        if not np.array_equal(union, read(folder/f'ships_{i}.png')):
            errors.append(f'source instance union mask mismatch view {i}')
        if meta.get('dynamic_events'):
            changed=np.zeros((size,size),np.uint8)
            for event in meta['dynamic_events'][i]:
                origin=source_masks[event['instance_id']]
                if origin.shape!=resized_shape:
                    origin=cv2.resize(origin,(resized_shape[1],resized_shape[0]),interpolation=cv2.INTER_NEAREST)
                origin=cv2.warpPerspective(origin,H,(size,size),flags=cv2.INTER_NEAREST)
                dx,dy=event['shift_xy']
                dest=cv2.warpAffine(origin,np.float32([[1,0,dx],[0,1,dy]]),(size,size),flags=cv2.INTER_NEAREST)
                dest[valid[i]==0]=0
                if np.count_nonzero(origin)!=event['origin_area_pixels'] or np.count_nonzero(dest)!=event['destination_area_pixels']:
                    errors.append(f'dynamic event area mismatch view {i}')
                changed|=origin|dest
            guard=meta.get('dynamic_exclusion_guard_pixels',0)
            if guard:changed=cv2.dilate(changed,np.ones((guard*2+1,guard*2+1),np.uint8))
            changed[valid[i]==0]=0
            if not np.array_equal(changed,dynamic[i]):errors.append(f'independent dynamic exclusion mismatch view {i}')
        if np.any(read(folder/f'view_{i}.png')[valid[i]==0]):
            errors.append(f'image intensity outside valid footprint view {i}')
    pair_matrices = {}
    for p in meta['pairs']:
        i,j = p['views']; H = np.asarray(p['H_i_to_j']); rev = np.asarray(p['H_j_to_i'])
        expected = matrices[j] @ np.linalg.inv(matrices[i]); expected /= expected[2,2]
        relative = max(relative, float(np.max(np.abs(H/H[2,2]-expected))))
        product = rev @ H; product /= product[2,2]
        inverse = max(inverse, float(np.max(np.abs(product-np.eye(3)))))
        pair_matrices[i,j] = H
        with np.load(folder/p['matches']) as data: a,b = data['points_i'],data['points_j']
        matches += len(a)
        if len(a):
            forward = max(forward, float(np.max(np.linalg.norm(points(H,a)-b, axis=1))))
            backward = max(backward, float(np.max(np.linalg.norm(points(rev,b)-a, axis=1))))
        for k,xy,op in ((i,a,p['overlap_mask_i']),(j,b,p['overlap_mask_j'])):
            over = read(folder/op)
            # Independent manual destination-valid sampling in the correct coordinate frame.
            other = j if k==i else i
            target_to_other = H if k==i else rev
            projected = nearest_reference(valid[other], np.linalg.inv(target_to_other), size)
            expected_overlap = np.where((valid[k]>0)&(projected>0),255,0).astype(np.uint8)
            if not np.array_equal(expected_overlap,over): errors.append(f'manual overlap mismatch {i}-{j} in {k}')
            overlap.append(float(np.count_nonzero(over)/max(1,np.count_nonzero(valid[k]))))
            if len(xy):
                q = np.rint(xy).astype(int); x,y = q.T
                if np.any(q<0) or np.any(q>=size): errors.append(f'match out of bounds view {k}')
                elif np.any(valid[k][y,x]==0) or np.any(over[y,x]==0) or np.any(dynamic[k][y,x]>0):
                    errors.append(f'match not in static common region view {k}')
    if n>=3:
        for i in range(n):
            for j in range(i+1,n):
                for k in range(j+1,n):
                    product = pair_matrices[j,k] @ pair_matrices[i,j]; product /= product[2,2]
                    expected = pair_matrices[i,k]/pair_matrices[i,k][2,2]
                    cycle = max(cycle,float(np.max(np.abs(product-expected))))
    if max(forward,backward,relative,inverse,cycle)>1e-3:
        errors.append('matrix/point/cycle tolerance exceeded (1e-3)')
    return {'folder': str(folder), 'errors': errors, 'passed': not errors, 'match_count': matches,
            'bbox_count': bbox_count, 'max_forward_error_px': forward, 'max_backward_error_px': backward,
            'max_relative_matrix_error': relative, 'max_inverse_matrix_error': inverse,
            'max_cycle_matrix_error': cycle, 'manual_valid_mismatch_pixels': mask_mismatch,
            'manual_sea_mismatch_pixels': sea_mismatch, 'overlap': overlap,
            'ship_visibility':visibility_records}, meta


def summarize(run_root, index):
    # Index paths in each generation_config determine the original experiment protocol.
    fallback = {r['image_path']:r for r in map(json.loads,Path(index).read_text().splitlines())}
    checks, samples, experiments = [], [], {}
    for root in sorted(run_root.iterdir()):
        if not root.is_dir() or not (root/'generation_config.json').exists(): continue
        config_path=root/'generation_config.json'; config=json.loads(config_path.read_text())
        source_index=Path(config['index']); lookup={r['image_path']:r for r in map(json.loads,source_index.read_text().splitlines())} if source_index.exists() else fallback
        generation=json.loads((root/'generation_report_all.json').read_text())
        validation=json.loads((root/'validation_report.json').read_text())
        local=[]
        for path in sorted(root.glob('*/pair_*/meta.json')):
            m=json.loads(path.read_text()); row=lookup[m['source_image_path']]
            check,meta=check_sample(path.parent,row);checks.append(check)
            record={'experiment':root.name,'folder':str(path.parent),'dataset':meta['dataset'],
                    'source_id':meta['source_id'],'source_group':meta['source_group'],'scene_type':row['scene_type'],
                    'source_split':meta['source_split'],'official_split':row['official_split'],
                    'source_height':meta['source_shape_hw'][0],'source_width':meta['source_shape_hw'][1],
                    'resize_x':meta['source_resize_factors_xy'][0],'resize_y':meta['source_resize_factors_xy'][1],
                    'upsampled':max(meta['source_resize_factors_xy'])>1,'views':meta['num_views'],
                    'ships_source':len(row['annotations']),'ships_output':check['bbox_count'],
                    'min_overlap':min(check['overlap']),'mean_overlap':float(np.mean(check['overlap'])),
                    'max_overlap':max(check['overlap']),'sea_mask':bool(row.get('sea_mask_path')),
                    'label_mask_sources':','.join(sorted({a['mask_source'] for view in meta['ship_annotations'] for a in view})),
                    'sample_seed':meta['seed'],'augmentation_speckle_sigma':config['config']['augmentation']['speckle_sigma'],
                    'independent_passed':check['passed']}
            samples.append(record);local.append(record)
        experiments[root.name]={'generation':generation,'validation':validation,'global_seed':config['global_seed'],
                                'index_sha256':hashlib.sha256(source_index.read_bytes()).hexdigest(),
                                'config_sha256':hashlib.sha256(config_path.read_bytes()).hexdigest(),
                                'datasets':dict(Counter(s['dataset'] for s in local)),
                                'scene_types':dict(Counter(s['scene_type'] for s in local)),
                                'upsampled':sum(s['upsampled'] for s in local),
                                'with_output_ships':sum(s['ships_output']>0 for s in local),
                                'min_overlap':min(s['min_overlap'] for s in local),
                                'mean_overlap':float(np.mean([s['mean_overlap'] for s in local])),
                                'max_overlap':max(s['max_overlap'] for s in local),
                                'independent_failed':sum(not s['independent_passed'] for s in local)}
    package_names=['numpy','opencv-python','opencv-python-headless','PyYAML','tqdm','pytest','maritime-sarreg-data']
    packages={}
    for name in package_names:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    metrics={key:max(c[key] for c in checks) for key in ('max_forward_error_px','max_backward_error_px','max_relative_matrix_error','max_inverse_matrix_error','max_cycle_matrix_error')}
    metrics.update(manual_valid_mismatch_pixels=sum(c['manual_valid_mismatch_pixels'] for c in checks),
                   manual_sea_mismatch_pixels=sum(c['manual_sea_mismatch_pixels'] for c in checks),
                   matches=sum(c['match_count'] for c in checks),bboxes=sum(c['bbox_count'] for c in checks))
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    code_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path('sarreg').glob('*.py')))).hexdigest()
    report={'checked':len(checks),'passed':sum(c['passed'] for c in checks),'failed':sum(not c['passed'] for c in checks),
            'metrics':metrics,'experiments':experiments,'errors':[c for c in checks if not c['passed']],
            'environment':{'system':platform.platform(),'python':platform.python_version(),'packages':packages,
                           'opencv_import_version':cv2.__version__,'opencv_import_file':cv2.__file__,
                           'opencv_threads':cv2.getNumThreads(),'workers':1,'rng':'numpy.default_rng (PCG64)',
                           'determinism':'Sequential sampling with fixed seeds; no GPU/PyTorch computation'},
            'validation_code_commit':git_commit,'sarreg_code_sha256':code_hash,
            'validation_git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
            'scientific_kind':'synthetic_2d_registration_proxy','tolerance_matrix_and_points':1e-3,
            'source_mask_pixel_tolerance':0}
    report['ship_visibility']=[r for c in checks for r in c['ship_visibility']]
    report['metrics']['truncated_ship_estimate_count']=sum(r['source_support_outside_view_estimate'] for r in report['ship_visibility'])
    return report,samples


def write_report(report, samples, out, stamp, run_root):
    audit=json.loads((out/f'phase1_data_audit_{stamp}.json').read_text())
    repair=json.loads((out/f'phase1_split_repair_{stamp}.json').read_text())
    testlog=out/f'phase1_pytest_{stamp}.log'
    test_results=re.findall(r'\d+ passed[^\n]*',testlog.read_text()) if testlog.exists() else []
    test_result=test_results[-1] if test_results else '未提供测试日志，请查看 execution 日志'
    lines=[f'# Phase 1 实测报告 — {stamp}', '',
           f'真实数据合成与独立验收：{report["passed"]}/{report["checked"]} 组通过。原始索引存在实际文件重复导致的划分泄漏；另交付显式协议修正版，未覆写原索引或原始数据。', '',
           '## 环境及来源', '',
           f'- {report["environment"]["system"]}，Python {report["environment"]["python"]}，cv2 实际导入 {report["environment"]["opencv_import_version"]}。',
           '- 项目 v0.3.0；依赖使用工作区 .venv 复用现有环境，未修改 base；没有训练、GPU、CUDA 或模型 checkpoint。',
           f'- 验证时 Git HEAD：`{report["validation_code_commit"]}`，工作树含未提交变更：{report["validation_git_dirty"]}；实际代码 SHA256：`{report["sarreg_code_sha256"]}`。',
           '- 本次依赖、线程、随机数与命令详见 validation_summary JSON 和 execution 日志。', '',
           f'- 软件自测：{test_result}；初始 v0.3.0 为 13 passed，均为虚构 fixture 测试，和下面真实数据验收区分。', '',
           '## 全量真实数据审计', '',
           '|来源|图像记录|成功解码|舰船实例|海陆掩膜|原始实验 train/val/test|',
           '|---|---:|---:|---:|---:|---|']
    for ds,s in audit['datasets'].items():
        lines.append(f'|{ds}|{s["images"]}|{s["decoded"]}|{s["ships"]}|{s["sea_masks"]}|{s["splits"].get("train",0)}/{s["splits"].get("val",0)}/{s["splits"].get("test",0)}|')
    groups=audit['byte_identical_sources']; extra=sum(len(x)-1 for x in groups)
    diff=audit['hrsid_annotation_discrepancy']
    lines += ['', f'规范文件记录 {audit["total_images"]} 条；发现 {len(groups)} 组字节相同文件，冗余 {extra} 条，因此字节不同文件 {audit["total_images"]-extra} 份。这仍不是独立采集场景数量。',
              f'读取/标注尺寸/掩膜错误：{sum(e["type"]!="identical_file_split_leakage" for e in audit["errors"])}；源框警告：{len(audit["warnings"])}。',
              'SL-SSDD 按 (official split, stem) 全量关联；0=land/255=sea，949 张全海、211 张包含两类，均为正确尺寸的 uint8 单通道。',
              f'HRSID 官方 train/test 合计 {diff["official_instances"]} 实例，合并文件 {diff["merged_instances"]}，差异涉及 {len(diff["different_images"])} 张图。保留官方两份标签，不混用发布版本。具体差异见 data_audit JSON。',
              'SSDD 只读取 BBox_RBox_PSeg_SSDD/coco_style 的 1,160 张文件及 2,587 条 polygon 标注；类别 ID 按 ship 名称识别。SL-SSDD README 的 2,456 个船目标不替代此分支的实际计数。', '',
              '## 显式划分修正', '',
              f'原始索引的内容重复隔离检查未通过；{len(audit["errors"][0]["files"]) if audit["errors"] and audit["errors"][0]["type"]=="identical_file_split_leakage" else 0} 组相同文件跨 split。',
              f'新协议 `{repair["protocol"]}` 把由相同文件连接的父场景整体合并；优先保留已有 test、其次 val 成员。调整 {repair["changed_split_rows"]} 条实验 split，原始 official_split 保持。',
              f'新索引：`{repair["index"]}`。当前已知字节重复跨 split={repair["cross_split_byte_duplicate_groups"]}，父场景跨 split={repair["cross_split_parent_groups"]}。',
              '四类基础 smoke 继续使用原始协议且 split=all，仅验证工程 GT；覆盖样例和三视图使用修正版。修正版不是官方 SSDD benchmark split，不能冒称官方实验。未核查近似重复和隐含同航次场景。', '',
              '## 实际生成与 GT 验证', '',
              '|子实验|seed|生成/目标|validate 通过|独立失败|上采样|重叠 min/mean/max|',
              '|---|---:|---:|---:|---:|---:|---|']
    for name,e in report['experiments'].items():
        lines.append(f'|{name}|{e["global_seed"]}|{e["generation"]["generated"]}/{e["generation"]["requested"]}|{e["validation"]["passed"]}|{e["independent_failed"]}|{e["upsampled"]}|{e["min_overlap"]:.4f}/{e["mean_overlap"]:.4f}/{e["max_overlap"]:.4f}|')
    lines += ['', '所有输入都来自真实 SAR 文件；程序 pytest 的虚构 fixture 与这些数据分开。增强、尺寸、场景、船数和上采样因子逐条见 sample_statistics CSV。三视图只约束 view0 与各视图的重叠阈值。',
              '表中重叠 min/mean/max 统计各方向，阈值约束的是每个参考视图对的双向最小值；因此单方向 max 可超过配置 max_overlap=0.95。上采样不增加真实 SAR 分辨率。',
              '独立验收直接计算齐次点投影、逐像素反采样、COCO 实例传播和三视图组合，不调用 sarreg.geometry 或 sarreg.labels。',
              f'矩阵/点误差容限 1e-3，源有效区域及海陆像素容限为 0。实际指标：`{json.dumps(report["metrics"],ensure_ascii=False)}`。',
              '所有输出船框均为传播 polygon 的实例支持包络；保留 instance_id 和可见比例估计。逐实例投影支持越界截断估计见 ship_visibility CSV，属于离散几何估计。背景源的标签严格为空。', '',
              '## 输出与复现', '',
              f'- 输出根目录：`{run_root.resolve()}`；预览见 [样例索引](phase1_previews_{stamp}.md)。',
              f'- 执行日志：`phase1_execution_{stamp}.log`；JSON 数值证据：`phase1_validation_summary_{stamp}.json`；源审计 CSV：`phase1_data_audit_{stamp}.csv`。',
              '- 每个子实验的 generation/validation 报告另导出为 reports 下的分钟时间戳文件；输出目录内保留 CLI 原生文件名以兼容现有读取接口。',
              '- 各输出目录保存配置、seed、源路径、manifest 和 generation/validation report；逐实验 config/index 哈希列于汇总 JSON。',
              '- 安装：`python -m venv --system-site-packages .venv`（先选择依赖齐全的 Python 3.10+）；`.venv/bin/python -m pip install --no-deps --no-build-isolation -e ".[dev]"`。',
              '- 重现命令：`bash scripts/run_phase1.sh /path/to/sar`；可设置 PHASE1_PYTHON 与 PHASE1_STAMP。已有分钟时间戳目录会直接拒绝覆盖。',
              '- 重现脚本设置 CPU 单线程；本次初始生成使用 OpenCV 默认线程，详见日志/环境。绝对源路径参与样例种子计算，迁移路径后不能承诺逐字节相同。', '',
              '## 代码变更与科学边界', '',
              '- 旧 v0.3.0 保存在原 ZIP 与 baseline Git commit；本次最小修改 qa.py 与 synthesis.py，新增审计、显式 split 修正、汇总、覆盖样例及全流程脚本。',
              '- 回归证实旧 validator 不识别动态区匹配、缺失 NPZ 中断整批报告，旧生成器重跑可改写 manifest/config；已修复。OpenCV BGR 预览的红/青通道语义也已修正。',
              '- 稀疏点检查有效/重叠/动态支持，图像及分类掩膜检查尺寸和二值性，输出源元信息补充 scene/official_split/protocol。基础几何和增强协议未改。',
              '- 这是 synthetic_2d_registration_proxy。图像平面角度不是雷达观测方位差，共享源散斑不会因独立加噪而成为真实独立观测；GT 自洽不证明低结构海面可被匹配算法解出。',
              '- 未生成可选大规模 P1 数据，未训练神经网络，动态 hard 配置未用于本次真实验收。下一步先固定显式修正的 split，进行 SIFT/RANSAC 小基线，再选择真实多视角数据。']
    (out/f'phase1_report_{stamp}.md').write_text('\n'.join(lines)+'\n')
    gallery=[f'# Phase 1 代表性预览 — {stamp}', '', '上排：两视图船框与 GT 对齐；中排：有效区域与双向重叠；SSDD 下排：蓝色海面、绿色陆地、黑色无效填充。', '']
    choices=[('coverage_examples_8','previews_coverage_8',8),('smoke_hrsid_20','previews_hrsid_20',3),('smoke_background_10','previews_background_10',3)]
    for source,preview,count in choices:
        metas=sorted((run_root/source).glob('*/pair_*/meta.json'))
        chosen=np.random.default_rng(42).choice(len(metas),size=min(8 if source=='coverage_examples_8' else 10 if source=='smoke_hrsid_20' else 5,len(metas)),replace=False)
        for i,idx in enumerate(chosen[:count]):
            meta=json.loads(metas[int(idx)].read_text()); path=run_root/preview/f'preview_{i:03d}.png'
            gallery += [f'## {meta["dataset"]} / {meta["source_id"]}', '', f'样本：`{metas[int(idx)].parent}`。船框：{sum(len(v) for v in meta["ship_annotations"])}；双向最小重叠：{min(meta["pairs"][0]["overlap_fraction_i"],meta["pairs"][0]["overlap_fraction_j"]):.3f}。', '', f'![{meta["source_id"]}](../{path.as_posix()})', '']
    (out/f'phase1_previews_{stamp}.md').write_text('\n'.join(gallery))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--index',required=True)
    p.add_argument('--reports',default='reports');p.add_argument('--stamp',required=True)
    a=p.parse_args();root=Path(a.root);out=Path(a.reports);out.mkdir(parents=True,exist_ok=True)
    report,samples=summarize(root,a.index)
    for name,experiment in report['experiments'].items():
        for kind in ('generation','validation'):
            (out/f'phase1_{name}_{kind}_{a.stamp}.json').write_text(json.dumps(experiment[kind],indent=2,ensure_ascii=False))
    summary=out/f'phase1_validation_summary_{a.stamp}.json'
    summary.write_text(json.dumps(report,indent=2,ensure_ascii=False))
    with (out/f'phase1_sample_statistics_{a.stamp}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    with (out/f'phase1_ship_visibility_{a.stamp}.csv').open('w',newline='') as f:
        records=report['ship_visibility']
        if records:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    write_report(report,samples,out,a.stamp,root)
    print(json.dumps({k:v for k,v in report.items() if k not in ('experiments','errors','ship_visibility')},indent=2))
    raise SystemExit(0 if report['failed']==0 else 1)


if __name__=='__main__':main()
