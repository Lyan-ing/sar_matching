"""Document immutable figure assets and perform bounded final delivery checks."""
from source_usage import *
import re
import subprocess
import xml.etree.ElementTree as ET

TITLES = {
    'A': '海事SAR双视场合成与监督构建流程',
    'B': '不同二维旋转与公共视场比例的合成SAR实例',
    'C': '同源同几何条件下的SAR显示灰度外观扰动',
    'C_SEA': '离岸来源语境中的随机外观实例',
    'D': '解析几何检查与局部结构对应监督的分工',
    'D_PURE': '结构监督示例的无标记原始双图',
    'E': '合成SAR场景来源语境与几何尺寸分布',
}
CAPTIONS = {
    'A': '以两幅同源相邻发布图像和一组正式合成双图说明构建过程。视场轮廓由保存的几何变换计算，随后展示随机外观、解析对应和结构监督；训练与验证按来源隔离，示意连线不代表真实跨方位观测。',
    'B': '六组正式样本覆盖严格零重叠至高重叠及不同二维旋转幅度，每行依次展示原始双图与两侧公共区域。灰色表示有效域，青色表示公共域；重叠率取两侧实测比例的较小值，零重叠仅示范评价负例。',
    'C': '五列读取同一来源、同一几何的既有外观展示样例，上下分别为图像A与B。P0不施加额外扰动，其余列按实际记录展示灰度、细纹理、乘性散斑及模糊等组合；这些显示域操作不代表独立雷达回波，样例不计入正式六千对。',
    'C_SEA': '各列为离岸来源语境下已保存的正式合成双图，分别体现无额外操作、灰度变化、细纹理、局部散射和单侧扰动。不同列的来源与几何并不相同，故仅展示实际外观实例，不能解释为同一图像的增强因果对比。',
    'D': '上部在正式双图中显示二十八个已保存且位于合法结构域的对应点，下部分别叠加几何网格与结构标记。网格用于检查解析投影，结构点用于局部监督；二者均不构成真人确认的真实跨方位物理真值。',
    'D_PURE': '本图保留结构监督示例的完整原始双图，不叠加连线、点位或公共区域。图像与带标记版对应同一正式样本，可用于论文排版或重新添加标注；原生像素副本另行保存，显示不附加去噪、锐化或对比度拉伸。',
    'E': '上部展示离岸、近岸、港口与其他陆地来源语境的正式双图，下部统计全部六千对的实测重叠率、二维旋转角和图像尺寸。场景名称仅表明来源语境，尚非逐片确认标签；旋转角不表示真实雷达观测方位差。',
}
SECTIONS = {'A': '4.1 数据来源与合成构建', 'B': '4.2 双视场几何与重叠控制',
            'C': '4.2 随机外观扰动', 'C_SEA': '4.2 随机外观扰动（补图）',
            'D': '4.3 几何真值与结构监督', 'D_PURE': '4.3 几何真值与结构监督（排版备用）',
            'E': '4.1 数据组成／4.4 数据分布'}
STEMS = {'A': 'FIG_A_CONSTRUCTION_FLOW', 'B': 'FIG_B_ROTATION_OVERLAP',
         'C': 'FIG_C_CONTROLLED_APPEARANCE', 'C_SEA': 'FIG_C_SEA_INSTANCES',
         'D': 'FIG_D_GEOMETRY_STRUCTURE', 'D_PURE': 'FIG_D_PURE_PAIR', 'E': 'FIG_E_SCENES_DISTRIBUTIONS'}
OPS = {'radiometry': '灰度辐射变化', 'fine_texture': '细纹理', 'multiplicative_speckle': '附加乘性散斑',
       'smooth_clutter': '平滑杂波', 'directional_blur': '方向模糊', 'correlated_field': '相关随机场', 'local_scatter': '局部散射'}


def rel(p):
    return str(Path(p).relative_to(OUT))


def table(headers, data):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, r)) + ' |' for r in data])


def run():
    t = stamp()
    plan = json.loads(sorted((OUT/'meta').glob('FIGURE_PLAN_*.json'))[-1].read_text())
    e, rr, audit, ss = inputs()
    provenance = json.loads(sorted((OUT/'meta').glob('SOURCE_PROVENANCE_*.json'))[-1].read_text())
    exports = {}
    for p in sorted((OUT/'meta').glob('FIGURE_EXPORTS*.json')):
        for ex in json.loads(p.read_text())['exports']:
            for key, prefix in STEMS.items():
                if ex['stem'].startswith(prefix+'_'):
                    if key not in exports or exports[key]['stem'] < ex['stem']:
                        exports[key] = ex
    assert set(exports) == set(STEMS)
    for k, caption in CAPTIONS.items():
        assert 60 <= len(caption) <= 120, (k, len(caption))
        assert not any(s in caption for s in ['数据Codex', '代码流程', 'V2.2', '审计', 'checksum', '内部版本'])
    native = plan['native_copies']
    assert len(native) == 40
    for original, copied in native.items():
        assert sha(original) == sha(copied), copied
    for s in plan['A_sources']:
        assert sha(s['original']) == sha(s['copied'])
    current = {k: list(v) for k, v in inventory().items()}
    assert current == provenance['metadata_inventory_before']
    assert all(sha(p) == h for p, h in provenance['manifest_hashes'].items())
    verified = []
    for key, ex in exports.items():
        with Image.open(ex['png']) as im:
            assert im.width >= 2500 and im.info['dpi'][0] >= 300
        root = ET.parse(ex['svg']).getroot()
        images = root.findall('.//{http://www.w3.org/2000/svg}image')
        assert images and all(n.get('{http://www.w3.org/1999/xlink}href', '').startswith('data:image/') for n in images)
        texts = root.findall('.//{http://www.w3.org/2000/svg}text')
        assert texts or key == 'D_PURE'
        info = subprocess.check_output(['pdfinfo', ex['pdf']], text=True)
        assert re.search(r'Pages:\s+1\b', info)
        assert not ex['missing_glyphs']
        verified.append(dict(figure=key, png=ex['png'], svg=ex['svg'], pdf=ex['pdf'], width=ex['width'],
                             height=ex['height'], dpi=ex['dpi'], caption_length=len(CAPTIONS[key]),
                             embedded_SAR_images=len(images), editable_vector_text_elements=len(texts)))
    assert not any(OUT.rglob('*.ttf')) and not any(OUT.rglob('*.otf')) and not any(OUT.rglob('*.ttc'))
    stats = []
    specs = [('overlap_bin', ['zero', '10_30', '30_50', '50_70', '70_90', '90_100']),
             ('angle_bin', list(range(6))), ('width', [640, 512]),
             ('source_scene', ['offshore_vessels_low_texture_sea', 'nearshore_vessels', 'port_quay_dense_vessels', 'other_land_structures'])]
    for field, order in specs:
        for split in ['train', 'val', 'all']:
            subset = [r for r in rr if split == 'all' or r['split'] == split]
            for b in order:
                n = sum(r[field] == b for r in subset)
                stats.append(dict(statistic=field, bin=b, split=split, pairs=n, denominator=len(subset), fraction=n/len(subset)))
    for r in rr:
        a = r['rotation_actual_deg']; assert sum(a >= x for x in [15, 30, 60, 90, 120]) == r['angle_bin']
        o = r['overlap_ratio']
        actual_bin = 'zero' if o == 0 else '10_30' if o < .3 else '30_50' if o < .5 else '50_70' if o < .7 else '70_90' if o < .9 else '90_100'
        assert actual_bin == r['overlap_bin']
    stats_path = OUT/f'FIGURE_E_STATISTICS_{t}.csv'; write_csv(stats_path, stats)
    index_csv = OUT/f'FIGURE_ASSET_INDEX_{t}.csv'
    write_csv(index_csv, [dict(figure=k, title=TITLES[k], caption=CAPTIONS[k], suggested_section=SECTIONS[k],
                             **{n: exports[k][n] for n in ['png', 'svg', 'pdf', 'pairs', 'width', 'height', 'physical_width_mm']}) for k in STEMS])
    source_md = sorted(OUT.glob('SOURCE_USAGE_FOR_REPORT_*.md'))[-1]
    source_csv = sorted(OUT.glob('SOURCE_USAGE_FOR_REPORT_*.csv'))[-1]
    marker_csv = sorted(OUT.glob('FIG_D_EDITABLE_MARKERS_*.csv'))[-1]
    paragraphs = ['# 合成SAR论文插图素材索引',
        f'本次交付只提取已有图像、真值和来源记录，不重新采样、不改变6,000对正式数据，不改训练验证划分、权重、真实SAR人工记录或报告Word。索引生成时间：{t[:8]} {t[9:11]}:{t[11:13]}（北京时间）。',
        '## 可直接采用的来源结论',
        '**实际采用SSDD为0对，760只是检查过的候选来源。** HRSID实际5,282对（Train 4,248／Val 1,034），SAR背景718对（Train 552／Val 166）。三类无跨类别混用，本次“至少使用该类像素”统计可加总。正式样本总数6,000对、12,000幅；方法展示材料另列。',
        f'完整来源口径、尺寸及重叠分布、可复核命令见[来源说明]({source_md.name})及[来源统计CSV]({source_csv.name})。源成员均在同一已登记父场景及已知来源组件内；发布资料缺少完整逐产品采集标识，不能据此宣称全部原始物理采集身份均已恢复。',
        '## 素材总表',
        table(['图号', '正式图题', '样本／角色', '建议第4章位置'],
              [[k.replace('_SEA', '补').replace('_PURE', '纯图'), TITLES[k], '、'.join(exports[k]['pairs']), SECTIONS[k]] for k in STEMS]),
        '建议小节仅供排版组织，未假定论文现有章节编号。以下每幅提供可直接采用的图题与60–120字技术图注。主要图A—E外，另有离岸补图与图D纯图备用。',
        '## 文件与排版约定',
        '所有推荐PNG宽2,834像素，450 dpi，设计宽160 mm；建议通栏插入，图B不宜缩为80 mm单栏。SVG/PDF中的文字、图例、线框及连线为矢量，SAR内容保留嵌入位图，并非虚构矢量细节。原生512/640 PNG副本不缩放、不改像素；组合图仅进行固定0–255灰度映射与显示布局，没有附加去噪、锐化或对比度拉伸。',
        '本机未安装宋体和Times New Roman，采用已安装的Noto Serif CJK与Liberation Serif，导出无缺失字形告警。未打包字体；SVG保留可编辑文字，其他机器需有相应字体或采用PDF嵌入字体版／PNG。早先图A底部说明距边缘过近，已上移并保留旧版，以下链接均指向修正后的推荐版。',
        f'机器可读文件索引：[图件CSV]({index_csv.name})；[原生PNG完整索引]({rel(plan["raw_index"])})，44条图号引用对应40幅唯一原生PNG；[统计底表]({stats_path.name})按Train、Val和合计列出。图D的[28点可编辑坐标]({marker_csv.name})保留原保存索引，坐标以输出原生像素中心为准。']
    for key in STEMS:
        ex = exports[key]
        paragraphs += [f'## 图{key}：{TITLES[key]}',
                       '**技术图注：** '+CAPTIONS[key],
                       '文件：'+' · '.join(f'[{ext.upper()}]({rel(ex[ext])})' for ext in ['png', 'svg', 'pdf']),
                       f'![图{key} {TITLES[key]}]({rel(ex["png"])})']
        group = plan['groups']['D' if key == 'D_PURE' else key]
        if key == 'A':
            paragraphs += ['实际双图：`'+group[0]['pair_id']+'`。几何轮廓来自该对已保存变换；流程箭头仅为构建示意。两幅原始发布文件的完整路径与同源范围见下表。',
                           table(['发布图像', '原始路径', '保留副本'], [[s['published_id'], s['original'], f'[文件]({rel(s["copied"])})'] for s in plan['A_sources']])]
        elif key == 'B':
            paragraphs += ['参数补注：a—f与图中六行对应，重叠率为有效域中的公共比例，图内采用两侧较小值；角度均为合成二维旋转。',
                           table(['行／正式ID', '尺寸', '重叠A／B／较小值', '二维旋转', '来源语境'],
                                 [[f'{chr(97+i)} / {r["pair_id"]}', f'{r["width"]}×{r["height"]}',
                                   f'{100*r["overlap_A"]:.3f}% / {100*r["overlap_B"]:.3f}% / {100*r["overlap_ratio"]:.3f}%',
                                   f'{r["rotation_actual_deg"]:.3f}°', r['source_scene']] for i, r in enumerate(group)]),
                           '覆盖规则在选图前保存：六个重叠槽位、指定小中大角度与离岸／近岸／港口来源语境，优先P0/P1；槽位内按固定原始亮度、对比度及ID排序。未读取模型预测或匹配成绩。规则见`meta/FIGURE_SELECTION_RULE_20261016_2337.json`。']
        elif key in ['C', 'C_SEA']:
            paragraphs += ['实际操作如下；P0—P4是预设类别，同类不保证每侧执行相同操作。',
                           table(['类别／样本', 'A实际操作', 'B实际操作'],
                                 [[r['enhancement_profile']+' / '+r['pair_id'],
                                   '、'.join(OPS[o] for o in r['enhancement_ops_A']) or '无额外操作',
                                   '、'.join(OPS[o] for o in r['enhancement_ops_B']) or '无额外操作'] for r in group])]
            if key == 'C':
                paragraphs += ['这5对为既有`smoke_20261016_2122`中的512方法展示样例，同源成员和两侧几何矩阵逐项一致；不加入6,000对计数。主图不包含局部散射操作，不将P3或P4当作“每种扰动都出现”的例子。']
            else:
                paragraphs += ['此补图5对属于正式数据，均为640；局部散射实际出现在P3，方向模糊实际出现在P2。未另外生成同源受控海面五联样例，因此不能借本补图隔离各增强操作的影响。']
        elif key in ['D', 'D_PURE']:
            paragraphs += [f'正式样本：`{group[0]["pair_id"]}`，640×640。标记版从已保存结构候选中取28个空间分散点，逐点检查两侧有效／静态／非忽略域以及解析映射一致性。网格仅是几何检查点；结构候选未经过独立真人物理核实，不能转写为真实跨角物理GT。无标记版与带标记版共用同一原始双图。']
        elif key == 'E':
            paragraphs += [table(['来源语境', '正式示例ID', '全量对数'],
                                 [[r['source_scene'], r['pair_id'], sum(x['source_scene'] == r['source_scene'] for x in rr)] for r in group]),
                           '重叠率计数依次为300、900、1,200、1,200、1,200、1,200；二维旋转区间0–15／15–30／30–60／60–90／90–120／120–180°计数依次为900、900、1,200、1,200、1,200、600。640为4,869对（81.15%），512为1,131对（18.85%）。所有分母为6,000对，非独立原图／物理采集数量；区间按左闭右开、末段含终点解释。',
                           '所有正式记录的`verified_visible_scene`仍为unknown。离岸2,100、近岸1,800、港口1,500、其他陆地600仅是来源语境统计，不能改写成逐裁片已确认舰船数量或场景标签。']
    paragraphs += ['## 复现与边界',
        '来源核查命令见来源说明；图件生成实现保留在`code/source_usage.py`、`code/prepare_assets.py`、`code/draw_figures.py`与`code/delivery_index.py`。这些代码只在本素材目录写入新文件；重复运行会产生新时间戳输出，不建议为时间戳对齐重复执行。',
        '已检查正式6,000条逐对来源链、4,530幅实际可见发布文件的格式与通道、40幅PNG和2幅源文件副本一致性、7组推荐图的尺寸与可解析性。正式24,000个pair文件的大小和修改时间及两份规范清单内容与本次起始快照一致；没有为交付重新开展全库大规模像素哈希。全量来源元数据与实际物理采集身份缺失的限制见来源说明。']
    path = OUT/f'FIGURE_ASSET_INDEX_{t}.md'
    with path.open('x') as f:
        f.write('\n\n'.join(paragraphs)+'\n')
    save(OUT/f'meta/FIGURE_DELIVERY_CHECK_{t}.json', dict(timestamp=t, index_markdown=str(path), index_csv=str(index_csv),
         figure_count=7, verified=verified, native_png_copies_verified=40, original_source_copies_verified=2,
         formal_pair_files_metadata_unchanged=len(current), canonical_manifests_unchanged=True,
         actual_bins_match_reported_bins=6000, font_files_packaged=0, dataset_regenerated=False, training_started=False,
         physical_GT_created=False, caption_lengths={k: len(v) for k, v in CAPTIONS.items()}))
    print(json.dumps(dict(index=str(path), csv=str(index_csv), check=str(OUT/f'meta/FIGURE_DELIVERY_CHECK_{t}.json')), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
