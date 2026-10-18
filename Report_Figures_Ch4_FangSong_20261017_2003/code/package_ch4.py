"""Short publication captions, source coordinates and final asset validation."""
import sys, re, subprocess, base64, hashlib, json, csv
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_ch4 as r

TITLES = {
 '4-1':'海事SAR合成双图及监督构建流程',
 '4-2':'不同公共视场比例与二维旋转的合成SAR实例',
 '4-2S':'零重叠与高重叠合成SAR补充实例',
 '4-3':'同源同几何条件下的随机外观扰动',
 '4-3S':'离岸海面与舰船的正式合成SAR双图',
 'D':'合成几何对应与结构监督',
 'D28':'合成结构对应的二十八点完整展示',
 'E1':'海事及陆地来源语境的合成SAR实例',
 'E2':'合成SAR双图的几何与尺寸分布',
}
CAPTIONS = {
 '4-1':'从海事SAR场景组织出发，依次构造含旋转、平移和公共区域的双视场，并随机施加或跳过外观扰动。解析变换生成几何对应，结构区域提供局部监督，最终按来源组件划分训练集与验证集。',
 '4-2':'四行依次展示低、中、高重叠及不同二维旋转的原生双图和公共区域。重叠率定义为两侧公共有效像素占各自有效像素比例的较小值；灰色表示有效视场，青色及其轮廓表示公共区域。',
 '4-2S':'两行分别展示严格零重叠与较高重叠的合成SAR双图，以及各自两侧有效域和公共域。零重叠样本仅作为评价负例；另一组保留低纹理海面与舰船外观，补充主图未展示的几何组合。',
 '4-3':'五列在同一来源和同一几何下比较原始外观、灰度变化、轻度及中度综合扰动和非对称扰动，上下分别为图像A与B。各列按保存参数组合不同操作，体现显示灰度域外观变化及结构保持。',
 '4-3S':'正式合成样本展示离岸低纹理海面与舰船在两侧视场中的外观及位置关系。该对不施加额外外观扰动，保留已发布SAR图像的固有纹理；二维旋转与公共视场参数独立记录，用于几何配准评价。',
 'D':'原生双图以十四组结构点及连线展示合成几何对应，下方给出两侧公共区域与监督位置。灰色十字检查点用于检验投影，青色方框表示局部结构监督点；稀疏显示兼顾对应可读性与图像结构辨识。',
 'D28':'本图使用与稀疏展示相同的原生SAR双图，保留二十八组已保存结构对应及其连线，下方同时显示公共区域与几何检查点。完整标记用于对照空间分布，点位坐标与稀疏版共享同一像素坐标系。',
 'E1':'四组正式合成双图分别来自离岸弱纹理、近岸舰船、港口码头和其他陆地结构的来源语境，每组左侧为图像A、右侧为图像B。名称描述来源场景，未将其改写为逐裁片已核实的地物标签。',
 'E2':'以全部六千对正式合成双图为分母，分别统计公共视场重叠率、二维旋转角与原生图像尺寸。柱高表示配对数量，尺寸面板同时标注占比；旋转为二维几何变化，不代表真实雷达观测方位差。',
}


def table(head, rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def relative(p):
    return str(Path(p).relative_to(r.OUT))


def run():
    t=r.stamp(); docs=r.OUT/f'FIGURES_FOR_CH4_{t}.md'
    metadata=json.loads(sorted((r.OUT/'meta').glob('EXPORTS_*.json'))[-1].read_text())
    exs={ex['key']:ex for ex in metadata['exports']}
    assert set(exs)==set(TITLES)
    for k,c in CAPTIONS.items(): assert 60<=len(c)<=100,(k,len(c))
    plan=json.loads((r.OLD/'meta/FIGURE_PLAN_20261016_2337.json').read_text())
    allrows={a['pair_id']:a for group in plan['groups'].values() for a in group}
    source_rows=[]
    for key in TITLES:
        for pid in exs[key]['pairs']:
            a=allrows[pid]
            source_rows.append(dict(figure=key,pair_id=pid,image_A=a['image_A'],image_B=a['image_B'],gt_path=a['gt_path'],
                width=a['width'],height=a['height'],overlap_A=a['overlap_A'],overlap_B=a['overlap_B'],overlap_min=a['overlap_ratio'],
                rotation_actual_2D_deg=a['rotation_actual_deg'],geometry_parameters=a['geometry_parameters'],H_AB=a['H_AB'],H_BA=a['H_BA'],
                source_to_view=a['H_source_to_view'],source_domain_bounds_xyxy=a['source_domain_bounds_xyxy'],
                anchor_source=a['source_path'],source_members=a['source_members'],source_component=a['source_component_id'],
                source_context=a['source_scene'],verified_visible_scene=a['verified_visible_scene'],
                appearance_operations_A=a['enhancement_ops_A'],appearance_operations_B=a['enhancement_ops_B'],
                appearance_parameters_A=a['enhancement_params_A'],appearance_parameters_B=a['enhancement_params_B'],
                formal_6000=not pid.startswith('PROFILE_CONTROL_'),seed=a['seed']))
    source_csv=r.OUT/f'FIGURE_SOURCES_AND_GEOMETRY_{t}.csv';r.write_csv(source_csv,source_rows)
    asset_csv=r.OUT/f'FIGURE_FILES_{t}.csv'
    r.write_csv(asset_csv,[dict(figure=k,title=TITLES[k],caption=CAPTIONS[k],**{j:exs[k][j] for j in ['png','pdf','svg','proof','pairs','physical_width_mm','physical_height_mm']}) for k in TITLES])
    core=plan['groups']['A'][0];z=r.loadgt(core);corners=np.array([[-.5,-.5],[639.5,-.5],[639.5,639.5],[-.5,639.5]])
    # Explicit panel provenance and both view coordinate systems, kept off the figure face.
    panels=[dict(stage='scene',source_file=plan['A_sources'][0]['original'],image_extent='full published image',
                 source_bounds_xyxy=plan['A_sources'][0]['bounds'],panel_position_mm=[3,23,31,31]),
            dict(stage='geometry',source_file=core['gt_path'],kind='source-domain diagram',panel_position_mm=[44,23,31,31],
                 A_source_corners=r.points(np.linalg.inv(z['T_A']),corners).tolist(),B_source_corners=r.points(np.linalg.inv(z['T_B']),corners).tolist(),
                 coordinate_system='parent published scene XY directly; T_A/T_B already include the source-domain offset; do not add it twice'),
            dict(stage='appearance',source_file=core['image_B'],side='B',pixel_centers_xyxy=[0,0,639,639],panel_position_mm=[85,23,31,31]),
            dict(stage='supervision',source_file=core['image_A'],side='A',pixel_centers_xyxy=[0,0,639,639],panel_position_mm=[126,23,31,31])]
    pixel_test=np.array([[0.,0.],[319.5,319.5],[639.,639.]])
    roundtrip={}
    for side in 'AB':
        parent=r.points(np.linalg.inv(z['T_'+side]),pixel_test)
        error=np.linalg.norm(r.points(z['T_'+side],parent)-pixel_test,axis=1).max()
        assert error<1e-8
        assert (parent[:,0]>=9000).all() and (parent[:,0]<10400).all()
        assert (parent[:,1]>=1800).all() and (parent[:,1]<3200).all()
        roundtrip[side]=dict(native_pixels=pixel_test.tolist(),parent_scene_xy=parent.tolist(),max_roundtrip_error_px=float(error))
    r.save(r.OUT/f'meta/FLOW_PANEL_COORDINATES_{t}.json',dict(pair_id=core['pair_id'],domain_bounds=core['source_domain_bounds_xyxy'],
           pixel_parent_roundtrip=roundtrip,
           T_A=z['T_A'].tolist(),T_B=z['T_B'].tolist(),panels=panels,pixel_centers_convention='XY pixel centers; image edges at -0.5 and 639.5'))
    sections=['# 第4章插图交付',
        '**正式来源：HRSID 5,282对，SAR背景影像718对，SSDD 0对；Train 4,800对、Val 1,200对。** 本轮只重排既有图件；历史素材、正式图像、GT、清单、生成配置和权重保持原样。',
        '## 使用方式',
        '图件均按160 mm通栏设计，PNG为450 dpi，SVG保留可编辑文字、箭头及几何图形，PDF保留矢量线条与文字。建议优先插入PDF或SVG，PNG适合直接预览。主标题及分图标签12 pt，注释10 pt，显式轮廓和对应线1.5 pt。',
        '全部图件的中文使用用户提供的仿宋字体FandolFang，英文和数值使用Times New Roman；直接加载`~/.local/share/fonts/SAR_Research/`中的实际字体，PNG/PDF不再采用旧版代换字体。SVG指定相同字体并保留文字编辑能力；在其他机器编辑时需安装相应字体。未打包字体文件。',
        f'先看[160 mm实际宽度A4检查PDF]({relative(metadata["proof_pdf"])})，按“实际大小／100%”打印，关闭“适合页面”。每图还提供150 dpi、160 mm宽的检查PNG。',
        f'[文件索引CSV]({asset_csv.name}) · [样本、原始路径和完整几何参数CSV]({source_csv.name})。原始A/B与GT均通过下方样本链接或CSV直达，不需要从组合图反向截取。',
        table(['图号','图名','用途'],[[k,TITLES[k],'备用素材' if k in ['4-2S','4-3S','D28'] else '主图'] for k in TITLES])]
    for key in TITLES:
        ex=exs[key]
        sections += [f'## 图{key}　{TITLES[key]}','**图注：** '+CAPTIONS[key],
                     '文件：'+' · '.join(f'[{ext.upper()}]({relative(ex[ext])})' for ext in ['svg','pdf','png'])+
                     f' · [160 mm检查图]({relative(ex["proof"])})',
                     f'![图{key} {TITLES[key]}]({relative(ex["proof"])})']
        if ex['pairs']:
            sections += [table(['样本ID','边长','重叠A／B／较小值','二维旋转'],
                [[pid,str(allrows[pid]['width']),f"{100*allrows[pid]['overlap_A']:.3f}% / {100*allrows[pid]['overlap_B']:.3f}% / {100*allrows[pid]['overlap_ratio']:.3f}%",f"{allrows[pid]['rotation_actual_deg']:.3f}°"] for pid in ex['pairs']])]
        if key=='4-1':
            sections += ['四环节图片依次为同源发布场景、保存变换计算的视场轮廓、核心样本B图、核心样本A图及其结构标记。后两幅完整像素中心范围均为(0,0)至(639,639)；图面位置、源域范围及两侧变换详见`meta/FLOW_PANEL_COORDINATES_'+t+'.json`。']
        if key=='4-3':
            sections += ['说明：五组512材料为已有同源、同几何对照，不参与正式6,000对统计。轻度／中度是所用预设类别名称，并不声称每侧、每个算子的强度都严格单调。下方算子词为可选操作集合；本五列未执行局部散射，实际各侧操作和参数详见来源CSV。']
        if key=='4-3S':
            sections += ['说明：本图仅取上一轮离岸素材中的一对正式图像，左右为同一合成关系的A/B；不把不同原图的列组合成受控增强对照。']
        if key in ['D','D28']:
            marker=sorted(r.OUT.glob('FIG_D_MARKERS_'+('28' if key=='D28' else '14')+'POINTS_*.csv'))[-1]
            sections += [f'[可编辑点位CSV]({marker.name})。14点为原28点的空间分散子集；点位属于已保存的合成结构监督，未据此新增真实跨方位物理GT。']
        if key=='E1':
            sections += ['来源语境与旧清单一致；四类全量配对数依次为2,100、1,800、1,500、600，仍未形成逐裁片人工确认标签。']
        if key=='E2':
            stats=sorted(r.OUT.glob('FIG_E2_STATISTICS_*.csv'))[-1]
            sections += [f'[统计底表]({stats.name})。分母均为6,000对；640为4,869对（81.15%），512为1,131对（18.85%）。区间左闭右开，末段包含终点；严格零重叠单列。']
    pids=list(dict.fromkeys(pid for ex in exs.values() for pid in ex['pairs']))
    sections += ['## 原始图像与真值入口',
        '下表直接链接原文件。正式双图统一位于`SyntheticSAR_Mixed640512_Multicondition_20261016_2114/pairs/<train或val>/<样本ID>/`；对照双图位于该目录的既有`smoke_20261016_2122`。更早的原始发布SAR路径、源成员、来源组件及采样变换均在来源CSV中逐项保存。',
        table(['样本ID','图像A','图像B','GT'],[[pid]+[f'[{label}]({allrows[pid][field]})' for label,field in [('A','image_A'),('B','image_B'),('NPZ','gt_path')]] for pid in pids])]
    with docs.open('x') as f:f.write('\n\n'.join(sections)+'\n')

    checks=[]
    for key,ex in exs.items():
        tree=ET.parse(ex['svg']);root=tree.getroot()
        assert abs(float(root.get('width').removesuffix('pt'))*25.4/72-160)<.02
        for embedded in ex['original_embedded_rasters']:
            node=root.find(f".//*[@id='{embedded['gid']}']")
            node=node if node.tag==r.NS+'image' else node.find('.//'+r.NS+'image')
            raw=base64.b64decode(node.get(r.HREF).split(',',1)[1])
            assert hashlib.sha256(raw).hexdigest()==r.sha(embedded['path'])==embedded['sha256']
        textnodes=root.findall('.//'+r.NS+'text')
        labels=[''.join(n.itertext()) for n in textnodes]
        assert not any(re.search(r'P[0-4]|PROFILE_|MIX_|\.png|内部|演示样本',s) for s in labels),labels
        assert all('Times New Roman' in n.get('style','') and 'FandolFang' in n.get('style','') for n in textnodes)
        assert min(float(x) for n in textnodes for x in re.findall(r'font-size:\s*([\d.]+)px',n.get('style','')))>=10
        info=subprocess.check_output(['pdfinfo',ex['pdf']],text=True)
        assert re.search(r'Pages:\s+1\b',info)
        fonts=subprocess.run(['pdffonts',ex['pdf']],capture_output=True,text=True,check=True)
        assert not fonts.stderr,fonts.stderr
        assert 'TimesNewRomanPSMT' in fonts.stdout
        assert not any(s in fonts.stdout for s in ['Liberation','DroidSans','NotoSerif'])
        with Image.open(ex['png']) as im: assert im.width>=2500 and abs(im.info['dpi'][0]-450)<1
        with Image.open(ex['proof']) as im: assert abs(im.width/im.info['dpi'][0]*25.4-160)<.3
        checks.append(dict(figure=key,SVG_raw_payloads_byte_identical=len(ex['original_embedded_rasters']),
                           font_stack_and_size_checked=True,caption_length=len(CAPTIONS[key]),proof_visually_reviewed=True))
    assert metadata['frozen_before_after']==r.snapshot(r.inputs()[0])
    assert not any(p.suffix.lower() in ['.ttf','.ttc','.otf'] for p in r.OUT.rglob('*'))
    for s in re.findall(r'\]\(([^)]+)\)',docs.read_text()): assert (docs.parent/s).exists(),s
    r.save(r.OUT/f'meta/FINAL_FIGURE_CHECK_{t}.json',dict(timestamp=t,markdown=str(docs),checks=checks,
        frozen_data_size_mtime_and_canonical_config_hashes_unchanged=True,physical_proof_width_mm=160,
        actual_preview_fonts=r.FAMILY,SVG_font_family=r.SVG_FAMILY,font_files_packaged=0,
        full_dataset_SHA_repeat=False,new_SAR_pixels_generated=False,new_physical_GT_generated=False,
        historical_report_or_Word_modified=False,render_manifest=str(sorted((r.OUT/'meta').glob('EXPORTS_*.json'))[-1])))
    print(json.dumps(dict(markdown=str(docs),asset_csv=str(asset_csv),source_csv=str(source_csv),figures=len(exs)),ensure_ascii=False),flush=True)


if __name__=='__main__':run()
