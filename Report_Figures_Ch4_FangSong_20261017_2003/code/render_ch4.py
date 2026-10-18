"""Read-only reuse of existing SAR samples; all writes stay in this figure directory."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/sar_ch4_fangsong_fonts_20261017')
import sys, json, csv, hashlib, base64, re, warnings, argparse, io
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, ConnectionPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text
from matplotlib import font_manager

BASE = Path('/home/yl/matching/sar_matching')
OLD = BASE/'Report_Figures_Mixed640512_20261016'
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OLD/'code'))
from source_usage import inputs, sha, inventory
from draw_figures import loadgt, points, pick_spread, mask

FONT_DIR = Path('/home/yl/.local/share/fonts/SAR_Research')
for font_file in sorted(FONT_DIR.iterdir()):
    if font_file.suffix.lower() in ['.ttf', '.otf']:
        font_manager.fontManager.addfont(str(font_file))
FAMILY = ['Times New Roman', 'FandolFang']
SVG_FAMILY = "'Times New Roman', 'FandolFang', 'FangSong', '仿宋', serif"
BLUE = '#287C8E'
INK = '#181818'
GRAY = '#d3d3d3'
EXPORTS = []
NS = '{http://www.w3.org/2000/svg}'
HREF = '{http://www.w3.org/1999/xlink}href'


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M')


def save(path, data):
    with Path(path).open('x') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(path, data):
    keys = list(dict.fromkeys(k for r in data for k in r))
    with Path(path).open('x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in data:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def style():
    # Latin/numerals use Times; unavailable CJK glyphs resolve to supplied Fandol FangSong.
    assert Path(font_manager.findfont(font_manager.FontProperties(family='Times New Roman'))).resolve() == FONT_DIR/'Times.TTF'
    assert Path(font_manager.findfont(font_manager.FontProperties(family='FandolFang'))).resolve() == FONT_DIR/'FandolFang-Regular.otf'
    plt.rcParams.update({'font.family': FAMILY, 'font.size': 10, 'axes.labelsize': 10,
        'axes.titlesize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
        'axes.linewidth': 1.5, 'lines.linewidth': 1.5, 'patch.linewidth': 1.5,
        'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.facecolor': 'white', 'savefig.facecolor': 'white',
        'image.interpolation': 'none', 'image.composite_image': False,
        'axes.unicode_minus': False})


def fig(height):
    f = plt.figure(figsize=(160/25.4, height/25.4))
    f.mm_height = height; f.raw_embeddings = {}; f.text_roles = []
    return f


def rect(f, x, y, w, h):
    return [x/160, y/f.mm_height, w/160, h/f.mm_height]


def text(f, x, y, s, size=10, ha='center', role='annotation', **kw):
    assert size >= (12 if role == 'heading' else 10)
    a = f.text(x/160, y/f.mm_height, s, ha=ha, va='center', fontsize=size,
               fontfamily=FAMILY, color=INK, **kw)
    f.text_roles.append((a, role))
    return a


def image(f, x, y, w, h, source):
    ax = f.add_axes(rect(f, x, y, w, h))
    if isinstance(source, (str, Path)):
        p = Path(source)
        with Image.open(p) as im: a = np.asarray(im).copy()
    else:
        p = None; a = source
    artist = ax.imshow(a, cmap='gray', vmin=0, vmax=255, origin='upper',
                       interpolation='none', resample=False)
    if p is not None:
        gid = f'original_SAR_{len(f.raw_embeddings):03d}'
        artist.set_gid(gid); f.raw_embeddings[gid] = str(p)
    ax.set_axis_off()
    return ax


def boundary(ax, m):
    if np.any(m) and not np.all(m):
        ax.contour(m.astype(np.uint8), levels=[.5], colors=[BLUE], linewidths=1.5)


def arrow(f, a, b):
    f.add_artist(FancyArrowPatch((a[0]/160, a[1]/f.mm_height),
                 (b[0]/160, b[1]/f.mm_height), transform=f.transFigure,
                 arrowstyle='-|>', mutation_scale=12, color=INK, lw=1.5))


def export(f, key, stem, t, pairs, note=''):
    target = OUT/'figures'/f'{stem}_{t}'
    assert not target.with_suffix('.png').exists()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        f.canvas.draw()
        renderer = f.canvas.get_renderer()
        for a, role in f.text_roles:
            b = a.get_window_extent(renderer)
            assert b.x0 >= 0 and b.y0 >= 0 and b.x1 <= f.bbox.width and b.y1 <= f.bbox.height, (key, a.get_text(), b)
        for ext in ['png', 'pdf', 'svg']:
            f.savefig(target.with_suffix('.'+ext), dpi=450, bbox_inches=None, facecolor='white')
        proof = OUT/'checks'/f'{stem}_160mm_{t}.png'
        f.savefig(proof, dpi=150, bbox_inches=None, facecolor='white')
        missing = [str(w.message) for w in caught if 'Glyph' in str(w.message) and 'missing' in str(w.message)]
        assert not missing, missing
    svg = target.with_suffix('.svg')
    tree = ET.parse(svg); root = tree.getroot(); embedded = []
    # Replace image payloads with byte-identical originals, retaining editable SVG overlays.
    for gid, source in f.raw_embeddings.items():
        g = root.find(f".//*[@id='{gid}']"); assert g is not None, gid
        im = g if g.tag == NS+'image' else g.find('.//'+NS+'image')
        assert im is not None
        mime = 'image/png' if Path(source).suffix.lower() == '.png' else 'image/jpeg'
        data = Path(source).read_bytes()
        im.set(HREF, 'data:'+mime+';base64,'+base64.b64encode(data).decode())
        embedded.append(dict(gid=gid, path=source, sha256=hashlib.sha256(data).hexdigest()))
    for n in root.iter():
        if n.tag in [NS+'text', NS+'tspan']:
            s = n.get('style', '')
            s = re.sub(r'font-family:[^;]+;?', '', s)
            n.set('style', s.rstrip(';')+'; font-family: '+SVG_FAMILY)
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    tree.write(svg, encoding='utf-8', xml_declaration=True)
    with Image.open(target.with_suffix('.png')) as a:
        assert a.width >= 2500
        width, height = a.size
    ex = dict(key=key, stem=target.name, png=str(target.with_suffix('.png')), pdf=str(target.with_suffix('.pdf')),
        svg=str(svg), proof=str(proof), width=width, height=height, dpi=450, physical_width_mm=160,
        physical_height_mm=f.mm_height, pairs=pairs, original_embedded_rasters=embedded,
        annotation_min_pt=10, heading_min_pt=12, line_min_pt=1.5, missing_glyphs=missing, note=note,
        text_sizes=[dict(text=a.get_text(), role=role, pt=a.get_fontsize()) for a, role in f.text_roles])
    EXPORTS.append(ex); plt.close(f)
    print(json.dumps(dict(exported=key, file=target.name, proof=str(proof)), ensure_ascii=False), flush=True)


def selected_ids(r, n):
    # Keep exactly the previously delivered 28 anchors; sparse version is a subset.
    z = loadgt(r)
    with (OLD/'FIG_D_EDITABLE_MARKERS_20261016_2340.csv').open() as f:
        all_ids = np.array([int(row['saved_structure_index']) for row in csv.DictReader(f)])
    ids = all_ids if n == 28 else all_ids[pick_spread(z['structure_points_A'][all_ids], n)]
    return z, ids


def flow(plan, t):
    r = plan['groups']['A'][0]; z, ids = selected_ids(r, 14)
    f = fig(73); xs = [3, 44, 85, 126]; w = 31
    for x, title in zip(xs, ['场景组织', '双视场几何', '随机外观', '监督与划分']):
        text(f, x+w/2, 65, title, 12, role='heading')
    image(f, xs[0], 23, w, w, plan['A_sources'][0]['original'])
    ax = f.add_axes(rect(f, xs[1], 23, w, w)); n = r['width']
    corners = np.array([[-.5,-.5],[n-.5,-.5],[n-.5,n-.5],[-.5,n-.5]])
    for side, ls, color in [('A', '-', INK), ('B', '--', BLUE)]:
        poly = points(np.linalg.inv(z['T_'+side]), corners)
        ax.plot(*np.vstack([poly, poly[0]]).T, ls=ls, color=color, lw=1.5)
        k = poly[0]
        ax.annotate(side, k, xytext=(0, 6 if side == 'A' else -12), textcoords='offset points', fontsize=12, color=INK)
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
    image(f, xs[2], 23, w, w, r['image_B'])
    ax = image(f, xs[3], 23, w, w, r['image_A'])
    pp = z['structure_points_A'][ids]
    ax.scatter(pp[:,0], pp[:,1], s=20, facecolors='none', edgecolors=BLUE, lw=1.5)
    for i in range(3): arrow(f, (xs[i]+w+1, 38.5), (xs[i+1]-1, 38.5))
    for x, label in zip(xs, ['原始影像组织', '旋转平移重叠', '含原始外观', '对应与分区']):
        text(f, x+w/2, 13, label, 10)
    export(f, '4-1', 'FIG_4_1_GENERATION_FLOW', t, [r['pair_id']],
           'Geometry is a coordinate-derived diagram; other panels are existing actual SAR images.')


def overlap(group, t, supplement=False):
    h = 131 if supplement else 230; f = fig(h)
    xs = [5, 43, 81, 119]; w = 35
    for x, title in zip(xs, ['图像A', '图像B', '公共区A', '公共区B']):
        text(f, x+w/2, h-7, title, 12, role='heading')
    for i, r in enumerate(group):
        y = h-23-i*49; z = loadgt(r)
        text(f, 80, y, f"重叠{r['overlap_ratio']*100:.1f}% / 旋转{r['rotation_actual_deg']:.1f}°", 12, role='heading')
        for x, side in zip(xs[:2], 'AB'): image(f, x, y-40, w, w, r['image_'+side])
        for x, side in zip(xs[2:], 'AB'):
            ax = image(f, x, y-40, w, w, mask(z, side)); boundary(ax, z['overlap_'+side])
    for x, color, label in [(38, GRAY, '有效视场'), (89, '#75b2be', '公共区域')]:
        f.add_artist(Rectangle((x/160, 5/h), 5/160, 3/h, transform=f.transFigure,
                              facecolor=color, edgecolor=INK, lw=1.5))
        text(f, x+8, 6.5, label, 10, ha='left')
    export(f, '4-2S' if supplement else '4-2', 'FIG_4_2_SUPPLEMENT' if supplement else 'FIG_4_2_OVERLAP_ROTATION', t,
           [r['pair_id'] for r in group], 'Native images plus masks read directly from saved GT.')


def appearance(plan, t):
    f = fig(93); xx = [9, 39, 69, 99, 129]; w = 26
    labels = ['原始外观', '灰度变化', '轻度综合扰动', '中度综合扰动', '非对称扰动']
    for x, r, label in zip(xx, plan['groups']['C'], labels):
        text(f, x+w/2, 85, label, 12, role='heading')
        for y, side in [(51,'A'),(19,'B')]: image(f, x, y, w, w, r['image_'+side])
    text(f, 3.5, 64, 'A', 12, role='heading'); text(f, 3.5, 32, 'B', 12, role='heading')
    text(f, 80, 7, '外观操作示例：灰度、细纹理、散斑、局部散射、方向模糊', 10)
    export(f, '4-3', 'FIG_4_3_CONTROLLED_APPEARANCE', t, [r['pair_id'] for r in plan['groups']['C']],
           'Five existing controls have identical source members and geometry; operator keywords denote a pool, not all operators in every panel.')
    r = plan['groups']['C_SEA'][0]; f = fig(86)
    text(f, 41, 79, '图像A', 12, role='heading'); text(f, 119, 79, '图像B', 12, role='heading')
    image(f, 5, 5, 72, 72, r['image_A']); image(f, 83, 5, 72, 72, r['image_B'])
    export(f, '4-3S', 'FIG_4_3_OFFSHORE_PAIR', t, [r['pair_id']],
           'One existing formal offshore pair; not a cross-source enhancement comparison.')


def supervision(r, t, n=14):
    z, ids = selected_ids(r, n); p, q = z['structure_points_A'][ids], z['structure_points_B'][ids]
    for pp, side in [(p, 'A'), (q, 'B')]:
        xy = np.floor(pp).astype(int); m = z['valid_'+side]&z['static_'+side]&~z['ignore_'+side]
        assert (xy>=0).all() and (xy<r['width']-1).all()
        assert np.all(m[xy[:,1],xy[:,0]]&m[xy[:,1]+1,xy[:,0]]&m[xy[:,1],xy[:,0]+1]&m[xy[:,1]+1,xy[:,0]+1])
    assert np.linalg.norm(points(z['H_AB'], p)-q, axis=1).max()<.001
    f = fig(175); text(f, 40, 167, '图像A', 12, role='heading'); text(f, 120, 167, '图像B', 12, role='heading')
    axA = image(f, 5, 89, 70, 70, r['image_A']); axB = image(f, 85, 89, 70, 70, r['image_B'])
    for ax, pp in [(axA,p),(axB,q)]: ax.scatter(pp[:,0],pp[:,1],s=24,facecolors='none',edgecolors=BLUE,lw=1.5,zorder=5)
    for a,b in zip(p,q):
        f.add_artist(ConnectionPatch(a,b,'data','data',axesA=axA,axesB=axB,color=BLUE,lw=1.5,alpha=.55,zorder=3))
    text(f, 80, 80, '解析对应与结构监督', 12, role='heading')
    for x, side, pp in [(12,'A',p),(92,'B',q)]:
        ax = image(f,x,16,56,56,mask(z,side)); boundary(ax,z['overlap_'+side])
        cp = z['check_points_'+side]
        cp = cp[np.linspace(0,len(cp)-1,min(32,len(cp))).astype(int)]
        ax.scatter(cp[:,0],cp[:,1],s=12,color='#555555',marker='+',lw=1.5)
        ax.scatter(pp[:,0],pp[:,1],s=24,facecolors='none',edgecolors=BLUE,marker='s',lw=1.5)
    text(f, 80, 7, '灰色＋：几何检查点；青色□：结构点；青色区域：公共域', 10)
    coords = [dict(saved_structure_index=int(k),pair_id=r['pair_id'],A_x=float(a[0]),A_y=float(a[1]),
                   B_x=float(b[0]),B_y=float(b[1]),gt_path=r['gt_path'],coordinate_frame='saved output native pixel centers xy')
              for k,a,b in zip(ids,p,q)]
    write_csv(OUT/f'FIG_D_MARKERS_{n}POINTS_{t}.csv',coords)
    export(f, 'D28' if n==28 else 'D', f'FIG_D_{n}POINTS_SUPERVISION', t, [r['pair_id']],
           f'{n} exact previously delivered anchors, no new points; all lines 1.5pt; markers diameter sqrt(24)pt.')


def scenes(plan,t):
    f = fig(122)
    names = ['离岸弱纹理','近岸舰船','港口码头','其他陆地结构']
    for i,(r,name) in enumerate(zip(plan['groups']['E'],names)):
        x = 3+(i%2)*80; y = 65-(i//2)*54
        text(f,x+37, y+48,name,12,role='heading')
        image(f,x,y+6,35,35,r['image_A']); image(f,x+39,y+6,35,35,r['image_B'])
    text(f,80,6,'每组左A、右B；场景按来源语境归类',10)
    export(f,'E1','FIG_E1_MARITIME_SCENES',t,[r['pair_id'] for r in plan['groups']['E']],
           'Existing source-context labels retained; no verified per-crop relabeling.')


def statistics(rr,t):
    f = fig(215)
    specs = [('overlap_bin',['zero','10_30','30_50','50_70','70_90','90_100'],['0','10–30','30–50','50–70','70–90','90–100'],'(a) 公共视场重叠率','重叠率（%）'),
             ('angle_bin',list(range(6)),['0–15','15–30','30–60','60–90','90–120','120–180'],'(b) 二维旋转角','旋转角（°）'),
             ('width',[640,512],['640×640','512×512'],'(c) 原生图像尺寸','尺寸（像素）')]
    rows = []
    for i,(key,order,labels,title,xlabel) in enumerate(specs):
        y = 153-i*67
        ax = f.add_axes(rect(f,25,y,128,43)); counts=Counter(r[key] for r in rr); vals=[counts[b] for b in order]
        ax.bar(range(len(order)),vals,color=BLUE,width=.56,zorder=3)
        ax.set_xticks(range(len(order)),labels); ax.set_ylim(0,max(vals)*1.30)
        ax.set_ylabel('配对数',fontsize=10);ax.set_xlabel(xlabel,fontsize=10,labelpad=4)
        ax.spines[['top','right']].set_visible(False); ax.tick_params(width=1.5,length=3,labelsize=10)
        ax.grid(axis='y',color='#888888',lw=1.5,alpha=.20,zorder=0)
        ax.set_yticks([0,600,1200] if i<2 else [0,2000,4000,6000])
        for j,n in enumerate(vals):
            label=f'{n:,}' if i<2 else f'{n:,}（{100*n/6000:.2f}%）'
            ax.text(j,n+max(vals)*.05,label,ha='center',va='bottom',fontsize=10)
        text(f,89,y+51,title,12,role='heading')
        rows.extend(dict(statistic=key,bin=b,count=n,denominator=6000) for b,n in zip(order,vals))
    text(f,80,5,'统计对象：全部6,000对合成双图',10)
    write_csv(OUT/f'FIG_E2_STATISTICS_{t}.csv',rows)
    export(f,'E2','FIG_E2_GEOMETRY_STATISTICS',t,[], 'All 6000 canonical pairs; measured geometric bin consistency checked separately.')


def snapshot(e):
    data = inventory()
    digest = hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()
    paths = list(e['manifests'].values())+[str(BASE/e['config'])]
    return dict(pair_file_count=len(data),pair_file_size_mtime_digest=digest,configuration_and_manifests={p:sha(p) for p in paths})


def proof_pdf(exports,t):
    p = OUT/'checks'/f'ALL_FIGURES_160mm_A4_{t}.pdf'
    with PdfPages(p) as pdf:
        for ex in exports:
            f = plt.figure(figsize=(210/25.4,297/25.4));h=ex['physical_height_mm']
            ax=f.add_axes([25/210,(297-h)/2/297,160/210,h/297]);ax.imshow(np.asarray(Image.open(ex['png'])),interpolation='none');ax.axis('off')
            f.text(.5,.96,'160 mm 通栏检查 · '+ex['key'],ha='center',fontsize=12,fontfamily=FAMILY)
            f.add_artist(FancyArrowPatch((25/210,12/297),(185/210,12/297),transform=f.transFigure,arrowstyle='|-|',lw=1.5,color=INK))
            f.text(.5,18/297,'160 mm',ha='center',fontsize=10,fontfamily=FAMILY)
            pdf.savefig(f,bbox_inches=None,dpi=450);plt.close(f)
    return str(p)


def run(only=None):
    for d in ['figures','checks','meta']:(OUT/d).mkdir(exist_ok=True)
    style();t=stamp();plan=json.loads((OLD/'meta/FIGURE_PLAN_20261016_2337.json').read_text());e,rr,audit,ss=inputs()
    before=snapshot(e)
    assert len(rr)==6000
    categories=Counter(audit[r['source_path']]['dataset'] for r in rr)
    assert categories=={'hrsid':5282,'background':718}
    for r in rr:
        assert r['angle_bin']==sum(r['rotation_actual_deg']>=x for x in [15,30,60,90,120])
    jobs={'flow':lambda:flow(plan,t),
          'overlap':lambda:overlap([plan['groups']['B'][i] for i in [1,2,3,5]],t),
          'overlap_supp':lambda:overlap([plan['groups']['B'][i] for i in [0,4]],t,True),
          'appearance':lambda:appearance(plan,t),
          'supervision':lambda:supervision(plan['groups']['D'][0],t,14),
          'supervision28':lambda:supervision(plan['groups']['D'][0],t,28),
          'scenes':lambda:scenes(plan,t),'statistics':lambda:statistics(rr,t)}
    for key,job in jobs.items():
        if only is None or key in only:job()
    assert before==snapshot(e),'Frozen dataset metadata changed during rendering'
    proof=proof_pdf(EXPORTS,t)
    save(OUT/f'meta/EXPORTS_{t}.json',dict(timestamp=t,exports=EXPORTS,proof_pdf=proof,
         frozen_before_after=before,source_usage=categories,requested_SVG_fonts=['FandolFang','Times New Roman'],
         font_files={p.name:sha(p) for p in FONT_DIR.iterdir() if p.suffix.lower() in ['.ttf','.otf']},
         preview_fonts=FAMILY,font_files_packaged=False,source_plan=str(OLD/'meta/FIGURE_PLAN_20261016_2337.json'),
         data_regenerated=False,training_started=False))
    print(json.dumps(dict(output=str(OUT),timestamp=t,figures=len(EXPORTS),proof=proof),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--only',nargs='+');a=p.parse_args();run(a.only)
