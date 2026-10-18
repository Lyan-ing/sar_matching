"""Publication assets from immutable existing PNGs/GT; no new SAR synthesis."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sar_report_figures_mpl_20261016')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
from source_usage import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch,Rectangle,ConnectionPatch
import warnings
FONT=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc')
BLUE='#287C8E';INK='#222222';GRAY='#777777'
plt.rcParams.update({'font.family':'serif','font.serif':[FONT.get_name(),'Liberation Serif'],'font.size':8,'axes.labelsize':8,
    'xtick.labelsize':7.2,'ytick.labelsize':7.2,'axes.linewidth':.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
    'image.interpolation':'nearest','image.composite_image':False,'savefig.facecolor':'white','figure.facecolor':'white','axes.unicode_minus':False})
EXPORTS=[]
def im(p):
    a=cv2.imread(str(p),cv2.IMREAD_GRAYSCALE);assert a is not None;return a
def figure(height):return plt.figure(figsize=(160/25.4,height/25.4))
def text(fig,x,y,s,size=8,ha='center',**kw):return fig.text(x,y,s,ha=ha,va='center',fontsize=size,fontproperties=FONT,color=INK,**kw)
def image_ax(fig,rect,p):
    ax=fig.add_axes(rect);a=im(p) if isinstance(p,(str,Path)) else p
    ax.imshow(a,cmap='gray',vmin=0,vmax=255,origin='upper',interpolation='nearest',resample=False);ax.set_axis_off();return ax
def mask(z,L):
    v=z['valid_'+L];over=z['overlap_'+L];a=np.ones((*v.shape,3),np.uint8)*250;a[v]=[211,211,211];a[over]=[117,178,190];return a
def export(fig,stem,t,role,pairs):
    stem=OUT/'figures'/f'{stem}_{t}';stem.parent.mkdir(exist_ok=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for ext in ['png','svg','pdf']:
            p=stem.with_suffix('.'+ext);assert not p.exists();fig.savefig(p,dpi=450,facecolor='white',bbox_inches=None)
        missing=[str(w.message) for w in caught if 'Glyph' in str(w.message) and 'missing' in str(w.message)]
        assert not missing,missing
    with Image.open(stem.with_suffix('.png')) as p:width,height=p.size;dpi=p.info.get('dpi')
    assert width>=2500
    EXPORTS.append(dict(stem=stem.name,png=str(stem.with_suffix('.png')),svg=str(stem.with_suffix('.svg')),pdf=str(stem.with_suffix('.pdf')),
        width=width,height=height,dpi=dpi,physical_width_mm=160,role=role,pairs=pairs,
        image_display='fixed original 0-255 grayscale; nearest-neighbor presentation only; unmodified full PNG copies separately',
        vector_content='editable text, axes and line work; SAR rasters embedded, not invented vector detail',missing_glyphs=missing))
    plt.close(fig);print(json.dumps(dict(exported=stem.name,width=width,height=height)),flush=True);return EXPORTS[-1]
def loadgt(r):
    with np.load(r['gt_path'],allow_pickle=False) as z:return {k:z[k] for k in z.files}
def points(H,p):
    a=np.c_[p,np.ones(len(p))]@H.T;return a[:,:2]/a[:,2:]
def pick_spread(p,n=28):
    if len(p)<=n:return np.arange(len(p))
    chosen=[int(np.argmin(np.sum((p-p.mean(0))**2,axis=1)))];dist=np.linalg.norm(p-p[chosen[0]],axis=1)
    for _ in range(n-1):
        k=int(np.argmax(dist));chosen.append(k);dist=np.minimum(dist,np.linalg.norm(p-p[k],axis=1))
    return np.array(chosen)

def flow(plan,t):
    r=plan['groups']['A'][0];z=loadgt(r);fig=figure(88);xs=[.025,.235,.445,.655,.86];ww=[.17,.17,.17,.17,.115]
    titles=['原始海事SAR','双视场几何','随机外观','对应与监督','输出分区']
    for x,w,title in zip(xs,ww,titles):text(fig,x+w/2,.93,title,9)
    for x,w,nx in zip(xs[:-1],ww[:-1],xs[1:]):fig.add_artist(FancyArrowPatch((x+w+.004,.56),(nx-.008,.56),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=8,lw=.7,color=BLUE))
    for i,s in enumerate(plan['A_sources']):image_ax(fig,[xs[0],.53-i*.33,ww[0],.28],s['original'])
    text(fig,xs[0]+ww[0]/2,.14,'同源相邻发布图',7.5);text(fig,xs[0]+ww[0]/2,.07,'来源先隔离',8)
    ax=fig.add_axes([xs[1],.30,ww[1],.49]);n=r['width'];corners=np.array([[-.5,-.5],[n-.5,-.5],[n-.5,n-.5],[-.5,n-.5]])
    for L,style,col in [('A','-',INK),('B','--',BLUE)]:
        poly=points(np.linalg.inv(z['T_'+L]),corners);ax.plot(*np.vstack([poly,poly[0]]).T,style,color=col,lw=.9);ax.text(*poly.mean(0),L,color=col,ha='center',fontsize=8)
    ax.set_aspect('equal');ax.invert_yaxis();ax.axis('off');text(fig,xs[1]+ww[1]/2,.23,'旋转 · 位移 · 重叠',7.5);text(fig,xs[1]+ww[1]/2,.14,'原生像素视场',7.5)
    for i,L in enumerate('AB'):image_ax(fig,[xs[2],.53-i*.33,ww[2],.28],r['image_'+L])
    text(fig,xs[2]+ww[2]/2,.14,'实际输出 A / B',7.5);text(fig,xs[2]+ww[2]/2,.07,'P0—P4随机选择',7.5)
    ids=pick_spread(z['structure_points_A'],24)
    for i,L in enumerate('AB'):
        ax=image_ax(fig,[xs[3],.53-i*.33,ww[3],.28],mask(z,L));p=z['structure_points_'+L][ids];ax.scatter(p[:,0],p[:,1],s=2,facecolors='none',edgecolors=INK,lw=.3)
    text(fig,xs[3]+ww[3]/2,.14,'解析几何 / 结构点',7.5);text(fig,xs[3]+ww[3]/2,.07,'分开保存',7.5)
    for y,title,count in [(.62,'Train','4,800对'),(.30,'Val','1,200对')]:
        fig.add_artist(Rectangle((xs[4],y),ww[4],.20,transform=fig.transFigure,fill=False,lw=.7,edgecolor=BLUE));text(fig,xs[4]+ww[4]/2,y+.135,title,8.5);text(fig,xs[4]+ww[4]/2,y+.065,count,8)
    text(fig,.5,.028,'构建流程示意；图像与视场参数均取自已保存材料',7.5)
    return export(fig,'FIG_A_CONSTRUCTION_FLOW',t,'method schematic with actual published inputs and formal output pair',[r['pair_id']])

def geometry(group,t):
    fig=figure(235);xx=[.035,.28,.525,.77];ww=.205
    for x,title in zip(xx,['图像 A','图像 B','公共视场 A','公共视场 B']):text(fig,x+ww/2,.974,title,8.5)
    for i,r in enumerate(group):
        y=.79-i*.147;z=loadgt(r);label=f"({chr(97+i)})  {r['overlap_ratio']*100:.1f}%  /  {r['rotation_actual_deg']:.1f}°"
        text(fig,.5,y+.137,label,8)
        image_ax(fig,[xx[0],y,ww,.127],r['image_A']);image_ax(fig,[xx[1],y,ww,.127],r['image_B'])
        for x,L in zip(xx[2:],'AB'):image_ax(fig,[x,y,ww,.127],mask(z,L))
    text(fig,.5,.012,'灰色：有效视场；青色：公共区域；零重叠仅为负例',7.5)
    return export(fig,'FIG_B_ROTATION_OVERLAP',t,'six formal pairs, fixed overlap/angle/source-context coverage',[r['pair_id'] for r in group])

def appearance(group,t,supplement=False):
    fig=figure(88 if not supplement else 94);xs=[.035+i*.195 for i in range(5)];w=.177
    labels=['无额外增强','轻度灰度变化','纹理 / 散斑','模糊 / 杂波','单侧变化']
    for j,r in enumerate(group):
        text(fig,xs[j]+w/2,.95,'P'+str(j),9)
        for i,L in enumerate('AB'):image_ax(fig,[xs[j],.52-i*.36,w,.33],r['image_'+L])
        text(fig,xs[j]+w/2,.075,labels[j] if not supplement else r['enhancement_profile']+' 实际样本',7.5)
    text(fig,.012,.69,'A',8);text(fig,.012,.33,'B',8)
    note='同源、同几何方法展示样例，不参与6,000对统计' if not supplement else '离岸来源语境；各列来源及几何不同，不作增强因果对比'
    text(fig,.5,.018,note,7.5)
    return export(fig,'FIG_C_SEA_INSTANCES' if supplement else 'FIG_C_CONTROLLED_APPEARANCE',t,'formal different-source illustrative examples' if supplement else 'existing controlled method demonstration; outside formal 6000',[r['pair_id'] for r in group])

def supervision(r,t):
    z=loadgt(r);fig=figure(150);ids=pick_spread(z['structure_points_A'],28);p,q=z['structure_points_A'][ids],z['structure_points_B'][ids]
    valid=[]
    for pts,L in zip([p,q],'AB'):
        xy=np.floor(pts).astype(int);m=z['static_'+L]&z['valid_'+L]&~z['ignore_'+L]
        assert (xy>=0).all() and (xy<r['width']-1).all();assert np.all(m[xy[:,1],xy[:,0]]&m[xy[:,1]+1,xy[:,0]]&m[xy[:,1],xy[:,0]+1]&m[xy[:,1]+1,xy[:,0]+1])
    assert np.linalg.norm(points(z['H_AB'],p)-q,axis=1).max()<.001
    axA=image_ax(fig,[.035,.55,.435,.40],r['image_A']);axB=image_ax(fig,[.53,.55,.435,.40],r['image_B'])
    for ax,pts in [(axA,p),(axB,q)]:ax.scatter(pts[:,0],pts[:,1],s=5,facecolors='none',edgecolors=BLUE,lw=.45,zorder=3)
    for a,b in zip(p,q):
        line=ConnectionPatch(a,b,'data','data',axesA=axA,axesB=axB,color=BLUE,lw=.3,alpha=.42,zorder=2);fig.add_artist(line)
    text(fig,.25,.974,'图像 A',8.5);text(fig,.75,.974,'图像 B',8.5)
    coords=[]
    for j,k in enumerate(ids):coords.append(dict(pair_id=r['pair_id'],display_index=j+1,saved_structure_index=int(k),A_x=float(p[j,0]),A_y=float(p[j,1]),B_x=float(q[j,0]),B_y=float(q[j,1]),GT_file=r['gt_path'],coordinate_convention='native output pixel centers xy; not independent physical GT'))
    write_csv(OUT/f'FIG_D_EDITABLE_MARKERS_{t}.csv',coords)
    for x,L,pp in [(0.035,'A',p),(.53,'B',q)]:
        ax=image_ax(fig,[x,.08,.435,.37],mask(z,L));cp=z['check_points_'+L];cp=cp[np.linspace(0,len(cp)-1,min(64,len(cp))).astype(int)] if len(cp) else cp
        ax.scatter(cp[:,0],cp[:,1],s=2.5,c=GRAY,marker='.',zorder=3);ax.scatter(pp[:,0],pp[:,1],s=7,facecolors='none',edgecolors=BLUE,lw=.5,marker='s',zorder=4)
    text(fig,.5,.49,'灰点：几何检查点；青框：结构监督点',8)
    text(fig,.5,.025,'网格只检验投影；结构点来自已保存的候选，不是独立物理真值',7.5)
    marked=export(fig,'FIG_D_GEOMETRY_STRUCTURE',t,'28 exact saved legal structure anchors; geometrical grid shown separately',[r['pair_id']])
    pure=figure(80)
    image_ax(pure,[.025,.05,.46,.90],r['image_A']);image_ax(pure,[.515,.05,.46,.90],r['image_B'])
    no=export(pure,'FIG_D_PURE_PAIR',t,'pure full images; no overlaid markers',[r['pair_id']])
    return marked,no

def scene_statistics(group,rr,t):
    fig=figure(175);names=['离岸弱纹理语境','近岸舰船语境','港口码头语境','其他陆地语境']
    for i,(r,name) in enumerate(zip(group,names)):
        left=.035+(i%2)*.50;y=.64-(i//2)*.29
        text(fig,left+.215,y+.245,name,8.5)
        for j,L in enumerate('AB'):image_ax(fig,[left+j*.232,y,.215,.23],r['image_'+L])
    kinds=[('overlap_bin',['zero','10_30','30_50','50_70','70_90','90_100'],['0','10–30','30–50','50–70','70–90','90–100'],'重叠率（%）'),
           ('angle_bin',list(range(6)),['0–15','15–30','30–60','60–90','90–120','120–180'],'二维旋转（°）'),
           ('width',[640,512],['640','512'],'图像边长（像素）')]
    for i,(key,order,labels,title) in enumerate(kinds):
        ax=fig.add_axes([.09+i*.315,.095,.255,.17]);c=Counter(r[key] for r in rr);values=[c[x] for x in order]
        ax.bar(range(len(order)),values,color=BLUE,width=.68);ax.set_xticks(range(len(order)),labels,rotation=40 if i<2 else 0,ha='right' if i<2 else 'center');ax.set_ylim(0,max(values)*1.20)
        ax.spines[['top','right']].set_visible(False);ax.tick_params(width=.5,length=2,pad=1)
        ax.set_xlabel(title,fontproperties=FONT,fontsize=8,labelpad=2)
        if i==0:ax.set_ylabel('配对数',fontproperties=FONT,fontsize=8,labelpad=1)
        if i==2:
            for j,n in enumerate(values):ax.text(j,n+max(values)*.035,f'{n/len(rr)*100:.2f}%',ha='center',fontsize=7)
    text(fig,.5,.30,'实测分布：全部6,000对，不代表独立采集次数',8)
    return export(fig,'FIG_E_SCENES_DISTRIBUTIONS',t,'four formal source-context examples and full 6000-pair statistics',[r['pair_id'] for r in group])

def run():
    plan=json.loads(sorted((OUT/'meta').glob('FIGURE_PLAN_*.json'))[-1].read_text());t=stamp();e,rr,audit,ss=inputs()
    flow(plan,t);geometry(plan['groups']['B'],t);appearance(plan['groups']['C'],t);appearance(plan['groups']['C_SEA'],t,True)
    supervision(plan['groups']['D'][0],t);scene_statistics(plan['groups']['E'],rr,t)
    save(OUT/f'meta/FIGURE_EXPORTS_{t}.json',dict(timestamp=t,exports=EXPORTS,figure_plan_timestamp=plan['timestamp'],font=FONT.get_name(),
        font_fallback='SimSun/Times New Roman unavailable; installed Noto Serif CJK + Liberation Serif used; no font files packaged',
        matplotlib=matplotlib.__version__,no_dataset_or_GT_modified=True,document_Word_not_modified=True))
    print(json.dumps(dict(export_manifest=str(OUT/f'meta/FIGURE_EXPORTS_{t}.json'),figures=len(EXPORTS)),ensure_ascii=False),flush=True)
if __name__=='__main__':run()
