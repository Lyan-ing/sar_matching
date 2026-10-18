from mixed_io import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False})

def architecture():
 fig,ax=plt.subplots(figsize=(12,7));ax.set_xlim(0,12);ax.set_ylim(0,7);ax.axis('off')
 def box(x,y,w,h,text,color='#E8F3FA'):
  p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.08',fc=color,ec='#444',lw=1);ax.add_patch(p);ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
 def arrow(a,b,text=None,dashed=False):
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='->',mutation_scale=12,lw=1.2,linestyle='--' if dashed else '-',color='#333'))
  if text:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.1,text,fontsize=8,ha='center')
 box(.2,4.7,1.5,.9,'SAR双图\n原生512/640','#EEE');box(2.15,4.7,2,.9,'共享编码器\n卷积1–4，固定','#EEE');arrow((1.7,5.15),(2.15,5.15))
 box(4.8,5.5,2.2,.9,'检测头 convPa/Pb\n65类→8倍响应图');arrow((4.15,5.3),(4.8,5.9))
 box(4.8,3.75,2.2,.9,'描述头 convDa/Db\n残差融合，256维');arrow((4.15,4.9),(4.8,4.2))
 box(2.15,2.7,2,.9,'已有三尺度结构分支\nlog/Scharr，轻量卷积');arrow((.95,4.7),(2.15,3.2));arrow((4.15,3.2),(5.15,3.75),'+结构残差')
 box(7.75,4.7,1.8,1,'NMS/Top-K\n官方坐标采样');arrow((7,5.8),(8.2,5.7));arrow((7,4.2),(8.2,4.7))
 box(10,4.7,1.65,1,'LightGlue固定\n统一几何门控','#EEE');arrow((9.55,5.2),(10,5.2))
 box(4.8,1.6,2.3,1,'可靠锚点±2局部响应\n可微坐标q；训练期','#FFF2DA');arrow((5.9,5.5),(5.9,2.6),dashed=True)
 box(7.75,1.6,3.2,1,'在q处读取描述子 + 双向判别\n解析几何约束，梯度传回检测头','#FFF2DA');arrow((7.1,2.1),(7.75,2.1),dashed=True);arrow((7,3.9),(8.7,2.6),dashed=True)
 ax.text(.3,.55,'三个同初始化消融：结构+检测迁移 / 仅描述迁移 / 结构+检测+描述联合迁移\n实线=原有推理路径；虚线=新增训练期耦合。共享编码器与LightGlue始终冻结。结构分支不直接接入检测头。',fontsize=10)
 p=ROOT/f'figures/network_and_gradient_{stamp()}.png';fig.savefig(p,dpi=180,bbox_inches='tight');plt.close(fig);return p

def curves():
 fig,axs=plt.subplots(2,2,figsize=(12,7.5),layout='constrained')
 colors=['#0072B2','#009E73','#D55E00','#CC79A7']
 for color,name in zip(colors,['t1','t2','t3','repeat']):
  rr=[]
  for p in sorted((ROOT/'runs'/name).glob('metrics_*.jsonl')):rr.extend(json.loads(s) for s in p.read_text().splitlines() if s)
  if not rr:continue
  nontrivial={}
  for path in sorted((ROOT/'runs'/name).glob('sampling_actual_*.jsonl')):
   for line in path.read_text().splitlines():
    row=json.loads(line);values=[z['local_R1'] for z in row['supervision'] if z['contrastive_queries']>0];nontrivial[row['step']]=float(np.mean(values)) if values else float('nan')
  for ax,key in [(axs[0,0],'loss'),(axs[0,1],'descriptor'),(axs[1,0],'local_R1')]:
   x=np.array([r['step'] for r in rr]);y=np.array([nontrivial.get(r['step'],float('nan')) if key=='local_R1' else r[key] for r in rr]);ax.plot(x,y,lw=.25,alpha=.15,c=color);win=min(100,len(y));ax.plot(x[win-1:],np.convolve(y,np.ones(win)/win,'valid'),label=NAMES[name],c=color)
  vals=[json.loads(p.read_text()) for p in sorted((ROOT/'runs'/name).glob('validation_step*.json'))]
  if vals:axs[1,1].plot([r['step'] for r in vals],[100*r['metrics']['all']['Success@5px'] for r in vals],marker='o',label=NAMES[name],c=color)
 for ax,title in zip(axs.flat,['原始总Loss（细线）与100步均值','描述目标（含无合法负例的正对齐）','有合法负例的局部Recall@1（Train诊断）','固定1200对Val Natural Success@5']):ax.set_title(title,fontsize=10);ax.set_xlabel('实际优化步');ax.grid(alpha=.2);ax.legend(fontsize=8)
 axs[1,0].set_ylim(0,1);axs[1,1].set_ylim(0,100)
 p=ROOT/f'figures/training_validation_curves_{stamp()}.png';fig.savefig(p,dpi=180);plt.close(fig);return p
if __name__=='__main__':print(architecture());print(curves())
