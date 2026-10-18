"""Read-only constant-input spatial diagnosis, no training/threshold adjustment."""
from mixed_io import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family']='Noto Sans CJK JP'
torch.set_num_threads(2)
checkpoint=ROOT/'runs/t1'/f'checkpoint_step2000_{TAG}.ckpt';models={'共同初始化':Model('b'),'结构检测迁移2000步':Model('b',checkpoint)};rr=[];plot=[]
for name,m in models.items():
 for size in [256,512,640]:
  for label,value in [('全黑',0.),('均匀灰',.5)]:
   with torch.no_grad():im=torch.full((1,1,size,size),value);f=m.extract(im);p=f['keypoints'][0].numpy()
   margin=np.min(np.c_[p,size-1-p],axis=1) if len(p) else np.empty(0)
   rr.append(dict(method=name,size=size,probe=label,keypoints=len(p),within32_fraction=float((margin<32).mean()) if len(p) else None,within64_fraction=float((margin<64).mean()) if len(p) else None,median_edge_distance=float(np.median(margin)) if len(p) else None,NMS_radius=4,border_exclusion=4,threshold=.0005,notes='人工常量输入，仅诊断，不参与训练或选阈值'))
   if size==512:plot.append((name,label,value,p))
fig,axs=plt.subplots(2,2,figsize=(9,9),layout='constrained')
for ax,(name,label,value,p) in zip(axs.flat,plot):
 ax.imshow(np.full((512,512),value),cmap='gray',vmin=0,vmax=1);ax.scatter(p[:,0],p[:,1],s=8,facecolors='none',edgecolors='#D55E00');ax.plot([32,479,479,32,32],[32,32,479,479,32],'--',c='#56B4E9',lw=.8);ax.set_title(f'{name} / {label}：{len(p)}点',fontsize=10);ax.set_xlim(0,511);ax.set_ylim(511,0)
fig.suptitle('固定NMS/阈值下的常量输入；虚线距边32像素，仅标示位置\n共享编码器未更新；不得把这些响应解释为新增有效结构',fontsize=11)
t=stamp();path=ROOT/f'figures/constant_input_boundary_failure_{t}.png';fig.savefig(path,dpi=170);plt.close(fig);csvwrite(ROOT/f'CONSTANT_INPUT_BOUNDARY_{t}.csv',rr);print(rr);print(path)
