from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
def matches(ims,p0,p1,inliers,row,target):
 fig,axs=plt.subplots(1,2,figsize=(10,5));mask=np.zeros(len(p0),bool) if inliers is None else inliers.astype(bool)
 for side in [0,1]:
  axs[side].imshow(ims[side],cmap='gray');p=[p0,p1][side];axs[side].scatter(p[:,0],p[:,1],s=5,c=np.where(mask,'#009E73','#D55E00'));axs[side].set_axis_off()
 for i in np.linspace(0,len(p0)-1,min(100,len(p0)),dtype=int):fig.add_artist(ConnectionPatch(p0[i],p1[i],coordsA='data',coordsB='data',axesA=axs[0],axesB=axs[1],lw=.5,alpha=.5,color='#009E73' if mask[i] else '#D55E00'))
 fig.suptitle(f"{row.get('variant',row.get('method'))}: {row.get('n_matches')} matches / {row.get('n_inliers')} self-fit support; {row.get('reason')}\nGreen is model fit support, NOT independently confirmed physical matches",fontsize=10);fig.tight_layout();fig.savefig(target,dpi=180);plt.close(fig)
