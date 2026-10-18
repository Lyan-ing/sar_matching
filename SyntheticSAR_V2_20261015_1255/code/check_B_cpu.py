"""One actual Model-B forward/loss on CPU, no optimization and no CUDA initialization."""
import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent
sys.path.insert(0,str(BASE/'SAR_SP_LG_StructureStable_20261013_2210/code'))
from network_corrected import SARNet
from dataset_v2 import load_training_pair,tensors
from loss_bridge import legacy_B_loss
import torch
torch.set_num_threads(2);torch.manual_seed(20261015)
m=SARNet('b').cpu().eval();teacher=SARNet('m2').cpu().eval()
inputs=json.loads(sys.argv[1]);inputs=inputs if isinstance(inputs,list) else [inputs];results=[]
for r in inputs:
 z=load_training_pair(r)
 with torch.no_grad():loss,info=legacy_B_loss(m,teacher,z,device='cpu')
 assert torch.isfinite(loss)
 if r['loss_role']=='structural_positive':assert info['n_points']>=2
 results.append(dict(sample_id=r['sample_id'],role=r['loss_role'],loss=float(loss),info=info,shape=list(tensors(z)['images'].shape)))
print(json.dumps(dict(passed=True,cases=results,device='cpu',optimizer_steps=0,weights_written=False)))
