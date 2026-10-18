"""Actual existing M2/Model-B CPU forward by role, no optimizer or checkpoint writes."""
import os,sys,json
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;variant=sys.argv[1]
if variant=='B':
 sys.path.insert(0,str(BASE/'SAR_SP_LG_StructureStable_20261013_2210/code'))
 from network_corrected import SARNet
 model=SARNet('b').cpu().eval();teacher=SARNet('m2').cpu().eval()
else:
 sys.path.insert(0,str(BASE/'SAR_SP_LG_Train_20261012_1544/code'))
 from model import SARModel
 model=SARModel('m2').cpu().eval();teacher=None
from data22 import load_pair
from common22 import rows
from loss22 import m2_loss,b_loss
import torch
torch.set_num_threads(2);results=[]
for r in rows(sys.argv[2]):
 for aux in ([False,True] if r['ship_descriptor_eligible'] else [False]):
  z=load_pair(r,use_ship_aux=aux)
  with torch.no_grad():loss,info=m2_loss(model,z) if variant=='M2' else b_loss(model,teacher,z)
  assert torch.isfinite(loss),(r['sample_id'],variant,info)
  if not aux:assert info['ship_points_used']==0
  results.append(dict(sample_id=r['sample_id'],arm=r['arm'],role=r['loss_role'],N=r['structural_anchor_count'],ship_aux_enabled=aux,shape=list(z['images'][0].shape),loss=float(loss),info=info))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(passed=True,variant=variant,cases=results,optimizer_steps=0,cuda_initialized=False)))
