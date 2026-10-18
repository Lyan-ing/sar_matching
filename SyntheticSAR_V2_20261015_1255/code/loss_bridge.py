"""Thin data compatibility bridge: legacy M2 and B loss, no model/weight edits.

M2 loss already accepts arbitrary shapes and explicit points. B's old helper rereads
a fixed 384 source window and hardcodes 256 sampling coordinates. Replace ONLY
those data assumptions and device placement in a newly compiled function. Keep
historical loss source untouched. Static points come exclusively from V2 masks.
"""
from common_v2 import *
import ast
def legacy_view(z):
    out=dict(z)
    out['all_vessel_masks']=z['ship_masks']
    # Legacy `ship_masks` is used as an EXCLUSION mask by B's loss. Also exclude
    # unsupported unknown regions: missing structural support is not a negative GT.
    out['ship_masks']=[~v|m|i|(~s&~c) for v,m,i,s,c in zip(z['valid'],z['ship_masks'],z['ignore'],z['static'],z['calibration_mask'])]
    return out
def legacy_B_loss(model,teacher,z,device='cpu'):
    import torch
    import torch.nn.functional as F
    from lightglue.superpoint import sample_descriptors
    path=BASE/'SAR_SP_LG_StructureStable_20261013_2210/code/network_corrected.py'
    source=path.read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='joint_loss')
    code=ast.get_source_segment(source,node)
    code=code.replace('def joint_loss(m,teacher,z):',"def joint_loss(m,teacher,z,device='cpu'):")
    assert "from m3_model import structure_targets" in code and '/255*2-1' in code
    code=code.replace('from m3_model import structure_targets','')
    code=code.replace("device='cuda'",'device=device')
    code=code.replace('/255*2-1','/torch.tensor([ims.shape[-1]-1,ims.shape[-2]-1],device=device)*2-1')
    def score_map(logits):
        s=logits.softmax(1)[:,:64].permute(0,2,3,1);b,h,w,_=s.shape
        return s.reshape(b,h,w,8,8).permute(0,1,3,2,4).reshape(b,h*8,w*8)
    namespace=dict(torch=torch,np=np,F=F,warp=transform_points,valid_at=bilinear_at,
                   sample_descriptors=sample_descriptors,score_map=score_map,
                   structure_targets=lambda pair,teacher:(pair['points0'],pair['points1']))
    exec(compile(code,str(path)+'::V2_DATA_BRIDGE','exec'),namespace)
    return namespace['joint_loss'](model,teacher,legacy_view(z),device=device)
