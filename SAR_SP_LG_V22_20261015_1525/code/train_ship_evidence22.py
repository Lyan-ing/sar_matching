"""Paired auxiliary ablation: identical image-evidence stream with ship aux off/on."""
import train22
from graded_loss import loss22 as original_loss
def paired_loss(m,teacher,z,device='cuda',ship_aux=False):
 loss,info=original_loss(m,teacher,z,device,ship_aux)
 # Model-independent eligibility ensures off/on consume identical pairs/seeds.
 # Ship-only unknown calls do not make a larger effective image batch in the ON arm.
 active=bool(info['ground_points_available'] or info['calibration_cells'] or info['negative_pairs'])
 info['artificial_blank_only']=not active;info['active']=active
 if not active:
  info['ship_points_computed_not_backpropagated']=info['ship_points_used'];info['ship_points_used']=0
 return loss,info
train22.loss22=paired_loss
if __name__=='__main__':train22.main()
