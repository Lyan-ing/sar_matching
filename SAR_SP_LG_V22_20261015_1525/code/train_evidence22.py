"""Explicit evidence-counting correction. Original pilot code and state retained."""
import train22
from graded_loss import loss22 as original_loss
def evidence_loss(m,teacher,z,device='cuda',ship_aux=False):
 loss,info=original_loss(m,teacher,z,device,ship_aux)
 if m.kind=='b':
  active=bool(info['ground_points_available'] or info['calibration_cells'] or info['negative_loss']>0 or (ship_aux and len(z['ship_points0'])))
  info['artificial_blank_only']=not active;info['active']=active
 return loss,info
train22.loss22=evidence_loss
if __name__=='__main__':train22.main()
