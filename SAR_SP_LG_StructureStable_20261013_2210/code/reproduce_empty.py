from network_corrected import *
from collections import defaultdict

def main():
 gpu(1);teacher=load('m2');m=load('b',ROOT/'runs/b/last.ckpt');rs=rows(OLD/'train_sources.jsonl');r=next(r for r in rs if r['source_id']=='P0130_2400_3200_4800_5600');z=pair(r,26118782,'medium');assert np.stack(z['ship_masks']).all()
 loss,info=joint_loss(m,teacher,z);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters());event('empty_dynamic_regression',dict(seed=26118782,source_id=r['source_id'],all_cells_dynamic=True,passed=True,metrics=info));print(json.dumps(info))
if __name__=='__main__':main()
