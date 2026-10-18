"""Original train22 joint detector/descriptor loss on frozen B, CPU forward only."""
from interface_check import *
sys.path.insert(0,str(BASE/'SAR_SP_LG_V22_20261015_1525/code'))
import graded_loss

def main():
    release=json.loads(latest('meta/RELEASE_METADATA_*.json').read_text());rr=rows(release['manifests']['train'])
    checkpoint=BASE/'SAR_SP_LG_StructureStable_20261013_2210/runs/b/best.ckpt';before=sha(checkpoint)
    model=SARNet('b').cpu().eval();model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'],strict=True);model.kind='b';model.requires_grad_(False)
    teacher=SARNet('m2').cpu().eval().requires_grad_(False);teacher.kind='m2';out=[]
    for size in [512,640]:
        r=next(r for r in rr if r['width']==size and r['global_eligible']);z=load_pair(r)
        with torch.no_grad():loss,info=graded_loss.loss22(model,teacher,z,device='cpu',ship_aux=False)
        assert torch.isfinite(loss) and info['positive_cells']>0
        out.append(dict(pair_id=r['pair_id'],size=size,points=len(z['points0']),loss=float(loss),info=info))
    assert not torch.cuda.is_initialized() and sha(checkpoint)==before
    save(ROOT/f'meta/JOINT_DETECTOR_INTERFACE_{stamp()}.json',dict(created=now(),passed=True,cases=out,optimizer_steps=0,cuda_initialized=False,
        original_graded_loss_file=str(BASE/'SAR_SP_LG_V22_20261015_1525/code/graded_loss.py'),checkpoint=str(checkpoint),checkpoint_hash_before=before,checkpoint_hash_after=sha(checkpoint),
        loss_not_model_performance=True));print(json.dumps(dict(passed=True,sizes=[512,640],joint_detector_descriptor=True,optimizer_steps=0)))
if __name__=='__main__':main()
