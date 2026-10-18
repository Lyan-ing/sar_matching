from evaluation import *
from network import joint_loss,SARNet

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();ident=gpu(a.gpu);train=rows(OLD/'train_sources.jsonl');r=next(r for r in train if r['dataset']=='hrsid' and r['scene_type']=='inshore');z=pair(r,8123,'easy');teacher=load('m2');test={}
 for name in ['a','b','control']:
  torch.manual_seed(42);m=load(name);orig={k:v.cpu().clone() for k,v in m.state_dict().items()};o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4);history=[]
  im=torch.tensor(z['images'][0].copy(),device='cuda').float()[None,None]/255
  with torch.no_grad():initial=float((m.maps(im)[1]-teacher.maps(im)[1]).abs().max());assert initial==0
  for step in range(30):
   o.zero_grad();loss,info=joint_loss(m,teacher,z);assert torch.isfinite(loss);loss.backward()
   for n,q in m.named_parameters():
    if q.requires_grad:assert q.grad is not None and torch.isfinite(q.grad).all(),n
    else:assert q.grad is None,n
   o.step();history.append(info['loss'])
  changed=[k for k,v in m.state_dict().items() if not torch.equal(v.cpu(),orig[k])];allowed={k for k,p in m.named_parameters() if p.requires_grad};assert set(changed)<=allowed and changed
  assert min(history[15:])<history[0],(name,history)
  temp=ROOT/'preflight'/f'{name}.ckpt';save_ckpt(temp,dict(model_state=m.state_dict(),variant=name));m2=load(name,temp)
  with torch.no_grad():assert torch.equal(m.maps(im)[1],m2.maps(im)[1])
  ev=FairEvaluator([]);result=ev.predict(m,*z['images']);test[name]=dict(initial_descriptor_delta=initial,loss_start=history[0],loss_end=history[-1],trainable_params=sum(p.numel() for p in m.parameters() if p.requires_grad),changed=changed,LG_matches=len(result[0].points0),probe=probes(m),passed=True)
  del m,m2,o;torch.cuda.empty_cache()
 # Correspondence warp algebra and official sampling convention checked directly.
 q=np.array([[40.,80.],[110.,130.]]);assert np.allclose(warp(z['H_AB'],warp(z['H_A'],q)),warp(z['H_B'],q),atol=1e-9)
 m=load('a');odd=torch.rand(1,1,517,531,device='cuda')
 with torch.no_grad():f=m.extract(odd);g=m.tiled(odd,tile=256);assert torch.equal(f['keypoints'],g['keypoints']);delta=float((f['descriptors']-g['descriptors']).abs().max());assert delta<2e-6,delta
 event('preflight',dict(process=ident,models=test,tile_descriptor_max_delta=delta,passed=True));print(json.dumps(test,indent=2))
if __name__=='__main__':main()
