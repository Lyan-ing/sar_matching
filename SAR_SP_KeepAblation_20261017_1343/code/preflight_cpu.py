"""No optimization: exact code, stream and gradient normalization checks on CPU."""
from keep_train import *
import ast,inspect

def main():
 torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
 planpath=FROZEN_ROOT/f'plans/sampling_seed42_{FROZEN_TAG}.jsonl';plan=[json.loads(s) for s in planpath.read_text().splitlines()];oldconfig=json.loads((FROZEN_ROOT/f'FORMAL_PROTOCOL_{FROZEN_TAG}.json').read_text())
 assert sha(planpath)==oldconfig['sampler_sha'] and len(plan)==6000
 assert all(e['roles']==['desc']*4+['keep']*4 and e['step']==i+1 for i,e in enumerate(plan))
 for n in ['keep_objective.py','keep_evaluate.py','old_objective.py','joint_evaluate.py']:assert sha(ROOT/'code'/n)==sha(FROZEN_ROOT/'code'/n)
 oldast=ast.parse((FROZEN_ROOT/'code/keep_train.py').read_text());newast=ast.parse((ROOT/'code/keep_train.py').read_text())
 for name in ['update','optim','schedule','frozen_check','checkpoint']:
  aa=[n for n in ast.walk(oldast) if isinstance(n,ast.FunctionDef) and n.name==name];bb=[n for n in ast.walk(newast) if isinstance(n,ast.FunctionDef) and n.name==name];assert ast.dump(aa[0],include_attributes=False)==ast.dump(bb[0],include_attributes=False),name
 rows=records('train');model=Model('b');teacher=Model('b').eval().requires_grad_(False);params=configure(model);assert frozen_check(model,teacher)==360960
 entry=plan[0];gd=None;tot=torch.zeros(sum(p.numel() for p in params));infos=[]
 for j in range(0,8,2):
  xx=[];rr=[];items=[]
  for i in entry['ids'][j:j+2]:
   row=rows[i];p=FROZEN_ROOT/'cache/train'/f"{row['pair_id']}_{FROZEN_TAG}.pt"
   if not p.exists():p=OLD/'cache/train'/f"{row['pair_id']}_{OLDTAG}.pt"
   q=torch.load(p,map_location='cpu',weights_only=False);xx.append(q['x']);rr.append(q['responses'])
   with np.load(FROZEN_ROOT/'cache/supervision'/f"{row['pair_id']}_{FROZEN_TAG}.npz") as z:items.append({k:z[k] for k in z.files})
  d,k,inf=terms(model,teacher,torch.cat(xx),torch.cat(rr),items,entry['roles'][j:j+2],amp=False);loss=(d+0.*k)/2;loss.backward();infos+=inf
  if j==2:gd=vector(model).clone()
 total=vector(model);assert torch.equal(total,gd) and torch.linalg.vector_norm(total)>0
 assert sum(i['structure_pairs'] for i in infos)==4 and sum(i['keep_pairs'] for i in infos)==4
 assert all(p.grad is None for p in teacher.parameters());assert all(p.grad is None for n,p in model.named_parameters() if not n.startswith('sp.convD'))
 frozen_check(model,teacher)
 out=dict(passed=True,optimizer_steps=0,cpu_real_features=True,same_sampler_sha=sha(planpath),sampling_6000_x_4desc_4keep=True,unchanged_functions=['update','optim','schedule','frozen_check','checkpoint'],unchanged_loss_and_evaluation=True,gradient_after_zero_weight_keep_bitwise_identical_to_4_descriptor_pairs=True,descriptor_gradient_norm=float(torch.linalg.vector_norm(total)),keep_stream_not_reallocated=True,structure_pairs=4,keep_pairs=4,only_descriptor_head_requires_grad=True,teacher_gradient_none=True,full_config_changes='lambda_keep=0 plus output/code identity paths; frozen sampling reference unchanged')
 write(ROOT/f'PREFLIGHT_CPU_{stamp()}.json',out);print(json.dumps(out),flush=True)
if __name__=='__main__':main()
